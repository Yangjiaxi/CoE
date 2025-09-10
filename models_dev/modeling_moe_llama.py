""" Moe-LLaMA model."""

import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.llama.modeling_llama import (
    LlamaRMSNorm,
    LlamaMLP,
    LlamaAttention,
    LlamaFlashAttention2,
    LlamaSdpaAttention,
)

from .configuration_moe_llama import MoeLlamaConfig


logger = logging.get_logger(__name__)


@dataclass
class MoeBaseOutput(BaseModelOutputWithPast):
    all_routing_idx: Optional[Tuple[torch.LongTensor]] = None  # num_layer * [B, L, TopK]
    all_routing_weight: Optional[Tuple[torch.FloatTensor]] = None  # num_layer * [B, L, TopK]
    all_router_logits: Optional[Tuple[torch.FloatTensor]] = None  # num_layer * [B, L, #E]


@dataclass
class MoeCLMOutput(CausalLMOutputWithPast, MoeBaseOutput):
    ce_loss: torch.FloatTensor = None
    router_loss: torch.FloatTensor = None


def _infer_shape(e):
    if isinstance(e, torch.Tensor):
        return f"{e.size()}"
    elif isinstance(e, (tuple, set, list)):
        return f"{len(e)} * {_infer_shape(e[0])}"
    else:
        return type(e)


def infer_shape(e):
    return f"[{_infer_shape(e)}]"


LLAMA_ATTENTION_CLASSES = {
    "eager": LlamaAttention,
    "flash_attention_2": LlamaFlashAttention2,
    "sdpa": LlamaSdpaAttention,
}


class MoeMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        self.gate = nn.Linear(self.hidden_size, self.n_routed_experts, bias=False)
        self.experts = nn.ModuleList([LlamaMLP(config) for i in range(self.n_routed_experts)])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden: [B, L, H]
        batch_size, sequence_length, hidden_dim = hidden_states.shape

        hidden_states = hidden_states.view(-1, hidden_dim)  # [B*L, H]
        router_logits = self.gate(hidden_states)  # [B*L, H] -> [B*L, #E], this is what we want to apply loss on

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)  # token-level experts selection
        # routing_weights: [B*L, TopK], float, the weight for selected experts
        # selected_experts: [B*L, TopK], uint, the index for selected experts

        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)  # normalize the routing weight
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )  # zeros: [B*L, H], selective fill them back

        # One hot encode the selected experts to create an expert mask
        # TopK-hot [B*L, #E] -> [B*L, TopK, #E] -> [#E, TopK, B*L], expert-by-expert
        expert_mask = F.one_hot(selected_experts, num_classes=self.n_routed_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.n_routed_experts):
            expert_layer = self.experts[expert_idx]
            # `True` values indicate that the expert(expert_idx) is one of the
            # top K selections for that particular token.
            idx, top_x = torch.where(expert_mask[expert_idx])  # where on [TopK, B*L]
            # top_x: tokens' indices that will be processed by this experts (expert_idx)
            # idx: positions within the top `K` choices for each token where this expert is selected
            # top_x: [num_of_active_tokens], range: [0, B*L-1] indicate the active tokens
            # idx: [num_of_active_tokens], range: [0, TopK-1], useful for weighting

            # [B*L, H] @ [None, num_of_active_tokens]
            # => [1(from None), num_active_tokens, H]
            # => [num_active_tokens, H]
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)  # [num_active_tokens, H]

            # routing_weights: [B*L, TopK] of float
            # top_x select tokens
            # idx select expert
            # => [num_active_tokens, 1(from None)]
            #
            # expert_layer(current_state): [num_active_tokens, H] => [num_active_tokens, H]
            #
            # output: [num_active_tokens, H] * [num_active_tokens, 1] => [num_active_tokens, 1(from None)]
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)

        routing_infos = {
            "topk_idx": selected_experts.view(batch_size, sequence_length, -1),  # [B, L, TopK]
            "topk_weight": routing_weights.view(batch_size, sequence_length, -1),  # [B, L, TopK]
            "router_logits": router_logits.view(batch_size, sequence_length, -1),  # [B, L, #E]
        }
        return final_hidden_states, routing_infos


# Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2 with Llama->MoeLlama
class MoeLlamaDecoderLayer(nn.Module):
    def __init__(self, config: MoeLlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LLAMA_ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.moe_mlp = MoeMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, routing_infos = self.moe_mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states, routing_infos)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs  # hidden, routing_infos, [attn], [kv_cache]


# Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2 with Llama->MoeLlama
class MoeLlamaPreTrainedModel(PreTrainedModel):
    config_class = MoeLlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["MoeLlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2 with Llama->MoeLlama
class MoeLlamaModel(MoeLlamaPreTrainedModel):
    def __init__(self, config: MoeLlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([MoeLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MoeBaseOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions)

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_routing_idx = ()
        all_routing_weight = ()
        all_router_logits = ()

        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )
            # layer_outputs: hidden, routing_infos, [attn], [kv_cache]

            hidden_states = layer_outputs[0]
            routing_infos = layer_outputs[1]

            if use_cache:
                next_decoder_cache = layer_outputs[3 if output_attentions else 2]

            if output_attentions:
                all_self_attns += (layer_outputs[2],)

            all_routing_idx += (routing_infos["topk_idx"],)
            all_routing_weight += (routing_infos["topk_weight"],)
            all_router_logits += (routing_infos["router_logits"],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return MoeBaseOutput(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            all_routing_idx=all_routing_idx,
            all_routing_weight=all_routing_weight,
            all_router_logits=all_router_logits,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else past_seen_tokens + sequence_length + 1

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask


# Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2 with Llama->MoeLlama
class MoeLlamaForCausalLM(MoeLlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.n_routed_experts = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.router_loss_type = config.router_loss_type
        self.router_loss_alpha = config.router_loss_alpha
        self.ordinal_to_expert = config.ordinal_to_expert

        self.model = MoeLlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        self._cached_routing_info = None

    def get_expert_name(self, idx):
        return self.ordinal_to_expert.get(str(idx), "unknown")

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    # @staticmethod
    # def consistency_loss(top_k_indices):
    #     # Flatten the indices to simplify set operations
    #     batch_size, seq_len, _ = top_k_indices.shape
    #     flattened_indices = top_k_indices.view(batch_size, seq_len * 2)  # Flattening as each position has TopK indices

    #     def set_intersection_loss(batch):
    #         unique_experts = batch.unique()
    #         intersection = 0
    #         for expert in unique_experts:
    #             min_count = (batch == expert).sum()
    #             intersection += min_count
    #         return 1 - (intersection / (seq_len * 2))  # Normalize by the total number of unique expert slots available

    #     loss = torch.mean(torch.stack([set_intersection_loss(batch) for batch in flattened_indices]))
    #     return loss

    @staticmethod
    def topk_margin_loss(router_logits, domain_labels, top_k):
        batch_size, seq_len, num_experts = router_logits.size()
        correct_expert_logits = router_logits.gather(-1, domain_labels.unsqueeze(-1).expand(-1, seq_len, -1))
        topk_scores = torch.topk(router_logits, top_k, dim=-1).values
        margin = correct_expert_logits - topk_scores[:, :, -1:]
        loss = F.relu(-margin).mean()  # Using ReLU to zero-out non-negative margins (where correct expert is in TopK)
        return loss

    @staticmethod
    def expert_selection_loss(router_logits, domain_labels):
        _, sequence_length, num_expert = router_logits.size()  # [B, L, #E]
        domain_labels = domain_labels.expand(-1, sequence_length)  # [B, 1] -> [B, L]

        # [B*L, #E] <- [B*L]
        loss = F.cross_entropy(router_logits.reshape(-1, num_expert), domain_labels.reshape(-1))
        return loss

    def set_cached_routing_info(self, all_routing_idx):
        # # all_routing_idx: 32 * [B, L/L', K]
        # if self._cached_routing_info is None:
        #     # third_tensor = torch.cat((first_tensor, second_tensor), 0)
        #     self._cached_routing_info = [e.cpu().detach() for e in all_routing_idx]

        # for i in range(len(self._cached_routing_info)):
        #     cur = all_routing_idx[i].cpu().detach()  # [B, L_new, K]
        #     self._cached_routing_info[i] = torch.cat((self._cached_routing_info[i], cur), dim=1)

        self._cached_routing_info = [e.cpu().detach() for e in all_routing_idx]
        # print(f"{infer_shape(self._cached_routing_info) = }")

    def get_cached_routing_info(self):
        return self._cached_routing_info

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        router_target: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, MoeCLMOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        if self.training:
            assert labels is not None
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss_ce = loss_fct(shift_logits, shift_labels)

            if router_target is not None:
                losses_expert = []
                for idx, layer_router_logits in enumerate(outputs.all_router_logits):
                    # print(f"{idx=} | {layer_router_logits.size()} {router_target.size()}")
                    if self.config.router_loss_type == "margin":
                        layer_expert_loss = self.topk_margin_loss(layer_router_logits, router_target, self.config.num_experts_per_tok)
                    elif self.config.router_loss_type == "ce":
                        layer_expert_loss = self.expert_selection_loss(layer_router_logits, router_target)
                    else:
                        raise ValueError(f"{self.config.router_loss_type = }")
                    losses_expert.append(layer_expert_loss)
                loss_expert = sum(losses_expert) / len(losses_expert)

                loss = loss_ce + self.config.router_loss_alpha * loss_expert
            else:
                loss_expert = None
                loss = loss_ce
        else:
            loss, loss_ce, loss_expert = None, None, None

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        self.set_cached_routing_info(all_routing_idx=outputs.all_routing_idx)
        # print(f"{infer_shape(outputs.all_routing_idx) = }")

        return MoeCLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            all_routing_idx=outputs.all_routing_idx,
            all_routing_weight=outputs.all_routing_weight,
            all_router_logits=outputs.all_router_logits,
            ce_loss=loss_ce,
            router_loss=loss_expert,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device) if past_key_values.get_max_length() is not None else None
                )
                cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
            # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if max_cache_length is not None and attention_mask is not None and cache_length + input_ids.shape[1] > max_cache_length:
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        elif use_cache:
            cache_position = cache_position[-input_length:]

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),)
        return reordered_past
