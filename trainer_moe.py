import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch
import transformers
from transformers import Trainer
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled


# Copied from transformers/trainer_pt_utils.py
def parameter_numel(p):
    if is_deepspeed_zero3_enabled():
        _numel = lambda p: p.ds_numel if hasattr(p, "ds_numel") else p.numel()
    else:
        _numel = lambda p: p.numel()
    return _numel(p)


def get_model_param_count(model, trainable_only=False):
    return sum(parameter_numel(p) for p in model.parameters() if not trainable_only or p.requires_grad)


class DeepSpeedTrainer(Trainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        checkpoint_folder = f"checkpoint-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = Path(run_dir).joinpath(checkpoint_folder)
        logging.info(f"\To save: {output_dir.resolve()}")

        self.save_model(output_dir, _internal_call=True)

        if self.is_world_process_zero():
            logging.info(f"\tSaved: {output_dir.resolve()}")
            saved_locker = output_dir.joinpath(".saved")
            with saved_locker.open("w") as f:
                f.write("saved")
            logging.info(f"\tMake: {saved_locker.resolve()}")

    def log(self, logs: dict):
        the_logs = {}
        for k, v in logs.items():
            if isinstance(v, torch.Tensor):
                the_v = v.item()
            else:
                the_v = v
            the_logs[k] = the_v

        return super().log(the_logs)

    def _save(self, output_dir=None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Saving model checkpoint to {output_dir}")

        self.model.save_pretrained(output_dir, state_dict=state_dict, max_shard_size="12GB", safe_serialization=False)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, str(output_dir.joinpath("training_args.bin").resolve()))


def _infer_shape(e):
    if isinstance(e, torch.Tensor):
        return f"{e.size()}"
    elif isinstance(e, (tuple, set, list)):
        return f"{len(e)} * {_infer_shape(e[0])}"
    else:
        return type(e)


def infer_shape(e):
    return f"[{_infer_shape(e)}]"


class MoeDeepSpeedTrainer(DeepSpeedTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.log_cache = {}

    def log(self, logs, just_cache=False):
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen

        if just_cache:
            self.log_cache.update(logs)
        else:
            output_logs = {**self.log_cache.copy(), **logs}
            self.log_cache = {}  # after log, make it empty

            output = {**output_logs, **{"step": self.state.global_step}}
            self.state.log_history.append(output)
            self.control = self.callback_handler.on_log(self.args, self.state, self.control, output_logs)

    def compute_loss(self, model, inputs, return_outputs=False):
        # 1. no `label_smooth`
        # 2. causal language model training with input and labels
        # 3. label shift is done by model itself

        rank = self.args.process_index

        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # print(f"Rank-{rank}: {loss=}")

        # [1] router_loss ------------------------------------------------------------
        if "router_loss" in outputs and outputs["router_loss"] is not None:
            router_loss = outputs["router_loss"]
            router_loss_gathered = self._nested_gather(router_loss)
            final_router_loss = sum(router_loss_gathered) / len(router_loss_gathered)
            final_router_loss = final_router_loss.cpu().item()
        else:
            final_router_loss = 0.0

        # [1.2] language ce loss ------------------------------------------------------------
        ce_loss = outputs["ce_loss"]
        ce_loss_gathered = self._nested_gather(ce_loss)
        final_ce_loss = sum(ce_loss_gathered) / len(ce_loss_gathered)
        final_ce_loss = final_ce_loss.cpu().item()

        # [2] routing_idx ---------------------------------------------------------
        all_rounting_idx_rank = outputs["all_routing_idx"]  # layers * [B_device, L_device, #E_per_token]
        attention_mask_rank = inputs["attention_mask"]  # [B_device, L_device]

        # print(f"[Rank-{rank:02d}] len={attention_mask_rank.shape[1]}")
        # total_tokens_rank = torch.sum(attention_mask_rank)

        num_experts = model.n_routed_experts
        placeholder_value = num_experts  # routing idx in [0, ..., num_experts-1], making `num_experts` a placeholder
        masked_all_routing_idx_rank = [
            torch.where(
                attention_mask_rank.unsqueeze(-1) != 0,
                e,
                torch.full_like(e, placeholder_value),
            )
            for e in all_rounting_idx_rank
        ]  # layers * [B_device, L_device, #E_per_token], masked elements replace by n_rounted+1

        flattened_routing_index_per_layer = [e.view(-1) for e in masked_all_routing_idx_rank]
        per_layer_count = [
            layer_rounting_idx.bincount(minlength=num_experts + 1)[:num_experts].unsqueeze(0)
            for layer_rounting_idx in flattened_routing_index_per_layer
        ]  # layers * [n_routed]

        per_layer_count_gathered = self._nested_gather(per_layer_count)  # layers * [1, n_routed], additional dimension is for .sum
        per_layer_count_global = [torch.sum(e, dim=0).squeeze(0) for e in per_layer_count_gathered]  # layers * [n_routed]
        overall_count_global = sum(per_layer_count_global)  # [n_routed]

        per_layer_router_ratio = [(e / torch.sum(e)).cpu().tolist() for e in per_layer_count_global]
        overall_router_ratio = (overall_count_global / torch.sum(overall_count_global)).cpu().tolist()

        overall_router_ratio_dict = {}
        for idx, ratio in enumerate(overall_router_ratio):
            overall_router_ratio_dict[f"routing_overall/{model.get_expert_name(idx)}"] = round(ratio * 100, 2)

        layer_router_ratio_dict = {}
        for layer_idx, layer_ratio in enumerate(per_layer_router_ratio):
            for router_idx, ratio in enumerate(layer_ratio):
                layer_router_ratio_dict[f"routing_layers/layer{layer_idx:02d}/{model.get_expert_name(router_idx)}"] = round(ratio * 100, 2)

        # if rank == 0:
        #     print(f"per_layer_router_ratio: {infer_shape(per_layer_router_ratio)}")
        #     print(f"overall_router_ratio: {infer_shape(overall_router_ratio)}")
        #     print(overall_router_ratio_dict)
        #     # print(overall_router_ratio_dict)
        self.log({"router_loss": final_router_loss}, just_cache=True)
        self.log({"ce_loss": final_ce_loss}, just_cache=True)
        self.log(overall_router_ratio_dict, just_cache=True)
        self.log(layer_router_ratio_dict, just_cache=True)

        # torch.sum(per_layer_count_whole) == torch.sum(attention_mask_rank) * num_layers * #E_per_token

        return (loss, outputs) if return_outputs else loss


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments:
    overwrite_cache: bool = field(default=False)
    data_path: str = field(default=None, metadata={"help": "Path to the training data, seperate by comma(,) if mixing is desired."})
    data_path_eval: str = field(default=None, metadata={"help": "Path to the evaluation data."})
    ensure_bos_token: bool = field(default=False)
    eval_data_limit: int = field(default=None)
    data_format: str = field(default="special_token")
    # special_token: <|system|> <|user|> <|assitant|>+<eos>
    # chatml: <im_start> <im_end>
    # chatml_full: apply_chat_template
    force_system_prompt: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    group_name: str = field(default="LLaMA2")
    exp_name: str = field(default=None)

    training_part: str = field(default="full")
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    max_seq_length: int = field(default=2048)
    report_to: str = field(default="wandb")
    log_level: str = field(default="debug")


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if "pad_token" in special_tokens_dict:
        model.config.pad_token_id = tokenizer.pad_token_id

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

    return num_new_tokens


def encode_with_messages_format(example, tokenizer, max_seq_length):
    """
    Here we assume each example has a 'messages' field Each message is a dict with 'role' and 'content' fields.
    We concatenate all messages with the roles as delimiters and tokenize them together.
    """
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")

    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() + tokenizer.eos_token + "\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text

    example_text = _concat_messages(messages).strip()
    tokenized_example = tokenizer(example_text, return_tensors="pt", max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]), return_tensors="pt", max_length=max_seq_length, truncation=True
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = _concat_messages(messages[: message_idx + 1]) + "<|assistant|>\n"
            else:
                messages_so_far = _concat_messages(messages[: message_idx + 1])
            message_end_idx = tokenizer(messages_so_far, return_tensors="pt", max_length=max_seq_length, truncation=True).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)

    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def encode_with_messages_format_chatml(example, tokenizer, max_seq_length, meta_mapping=None):
    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")

    def _concat_messages(messages):
        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|im_start|>system\n" + message["content"].strip() + "<|im_end|>\n"
            elif message["role"] == "user":
                message_text += "<|im_start|>user\n" + message["content"].strip() + "<|im_end|>\n"
            elif message["role"] == "assistant":
                message_text += "<|im_start|>assistant\n" + message["content"].strip() + "<|im_end|>\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text

    example_text = _concat_messages(messages).strip()
    tokenized_example = tokenizer(example_text, return_tensors="pt", max_length=max_seq_length, truncation=True)
    input_ids = tokenized_example.input_ids
    labels = input_ids.clone()

    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer(
                    _concat_messages(messages[:message_idx]),
                    return_tensors="pt",
                    max_length=max_seq_length,
                    truncation=True,
                ).input_ids.shape[1]
            if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
                # here we also ignore the role of the assistant
                messages_so_far = _concat_messages(messages[: message_idx + 1]) + "<|im_start|>assistant\n"
            else:
                messages_so_far = _concat_messages(messages[: message_idx + 1])

            message_end_idx = tokenizer(messages_so_far, return_tensors="pt", max_length=max_seq_length, truncation=True).input_ids.shape[1]
            labels[:, message_start_idx:message_end_idx] = -100

            if message_end_idx >= max_seq_length:
                break

    attention_mask = torch.ones_like(input_ids)
    if meta_mapping is not None and isinstance(meta_mapping, dict):
        meta_tag = example["meta"]
        router_target = meta_mapping[meta_tag]
        router_target = torch.tensor([router_target])
        router_target_pack = {"router_target": router_target}
    else:
        router_target_pack = {}

    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
        **router_target_pack,
    }


def encode_with_messages_format_chatml_full(example, tokenizer, max_seq_length):
    messages = example["messages"]

    ret = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        max_length=max_seq_length,
        truncation=True,
        return_dict=True,
        add_generation_prompt=False,
    )

    input_ids = ret.input_ids
    labels = input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    attention_mask = ret.attention_mask

    return dict(
        input_ids=input_ids[0],
        labels=labels[0],
        attention_mask=attention_mask[0],
    )


def add_system_prompt(example, system_prompt="You are a very helpful assistant."):
    messages = example["messages"]

    for msg in messages:
        if msg["role"] == "system":
            return example

    system_msg = {"role": "system", "content": system_prompt}
    example["messages"] = [system_msg] + messages

    return example
