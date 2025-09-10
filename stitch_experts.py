"""
Stitch domain expert models into a MoE architecture.
Loads FFN-only trained experts from Stage 1, merges them into a single MoE model.
"""

import argparse
import gc
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models_dev.modeling_moe_llama import MoeLlamaForCausalLM
from models_dev.configuration_moe_llama import MoeLlamaConfig


def merge_state_dict(experts, moe_config):
    """
    Merge multiple expert state dicts into a single MoE state dict.
    - Embedding and lm_head: averaged across experts
    - Attention, layernorm: replicated from first expert (should be identical)
    - MLP: mapped to moe_mlp.experts.{idx}.*
    """
    model_to_state_dict = {}
    for idx, (expert_name, model_path) in enumerate(experts.items()):
        print(f"[{idx}] load expert [{expert_name}]: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            device_map="cpu",
        )
        model_state_dict = model.state_dict()
        model_to_state_dict[expert_name] = {k: v.cpu() for k, v in model_state_dict.items()}
        del model
        gc.collect()

    # Parameter consistency check (compare first two experts)
    expert_names = list(model_to_state_dict.keys())
    if len(expert_names) >= 2:
        ref1, ref2 = model_to_state_dict[expert_names[0]], model_to_state_dict[expert_names[1]]
        param_keys = list(ref1.keys())
        for key in param_keys:
            params_equal = torch.equal(ref1[key], ref2[key])
            if ".mlp." in key:
                assert not params_equal, f"MLP params should differ: {key}"
            else:
                if any(e in key for e in [".embed_tokens.", "lm_head."]):
                    # vocab params can differ per token
                    pass
                else:
                    assert params_equal, f"Non-MLP params should be equal: {key}"

    num_layers = moe_config.num_hidden_layers
    print(f"*** parameters checks passed *** num_layers={num_layers}")

    state_dict = {}

    # 1. Vocab-related: average
    tensors_mean = lambda e: torch.mean(torch.stack(e), dim=0)
    all_embed_tokens = tensors_mean([sd["model.embed_tokens.weight"] for sd in model_to_state_dict.values()])
    all_lm_head = tensors_mean([sd["lm_head.weight"] for sd in model_to_state_dict.values()])
    state_dict["model.embed_tokens.weight"] = all_embed_tokens
    state_dict["lm_head.weight"] = all_lm_head

    expert_name_to_idx = {v: int(k) for k, v in moe_config.ordinal_to_expert.items()}
    for expert_name, expert_state_dict in model_to_state_dict.items():
        expert_idx = expert_name_to_idx[expert_name]
        for param_name, param in expert_state_dict.items():
            if ".mlp." in param_name:
                leading_part, rest_part = param_name.split(".mlp.", maxsplit=1)
                assert leading_part.startswith("model.layers.")
                routed_expert_name = f"{leading_part}.moe_mlp.experts.{expert_idx}.{rest_part}"
                state_dict[routed_expert_name] = param.cpu()
            elif any(e in param_name for e in [".embed_tokens.", "lm_head."]):
                pass
            else:
                if param_name not in state_dict:
                    state_dict[param_name] = param

    print(f"Collected key-values: {len(state_dict)}")
    return state_dict


def main():
    parser = argparse.ArgumentParser(description="Stitch domain experts into MoE")
    parser.add_argument(
        "--experts",
        type=str,
        nargs="+",
        required=True,
        help="Expert paths as key=value, e.g. coding=/path/to/coding math=/path/to/math",
    )
    parser.add_argument(
        "--output_dir", type=str, default="output/moe_prebuilt", help="Output directory for stitched MoE model"
    )
    parser.add_argument("--num_experts_per_tok", type=int, default=2, help="TopK")
    parser.add_argument("--router_loss_type", type=str, default="ce", choices=["ce", "margin"])
    parser.add_argument("--router_loss_alpha", type=float, default=1.0)
    args = parser.parse_args()

    experts = {}
    for kv in args.experts:
        k, v = kv.split("=", 1)
        experts[k.strip()] = v.strip()

    # Use first expert as backbone for tokenizer/config
    first_expert_path = list(experts.values())[0]
    tokenizer = AutoTokenizer.from_pretrained(first_expert_path, local_files_only=True)

    moe_config = MoeLlamaConfig.from_pretrained(
        first_expert_path,
        n_routed_experts=len(experts),
        num_experts_per_tok=args.num_experts_per_tok,
        ordinal_to_expert={str(e): k for e, k in enumerate(experts)},
        router_loss_type=args.router_loss_type,
        router_loss_alpha=args.router_loss_alpha,
    )

    merged_state_dict = merge_state_dict(experts, moe_config)
    gc.collect()
    torch.cuda.empty_cache()

    moe_llama = MoeLlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=None,
        config=moe_config,
        state_dict=merged_state_dict,
        torch_dtype=torch.bfloat16,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    moe_llama.save_pretrained(str(output_dir), max_shard_size="12GB", safe_serialization=False)
    tokenizer.save_pretrained(str(output_dir))
    print(f"=> Saved to {output_dir}")


if __name__ == "__main__":
    main()
