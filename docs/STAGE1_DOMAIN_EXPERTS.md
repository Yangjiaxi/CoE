# Stage 1: Domain Expert Training

Train one LLaMA model per domain (math, coding, general), **only updating FFN (MLP) parameters**. All other parameters (embedding, attention, lm_head) remain frozen.

## Implementation Details

- **Trainable**: `model.layers.*.mlp.*` (gate_proj, up_proj, down_proj)
- **Frozen**: embedding, attention, layernorm, lm_head
- **Special tokens**: If new tokens are added (e.g., chat template), only those token positions in embedding/lm_head receive gradients (via hook mask)

## Key Parameters

| Parameter                       | Typical Value | Description        |
| ------------------------------- | ------------- | ------------------ |
| `--training_part`               | `ffn`         | FFN-only training  |
| `--max_seq_length`              | 2048          | Sequence length    |
| `--num_train_epochs`            | 5             | Epochs per domain  |
| `--learning_rate`               | 2e-5          | LR for FFN         |
| `--per_device_train_batch_size` | 2             | Batch size per GPU |
| `--gradient_checkpointing`      | True          | Save memory        |
| `--data_format`                 | chatml        | ChatML format      |

## Data Combinations

Use `-` to concatenate multiple datasets. Examples:

- **Math**: `tulu_math-mammoth_cot`
- **Coding**: `tulu_coding-code_alpaca-oss_instruct`
- **General**: `tulu_g.growth`

## Script Usage

```bash
# Set BASE_MODEL_PATH and OUTPUT_BASE in the script or via env
./scripts/stage1/train_domain_expert.sh <model_name> <batch_size> <data_names> <learning_rate>
```

**Example**:
```bash
export BASE_MODEL_PATH=/path/to/llama-2-7b
./scripts/stage1/train_domain_expert.sh 7b 2 tulu_math-mammoth_cot_no_aqua 2e-5
```

Output: `{OUTPUT_BASE}/s1-domain-expert-ffn/{exp_name}/final/`

## Multi-GPU

The script auto-detects single-node vs multi-node. For multi-node, set:
- `WORLD_SIZE`, `RANK`, `MASTER_ADDR`, `MASTER_PORT`

## DeepSpeed

Uses ZeRO-2 by default (`configs/zero2_bf16_cos_detailed.json`). For larger models, consider ZeRO-3.
