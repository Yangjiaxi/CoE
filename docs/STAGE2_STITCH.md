# Stage 2: Stitch Experts into MoE

Load the FFN-trained expert checkpoints and merge them into a single MoE architecture.

## Merge Logic

1. **Embedding & lm_head**: Averaged across all experts (vocab-related params may differ slightly due to different training).
2. **Attention & Layernorm**: Replicated from first expert (should be identical across experts since they were frozen).
3. **MLP**: Each expert's MLP is placed in `model.layers.{i}.moe_mlp.experts.{expert_idx}.*`
4. **Router (gate)**: Newly initialized; trained in Stage 3.

## Consistency Checks

Before merging, the script verifies:
- MLP params differ between experts (expected)
- Non-MLP, non-vocab params are identical across experts

## Usage

```bash
python stitch_experts.py \
  --experts "math=/path/to/math/final" "coding=/path/to/coding/final" "general_growth=/path/to/general/final" \
  --output_dir output/moe_prebuilt \
  --num_experts_per_tok 2 \
  --router_loss_type ce \
  --router_loss_alpha 1.0
```

## Parameters

| Parameter               | Default               | Description                                    |
| ----------------------- | --------------------- | ---------------------------------------------- |
| `--experts`             | (required)            | `key=path` pairs, e.g. `math=/path math=/path` |
| `--output_dir`          | `output/moe_prebuilt` | Save path                                      |
| `--num_experts_per_tok` | 2                     | TopK for routing                               |
| `--router_loss_type`    | `ce`                  | `ce` or `margin`                               |
| `--router_loss_alpha`   | 1.0                   | Router loss weight (used in Stage 3)           |

## Output

The stitched model is a standard HuggingFace model directory with:
- `config.json` (MoeLlamaConfig)
- `model.safetensors` or `pytorch_model.bin`
- `tokenizer.json`, etc.

## Shell Script

```bash
MATH_EXPERT=.../math/final \
CODING_EXPERT=.../coding/final \
GENERAL_EXPERT=.../general/final \
OUTPUT_DIR=./moe_prebuilt \
./scripts/stage2/stitch_only.sh
```
