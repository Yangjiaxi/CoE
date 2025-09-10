# Stage 3: Finetune Stitched MoE on Full Data

After stitching, finetune the MoE on **full mixed data**. The trainable part is the **complement of FFN**: embedding, attention, router, lm_head. FFN experts remain frozen.

## Rationale

- Stage 1: Experts learned domain-specific FFN representations.
- Stage 2: Experts are stitched; router is randomly initialized.
- Stage 3: Train the router (and shared components) to route correctly on mixed data, while keeping expert FFNs fixed.

## Training Modes

| Mode         | Trainable                  | Use Case                                              |
| ------------ | -------------------------- | ----------------------------------------------------- |
| `moe-rest`   | emb, attn, router, lm_head | **Recommended** for Stage 3; complementary to Stage 1 |
| `moe-full`   | All                        | Full finetune of MoE                                  |
| `moe-router` | Router only                | Fast router-only tuning                               |

## Data Format for Router Supervision

When using `data_format=chatml` and MoE training, each example can have a `meta` field:
```json
{"meta": "math", "messages": [...]}
```
`meta` is used as the router target (CE or margin loss). The `meta` values must match `ordinal_to_expert` in the MoE config (e.g., `math`, `coding`, `general`).

If `meta` is missing, only the language modeling loss is used (no router loss).

**Adding meta fields**: Use `data_prepare/add_meta_expert.py` to inject `meta` and `data` into prepared JSONL files. See [Stage 0: Data Preparation](STAGE0_DATA_PREPARATION.md#adding-meta-expert-fields-for-stage-3-router-supervision) for usage.

## Key Parameters

| Parameter              | Typical Value        |
| ---------------------- | -------------------- |
| `--training_part`      | `moe-rest`           |
| `--model_name_or_path` | Path to stitched MoE |
| `--data_path`          | Full data mix        |
| `--num_train_epochs`   | 3                    |
| `--learning_rate`      | 2e-5                 |

## Full Data Mix

Combine all domains for Stage 3:
```
tulu_coding-code_alpaca-oss_instruct-tulu_math-mammoth_cot_no_camel-tulu_g.growth
```

Ensure your data has `meta` field for router supervision.

## Script Usage

```bash
# Prerequisite: Run stitch_only.sh first to get the MoE model
MOE_MODEL_PATH=/path/to/moe_prebuilt \
FULL_DATA=tulu_coding-code_alpaca-oss_instruct-tulu_math-mammoth_cot_no_camel-tulu_g.growth \
OUTPUT_BASE=./ckpt \
./scripts/stage2/finetune_moe.sh
```
