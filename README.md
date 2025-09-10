# Sewing-MoE: Code Release

This repository contains the code for training domain-specific Mixture-of-Experts (MoE) models via a "sewing" pipeline: train experts on domain data with FFN-only updates, stitch them into an MoE, then finetune the complement (non-FFN) on full data.

## Pipeline Overview

**Stage 0**: Data Preparation
- Tag/label data from various sources → math / code / general
- Output: domain expert training data (JSONL)

**Stage 1**: Domain Expert Training
- Train one LLaMA model per domain (math, coding, general)
- Only FFN (mlp) parameters are trainable; rest frozen
- Output: expert checkpoints

**Stage 2**: Stitch Experts into MoE
- Load expert checkpoints, merge FFNs into MoE architecture
- Embedding/lm_head: averaged; Attention: shared; MLP → experts
- Output: prebuilt MoE model

**Stage 3**: Finetune on Full Data
- Train the stitched MoE with full mixed data
- Trainable: emb, attn, router, lm_head (complement of FFN)
- FFN experts remain frozen
- Output: final MoE model


## Quick Start

**Note**: Base LLaMA models must be available locally. Set `local_files_only=False` in `train_moe.py` if loading from HuggingFace Hub.

1. **Configure paths** in `factory.py` (`DATA_ROOT`) and in the shell scripts (`BASE_MODEL_PATH`, expert paths).
2. **Prepare data** (Stage 0): See [STAGE0_DATA_PREPARATION.md](docs/STAGE0_DATA_PREPARATION.md) for details.
3. **Train domain experts** (Stage 1): See [STAGE1_DOMAIN_EXPERTS.md](docs/STAGE1_DOMAIN_EXPERTS.md) for details.
4. **Stitch experts** (Stage 2): See [STAGE2_STITCH.md](docs/STAGE2_STITCH.md) for details.
5. **Finetune stitched MoE** (Stage 3): See [STAGE3_FINETUNE.md](docs/STAGE3_FINETUNE.md) for details.

## Directory Structure

```
coe-released/
├── README.md                 # This file
├── factory.py                # Data path mapping (configure DATA_ROOT)
├── train_moe.py              # Main training script
├── trainer_moe.py            # Custom Trainer for MoE
├── stitch_experts.py         # Stitch experts into MoE
├── data_prepare/             # Data preparation scripts
│   ├── prepare_tagged_tulu.py
│   ├── prepare_mammoth.py
│   ├── prepare_code_alpaca.py
│   ├── prepare_oss_instruct.py
│   └── add_meta_expert.py    # Add meta/data fields for Stage 3 router supervision
├── scripts/
│   ├── stage1/               # Domain expert training
│   │   └── train_domain_expert.sh
│   └── stage2/               # Stitch & finetune
│       ├── stitch_only.sh
│       └── finetune_moe.sh
├── models_dev/               # MoE-LLaMA model
│   ├── configuration_moe_llama.py
│   └── modeling_moe_llama.py
├── configs/
│   └── zero2_bf16_cos_detailed.json
└── docs/                     # Stage-specific documentation
    ├── STAGE0_DATA_PREPARATION.md
    ├── STAGE1_DOMAIN_EXPERTS.md
    ├── STAGE2_STITCH.md
    └── STAGE3_FINETUNE.md
```

## Training Modes

| `--training_part` | Trainable Parameters       | Use Case                        |
| ----------------- | -------------------------- | ------------------------------- |
| `ffn`             | MLP only                   | Stage 1: domain expert training |
| `moe-full`        | All                        | Full finetune of MoE            |
| `moe-rest`        | emb, attn, router, lm_head | Stage 3: complement of FFN      |
| `moe-router`      | Router (gate) only         | Router-only finetune            |

## Data Format

Training data: JSONL with `messages` field (list of `{role, content}`). Optional `meta` for router targets when using `chatml` + MoE.

## Citation

```bibtex
@inproceedings{yang2025fine,
  title={Fine-tuning language models with collaborative and semantic experts},
  author={Yang, Jiaxi and Hui, Binyuan and Yang, Min and Yang, Jian and Zhang, Lei and Qu, Qiang and Lin, Junyang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={24},
  pages={25624--25632},
  year={2025}
}
```