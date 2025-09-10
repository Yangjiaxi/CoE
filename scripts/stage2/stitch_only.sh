#!/bin/bash
# Stitch experts into MoE (standalone, no finetune)
# Usage: Set EXPERT_PATHS or pass as env, e.g.:
#   MATH_EXPERT=/path/math CODING_EXPERT=/path/coding GENERAL_EXPERT=/path/general ./stitch_only.sh

MATH_EXPERT="${MATH_EXPERT:-/path/to/math/final}"
CODING_EXPERT="${CODING_EXPERT:-/path/to/coding/final}"
GENERAL_EXPERT="${GENERAL_EXPERT:-/path/to/general/final}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/moe_prebuilt}"

cd "$(dirname "$0")/../.."
python stitch_experts.py \
    --experts "math=${MATH_EXPERT}" "coding=${CODING_EXPERT}" "general=${GENERAL_EXPERT}" \
    --output_dir "$OUTPUT_DIR" \
    --num_experts_per_tok 2 \
    --router_loss_type ce \
    --router_loss_alpha 1.0

echo "MoE model saved to: $OUTPUT_DIR"

