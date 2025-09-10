#!/bin/bash
# Stage 3: Finetune stitched MoE with moe-rest (FFN complement)
# Prerequisites: Run stitch_only.sh first to get the prebuilt MoE model

set -e
export OMP_NUM_THREADS=1

# ============ CONFIGURATION ============
# Path to stitched MoE model (output of stitch_only.sh)
MOE_MODEL_PATH="${MOE_MODEL_PATH:-/path/to/moe_prebuilt}"

# Full data for finetune (abbreviations separated by -)
FULL_DATA="${FULL_DATA:-tulu_coding-code_alpaca-oss_instruct-tulu_math-mammoth_cot_no_aqua-tulu_g.growth}"

OUTPUT_BASE="${OUTPUT_BASE:-./ckpt}"
# =======================================

DEBUG_MODE=false
if [[ -z $WORLD_SIZE || -z $RANK || -z $MASTER_ADDR || -z $MASTER_PORT ]]; then
    DEBUG_MODE=true
fi

if $DEBUG_MODE; then
    export WANDB_MODE=disabled
    N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    DISTRIBUTED_ARGS="--nproc_per_node=$N_GPUS --master_port=6656"
else
    N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    DISTRIBUTED_ARGS="--nproc_per_node ${N_GPUS} --nnodes ${WORLD_SIZE} --node_rank ${RANK} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT}"
fi

time_str=$(date +"%m.%d_%H.%M.%S")
exp_name="moe-rest-fulldata-lr2e-5-${time_str}"
save_folder="${OUTPUT_BASE}/s2-moe-rest/${exp_name}"

cd "$(dirname "$0")/../.."
torchrun $DISTRIBUTED_ARGS train_moe.py \
    --exp_name "$exp_name" --group_name "s2-moe-rest" \
    --model_name_or_path "$MOE_MODEL_PATH" \
    --training_part "moe-rest" \
    --max_seq_length 2048 \
    --dataloader_num_workers 8 \
    --data_format chatml \
    --force_system_prompt True \
    --data_path "$FULL_DATA" \
    --output_dir "$save_folder" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 0.1 \
    --save_total_limit -1 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --max_grad_norm 1.0 \
    --gradient_checkpointing True \
    --deepspeed "./configs/zero2_bf16_cos_detailed.json" \
    --bf16 True

echo "Done. Final model: ${save_folder}/final"
