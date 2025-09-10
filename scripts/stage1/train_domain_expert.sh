#!/bin/bash
# Stage 1: Train domain experts (FFN-only on LLaMA)
# Usage: ./train_domain_expert.sh <model_name> <batch_size> <data_names> <learning_rate>
# Example: ./train_domain_expert.sh 7b 2 tulu_coding-code_alpaca 2e-5

set -e
export OMP_NUM_THREADS=1

# ============ CONFIGURATION ============
# Set these to your paths
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/path/to/llama-2/7b-base}"  # LLaMA-2 7B base
OUTPUT_BASE="${OUTPUT_BASE:-./ckpt}"
# =======================================

DEBUG_MODE=false
if [[ -z $WORLD_SIZE || -z $RANK || -z $MASTER_ADDR || -z $MASTER_PORT ]]; then
    DEBUG_MODE=true
fi

if $DEBUG_MODE; then
    export WANDB_MODE=disabled
    N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    DISTRIBUTED_ARGS="--nproc_per_node=$N_GPUS --master_port=6655"
else
    N_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    DISTRIBUTED_ARGS="--nproc_per_node ${N_GPUS} --nnodes ${WORLD_SIZE} --node_rank ${RANK} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT}"
fi

model_name=${1:-7b}
train_per_device_bs=${2:-2}
data_names=${3:-tulu_coding-code_alpaca}
learning_rate=${4:-2e-5}

task_name="s1-domain-expert-ffn"
model_path="${BASE_MODEL_PATH}"
shell_name=$(basename "$0" .sh)
time_str=$(date +"%m.%d_%H.%M.%S")
exp_name="${model_name}-${data_names}-lr${learning_rate}-${time_str}"
save_folder="${OUTPUT_BASE}/${task_name}/${exp_name}"

echo "Model: ${model_path}"
echo "Data: ${data_names}"
echo "Exp: ${exp_name}"
echo "Save: ${save_folder}"

cd "$(dirname "$0")/../.."
torchrun $DISTRIBUTED_ARGS train_moe.py \
    --exp_name "$exp_name" --group_name "$task_name" \
    --model_name_or_path "$model_path" \
    --training_part "ffn" \
    --max_seq_length 2048 \
    --dataloader_num_workers 8 \
    --data_format chatml \
    --force_system_prompt True \
    --data_path "$data_names" \
    --output_dir "$save_folder" \
    --num_train_epochs 5 \
    --per_device_train_batch_size $train_per_device_bs \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 0.05 \
    --save_total_limit -1 \
    --learning_rate $learning_rate \
    --weight_decay 0.0 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.03 \
    --logging_steps 1 \
    --max_grad_norm 1.0 \
    --gradient_checkpointing True \
    --deepspeed "./configs/zero2_bf16_cos_detailed.json" \
    --bf16 True
