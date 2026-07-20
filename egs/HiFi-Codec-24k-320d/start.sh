#!/bin/bash
source path.sh
set -e

log_root="logs"
# .lst save the wav path.
input_training_file="train.lst" 
input_validation_file="valid.lst"
training_epochs="${TRAINING_EPOCHS:-50}"

# Forward optional W&B settings to training script.
wandb_args=()
if [ -n "${WANDB_PROJECT}" ]; then
  wandb_args+=(--wandb_project "${WANDB_PROJECT}")
fi
if [ -n "${WANDB_NAME}" ]; then
  wandb_args+=(--wandb_name "${WANDB_NAME}")
fi
if [ -n "${WANDB_ENTITY}" ]; then
  wandb_args+=(--wandb_entity "${WANDB_ENTITY}")
fi

wandb_mode="${WANDB_MODE:-online}"
wandb_args+=(--wandb --wandb_mode "${wandb_mode}")

#mode=debug
mode=train

if [ "${mode}" == "debug" ]; then
  ## debug
  echo "Debug"
  log_root=${log_root}_debug
  export CUDA_VISIBLE_DEVICES=0
  python ${BIN_DIR}/train.py \
    --config config_24k_320d.json \
    --checkpoint_path ${log_root} \
    --input_training_file ${input_training_file} \
    --input_validation_file ${input_validation_file} \
    --checkpoint_interval 100 \
    --summary_interval 10 \
    --validation_interval 100 \

elif [ "$mode" == "train" ]; then
  ## train
  echo "Train model..."
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
  python ${BIN_DIR}/train.py \
    --config config_24k_320d.json \
    --checkpoint_path ${log_root} \
    --input_training_file ${input_training_file} \
    --input_validation_file ${input_validation_file} \
    --training_epochs ${training_epochs} \
    --checkpoint_interval 5000 \
    --summary_interval 100 \
    --validation_interval 5000 \
    "${wandb_args[@]}"
fi
