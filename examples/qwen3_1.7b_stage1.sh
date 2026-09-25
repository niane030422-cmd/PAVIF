#!/bin/bash
set -euo pipefail

# Run from the repository root. Edit the settings below before training.
HEAD_NODE="10.0.0.1"  # Replace with your Ray head node IP.
NNODES=4
GPUS_PER_NODE=4
DATA_ROOT="./data"
OUTPUT_ROOT="./outputs"
EXPERIMENT_NAME="qwen3_1.7b_stage1"

MODEL_PATH="/path/to/Qwen3-1.7B"

LOG_DIR="${OUTPUT_ROOT}/logs/${EXPERIMENT_NAME}"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

# Show training logs in the terminal and save a copy to the log file.
python3 -m verl.trainer.main \
    config=examples/stage1.yaml \
    ray_ip="${HEAD_NODE}:6379" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${GPUS_PER_NODE}" \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.reward.compute_score=verification \
    data.train_files="${DATA_ROOT}/train.parquet" \
    data.val_files="${DATA_ROOT}/test.parquet" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.load_checkpoint_path=null \
    trainer.save_checkpoint_path="${OUTPUT_ROOT}/checkpoints/instruct_stage1/${EXPERIMENT_NAME}" \
    trainer.rollout_data_dir="${OUTPUT_ROOT}/rollout_data/instruct_stage1/${EXPERIMENT_NAME}" \
    2>&1 | tee "${LOG_FILE}"
