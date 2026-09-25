#!/bin/bash
set -euo pipefail

# Run from the repository root. Edit the settings below before training.
HEAD_NODE="10.0.0.1"  # Replace with your Ray head node IP.
NNODES=4
GPUS_PER_NODE=4
DATA_ROOT="./data"
OUTPUT_ROOT="./outputs"
EXPERIMENT_NAME="qwen3_8b_stage2"

# Start from a Stage 1 checkpoint. Replace global_step_50 as needed.
MODEL_PATH="${OUTPUT_ROOT}/checkpoints/instruct_stage1/qwen3_8b_stage1/global_step_50/actor/huggingface"
# Keep null for a new Stage 2 run, or set a Stage 2 global_step directory to resume.
RESUME_CHECKPOINT_PATH=null

LOG_DIR="${OUTPUT_ROOT}/logs/${EXPERIMENT_NAME}/stage2"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/train_$(date +%Y%m%d_%H%M%S).log"

# Show training logs in the terminal and save a copy to the log file.
python3 -m verl.trainer.stage2_main \
    config=examples/stage2.yaml \
    ray_ip="${HEAD_NODE}:6379" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${GPUS_PER_NODE}" \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.reward.compute_score=instruction \
    data.train_files="${DATA_ROOT}/stage2/train.parquet" \
    data.val_files="${DATA_ROOT}/stage2/test.parquet" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.load_checkpoint_path="${RESUME_CHECKPOINT_PATH}" \
    trainer.save_checkpoint_path="${OUTPUT_ROOT}/checkpoints/instruct_stage2/${EXPERIMENT_NAME}" \
    trainer.rollout_data_dir="${OUTPUT_ROOT}/rollout_data/instruct_stage2/${EXPERIMENT_NAME}" \
    2>&1 | tee "${LOG_FILE}"
