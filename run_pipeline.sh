#!/usr/bin/env bash
# run_pipeline.sh
#
# Chains training (self-training / on-the-fly synthetic degradation) and
# batched evaluation into a single command for NAFNet-SR semiconductor
# image restoration.
#
# Usage:
#   ./run_pipeline.sh
#
# Override any setting by exporting an env var before running, e.g.:
#   BATCH_SIZE=32 UPSCALE_FACTOR=2 ./run_pipeline.sh
#
# To skip training and only run evaluation with existing weights:
#   SKIP_TRAIN=1 ./run_pipeline.sh

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------
CLEAN_DIR="${CLEAN_DIR:-data/clean}"
VAL_CLEAN_DIR="${VAL_CLEAN_DIR:-data/val_clean}"
TEST_INPUT_DIR="${TEST_INPUT_DIR:-data/test_input}"
OUTPUT_DIR="${OUTPUT_DIR:-data/restored_output}"
WEIGHTS_PATH="${WEIGHTS_PATH:-final_model_weights.pt}"

EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PATCH_SIZE="${PATCH_SIZE:-128}"
UPSCALE_FACTOR="${UPSCALE_FACTOR:-4}"
WIDTH="${WIDTH:-32}"
NUM_BLOCKS="${NUM_BLOCKS:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"

SKIP_TRAIN="${SKIP_TRAIN:-0}"

echo "=============================================================="
echo " NAFNet-SR pipeline"
echo "=============================================================="
echo " clean_dir        : ${CLEAN_DIR}"
echo " val_clean_dir     : ${VAL_CLEAN_DIR}"
echo " test_input_dir    : ${TEST_INPUT_DIR}"
echo " output_dir        : ${OUTPUT_DIR}"
echo " weights_path      : ${WEIGHTS_PATH}"
echo " epochs            : ${EPOCHS}"
echo " batch_size (train): ${BATCH_SIZE}"
echo " batch_size (eval) : ${EVAL_BATCH_SIZE}"
echo " patch_size        : ${PATCH_SIZE}"
echo " upscale_factor    : ${UPSCALE_FACTOR}"
echo " width / num_blocks: ${WIDTH} / ${NUM_BLOCKS}"
echo "=============================================================="

if [ "${SKIP_TRAIN}" != "1" ]; then
    echo
    echo "----- Step 1/2: Training (self-training mode) -----"
    python train.py \
        --self_train \
        --clean_dir "${CLEAN_DIR}" \
        --val_clean_dir "${VAL_CLEAN_DIR}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --patch_size "${PATCH_SIZE}" \
        --upscale_factor "${UPSCALE_FACTOR}" \
        --width "${WIDTH}" \
        --num_blocks "${NUM_BLOCKS}" \
        --num_workers "${NUM_WORKERS}" \
        --output_path "${WEIGHTS_PATH}"
else
    echo
    echo "----- Step 1/2: Skipped (SKIP_TRAIN=1) -----"
fi

echo
echo "----- Step 2/2: Batched FP16 evaluation -----"
python evaluation.py \
    --input_dir "${TEST_INPUT_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --weights_path "${WEIGHTS_PATH}" \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --upscale_factor "${UPSCALE_FACTOR}" \
    --width "${WIDTH}" \
    --num_blocks "${NUM_BLOCKS}" \
    --num_workers "${NUM_WORKERS}"

echo
echo "Done. Restored images saved to ${OUTPUT_DIR}"
