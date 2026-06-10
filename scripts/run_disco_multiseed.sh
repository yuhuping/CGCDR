#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 TASK GPU [TAG]"
  exit 1
fi

TASK="$1"
GPU="$2"
TAG="${3:-dynamic}"
PYTHON="${PYTHON:-/home/yuhp/.conda/envs/torch/bin/python}"
SEEDS=(2021 2022 2023 2024 2025)

mkdir -p log/DisCo

for SEED in "${SEEDS[@]}"; do
  INFO="${TAG}_seed${SEED}"
  echo "[$(date '+%F %T')] ${TASK} seed=${SEED} GPU=${GPU}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" run.py \
    --model=DisCo \
    --Task="${TASK}" \
    --seed="${SEED}" \
    --epoch=100 \
    --stopping_step=20 \
    --emb_dim=128 \
    --batch_size=1024 \
    --num_intents=4 \
    --graph_neighbors=10 \
    --random_walk_steps=3 \
    --disco_beta=0.3 \
    --disco_lambda=0.1 \
    --dynamic_neg_sampling \
    --info="${INFO}" \
    > "log/DisCo/${TASK}_${INFO}_nohup.log" 2>&1
done

"${PYTHON}" scripts/summarize_disco_multiseed.py \
  --task="${TASK}" \
  --tag="${TAG}" \
  --seeds "${SEEDS[@]}" \
  | tee "log/DisCo/${TASK}_${TAG}_summary.txt"
