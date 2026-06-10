#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
mkdir -p log/DisCo

run_queue() {
  local gpu="$1"
  shift
  local task
  for task in "$@"; do
    scripts/run_disco_multiseed.sh "${task}" "${gpu}" dynamic
  done
}

run_queue 1 Sport_Cloth Game_Video Movies_CD Elec_Phone &
gpu1_pid=$!
run_queue 3 Cloth_Sport Video_Game CD_Movies Phone_Elec &
gpu3_pid=$!

wait "${gpu1_pid}"
wait "${gpu3_pid}"
