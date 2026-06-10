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

run_queue 1 Sport_Cloth Movies_CD Phone_Elec &
gpu1_pid=$!
run_queue 2 Game_Video CD_Movies &
gpu2_pid=$!
run_queue 3 Video_Game Elec_Phone &
gpu3_pid=$!

wait "${gpu1_pid}"
wait "${gpu2_pid}"
wait "${gpu3_pid}"
