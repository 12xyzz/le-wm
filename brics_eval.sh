#!/bin/bash
# Usage:
#   sbatch brics_eval.sh /path/to/*_object.ckpt
# Layout: $STABLEWM_HOME/<task>/<run>/ckpt/<stem>_object.ckpt
# Expects *_epoch_N_object.ckpt; sets output.filename to <CONFIG_NAME>_eval_results_N.txt
# Logs: copy stdout → <run>/logs/slurm-eval-<jobid>.out

#SBATCH --job-name=lewm-eval
#SBATCH --output=/home/u6gn/xuanya.u6gn/le-wm/logs/eval.%j.out
#SBATCH --error=/home/u6gn/xuanya.u6gn/le-wm/logs/eval.%j.err
#SBATCH --time=12:00:00
#SBATCH --gpus=1

set -euo pipefail

LEWM_ROOT="/home/u6gn/xuanya.u6gn/le-wm"
export STABLEWM_HOME="/projects/u6gn/xuanya.u6gn/stablewm"
cd "${LEWM_ROOT}"

CKPT="${1:?Usage: sbatch brics_eval.sh /path/to/*_object.ckpt}"
REL="${CKPT#"$STABLEWM_HOME"/}"
CONFIG_NAME="${REL%%/*}"
TAIL="${REL#*/}"
RUN_SUB="${TAIL%/*}"
FILE="${TAIL##*/}"
STEM="${FILE%_object.ckpt}"
EPOCH="${STEM##*_epoch_}"
OUT_NAME="${CONFIG_NAME}_eval_results_${EPOCH}.txt"

python eval.py --config-name="${CONFIG_NAME}" policy="${CONFIG_NAME}/${RUN_SUB}/${STEM}" "output.filename=${OUT_NAME}"

LOG_DIR="${LEWM_ROOT}/logs"
RUN_DIR="$(dirname "$(dirname "$CKPT")")"

mkdir -p "${RUN_DIR}/logs"
cp "${LOG_DIR}/eval.${SLURM_JOB_ID}.out" "${RUN_DIR}/logs/slurm-eval-${SLURM_JOB_ID}.out"
