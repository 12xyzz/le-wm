#!/bin/bash
# Usage:
#   sbatch brics_eval.sh /path/to/*_object.ckpt
# Layout: $STABLEWM_HOME/checkpoints/<task>/<run>/ckpt/<stem>_object.ckpt
# Expects *_epoch_N_object.ckpt; sets output.filename to <CONFIG_NAME>_eval_results_N.txt
# Logs: copy stdout → <run>/logs/slurm-eval-<jobid>.out

#SBATCH --job-name=lewm-eval
#SBATCH --account=brics.u6mw
#SBATCH --output=/home/u6mw/xuanya.u6mw/le-wm/logs/eval.%j.out
#SBATCH --error=/home/u6mw/xuanya.u6mw/le-wm/logs/eval.%j.err
#SBATCH --time=00:30:00
#SBATCH --gpus=1

set -euo pipefail

LEWM_ROOT="/home/u6mw/xuanya.u6mw/le-wm"
export STABLEWM_HOME="/projects/u6mw/xuanya.u6mw/stablewm"
cd "${LEWM_ROOT}"

CKPT="${1:?Usage: sbatch brics_eval.sh /path/to/*_object.ckpt}"
CKPT_PREFIX="${STABLEWM_HOME}/checkpoints/"
[[ "${CKPT}" == "${CKPT_PREFIX}"* ]] || {
  echo "Error: checkpoint must be under ${CKPT_PREFIX}" >&2
  exit 1
}
REL="${CKPT#"${CKPT_PREFIX}"}"
CONFIG_NAME="${REL%%/*}"
TAIL="${REL#*/}"
RUN_SUB="${TAIL%/*}"
FILE="${TAIL##*/}"
STEM="${FILE%_object.ckpt}"
EPOCH="${STEM##*_epoch_}"
OUT_NAME="${CONFIG_NAME}_eval_results_${EPOCH}.txt"

# Quote policy value for Hydra
POLICY="${CONFIG_NAME}/${RUN_SUB}/${STEM}"
python eval.py --config-name="${CONFIG_NAME}" "policy='${POLICY}'" "output.filename=${OUT_NAME}"

LOG_DIR="${LEWM_ROOT}/logs"
RUN_DIR="$(dirname "$(dirname "$CKPT")")"

mkdir -p "${RUN_DIR}/logs"
cp "${LOG_DIR}/eval.${SLURM_JOB_ID}.out" "${RUN_DIR}/logs/slurm-eval-${SLURM_JOB_ID}.out"
