#!/bin/bash
# Usage:
#   sbatch brics_train.sh <data> [config_name]
# Run dir: $STABLEWM_HOME/checkpoints/<data>/<mmddHHMM>_<SLURM_JOB_ID>/ (subdir from config).
# Logs: copy stdout → <run>/logs/slurm-train-<jobid>.out

#SBATCH --job-name=lewm-train
#SBATCH --account=brics.u6mw
#SBATCH --output=/home/u6mw/xuanya.u6mw/le-wm/logs/train.%j.out
#SBATCH --error=/home/u6mw/xuanya.u6mw/le-wm/logs/train.%j.err
#SBATCH --time=12:00:00
#SBATCH --gpus=1

set -euo pipefail

LEWM_ROOT="/home/u6mw/xuanya.u6mw/le-wm"
export STABLEWM_HOME="/projects/u6mw/xuanya.u6mw/stablewm"
export SPT_CACHE_DIR="${SPT_CACHE_DIR:-${STABLEWM_HOME}/.cache}"
mkdir -p "${SPT_CACHE_DIR}"
cd "${LEWM_ROOT}"

DATA="${1:?Usage: sbatch brics_train.sh <data> [config_name]}"
CFG="${2:-lewm}"
python train.py --config-name "${CFG}" "data=${DATA}"

LOG_DIR="${LEWM_ROOT}/logs"
shopt -s nullglob
RUN_DIRS=("${STABLEWM_HOME}/checkpoints/${DATA}/"*_"${SLURM_JOB_ID}")
shopt -u nullglob
RUN_DIR="${RUN_DIRS[0]}"

mkdir -p "${RUN_DIR}/logs"
cp "${LOG_DIR}/train.${SLURM_JOB_ID}.out" "${RUN_DIR}/logs/slurm-train-${SLURM_JOB_ID}.out"