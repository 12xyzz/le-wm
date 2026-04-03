#!/bin/bash
# Usage:
#   sbatch brics_train.sh <data>
# Run dir: $STABLEWM_HOME/<data>/<mmddHHMM>_<SLURM_JOB_ID>/ (subdir from config).
# Logs: copy stdout → <run>/logs/slurm-train-<jobid>.out

#SBATCH --job-name=lewm-train
#SBATCH --output=/home/u6gn/xuanya.u6gn/le-wm/logs/train.%j.out
#SBATCH --error=/home/u6gn/xuanya.u6gn/le-wm/logs/train.%j.err
#SBATCH --time=12:00:00
#SBATCH --gpus=1

set -euo pipefail

LEWM_ROOT="/home/u6gn/xuanya.u6gn/le-wm"
export STABLEWM_HOME="/scratch/u6gn/xuanya.u6gn/stablewm"
cd "${LEWM_ROOT}"

DATA="${1:?Usage: sbatch brics_train.sh <data>}"
python train.py "data=${DATA}"

LOG_DIR="${LEWM_ROOT}/logs"
shopt -s nullglob
RUN_DIRS=("${STABLEWM_HOME}/${DATA}/"*_"${SLURM_JOB_ID}")
shopt -u nullglob
RUN_DIR="${RUN_DIRS[0]}"

cp "${LOG_DIR}/train.${SLURM_JOB_ID}.out" "${RUN_DIR}/logs/slurm-train-${SLURM_JOB_ID}.out"