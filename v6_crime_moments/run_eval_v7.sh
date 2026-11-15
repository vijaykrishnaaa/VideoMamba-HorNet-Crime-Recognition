#!/bin/bash
#PBS -N V7_EVAL
#PBS -q workq
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o /home/y_rajeshwar/vlil_40/v6_crime_moments/logs/eval_v7_pbs.log

# V7 step 1: cache the 140 whole test videos, then test the 3 V6 models on the
# CVPR 2024/2025 protocol (no training). Live progress:
#   tail -f ~/vlil_40/v6_crime_moments/logs/eval_v7_live.log

EXP=/home/y_rajeshwar/vlil_40/v6_crime_moments
LIVE=$EXP/logs/eval_v7_live.log
CONDA_SH=/home/y_rajeshwar/anaconda3/etc/profile.d/conda.sh
mkdir -p $EXP/logs
exec >> "$LIVE" 2>&1          # everything below (including errors) goes to the live log

echo "===== JOB $PBS_JOBID on $(hostname) started $(date) ====="
nvidia-smi
source "$CONDA_SH" || { echo "ERROR: $CONDA_SH not found"; exit 2; }
conda activate videomamba_env || { echo "ERROR: conda env videomamba_env not found"; exit 2; }
cd "$EXP"
export OMP_NUM_THREADS=4

python -u prepare_test_whole_v7.py && python -u eval_v7_protocol.py
STATUS=$?
echo "===== EXIT CODE $STATUS at $(date) ====="
exit $STATUS
