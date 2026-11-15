#!/bin/bash
#PBS -N V10_2STAGE
#PBS -q workq
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=08:00:00
#PBS -j oe
#PBS -o /home/y_rajeshwar/vlil_40/v10_experiments/logs/two_stage_pbs.log

# V10: crime-or-not, then which crime, read from the three V9 models. No training.
# First a short smoke run on a few videos; the full test starts only if it passes.
# (No ngpus=1: on this cluster that puts the job on Hold.)
# Live progress:  tail -f ~/vlil_40/v10_experiments/logs/two_stage_live.log

EXP=/home/y_rajeshwar/vlil_40/v10_experiments
LIVE=$EXP/logs/two_stage_live.log
CONDA_SH=/home/y_rajeshwar/anaconda3/etc/profile.d/conda.sh
mkdir -p $EXP/logs
exec >> "$LIVE" 2>&1          # everything below (including errors) goes to the live log

echo "===== JOB $PBS_JOBID on $(hostname) started $(date) ====="
nvidia-smi
source "$CONDA_SH" || { echo "ERROR: $CONDA_SH not found"; exit 2; }
conda activate videomamba_env || { echo "ERROR: conda env videomamba_env not found"; exit 2; }
cd /home/y_rajeshwar/vlil_40
export OMP_NUM_THREADS=4

python -u "$EXP/two_stage_v10.py" --smoke && python -u "$EXP/two_stage_v10.py"
STATUS=$?
echo "===== EXIT CODE $STATUS at $(date) ====="
exit $STATUS
