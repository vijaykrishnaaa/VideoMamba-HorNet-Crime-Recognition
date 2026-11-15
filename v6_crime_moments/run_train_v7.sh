#!/bin/bash
#PBS -N V7_TRAIN
#PBS -q workq
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /home/y_rajeshwar/vlil_40/v6_crime_moments/logs/train_v7_pbs.log

# V7 step 2: train with the Background class, then test on the 140 whole test videos.
# (No ngpus=1: on this cluster that puts the job on Hold; the node's GPU is visible anyway.)
#   qsub -v SEED=42 run_train_v7.sh       (then SEED=7 and SEED=123, ONE AFTER ANOTHER:
#   qsub -W depend=afterany:<previous job id> -v SEED=7 run_train_v7.sh; jobs started together can
#   land on the same node and run its one 16 GB GPU out of memory)
# Live progress:  tail -f ~/vlil_40/v6_crime_moments/runs_v7/seed42_ucf101/logs/train_live.log
# If the job is killed (walltime / GPU taken), submit the same command again: it resumes from last.pth.

SEED=${SEED:-42}
EXP=/home/y_rajeshwar/vlil_40/v6_crime_moments
RUN=$EXP/runs_v7/seed${SEED}_ucf101
LIVE=$RUN/logs/train_live.log
CONDA_SH=/home/y_rajeshwar/anaconda3/etc/profile.d/conda.sh
mkdir -p $RUN/logs $EXP/logs
exec >> "$LIVE" 2>&1          # everything below (including errors) goes to the live log

echo "===== JOB $PBS_JOBID on $(hostname) started $(date) | SEED=$SEED ====="
nvidia-smi
source "$CONDA_SH" || { echo "ERROR: $CONDA_SH not found"; exit 2; }
conda activate videomamba_env || { echo "ERROR: conda env videomamba_env not found (run v5_ucf_crime/restore_env.sh)"; exit 2; }
cd /home/y_rajeshwar/vlil_40
export OMP_NUM_THREADS=4

python -u "$EXP/train_v7.py" --seed "$SEED" --init ucf101
STATUS=$?
echo "===== EXIT CODE $STATUS at $(date) ====="
exit $STATUS
