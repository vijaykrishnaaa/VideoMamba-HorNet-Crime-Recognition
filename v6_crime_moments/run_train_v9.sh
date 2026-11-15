#!/bin/bash
#PBS -N V9_TRAIN
#PBS -q workq
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=24:00:00
#PBS -j oe
#PBS -o /home/y_rajeshwar/vlil_40/v6_crime_moments/logs/train_v9_pbs.log

# V9: train one seed, pick the best epoch on whole-video validation, test on the 140 test videos.
# (No ngpus=1: on this cluster that puts the job on Hold; the node's GPU is visible anyway.)
# Normally started by submit_v9.sh, which queues the jobs ONE AFTER ANOTHER (never together:
# jobs started together can land on the same node and run its one 16 GB GPU out of memory).
#   qsub -v SEED=42 run_train_v9.sh                       the main model, seed 42
#   qsub -v SEED=42,GNCONV=st run_train_v9.sh             updated gnConv
#   qsub -v SEED=42,SMOKE=1 run_train_v9.sh               2-minute check of the whole pipeline
# Live progress:  tail -f ~/vlil_40/v6_crime_moments/logs/v9_guide_ucf101_seed42_live.log
# If the job is killed (walltime / GPU taken), submit the same command again: it resumes from last.pth.

SEED=${SEED:-42}
GNCONV=${GNCONV:-guide}       # guide | st | none
INIT=${INIT:-ucf101}          # ucf101 | k400
TAU=${TAU:-1}                 # 1 = balanced crimes, 0 = off
SMOKE=${SMOKE:-0}
EXP=/home/y_rajeshwar/vlil_40/v6_crime_moments
NAME=v9_${GNCONV}_${INIT}_seed${SEED}
[ "$TAU" != "1" ] && NAME=${NAME}_tau${TAU}
[ "$SMOKE" != "0" ] && NAME=${NAME}_smoke
LIVE=$EXP/logs/${NAME}_live.log
CONDA_SH=/home/y_rajeshwar/anaconda3/etc/profile.d/conda.sh
mkdir -p $EXP/logs
exec >> "$LIVE" 2>&1          # everything below (including errors) goes to the live log

echo "===== JOB $PBS_JOBID on $(hostname) started $(date) | SEED=$SEED GNCONV=$GNCONV INIT=$INIT TAU=$TAU SMOKE=$SMOKE ====="
nvidia-smi
source "$CONDA_SH" || { echo "ERROR: $CONDA_SH not found"; exit 2; }
conda activate videomamba_env || { echo "ERROR: conda env videomamba_env not found (run v5_ucf_crime/restore_env.sh)"; exit 2; }
cd /home/y_rajeshwar/vlil_40
export OMP_NUM_THREADS=4

ARGS="--seed $SEED --gnconv $GNCONV --init $INIT --tau $TAU"
if [ "$SMOKE" != "0" ]; then
  python -u "$EXP/train_v9.py" $ARGS --smoke 1 && python -u "$EXP/train_v9.py" $ARGS --smoke 2
else
  python -u "$EXP/train_v9.py" $ARGS
fi
STATUS=$?
echo "===== EXIT CODE $STATUS at $(date) ====="
exit $STATUS
