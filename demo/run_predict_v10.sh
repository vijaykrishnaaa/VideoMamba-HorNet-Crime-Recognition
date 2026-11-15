#!/bin/bash
#PBS -N V10_DEMO
#PBS -q workq
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=00:30:00
#PBS -j oe
#PBS -o /home/y_rajeshwar/vlil_40/demo-16novpanel/logs/predict_pbs.log

# Demo: the V9 model on one video or one folder of videos. Started by demo.sh:
#   bash demo.sh demo_videos/my_video.mp4
# (No ngpus=1: on this cluster that puts the job on Hold.)

EXP=${PBS_O_WORKDIR:-/home/y_rajeshwar/vlil_40/demo-16novpanel}   # the folder demo.sh was started from
OUT=$EXP/logs/predict_${PBS_JOBID%%.*}.log
CONDA_SH=/home/y_rajeshwar/anaconda3/etc/profile.d/conda.sh
mkdir -p $EXP/logs
exec >> "$OUT" 2>&1

[ -n "$VIDEO" ] || { echo "ERROR: no video given. Use: bash demo.sh <video>"; exit 2; }
source "$CONDA_SH" || { echo "ERROR: $CONDA_SH not found"; exit 2; }
conda activate videomamba_env || { echo "ERROR: conda env videomamba_env not found"; exit 2; }
cd /home/y_rajeshwar/vlil_40
export OMP_NUM_THREADS=4

python -u "$EXP/predict_v10.py" "$VIDEO" --seed "${SEED:-42}" 2>&1 | grep --line-buffered -v -E "^(Use checkpoint|Checkpoint number)"
STATUS=${PIPESTATUS[0]}
echo "===== EXIT CODE $STATUS ====="
exit $STATUS
