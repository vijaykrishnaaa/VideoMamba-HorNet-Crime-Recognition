#!/bin/bash
#PBS -N V9_PREP
#PBS -q workq
#PBS -l select=1:ncpus=4:mem=32gb
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o /home/y_rajeshwar/vlil_40/v6_crime_moments/logs/prep_v9_pbs.log

# V9 prep: store the 121 validation crime videos whole (CPU only, a few minutes).
# Videos already stored are skipped, so running it again is harmless. Live progress:
#   tail -f ~/vlil_40/v6_crime_moments/logs/prep_v9_live.log

EXP=/home/y_rajeshwar/vlil_40/v6_crime_moments
LIVE=$EXP/logs/prep_v9_live.log
CONDA_SH=/home/y_rajeshwar/anaconda3/etc/profile.d/conda.sh
mkdir -p $EXP/logs
exec >> "$LIVE" 2>&1          # everything below (including errors) goes to the live log

echo "===== JOB $PBS_JOBID on $(hostname) started $(date) ====="
source "$CONDA_SH" || { echo "ERROR: $CONDA_SH not found"; exit 2; }
conda activate videomamba_env || { echo "ERROR: conda env videomamba_env not found"; exit 2; }
cd "$EXP"
export OMP_NUM_THREADS=4

python -u prepare_val_whole_v9.py
STATUS=$?
echo "===== EXIT CODE $STATUS at $(date) ====="
exit $STATUS
