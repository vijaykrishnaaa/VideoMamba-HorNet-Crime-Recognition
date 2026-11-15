#!/bin/bash
# Queues one V9 variant: prep -> 2-minute smoke test -> seeds 42, 7, 123, ONE AFTER ANOTHER.
# The seeds start only if the smoke test passes, so a mistake costs 2 minutes, not a night.
#   bash submit_v9.sh                       the main model (the guide's gnConv, UCF-101 start)
#   AFTER=12345 bash submit_v9.sh st        updated gnConv, starting when job 12345 has ended
#   AFTER=12345 bash submit_v9.sh none      no gnConv (shows what gnConv adds)
#   AFTER=12345 bash submit_v9.sh guide k400      Kinetics-400 start instead of UCF-101
#   AFTER=12345 TAU=0 bash submit_v9.sh     without the balanced loss
# Always give AFTER=<last job number of the chain before> when something is still queued or running.
cd "$(dirname "$0")" || exit 1
G=${1:-guide}
I=${2:-ucf101}
V="GNCONV=$G,INIT=$I,TAU=${TAU:-1}"

if [ -n "$AFTER" ]; then
  prep=$(qsub -W depend=afterany:$AFTER run_prepare_v9.sh) || exit 1
else
  prep=$(qsub run_prepare_v9.sh) || exit 1
fi
echo "prep        -> job $prep"
prev=$(qsub -W depend=afterok:$prep -v SMOKE=1,SEED=42,$V run_train_v9.sh) || exit 1
echo "smoke test  -> job $prev"
dep=afterok                  # seed 42 needs the smoke test to have passed
for seed in 42 7 123; do
  prev=$(qsub -W depend=$dep:$prev -v SEED=$seed,$V run_train_v9.sh) || exit 1
  echo "$G $I seed $seed -> job $prev"
  dep=afterany               # later seeds start even if the one before failed
done
echo "To queue another variant after this one:  AFTER=${prev%%.*} bash submit_v9.sh st"
