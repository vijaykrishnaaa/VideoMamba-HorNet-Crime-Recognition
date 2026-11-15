#!/bin/bash
# One line for the demo: sends the video (or a folder of videos) to a GPU node, waits, shows the answer.
#   bash demo.sh demo_videos/my_video.mp4
#   bash demo.sh demo_videos            (every video in the folder, the model is loaded once)
# The file or folder name must not contain spaces or commas.
cd "$(dirname "$0")" || exit 1
[ $# -eq 1 ] || { echo "usage: bash demo.sh <video file or folder>"; exit 1; }
VIDEO=$(readlink -f "$1")
[ -e "$VIDEO" ] || { echo "not found: $1"; exit 1; }
case "$VIDEO" in *" "*|*","*) echo "rename it first: no spaces or commas in the name"; exit 1;; esac

id=$(qsub -v VIDEO="$VIDEO" run_predict_v10.sh) || exit 1
out=logs/predict_${id%%.*}.log
echo "job $id sent. Waiting for a free GPU node..."
while qstat "$id" > /dev/null 2>&1 && [ ! -s "$out" ]; do sleep 2; printf "."; done
echo " running (loading the model takes about half a minute)"
while qstat "$id" > /dev/null 2>&1 && ! grep -q "EXIT CODE" "$out" 2>/dev/null; do sleep 2; printf "."; done
echo
cat "$out"
