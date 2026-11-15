# V9 run steps (final UCF-Crime model)

V9 uses the same HPC folder as V6 and V7, because it reuses their data and code.
What V9 changes and why is in `V9_EXPLAINED.md`.

## What must already be on the HPC

In `~/vlil_40/v6_crime_moments/` (all there since the V7 runs of 26-27 October):

| Needed | Made by |
|---|---|
| `train_v6.py`, `eval_v7_protocol.py`, `prepare_data_v6.py`, `prepare_test_whole_v7.py`, `mamba_simple_v7.py` | V6 / V7 upload |
| `frame_cache/`, `splits/train.csv`, `val.csv`, `classes.txt` | `run_prepare_v6.sh` |
| `frame_cache_bg/`, `splits/train_bg.csv`, `val_bg.csv`, `classes_v7.txt` | `run_prepare_bg_v7.sh` |
| `frame_cache_whole/`, `splits/test_whole.csv` | `run_eval_v7.sh` |

## Step 1: upload 6 files (WinSCP)

Left panel: `C:\Users\LENOVO\Desktop\Vlil40\Current work\v6_crime_moments\`
Right panel: `/home/y_rajeshwar/vlil_40/v6_crime_moments/`

Drag these 6 files across (do not replace the folder, only add the files):

| File | What it does |
|---|---|
| `train_v9.py` | Training, whole-video validation every epoch, one final test |
| `gnconv_st_v9.py` | The updated gnConv (used only by the `st` variant) |
| `prepare_val_whole_v9.py` | Stores the 121 validation videos whole (CPU, a few minutes) |
| `run_prepare_v9.sh` | PBS job for the prep |
| `run_train_v9.sh` | PBS job for one training run |
| `submit_v9.sh` | Queues prep, smoke test and the three seeds one after another |

## Step 2: fix line endings and check the V7 data is there (PuTTY)

```text
cd ~/vlil_40/v6_crime_moments
sed -i 's/\r$//' *.py *.sh
ls splits/train_bg.csv splits/val_bg.csv splits/classes_v7.txt splits/test_whole.csv
```

The `ls` must list all four files. If one is missing, stop and tell me.

## Step 3: submit the main model

```text
bash submit_v9.sh
qstat -u y_rajeshwar
```

It prints five job numbers: prep, smoke test, seed 42, seed 7, seed 123. Waiting jobs show **H**
(they wait for the job before them). Write down the last job number.

The smoke test trains on 16 clips for 2 epochs, stops, resumes, validates and tests on 3 videos.
It takes a few minutes and its numbers mean nothing. Its only job is to prove that every part
works before the real runs start. If it fails, the three seeds do not start (PBS removes them,
or leaves them on hold: then delete them with `qdel <job number>`).

## Step 4: watch (Ctrl+C stops watching, not the job)

```text
tail -f ~/vlil_40/v6_crime_moments/logs/prep_v9_live.log
tail -f ~/vlil_40/v6_crime_moments/logs/v9_guide_ucf101_seed42_smoke_live.log
tail -f ~/vlil_40/v6_crime_moments/logs/v9_guide_ucf101_seed42_live.log
```

| Log | Must show |
|---|---|
| prep | `V9 PREP DONE: 121/121 validation videos cached` |
| smoke | `SMOKE STAGE 1 OK`, then `RESUMED from ...`, then `SMOKE TEST PASSED` |
| seed 42 | `[CHECK]` lines all OK, the data table with the loss offsets, then two lines per epoch |

Each epoch prints the clip numbers (train, clip val, gap) and one `WHOLE-VIDEO VAL` line.
`>>> NEW BEST` means a better epoch was saved. Time per epoch is printed at the end of the first
line. My estimate is 7 to 8 minutes per epoch and 4 to 6 hours per seed; the log shows the real time.

## Step 5: results

```text
cat ~/vlil_40/v6_crime_moments/results_v9_all_runs.csv
tail -60 ~/vlil_40/v6_crime_moments/logs/v9_guide_ucf101_seed123_live.log
```

The end of every training log has the test result of that seed, the table `ALL V9 RUNS SO FAR`
(mean over seeds for each variant) and, from the second seed on, the ensemble line.

| Number | Meaning | Use it as |
|---|---|---|
| `acc` (V9 pooling) | Top-1 on the 140 test videos, four time scales | **The V9 result: mean and spread of the 3 seeds** |
| `acc_v7_pooling` | Same model, tested exactly as V7 | Shows what the training changes did (V7: 38.33%) |
| `balanced_acc` | Average of the 13 per-class recalls | Second number in the paper |
| `val_video_bal` | Balanced accuracy on the 121 validation videos | Decides which variant is the final one |
| Ensemble line | The 3 seed models voting together | Only as "ensemble of 3", never as the main number |

## Step 6: optional runs (each is 3 more seeds)

Give `AFTER=` the last job number that is still queued or running, so nothing runs together.

```text
AFTER=<last job number> bash submit_v9.sh st
AFTER=<last job number> bash submit_v9.sh none
AFTER=<last job number> bash submit_v9.sh guide k400
AFTER=<last job number> TAU=0 bash submit_v9.sh
```

| Command | What it answers |
|---|---|
| `st` | Is the updated gnConv (2.2 times fewer parameters) as good or better? Run this one first |
| `none` | What does gnConv add at all? Needed if anyone asks for an ablation |
| `guide k400` | Is the plain Kinetics-400 start better than the UCF-101 start? Needs `~/vlil_40/checkpoints/videomamba_s16_k400_f8_res224.pth` |
| `TAU=0` | What does the balanced loss add? |

The final model is the variant with the highest mean `val_video_bal`. The test numbers of all
variants are reported, but the choice is made on validation.

## Step 7: files to download for me (WinSCP)

- `results_v9_all_runs.csv`
- `logs/v9_*_live.log`
- from each `runs_v9/<variant>_seed<S>/`: `logs/history.csv` and the whole `results/` folder
  (`test_results.json`, `test_predictions.csv`, `confusion_matrix.csv`, `per_class_metrics.csv`)

Put them in `Current work\v6_crime_moments\v9_results\` on the laptop.

## If something goes wrong

| What you see | What to do |
|---|---|
| Smoke test failed, seeds gone from `qstat` or stuck on **H** | `tail -40 logs/v9_guide_ucf101_seed42_smoke_live.log` and send it to me |
| `ERROR: splits/... not found` | The V6 / V7 data is missing; send me the line |
| `CUDA out of memory`, or `[WARN] less than 8 GB free` | Another user is on the GPU. Submit the same seed again later: `qsub -v SEED=42 run_train_v9.sh` |
| Job killed by walltime or node failure | Same `qsub` again; it resumes from `last.pth` |
| Jobs stay in **Q** | The cluster is full; they start by themselves |
| You want to start one variant from scratch | Delete its folder `runs_v9/<variant>_seed<S>/` first |
