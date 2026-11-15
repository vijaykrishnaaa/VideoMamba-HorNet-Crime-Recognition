"""
V7 step 2: V6 + a "Background" class, tested on the CVPR 2024 / 2025 protocol.
==========================================================================
Same model (VideoMamba-Small + gnConv2D in layers 16-23), same starting checkpoint (UCF-101) and
the same training recipe as train_v6.py, which this script reuses unchanged. Differences:
   1. 14 classes: the 13 crimes + Background (clips outside the crime window and normal videos,
      made by prepare_background_v7.py). Train and val get background clips; test does not.
   2. Final test = the step-1 protocol: the 140 official test crime videos, WHOLE, 32 snippets x
      3 crops. Background is left out of the decision (the answer is always one of the 13 crimes).
      Main number: top-2 pooling (Anomize), decided before training. Mean pooling is also shown.
   3. Runs go to runs_v7/seed{S}_ucf101/, the summary over seeds to results_v7_all_runs.csv.

Step 1 (V6 models, no Background class, same test): mean 31.19%, top-2 34.52%.

Run (on HPC, through PBS):  qsub -v SEED=42 run_train_v7.sh
"""

import csv
import os
import sys
import time

import numpy as np
from torch.utils.data import DataLoader

import eval_v7_protocol as e7
import train_v6 as t6

RUNS_DIR = os.path.join(t6.EXP_DIR, 'runs_v7')
ALL_RUNS_CSV = os.path.join(t6.EXP_DIR, 'results_v7_all_runs.csv')
STEP1 = {'mean': 31.19, 'topk': 34.52}

_setup_run_v6, _read_split_v6 = t6.setup_run, t6.read_split


def setup_run(seed, init):
    _setup_run_v6(seed, init)
    t6.RUN_DIR = os.path.join(RUNS_DIR, f'seed{seed}_{init}')
    t6.CKPT_DIR = os.path.join(t6.RUN_DIR, 'checkpoints')
    t6.LOG_DIR = os.path.join(t6.RUN_DIR, 'logs')
    t6.RESULT_DIR = os.path.join(t6.RUN_DIR, 'results')
    t6.BEST_PATH = os.path.join(t6.CKPT_DIR, 'best_model.pth')
    t6.LAST_PATH = os.path.join(t6.CKPT_DIR, 'last.pth')


def read_split(name):
    rows = _read_split_v6(name)
    if name in ('train', 'val'):
        rows += _read_split_v6(name + '_bg')
    return rows


def whole_video_test(model, classes, test_rows, device, best_info, params_info, compute_info):
    """Replaces train_v6.final_test: the step-1 protocol on whole test videos."""
    n_crime = len(classes) - 1                      # Background is the last class
    with open(os.path.join(t6.SPLIT_DIR, 'test_whole.csv'), newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['label'] = int(r['label'])
    loader = DataLoader(e7.WholeVideo(rows), batch_size=1, shuffle=False, num_workers=t6.NUM_WORKERS,
                        worker_init_fn=t6._worker_init)
    t6.log(f'\n  TEST: {len(rows)} whole test crime videos, {e7.SNIPPETS} snippets x {e7.CROPS} crops each...')
    probs, labels = e7.snippet_probs(model, loader, device)
    crime = [p[:, :n_crime] for p in probs]
    pred_mean = np.array([int(p.mean(0).argmax()) for p in crime])
    pred_topk = np.array([int(p.topk(e7.TOPK, dim=0).values.mean(0).argmax()) for p in crime])
    acc_m, bal_m = e7.scores(pred_mean, labels, n_crime)
    acc_k, bal_k = e7.scores(pred_topk, labels, n_crime)
    bg_share = 100 * float(np.mean([float((p.argmax(1) == n_crime).float().mean()) for p in probs]))

    t6.log('')
    t6.log('=' * 70)
    t6.log(f'V7 STEP 2 TEST — seed {t6.SEED}, model from epoch {best_info["epoch"]} '
           f'({best_info["weights"]} weights, val acc {best_info["val_acc"]:.2f}%)')
    t6.log('=' * 70)
    t6.log(f'  top-{e7.TOPK} pooling (main): {acc_k:.2f}%  (balanced {bal_k:.2f}%)   step 1: {STEP1["topk"]:.2f}%')
    t6.log(f'  mean pooling        : {acc_m:.2f}%  (balanced {bal_m:.2f}%)   step 1: {STEP1["mean"]:.2f}%')
    t6.log(f'  snippets called Background: {bg_share:.1f}% (the crime is usually a small part of the video)')
    t6.log('  per class, top-k (correct/total): ' + ' | '.join(
        f'{c[:10]} {int(((pred_topk == i) & (labels == i)).sum())}/{int((labels == i).sum())}'
        for i, c in enumerate(classes[:n_crime])))
    t6.log('=' * 70)

    os.makedirs(t6.RESULT_DIR, exist_ok=True)
    with open(os.path.join(t6.RESULT_DIR, 'test_whole_predictions.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['video', 'true', 'pred_mean', 'pred_topk'])
        for r, t_, a, b in zip(rows, labels, pred_mean, pred_topk):
            w.writerow([r['video_path'], classes[t_], classes[a], classes[b]])

    runs = []
    if os.path.exists(ALL_RUNS_CSV):
        with open(ALL_RUNS_CSV, newline='') as f:
            runs = [r for r in csv.DictReader(f) if r['seed'] != str(t6.SEED)]
    runs.append({'seed': t6.SEED, 'acc_topk': f'{acc_k:.2f}', 'bal_topk': f'{bal_k:.2f}',
                 'acc_mean': f'{acc_m:.2f}', 'bal_mean': f'{bal_m:.2f}', 'best_epoch': best_info['epoch'],
                 'val_acc': f"{best_info['val_acc']:.2f}", 'finished': time.strftime('%Y-%m-%d %H:%M')})
    runs.sort(key=lambda r: int(r['seed']))
    with open(ALL_RUNS_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(runs[-1].keys()))
        w.writeheader()
        w.writerows(runs)
    for key in ('acc_topk', 'acc_mean'):
        v = np.array([float(r[key]) for r in runs])
        t6.log(f'  Mean over seeds {[int(r["seed"]) for r in runs]}: {key} {v.mean():.2f}%'
               + (f' (std {v.std():.2f})' if len(v) > 1 else ''))
    t6.log('  Published on this protocol (CLIP, open-vocabulary): Wu et al. 2024 41.43%, Anomize 2025 47.14%')
    t6.log(f'  Saved: {ALL_RUNS_CSV}')


def main():
    for fn, how in (('train_bg.csv', 'qsub run_prepare_bg_v7.sh'), ('val_bg.csv', 'qsub run_prepare_bg_v7.sh'),
                    ('classes_v7.txt', 'qsub run_prepare_bg_v7.sh'), ('test_whole.csv', 'qsub run_eval_v7.sh')):
        if not os.path.exists(os.path.join(t6.SPLIT_DIR, fn)):
            sys.exit(f'ERROR: splits/{fn} not found. Run first: {how}')
    t6.CLASSES_FILE = os.path.join(t6.SPLIT_DIR, 'classes_v7.txt')
    t6.setup_run, t6.read_split, t6.final_test = setup_run, read_split, whole_video_test
    t6.log('V7 step 2: 13 crimes + Background; the train_v6 log below is reused (runs go to runs_v7/)')
    t6.main()


if __name__ == '__main__':
    main()
