"""
V10: two-stage reading of the V9 models. NO TRAINING.
  Stage 1: is there a crime in this video at all?
  Stage 2: if yes, which crime (and which kind of emergency)?

The V9 model already has a "Background" output next to the 13 crimes, so stage 1 is read from it:
  crime score of a snippet = 1 - P(Background);  crime score of a video = mean of its 4 highest
  snippet scores (the same top-k rule V9 uses for the crime type).

Test set: the official UCF-Crime test set, 140 crime videos + 150 normal videos, whole.
Threshold for "crime": picked on VALIDATION videos (121 crime + the 30 normal videos of val_bg.csv,
none of them ever trained on), never on the test set. AUC needs no threshold and is the main number.

This is VIDEO-level detection. Papers report FRAME-level AUC on UCF-Crime, so do not put these
numbers next to theirs.

Everything is read from ~/vlil_40/v6_crime_moments (V6/V7/V9 caches, code and checkpoints).
Everything is written here, in ~/vlil_40/v10_experiments. Nothing of V4-V9 is changed.

Run: qsub run_two_stage_v10.sh
"""

import argparse
import csv
import multiprocessing as mp
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V6 = os.path.expanduser('~/vlil_40/v6_crime_moments')
sys.path.insert(0, V6)

import prepare_data_v6 as p6          # noqa: E402
import prepare_test_whole_v7 as p7    # noqa: E402  (its _init makes process_video store 256 frames)

SEEDS = (42, 7, 123)
VARIANT = ('guide', 'ucf101')
CACHE = os.path.join(HERE, 'frame_cache_normal')
GROUPS = {'Violence': ['Abuse', 'Arrest', 'Assault', 'Fighting', 'Robbery', 'Shooting'],
          'Property': ['Burglary', 'Shoplifting', 'Stealing', 'Vandalism'],
          'Hazard': ['Arson', 'Explosion', 'RoadAccidents']}


# ---------------------------------------------------------------- prep (CPU)
def normal_lists():
    """Normal test videos (named in the official test annotation) and the normal validation videos."""
    ann, files = set(), {}
    for dirpath, _, filenames in os.walk(p6.DATA_ROOT):
        for fn in sorted(filenames):
            if fn == p6.ANNOTATION_FILE:
                with open(os.path.join(dirpath, fn)) as f:
                    ann |= {line.split()[0].lower() for line in f if line.strip()}
            elif (fn.lower().startswith('normal_videos') and not fn.startswith('._')
                  and os.path.splitext(fn)[1].lower() in p6.VIDEO_EXTS):
                files.setdefault(fn.lower(), os.path.join(dirpath, fn))      # one copy per file name
    test = sorted(p for name, p in files.items() if name in ann)
    with open(os.path.join(p6.EXP_DIR, 'splits', 'val_bg.csv'), newline='') as f:
        val = sorted({r['video_path'] for r in csv.DictReader(f) if r['window_source'] == 'normal video'})
    overlap = {os.path.basename(p).lower() for p in val} & ann
    if overlap:
        sys.exit(f'ERROR: validation normal videos are in the test list: {sorted(overlap)[:5]}')
    return test, val


def cache_whole(paths, sub, label):
    jobs = [{'key': os.path.basename(p), 'video_path': p, 'window_f': None, 'window_s': None,
             'cache_path': os.path.join(CACHE, sub, p6.safe_name(os.path.splitext(os.path.basename(p))[0]) + '.npz')}
            for p in paths]
    print(f'  caching {len(jobs)} whole normal {sub} videos ({p7.N_FRAMES} frames each)', flush=True)
    with mp.Pool(4, initializer=p7._init) as pool:
        results = pool.map(p6.process_video, jobs, chunksize=1)
    for r in results:
        if not r['ok']:
            print(f"  UNREADABLE: {r['video_path']} -> {r['error']}")
    return [{'video_path': r['video_path'], 'cache_path': r['cache_path'], 'label': label}
            for r in results if r['ok']]


# ---------------------------------------------------------------- numbers
def auc(score, is_crime):
    """Chance that a crime video scores higher than a normal one (ties count half)."""
    pos, neg = score[is_crime][:, None], score[~is_crime][None, :]
    return 100 * float((pos > neg).mean() + 0.5 * (pos == neg).mean())


def pick_threshold(score, is_crime):
    """The validation score that best separates crime from normal (balanced accuracy)."""
    best_bal, best_t = -1.0, 0.5
    for t in np.unique(score):
        flag = score >= t
        bal = (flag[is_crime].mean() + (~flag[~is_crime]).mean()) / 2
        if bal > best_bal:
            best_bal, best_t = bal, float(t)
    return best_t


def thin(rows, n):
    return rows[::max(1, len(rows) // n)][:n]


# ---------------------------------------------------------------- evaluation (GPU)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', action='store_true', help='4 videos of each kind, seed 42 only: checks the code')
    args = ap.parse_args()
    t_start = time.time()
    print('=' * 74)
    print('V10 two-stage test of the V9 models' + ('   [SMOKE: a few videos, numbers mean nothing]' if args.smoke else ''))
    print('=' * 74, flush=True)

    test_paths, val_paths = normal_lists()
    print(f'  normal videos found: {len(test_paths)} test (expected 150), {len(val_paths)} validation (expected 30)')
    if args.smoke:
        test_paths, val_paths = thin(test_paths, 4), thin(val_paths, 4)

    import torch                                   # after the CPU prep's worker pool is defined, before any GPU use
    from torch.utils.data import DataLoader
    import train_v6 as t6
    import train_v9 as v9

    v9.setup_run(SEEDS[0], VARIANT[0], VARIANT[1], 1.0, 0)
    classes = t6.read_classes()
    n_crime = len(classes) - 1
    crimes = classes[:n_crime]
    group_of = {c: g for g, members in GROUPS.items() for c in members}
    missing = [c for c in crimes if c not in group_of]
    if missing:
        sys.exit(f'ERROR: classes without a group: {missing}')
    grp = np.array([list(GROUPS).index(group_of[c]) for c in crimes])

    normal_test = cache_whole(test_paths, 'test', n_crime)
    normal_val = cache_whole(val_paths, 'val', n_crime)
    crime_test, crime_val = v9.read_whole('test'), v9.read_whole('val')
    if args.smoke:
        crime_test, crime_val = thin(crime_test, 4), thin(crime_val, 4)
    test_rows, val_rows = crime_test + normal_test, crime_val + normal_val
    print(f'  test: {len(crime_test)} crime + {len(normal_test)} normal | validation: {len(crime_val)} crime + '
          f'{len(normal_val)} normal', flush=True)

    if not torch.cuda.is_available():
        sys.exit('ERROR: no GPU visible. Submit through PBS.')
    device = torch.device('cuda')
    vm, ms7 = t6.load_backbone_module()
    model, _ = v9.build_model(vm, ms7, len(classes), device)

    def score(rows, crops):
        loader = DataLoader(v9.WholeVideo(rows, crops), batch_size=1, shuffle=False,
                            num_workers=t6.NUM_WORKERS, worker_init_fn=t6._worker_init)
        probs, labels = v9.video_probs(model, loader, device, crops)
        k = max(1, round(len(probs[0]) / v9.TOPK_DIV))
        crime = np.array([float((1 - p[:, n_crime]).topk(k).values.mean()) for p in probs])
        kind = np.array([int(v9.pool(p, n_crime)[0].argmax()) for p in probs])
        return crime, kind, labels

    out_dir = os.path.join(HERE, 'results_smoke' if args.smoke else 'results')
    os.makedirs(out_dir, exist_ok=True)
    fields = ['seed', 'auc', 'threshold', 'detect_acc', 'crime_found', 'false_alarms', 'type_acc_all_crime',
              'type_acc_found_crime', 'group_acc_all_crime', 'end_to_end_acc']
    summary = []
    for seed in SEEDS[:1] if args.smoke else SEEDS:
        v9.setup_run(seed, VARIANT[0], VARIANT[1], 1.0, 0)
        if not os.path.exists(t6.BEST_PATH):
            print(f'\n  seed {seed}: {t6.BEST_PATH} not found, skipped')
            continue
        model.load_state_dict(t6.torch_load(t6.BEST_PATH)['state_dict'])
        t0 = time.time()
        v_score, _, v_lab = score(val_rows, v9.VAL_CROPS)
        thr = pick_threshold(v_score, v_lab < n_crime)
        s, kind, lab = score(test_rows, v9.TEST_CROPS)
        is_c = lab < n_crime
        flag = s >= thr
        type_ok = kind == lab                                            # only meaningful for crime videos
        r = {'seed': seed, 'auc': auc(s, is_c), 'threshold': thr,
             'detect_acc': 100 * float((flag == is_c).mean()),
             'crime_found': 100 * float(flag[is_c].mean()),
             'false_alarms': 100 * float(flag[~is_c].mean()),
             'type_acc_all_crime': 100 * float(type_ok[is_c].mean()),
             'type_acc_found_crime': 100 * float(type_ok[is_c & flag].mean()) if (is_c & flag).any() else 0.0,
             'group_acc_all_crime': 100 * float((grp[kind[is_c]] == grp[lab[is_c]]).mean()),
             'end_to_end_acc': 100 * float(np.where(is_c, flag & type_ok, ~flag).mean())}
        summary.append(r)
        print(f'\n  SEED {seed}  ({time.time() - t0:.0f}s)')
        print(f"    Stage 1, crime or not ({len(lab)} test videos): AUC {r['auc']:.2f}% | at the validation threshold "
              f"{thr:.3f}: accuracy {r['detect_acc']:.2f}%, crime videos found {int(flag[is_c].sum())}/{int(is_c.sum())}, "
              f"false alarms {int(flag[~is_c].sum())}/{int((~is_c).sum())}")
        print(f"    Stage 2, which crime ({int(is_c.sum())} crime videos): {r['type_acc_all_crime']:.2f}% "
              f"(must equal this seed's V9 accuracy) | on the crime videos stage 1 found: {r['type_acc_found_crime']:.2f}%")
        print(f"    Kind of emergency (Violence / Property / Hazard): {r['group_acc_all_crime']:.2f}%")
        print(f"    Both stages together, 14 answers (13 crimes + no crime), {len(lab)} videos: {r['end_to_end_acc']:.2f}%",
              flush=True)
        with open(os.path.join(out_dir, f'two_stage_seed{seed}.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['video', 'true', 'crime_score', 'flagged_as_crime', 'predicted_crime'])
            for row, lb, sc, fl, kd in zip(test_rows, lab, s, flag, kind):
                w.writerow([os.path.basename(row['video_path']), classes[lb], f'{sc:.4f}', int(fl), crimes[kd]])

    if not summary:
        sys.exit('ERROR: no V9 checkpoint found.')
    with open(os.path.join(out_dir, 'two_stage_all_seeds.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows([{k: (v if k == 'seed' else f'{v:.4f}' if k == 'threshold' else f'{v:.2f}') for k, v in r.items()}
                     for r in summary])
    print('\n' + '=' * 74)
    print(f'MEAN OVER SEEDS {[r["seed"] for r in summary]} (V9 guide_ucf101 models, no training)')
    names = {'auc': 'Stage 1 AUC, crime or not (video level)', 'detect_acc': 'Stage 1 accuracy',
             'crime_found': 'Crime videos found', 'false_alarms': 'Normal videos flagged (false alarms)',
             'type_acc_all_crime': 'Stage 2, which crime (all 140)', 'group_acc_all_crime': 'Kind of emergency (3 groups)',
             'end_to_end_acc': 'Both stages together (14 answers)'}
    for k, name in names.items():
        v = np.array([r[k] for r in summary])
        print(f'  {name:<42s}{v.mean():>7.2f}%' + (f' +- {v.std():.2f}' if len(v) > 1 else ''))
    print('  Video-level detection: not comparable with the frame-level AUC of the papers.')
    print('  The 3 groups were chosen after seeing V9 results: say so when reporting.')
    print(f'  Saved: {out_dir}/   ({time.time() - t_start:.0f}s)')
    print('SMOKE TEST PASSED' if args.smoke else 'V10 TWO-STAGE TEST DONE')
    print('=' * 74, flush=True)


if __name__ == '__main__':
    main()
