"""
V7 step 2, prep: clips for a 14th class "Background" (no crime happening in this clip).

Why: the test of the CVPR 2024 / 2025 papers shows the WHOLE untrimmed video, and most of each
video is not the crime. V6 only ever saw crime moments, so every snippet of a test video had to
be called some crime. With a Background class the model can say "nothing here", and the top-k
pooling of the test then picks the snippets where a crime really shows.

Background clips (12 s each, 32 frames cached, same format as frame_cache/):
  outside : from the V6 train/val crime videos, the longest stretch OUTSIDE their UCA crime window
            (2 s margin on each side, at least 4 s long). Train videos -> train, val videos -> val.
  normal  : random 12 s from UCF-Crime normal videos. Videos listed in the official test
            annotation file are never used.
The 140 test videos are not touched. Writes splits/train_bg.csv, splits/val_bg.csv,
splits/classes_v7.txt and logs/prep_bg_report.txt.

Run: qsub run_prepare_bg_v7.sh   (after V6 prep; CPU only)
"""

import ast
import csv
import hashlib
import multiprocessing as mp
import os
import random
import sys
import time

import prepare_data_v6 as p6

BG = 'Background'
WIN_S = 12.0              # length of one background clip
MARGIN_S = 2.0            # keep this far from the crime window
MIN_S = 4.0               # shortest usable stretch
N_OUTSIDE = {'train': 250, 'val': 30}
N_NORMAL = {'train': 250, 'val': 30}
SEED = 42


def outside_window(n, fps, window, rng):
    """A random WIN_S piece of the longest stretch outside the crime window, in frames, or None."""
    m, need, win = MARGIN_S * fps, MIN_S * fps, WIN_S * fps
    free, cur = [], 0.0
    for s, e in sorted(window):
        free.append((cur, s - m))
        cur = max(cur, e + m)
    free.append((cur, n - 1))
    s, e = max(free, key=lambda g: g[1] - g[0])
    if e - s < need:
        return None
    start = rng.uniform(s, max(s, e - win))
    return [[int(start), int(min(e, start + win))]]


def header(path):
    import cv2
    cap = cv2.VideoCapture(path)
    n, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0), float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return path, n, fps


def main():
    t0 = time.time()
    rng = random.Random(SEED)
    split_dir = os.path.join(p6.EXP_DIR, 'splits')
    cache_dir = os.path.join(p6.EXP_DIR, 'frame_cache_bg')
    log_dir = os.path.join(p6.EXP_DIR, 'logs')

    # ---------- V6 classes + Background ----------
    with open(os.path.join(split_dir, 'classes.txt')) as f:
        classes = [line.rstrip('\n').split('\t', 1)[1] for line in f if line.strip()]
    bg_label = len(classes)
    with open(os.path.join(split_dir, 'classes_v7.txt'), 'w') as f:
        for i, c in enumerate(classes + [BG]):
            f.write(f'{i}\t{c}\n')

    # ---------- outside-the-crime clips from V6 train / val videos ----------
    videos = {}
    for split in ('train', 'val'):
        with open(os.path.join(split_dir, f'{split}.csv'), newline='') as f:
            for r in csv.DictReader(f):
                videos[os.path.splitext(os.path.basename(r['video_path']))[0]] = dict(r, split=split)
    cand = {'train': [], 'val': []}
    with open(os.path.join(split_dir, 'crime_windows.csv'), newline='') as f:
        for w in csv.DictReader(f):
            v = videos.get(w['video'])
            if not v or w['window_source'] != 'uca':
                continue
            n, fps = int(v['n_frames']), float(v['fps']) or 30.0
            win = outside_window(n, fps, ast.literal_eval(w['window_frames']), rng)
            if win:
                cand[v['split']].append((v['video_path'], win, f"outside {w['class']} window"))

    # ---------- normal videos (never the ones in the official test annotation) ----------
    ann_names, normals = set(), []
    for dirpath, _, filenames in os.walk(p6.DATA_ROOT):
        for fn in filenames:
            if fn == p6.ANNOTATION_FILE:
                with open(os.path.join(dirpath, fn)) as af:
                    ann_names |= {line.split()[0].lower() for line in af if line.strip()}
            elif (fn.lower().startswith('normal_videos') and not fn.startswith('._')
                  and os.path.splitext(fn)[1].lower() in p6.VIDEO_EXTS):
                normals.append(os.path.join(dirpath, fn))
    by_name = {}
    for p in sorted(normals):
        if os.path.basename(p).lower() not in ann_names:
            by_name.setdefault(os.path.basename(p).lower(), p)       # one copy per file name
    normals = sorted(by_name.values())
    rng.shuffle(normals)
    with mp.Pool(4) as pool:
        heads = pool.map(header, normals[:3 * sum(N_NORMAL.values())], chunksize=4)
    normal_clips = []
    for path, n, fps in heads:
        fps = fps or 30.0
        if n >= MIN_S * fps:
            start = rng.uniform(0, max(0.0, n - 1 - WIN_S * fps))
            normal_clips.append((path, [[int(start), int(min(n - 1, start + WIN_S * fps))]], 'normal video'))
    print(f'Normal videos usable for Background: {len(by_name)} on disk (test ones excluded), '
          f'{len(normal_clips)} checked', flush=True)

    # ---------- pick, decode, write ----------
    jobs = []
    for split in ('train', 'val'):
        rng.shuffle(cand[split])
        picked = cand[split][:N_OUTSIDE[split]]
        start = 0 if split == 'train' else N_NORMAL['train']
        picked += normal_clips[start:start + N_NORMAL[split]]
        for path, win, source in picked:
            stem = os.path.splitext(os.path.basename(path))[0]
            name = f"{p6.safe_name(stem)}_{hashlib.md5((path + str(win)).encode()).hexdigest()[:8]}.npz"
            jobs.append({'key': stem, 'video_path': path, 'window_f': win, 'window_s': None,
                         'cache_path': os.path.join(cache_dir, split, name), 'split': split,
                         'class_name': BG, 'label': bg_label, 'window_source': source})
    print(f'Decoding {len(jobs)} background clips -> {cache_dir}', flush=True)
    with mp.Pool(4, initializer=p6._init_worker) as pool:
        results = pool.map(p6.process_video, jobs, chunksize=1)

    lines = ['=' * 70, 'V7 BACKGROUND CLIPS (class %d "%s")' % (bg_label, BG), '=' * 70]
    fields = ['split', 'class_name', 'label', 'video_path', 'cache_path', 'window_source', 'window_s',
              'duration_s', 'n_frames', 'fps']
    counts = {}
    for split in ('train', 'val'):
        rows = [r for r in results if r['ok'] and r['split'] == split]
        with open(os.path.join(split_dir, f'{split}_bg.csv'), 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            w.writerows(rows)
        counts[split] = len(rows)
        n_out = sum(r['window_source'].startswith('outside') for r in rows)
        lines.append(f'{split:5s}: {len(rows)} clips ({n_out} outside a crime window, {len(rows) - n_out} normal) '
                     f'| usable outside-window candidates: {len(cand[split])}')
    bad = [r for r in results if not r['ok']]
    for r in bad:
        lines.append(f"  UNREADABLE: {r['video_path']} -> {r['error']}")
    lines.append(f'Classes: splits/classes_v7.txt ({bg_label + 1} classes) | time {time.time() - t0:.0f}s')
    failed = counts['train'] < 100 or counts['val'] < 10
    lines.append('PREP FAILED: too few background clips' if failed else 'PREP DONE')
    report = '\n'.join(lines)
    print(report)
    with open(os.path.join(log_dir, 'prep_bg_report.txt'), 'w') as f:
        f.write(report + '\n')
    if failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
