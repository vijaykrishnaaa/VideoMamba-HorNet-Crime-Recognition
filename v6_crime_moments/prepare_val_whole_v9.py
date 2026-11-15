"""
V9 prep: cache the 121 validation crime videos WHOLE (256 frames each), exactly as
prepare_test_whole_v7.py did for the 140 test videos.

Why: V9 picks its best epoch with the same test the papers use (whole untrimmed video), run on
the validation videos. The model never trains on these videos, and the test videos are not
touched. Writes frame_cache_whole_val/ and splits/val_whole.csv. CPU only, a few minutes.

Run: qsub run_prepare_v9.sh   (bash submit_v9.sh does it for you)
"""

import csv
import multiprocessing as mp
import os
import sys

import prepare_data_v6 as p6
import prepare_test_whole_v7 as p7        # its _init makes process_video store 256 frames


def main():
    split_dir = os.path.join(p6.EXP_DIR, 'splits')
    cache_dir = os.path.join(p6.EXP_DIR, 'frame_cache_whole_val')
    for need in ('val.csv', 'train_bg.csv', 'val_bg.csv', 'classes_v7.txt', 'test_whole.csv'):
        if not os.path.exists(os.path.join(split_dir, need)):
            sys.exit(f'ERROR: splits/{need} not found. V9 needs the V6 and V7 data first '
                     f'(run_prepare_v6.sh, run_prepare_bg_v7.sh, run_eval_v7.sh).')
    with open(os.path.join(split_dir, 'val.csv'), newline='') as f:
        rows = list(csv.DictReader(f))

    jobs = [{'key': os.path.basename(r['video_path']), 'video_path': r['video_path'],
             'window_f': None, 'window_s': None,                       # None = the whole video
             'cache_path': os.path.join(cache_dir, os.path.basename(r['cache_path'])), 'row': r}
            for r in rows]
    print(f'V9 prep: caching {len(jobs)} whole validation videos ({p7.N_FRAMES} frames each) -> {cache_dir}',
          flush=True)
    with mp.Pool(4, initializer=p7._init) as pool:
        results = pool.map(p6.process_video, jobs, chunksize=1)

    bad = [r for r in results if not r['ok']]
    for r in bad:
        print(f"  UNREADABLE: {r['video_path']} -> {r['error']}")
    with open(os.path.join(split_dir, 'val_whole.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['split', 'class_name', 'label', 'video_path', 'cache_path', 'duration_s'])
        for r in results:
            if r['ok']:
                w.writerow(['val', r['row']['class_name'], r['row']['label'], r['video_path'],
                            r['cache_path'], r['duration_s']])
    print(f'V9 PREP DONE: {len(results) - len(bad)}/{len(results)} validation videos cached', flush=True)
    if len(results) - len(bad) < 100:
        sys.exit(1)


if __name__ == '__main__':
    main()
