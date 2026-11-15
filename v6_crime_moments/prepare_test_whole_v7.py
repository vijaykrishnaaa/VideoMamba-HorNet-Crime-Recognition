"""
V7 step 1, prep: cache the 140 official test crime videos WHOLE (not cut to the crime window).

This is the test of the CVPR 2024 / 2025 papers (Wu et al. "Open-Vocabulary Video Anomaly
Detection", Li et al. "Anomize"): top-1 accuracy on the anomalous test videos, where the model
sees the whole untrimmed video. Stores 256 frames spread over each whole video in
frame_cache_whole/ and writes splits/test_whole.csv. Reuses the decoding code of prepare_data_v6.py.
CPU only; run_eval_v7.sh runs it before eval_v7_protocol.py.
"""

import csv
import multiprocessing as mp
import os
import sys

import prepare_data_v6 as p6

N_FRAMES = 256          # 32 snippets x 8 frames over the whole video


def _init():
    p6.NUM_CACHE_FRAMES = N_FRAMES     # process_video reads this module global (also under spawn)
    p6._init_worker()


def main():
    split_dir = os.path.join(p6.EXP_DIR, 'splits')
    cache_dir = os.path.join(p6.EXP_DIR, 'frame_cache_whole')
    with open(os.path.join(split_dir, 'test.csv'), newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit('ERROR: splits/test.csv is empty. Run prepare_data_v6.py first.')

    jobs = [{'key': os.path.basename(r['video_path']), 'video_path': r['video_path'],
             'window_f': None, 'window_s': None,                       # None = the whole video
             'cache_path': os.path.join(cache_dir, os.path.basename(r['cache_path'])), 'row': r}
            for r in rows]
    print(f'V7 prep: caching {len(jobs)} whole test videos ({N_FRAMES} frames each) -> {cache_dir}', flush=True)
    with mp.Pool(4, initializer=_init) as pool:
        results = pool.map(p6.process_video, jobs, chunksize=1)

    bad = [r for r in results if not r['ok']]
    for r in bad:
        print(f"  UNREADABLE: {r['video_path']} -> {r['error']}")
    with open(os.path.join(split_dir, 'test_whole.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['split', 'class_name', 'label', 'video_path', 'cache_path', 'duration_s'])
        for r in results:
            if r['ok']:
                w.writerow(['test', r['row']['class_name'], r['row']['label'], r['video_path'],
                            r['cache_path'], r['duration_s']])
    print(f'V7 prep done: {len(results) - len(bad)}/{len(results)} videos cached', flush=True)
    if bad:
        sys.exit(1)            # the protocol needs all 140 videos


if __name__ == '__main__':
    main()
