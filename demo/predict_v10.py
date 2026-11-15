"""
V10 demo: give the trained V9 model any video (or a folder of videos) and print its answer.
  Stage 1: is there a crime?      Stage 2: which crime, which kind of emergency, and when.

Same steps as the V9 test: 256 frames are stored from the whole video, the video is cut into
32 / 16 / 8 / 4 pieces (60 snippets, 3 crops each) and the 4 strongest snippets decide.
The model knows 13 crimes (the UCF-Crime ones). Its measured accuracy on naming the crime is
43.10% +- 1.47, so a single answer can be wrong: show the top 3.

Run through PBS (the model needs the GPU):   bash demo.sh /path/to/video.mp4
"""

import argparse
import csv
import hashlib
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
V6 = os.path.expanduser('~/vlil_40/v6_crime_moments')
sys.path.insert(0, V6)

import prepare_data_v6 as p6          # noqa: E402
import prepare_test_whole_v7 as p7    # noqa: E402

GROUPS = {'Violence': ['Abuse', 'Arrest', 'Assault', 'Fighting', 'Robbery', 'Shooting'],
          'Property': ['Burglary', 'Shoplifting', 'Stealing', 'Vandalism'],
          'Hazard (fire, explosion, accident)': ['Arson', 'Explosion', 'RoadAccidents']}


def clock(seconds):
    return f'{int(seconds) // 60}:{int(seconds) % 60:02d}'


def find_videos(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(os.path.join(p, f) for f in os.listdir(p)
                          if os.path.splitext(f)[1].lower() in p6.VIDEO_EXTS and not f.startswith('._'))
        elif os.path.isfile(p):
            out.append(p)
        else:
            print(f'  NOT FOUND: {p}')
    return out


def threshold_for(seed):
    """The crime-or-not threshold the two-stage test picked on validation videos, if that test has run."""
    path = os.path.expanduser('~/vlil_40/v10_experiments/results/two_stage_all_seeds.csv')
    if os.path.exists(path):
        with open(path, newline='') as f:
            for r in csv.DictReader(f):
                if r['seed'] == str(seed):
                    return float(r['threshold']), 'picked on validation videos'
    return 0.5, 'default 0.5: the two-stage test has not finished for this seed'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('videos', nargs='+', help='video files or folders')
    ap.add_argument('--seed', type=int, default=42, help='which V9 model (42 has the best validation score)')
    args = ap.parse_args()
    videos = find_videos(args.videos)
    if not videos:
        sys.exit('ERROR: no video file found.')

    import torch
    from torch.utils.data import DataLoader
    import train_v6 as t6
    import train_v9 as v9

    if not torch.cuda.is_available():
        sys.exit('ERROR: no GPU visible. Run it with:  bash demo.sh <video>')
    device = torch.device('cuda')
    v9.setup_run(args.seed, 'guide', 'ucf101', 1.0, 0)
    if not os.path.exists(t6.BEST_PATH):
        sys.exit(f'ERROR: model not found: {t6.BEST_PATH}')
    classes = t6.read_classes()
    n_crime = len(classes) - 1
    crimes = classes[:n_crime]
    group_of = {c: g for g, members in GROUPS.items() for c in members}
    thr, thr_note = threshold_for(args.seed)

    t0 = time.time()
    vm, ms7 = t6.load_backbone_module()
    model, _ = v9.build_model(vm, ms7, len(classes), device)
    model.load_state_dict(t6.torch_load(t6.BEST_PATH)['state_dict'])
    print(f'\n  Model: VideoMamba-Small + gnConv2D (V9, seed {args.seed}), loaded in {time.time() - t0:.0f}s')
    print(f'  Crime-or-not threshold: {thr:.3f} ({thr_note})')

    p7._init()                                              # store 256 frames per video
    os.makedirs(os.path.join(HERE, 'demo_cache'), exist_ok=True)
    for path in videos:
        t1 = time.time()
        st = os.stat(path)
        tag = hashlib.md5(f'{os.path.abspath(path)}|{st.st_size}|{st.st_mtime}'.encode()).hexdigest()[:10]
        cache = os.path.join(HERE, 'demo_cache',
                             p6.safe_name(os.path.splitext(os.path.basename(path))[0]) + f'_{tag}.npz')
        r = p6.process_video({'key': os.path.basename(path), 'video_path': path, 'window_f': None,
                              'window_s': None, 'cache_path': cache})
        print('\n' + '=' * 70)
        print(f'  VIDEO: {os.path.basename(path)}')
        if not r['ok']:
            print(f"  COULD NOT READ THIS VIDEO: {r['error']}")
            continue
        loader = DataLoader(v9.WholeVideo([{'cache_path': cache, 'label': 0}], v9.TEST_CROPS), batch_size=1)
        p = v9.video_probs(model, loader, device, v9.TEST_CROPS)[0][0]          # (60 snippets, 14 classes)
        k = max(1, round(len(p) / v9.TOPK_DIV))
        crime_score = float((1 - p[:, n_crime]).topk(k).values.mean())
        scores = v9.pool(p, n_crime)[0]
        share = scores / scores.sum()
        order = share.argsort(descending=True).tolist()
        top = crimes[order[0]]
        fine = p[:v9.SCALES[0], order[0]]                                       # the 32 short pieces, top crime
        piece = int(fine.argmax())
        dur = float(r.get('duration_s') or 0)
        is_crime = crime_score >= thr

        print(f"  Length {clock(dur)} | {r.get('width')}x{r.get('height')} | {r.get('n_frames')} frames")
        print('-' * 70)
        print(f'  STAGE 1  Is there a crime?   {"YES" if is_crime else "NO"}   '
              f'(crime score {crime_score:.2f}, threshold {thr:.2f})')
        second = crimes[order[1]]
        tie = float(share[order[0]] - share[order[1]]) < 0.05           # top two within 5 points: say so
        if is_crime and tie:
            print(f'  STAGE 2  Which crime?        NOT SURE: {top.upper()} or {second.upper()}')
        elif is_crime:
            print(f'  STAGE 2  Which crime?        {top.upper()}')
        else:
            print(f'  STAGE 2  (if it were a crime, the closest would be {top})')
        print('           ' + '   '.join(f'{i + 1}. {crimes[c]} {100 * float(share[c]):.0f}%'
                                         for i, c in enumerate(order[:3])))
        kinds = [group_of.get(top, '?')]
        if tie and group_of.get(second, '?') != kinds[0]:
            kinds.append(group_of.get(second, '?'))
        print(f'  Kind of emergency: {" or ".join(kinds)}')
        if dur > 0:
            a, b = piece * dur / v9.SCALES[0], (piece + 1) * dur / v9.SCALES[0]
            when = f'{a:.1f} s to {b:.1f} s' if dur < 60 else f'{clock(a)} to {clock(b)}'   # short video: show tenths
            print(f'  Strongest moment for {top}: {when}')
        print(f'  ({time.time() - t1:.0f}s for this video; the percentages share out the 13 crime scores)')
    print('=' * 70)
    print('  The model names the crime correctly in 43.10% of the 140 UCF-Crime test videos (13 crimes, chance 7.7%).')
    print('DEMO DONE', flush=True)


if __name__ == '__main__':
    main()
