"""
V9: the final UCF-Crime model. V7 with four changes that come from our own results
==================================================================================
Same backbone (VideoMamba-Small + HorNet gnConv2D in layers 16-23, layers 0-15 frozen), same data
(the crime moments and Background clips made for V6 / V7), same test as the CVPR 2024 / 2025
papers (top-1 accuracy on the 140 official test crime videos, whole and untrimmed).
What is new, and why (the full story with numbers and sources is in V9_EXPLAINED.md):

  1. Balanced crimes (logit-adjusted loss, Menon et al., ICLR 2021). Training has 123 Robbery but
     23 Shooting and 25 Shoplifting videos; the test has 5 Robbery but 23 Shooting and 21
     Shoplifting. V6 called 14-15 of the 21 Shoplifting test videos Robbery in all 3 seeds.
     The loss now removes the "how often did I see this crime" habit among the 13 crimes.
     Background keeps its natural share, because most of a real video is background.
  2. Pieces of the crime moment (temporal crop). Each training clip is a random 40-100% piece of
     the stored crime moment, so the model sees short and long views of a crime.
  3. Four time scales in the test. The video is cut into 32, 16, 8 and 4 equal pieces
     (60 snippets). For each crime the 4 highest snippet scores are averaged (top-k,
     k = snippets / 16, the Anomize rule). V7 used only the 32 short pieces, which are much
     shorter than the clips the model was trained on.
  4. The best epoch is picked with the test protocol itself on the 121 validation videos
     (whole video, balanced accuracy, EMA weights). V7 picked it on clips; its validation
     accuracy (about 60%) did not follow its test accuracy (33.6 to 42.9%).

Unchanged from V7: learning rates, weight decay, dropout, stochastic depth, label smoothing,
mixup, EMA, augmentation, 8 frames, batch 2 x 8, 40 epochs, early stopping (train_v6.py).

Options (run_train_v9.sh passes them):
  --gnconv guide   the guide's gnConv2D, unchanged (default: the main run)
  --gnconv st      updated gnConv (gnconv_st_v9.py): separable 3x3/5x5/7x7 + neighbouring frames
  --gnconv none    no gnConv at all (shows what gnConv adds)
  --init ucf101    Kinetics-400 -> UCF-101 start, as V4-V8 (default);  --init k400: Kinetics-400 only
  --tau 0          change 1 switched off (for comparison)
  --smoke 1 / 2    2-minute check of the whole pipeline on a few clips (stage 2 tests resuming)

Run (HPC, PBS):  bash submit_v9.sh      (prep + smoke test + seeds 42, 7, 123, one after another)
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset

import eval_v7_protocol as e7
import train_v6 as t6

log = t6.log

RUNS_DIR     = os.path.join(t6.EXP_DIR, 'runs_v9')
ALL_RUNS_CSV = os.path.join(t6.EXP_DIR, 'results_v9_all_runs.csv')
INIT_CHECKPOINTS = {
    'ucf101': t6.INIT_CHECKPOINTS['ucf101'],
    'k400':   os.path.join(t6.VLIL_ROOT, 'checkpoints', 'videomamba_s16_k400_f8_res224.pth'),
}

# ---- what V9 changes (everything else is read from train_v6.py) ----
TAU        = 1.0              # 1 = the 13 crimes fully balanced in the loss, 0 = off
TCROP_MIN  = 0.4              # a training clip covers a random 40-100% piece of the stored moment
SCALES     = (32, 16, 8, 4)   # test: the video cut into this many equal pieces, one clip per piece
TOPK_DIV   = 16               # top-k pooling, k = snippets / 16 (Anomize)
VAL_CROPS  = 1                # whole-video validation: centre crop (runs every epoch)
TEST_CROPS = 3                # final test: 3 crops, as V7
V7_RESULT  = '38.33% +- 3.8 (seeds 42 / 7 / 123: 42.86 / 33.57 / 38.57)'

GNCONV, INIT, TAG, SMOKE = 'guide', 'ucf101', 'guide_ucf101', 0


def setup_run(seed, gnconv, init, tau, smoke):
    global GNCONV, INIT, TAU, TAG, SMOKE, ALL_RUNS_CSV
    GNCONV, INIT, TAU, SMOKE = gnconv, init, tau, smoke
    TAG = f'{gnconv}_{init}' + ('' if tau == 1.0 else f'_tau{tau:g}')
    t6.SEED, t6.INIT = seed, init
    t6.RUN_DIR = os.path.join(RUNS_DIR, ('smoke_' if smoke else '') + f'{TAG}_seed{seed}')
    t6.CKPT_DIR = os.path.join(t6.RUN_DIR, 'checkpoints')
    t6.LOG_DIR = os.path.join(t6.RUN_DIR, 'logs')
    t6.RESULT_DIR = os.path.join(t6.RUN_DIR, 'results')
    t6.BEST_PATH = os.path.join(t6.CKPT_DIR, 'best_model.pth')
    t6.LAST_PATH = os.path.join(t6.CKPT_DIR, 'last.pth')
    t6.PRETRAINED_PATH = INIT_CHECKPOINTS[init]
    t6.CLASSES_FILE = os.path.join(t6.SPLIT_DIR, 'classes_v7.txt')
    if smoke:                                       # the smoke test never touches the real results
        t6.EPOCHS, t6.WARMUP_EPOCHS = 2, 1
        ALL_RUNS_CSV = os.path.join(t6.EXP_DIR, 'results_v9_smoke.csv')


def config_dict():
    cfg = t6.config_dict()
    cfg.update({'GNCONV': GNCONV, 'TAU': TAU, 'TCROP_MIN': TCROP_MIN, 'SCALES': SCALES, 'TOPK_DIV': TOPK_DIV,
                'VIDEO_VAL_CROPS': VAL_CROPS, 'VIDEO_TEST_CROPS': TEST_CROPS})
    return cfg


# ============================================================
# DATA
# ============================================================
def read_rows(name):
    """Crime-moment clips + Background clips, as V7."""
    return t6.read_split(name) + t6.read_split(name + '_bg')


def read_whole(name):
    path = os.path.join(t6.SPLIT_DIR, f'{name}_whole.csv')
    if not os.path.exists(path):
        how = 'qsub run_prepare_v9.sh' if name == 'val' else 'qsub run_eval_v7.sh'
        sys.exit(f'  ERROR: {path} not found. Run first: {how}')
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['label'] = int(r['label'])
    return rows


def thin(rows, n):
    """Smoke test only: n rows spread over the list, so several classes are in."""
    return rows[::max(1, len(rows) // n)][:n]


class ClipDatasetV9(t6.ClipDataset):
    """Training clips: change 2. A random 40-100% piece of the stored moment, then one random
    frame from each eighth of that piece. Evaluation clips are the V6 ones, unchanged."""

    def __getitem__(self, idx):
        if not self.train:
            return super().__getitem__(idx)
        r = self.rows[idx]
        jpg, off = t6._open_cache(r['cache_path'])
        n = len(off) - 1
        span = min(n, max(t6.NUM_FRAMES, round(n * random.uniform(TCROP_MIN, 1.0))))
        start = random.randint(0, n - span)
        seg = span / t6.NUM_FRAMES
        ids = [min(n - 1, start + int((s + random.random()) * seg)) for s in range(t6.NUM_FRAMES)]
        return self._train_clip([t6._decode(jpg, off, k) for k in ids]), r['label']


class WholeVideo(Dataset):
    """Change 3. One whole video -> all its snippets, uint8, (snippets * crops, C, T, H, W).
    The 256 stored frames are cut into 32, then 16, 8 and 4 equal pieces; each piece gives one
    8-frame clip. The first 32 snippets are exactly the V7 snippets."""

    def __init__(self, rows, crops):
        self.rows, self.crops = rows, crops

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        jpg, off = t6._open_cache(r['cache_path'])
        n = len(off) - 1
        resized, views = {}, []
        for pieces in SCALES:
            per = n / pieces
            for s in range(pieces):
                frames = []
                for k in range(t6.NUM_FRAMES):
                    j = min(n - 1, int(s * per + (k + 0.5) * per / t6.NUM_FRAMES))
                    if j not in resized:
                        resized[j] = TF.resize(t6._decode(jpg, off, j), t6.IMG_SIZE)
                    frames.append(resized[j])
                w, h = frames[0].size
                for c in range(self.crops):
                    if w >= h:
                        top, left = (h - t6.IMG_SIZE) // 2, t6.crop_positions(w, self.crops)[c]
                    else:
                        top, left = t6.crop_positions(h, self.crops)[c], (w - t6.IMG_SIZE) // 2
                    clip = np.stack([np.asarray(TF.crop(f, top, left, t6.IMG_SIZE, t6.IMG_SIZE)) for f in frames])
                    views.append(torch.from_numpy(clip).permute(3, 0, 1, 2))     # (C, T, H, W) uint8
        return torch.stack(views), r['label']


# ============================================================
# WHOLE-VIDEO SCORING (validation every epoch, and the final test)
# ============================================================
@torch.no_grad()
def video_probs(model, loader, device, crops):
    """Per video: (snippets, classes) probabilities, averaged over the crops."""
    model.eval()
    mean, std = t6.MEAN.to(device)[None], t6.STD.to(device)[None]
    out, labels = [], []
    for x, y in loader:
        flat = x[0]
        ps = []
        for s in range(0, len(flat), t6.EVAL_CHUNK):
            xb = flat[s:s + t6.EVAL_CHUNK].to(device, non_blocking=True).float().div_(255)
            ps.append(model((xb - mean) / std).float().softmax(-1).cpu())
        p = torch.cat(ps)
        out.append(p.view(-1, crops, p.shape[-1]).mean(1))
        labels.append(int(y[0]))
    return out, np.array(labels)


def pool(p, n_crime):
    """(snippets, classes) -> three score vectors over the crimes. Background is never an answer."""
    crime = p[:, :n_crime]
    n0 = SCALES[0]
    k = max(1, round(len(crime) / TOPK_DIV))                                  # 60 snippets -> top-4
    return (crime.topk(k, dim=0).values.mean(0),                               # V9: all four scales
            crime[:n0].topk(max(1, n0 // TOPK_DIV), dim=0).values.mean(0),     # V7: 32 pieces, top-2
            crime[:n0].mean(0))                                               # plain average


def video_eval(model, loader, device, n_crime, crops):
    probs, labels = video_probs(model, loader, device, crops)
    pooled = [pool(p, n_crime) for p in probs]
    res = {'labels': labels, 'probs': probs}
    for i, name in enumerate(('v9', 'v7', 'mean')):
        s = torch.stack([q[i] for q in pooled])
        pred = s.argmax(1).numpy()
        acc, bal = e7.scores(pred, labels, n_crime)
        res[name] = {'scores': s.numpy(), 'pred': pred, 'acc': acc, 'bal': bal}
    return res


def balanced_from_cm(cm):
    support = cm.sum(1)
    ok = support > 0
    return 100 * float((cm.diag()[ok].float() / support[ok].float()).mean())


# ============================================================
# LOSS (change 1)
# ============================================================
class BalancedCrimeLoss(nn.Module):
    """Cross-entropy on logits + tau * log(share of this crime among the crime clips x 13).
    Frequent crimes get a plus, rare ones a minus, so to be right in training a rare crime
    needs a higher raw score; at test time the raw scores are used and no crime is favoured
    for being frequent. Background gets 0 (its share is left as it is)."""

    def __init__(self, counts, n_crime, tau, smoothing):
        super().__init__()
        c = torch.as_tensor(np.asarray(counts[:n_crime]), dtype=torch.float32).clamp(min=1)
        adj = torch.zeros(len(counts))
        adj[:n_crime] = tau * torch.log(c * n_crime / c.sum())
        self.register_buffer('adj', adj)
        self.smoothing = smoothing

    def forward(self, logits, target):
        return F.cross_entropy(logits + self.adj, target, label_smoothing=self.smoothing)


# ============================================================
# MODEL
# ============================================================
def build_model(vm, ms7, num_classes, device):
    freeze_until = t6.NUM_LAYERS - t6.UNFREEZE_LAST
    model = vm.videomamba_small(
        pretrained=False, num_classes=num_classes, num_frames=t6.NUM_FRAMES,
        fc_drop_rate=t6.HEAD_DROPOUT, drop_path_rate=t6.DROP_PATH,
        ssm_cfg={'gnconv_from_layer': freeze_until},
    )
    assert isinstance(model.layers[0].mixer, ms7.Mamba), 'model is not using mamba_simple_v7'
    assert model.layers[0].mixer.gnconv is None and model.layers[-1].mixer.gnconv is not None

    if GNCONV == 'st':
        from gnconv_st_v9 import gnConvST
    for i in range(freeze_until, t6.NUM_LAYERS):
        mx = model.layers[i].mixer
        if GNCONV == 'none':
            mx.gnconv = mx.gnconv_b = None
        elif GNCONV == 'st':
            mx.gnconv = gnConvST(mx.d_inner, t6.NUM_FRAMES)
            mx.gnconv_b = gnConvST(mx.d_inner, t6.NUM_FRAMES)

    # ---- starting weights: everything except gnConv (starts at zero output) and the head (new) ----
    log(f'\n  Loading the {INIT} checkpoint: {t6.PRETRAINED_PATH}')
    if not os.path.exists(t6.PRETRAINED_PATH):
        sys.exit('  ERROR: starting checkpoint not found. Refusing to train from random weights.')
    ckpt = t6.torch_load(t6.PRETRAINED_PATH)
    for key in ('model', 'state_dict', 'module'):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            ckpt = ckpt[key]
    ckpt = {(k[7:] if k.startswith('module.') else k): v for k, v in ckpt.items()}
    state = {k: v for k, v in ckpt.items() if not k.startswith('head.') and '.gnconv' not in k}
    msg = model.load_state_dict(state, strict=False)
    bad_missing = [k for k in msg.missing_keys if not k.startswith('head.') and '.gnconv' not in k]
    log(f'  Loaded {len(state)} tensors | not loaded from the file: {len(ckpt) - len(state)} (old gnConv, old head) '
        f'| new in the model: {len(msg.missing_keys)} (gnConv of layers {freeze_until}-{t6.NUM_LAYERS - 1}, head)')
    if bad_missing or msg.unexpected_keys:
        log(f'  Missing backbone keys: {bad_missing[:10]}')
        log(f'  Unexpected keys: {msg.unexpected_keys[:10]}')
        sys.exit('  ERROR: checkpoint does not match the backbone.')
    del ckpt, state

    # ---- freeze layers 0-15, train layers 16-23 (with their gnConv) + final norm + head ----
    for p in model.parameters():
        p.requires_grad = False
    for i in range(freeze_until, t6.NUM_LAYERS):
        for p in model.layers[i].parameters():
            p.requires_grad = True
    for p in model.norm_f.parameters():
        p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True
    return model.to(device), freeze_until


def param_table(model):
    names = ['Patch embedding + positions', 'Mamba layers', 'gnConv modules', 'Norm layers',
             'Classification head']
    tot, trn = dict.fromkeys(names, 0), dict.fromkeys(names, 0)
    for n, p in model.named_parameters():
        if n.startswith('patch_embed') or n in ('cls_token', 'pos_embed', 'temporal_pos_embedding'):
            k = names[0]
        elif n.startswith('head.'):
            k = names[4]
        elif '.gnconv' in n:
            k = names[2]
        elif 'norm' in n:
            k = names[3]
        else:
            k = names[1]
        tot[k] += p.numel()
        trn[k] += p.numel() if p.requires_grad else 0
    T_, R_ = sum(tot.values()), sum(trn.values())
    log('')
    log(f"  {'=' * 70}")
    log(f'  MODEL PARAMETERS (V9, gnConv = {GNCONV})')
    log(f"  {'-' * 70}")
    log(f"  {'Component':<30s}{'Total':>13s}{'Trained':>13s}{'Frozen':>13s}")
    for k in names:
        log(f'  {k:<30s}{tot[k]:>13,}{trn[k]:>13,}{tot[k] - trn[k]:>13,}')
    log(f"  {'-' * 70}")
    log(f"  {'TOTAL':<30s}{T_:>13,}{R_:>13,}{T_ - R_:>13,}")
    log('  V7 (guide gnConv, 14 classes): 116,237,582 total, 99,169,934 trained')
    log(f"  {'=' * 70}")
    return {'total': T_, 'trainable': R_, 'frozen': T_ - R_, 'per_component_total': tot,
            'per_component_trainable': trn}


# ============================================================
# FINAL TEST (once, with the model chosen on validation)
# ============================================================
def final_test(model, classes, test_rows, device, best, params_info, compute_info):
    n_crime = len(classes) - 1
    crimes = classes[:n_crime]
    loader = DataLoader(WholeVideo(test_rows, TEST_CROPS), batch_size=1, shuffle=False,
                        num_workers=t6.NUM_WORKERS, worker_init_fn=t6._worker_init)
    log(f'\n  TEST: {len(test_rows)} whole test crime videos, {sum(SCALES)} snippets x {TEST_CROPS} crops each...')
    t0 = time.time()
    res = video_eval(model, loader, device, n_crime, TEST_CROPS)
    labels, main = res['labels'], res['v9']
    cm = torch.zeros(n_crime, n_crime, dtype=torch.long)
    for t_, p_ in zip(labels.tolist(), main['pred'].tolist()):
        cm[t_, p_] += 1
    metrics = t6.class_metrics(cm, crimes)
    macro_f1 = 100 * float(np.mean([m['f1'] for m in metrics if m['support'] > 0]))
    n = len(test_rows)
    half = 196 * math.sqrt(max(main['acc'] / 100 * (1 - main['acc'] / 100), 1e-9) / n)
    bg_share = 100 * float(np.mean([float((p[:SCALES[0]].argmax(1) == n_crime).float().mean())
                                    for p in res['probs']]))

    log('')
    log('=' * 74)
    log(f'V9 TEST: {TAG}, seed {t6.SEED}, model from epoch {best["epoch"]} (EMA weights), chosen on whole-video')
    log(f'validation (balanced {best["val_video_bal"]:.2f}%, accuracy {best["val_video_acc"]:.2f}%)')
    log('=' * 74)
    log(f"  V9 pooling, 4 time scales (main)  : {main['acc']:.2f}%  (balanced {main['bal']:.2f}%, "
        f"macro-F1 {macro_f1:.2f}%, {int(cm.diag().sum())}/{n} correct)")
    log(f"  V7 pooling (32 pieces, top-2)     : {res['v7']['acc']:.2f}%  (balanced {res['v7']['bal']:.2f}%)  same model")
    log(f"  plain average of the 32 pieces    : {res['mean']['acc']:.2f}%  (balanced {res['mean']['bal']:.2f}%)")
    log(f'  short pieces called Background    : {bg_share:.1f}%')
    log(f'  One test video = {100 / n:.2f} points. A single run on {n} videos has a 95% range of about '
        f'+-{half:.1f} points.')
    log(f'  V7 on the same test: {V7_RESULT}')
    log('  Published on this test (CLIP, open-vocabulary, 6 of 13 crimes trained): '
        + ', '.join(f'{acc:.2f}%' for _, acc in e7.PAPERS) + ' (Wu et al. 2024, Anomize 2025)')
    log('')
    log(f"  {'Class':<16s}{'V9 correct':>12s}{'Recall':>9s}{'Precision':>11s}{'F1':>8s}{'V7 pooling':>13s}")
    for i, m in enumerate(metrics):
        v7_ok = int(((res['v7']['pred'] == i) & (labels == i)).sum())
        log(f"  {m['class']:<16s}{m['correct']:>7d}/{m['support']:<4d}{100 * m['recall']:>8.1f}%"
            f"{100 * m['precision']:>10.1f}%{100 * m['f1']:>7.1f}%{v7_ok:>9d}/{m['support']:<3d}")
    log('')
    t6.print_confusion(cm, crimes)
    log(f'  ({time.time() - t0:.0f}s)')
    log('=' * 74)

    os.makedirs(t6.RESULT_DIR, exist_ok=True)
    with open(os.path.join(t6.RESULT_DIR, 'test_predictions.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['video', 'true', 'predicted', 'score', 'correct', 'predicted_v7_pooling', 'predicted_average'])
        for r, t_, a, b, c, s in zip(test_rows, labels, main['pred'], res['v7']['pred'], res['mean']['pred'],
                                     main['scores']):
            w.writerow([os.path.basename(r['video_path']), crimes[t_], crimes[a], f'{s.max():.3f}', int(t_ == a),
                        crimes[b], crimes[c]])
    with open(os.path.join(t6.RESULT_DIR, 'confusion_matrix.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['true \\ predicted'] + crimes)
        for i, c in enumerate(crimes):
            w.writerow([c] + cm[i].tolist())
    with open(os.path.join(t6.RESULT_DIR, 'per_class_metrics.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        w.writeheader()
        w.writerows(metrics)
    np.savez(os.path.join(t6.RESULT_DIR, 'test_scores.npz'), v9=main['scores'], v7=res['v7']['scores'],
             labels=labels)
    summary = {'tag': TAG, 'seed': t6.SEED, 'test_acc': main['acc'], 'test_balanced_acc': main['bal'],
               'test_macro_f1': macro_f1, 'test_acc_v7_pooling': res['v7']['acc'],
               'test_acc_average': res['mean']['acc'], 'test_videos': n, 'ci95_half_width': half,
               'background_share': bg_share, 'best': best, 'params': params_info, 'compute': compute_info,
               'per_class': metrics, 'config': config_dict()}
    with open(os.path.join(t6.RESULT_DIR, 'test_results.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log(f'  Saved: {t6.RESULT_DIR}/ (test_results.json, test_predictions.csv, confusion_matrix.csv, '
        f'per_class_metrics.csv, test_scores.npz)')
    record_run(res, best)


def record_run(res, best):
    """One row per tag + seed in results_v9_all_runs.csv, the mean over seeds, and the ensemble."""
    fields = ['tag', 'seed', 'acc', 'balanced_acc', 'acc_v7_pooling', 'acc_average', 'val_video_bal',
              'val_video_acc', 'best_epoch', 'finished']
    rows = []
    if os.path.exists(ALL_RUNS_CSV):
        with open(ALL_RUNS_CSV, newline='') as f:
            rows = [r for r in csv.DictReader(f) if not (r['tag'] == TAG and r['seed'] == str(t6.SEED))]
    rows.append({'tag': TAG, 'seed': t6.SEED, 'acc': f"{res['v9']['acc']:.2f}",
                 'balanced_acc': f"{res['v9']['bal']:.2f}", 'acc_v7_pooling': f"{res['v7']['acc']:.2f}",
                 'acc_average': f"{res['mean']['acc']:.2f}", 'val_video_bal': f"{best['val_video_bal']:.2f}",
                 'val_video_acc': f"{best['val_video_acc']:.2f}", 'best_epoch': best['epoch'],
                 'finished': time.strftime('%Y-%m-%d %H:%M')})
    rows.sort(key=lambda r: (r['tag'], int(r['seed'])))
    with open(ALL_RUNS_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    log('')
    log('  ALL V9 RUNS SO FAR (mean over the seeds of each variant; validation decides which variant is final):')
    log(f"  {'variant':<22s}{'seeds':>7s}{'test acc':>18s}{'balanced':>10s}{'V7 pooling':>12s}{'val balanced':>14s}")
    for tag in sorted({r['tag'] for r in rows}):
        same = [r for r in rows if r['tag'] == tag]
        v = {k: np.array([float(r[k]) for r in same]) for k in ('acc', 'balanced_acc', 'acc_v7_pooling',
                                                                 'val_video_bal')}
        spread = f" +- {v['acc'].std():.2f}" if len(same) > 1 else ''
        log(f"  {tag:<22s}{len(same):>7d}{v['acc'].mean():>10.2f}%{spread:<8s}{v['balanced_acc'].mean():>9.2f}%"
            f"{v['acc_v7_pooling'].mean():>11.2f}%{v['val_video_bal'].mean():>13.2f}%")

    scores, labels = [], None
    for r in rows:
        if r['tag'] != TAG:
            continue
        path = os.path.join(RUNS_DIR, ('smoke_' if SMOKE else '') + f"{TAG}_seed{r['seed']}", 'results',
                            'test_scores.npz')
        if os.path.exists(path):
            with np.load(path) as d:
                scores.append(d['v9'])
                labels = d['labels']
    if SMOKE:
        scores = scores * 2                       # smoke test: run the ensemble code with one model twice
    if len(scores) > 1:
        pred = np.mean(scores, axis=0).argmax(1)
        acc, bal = e7.scores(pred, labels, scores[0].shape[1])
        log(f'  Ensemble of the {len(scores)} {TAG} models (their scores averaged, so {len(scores)}x the compute): '
            f'{acc:.2f}%  (balanced {bal:.2f}%). Report it only as an ensemble.')
    log(f'  V7: {V7_RESULT}. Table: {ALL_RUNS_CSV}')


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--gnconv', default='guide', choices=['guide', 'st', 'none'])
    ap.add_argument('--init', default='ucf101', choices=sorted(INIT_CHECKPOINTS))
    ap.add_argument('--tau', type=float, default=1.0, help='1 = balanced crimes (default), 0 = off')
    ap.add_argument('--test-only', action='store_true', help='skip training, test best_model.pth')
    ap.add_argument('--no-resume', action='store_true', help='ignore last.pth and start fresh')
    ap.add_argument('--smoke', type=int, default=0, choices=[0, 1, 2],
                    help='1 = one epoch on 16 clips then stop; 2 = resume, second epoch, test on 3 videos')
    args = ap.parse_args()
    setup_run(args.seed, args.gnconv, args.init, args.tau, args.smoke)
    for d in (t6.CKPT_DIR, t6.LOG_DIR, t6.RESULT_DIR):
        os.makedirs(d, exist_ok=True)
    t6.seed_everything(t6.SEED)

    log('=' * 74)
    log(f'VideoMamba V9, UCF-Crime final model: variant {TAG}, seed {t6.SEED}'
        + (f'   [SMOKE TEST stage {SMOKE}: a few clips only, numbers mean nothing]' if SMOKE else ''))
    log('=' * 74)
    log(f'  Started: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    log(f'  Run folder: {t6.RUN_DIR}')
    if not torch.cuda.is_available():
        sys.exit('  ERROR: no GPU visible. Mamba selective scan needs CUDA: submit through PBS on a GPU node.')
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True
    free, total = torch.cuda.mem_get_info()
    log(f'  GPU: {torch.cuda.get_device_name(0)} | free {free / 1e9:.1f} / {total / 1e9:.1f} GB '
        f'| PyTorch {torch.__version__}')
    if free < 8e9:
        log('  [WARN] less than 8 GB free on this GPU (another user?). Training may run out of memory.')

    log('\n  MODEL: VideoMamba-Small, 24 bidirectional Mamba layers; layers 0-15 frozen, 16-23 trained')
    log({'guide': '    gnConv: the guide\'s HorNet gnConv2D (order 5, kernels 3/5/7/1, GhostModule), unchanged',
         'st': '    gnConv: UPDATED (gnconv_st_v9.py): separable 3x3/5x5/7x7 + gates that see neighbouring frames',
         'none': '    gnConv: NONE (comparison run: plain VideoMamba layers)'}[GNCONV])
    log(f'  START: {"Kinetics-400 -> UCF-101" if INIT == "ucf101" else "Kinetics-400 only"} checkpoint; '
        f'gnConv starts at zero output')
    log('  WHAT V9 CHANGES (rest = V7 recipe from train_v6.py):')
    log(f'    1. loss: crimes balanced by logit adjustment, tau = {TAU:g}' + ('' if TAU else '  (OFF)'))
    log(f'    2. training clips: a random {TCROP_MIN:.0%}-100% piece of the stored crime moment')
    log(f'    3. test: the video cut into {" / ".join(map(str, SCALES))} pieces = {sum(SCALES)} snippets, '
        f'top-{max(1, round(sum(SCALES) / TOPK_DIV))} per crime')
    log('    4. best epoch: whole-video validation (balanced accuracy, EMA weights), '
        f'early stop after {t6.PATIENCE} epochs without a new best')
    log(f'  RECIPE: {t6.NUM_FRAMES} frames | batch {t6.BATCH_SIZE} x {t6.ACCUM_STEPS} | {t6.EPOCHS} epochs '
        f'(warmup {t6.WARMUP_EPOCHS}) | LR {t6.LR_BACKBONE}/{t6.LR_GNCONV}/{t6.LR_HEAD} | wd {t6.WEIGHT_DECAY} '
        f'| dropout {t6.HEAD_DROPOUT} | drop path {t6.DROP_PATH} | smoothing {t6.LABEL_SMOOTH} '
        f'| mixup {t6.MIXUP_ALPHA} | EMA {t6.EMA_DECAY}')
    log('=' * 74)

    # ---------------- data ----------------
    classes = t6.read_classes()
    num_classes, n_crime = len(classes), len(classes) - 1
    if classes[-1] != 'Background':
        sys.exit(f'  ERROR: the last class in {t6.CLASSES_FILE} must be Background.')
    train_rows, val_rows = read_rows('train'), read_rows('val')
    val_whole, test_whole = read_whole('val'), read_whole('test')
    if SMOKE:
        train_rows, val_rows = thin(train_rows, 16), thin(val_rows, 8)
        val_whole, test_whole = thin(val_whole, 3), thin(test_whole, 3)
    missing = [r['cache_path'] for r in train_rows + val_rows + val_whole + test_whole
               if not os.path.exists(r['cache_path'])]
    if missing:
        sys.exit(f'  ERROR: {len(missing)} cached videos missing (e.g. {missing[0]}).')
    counts = {name: np.bincount(np.array([r['label'] for r in rows], dtype=np.int64), minlength=num_classes)
              for name, rows in (('train', train_rows), ('val', val_rows), ('val_whole', val_whole),
                                 ('test', test_whole))}
    criterion = BalancedCrimeLoss(counts['train'], n_crime, TAU, t6.LABEL_SMOOTH).to(device)
    log(f'\n  DATA (from {t6.SPLIT_DIR}):')
    log(f"  {'Class':<16s}{'Train clips':>12s}{'Val clips':>11s}{'Val videos':>12s}{'Test videos':>13s}"
        f"{'Loss offset':>13s}")
    for i, c in enumerate(classes):
        log(f"  {c:<16s}{counts['train'][i]:>12d}{counts['val'][i]:>11d}{counts['val_whole'][i]:>12d}"
            f"{counts['test'][i]:>13d}{criterion.adj[i].item():>+13.2f}")
    log(f"  {'TOTAL':<16s}{len(train_rows):>12d}{len(val_rows):>11d}{len(val_whole):>12d}{len(test_whole):>13d}")
    log('  (Loss offset: plus = seen often in training, so it must earn its score; minus = rare.)')

    # ---------------- model ----------------
    vm, ms7 = t6.load_backbone_module()
    t6.self_test_gnconv(ms7, device)
    if GNCONV == 'st':
        from gnconv_st_v9 import self_test as self_test_st
        start_ok, time_ok, cls_ok, text = self_test_st(device, t6.NUM_FRAMES)
        log(f'  [CHECK] gnConv-ST: {text}')
        if not (start_ok and time_ok and cls_ok):
            sys.exit('  ERROR: gnConv-ST self-test failed.')
    model, freeze_until = build_model(vm, ms7, num_classes, device)
    params_info = param_table(model)
    compute_info = t6.measure_complexity(model, freeze_until, device)
    torch.cuda.empty_cache()

    if args.test_only:
        if not os.path.exists(t6.BEST_PATH):
            sys.exit(f'  ERROR: {t6.BEST_PATH} not found.')
        ck = t6.torch_load(t6.BEST_PATH)
        model.load_state_dict(ck['state_dict'])
        final_test(model, classes, test_whole, device, ck['best'], params_info, compute_info)
        return

    workers = dict(num_workers=t6.NUM_WORKERS, persistent_workers=t6.NUM_WORKERS > 0,
                   worker_init_fn=t6._worker_init)
    train_loader = DataLoader(ClipDatasetV9(train_rows, train=True), batch_size=t6.BATCH_SIZE, shuffle=True,
                              pin_memory=True, drop_last=True, **workers)
    val_loader = DataLoader(t6.ClipDataset(val_rows, train=False, n_clips=t6.VAL_CLIPS, n_crops=t6.VAL_CROPS),
                            batch_size=1, shuffle=False, pin_memory=True, **workers)
    vid_loader = DataLoader(WholeVideo(val_whole, VAL_CROPS), batch_size=1, shuffle=False, **workers)

    optimizer = t6.build_optimizer(model)
    log('\n  Optimizer groups:')
    for g in optimizer.param_groups:
        log(f"    {g['name']:<18s} {sum(p.numel() for p in g['params']):>12,} params | "
            f"lr {g['base_lr']:.0e} | wd {g['weight_decay']}")
    ema = t6.ModelEMA(model, t6.EMA_DECAY)

    best = {'epoch': 0, 'val_video_bal': -1.0, 'val_video_acc': -1.0, 'val_clip_loss': float('inf'),
            'val_clip_acc': None, 'train_acc': None}
    history, patience, start_epoch = [], 0, 1
    resume = os.path.exists(t6.LAST_PATH) and not args.no_resume and SMOKE != 1
    if SMOKE == 2 and not resume:
        sys.exit(f'  ERROR: smoke stage 2 needs {t6.LAST_PATH} from stage 1.')
    if resume:
        ck = t6.torch_load(t6.LAST_PATH)
        cur = t6.trainable_state(model)
        with torch.no_grad():
            for n, t in ck['trainable'].items():
                cur[n].copy_(t)
        optimizer.load_state_dict(ck['optimizer'])
        ema.load_state_dict(ck['ema'])
        best, history, patience = ck['best'], ck['history'], ck['patience']
        start_epoch = ck['epoch'] + 1
        random.setstate(ck['rng']['py'])
        np.random.set_state(ck['rng']['np'])
        torch.set_rng_state(ck['rng']['torch'])
        log(f"\n  RESUMED from {t6.LAST_PATH}: continuing at epoch {start_epoch} "
            f"(best so far: validation balanced {best['val_video_bal']:.2f}% at epoch {best['epoch']})")
        del ck

    hist_path = os.path.join(t6.LOG_DIR, 'history.csv')
    hist_fields = ['epoch', 'train_loss', 'train_acc', 'val_clip_acc', 'val_clip_bal', 'val_clip_loss',
                   'val_video_bal', 'val_video_acc', 'val_video_bal_v7_pooling', 'gap', 'lr_backbone',
                   'lr_gnconv', 'lr_head', 'time_s', 'best_val_video_bal']
    log(f'\n  Training: {len(train_loader)} batches/epoch | clip validation {len(val_rows)} clips | '
        f'whole-video validation {len(val_whole)} videos x {sum(SCALES)} snippets')

    for epoch in range(start_epoch, t6.EPOCHS + 1):
        if patience >= t6.PATIENCE:
            break
        t0 = time.time()
        t6.set_lr(optimizer, epoch - 1)
        tr_loss, tr_acc = t6.train_one_epoch(model, train_loader, optimizer, criterion, ema, freeze_until,
                                             device, epoch, first_check=(epoch == start_epoch))
        with ema.applied(model):
            clip = t6.evaluate(model, val_loader, device, num_classes)
            vid = video_eval(model, vid_loader, device, n_crime, VAL_CROPS)
            improved = ((vid['v9']['bal'], vid['v9']['acc'], -clip['loss'])
                        > (best['val_video_bal'], best['val_video_acc'], -best['val_clip_loss']))
            if improved:
                best = {'epoch': epoch, 'val_video_bal': vid['v9']['bal'], 'val_video_acc': vid['v9']['acc'],
                        'val_clip_loss': clip['loss'], 'val_clip_acc': clip['acc'], 'train_acc': tr_acc}
                t6.save_checkpoint(t6.BEST_PATH, {'state_dict': model.state_dict(), 'best': best,
                                                  'classes': classes, 'config': config_dict()})
        patience = 0 if improved else patience + 1

        lrs = {g['name']: g['lr'] for g in optimizer.param_groups}
        elapsed = time.time() - t0
        gap = tr_acc - clip['acc']
        clip_bal = balanced_from_cm(clip['cm'])
        log(f"  Epoch {epoch:2d}/{t6.EPOCHS}: loss {tr_loss:.4f} | train {tr_acc:.1f}% | clip val {clip['acc']:.1f}% "
            f"(balanced {clip_bal:.1f}%, loss {clip['loss']:.3f}) | gap {gap:.1f}% | {elapsed:.0f}s")
        log(f"    WHOLE-VIDEO VAL: balanced {vid['v9']['bal']:.2f}% | accuracy {vid['v9']['acc']:.2f}% "
            f"| with V7 pooling: balanced {vid['v7']['bal']:.2f}%, accuracy {vid['v7']['acc']:.2f}%")
        if improved:
            log(f"    >>> NEW BEST (validation balanced {best['val_video_bal']:.2f}%) -> {t6.BEST_PATH}")
        else:
            log(f"    (no improvement {patience}/{t6.PATIENCE}; best {best['val_video_bal']:.2f}% at epoch "
                f"{best['epoch']})")
        if epoch == start_epoch:
            log(f'    peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB')

        history.append({'epoch': epoch, 'train_loss': round(tr_loss, 4), 'train_acc': round(tr_acc, 2),
                        'val_clip_acc': round(clip['acc'], 2), 'val_clip_bal': round(clip_bal, 2),
                        'val_clip_loss': round(clip['loss'], 4), 'val_video_bal': round(vid['v9']['bal'], 2),
                        'val_video_acc': round(vid['v9']['acc'], 2),
                        'val_video_bal_v7_pooling': round(vid['v7']['bal'], 2), 'gap': round(gap, 2),
                        'lr_backbone': lrs.get('backbone'), 'lr_gnconv': lrs.get('gnconv'),
                        'lr_head': lrs.get('head'), 'time_s': round(elapsed),
                        'best_val_video_bal': round(best['val_video_bal'], 2)})
        with open(hist_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=hist_fields)
            w.writeheader()
            w.writerows(history)
        t6.save_checkpoint(t6.LAST_PATH, {
            'epoch': epoch, 'trainable': {n: t.detach().cpu() for n, t in t6.trainable_state(model).items()},
            'optimizer': optimizer.state_dict(), 'ema': ema.state_dict(), 'best': best,
            'history': history, 'patience': patience,
            'rng': {'py': random.getstate(), 'np': np.random.get_state(), 'torch': torch.get_rng_state()}})
        if SMOKE == 1:
            log('\n  SMOKE STAGE 1 OK: one epoch trained, validated and saved. Stage 2 resumes from it.')
            return

    if patience >= t6.PATIENCE:
        log(f'\n  EARLY STOPPING: no better validation result for {t6.PATIENCE} epochs.')
    log('')
    log('=' * 74)
    log('TRAINING COMPLETE')
    log(f"  Best whole-video validation: balanced {best['val_video_bal']:.2f}%, accuracy "
        f"{best['val_video_acc']:.2f}% at epoch {best['epoch']} (train acc then {best['train_acc']:.1f}%)")
    log(f'  History (one row per epoch): {hist_path}')
    log('=' * 74)

    ck = t6.torch_load(t6.BEST_PATH)
    model.load_state_dict(ck['state_dict'])
    final_test(model, classes, test_whole, device, best, params_info, compute_info)
    if SMOKE:
        log('\n  SMOKE TEST PASSED: training, resuming, validation, test and result files all work.')
    log(f'\n  Finished: {time.strftime("%Y-%m-%d %H:%M:%S")}')


if __name__ == '__main__':
    main()
