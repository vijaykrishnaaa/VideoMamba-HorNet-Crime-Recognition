"""
V7 step 1: test the V6 models on the protocol of the CVPR 2024 / 2025 papers
(Wu et al. "Open-Vocabulary Video Anomaly Detection", Li et al. "Anomize"):
top-1 accuracy on the 140 official test crime videos, WHOLE untrimmed videos, 13 classes.

Same model, same backbone, same V6 checkpoints: no retraining. The V6 models never saw a test
video (they were trained on the other 810 crime videos), so this test is clean.

Each test video is cut into 32 snippets of 8 frames (256 cached frames over the whole video),
3 crops per snippet. Two ways to turn the snippet scores into one answer, both fixed in advance:
  mean  : average all snippet probabilities (what V5 / V6 did)
  top-k : Anomize Eq. 7-8 — for each class, average its k highest snippet probabilities,
          k = snippets / 16 = 2, then take the class with the highest value
Run: qsub run_eval_v7.sh
"""

import csv
import os
import sys
import time

import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset

import train_v6 as t6

SNIPPETS = 32
TOPK = SNIPPETS // 16                 # Anomize: M = T / 16
CROPS = 3
SEEDS = (42, 7, 123)
PAPERS = [('Wu et al., CVPR 2024 (CLIP, trained on 6 of the 13 classes)', 41.43),
          ('Anomize, CVPR 2025 (CLIP, trained on 6 of the 13 classes)', 47.14)]
OUT_CSV = os.path.join(t6.EXP_DIR, 'results_protocol_v7.csv')


class WholeVideo(Dataset):
    """(SNIPPETS * CROPS, C, T, H, W) views of one whole test video."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        jpg, off = t6._open_cache(r['cache_path'])
        n = len(off) - 1
        per = max(1, n // SNIPPETS)
        views = []
        for s in range(SNIPPETS):
            idx = [min(s * per + k * per // t6.NUM_FRAMES, n - 1) for k in range(t6.NUM_FRAMES)]
            frames = [TF.resize(t6._decode(jpg, off, k), t6.IMG_SIZE) for k in idx]
            w, h = frames[0].size
            for c in range(CROPS):
                if w >= h:
                    top, left = (h - t6.IMG_SIZE) // 2, t6.crop_positions(w, CROPS)[c]
                else:
                    top, left = t6.crop_positions(h, CROPS)[c], (w - t6.IMG_SIZE) // 2
                x = torch.stack([TF.to_tensor(TF.crop(f, top, left, t6.IMG_SIZE, t6.IMG_SIZE)) for f in frames], dim=1)
                views.append((x - t6.MEAN) / t6.STD)
        return torch.stack(views), r['label']


@torch.no_grad()
def snippet_probs(model, loader, device):
    """Per video: (SNIPPETS, classes) probabilities, averaged over the crops."""
    model.eval()
    out, labels = [], []
    for x, y in loader:
        flat = x[0]
        p = torch.cat([model(flat[s:s + t6.EVAL_CHUNK].to(device, non_blocking=True)).float().softmax(-1).cpu()
                       for s in range(0, len(flat), t6.EVAL_CHUNK)])
        out.append(p.view(SNIPPETS, CROPS, -1).mean(1))
        labels.append(int(y[0]))
    return out, np.array(labels)


def scores(pred, labels, n_classes):
    acc = 100 * float((pred == labels).mean())
    recalls = [float((pred[labels == c] == c).mean()) for c in range(n_classes) if (labels == c).any()]
    return acc, 100 * float(np.mean(recalls))


def main():
    t6.log('=' * 70)
    t6.log('V7 step 1 — V6 models on the CVPR 2024/2025 protocol (140 whole test crime videos)')
    t6.log('=' * 70)
    if not torch.cuda.is_available():
        sys.exit('ERROR: no GPU visible — submit through PBS.')
    device = torch.device('cuda')
    with open(os.path.join(t6.SPLIT_DIR, 'test_whole.csv'), newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['label'] = int(r['label'])
    classes = t6.read_classes()
    t6.log(f'  Test videos: {len(rows)} | snippets per video: {SNIPPETS} x {t6.NUM_FRAMES} frames, '
           f'{CROPS} crops | top-k: k = {TOPK}')
    loader = DataLoader(WholeVideo(rows), batch_size=1, shuffle=False, num_workers=4,
                        worker_init_fn=t6._worker_init)

    vm, ms7 = t6.load_backbone_module()
    t6.setup_run(SEEDS[0], 'ucf101')
    model, _ = t6.build_model(vm, ms7, len(classes), device)     # the structure; weights replaced below

    results = []
    for seed in SEEDS:
        t6.setup_run(seed, 'ucf101')
        if not os.path.exists(t6.BEST_PATH):
            t6.log(f'\n  seed {seed}: {t6.BEST_PATH} not found — skipped')
            continue
        model.load_state_dict(t6.torch_load(t6.BEST_PATH)['state_dict'])
        t0 = time.time()
        probs, labels = snippet_probs(model, loader, device)
        pred_mean = np.array([int(p.mean(0).argmax()) for p in probs])
        pred_topk = np.array([int(p.topk(TOPK, dim=0).values.mean(0).argmax()) for p in probs])
        acc_m, bal_m = scores(pred_mean, labels, len(classes))
        acc_k, bal_k = scores(pred_topk, labels, len(classes))
        results.append({'seed': seed, 'acc_mean': acc_m, 'bal_mean': bal_m, 'acc_topk': acc_k, 'bal_topk': bal_k})
        t6.log(f'\n  seed {seed}: mean pooling {acc_m:.2f}% (balanced {bal_m:.2f}%) | '
               f'top-{TOPK} pooling {acc_k:.2f}% (balanced {bal_k:.2f}%) | {time.time() - t0:.0f}s')
        t6.log('    per class, top-k (correct/total): ' + ' | '.join(
            f'{c[:10]} {int(((pred_topk == i) & (labels == i)).sum())}/{int((labels == i).sum())}'
            for i, c in enumerate(classes)))
        with open(os.path.join(t6.RESULT_DIR, 'protocol_v7_predictions.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['video', 'true', 'pred_mean', 'pred_topk'])
            for r, t_, a, b in zip(rows, labels, pred_mean, pred_topk):
                w.writerow([r['video_path'], classes[t_], classes[a], classes[b]])

    if not results:
        sys.exit('ERROR: no V6 checkpoint found.')
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    mean = {k: np.mean([r[k] for r in results]) for k in ('acc_mean', 'bal_mean', 'acc_topk', 'bal_topk')}
    t6.log('')
    t6.log('=' * 70)
    t6.log(f'RESULT over seeds {[r["seed"] for r in results]} (top-1 on the {len(rows)} whole test crime videos):')
    t6.log(f"  mean pooling : {mean['acc_mean']:.2f}%  (balanced {mean['bal_mean']:.2f}%)")
    t6.log(f"  top-{TOPK} pooling: {mean['acc_topk']:.2f}%  (balanced {mean['bal_topk']:.2f}%)")
    t6.log('  Published on this protocol (open-vocabulary, so not identical: we train on all 13 classes):')
    for name, acc in PAPERS:
        t6.log(f'    {name}: {acc:.2f}%')
    t6.log(f'  Saved: {OUT_CSV}')
    t6.log('=' * 70)


if __name__ == '__main__':
    main()
