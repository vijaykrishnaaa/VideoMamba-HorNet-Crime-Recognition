# V9 explained: what we changed, why, and what to expect

Written 11 November 2026, before the V9 runs. Section 10 has the results of the main run (12 November).

## 1. In short

V9 is the last attempt on UCF-Crime, on the same test the CVPR 2024 and 2025 papers use
(top-1 accuracy on the 140 official test crime videos, whole and untrimmed).

- **Same model:** VideoMamba-Small with the guide's HorNet gnConv2D in layers 16 to 23, layers 0 to 15 frozen.
- **Same data:** the crime moments and Background clips already made for V6 and V7. Nothing new is downloaded.
- **Same test:** so V9 can be compared with V7 (38.33%) and with the papers (41.43%, 47.14%).

We did not guess what to change. We read our own V6 and V7 results, found four problems, and each
change answers one of them:

| # | Change | Problem it answers | Source |
|---|---|---|---|
| 1 | Balanced crimes in the loss | Shoplifting is called Robbery in 14 to 15 of 21 test videos | Menon et al., ICLR 2021 |
| 2 | Train on pieces of the crime moment | Training clips are much longer than the test pieces | TSN sampling, Wang et al., ECCV 2016 |
| 3 | Test at four time scales | Same problem, from the test side | Top-k rule of Anomize, CVPR 2025 |
| 4 | Pick the best epoch with the real test, on validation videos | V7's validation score did not follow its test score | Standard practice |

A fifth item is optional and runs separately: an **updated gnConv** that is 3.4 times smaller per
module and also looks at neighbouring frames (section 5).

## 2. Where we stand before V9

| Model | Test | Result |
|---|---|---|
| V7 (ours, 3 seeds) | 140 whole test crime videos | 38.33% +- 3.8 (42.86 / 33.57 / 38.57) |
| Wu et al., CVPR 2024 | same | 41.43% |
| Anomize, CVPR 2025 | same | 47.14% |

The two papers use CLIP (pretrained on 400 million image-caption pairs) and train on 6 of the 13
crimes. We train on all 13 and use the UCA time stamps. It is not a like-for-like comparison.
Two searches on 11 November 2026 found no newer paper that reports this accuracy.

## 3. What we found in our own results

### 3.1 The test asks most about the crimes we have the fewest training videos for

From `logs/prep_report.txt` (V6 prep):

| Class | Train | Val | Test |
|---|---|---|---|
| Robbery | 123 | 22 | 5 |
| RoadAccidents | 108 | 19 | 23 |
| Stealing | 81 | 14 | 5 |
| Burglary | 74 | 13 | 13 |
| Abuse | 41 | 7 | 2 |
| Assault | 40 | 7 | 3 |
| Arrest | 38 | 7 | 5 |
| Fighting | 38 | 7 | 5 |
| Vandalism | 38 | 7 | 5 |
| Arson | 35 | 6 | 9 |
| **Explosion** | **25** | 4 | **21** |
| **Shoplifting** | **25** | 4 | **21** |
| **Shooting** | **23** | 4 | **23** |
| Total | 689 | 121 | 140 |

- Shooting, Shoplifting and Explosion are **65 of the 140 test videos (46%)** but only **73 of the 689 training videos (11%)**.
- Robbery, Stealing and Burglary are 278 training videos (40%) but only 23 test videos (16%).

So the model learns that Robbery is common, and the test then asks mostly about other crimes.

### 3.2 Shoplifting is called Robbery, and Shooting is not recognised

From the three V6 confusion matrices (`runs/seed*/results/confusion_matrix.csv`):

| | Seed 42 | Seed 7 | Seed 123 |
|---|---|---|---|
| Shoplifting videos called Robbery (of 21) | 15 | 14 | 15 |
| Shoplifting correct (of 21) | 4 | 5 | 4 |
| Shooting correct (of 23) | 1 | 2 | 0 |

The pattern is the same in every seed, so it is not bad luck. Both crimes happen in shops, and the
model has seen 123 Robbery videos against 25 Shoplifting videos. For Shooting the answers are
spread over eight other classes with low confidence (0.17 to 0.43 in seed 42): the model is unsure.
These two classes are 44 of the 140 test videos, and we get about 5 of them right.

### 3.3 Training clips are longer than the pieces the test looks at

- **Training:** the crime moment has a median length of 22.6 s. A clip is 8 frames spread over the whole moment, so about one frame every 3 s.
- **Test (V7):** the video is cut into 32 pieces. For a 2-minute video one piece is under 4 s, so 8 frames about half a second apart.

The model was trained on long, sparse views and tested on short, dense ones. In V7 (seed 42)
85% of the test pieces were called Background.

### 3.4 The best epoch was chosen with a different test than the final one

V7 chose its best epoch on validation **clips** (crime moments and Background clips). The
validation accuracy was about 60% in all three seeds, but the test accuracy was 42.86, 33.57 and
38.57%. A validation score that stays the same while the test moves by 9 points cannot tell us
which epoch is best for whole videos.

### 3.5 The test is small

140 videos means one video is 0.71 points. A single run at 40% has a 95% range of about +-8 points
(binomial). So 38.33% against 41.43% is inside the noise, and any claim needs the mean of several seeds.

## 4. The four changes

### Change 1: balanced crimes in the loss (logit adjustment)

**What.** During training a fixed number is added to the score of each crime before the loss:
`ln(13 x clips of this crime / 689)`. Frequent crimes get a plus, rare crimes a minus.
At test time nothing is added.

| Class | Offset | Class | Offset |
|---|---|---|---|
| Robbery | +0.84 | Assault | -0.28 |
| RoadAccidents | +0.71 | Arrest, Fighting, Vandalism | -0.33 |
| Stealing | +0.42 | Arson | -0.41 |
| Burglary | +0.33 | Explosion, Shoplifting | -0.75 |
| Abuse | -0.26 | Shooting | -0.83 |
| Background | 0.00 | | |

**In plain words.** To be marked right in training, a Shoplifting clip must beat Robbery by a
clear margin (1.59 = ln(123/25)), because Robbery starts with a head start. The model learns to
give rare crimes higher raw scores. In the test the head start is gone, so no crime wins just
because it was seen often.

**Why not the class weights V7 already had.** Weights make rare clips count more in the loss, but
their effect fades as training goes on (Byrd and Lipton, 2019), and the logit-adjustment paper
notes that re-weighting has little effect once the model separates the training data. That paper
reports a lower balanced error than plain training on all four of its datasets:

| Dataset | Plain training | Logit-adjusted loss |
|---|---|---|
| CIFAR-10-LT | 27.16 | 22.33 |
| CIFAR-100-LT | 61.64 | 56.11 |
| ImageNet-LT | 53.11 | 48.89 |
| iNaturalist | 38.66 | 33.64 |

**Why Background gets 0.** In a real video most pieces are background. If Background were
"balanced" too, the model would see a crime in every calm piece. So only the 13 crimes are
balanced against each other.

**Why we do not oversample rare crimes.** Kang et al. (ICLR 2020) found that ordinary sampling
learns the best features; the balance belongs in the classifier, which is where this loss puts it.

**Code:** `BalancedCrimeLoss` in `train_v9.py`. Off switch: `TAU=0`.

### Change 2: train on pieces of the crime moment

**What.** Each training clip now covers a random 40 to 100% piece of the stored moment. Inside
the piece, one frame is taken from each eighth (the TSN way, as before).

**Why.** The model sees short and long views of the same crime, closer to what the test shows it.
It is also a new augmentation: every epoch shows a different part of each rare-class video.

**Code:** `ClipDatasetV9` in `train_v9.py`. No new data is stored: it uses the 32 frames per moment that V6 saved.

### Change 3: test at four time scales

**What.** The video is cut into 32, then 16, then 8, then 4 equal pieces: 60 snippets of 8 frames.
For each crime the 4 highest snippet scores are averaged (top-k with k = snippets / 16, the rule
of Anomize), and the crime with the highest value is the answer. Background is never an answer.

**Why.** The longer pieces look like the training clips. A short crime is still caught by the
short pieces; a long one (an arrest, a burglary) is seen as a whole by the long pieces.

**Fair comparison with V7.** The first 32 snippets are exactly the V7 snippets, so the log also
prints the V7-style result of the same model. Both numbers are reported.

**Code:** `WholeVideo` and `pool` in `train_v9.py`. It reuses the 256 frames per test video that V7 saved.

### Change 4: pick the best epoch with the real test, on validation videos

**What.** After every epoch the model is tested on the 121 validation videos, whole, exactly as
in change 3. The epoch with the highest **balanced** accuracy is kept. Only the EMA weights are
tested. Training stops after 10 epochs without a new best.

**Why balanced accuracy.** The validation videos have the training mix (22 Robbery, 4 Shooting).
Balanced accuracy counts every crime the same, which is what the loss now aims for.

**Is this fair?** Yes. The validation videos are never trained on, and the 140 test videos are
touched once, at the end.

**Code:** `video_eval` and the epoch loop in `train_v9.py`; `prepare_val_whole_v9.py` stores the validation videos.

## 5. The updated gnConv (optional run `st`)

| Part | Guide's gnConv2D | Updated gnConv-ST |
|---|---|---|
| Order, channel steps | 5; 48, 96, 192, 384, 768 | same |
| Input | GhostModule 768 to 1,536 | same |
| Gates | 7x7 depthwise conv on each frame | same, **then a depthwise conv over 3 frames in time** |
| Between the orders | full 3x3, 5x5, 7x7 convs, then 1x1 | **depthwise 3x3, 5x5, 7x7, each followed by 1x1**, then 1x1 |
| Output | 1x1 conv, starts at zero | same |
| Place in the model | beside Conv1d, both scan directions, layers 16 to 23 | same |
| Parameters, one module | 5,676,096 | 1,674,960 |
| Whole model (worked out by hand; the job measures it) | 116.2M, 185 GMACs per clip | about 52.2M, about 85 GMACs |

- **Separable convolutions.** The three full convolutions hold 4.12M of the 5.68M parameters of a module. Splitting each into "look around" (depthwise) and "mix channels" (1x1) is the MobileNet idea. The original HorNet uses only 1x1 here, so the update sits between HorNet and the guide's version.
- **Neighbouring frames.** gnConv2D looks at one frame at a time. The time convolution lets the gates see the frame before and after, the "(2+1)D" idea of Tran et al. (CVPR 2018). It starts as "this frame only", so training begins exactly like the per-frame version.
- **Why it might help:** 3.4 times fewer new parameters to learn from 689 videos, and direct access to motion.
- **Why it might not:** less capacity. That is why it is a separate run, and validation decides.

**Code:** `gnconv_st_v9.py` (the guide's class is reused; only two parts are swapped).

## 6. What we did not change, and why

| Kept | Why |
|---|---|
| VideoMamba-Small, layers 0 to 15 frozen, gnConv in 16 to 23 | The guide asked for the same backbone; the paper and diagrams describe it |
| 8 frames, 224 x 224 | Keeps the 370 GFLOPs and 58 ms figures valid |
| UCF-101 starting weights | Same as V4 to V8 (`k400` is an optional comparison) |
| Learning rates, weight decay, dropout, drop path, mixup, EMA | They worked in V6 and V7; changing them too would hide what helped |
| The data | Same 689 + 500 training clips, same 140 test videos |

## 7. How overtraining is kept out

- Layers 0 to 15 are frozen.
- Dropout 0.5 before the classifier, stochastic depth 0.2, weight decay 0.05.
- Label smoothing 0.1 and mixup 0.3.
- Augmentation: random crop, flip, colour, grey, erasing, and now the random piece in time.
- EMA: the tested model is a running average of the weights, which is steadier than any single step.
- Early stopping on validation **videos**, and the saved model is the best epoch, not the last.
- The test set is used once per run. Three seeds are run and the mean is reported.
- Every epoch the log prints `gap` (training accuracy minus validation accuracy). A gap that keeps growing while validation stops rising is overtraining; `history.csv` keeps all of it as proof.

## 8. What to expect

Nobody can promise a number before the run. My estimate for the mean of three seeds is **40 to
46%**. Reaching 47.14% would need the balanced loss to fix most of the Shoplifting errors.

- Each video is 0.71 points. If 8 of the 15 "Shoplifting called Robbery" videos flip, that alone is +5.7 points.
- Shooting will stay hard: 23 training videos, small figures, and the shot itself is barely visible.
- The balanced loss can cost a few RoadAccidents or Burglary videos.
- With seeds spread by 3.8 points, only a difference of several points in the mean is real.

How to read the result:

| Compare | It shows |
|---|---|
| V9 "V7 pooling" against V7's 38.33% | The effect of the training changes (1, 2, 4) |
| V9 pooling against V9 "V7 pooling" | The effect of the four time scales (3) |
| `st` against `guide` | The effect of the updated gnConv |
| `none` against `guide` | What gnConv adds |

## 9. Ideas we looked at and did not use

| Idea | Why not |
|---|---|
| VideoMamba-Middle with masked pretraining (82.0% on Kinetics-400 against 79.3% for our Small) | 74M parameters before gnConv, not the same backbone, and its pretraining uses a CLIP teacher |
| Teaching our model from CLIP features | The project decision is not to use the CLIP image encoder |
| 16 frames | Doubles the compute; no evidence yet that it helps on this data |
| Unfreezing more layers | Changes the model we describe; risky with 689 videos |
| Using how many test videos each class has | That is looking at the test set |
| An ensemble as the main number | 3 times the compute; it is printed, but only as an ensemble |

## 10. Results

Main run (`guide`, the guide's gnConv unchanged), finished 12 November 2026. Source:
`results_v9_all_runs.csv` and `logs/v9_guide_ucf101_seed*_live.log` on the HPC.

| Variant | Seeds | Test accuracy (V9 pooling) | Balanced | V7 pooling | Validation balanced | Ensemble of 3 |
|---|---|---|---|---|---|---|
| V7 (reference) | 42, 7, 123 | | 32.44 | 38.33 +- 3.8 (42.86 / 33.57 / 38.57) | | |
| V9 `guide` | 42, 7, 123 | **43.10 +- 1.47** (41.43 / 45.00 / 42.86) | 37.10 | 41.90 (42.14 / 41.43 / 42.14) | 53.71 | 43.57 |
| V9 `st` | not run yet | | | | | |
| V9 `none` | not run yet | | | | | |

Best epochs 13 / 10 / 11. About 3.3 hours per seed.

What the numbers show:

- **V9 is 4.8 points above V7 on average** (43.10 against 38.33), about 7 more videos right out of 140.
- **The seeds agree much more:** the spread fell from 3.8 to 1.47, and the worst V9 seed (41.43) is above V7's mean.
- **Where the gain comes from:** the same V9 models tested the V7 way reach 41.90, so the training changes (1, 2, 4) give about +3.6 points and the four time scales (3) about +1.2.
- **Balanced accuracy rose from 32.44 to 37.10,** so the rare crimes improved and not only the common ones.
- **Against the papers:** the mean is above Wu et al. (41.43) by 1.7 points and below Anomize (47.14) by 4 points. One run on 140 videos has a range of about +-8 points, so "above Wu et al." is a numerical statement, not a proven one. The settings also differ (CLIP and 6 trained crimes for them; all 13 crimes and UCA time stamps for us).
- **Is V9 better than V7 for sure?** With 3 seeds each the difference is about 1.7 standard errors. It is likely real but 3 seeds cannot prove it.

Per class, seed 123 (the only seed whose table was read so far):

| Class | V9 correct | Note |
|---|---|---|
| Shoplifting | 8 of 21 | V6 had 4 to 5; still 12 called Robbery (V6: 14 to 15) |
| Shooting | 0 of 23 | Not fixed: answers spread over ten other classes |
| Explosion | 14 of 21 | |
| RoadAccidents | 17 of 23 | |
| Arson | 7 of 9 | |
| Burglary | 7 of 13 | |
| Abuse, Arrest, Assault, Vandalism | 0 of 15 together | Small classes in the test |

Shooting stays the open problem: 23 of the 140 test videos, none right. Short pieces called
Background: 87.7%.

## 11. Sources (checked on 4, 7 and 11 November 2026)

| Used for | Source | What was checked |
|---|---|---|
| Change 1 | Menon et al., "Long-tail learning via logit adjustment", ICLR 2021, https://arxiv.org/abs/2007.07314 | Equation 10 (the loss), Table 3 (balanced error), the remark on re-weighting |
| Change 1 | Byrd and Lipton, "What is the Effect of Importance Weighting in Deep Learning?", https://arxiv.org/abs/1812.03372 | Abstract: the effect of weights fades over training |
| Change 1 | Kang et al., "Decoupling Representation and Classifier for Long-Tailed Recognition", ICLR 2020, https://arxiv.org/abs/1910.09217 | Ordinary sampling learns the best features |
| Change 2 | Wang et al., "Temporal Segment Networks", ECCV 2016, https://arxiv.org/abs/1608.00859 | Sparse sampling, one frame per segment |
| Change 3 | Li et al., "Anomize", CVPR 2025, https://arxiv.org/abs/2503.18094 | Top-k rule, 47.14%, base classes |
| Comparison | Wu et al., "Open-Vocabulary Video Anomaly Detection", CVPR 2024, https://arxiv.org/abs/2311.07042 | 41.43% (Table 7) |
| gnConv | Rao et al., "HorNet", NeurIPS 2022, https://arxiv.org/abs/2207.14284 and https://github.com/raoyongming/HorNet (`hornet.py`) | The original gnConv: 1x1 convs between orders, 7x7 depthwise |
| gnConv update | Howard et al., "MobileNets", https://arxiv.org/abs/1704.04861 | Depthwise separable convolutions |
| gnConv update | Tran et al., "A Closer Look at Spatiotemporal Convolutions", CVPR 2018, https://arxiv.org/abs/1711.11248 | Splitting space and time convolutions |
| Backbone | VideoMamba model zoo, https://github.com/OpenGVLab/VideoMamba/blob/main/videomamba/video_sm/MODEL_ZOO.md | Small 79.3% at 8 frames; Middle with masked pretraining 82.0% |
| Backbone | VideoMamba paper, `researchpapers/2403.06977v2.pdf`, tables on pages 9 to 11 | Middle has 74M parameters |
| Data | Yuan et al., UCA, CVPR 2024, https://arxiv.org/abs/2309.13925 | 23,542 timed sentences on UCF-Crime |
| Ensemble | Lakshminarayanan et al., NIPS 2017, https://arxiv.org/abs/1612.01474 | Averaging several models |
| Our numbers | `github_repo/v6_crime_moments/` (prep report, confusion matrices, histories) and the V7 result files on the cluster | Sections 2 and 3 |

## 12. Files

| File | Role |
|---|---|
| `train_v9.py` | The trainer (reuses `train_v6.py` for everything that did not change) |
| `gnconv_st_v9.py` | Updated gnConv |
| `prepare_val_whole_v9.py`, `run_prepare_v9.sh` | Validation videos, whole |
| `run_train_v9.sh`, `submit_v9.sh` | PBS jobs |
| `V9_RUN_STEPS.md` | Upload and PuTTY steps |
