# VideoMamba + HorNet gnConv for Crime-Type Recognition

This model watches a short surveillance clip and names the crime in it: one of 13 crime types from UCF-Crime, or normal activity. It is VideoMamba-Small with HorNet's recursive gated convolution (gnConv2D) added beside the 1D convolution in its last eight layers.

It is the perception part of NavPulse, a first-response robot for crime scenes. The robot's exploration and SLAM code is in [Autonomous-Quadruped-SLAM-Navigation-Perception](https://github.com/yokshith09/Autonomous-Quadruped-SLAM-Navigation-Perception).

## Model

- **Input.** 8 frames of 224 x 224 pixels.
- **Patch embedding.** A 3D convolution (kernel 1 x 16 x 16, stride 16) gives 8 x 14 x 14 = 1,568 tokens of size 384. A class (CLS) token and position embeddings are added, giving 1,569 tokens.
- **Backbone.** 24 bidirectional Mamba layers. Layers 0 to 15 are plain VideoMamba layers and stay frozen. Layers 16 to 23 are hybrid layers and are trained.
- **Head.** The CLS token of the last layer goes through a final norm, dropout 0.5 and a linear layer. The patch tokens are not pooled.

### The hybrid layer

Inside each trained Mamba layer, the input is projected into a stream `x` and a gate `z` (768 channels each). For the forward scan and for the backward scan:

```
u = Conv1d(x) + gnConv2D(x)
```

`u` then goes through SiLU and the selective scan, gated by `z`. The backward result is flipped back, added to the forward result and projected from 768 to 384 channels.

gnConv2D runs on each frame's 14 x 14 patch grid and skips the CLS token:

1. A GhostModule (1 x 1 conv plus a cheap 3 x 3 depthwise conv) maps 768 channels to 1,536, split into `a` (48) and `b` (1,488).
2. A 7 x 7 depthwise conv on `b` gives five gates of 48, 96, 192, 384 and 768 channels.
3. Recursive gating of order 5: `p0 = a * q0`, then `pk = gk(p(k-1)) * qk`, where `g1` to `g4` are convolutions with kernel sizes 3, 5, 7 and 1 that double the channels.
4. A 1 x 1 output conv, initialized to zero, so the module adds nothing at the start and cannot disturb the pretrained backbone.

### Two fixes and a smaller model

The first version had two bugs in how data reached gnConv2D:

- **Frame reshape.** Tokens of shape `(B, C, T*H*W)` were reshaped straight to `(B*T, C, H, W)`, so each "frame" mixed channels from all 8 frames. They are now reshaped to `(B, C, T, H, W)` before time is moved into the batch. A self-check before training perturbs frame 3 and confirms only frame 3's output changes.
- **CLS position.** In the backward scan the CLS token is last, but the code always skipped the first token. It now skips index 0 going forward and the last index going backward.

In the first version every layer had gnConv2D. In the frozen layers 0 to 15 these modules could only output zero, so they were removed. The predictions stay the same, and the model shrinks from 297.9M to 116.2M parameters.

| Part | Parameters | Trained |
|---|---:|---:|
| Patch embedding | 374,400 | 0 |
| Mamba layers (24) | 25,030,656 | 8,343,552 |
| gnConv2D (16 modules) | 90,817,536 | 90,817,536 |
| Norm layers | 9,600 | 3,456 |
| Classifier | 5,390 | 5,390 |
| **Total** | **116,237,582** | **99,169,934** |

Compute is 185.0 GMACs (370.1 GFLOPs) per clip. Inference takes 59 ms per clip on a Tesla V100 (batch 1, float32).

## Training

- **Transfer chain.** VideoMamba-Small pretrained on Kinetics-400, then fine-tuned on UCF-101 (94.6%), then trained on the crime data. gnConv2D starts from zero.
- **Optimizer.** AdamW with weight decay 0.05. Learning rates: backbone 1e-5, gnConv2D 1e-4, classifier 1e-3. 3 warm-up epochs, then cosine decay.
- **Batch.** 2 clips with 8 accumulation steps. Up to 40 epochs, stopping after 10 without improvement.
- **Regularization.** Dropout 0.5, stochastic depth 0.2, label smoothing 0.1, mixup 0.3, weight EMA 0.995, class-weighted loss.
- **Clip sampling.** 8 frames per clip, one random frame from each eighth of the video. Augmentation is the same for all 8 frames.
- **Model selection.** 6 videos per class are held out from training for validation and pick the epoch. The test set is used once, at the end.
- **Precision.** float32, because order-5 gating overflows in float16.
- **Testing.** 8 clips x 3 crops per video, with the softmax scores averaged.

## Results

UCF-Crime official recognition split: 4 folds, 14 classes, 532 training and 168 test videos per fold.

| Fold | Test accuracy (%) | Macro-F1 (%) |
|---|---:|---:|
| 1 | 41.07 | 38.77 |
| 2 | 29.76 | 27.87 |
| 3 | 36.90 | 35.75 |
| 4 | 40.48 | 38.45 |
| **Mean** | **37.05** | **35.21** |

Published results on the same split: C3D 23.0%, TCNN 28.4% (Sultani et al., CVPR 2018) and TCNN with motion-aware features 31.0% (Zhu and Newsam, BMVC 2019).

- **Crime window only.** On the 119 test videos with temporal annotations, scoring frames from inside the crime window raises accuracy from 36.97% to 42.86%.
- **Our own 13-class violence dataset.** 73.47% test accuracy (108 of 147). Excluding 15 clips that come from the same recordings as training clips, it is 70.45% (93 of 132).

## Code

| Path | What it is |
|---|---|
| `v5_ucf_crime/mamba_simple_v7.py` | Mamba layer with gnConv2D (the model change) |
| `v5_ucf_crime/prepare_data_v5.py`, `train_v5.py` | Frame cache and splits, then training with validation and one final test |
| `v5_ucf_crime/run_*.sh`, `V5_RUN_STEPS.md` | PBS job scripts and the exact run steps |
| `v5_ucf_crime/runs/` | Logs and results of the four folds |
| `v4_crime_fixed/` | The same model on our own dataset |
| `environment/` | Conda environment and pip freeze |

You will need:

- Python 3.10, PyTorch 2.1.1 with CUDA 11.8, and a CUDA GPU.
- `mamba_ssm` 1.1.1 and `causal_conv1d` built from [hustvl/Vim](https://github.com/hustvl/Vim).
- The [VideoMamba](https://github.com/OpenGVLab/VideoMamba) repository, which provides the model definition.
- UCF-Crime from its [dataset page](https://www.crcv.ucf.edu/projects/real-world/).

Paths at the top of each script point to our HPC account, so change them for your machine. Checkpoints and frame caches are not included.
