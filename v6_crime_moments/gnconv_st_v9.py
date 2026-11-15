"""
V9: updated HorNet gnConv for video ("gnConv-ST"), used with  --gnconv st  in train_v9.py.

It is the guide's gnConv2D class from mamba_simple_v7.py (order 5, GhostModule input, kernels
3 / 5 / 7 / 1, zero-initialised output) with two parts swapped. Everything else, including the
forward pass and the two V4 fixes (frame reshape, cls token), is inherited unchanged.

  1. Separable multi-scale convolutions. The guide's 3x3, 5x5 and 7x7 convolutions between the
     orders are full convolutions: 4.12M of the 5.68M parameters of one module. Here each one
     is a depthwise k x k convolution followed by a 1x1 convolution (as in MobileNet). Same
     kernel sizes, same channel steps 48 -> 96 -> 192 -> 384 -> 768.
     One module: 5,676,096 -> 1,674,960 parameters. Whole model (by hand, the job measures it):
     116.2M -> about 52.2M parameters, 185 -> about 85 GMACs per clip.
     (The original HorNet uses only 1x1 convolutions here, so this sits between the two.)
  2. Gates that see the neighbouring frames. gnConv2D looks at one frame at a time. After
     HorNet's 7x7 depthwise convolution we add a depthwise convolution over 3 frames in time
     (the "(2+1)D" idea of Tran et al., CVPR 2018). It starts as "this frame only", so at the
     start of training the module behaves exactly like the per-frame version.

Run  python gnconv_st_v9.py  for the parameter count (CPU, a few seconds).
"""

from functools import partial

import torch
import torch.nn as nn

import mamba_simple_v7 as ms7          # the guide's gnConv2D lives here, unchanged


class SpaceTimeDW(nn.Module):
    """HorNet's 7x7 depthwise conv on each frame, then a depthwise conv over 3 frames in time."""

    def __init__(self, dim, frames, h=14, w=14):
        super().__init__()
        self.frames = frames
        self.space = ms7.get_dwconv(dim, 7, True)
        self.time = nn.Conv3d(dim, dim, kernel_size=(3, 1, 1), padding=(1, 0, 0), groups=dim, bias=False)
        with torch.no_grad():                      # start as identity: this frame only
            self.time.weight.zero_()
            self.time.weight[:, 0, 1, 0, 0] = 1.0

    def forward(self, x):                          # (B*T, C, H, W); the T frames of a clip are consecutive
        x = self.space(x)
        bt, c, h, w = x.shape
        if bt % self.frames:
            raise ValueError(f'SpaceTimeDW: {bt} frames is not a multiple of {self.frames} frames per clip')
        x = x.reshape(bt // self.frames, self.frames, c, h, w).transpose(1, 2)      # (B, C, T, H, W)
        return self.time(x).transpose(1, 2).reshape(bt, c, h, w)


def separable(cin, cout, k):
    """Depthwise k x k (looks around) then 1x1 (mixes channels). k = 1 is a plain 1x1 conv."""
    if k == 1:
        return nn.Conv2d(cin, cout, 1)
    return nn.Sequential(nn.Conv2d(cin, cin, k, padding=k // 2, groups=cin), nn.Conv2d(cin, cout, 1))


class gnConvST(ms7.gnConv):
    KERNELS = (3, 5, 7, 1)                         # the guide's multi-scale kernels

    def __init__(self, dim, frames, order=5, h=14, w=14, s=1.0, bias=True):
        super().__init__(dim, order=order, gflayer=partial(SpaceTimeDW, frames=frames), h=h, w=w, s=s, bias=bias)
        self.pws = nn.ModuleList([separable(self.dims[i], self.dims[i + 1], self.KERNELS[i % 4])
                                  for i in range(order - 1)])


@torch.no_grad()
def self_test(device, frames=8):
    """Checks the frame handling. Returns (start_ok, time_ok, cls_ok, text)."""
    g = gnConvST(768, frames).to(device).eval()
    nn.init.normal_(g.proj_out.weight, std=0.02)   # non-zero output for the test only
    hw = 14 * 14
    x = torch.randn(1, 768, 1 + frames * hw, device=device)
    x2 = x.clone()
    x2[:, :, 1 + 3 * hw:1 + 4 * hw] += 1.0          # change frame 3 only

    def change_per_frame():
        d = (g(x2, cls_first=True) - g(x, cls_first=True)).abs().amax(dim=1)[0]
        return [d[1 + t * hw:1 + (t + 1) * hw].max().item() for t in range(frames)]

    a = change_per_frame()                          # time conv = identity: only frame 3 may change
    start_ok = a[3] > 1e-4 and max(v for t, v in enumerate(a) if t != 3) <= 1e-6 + 1e-4 * a[3]
    g.dwconv.time.weight[:, 0, :, 0, 0] = torch.tensor([0.25, 0.5, 0.25], device=device)
    b = change_per_frame()                          # now frames 2, 3 and 4 change, no other frame
    time_ok = (min(b[2], b[3], b[4]) > 1e-5
               and max(v for t, v in enumerate(b) if t not in (2, 3, 4)) <= 1e-6 + 1e-4 * b[3])
    cls_ok = (g(x, cls_first=True)[:, :, 0].abs().max().item() == 0.0
              and g(x.flip([-1]), cls_first=False)[:, :, -1].abs().max().item() == 0.0)
    text = (f'at start only frame 3 changes: {"OK" if start_ok else "FAILED"} | with a time kernel frames '
            f'2-4 change: {"OK" if time_ok else "FAILED"} | cls token untouched: {"OK" if cls_ok else "FAILED"}')
    return start_ok, time_ok, cls_ok, text


if __name__ == '__main__':
    old, new = ms7.gnConv(768), gnConvST(768, 8)
    n_old, n_new = (sum(p.numel() for p in m.parameters()) for m in (old, new))
    print(f'one gnConv module: guide {n_old:,} parameters -> gnConv-ST {n_new:,} ({n_old / n_new:.1f}x fewer)')
    print(f'16 modules (layers 16-23, both scan directions): {16 * n_old:,} -> {16 * n_new:,}')
    print('self-test:', self_test(torch.device('cpu'))[3])
