"""NAFNet-SR: Nonlinear Activation Free Network for image restoration + SR.

Grayscale (1-channel) in, grayscale (1-channel) out. Final PixelShuffle upsample
lets the same model function as a super-resolution head; when `upscale=1` the
network acts purely as a restoration model at the input resolution.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for (N, C, H, W) tensors."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    """Splits the channel dim in half and returns their elementwise product."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b


class SimplifiedChannelAttention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.pool(x))
        return x * w


class NAFBlock(nn.Module):
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2, drop_path: float = 0.0):
        super().__init__()
        dw_c = c * dw_expand

        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw_c, 1, 1, 0)
        self.conv2 = nn.Conv2d(dw_c, dw_c, 3, 1, 1, groups=dw_c)  # depthwise
        self.sg1 = SimpleGate()
        self.sca = SimplifiedChannelAttention(dw_c // 2)
        self.conv3 = nn.Conv2d(dw_c // 2, c, 1, 1, 0)

        self.norm2 = LayerNorm2d(c)
        ffn_c = c * ffn_expand
        self.conv4 = nn.Conv2d(c, ffn_c, 1, 1, 0)
        self.sg2 = SimpleGate()
        self.conv5 = nn.Conv2d(ffn_c // 2, c, 1, 1, 0)

        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg1(y)
        y = self.sca(y)
        y = self.conv3(y)
        x = x + y * self.beta

        y = self.norm2(x)
        y = self.conv4(y)
        y = self.sg2(y)
        y = self.conv5(y)
        return x + y * self.gamma


class NAFNet_SR(nn.Module):
    """Encoder-decoder NAFNet with a PixelShuffle super-resolution tail.

    Args:
        in_channels: input channels (1 for grayscale).
        out_channels: output channels (1 for grayscale).
        width: base feature width.
        enc_blocks: NAFBlocks per encoder stage.
        middle_blocks: NAFBlocks at the bottleneck.
        dec_blocks: NAFBlocks per decoder stage.
        upscale: final PixelShuffle spatial scale factor (1, 2, or 4).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 32,
        enc_blocks=(2, 2, 4),
        middle_blocks: int = 8,
        dec_blocks=(2, 2, 2),
        upscale: int = 1,
    ):
        super().__init__()
        assert upscale in (1, 2, 4), "upscale must be 1, 2, or 4"
        self.upscale = upscale

        self.intro = nn.Conv2d(in_channels, width, 3, 1, 1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for n in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(n)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, 2))
            chan *= 2

        self.middle = nn.Sequential(*[NAFBlock(chan) for _ in range(middle_blocks)])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for n in dec_blocks:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2),
                )
            )
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(n)]))

        # Super-resolution tail (PixelShuffle-based)
        if upscale > 1:
            tail = []
            tc = chan
            up = upscale
            while up > 1:
                tail += [nn.Conv2d(tc, tc * 4, 3, 1, 1), nn.PixelShuffle(2)]
                up //= 2
            tail += [nn.Conv2d(tc, out_channels, 3, 1, 1)]
            self.tail = nn.Sequential(*tail)
        else:
            self.tail = nn.Conv2d(chan, out_channels, 3, 1, 1)

        self.padder_size = 2 ** len(enc_blocks)

    def _pad(self, x: torch.Tensor):
        _, _, h, w = x.shape
        ph = (self.padder_size - h % self.padder_size) % self.padder_size
        pw = (self.padder_size - w % self.padder_size) % self.padder_size
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        return x, h, w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp = x
        x, h, w = self._pad(x)

        feat = self.intro(x)
        skips = []
        for enc, down in zip(self.encoders, self.downs):
            feat = enc(feat)
            skips.append(feat)
            feat = down(feat)

        feat = self.middle(feat)

        for up, dec, skip in zip(self.ups, self.decoders, reversed(skips)):
            feat = up(feat)
            feat = feat + skip
            feat = dec(feat)

        out = self.tail(feat)

        # Global residual: upsample input to match output scale
        if self.upscale > 1:
            inp_up = F.interpolate(
                inp, scale_factor=self.upscale, mode="bilinear", align_corners=False
            )
            # Crop tail output to original padded region * upscale
            out = out[:, :, : h * self.upscale, : w * self.upscale]
            out = out + inp_up
        else:
            out = out[:, :, :h, :w] + inp
        return out
