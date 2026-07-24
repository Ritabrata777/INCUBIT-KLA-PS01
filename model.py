"""
model.py
NAFNet-SR: Nonlinear Activation Free Network for Super-Resolution.

Building blocks:
- SimpleGate: activation-free gating (channel-split + elementwise multiply).
- SimplifiedChannelAttention (SCA): cheap channel attention via global
  average pooling + a single 1x1 conv, with no activation function at all.
- LayerNorm2d: FP16-safe channel-wise normalization.
- NAFBlock: LayerNorm + depthwise-conv spatial mixing (with SCA) followed
  by a SimpleGate-based channel-mixing FFN, all activation-function-free.
- NAFNet_SR: NAFNet body operating on a low-resolution input, followed by
  a Sub-pixel Convolution (PixelShuffle) head that upsamples back to full
  resolution with a long bicubic residual skip.

Accepts 1-channel grayscale input, produces 1-channel restored grayscale output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGate(nn.Module):
    """Splits the channel dimension in half and multiplies the two halves
    together. Entirely removes the need for a nonlinear activation
    function (ReLU/GELU/etc.)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """Simplified Channel Attention (SCA): global average pool followed by
    a single 1x1 convolution -- no activation, no bottleneck MLP."""

    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.conv(self.pool(x))
        return x * attn


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm applied directly to NCHW tensors.
    Calculates statistics in FP32 for numerical stability under FP16 autocast."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_fp32 = x.float()
        mu = x_fp32.mean(dim=1, keepdim=True)
        var = x_fp32.var(dim=1, keepdim=True, unbiased=False)
        normalized = (x_fp32 - mu) / torch.sqrt(var + self.eps)
        out = normalized.to(dtype) * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out


class NAFBlock(nn.Module):
    """Nonlinear Activation Free Block."""

    def __init__(self, channels: int, expand_ratio: int = 2, ffn_expand_ratio: int = 2):
        super().__init__()
        dw_channels = channels * expand_ratio

        # --- Spatial mixing branch ---
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channels, kernel_size=1, bias=True)
        self.dwconv = nn.Conv2d(
            dw_channels, dw_channels, kernel_size=3, padding=1,
            groups=dw_channels, bias=True,
        )
        self.sg1 = SimpleGate()
        self.sca = SimplifiedChannelAttention(dw_channels // 2)
        self.conv2 = nn.Conv2d(dw_channels // 2, channels, kernel_size=1, bias=True)

        # --- Channel mixing (FFN) branch ---
        ffn_channels = channels * ffn_expand_ratio
        self.norm2 = LayerNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, ffn_channels, kernel_size=1, bias=True)
        self.sg2 = SimpleGate()
        self.conv4 = nn.Conv2d(ffn_channels // 2, channels, kernel_size=1, bias=True)

        # Learnable residual scaling factors
        self.beta = nn.Parameter(torch.ones((1, channels, 1, 1)))
        self.gamma = nn.Parameter(torch.ones((1, channels, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial mixing
        residual = x
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.dwconv(y)
        y = self.sg1(y)
        y = self.sca(y)
        y = self.conv2(y)
        x = residual + y * self.beta

        # Channel mixing
        residual = x
        y = self.norm2(x)
        y = self.conv3(y)
        y = self.sg2(y)
        y = self.conv4(y)
        return residual + y * self.gamma


class NAFNet_SR(nn.Module):
    """NAFNet-SR architecture with PixelShuffle upsampling head."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 64,          # Updated default width for optimal capacity
        num_blocks: int = 8,
        upscale_factor: int = 2,  # Updated default to 2x matching KLA dataset
    ):
        super().__init__()
        self.upscale_factor = upscale_factor

        self.intro = nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=True)
        self.body = nn.Sequential(*[NAFBlock(width) for _ in range(num_blocks)])
        self.body_tail_conv = nn.Conv2d(width, width, kernel_size=3, padding=1, bias=True)

        # Sub-pixel convolution (PixelShuffle) upsampling head
        up_channels = out_channels * (upscale_factor ** 2)
        self.upsample = nn.Sequential(
            nn.Conv2d(width, up_channels, kernel_size=3, padding=1, bias=True),
            nn.PixelShuffle(upscale_factor),
        )
        self.final_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.intro(x)
        body_feat = self.body(feat)
        body_feat = self.body_tail_conv(body_feat)
        feat = feat + body_feat

        out = self.upsample(feat)
        out = self.final_conv(out)

        # Long residual skip: add bicubic upsampled base input
        base = F.interpolate(
            x, scale_factor=self.upscale_factor, mode="bicubic", align_corners=False
        )
        return out + base
