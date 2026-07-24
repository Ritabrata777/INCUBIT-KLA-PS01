"""
model.py
NAFNet-SR: Nonlinear Activation Free Network for Super-Resolution.

Building blocks:
- SimpleGate: activation-free gating (channel-split + elementwise multiply).
- SimplifiedChannelAttention (SCA): cheap channel attention via global
  average pooling + a single 1x1 conv, with no activation function at all.
- NAFBlock: LayerNorm + depthwise-conv spatial mixing (with SCA) followed
  by a SimpleGate-based channel-mixing FFN, all activation-function-free.
- NAFNet_SR: NAFNet body operating on a low-resolution input, followed by
  a Sub-pixel Convolution (PixelShuffle) head that upsamples back to full
  (clean) resolution.

Accepts 1-channel grayscale input, produces 1-channel restored grayscale
output.
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
    a single 1x1 convolution -- no activation, no bottleneck MLP. Far
    cheaper than SE-style attention while retaining most of the benefit."""

    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.conv(self.pool(x))
        return x * attn


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm applied directly to NCHW tensors, as used
    throughout NAFNet (normalizes across the channel dimension at every
    spatial location)."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class NAFBlock(nn.Module):
    """
    Nonlinear Activation Free Block.

    Spatial mixing branch:
        LayerNorm -> 1x1 conv (expand) -> depthwise 3x3 conv -> SimpleGate
        -> Simplified Channel Attention -> 1x1 conv (project) -> residual

    Channel mixing (FFN) branch:
        LayerNorm -> 1x1 conv (expand) -> SimpleGate -> 1x1 conv (project)
        -> residual

    No ReLU/GELU/etc. appears anywhere in this block.
    """

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

        # Learnable residual scaling factors (as in the original NAFNet).
        self.beta = nn.Parameter(torch.ones((1, channels, 1, 1)))
        self.gamma = nn.Parameter(torch.ones((1, channels, 1, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial mixing.
        residual = x
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.dwconv(y)
        y = self.sg1(y)
        y = self.sca(y)
        y = self.conv2(y)
        x = residual + y * self.beta

        # Channel mixing.
        residual = x
        y = self.norm2(x)
        y = self.conv3(y)
        y = self.sg2(y)
        y = self.conv4(y)
        return residual + y * self.gamma


class NAFNet_SR(nn.Module):
    """
    NAFNet-SR: a NAFNet body operating at low resolution followed by a
    Sub-pixel Convolution (PixelShuffle) upsampling head that restores
    the image to full (clean) resolution.

    Args:
        in_channels: number of input channels (1 for grayscale).
        out_channels: number of output channels (1 for grayscale).
        width: base feature width of the NAFNet body.
        num_blocks: number of stacked NAFBlocks in the body.
        upscale_factor: spatial upscaling ratio applied by the
            PixelShuffle head (must match the downsampling factor used by
            the degradation pipeline / dataset during training).
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 32,
        num_blocks: int = 8,
        upscale_factor: int = 4,
    ):
        super().__init__()
        self.upscale_factor = upscale_factor

        self.intro = nn.Conv2d(in_channels, width, kernel_size=3, padding=1, bias=True)
        self.body = nn.Sequential(*[NAFBlock(width) for _ in range(num_blocks)])
        self.body_tail_conv = nn.Conv2d(width, width, kernel_size=3, padding=1, bias=True)

        # Sub-pixel convolution (PixelShuffle) upsampling head.
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

        # Long residual skip: the network only needs to learn a residual
        # correction on top of a simple bicubic upsample of the (noisy,
        # low-res) input, rather than reconstructing the image from scratch.
        base = F.interpolate(
            x, scale_factor=self.upscale_factor, mode="bicubic", align_corners=False
        )
        return out + base
