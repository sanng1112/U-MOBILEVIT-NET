import argparse
from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn

from cv_nets.layers.base_layer import BaseLayer
from cv_nets.layers.conv_layer import Conv2d


class FourierAttention2D(BaseLayer):
    """
    Global Fourier-domain attention for 2D feature maps.

    Pipeline:
        x : [B, C, H, W]                          (real)
        h = GroupNorm(x)                          (pre-norm, real)
        X = rfft2(h)        -> [B, C, H, W//2+1]  (complex)
        X = X * F           per-(channel, freq) complex filter
        X = mix(X)          block-diagonal complex channel mix
        y = irfft2(X)       -> [B, C, H, W]       (real)
        y = out_proj(y)     1x1 conv (real)
        return y            (NO internal residual; the caller fuses)

    Why these two complex operations:
      * F gives every (channel, frequency) coordinate its own learnable
        complex weight. Multiplication in the frequency domain is
        convolution in the spatial domain, so a single weight tensor of
        size [C, H, W//2+1] is equivalent to a per-channel global
        convolution with an H x W kernel. Costs ~2*C*H*(W//2+1)
        parameters - tiny compared to a dense H*W kernel per channel.
      * The block-diagonal complex matrix mixes channels at every
        frequency (AFNO-style). It is the spectral analogue of a 1x1
        conv: each frequency component sees a learned linear combination
        of channels. We block-diagonalise it (num_heads heads) so the
        parameter count stays O(C^2 / num_heads).
    """

    def __init__(
        self,
        opts,
        embed_dim: int,
        spatial_size: Tuple[int, int],
        num_heads: int = 4,
        use_channel_mix: bool = True,
        dropout: float = 0.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        assert isinstance(spatial_size, (tuple, list)) and len(spatial_size) == 2, (
            "spatial_size must be a (H, W) tuple matching the feature map at this stage"
        )
        H, W = int(spatial_size[0]), int(spatial_size[1])
        W_freq = W // 2 + 1

        assert embed_dim % num_heads == 0, (
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.H = H
        self.W = W
        self.W_freq = W_freq
        self.use_channel_mix = use_channel_mix

        scale = 0.02

        # Per-(channel, frequency) complex modulation, init near identity (real=1, imag=0).
        self.filter_real = nn.Parameter(
            torch.ones(1, embed_dim, H, W_freq)
            + torch.randn(1, embed_dim, H, W_freq) * scale
        )
        self.filter_imag = nn.Parameter(
            torch.randn(1, embed_dim, H, W_freq) * scale
        )

        # Block-diagonal complex channel mixer, init near identity per head.
        if use_channel_mix:
            eye = torch.eye(self.head_dim).unsqueeze(0).repeat(num_heads, 1, 1)
            self.mix_real = nn.Parameter(
                eye + torch.randn(num_heads, self.head_dim, self.head_dim) * scale
            )
            self.mix_imag = nn.Parameter(
                torch.randn(num_heads, self.head_dim, self.head_dim) * scale
            )
        else:
            self.register_parameter("mix_real", None)
            self.register_parameter("mix_imag", None)

        # Pre-norm in spatial domain. GroupNorm is quantization-friendlier than
        # LayerNorm-2d and works at any batch size.
        num_groups = min(8, embed_dim)
        while embed_dim % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.pre_norm = nn.GroupNorm(num_groups=num_groups, num_channels=embed_dim)

        # Real-valued output projection.
        self.out_proj = Conv2d(
            opts=opts,
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def _complex_filter(self) -> Tensor:
        return torch.complex(self.filter_real, self.filter_imag)

    def _complex_mix(self) -> Tensor:
        return torch.complex(self.mix_real, self.mix_imag)

    def forward(self, x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        if H != self.H or W != self.W:
            raise ValueError(
                f"FourierAttention2D was built for spatial size "
                f"({self.H}, {self.W}) but received ({H}, {W}). "
                "Rebuild the module with the correct spatial_size or "
                "interpolate the input."
            )

        h = self.pre_norm(x)

        # rfft2 returns a complex tensor of shape [B, C, H, W//2+1].
        spec = torch.fft.rfft2(h, dim=(-2, -1), norm="ortho")

        # Per-(channel, freq) complex modulation.
        spec = spec * self._complex_filter()

        # Multi-head, block-diagonal complex channel mixing.
        if self.use_channel_mix:
            spec = spec.view(B, self.num_heads, self.head_dim, H, self.W_freq)
            mix = self._complex_mix()  # [num_heads, head_dim, head_dim]
            spec = torch.einsum("nij,bnjhw->bnihw", mix, spec)
            spec = spec.reshape(B, C, H, self.W_freq)

        out = torch.fft.irfft2(spec, s=(H, W), dim=(-2, -1), norm="ortho")
        out = self.out_proj(out)
        out = self.dropout(out)
        return out

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group("FourierAttention2D Arguments")
        group.add_argument(
            "--fourier-num-heads", type=int, default=4,
            help="Number of heads for block-diagonal complex channel mixing.",
        )
        group.add_argument(
            "--fourier-use-channel-mix", action="store_true",
            help="Enable the complex channel-mixing matrix.",
        )
        group.add_argument(
            "--fourier-dropout", type=float, default=0.0,
            help="Dropout on the Fourier branch output.",
        )
        return parser

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(embed_dim={self.embed_dim}, "
            f"spatial=({self.H}x{self.W}), spectrum=({self.H}x{self.W_freq}), "
            f"num_heads={self.num_heads}, use_channel_mix={self.use_channel_mix})"
        )


if __name__ == "__main__":
    import argparse as _argparse

    opts = _argparse.Namespace()
    B, C, H, W = 2, 64, 20, 20

    m = FourierAttention2D(
        opts=opts, embed_dim=C, spatial_size=(H, W),
        num_heads=4, use_channel_mix=True, dropout=0.0,
    )
    x = torch.randn(B, C, H, W)
    y = m(x)
    print(m)
    print("input :", tuple(x.shape))
    print("output:", tuple(y.shape))
    n_params = sum(p.numel() for p in m.parameters())
    print(f"params: {n_params:,}")
    loss = y.mean()
    loss.backward()
    print("backward OK, grad-norm:",
          sum((p.grad.norm().item() ** 2 for p in m.parameters() if p.grad is not None)) ** 0.5)
