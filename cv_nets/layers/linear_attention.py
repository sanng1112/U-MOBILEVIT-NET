from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from cv_nets.layers.base_layer import BaseLayer
from cv_nets.layers.conv_layer import Conv2d
from cv_nets.layers.dropout import Dropout


class LinearSelfAttention(BaseLayer):
    def __init__(
        self,
        opts,
        embed_dim: int,
        attn_dropout: Optional[float] = 0.0,
        bias: Optional[bool] = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        self.qkv_proj = Conv2d(
            opts=opts,
            in_channels=embed_dim,
            out_channels=1 + (2 * embed_dim),
            bias=bias,
            kernel_size=1,
            padding = 0,
            use_norm=False,
            use_act=False,
        )

        self.attn_dropout = Dropout(p=attn_dropout)
        self.out_proj = Conv2d(
            opts=opts,
            in_channels=embed_dim,
            out_channels=embed_dim,
            bias=bias,
            kernel_size=1,
            padding = 0,
            use_norm=False,
            use_act=False,
        )
        self.embed_dim = embed_dim

    def __repr__(self):
        return "{}(embed_dim={}, attn_dropout={})".format(
            self.__class__.__name__, self.embed_dim, self.attn_dropout.p
        )

    @staticmethod
    def visualize_context_scores(context_scores):
        batch_size, channels, num_pixels, num_patches = context_scores.shape

        assert batch_size == 1, "For visualization purposes, use batch size of 1"
        assert (
            channels == 1
        ), "The inner-product between input and latent node (query) is a scalar"

        up_scale_factor = int(num_pixels**0.5)
        patch_h = patch_w = int(context_scores.shape[-1] ** 0.5)
        # [1, 1, P, N] --> [1, P, h, w]
        context_scores = context_scores.reshape(1, num_pixels, patch_h, patch_w)
        context_map = F.pixel_shuffle(context_scores, upscale_factor=up_scale_factor)
        context_map = context_map.squeeze()

        min_val = torch.min(context_map)
        max_val = torch.max(context_map)
        context_map = (context_map - min_val) / (max_val - min_val)

        try:
            import os
            from glob import glob

            import cv2
            context_map = (context_map * 255).byte().cpu().numpy()
            context_map = cv2.resize(
                context_map, (80, 80), interpolation=cv2.INTER_NEAREST
            )

            colored_context_map = cv2.applyColorMap(context_map, cv2.COLORMAP_JET)
            res_dir_name = "attn_res"
            if not os.path.isdir(res_dir_name):
                os.makedirs(res_dir_name)
            f_name = "{}/h_{}_w_{}_index_".format(res_dir_name, patch_h, patch_w)

            files_cmap = glob(
                "{}/h_{}_w_{}_index_*.png".format(res_dir_name, patch_h, patch_w)
            )
            idx = len(files_cmap)
            f_name += str(idx)

            cv2.imwrite("{}.png".format(f_name), colored_context_map)
            return colored_context_map
        except ModuleNotFoundError as mnfe:
            print("Please install OpenCV to visualize context maps")
            return context_map

    def _forward_self_attn(self, x: Tensor, *args, **kwargs) -> Tensor:
        qkv = self.qkv_proj(x)

        query, key, value = torch.split(
            qkv, split_size_or_sections=[1, self.embed_dim, self.embed_dim], dim=1
        )

        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)
        context_vector = key * context_scores
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out

    def _forward_cross_attn(
        self, x: Tensor, x_prev: Optional[Tensor] = None, *args, **kwargs
    ) -> Tensor:

        batch_size, in_dim, kv_patch_area, kv_num_patches = x.shape

        q_patch_area, q_num_patches = x.shape[-2:]

        assert (
            kv_patch_area == q_patch_area
        ), "The number of pixels in a patch for query and key_value should be the same"

        qk = F.conv2d(
            x_prev,
            weight=self.qkv_proj.weight[: self.embed_dim + 1, ...],
            bias=self.qkv_proj.bias[: self.embed_dim + 1, ...],
        )
        query, key = torch.split(qk, split_size_or_sections=[1, self.embed_dim], dim=1)
        value = F.conv2d(
            x,
            weight=self.qkv_proj.weight[self.embed_dim + 1 :, ...],
            bias=self.qkv_proj.bias[self.embed_dim + 1 :, ...],
        )
        
        context_scores = F.softmax(query, dim=-1)
        context_scores = self.attn_dropout(context_scores)
        context_vector = key * context_scores
        context_vector = torch.sum(context_vector, dim=-1, keepdim=True)
        out = F.relu(value) * context_vector.expand_as(value)
        out = self.out_proj(out)
        return out

    def forward(
        self, x: Tensor, x_prev: Optional[Tensor] = None, *args, **kwargs
    ) -> Tensor:
        if x_prev is None:
            return self._forward_self_attn(x, *args, **kwargs)
        else:
            return self._forward_cross_attn(x, x_prev=x_prev, *args, **kwargs)

if __name__ == "__main__":
    import argparse
    opts = argparse.Namespace()

    # 2. Thiết lập các thông số kích thước (Shape parameters)
    B = 2       # Batch size
    d = 64      # Embed dimension (Channels)
    P = 16      # Số pixels trong một patch (ví dụ patch 4x4 = 16)
    N = 49      # Số lượng patches của đầu vào hiện tại (x)
    M = 25      # Số lượng patches của đầu vào trước đó (x_prev)

    # 3. Khởi tạo model
    print("Khởi tạo module LinearSelfAttention...")
    try:
        model = LinearSelfAttention(opts=opts, embed_dim=d)
        print(model)
    except Exception as e:
        print(f"Lỗi khi khởi tạo model: {e}")
        exit(1)

    x = torch.randn(B, d, P, N)
    x_prev = torch.randn(B, d, P, M)

    print("\n" + "="*30)
    print("--- Testing Self-Attention ---")
    try:
        out_self = model(x)
        print(f"Input x shape:  {x.shape}")
        print(f"Output shape:   {out_self.shape}")
        
        assert out_self.shape == x.shape, "Lỗi: Output shape khác Input shape!"
        print("[SUCCESS] Self-Attention chạy thành công!")
    except Exception as e:
        print(f"[FAILED] Lỗi khi chạy Self-Attention:\n{e}")
    print("\n" + "="*30)
    print("--- Testing Cross-Attention ---")
    try:
        out_cross = model(x, x_prev=x_prev)
        print(f"Input x shape:       {x.shape}")
        print(f"Input x_prev shape:  {x_prev.shape}")
        print(f"Output shape:        {out_cross.shape}")
        assert out_cross.shape == x.shape, "Lỗi: Output shape khác Input shape của x!"
        print("[SUCCESS] Cross-Attention chạy thành công!")
    except Exception as e:
        print(f"[FAILED] Lỗi khi chạy Cross-Attention:\n{e}")