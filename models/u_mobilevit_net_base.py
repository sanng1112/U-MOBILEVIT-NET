import argparse
import torch
from torch import nn, Tensor, optim
from typing import Optional, Union, Tuple, List, Any
import torch.optim as optim
from torch.ao.quantization import QuantStub, DeQuantStub
from torch.nn.quantized import FloatFunctional


import os
import json
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch.nn.functional as F


from cv_nets.utils.config_helper import get_param
from cv_nets.layers.activation import build_activation_layer
from cv_nets.layers.normalization import build_normalization_layer
from cv_nets.layers.pooling import build_pooling_layer
from cv_nets.layers import *

class MV2Block(nn.Module):
    """
    Inverted Residual Block của MobileNetV2.
    Dùng depthwise separable conv để giảm tham số và FLOPs.
    """
    def __init__(
        self,
        opts: Any,
        in_channels: int,
        out_channels: int,
        expand_ratio: int = 2
    ) -> None:
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = (in_channels == out_channels)
        self.f_add = FloatFunctional()

        layers = []

        # 1x1 expansion
        if expand_ratio != 1:
            layers.extend([
                Conv2d(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    opts=opts
                ),
                build_normalization_layer(opts, num_features=hidden_dim),
                build_activation_layer(opts),
            ])
        else:
            hidden_dim = in_channels

        # 3x3 depthwise
        layers.extend([
            Conv2d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=hidden_dim,
                opts=opts
            ),
            build_normalization_layer(opts, num_features=hidden_dim),
            build_activation_layer(opts),
        ])

        # 1x1 projection
        layers.extend([
            Conv2d(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                opts=opts
            ),
            build_normalization_layer(opts, num_features=out_channels),
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out = self.block(x)
        if self.use_res_connect:
            return self.f_add.add(x, out)
        return out


class Attention2DWrapper(nn.Module):
    """
    Bọc LinearSelfAttention để dùng trực tiếp với tensor [B, C, H, W].

    Quy ước:
    - Input:  [B, C, H, W]
    - Unfold: [B, C, P, N]
      trong đó P = patch_size * patch_size
               N = số patch
    - Output: [B, C, H, W]
    """
    def __init__(self, opts: Any, embed_dim: int, patch_size: int = 2) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.P = patch_size * patch_size
        self.attn = LinearSelfAttention(opts=opts, embed_dim=embed_dim)

    def unfold(self, x: Tensor):
        b, c, h, w = x.shape
        ph = pw = self.patch_size

        if h % ph != 0 or w % pw != 0:
            raise ValueError(
                f"Feature map size ({h}, {w}) must be divisible by patch_size={self.patch_size}"
            )

        num_h, num_w = h // ph, w // pw
        n = num_h * num_w

        # [B, C, H, W] -> [B, C, num_h, ph, num_w, pw]
        x = x.view(b, c, num_h, ph, num_w, pw)
        # -> [B, C, num_h, num_w, ph, pw]
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        # -> [B, C, N, P]
        x = x.view(b, c, n, self.P)
        # -> [B, C, P, N]
        x = x.permute(0, 1, 3, 2).contiguous()
        return x, (num_h, num_w)

    def fold(self, x: Tensor, grid_shape: tuple):
        b, c, p, n = x.shape
        num_h, num_w = grid_shape
        ph = pw = self.patch_size

        h, w = num_h * ph, num_w * pw

        # [B, C, P, N] -> [B, C, N, P]
        x = x.permute(0, 1, 3, 2).contiguous()
        # -> [B, C, num_h, num_w, ph, pw]
        x = x.view(b, c, num_h, num_w, ph, pw)
        # -> [B, C, num_h, ph, num_w, pw]
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous()
        # -> [B, C, H, W]
        x = x.view(b, c, h, w)
        return x

    def forward(self, x: Tensor, x_prev: Optional[Tensor] = None) -> Tensor:
        x_patch, shape_x = self.unfold(x)

        if x_prev is None:
            out_patch = self.attn(x=x_patch)
        else:
            x_prev_patch, _ = self.unfold(x_prev)
            out_patch = self.attn(x=x_patch, x_prev=x_prev_patch)

        return self.fold(out_patch, shape_x)


class UNetMobileViT(nn.Module):
    """
    Hybrid U-Net + MobileNetV2 + MobileViTv2-style Linear Self Attention.

    Thiết kế tiết kiệm VRAM:
    - attention ở bottleneck
    - attention ở 2 decoder stage sâu nhất
    - 2 decoder stage còn lại dùng residual add
    """
    def __init__(self, opts: Any = None, num_classes: int = 1) -> None:
        super().__init__()

        features = [16, 32, 64, 128]

        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.f_add = FloatFunctional()

        # -----------------------------
        # Encoder
        # -----------------------------
        self.encoder_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()

        in_ch = 3
        for feat in features:
            self.encoder_blocks.append(
                MV2Block(
                    opts=opts,
                    in_channels=in_ch,
                    out_channels=feat,
                    expand_ratio=2
                )
            )
            self.downsample_layers.append(
                Conv2d(
                    in_channels=feat,
                    out_channels=feat,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    opts=opts
                )
            )
            in_ch = feat

        # -----------------------------
        # Bottleneck
        # -----------------------------
        bottleneck_dim = features[-1] * 2  # 128 -> 256
        self.bottleneck_in = MV2Block(
            opts=opts,
            in_channels=features[-1],
            out_channels=bottleneck_dim,
            expand_ratio=2
        )
        self.bottleneck_attn = Attention2DWrapper(
            opts=opts,
            embed_dim=bottleneck_dim,
            patch_size=2
        )
        self.bottleneck_out = MV2Block(
            opts=opts,
            in_channels=bottleneck_dim,
            out_channels=bottleneck_dim,
            expand_ratio=2
        )

        # -----------------------------
        # Decoder
        # -----------------------------
        self.decoder_blocks = nn.ModuleList()
        self.upsample_layers = nn.ModuleList()
        self.cross_attns = nn.ModuleList()

        decoder_feats = list(reversed(features))  # [128, 64, 32, 16]

        for idx, feat in enumerate(decoder_feats):
            # upsample from current channels to feat channels
            self.upsample_layers.append(
                ConvTranspose2d(
                    in_channels=feat * 2 if idx == 0 else decoder_feats[idx - 1],
                    out_channels=feat,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    opts=opts
                )
            )

            # Chỉ giữ attention ở 2 stage sâu nhất (128, 64)
            if idx < 2:
                self.cross_attns.append(
                    Attention2DWrapper(
                        opts=opts,
                        embed_dim=feat,
                        patch_size=2
                    )
                )
            else:
                self.cross_attns.append(None)

            self.decoder_blocks.append(
                MV2Block(
                    opts=opts,
                    in_channels=feat,
                    out_channels=feat,
                    expand_ratio=2
                )
            )

        self.final_conv = Conv2d(
            in_channels=features[0],
            out_channels=num_classes,
            kernel_size=1,
            stride=1,
            padding=0,
            opts=opts
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.quant(x)

        # Encoder
        skip_connections: List[Tensor] = []
        for i in range(len(self.encoder_blocks)):
            x = self.encoder_blocks[i](x)
            skip_connections.append(x)
            x = self.downsample_layers[i](x)

        # Bottleneck
        x = self.bottleneck_in(x)
        x = self.bottleneck_attn(x)
        x = self.bottleneck_out(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for i in range(len(self.decoder_blocks)):
            x = self.upsample_layers[i](x)
            skip_connection = skip_connections[i]

            if self.cross_attns[i] is not None:
                x = self.cross_attns[i](x=x, x_prev=skip_connection)
            else:
                x = self.f_add.add(x, skip_connection)

            x = self.decoder_blocks[i](x)

        x = self.final_conv(x)
        x = self.dequant(x)
        return x 
    
def check_model_convergence(model, device='cpu'):
    print(f"Đưa mô hình lên {device.upper()} và bắt đầu kiểm tra...")
    model.to(device)
    model.train() 

    batch_size = 2
    x = torch.randn(batch_size, 3, 320, 320, device=device) 
    
    y_true = torch.zeros((batch_size, 1, 320, 320), dtype=torch.float32, device=device)
    y_true[:, :, 100:220, 100:220] = 1.0  
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("Bắt đầu ép mô hình overfit trên 1 batch dữ liệu duy nhất:\n")

    epochs = 100
    for epoch in range(epochs):
        optimizer.zero_grad()     

        y_pred = model(x)         
        
        loss = criterion(y_pred, y_true) 
        
        loss.backward()           
        optimizer.step()           

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1:3d}/{epochs}] - Loss: {loss.item():.6f}")

    print("\n=> Kiểm tra hoàn tất!")
    return loss.item()

def save_and_profile_model(model, filepath="unet_mobilevit.pth"):
    print("\n" + "="*40)
    print("THỐNG KÊ CHI TIẾT MÔ HÌNH")
    print("="*40)
    
    # 1. Đếm số lượng tham số
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Tổng số tham số:              {total_params:,}")
    print(f"Tham số cần huấn luyện:       {trainable_params:,}")
    
    model.to('cpu') 
    torch.save(model.state_dict(), filepath)
    file_size_bytes = os.path.getsize(filepath)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    print(f"Đã lưu mô hình tại:           {filepath}")
    print(f"Dung lượng file tĩnh (Size):  {file_size_mb:.2f} MB")
    print("="*40)
    
    return total_params, file_size_mb

if __name__ == "__main__":
    class DummyOpts:
        pass
    opts = DummyOpts()

    model = UNetMobileViT(opts=opts, num_classes=1)
    print(model)
    
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    final_loss = check_model_convergence(model, device)

    save_and_profile_model(model, filepath="unet_mobilevit_weights.pth")