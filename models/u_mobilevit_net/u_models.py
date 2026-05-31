import argparse
from typing import (
    Tuple, 
    Callable,
    Optional,
    Union,
    Any,
    List
)

import torch
from torch import Tensor

from cv_nets.layers.base_layer import BaseLayer
from cv_nets.utils.config_helper import get_param

from models.u_mobilevit_net.encoder_block import UMobileViTEncoder
from models.u_mobilevit_net.decoder_block import UMobileViTDecoder

# [ĐÃ SỬA]: Import đúng các class Head mới
from models.u_mobilevit_net.seg_head import MultiTaskDenseHead, ContextAwareSegHead

     
class UMobileViT(BaseLayer):
    def __init__(
        self,
        in_channels: Optional[int] = None,
        out_channels: Optional[Union[int, Tuple[int, ...], List[int]]] = None,
        d_model: Optional[int] = None,
        expansion_factor: Optional[float] = None,
        alpha: Optional[float] = None,
        patch_size: Optional[Union[int, Tuple[int, int]]] = None,
        dropout_p: Optional[float] = None,
        norm_num_groups: Optional[int] = None,
        bias: Optional[bool] = None, 
        num_transformer_block: Optional[int] = None,
        initializer: Optional[Union[str, Callable[[Tensor], Tensor]]] = None,
        opts: Optional[Any] = None,
        device=None,
        dtype=None,
        *args,
        **kwargs
    ) -> None:
        """
        Initializer of UmobileViT model wrapper. 
        """
        super().__init__(*args, **kwargs)
        
        self.opts = opts or kwargs.get("opts", None)
        
        self.in_channels = get_param(self.opts, in_channels, "in_channels", 3)
        self.out_channels = get_param(self.opts, out_channels, "out_channels", (2, 2))
        self.base_d_model = get_param(self.opts, d_model, "d_model", 64)
        self.expansion_factor = get_param(self.opts, expansion_factor, "expansion_factor", 3.0)
        self.alpha = get_param(self.opts, alpha, "alpha", 1.0)
        self.patch_size = get_param(self.opts, patch_size, "patch_size", (2, 2))
        self.dropout_p = get_param(self.opts, dropout_p, "dropout_p", 0.1)
        self.norm_num_groups = get_param(self.opts, norm_num_groups, "norm_num_groups", 4)
        self.bias = get_param(self.opts, bias, "bias", True)
        self.num_transformer_block = get_param(self.opts, num_transformer_block, "num_transformer_blocks", 2)
        self.init_str = get_param(self.opts, initializer, "initializer", "he_uniform")
        
        assert self.alpha > 0, f"alpha must be greater than zero, got alpha={self.alpha}"

        scaled_d_model = int(self.alpha * self.base_d_model)
        
        module_kwargs = {
            "opts": self.opts,
            "d_model": scaled_d_model,
            "expansion_factor": self.expansion_factor,
            "patch_size": self.patch_size,
            "dropout_p": self.dropout_p,
            "norm_num_groups": self.norm_num_groups,
            "bias": self.bias, 
            "num_transformer_block": self.num_transformer_block,
            "initializer": self.init_str,
            "device": device,
            "dtype": dtype
        }
        
        self.encoder = UMobileViTEncoder(in_channels=self.in_channels, **module_kwargs)
        self.decoder = UMobileViTDecoder(**module_kwargs)
        
        # [ĐÃ SỬA]: Mặc định khởi tạo mô hình đa nhiệm (Multi-task) thay vì fix cứng DrivableAndLane
        self.seg_head = MultiTaskDenseHead(
            out_channels=self.out_channels,
            **module_kwargs 
        )
    
    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        group = parser.add_argument_group(f"Arguments for {cls.__name__} (Model Wrapper)")
        group.add_argument("--alpha", type=float, default=1.0, help="Hệ số kiểm soát độ rộng mạng (Width multiplier)")
        group.add_argument("--head-type", type=str, default="dual", choices=["dual", "single"], help="Loại Segmentation Head (dual hoặc single)")
        
        parser = UMobileViTEncoder.add_arguments(parser)
        parser = UMobileViTDecoder.add_arguments(parser)
        return parser

    def forward(self, input: Tensor) -> Union[Tensor, Tuple[Tensor, ...]]:
        stem_outputs, stage_outputs = self.encoder(input)
        
        output_decoder = self.decoder(tuple(reversed(stage_outputs)))
        
        output = self.seg_head(output_decoder, tuple(reversed(stem_outputs)))
        
        return output


def umobilevit(
    opts: Optional[Any] = None,
    head: Optional[str] = None,
    in_channels: Optional[int] = None,
    out_channels: Optional[Union[int, Tuple[int, ...], List[int]]] = None,
    d_model: Optional[int] = None,
    expansion_factor: Optional[float] = None,
    alpha: Optional[float] = None,
    patch_size: Optional[Union[int, Tuple[int, int]]] = None,
    dropout_p: Optional[float] = None,
    norm_num_groups: Optional[int] = None,
    bias: Optional[bool] = None, 
    num_transformer_block: Optional[int] = None,
    initializer: Optional[Union[str, Callable[[Tensor], Tensor]]] = None,
    device=None,
    dtype=None
) -> UMobileViT:
    """
    Factory function để khởi tạo mạng UMobileViT.
    """
    head_type = head if head is not None else get_param(opts, None, "head_type", "dual")
    
    model = UMobileViT(
        opts=opts,
        in_channels=in_channels,
        out_channels=out_channels,
        d_model=d_model,
        expansion_factor=expansion_factor,
        alpha=alpha,
        patch_size=patch_size,
        dropout_p=dropout_p,
        norm_num_groups=norm_num_groups,
        bias=bias,
        num_transformer_block=num_transformer_block,
        initializer=initializer,
        device=device,
        dtype=dtype
    )
    
    # [ĐÃ SỬA]: Xử lý nếu người dùng muốn khởi tạo mạng tác vụ đơn lẻ (Single Task)
    if head_type != "dual":
        # Xử lý an toàn: Nếu out_channels đang là tuple (vd: (2, 4)) thì chỉ lấy class đầu tiên
        single_out_channels = model.out_channels
        if isinstance(single_out_channels, (tuple, list)):
            single_out_channels = single_out_channels[0]
            
        head_kwargs = {
            "opts": opts,
            "out_channels": single_out_channels, # Dùng int cho ContextAwareSegHead
            "d_model": int(model.base_d_model * model.alpha),
            "expansion_factor": model.expansion_factor,
            "patch_size": model.patch_size,
            "dropout_p": model.dropout_p,
            "norm_num_groups": model.norm_num_groups,
            "bias": model.bias, 
            "num_transformer_block": model.num_transformer_block,
            "initializer": model.init_str,
            "device": device,
            "dtype": dtype
        }
        
        # Ghi đè seg_head thành loại tác vụ đơn (ContextAwareSegHead)
        model.seg_head = ContextAwareSegHead(**head_kwargs)
    
    return model

if __name__ == "__main__":
    formatter = lambda prog: argparse.HelpFormatter(prog, max_help_position=60, width=240)
    parser = argparse.ArgumentParser(description="Kiểm tra thông số UMobileViT", formatter_class=formatter)
    UMobileViT.add_arguments(parser)
    args = parser.parse_args()
    
    # Test 1: Khởi tạo mặc định (Đa tác vụ - Dual Head)
    model = umobilevit(opts=args, out_channels=(2, 4))
    print("\n[+] Đã khởi tạo UMobileViT Đa tác vụ (Multi-task)")
    print(f"Số nhánh: {len(model.seg_head.task_classifiers)} | Classes: {model.out_channels}")
    
    # Test 2: Khởi tạo với Single Head (Tác vụ đơn)
    # args.head_type = "single"
    # model_single = umobilevit(opts=args, out_channels=(2, 4))
    # print("\n[+] Đã khởi tạo UMobileViT Đơn tác vụ (Single-task)")
    # print(f"Sử dụng Module: {model_single.seg_head.__class__.__name__} | Classes: {model_single.seg_head.out_channels}")
    
    
    
    model.eval() # Chuyển sang chế độ đánh giá
    
    # Tạo 2 Tensor giả lập (Batch_size=1, Channels=3, Height, Width)
    dummy_image_small = torch.randn(1, 3, 320, 320)   # Ảnh vuông nhỏ
    dummy_image_large = torch.randn(1, 3, 640, 320)   # Ảnh chữ nhật to
    
    print("\n[--- TEST KÍCH THƯỚC ĐẦU VÀO ĐỘNG ---]")
    
    # Thử với ảnh nhỏ
    out_small = model(dummy_image_small)
    print(f"1. Ảnh đầu vào: {dummy_image_small.shape}")
    print(f"   => Nhánh 1 (2 class) ra: {out_small[0].shape}")
    print(f"   => Nhánh 2 (4 class) ra: {out_small[1].shape}")
    
    print("-" * 40)
    
    # Thử với ảnh to
    out_large = model(dummy_image_large)
    print(f"2. Ảnh đầu vào: {dummy_image_large.shape}")
    print(f"   => Nhánh 1 (2 class) ra: {out_large[0].shape}")
    print(f"   => Nhánh 2 (4 class) ra: {out_large[1].shape}")
    
    
    