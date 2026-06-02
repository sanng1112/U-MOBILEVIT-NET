import argparse
from typing import Tuple, Callable, Optional, Union, Any, List
import torch.nn.functional as F
import torch
from torch import Tensor
from torch.nn import (
    ModuleList,
    Sequential,
    ReLU,
    Sigmoid
)
from torch.nn.init import zeros_
from torch.nn.modules.utils import _pair

from cv_nets.layers.base_layer import BaseLayer
from cv_nets.layers.conv_layer import Conv2d
from cv_nets.utils.config_helper import get_param

from models.u_mobilevit_net.transfomer import _get_initializer
from models.u_mobilevit_net.decoder_block import (
    UMobileViTDecoderConcatLayer,
    _get_upsample_block
)
from models.u_mobilevit_net.module import get_groupnorm_groups

# -----------------------------------------------------------------------------
# 1. UPSAMPLE HEAD (Mô-đun phục hồi độ phân giải)
# -----------------------------------------------------------------------------
class UpsampleHead(BaseLayer):
    """
    Mô-đun phục hồi độ phân giải của feature map về kích thước tương đương ảnh gốc.
    Tích hợp skip-connection từ Encoder (outputs_stem) thông qua UMobileViTDecoderConcatLayer.
    """
    def __init__(
        self,  
        d_model: Optional[int] = None,
        dropout_p: Optional[float] = None,
        norm_num_groups: Optional[int] = None,
        bias: Optional[bool] = None, 
        initializer: Optional[Union[str, Callable[[Tensor], Tensor]]] = None,
        opts: Optional[Any] = None,
        device=None,
        dtype=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        
        self.opts = opts or kwargs.get("opts", None)
        self.d_model = get_param(self.opts, d_model, "d_model", 64)
        self.dropout_p = get_param(self.opts, dropout_p, "dropout_p", 0.1)
        self.norm_num_groups = get_param(self.opts, norm_num_groups, "norm_num_groups", 4)
        self.bias = get_param(self.opts, bias, "bias", True)
        self.init_str = get_param(self.opts, initializer, "initializer", "he_uniform")

        self.initializer = _get_initializer(self.init_str)
        
        factory_kwargs = {"device": device, "dtype": dtype}
        decoder_layer_kwargs = {
            "dropout_p": self.dropout_p,
            "norm_num_groups": self.norm_num_groups,
            "bias": self.bias,
            "initializer": self.init_str,
            "opts": self.opts,
        }
        upsampling_kwargs = {
            "scale_factor": 2,
            "mode": "nearest"
        }
        upsampling_conv_kwargs = {
            "kernel_size": 3,
            "stride": 1,
            "padding": 1,
            "norm_num_groups": self.norm_num_groups,
            "bias": self.bias,
        }
        
        self.upsample_head_x2 = ModuleList([
           Sequential(
                Conv2d(
                    opts=self.opts,
                    in_channels=self.d_model,
                    out_channels=self.d_model // 2,
                    kernel_size=1,
                    bias=self.bias,
                    **factory_kwargs    
                ),
                _get_upsample_block(
                    self.initializer,
                    opts=self.opts,
                    in_channels=self.d_model // 2,
                    out_channels=self.d_model // 2,
                    groups=self.d_model // 2,
                    **upsampling_kwargs, 
                    **upsampling_conv_kwargs,
                    **factory_kwargs
                ),
            ),
            UMobileViTDecoderConcatLayer(
                in_channels=self.d_model // 2,
                **decoder_layer_kwargs, 
                **factory_kwargs,
                **kwargs,
            )
        ])
        
        self.upsample_head_x4 = ModuleList([
            Sequential(
                Conv2d(
                    opts=self.opts,
                    in_channels=self.d_model // 2,
                    out_channels=self.d_model // 4,
                    kernel_size=1,
                    bias=self.bias,
                    **factory_kwargs    
                ),
                _get_upsample_block(
                    self.initializer,
                    opts=self.opts,
                    in_channels=self.d_model // 4,
                    out_channels=self.d_model // 4,
                    groups=self.d_model // 4,
                    **upsampling_kwargs, 
                    **upsampling_conv_kwargs,
                    **factory_kwargs
                ),
            ),
            UMobileViTDecoderConcatLayer(
                in_channels=self.d_model // 4,
                **decoder_layer_kwargs, 
                **factory_kwargs,
                **kwargs,
            )
        ])
        
        self.upsample_head_x8 = ModuleList([
            Sequential(
                Conv2d(
                    opts=self.opts,
                    in_channels=self.d_model // 4,
                    out_channels=self.d_model // 8,
                    kernel_size=1,
                    bias=self.bias,
                    **factory_kwargs    
                ),
                _get_upsample_block(
                    self.initializer,
                    opts=self.opts,
                    in_channels=self.d_model // 8,
                    out_channels=self.d_model // 8,
                    groups=self.d_model // 8,
                    **upsampling_kwargs, 
                    **upsampling_conv_kwargs,
                    **factory_kwargs
                ),
            )
        ])
        
        self.layers = ModuleList([
            self.upsample_head_x2,
            self.upsample_head_x4,
            self.upsample_head_x8,
        ])

        # [ĐÃ SỬA] Khởi tạo các Conv2d 1x1 pointwise trong UpsampleHead
        # Trước đây chúng chỉ dựa vào default init của custom Conv2d,
        # không nhất quán với he_uniform của phần còn lại của model.
        self._reset_parameters()

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser

    def _reset_parameters(self) -> None:
        """Khởi tạo các Conv2d pointwise (1x1) trong UpsampleHead với he_uniform."""
        for stage in self.layers:
            # Mỗi stage có Sequential[0] chứa Conv2d 1x1 + UpsampleBlock
            sequential = stage[0]
            if isinstance(sequential, Sequential):
                for layer in sequential:
                    if isinstance(layer, Conv2d):
                        self.initializer(layer.weight)
                        if getattr(layer, "bias", None) is not None:
                            zeros_(layer.bias)

    def forward(self, input: Tensor, outputs_stem: Tuple[Tensor, ...]) -> Tensor:
        assert (
            len(outputs_stem) == len(self.layers) - 1
        ), f"length of output_stems must be {len(self.layers) - 1}, got {len(outputs_stem)}"
        
        Z = input
        for i, output_stem in enumerate(outputs_stem):
            # 1. Phóng to feature map
            Z = self.layers[i][0](Z) 
            
            # 2. [FIX] Ép kích thước Z phải BẰNG ĐÚNG kích thước của output_stem
            if Z.shape[2:] != output_stem.shape[2:]:
                Z = F.interpolate(Z, size=output_stem.shape[2:], mode="bilinear", align_corners=False)
            
            # 3. Đưa vào khối Concat/Add (Lúc này 2 tensor đã khớp 100%)
            for layer in self.layers[i][1:]: 
                Z = layer(Z, output_stem)
                
        # Phóng to bước cuối cùng để ra output
        Z = self.layers[-1][0](Z)
        
        return Z


# -----------------------------------------------------------------------------
# 2. FEATURE REFINEMENT BLOCK (Mô-đun tinh chỉnh đặc trưng tổng quát)
# -----------------------------------------------------------------------------
class FeatureRefinementBlock(BaseLayer):
    """
    Khối tinh chỉnh đặc trưng tổng quát.
    Sử dụng Dilated Convolution để mở rộng Receptive Field và Spatial Attention 
    để lọc nhiễu không gian, phù hợp cho các bài toán Dense Prediction cần độ chi tiết cao.
    """
    def __init__(
        self, 
        in_channels: int, 
        norm_num_groups: int,
        bias: bool,
        opts: Optional[Any] = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        
        self.dilated_conv = Conv2d(
            opts=opts,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=2,
            dilation=2,
            groups=in_channels, 
            bias=bias,
            **factory_kwargs
        )
        
        self.spatial_attention = Sequential(
            Conv2d(
                opts=opts, 
                in_channels=2, 
                out_channels=1, 
                kernel_size=7, 
                stride=1, 
                padding=3, 
                bias=False, 
                **factory_kwargs
            ),
            Sigmoid()
        )
        
        self.mix_conv = Conv2d(
            opts=opts, 
            in_channels=in_channels, 
            out_channels=in_channels, 
            kernel_size=1, 
            bias=bias, 
            **factory_kwargs
        )

    def forward(self, x: Tensor) -> Tensor:
        feat = self.dilated_conv(x)
        feat = self.mix_conv(feat)
        
        avg_out = torch.mean(feat, dim=1, keepdim=True)
        max_out, _ = torch.max(feat, dim=1, keepdim=True)
        att_map = torch.cat([avg_out, max_out], dim=1)
        att_map = self.spatial_attention(att_map)
        
        return x + (feat * att_map)


def _build_seg_classifier(
    opts: Optional[Any],
    in_channels: int,
    out_channels: int,
    bias: bool,
    initializer: Callable[[Tensor], Tensor],
    **factory_kwargs,
) -> Sequential:
    """
    Bộ phân loại pixel nhẹ (depthwise 3x3 -> pointwise 1x1) dùng cho mỗi nhánh tác vụ.
    Tách riêng để các nhánh trong MultiTaskDenseHead có thể chia sẻ phần upsample/refinement
    chung và chỉ nhân đôi đúng lớp phân loại cuối (rất nhẹ về FLOPs).
    """
    classifier = Sequential(
        Conv2d(
            opts=opts,
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
            bias=bias,
            **factory_kwargs,
        ),
        ReLU(inplace=True),
        Conv2d(
            opts=opts,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
            **factory_kwargs,
        ),
    )
    for layer in classifier:
        if isinstance(layer, Conv2d):
            initializer(layer.weight)
            if getattr(layer, "bias", None) is not None:
                zeros_(layer.bias)
    return classifier


class ContextAwareSegHead(BaseLayer):
    """
    Đầu ra phân vùng ngữ nghĩa đơn lẻ, tích hợp cơ chế nhận thức ngữ cảnh
    thông qua khối Feature Refinement.
    """
    def __init__(
        self, 
        d_model: Optional[int] = None,
        dropout_p: Optional[float] = None,
        out_channels: Optional[int] = None,
        norm_num_groups: Optional[int] = None,
        bias: Optional[bool] = None, 
        initializer: Optional[Union[str, Callable[[Tensor], Tensor]]] = None,
        opts: Optional[Any] = None,
        device=None,
        dtype=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        
        self.opts = opts or kwargs.get("opts", None)
        self.d_model = get_param(self.opts, d_model, "d_model", 64)
        self.dropout_p = get_param(self.opts, dropout_p, "dropout_p", 0.1)
        self.out_channels = get_param(self.opts, out_channels, "out_channels", 2)
        self.norm_num_groups = get_param(self.opts, norm_num_groups, "norm_num_groups", 4)
        self.bias = get_param(self.opts, bias, "bias", True)
        self.init_str = get_param(self.opts, initializer, "initializer", "he_uniform")
        
        factory_kwargs = {"device": device, "dtype": dtype}
        self.initializer = _get_initializer(self.init_str)
        
        # [ĐÃ SỬA] Chỉ truyền các tham số UpsampleHead thực sự cần.
        # UpsampleHead không dùng expansion_factor, patch_size, num_transformer_block —
        # trước đây **kwargs vô tình truyền chúng qua, gây inconsistency với
        # MultiTaskDenseHead (vốn không truyền **kwargs).
        self.upsample_module = UpsampleHead(
            opts=self.opts,
            d_model=self.d_model,
            dropout_p=self.dropout_p,
            norm_num_groups=self.norm_num_groups,
            bias=self.bias,
            initializer=self.init_str,
            **factory_kwargs
        )
        
        self.refinement_module = FeatureRefinementBlock(
            in_channels=self.d_model // 8,
            norm_num_groups=self.norm_num_groups,
            bias=self.bias,
            opts=self.opts,
            **factory_kwargs
        )
        
        self.classifier = _build_seg_classifier(
            opts=self.opts,
            in_channels=self.d_model // 8,
            out_channels=self.out_channels,
            bias=self.bias,
            initializer=self.initializer,
            **factory_kwargs,
        )

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser

    def forward(self, input: Tensor, outputs_stem: Tuple[Tensor, ...]) -> Tensor:
        seg_head_feat = self.upsample_module(input, outputs_stem)
        refined_feat = self.refinement_module(seg_head_feat)
        seg_mask = self.classifier(refined_feat)
        return seg_mask
        

class MultiTaskDenseHead(BaseLayer):
    """
    Đầu ra Đa tác vụ (Multi-task).
    Số lượng tác vụ và số class của từng tác vụ được cấu hình tự động
    thông qua tham số `out_channels` (Tuple hoặc List).

    [TỐI ƯU EDGE] Toàn bộ pipeline phục hồi độ phân giải (UpsampleHead) và khối
    tinh chỉnh đặc trưng (FeatureRefinementBlock) được CHIA SẺ chung cho mọi tác vụ
    và chỉ chạy MỘT lần. Mỗi tác vụ chỉ sở hữu một bộ phân loại pixel rất nhẹ
    (depthwise 3x3 + pointwise 1x1). Đây là chuẩn thiết kế multi-task (YOLOP/HybridNets):
    chia sẻ phần decode tốn kém, chỉ tách nhánh ở lớp phân loại cuối.

    Trước: mỗi tác vụ chạy lại upsample + refinement => O(N_task) phần đắt nhất của mạng.
    Sau:   upsample + refinement chạy 1 lần => chi phí gần như không đổi khi tăng số tác vụ.
    """
    def __init__(
        self,
        d_model: Optional[int] = None,
        dropout_p: Optional[float] = None,
        out_channels: Optional[Union[int, Tuple[int, ...], List[int]]] = None,
        norm_num_groups: Optional[int] = None,
        bias: Optional[bool] = None,
        initializer: Optional[Union[str, Callable[[Tensor], Tensor]]] = None,
        opts: Optional[Any] = None,
        device=None,
        dtype=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.opts = opts or kwargs.get("opts", None)

        # Xử lý linh hoạt out_channels (hỗ trợ 1 task, 2 task, hoặc N task)
        out_c_param = get_param(self.opts, out_channels, "out_channels", (2, 2))
        if isinstance(out_c_param, int):
            self.out_channels_list = (out_c_param,)
        else:
            self.out_channels_list = tuple(out_c_param)

        self.d_model = get_param(self.opts, d_model, "d_model", 64)
        self.dropout_p = get_param(self.opts, dropout_p, "dropout_p", 0.1)
        self.norm_num_groups = get_param(self.opts, norm_num_groups, "norm_num_groups", 4)
        self.bias = get_param(self.opts, bias, "bias", True)
        self.init_str = get_param(self.opts, initializer, "initializer", "he_uniform")

        factory_kwargs = {"device": device, "dtype": dtype}
        self.initializer = _get_initializer(self.init_str)

        # --- Phần CHIA SẺ chung cho mọi tác vụ (chạy 1 lần) ---
        self.upsample_module = UpsampleHead(
            opts=self.opts,
            d_model=self.d_model,
            dropout_p=self.dropout_p,
            norm_num_groups=self.norm_num_groups,
            bias=self.bias,
            initializer=self.init_str,
            **factory_kwargs,
        )

        self.refinement_module = FeatureRefinementBlock(
            in_channels=self.d_model // 8,
            norm_num_groups=self.norm_num_groups,
            bias=self.bias,
            opts=self.opts,
            **factory_kwargs,
        )

        # --- Phần RIÊNG mỗi tác vụ: chỉ bộ phân loại pixel nhẹ ---
        self.task_classifiers = ModuleList([
            _build_seg_classifier(
                opts=self.opts,
                in_channels=self.d_model // 8,
                out_channels=c,
                bias=self.bias,
                initializer=self.initializer,
                **factory_kwargs,
            )
            for c in self.out_channels_list
        ])

    @classmethod
    def add_arguments(cls, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        return parser

    def forward(
        self,
        input: Tensor,
        outputs_stem: Tuple[Tensor, ...]
    ) -> Tuple[Tensor, ...]:
        """
        Returns:
            Tuple[Tensor, ...]: Danh sách các Tensors đầu ra, tương ứng với số lượng tác vụ.
        """
        # Decode dùng chung (tốn kém) -> chạy đúng 1 lần
        shared_feat = self.upsample_module(input, outputs_stem)
        shared_feat = self.refinement_module(shared_feat)

        # Mỗi tác vụ chỉ chạy thêm bộ phân loại nhẹ trên đặc trưng chung
        return tuple(classifier(shared_feat) for classifier in self.task_classifiers)


if __name__ == "__main__":
    formatter = lambda prog: argparse.HelpFormatter(prog, max_help_position=60, width=240)
    parser = argparse.ArgumentParser(description="Kiểm tra MultiTaskDenseHead", formatter_class=formatter)
    args = parser.parse_args()
    
    # Test khởi tạo mô hình đa tác vụ: Task 1 (Segmentation - 2 classes), Task 2 (Grading - 4 classes)
    model = MultiTaskDenseHead(opts=args, d_model=64, out_channels=(2, 4))
    
    print(model)
    print(f"\n[+] Khởi tạo thành công MultiTaskDenseHead với {len(model.out_channels_list)} nhánh tác vụ độc lập!")
    print(f"[+] Số classes tương ứng mỗi nhánh: {model.out_channels_list}")