import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Union


def _get_patch_size(kernel_size: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    """Hàm hỗ trợ lấy kích thước patch_h và patch_w."""
    if isinstance(kernel_size, int):
        return kernel_size, kernel_size
    return kernel_size[0], kernel_size[1]


def unfold_custom(input: Tensor, kernel_size: Union[int, Tuple[int, int]]) -> Tensor:
    """
    Biến đổi feature map 2D thành chuỗi các patches cho Transformer.
    
    Args:
        input (Tensor): Tensor đầu vào có kích thước (N, C, H, W).
        kernel_size (int hoặc Tuple[int, int]): Kích thước của patch (patch_size).
        
    Returns:
        Tensor: Tensor đã được unfold với kích thước (N, C, spatial, seq_len)
                trong đó spatial = patch_h * patch_w (số pixel trong 1 patch)
                và seq_len = (H // patch_h) * (W // patch_w) (số lượng patches).
    """
    assert input.dim() == 4, f"Expected 4D tensor, got {input.dim()}D tensor."
    
    N, C, H, W = input.shape
    patch_h, patch_w = _get_patch_size(kernel_size)

    assert H % patch_h == 0, f"Height ({H}) must be divisible by patch height ({patch_h})"
    assert W % patch_w == 0, f"Width ({W}) must be divisible by patch width ({patch_w})"

    num_patches_h = H // patch_h
    num_patches_w = W // patch_w
    x = input.view(N, C, num_patches_h, patch_h, num_patches_w, patch_w)
    x = x.permute(0, 1, 3, 5, 2, 4)
    spatial = patch_h * patch_w
    seq_len = num_patches_h * num_patches_w
    x = x.contiguous().view(N, C, spatial, seq_len)
    
    return x


def fold_custom(input: Tensor, output_size: Tuple[int, int], kernel_size: Union[int, Tuple[int, int]]) -> Tensor:
    """
    Phục hồi lại chuỗi các patches từ Transformer về dạng feature map 2D ban đầu.
    
    Args:
        input (Tensor): Tensor đầu ra từ Transformer có kích thước (N, C, spatial, seq_len).
        output_size (Tuple[int, int]): Kích thước (H, W) mong muốn của feature map.
        kernel_size (int hoặc Tuple[int, int]): Kích thước của patch đã dùng khi unfold.
        
    Returns:
        Tensor: Feature map 2D phục hồi với kích thước (N, C, H, W).
    """
    assert input.dim() == 4, f"Expected 4D tensor, got {input.dim()}D tensor."
    
    N, C, spatial, seq_len = input.shape
    H, W = output_size
    patch_h, patch_w = _get_patch_size(kernel_size)

    num_patches_h = H // patch_h
    num_patches_w = W // patch_w
    assert spatial == patch_h * patch_w, "Kích thước spatial không khớp với kernel_size."
    assert seq_len == num_patches_h * num_patches_w, "Kích thước seq_len không khớp với output_size."
    x = input.view(N, C, patch_h, patch_w, num_patches_h, num_patches_w)
    x = x.permute(0, 1, 4, 2, 5, 3)
    x = x.contiguous().view(N, C, H, W)
    
    return x


def separable_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    channels_to_check: int,
    in_proj_weight: Optional[Tensor],
    in_proj_bias: Optional[Tensor],
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    dropout_p: float,
    q_proj_weight: Optional[Tensor],
    k_proj_weight: Optional[Tensor],
    v_proj_weight: Optional[Tensor],
    use_separate_proj_weight: bool,
    training: bool = True,
    need_context_scores: bool = False
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """
    Computes Separable Attention / Linear Context Attention.
    
    Args:
        - query, key, value: Tensors of shape (N, C, spatial, seq_len).
        - channels_to_check: Number of features (in_channels) to extract for key/value.
        - ... (Projection weights and biases)
        - use_separate_proj_weight: Boolean indicating whether to use q, k, v separate 
          weights or a unified in_proj_weight.
          
    Returns:
        - out: Tensor of shape (N, in_channels, spatial, seq_len)
        - context_scores (optional): Tensor of shape (N, 1, spatial, seq_len)
    """
    
    N, _, spatial, seq_len = query.shape


    if not use_separate_proj_weight:
        x = query.permute(0, 2, 3, 1)  
        qkv = F.linear(x, in_proj_weight, in_proj_bias) 
        qkv = qkv.permute(0, 3, 1, 2) 
        
        q, k, v = torch.split(qkv, [1, channels_to_check, channels_to_check], dim=1)
    else:
        q_x = query.permute(0, 2, 3, 1)
        k_x = key.permute(0, 2, 3, 1)
        v_x = value.permute(0, 2, 3, 1)
        
        if in_proj_bias is not None:
            q_b, k_b, v_b = torch.split(in_proj_bias, [1, channels_to_check, channels_to_check])
        else:
            q_b, k_b, v_b = None, None, None
            
        q = F.linear(q_x, q_proj_weight, q_b).permute(0, 3, 1, 2)
        k = F.linear(k_x, k_proj_weight, k_b).permute(0, 3, 1, 2) 
        v = F.linear(v_x, v_proj_weight, v_b).permute(0, 3, 1, 2) 

    # [FIX NaN] Tính softmax trong FP32 để tránh overflow khi dùng autocast FP16.
    # Query q có shape (N, 1, spatial, seq_len) — giá trị không được scale bởi sqrt(d_k)
    # nên dễ vượt ngưỡng exp(11) ≈ 59874 trong FP16, gây Inf → NaN.
    context_scores = F.softmax(q.float(), dim=-1).to(q.dtype)
    context_scores = F.dropout(context_scores, p=dropout_p, training=training)

    context_vector = k * context_scores
    context_vector = torch.sum(context_vector, dim=-1, keepdim=True)

    # [FIX NaN] relu(v) * Inf = NaN (vì 0 * Inf = NaN trong IEEE 754).
    # Đã được phòng ngừa bằng softmax FP32 bên trên, nhưng clamp context_vector
    # để bảo vệ thêm một lớp an toàn.
    context_vector = torch.clamp(context_vector, max=1e4)
    out = F.relu(v) * context_vector

    out_x = out.permute(0, 2, 3, 1)
    out = F.linear(out_x, out_proj_weight, out_proj_bias)
    out = out.permute(0, 3, 1, 2)   
    if need_context_scores:
        return out, context_scores
    
    return out