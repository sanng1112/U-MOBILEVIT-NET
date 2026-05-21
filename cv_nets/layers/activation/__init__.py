import argparse
import importlib
import inspect  # <-- Thêm thư viện này để đọc signature của class
import os
from typing import Optional, Union, Any
from types import SimpleNamespace
import torch.nn as nn

from utils import logger
from utils.config_helper import get_param

SUPPORTED_ACT_FNS = []
ACT_FN_MODULES = {}


def register_act_fn(name: str):
    def register_fn(cls):
        if name in SUPPORTED_ACT_FNS:
            raise ValueError(f"Cannot register duplicate activation function ({name})")
        SUPPORTED_ACT_FNS.append(name)
        ACT_FN_MODULES[name] = cls
        return cls
    return register_fn

def arguments_activation_fn(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--type",
        default="relu",
        type=str,
        help="Non-linear function name",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Use non-linear functions inplace",
    )
    parser.add_argument(
        "--neg-slope",
        default=0.1,
        type=float,
        help="Negative slope in leaky relu function",
    )
    return parser


def get_config_prop(opts: Any, prop_path: str, default: Any = None) -> Any:
    if opts is None:
        return default
    try:
        parts = prop_path.split('.')
        for part in parts:
            if isinstance(opts, dict):
                opts = opts.get(part)
            else:
                opts = getattr(opts, part)
        return opts if opts is not None else default
    except (AttributeError, KeyError):
        return default


def build_activation_layer(
    opts: Any,
    act_type: Optional[str] = None,
    **kwargs  # <-- Sử dụng **kwargs để nhận mọi tham số bổ sung linh hoạt từ bên ngoài
) -> Optional[nn.Module]:
    """
    Hàm khởi tạo tầng kích hoạt tổng quát.
    Tự động phân giải cấu hình từ đa nguồn (opts phẳng, opts lồng, hoặc tham số truyền trực tiếp)
    và chỉ nạp các biến tương thích với hàm __init__ của lớp đích.
    """
    
    # 1. Định danh act_type đa kênh (Ưu tiên: tham số gọi trực tiếp -> cấu hình lồng -> cấu hình phẳng -> mặc định)
    if not act_type:
        act_type = (
            get_config_prop(opts, "act.name") or 
            get_config_prop(opts, "act.type") or 
            get_config_prop(opts, "type") or 
            "relu"
        )
        
    if not act_type or not isinstance(act_type, str):
        return None

    act_type = act_type.lower()

    if act_type not in SUPPORTED_ACT_FNS:
        logger.error(
            f"Supported activation layers: {SUPPORTED_ACT_FNS}. Supplied: {act_type}"
        )
        raise NotImplementedError(f"Activation function '{act_type}' is not supported/registered.")

    # 2. Thu thập toàn bộ các tham số có thể cấu hình từ file cấu hình / giá trị mặc định
    # Đồng bộ hóa tên biến tiềm năng
    inplace_val = kwargs.get("inplace", get_config_prop(opts, "act.inplace") or get_config_prop(opts, "inplace", False))
    neg_slope_val = kwargs.get("negative_slope", get_config_prop(opts, "act.neg_slope") or get_config_prop(opts, "neg_slope", 0.1))
    num_params_val = kwargs.get("num_parameters", get_config_prop(opts, "act.num_parameters") or get_config_prop(opts, "num_parameters", 1))

    # Đóng gói thành một "kho chứa dữ liệu thô"
    raw_args = {
        "inplace": inplace_val,
        "negative_slope": neg_slope_val,
        "neg_slope": neg_slope_val,  # Mapping dự phòng cho cả 2 cách đặt tên biến
        "num_parameters": num_params_val,
        "num_params": num_params_val
    }
    # Gộp thêm bất kỳ cặp key-value tùy biến nào mà người dùng truyền trực tiếp qua hàm
    raw_args.update(kwargs)

    # 3. Trích xuất Lớp đích được đăng ký
    act_class = ACT_FN_MODULES[act_type]

    # 4. CỐT LÕI TỔNG QUÁT: Lọc tham số tự động dựa trên Signature thực tế của Class đích
    sig = inspect.signature(act_class.__init__)
    allowed_params = sig.parameters

    filtered_args = {}
    for param_name, param in allowed_params.items():
        # Bỏ qua các tham số hệ thống mặc định của hàm khởi tạo
        if param_name in ["self", "args", "kwargs"]:
            continue
        
        # Nếu tham số mà lớp đích yêu cầu nằm trong kho dữ liệu thô của chúng ta -> lấy ra truyền vào
        if param_name in raw_args:
            filtered_args[param_name] = raw_args[param_name]

    # Nếu bản thân lớp đích nhận biến mở rộng (**kwargs), đổ toàn bộ dữ liệu thô vào cho nó tự xử lý
    has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in allowed_params.values())
    if has_var_keyword:
        filtered_args.update(raw_args)

    # 5. Khởi tạo thực thể an toàn tuyệt đối
    return act_class(**filtered_args)


# --- Giữ nguyên logic auto-import ---
act_dir = os.path.dirname(__file__)
for file in os.listdir(act_dir):
    path = os.path.join(act_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        # THAY THẾ CÁCH CŨ BẰNG CÁCH NÀY:
        # Tự động tách 'relu.py' thành ['relu', '.py'] và lấy phần tử [0]
        model_name = os.path.splitext(file)[0]
        
        try:
            importlib.import_module("layers.activation." + model_name)
        except Exception as e:
            logger.warning(f"Failed to auto-import module '{model_name}': {e}")