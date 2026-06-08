from types import SimpleNamespace
from typing import Any, Dict, List


def dict_to_namespace(d: Any) -> Any:
    """Chuyển dict thành SimpleNamespace lồng nhau (dùng cho config)."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(v) for v in d]
    return d


def namespace_to_dict(ns: Any) -> Any:
    """Chuyển SimpleNamespace về dict (dùng cho serialize config)."""
    if isinstance(ns, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    elif isinstance(ns, dict):
        return {k: namespace_to_dict(v) for k, v in ns.items()}
    elif isinstance(ns, list):
        return [namespace_to_dict(v) for v in ns]
    return ns
