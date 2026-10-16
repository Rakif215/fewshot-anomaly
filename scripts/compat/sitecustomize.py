"""Compatibility shims, loaded automatically when scripts/compat is on PYTHONPATH.

The official third_party code is kept unmodified; newer library versions renamed a few functions.
"""
try:
    import numpy as np
except ImportError:  # nothing to patch
    np = None

# NumPy 2.x removed np.trapz (renamed np.trapezoid); SubspaceAD's AU-PRO uses np.trapz.
if np is not None and not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid

# AnomalyDINO hardcodes device 'cuda' for the model (the --device flag is ignored) and calls torch.cuda.synchronize().
# On a CPU-only machine (Codespaces, Mac) map 'cuda' to 'cpu' and make synchronize a no-op.
try:
    import torch
except ImportError:  # nothing to patch
    torch = None

if torch is not None and not torch.cuda.is_available():
    _orig_to = torch.nn.Module.to

    def _cpu(x):
        if isinstance(x, str) and x.startswith("cuda"):
            return "cpu"
        if isinstance(x, torch.device) and x.type == "cuda":
            return torch.device("cpu")
        return x

    def _module_to(self, *args, **kwargs):
        args = tuple(_cpu(a) for a in args)
        if "device" in kwargs:
            kwargs["device"] = _cpu(kwargs["device"])
        return _orig_to(self, *args, **kwargs)

    _orig_tensor_to = torch.Tensor.to

    def _tensor_to(self, *args, **kwargs):
        args = tuple(_cpu(a) for a in args)
        if "device" in kwargs:
            kwargs["device"] = _cpu(kwargs["device"])
        return _orig_tensor_to(self, *args, **kwargs)

    torch.nn.Module.to = _module_to
    torch.Tensor.to = _tensor_to
    torch.cuda.synchronize = lambda *a, **k: None
