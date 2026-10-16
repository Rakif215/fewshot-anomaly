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

# Memory: SubspaceAD asks for output_attentions=True, which keeps every layer's attention map. For DINOv2-giant at 672 px
# (2304 tokens, 40 layers) that is ~20 GB and overflows a 15 GB T4. Its saliency mask reads only ONE layer
# (--dino_saliency_layer, default 6), so with SUBSPACEAD_KEEP_ATTN_LAYER set we keep that layer and replace the others with
# a 1-element tensor. Scores are unchanged because no other layer's attention is ever read.
import os
if torch is not None and os.environ.get("SUBSPACEAD_KEEP_ATTN_LAYER") is not None:
    try:
        import transformers.models.dinov2_with_registers.modeling_dinov2_with_registers as _m
        _keep = int(os.environ["SUBSPACEAD_KEEP_ATTN_LAYER"])
        _orig_attn = _m.eager_attention_forward
        _order = {}

        def _slim_attn(module, *args, **kwargs):
            out, weights = _orig_attn(module, *args, **kwargs)
            idx = _order.setdefault(id(module), len(_order))
            if idx != _keep:
                weights = weights.new_zeros(1)
            return out, weights

        _m.eager_attention_forward = _slim_attn
    except Exception as e:  # leave behaviour unchanged if the library layout differs
        print("compat: attention slimming not applied:", e)
