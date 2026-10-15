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
