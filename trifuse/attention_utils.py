import math
import torch
import numpy as np

def _to_np(x):
    """Internal helper."""
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().contiguous().numpy()
    else:
        x = np.asarray(x)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x

def _nearest_hw_from_tokens(n, target_ratio):
    """Internal helper."""
    best = (int(math.sqrt(n)), max(1, n // int(math.sqrt(n))))
    best_err = 1000000000.0
    for h in range(1, int(math.sqrt(n)) + 1):
        if n % h == 0:
            w = n // h
            err = abs(w / h - target_ratio)
            if err < best_err:
                best_err = err
                best = (h, w)
    return best
