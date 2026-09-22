import torch
import numpy as np

def stabilize_gui_heatmap(hm):
    """Internal helper."""
    hm = torch.as_tensor(hm).float()
    lap = 4 * hm - torch.roll(hm, 1, 0) - torch.roll(hm, -1, 0) - torch.roll(hm, 1, 1) - torch.roll(hm, -1, 1)
    lap = lap.clamp_min(0)
    s = 0.7 * hm + 0.3 * (lap / (lap.max() + 1e-06))
    return (s - s.min()) / (s.max() - s.min() + 1e-06)

def _otsu_threshold(H):
    """Otsu on [0,1] array"""
    H = np.asarray(H, dtype=np.float32)
    H = np.clip(H, 0.0, 1.0)
    hist, bin_edges = np.histogram(H.ravel(), bins=256, range=(0.0, 1.0))
    total = H.size
    sum_total = np.dot(hist, (bin_edges[:-1] + bin_edges[1:]) / 2.0)
    sum_bg, w_bg, max_var, thresh = (0.0, 0.0, 0.0, 0.0)
    for i in range(256):
        w_bg += hist[i]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        val = (bin_edges[i] + bin_edges[i + 1]) / 2.0
        sum_bg += hist[i] * val
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_total - sum_bg) / w_fg
        between = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        if between > max_var:
            max_var, thresh = (between, val)
    return float(thresh)

def _connected_components(mask, H):
    """Internal helper."""
    mask = mask.astype(np.uint8) > 0
    H = np.asarray(H, dtype=np.float32)
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=np.uint8)
    comps = []
    for y in range(h):
        for x in range(w):
            if mask[y, x] and (not visited[y, x]):
                q = [(x, y)]
                visited[y, x] = 1
                xs, ys, vals = ([], [], [])
                while q:
                    cx, cy = q.pop()
                    xs.append(cx)
                    ys.append(cy)
                    vals.append(H[cy, cx])
                    for nx in (cx - 1, cx, cx + 1):
                        for ny in (cy - 1, cy, cy + 1):
                            if 0 <= nx < w and 0 <= ny < h and (not (nx == cx and ny == cy)):
                                if mask[ny, nx] and (not visited[ny, nx]):
                                    visited[ny, nx] = 1
                                    q.append((nx, ny))
                x1, x2 = (min(xs), max(xs))
                y1, y2 = (min(ys), max(ys))
                comps.append({'bbox': (x1, y1, x2 + 1, y2 + 1), 'peak': float(np.max(vals)), 'mean': float(np.mean(vals)), 'size': len(vals)})
    return comps







def _scale_bbox(bbox, from_shape, to_shape):
    """Internal helper."""
    x1, y1, x2, y2 = map(float, bbox)
    Hh, Hw = from_shape
    W, H = to_shape
    sx = float(W) / float(Hw)
    sy = float(H) / float(Hh)
    X1 = int(round(x1 * sx))
    Y1 = int(round(y1 * sy))
    X2 = int(round(x2 * sx))
    Y2 = int(round(y2 * sy))
    X1 = max(0, min(W - 1, X1))
    Y1 = max(0, min(H - 1, Y1))
    X2 = max(X1 + 1, min(W, X2))
    Y2 = max(Y1 + 1, min(H, Y2))
    return (X1, Y1, X2, Y2)
