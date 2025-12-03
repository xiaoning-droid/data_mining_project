import os
import runpy
import math
from typing import Tuple, List

import numpy as np
import torch
import matplotlib.pyplot as plt

# ----------------- 加：只看这三个 kernel -----------------
KERNELS_TOP3 = [
    "spacextime+linear",  # 最佳 kernel
    "Matern52",           # 强 baseline
    "RBF",                # 经典 baseline
]

# 尝试载入 gp_train_svgp.py 中定义的常量和工具函数
trainer_globals = runpy.run_path(os.path.join(os.path.dirname(__file__), 'gp_train_svgp.py'))
# expected: OUT_DIR, DATA_NPZ, _canonicalize_kernel_name, sanitize
# Force diagnostics output to the project's Figures directory (override trainer setting)
OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Figures'))

# Use RESULTS_DIR to read prediction files (pred_*.pt). Prefer trainer setting if available.
if 'OUT_DIR' in trainer_globals:
    RESULTS_DIR = trainer_globals['OUT_DIR']
else:
    RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Results'))

if 'DATA_NPZ' in trainer_globals:
    DATA_NPZ = trainer_globals['DATA_NPZ']
else:
    DATA_NPZ = os.path.join(OUT_DIR, 'data_prep.npz')

if '_canonicalize_kernel_name' in trainer_globals and 'sanitize' in trainer_globals:
    _canonicalize_kernel_name = trainer_globals['_canonicalize_kernel_name']
    sanitize = trainer_globals['sanitize']
else:
    raise RuntimeError('gp_train_svgp.py must define _canonicalize_kernel_name and sanitize')


def get_tag_from_kernel_name(name: str) -> str:
    """Return the file tag for a given kernel name using trainer canonicalizer."""
    try:
        key, params = _canonicalize_kernel_name(name)
    except Exception:
        key = name
        params = {}

    if key in ('sm', 'sm+linear'):
        q = params.get('q', 4)
        raw_tag = f"{key}(q={q})"
    else:
        raw_tag = key
    return sanitize(raw_tag)


def load_predictions_and_data(kernel_name: str):
    """Load pred_{tag}.pt and data_prep.npz and return a dict with numpy arrays."""
    tag = get_tag_from_kernel_name(kernel_name)
    # Predictions live in RESULTS_DIR; outputs written to OUT_DIR (Figures)
    pred_path = os.path.join(RESULTS_DIR, f"pred_{tag}.pt")
    if not os.path.exists(pred_path):
        raise FileNotFoundError(pred_path)

    pred_pack = torch.load(pred_path, map_location='cpu')
    if 'yhat' not in pred_pack or 'Yte' not in pred_pack:
        raise KeyError(f"pred file {pred_path} missing required keys")

    yhat = pred_pack['yhat']
    Yte = pred_pack['Yte']
    ystd = pred_pack.get('ystd', None)

    data = np.load(DATA_NPZ)
    X_test = data['X_test']
    y_test = data['y_test']
    H = int(data['H'])
    W = int(data['W'])
    T = int(data.get('T', 0))
    if T == 0:
        # 如果 npz 里没存 T，就从 X_test 的第三列推
        if X_test.size:
            T = int(np.max(X_test[:, 2]))
        else:
            T = 0

    out = {
        'kernel_name': kernel_name,
        'tag': tag,
        'X_test': X_test,
        'y_test': y_test,
        'yhat': yhat.numpy() if isinstance(yhat, torch.Tensor) else np.asarray(yhat),
        'ystd': ystd.numpy() if (ystd is not None and isinstance(ystd, torch.Tensor)) else (np.asarray(ystd) if ystd is not None else None),
        'Yte': Yte.numpy() if isinstance(Yte, torch.Tensor) else np.asarray(Yte),
        'H': H,
        'W': W,
        'T': T
    }
    return out


# ---- helpers to map lat/lon to grid indices ----
def _to_idx(val, vmin, vmax, N):
    return np.clip(((val - vmin) / (vmax - vmin) * (N - 1) + 0.5).astype(int), 0, N - 1)


def _build_grid_from_day(X_test: np.ndarray, values: np.ndarray, day: int, H: int, W: int):
    """Return an HxW grid with values filled at positions where X_test[:,2]==day."""
    # 注意：day 存在 float 精度问题的话，可以换成 np.isclose
    is_day = (X_test[:, 2] == day)
    grid = np.full((H, W), np.nan, dtype=np.float32)
    if np.sum(is_day) == 0:
        return grid
    lat = X_test[is_day, 0]
    lon = X_test[is_day, 1]
    lat_idx = _to_idx(lat, 35, 40, H)
    lon_idx = _to_idx(lon, -115, -105, W)
    grid[lat_idx, lon_idx] = values[is_day]
    return grid


# ========== 1. Spatial Error Map：每个 kernel 一张大图，4 列排 31 天 ==========

def plot_spatial_error_grid_for_kernel(kernel_name: str, ncols: int = 4) -> str:
    """
    对单个 kernel，把所有天（1..T）的绝对误差 |yhat - ytrue|
    画成一个多子图：每行 ncols 个，共 rows 行。
    """
    d = load_predictions_and_data(kernel_name)
    X_test = d['X_test']
    y_true = d['Yte']
    yhat = d['yhat']
    H, W, T = d['H'], d['W'], d['T']
    tag = d['tag']

    abs_err = np.abs(yhat - y_true)
    days = np.arange(1, T + 1, dtype=int)

    # 先把每一天的误差 grid 算好，同时算一个全局 vmax，方便统一色标
    grids = []
    vmax = 0.0
    for day in days:
        g = _build_grid_from_day(X_test, abs_err, day, H, W)
        grids.append(g)
        if not np.all(np.isnan(g)):
            vmax = max(vmax, float(np.nanmax(g)))
    if vmax == 0.0:
        vmax = 1.0
    vmin = 0.0

    n = len(days)
    rows = int(math.ceil(n / float(ncols)))
    fig, axes = plt.subplots(rows, ncols, figsize=(4 * ncols, 3 * rows))
    axes = np.atleast_2d(axes)
    axes_flat = axes.ravel()

    im_last = None
    for i, day in enumerate(days):
        ax = axes_flat[i]
        g = grids[i]
        im = ax.imshow(g, origin='lower', cmap='coolwarm', vmin=vmin, vmax=vmax, aspect='auto')
        im_last = im
        ax.set_title(f"Day {day}", fontsize=20)
        ax.set_xticks([]); ax.set_yticks([])
    # 多余的子图关掉
    for j in range(len(days), len(axes_flat)):
        axes_flat[j].axis('off')
    # 添加统一 colorbar：在子图右侧新建一个轴，避免叠加到子图上
    if im_last is not None:
        # compute union bbox of used axes (in figure coordinates)
        used = axes_flat[:len(days)]
        xs = [a.get_position() for a in used]
        x0 = min(b.x0 for b in xs)
        x1 = max(b.x1 for b in xs)
        y0 = min(b.y0 for b in xs)
        y1 = max(b.y1 for b in xs)
        cb_width = 0.02
        cb_pad = 0.15
        cax = fig.add_axes([x1 + cb_pad, y0, cb_width, y1 - y0])
        cbar = fig.colorbar(im_last, cax=cax)
        cbar.set_label("Absolute Error", fontsize=20)
        cbar.ax.tick_params(labelsize=20)

    fig.suptitle(f"{kernel_name} - Spatial Absolute Error (All Days)", fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fname = os.path.join(OUT_DIR, f"errormap_grid_{tag}.png")
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fname


# ========== 2. Uncertainty Map：每个 kernel 一张大图，4 列排 31 天 ==========

def plot_uncertainty_grid_for_kernel(kernel_name: str, ncols: int = 4) -> str:
    """
    对单个 kernel，把所有天（1..T）的预测标准差 ystd
    画成一个多子图：每行 ncols 个，共 rows 行。
    """
    d = load_predictions_and_data(kernel_name)
    if d['ystd'] is None:
        raise RuntimeError(f"No predictive std (ystd) found for {kernel_name}")

    X_test = d['X_test']
    ystd = d['ystd']
    H, W, T = d['H'], d['W'], d['T']
    tag = d['tag']

    days = np.arange(1, T + 1, dtype=int)

    grids = []
    all_vals = []
    for day in days:
        g = _build_grid_from_day(X_test, ystd, day, H, W)
        grids.append(g)
        if not np.all(np.isnan(g)):
            all_vals.append(g[~np.isnan(g)])
    if len(all_vals):
        all_vals = np.concatenate(all_vals)
        vmin = float(np.nanpercentile(all_vals, 2))
        vmax = float(np.nanpercentile(all_vals, 98))
    else:
        vmin, vmax = 0.0, 1.0

    n = len(days)
    rows = int(math.ceil(n / float(ncols)))
    fig, axes = plt.subplots(rows, ncols, figsize=(4 * ncols, 3 * rows))
    axes = np.atleast_2d(axes)
    axes_flat = axes.ravel()

    im_last = None
    for i, day in enumerate(days):
        ax = axes_flat[i]
        g = grids[i]
        im = ax.imshow(g, origin='lower', cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
        im_last = im
        ax.set_title(f"Day {day}", fontsize=20)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(len(days), len(axes_flat)):
        axes_flat[j].axis('off')

    if im_last is not None:
        # compute union bbox of used axes (in figure coordinates)
        used = axes_flat[:len(days)]
        xs = [a.get_position() for a in used]
        x0 = min(b.x0 for b in xs)
        x1 = max(b.x1 for b in xs)
        y0 = min(b.y0 for b in xs)
        y1 = max(b.y1 for b in xs)
        cb_width = 0.02
        cb_pad = 0.15
        cax = fig.add_axes([x1 + cb_pad, y0, cb_width, y1 - y0])
        cbar = fig.colorbar(im_last, cax=cax)
        cbar.set_label("Predictive Std", fontsize=20)
        cbar.ax.tick_params(labelsize=20)

    fig.suptitle(f"{kernel_name} - Predictive Uncertainty (All Days)", fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fname = os.path.join(OUT_DIR, f"uncertainty_grid_{tag}.png")
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return fname

# ========== main：一次性画完 6 张 + 2 张汇总图 ==========

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1）每个 kernel：Spatial Error 多子图
    for k in KERNELS_TOP3:
        try:
            f = plot_spatial_error_grid_for_kernel(k, ncols=4)
            print("Saved spatial error grid:", f)
        except Exception as e:
            print("Error in spatial error grid for", k, ":", e)

    # 2）每个 kernel：Uncertainty 多子图
    for k in KERNELS_TOP3:
        try:
            f = plot_uncertainty_grid_for_kernel(k, ncols=4)
            print("Saved uncertainty grid:", f)
        except Exception as e:
            print("Error in uncertainty grid for", k, ":", e)

if __name__ == "__main__":
    main()
