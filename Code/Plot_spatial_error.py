import os
import numpy as np
import torch
import matplotlib

# === 强制使用非交互式后端 Agg (防报错) ===
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

# === 全局字体设置 ===
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

# Paths
RESULTS_DIR = r"C:\Users\Xiaonongzi\Desktop\CIVE 650\Project\Results"
FIG_DIR = r"C:\Users\Xiaonongzi\Desktop\CIVE 650\Project\Figures\Grid_Summaries"
os.makedirs(FIG_DIR, exist_ok=True)

DATA_NPZ = os.path.join(RESULTS_DIR, "data_prep.npz")

# === 显示名称映射 ===
DISPLAY_NAMES = {
    'n100_seed2': 'Random Forest',
    'idw': 'IDW',
    'linear': 'Linear',
    'matern32': 'Matern32',
    'matern52': 'Matern52',
    'rbf': 'RBF',
    'spacextime': 'Space x Time',
    'matern52_elev': 'matern52_Elev',
    'rbf_elev': 'rbf_Elev',
    'spacextime_elev': 'spacextime_Elev',
    'matern52_sharedkernel_temp_elev': 'matern52_multi',
    'rbf_sharedkernel_temp_elev': 'rbf_multi',
    'spacextime_sharedkernel_temp_elev': 'spacextime_multi',
}

def get_display_name(tag):
    return DISPLAY_NAMES.get(tag, tag)

def load_pred(tag):
    pth = os.path.join(RESULTS_DIR, f'pred_{tag}.pt')
    try:
        d = torch.load(pth, map_location='cpu', weights_only=False)
        return d
    except Exception as e:
        print(f"Error loading {tag}: {e}")
        return None

def choose_prediction_array(pred_pack):
    candidates = ['yhat', 'y_hat', 'y_pred', 'pred', 'preds', 'ymean', 'mean', 'mu', 'Yte']
    if hasattr(pred_pack, 'keys'):
        for k in candidates:
            if k in pred_pack: return pred_pack[k]
        for k in pred_pack.keys():
            if str(k).endswith('_hat'): return pred_pack[k]
    return None

def choose_uncertainty_array(pred_pack):
    candidates = ['ystd', 'y_std', 'std', 'sigma', 'uncertainty', 'y_stddev']
    if hasattr(pred_pack, 'keys'):
        for k in candidates:
            if k in pred_pack: return pred_pack[k]
    return None

def plot_grid_for_kernel(tag, data_list, title_suffix, cmap, vmin, vmax, out_name):
    """
    绘制 8x4 网格图 (31天)
    data_list: list of 31 numpy arrays (H, W)
    """
    # 创建画布
    cols = 4
    rows = 8
    fig, axes = plt.subplots(rows, cols, figsize=(12, 18)) # 调整宽高比以适应布局
    axes = axes.flatten()
    
    display_name = get_display_name(tag)
    fig.suptitle(f"{display_name} - {title_suffix}", fontsize=20, y=0.92) # 主标题

    # 遍历 32 个格子 (31天 + 1个空)
    for i in range(len(axes)):
        ax = axes[i]
        
        if i < len(data_list):
            # 获取当天数据
            day_data = data_list[i]
            
            # === 核心：垂直翻转 ===
            day_data = np.flipud(day_data) 
            
            # 绘图
            # 使用 copy 避免修改原数据，将 NaN 设为背景色需要用 set_bad
            current_cmap = matplotlib.colormaps[cmap].copy()
            current_cmap.set_bad(color='white')
            
            im = ax.imshow(day_data, cmap=current_cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"Day {i+1}", fontsize=12)
            ax.axis('off') # 去除坐标轴
        else:
            # 多余的格子隐藏
            ax.axis('off')

    # === 添加统一的 Colorbar ===
    # 在右侧添加一个轴用于放 colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(title_suffix, fontsize=14)
    
    # 保存
    plt.subplots_adjust(wspace=0.1, hspace=0.3, left=0.05, right=0.9)
    out_path = os.path.join(FIG_DIR, out_name)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out_name}")

def process_kernels(tags, pack, day_range):
    H = int(pack['H']); W = int(pack['W'])
    
    for tag in tags:
        print(f"\nProcessing Kernel: {tag} ...")
        pred_pack = load_pred(tag)
        if not pred_pack: continue

        # 1. 提取预测值
        yhat = choose_prediction_array(pred_pack)
        if yhat is None:
            print(f"Skipping {tag}: No prediction data found.")
            continue
        yhat_np = np.asarray(yhat) if not isinstance(yhat, torch.Tensor) else yhat.cpu().numpy()
        
        # 2. 提取真实值 (用于计算 Error)
        Yte = pred_pack.get('Yte', pack['y_test'])
        Yte_np = np.asarray(Yte) if not isinstance(Yte, torch.Tensor) else Yte.cpu().numpy()
        
        # 3. 提取不确定性
        ystd = choose_uncertainty_array(pred_pack)
        ystd_np = None
        if ystd is not None:
            ystd_np = np.asarray(ystd) if not isinstance(ystd, torch.Tensor) else ystd.cpu().numpy()

        # --- 收集该核 31 天的数据 ---
        list_pred = []
        list_err = []
        list_std = []
        
        # 用于计算该核的全局 min/max (为了色标统一)
        vals_pred = []
        vals_err = []
        vals_std = []

        for day in day_range:
            is_day = (pack['X_test'][:,2] == day)
            
            # 初始化空画布 (NaN)
            grid_pred = np.full((H,W), np.nan, dtype=np.float32)
            grid_err = np.full((H,W), np.nan, dtype=np.float32)
            grid_std = np.full((H,W), np.nan, dtype=np.float32)

            if np.any(is_day):
                # 坐标映射
                lat = pack['X_test'][is_day,0]
                lon = pack['X_test'][is_day,1]
                def to_idx(val, vmin, vmax, N):
                    return np.clip(((val - vmin) / (vmax - vmin) * (N - 1) + 0.5).astype(int), 0, N - 1)
                lat_idx = to_idx(lat, 35, 40, H)
                lon_idx = to_idx(lon, -115, -105, W)
                
                # 填值
                y_p = yhat_np[is_day]
                y_t = Yte_np[is_day]
                
                grid_pred[lat_idx, lon_idx] = y_p
                
                # Abs Error
                if len(y_t) == len(y_p):
                    # 简单的 mask 处理
                    diff = np.abs(y_p - y_t)
                    grid_err[lat_idx, lon_idx] = diff
                
                # Std
                if ystd_np is not None:
                    grid_std[lat_idx, lon_idx] = ystd_np[is_day]

            list_pred.append(grid_pred)
            list_err.append(grid_err)
            list_std.append(grid_std)
            
            # 收集非空值用于计算 range
            vals_pred.append(grid_pred[~np.isnan(grid_pred)])
            vals_err.append(grid_err[~np.isnan(grid_err)])
            vals_std.append(grid_std[~np.isnan(grid_std)])

        # --- 计算 Scale ---
        def get_range(v_list, p_min=2, p_max=98):
            merged = np.concatenate(v_list)
            if len(merged) == 0: return 0, 1
            return np.nanpercentile(merged, p_min), np.nanpercentile(merged, p_max)

        vp_min, vp_max = get_range(vals_pred)
        ve_min, ve_max = get_range(vals_err, 0, 98) # Error 从 0 开始
        vs_min, vs_max = get_range(vals_std, 2, 98)

        # --- 绘图 1: Prediction ---
        plot_grid_for_kernel(tag, list_pred, "Prediction (All Days)", 
                             'plasma', vp_min, vp_max, f"{tag}_prediction_all_days.png")
        
        # --- 绘图 2: Absolute Error ---
        plot_grid_for_kernel(tag, list_err, "Spatial Absolute Error (All Days)", 
                             'coolwarm', 0, ve_max, f"{tag}_absolute_error_all_days.png")
        
        # --- 绘图 3: Uncertainty (如果存在) ---
        if ystd_np is not None:
            plot_grid_for_kernel(tag, list_std, "Predictive Uncertainty (All Days)", 
                                 'viridis', vs_min, vs_max, f"{tag}_uncertainty_all_days.png")
        else:
            print(f"  -> Skipping Uncertainty plot for {tag} (Not available)")


def main():
    if not os.path.exists(DATA_NPZ):
        print("Data NPZ not found.")
        return
    pack = np.load(DATA_NPZ)
    day_range = list(range(1, int(pack['T'])+1))

    # 指定的 11 个模型
    TARGET_ORDER = [
        'idw', 'n100_seed2', 
        'matern52', 'rbf', 'spacextime',
        'matern52_elev', 'rbf_elev', 'spacextime_elev',
        'matern52_sharedkernel_temp_elev', 'rbf_sharedkernel_temp_elev', 'spacextime_sharedkernel_temp_elev'
    ]
    
    # 过滤存在的文件
    available_tags = []
    for fn in os.listdir(RESULTS_DIR):
        if fn.startswith('pred_'): available_tags.append(fn[5:-3])
    
    valid_tags = [t for t in TARGET_ORDER if t in available_tags]
    print(f"Found {len(valid_tags)} valid kernels to process.")

    process_kernels(valid_tags, pack, day_range)
    print("\nAll Done! Check Figures/Grid_Summaries folder.")

if __name__ == '__main__':
    main()