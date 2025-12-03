import os
import numpy as np
import torch
import imageio.v2 as imageio
import matplotlib

# === 强制使用非交互式后端 Agg ===
matplotlib.use('Agg') 

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colorbar
import matplotlib as mpl
from matplotlib.colors import Normalize
from PIL import Image, ImageDraw, ImageFont

# Paths
RESULTS_DIR = r"C:\Users\Xiaonongzi\Desktop\CIVE 650\Project\Results"
FIG_DIR = r"C:\Users\Xiaonongzi\Desktop\CIVE 650\Project\Figures"
os.makedirs(FIG_DIR, exist_ok=True)

DATA_NPZ = os.path.join(RESULTS_DIR, "data_prep.npz")

# Config
MAX_KERNELS = 17 
GRID_COLS = 4 
CELL_W = 200
CELL_H = 100
TITLE_H = 30        # 标题区域高度
GIF_DURATION = 0.6 

# === 显示名称映射 ===
DISPLAY_NAMES = {
    'n100_seed2': 'Random Forest',
    'idw': 'IDW',
    'linear': 'Linear',
    'matern32': 'Matern32',
    'matern52': 'Matern52',
    'rbf': 'RBF',
    'rq_like': 'RQ Like',
    'rbfpluslinear': 'RBF + Linear',
    'periodicxtime': 'Periodic x Time',
    'spacextime': 'Space x Time',
    'matern52_elev': 'matern52_Elev',
    'rbf_elev': 'RBF_Elev',
    'spacextime_elev': 'SpacexTime_Elev',
    'matern52_sharedkernel_temp_elev': 'matern52_multi',
    'rbf_sharedkernel_temp_elev': 'rbf_multi',
    'spacextime_sharedkernel_temp_elev': 'spacextime_multi',
}

def get_display_name(tag):
    return DISPLAY_NAMES.get(tag, tag)

def discover_kernel_tags():
    if not os.path.exists(RESULTS_DIR):
        print(f"Error: Directory not found {RESULTS_DIR}")
        return []
    
    existing_tags = set()
    for fn in os.listdir(RESULTS_DIR):
        if fn.startswith('pred_') and fn.endswith('.pt'):
            tag = fn[len('pred_'):-3]
            existing_tags.add(tag)
    
    # === 强制排序列表 ===
    TARGET_ORDER = [
        'n100_seed2',                        # 1
        'idw',                               # 2
        'linear',                            # 3
        'matern32',                          # 4
        'matern52',                          # 5
        'rbf',                               # 6
        'rq_like',                           # 7
        'rbfpluslinear',                     # 8
        'periodicxtime',                     # 9
        'spacextime',                        # 10
        'matern52_elev',                     # 11
        'rbf_elev',                          # 12
        'spacextime_elev',                   # 13
        'matern52_sharedkernel_temp_elev',   # 14
        'rbf_sharedkernel_temp_elev',        # 15
        'spacextime_sharedkernel_temp_elev'  # 16
    ]
    
    final_tags = []
    for tag in TARGET_ORDER:
        if tag in existing_tags:
            final_tags.append(tag)
        else:
            print(f"Warning: Expected kernel '{tag}' not found in Results folder.")
            
    return final_tags


def load_pred(tag):
    pth = os.path.join(RESULTS_DIR, f'pred_{tag}.pt')
    d = torch.load(pth, map_location='cpu', weights_only=False)
    return d


def choose_prediction_array(pred_pack):
    candidates = ['yhat', 'y_hat', 'y_pred', 'pred', 'preds', 'ymean', 'mean', 'mu', 'Yte']
    if hasattr(pred_pack, 'keys'):
        for k in candidates:
            if k in pred_pack:
                return pred_pack[k]
        for k in pred_pack.keys():
            ks = str(k)
            if ks.endswith('_hat') or ks.lower().endswith('hat'):
                return pred_pack[k]
    if isinstance(pred_pack, (np.ndarray,)):
        return pred_pack
    return None


def get_cmap_safe(cmap_name):
    try:
        return mpl.colormaps[cmap_name]
    except AttributeError:
        return plt.get_cmap(cmap_name)


def array_to_rgba(arr, cmap_name='plasma', vmin=None, vmax=None):
    cmap = get_cmap_safe(cmap_name)
    if vmin is None: vmin = np.nanmin(arr)
    if vmax is None: vmax = np.nanmax(arr)
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    mapped = cmap(norm(arr)) 
    mask = np.isnan(arr)
    mapped[mask] = [1.0, 1.0, 1.0, 1.0] 
    rgba_uint8 = (mapped * 255).astype(np.uint8)
    return rgba_uint8


# === 核心修改: 在子图之间增加间距 (GAP) ===
def make_grid_image(arrays, cols, cell_w, cell_h, cmap='plasma', vmin=None, vmax=None, is_error=False, eabs=None, labels=None):
    n = len(arrays)
    cols = min(cols, n)
    rows = (n + cols - 1) // cols
    
    GAP = 15  # <--- 定义子图之间的间距 (像素)
    
    # 计算画布总宽: 列宽总和 + (列数-1)个间隙
    # 如果 cols=1, (cols-1)*GAP = 0, 逻辑依然成立
    grid_w = cols * cell_w + (cols - 1) * GAP
    
    # 计算画布总高: 行高总和 (含标题) + (行数-1)个间隙
    grid_h = rows * (cell_h + TITLE_H) + (rows - 1) * GAP
    
    canvas = Image.new('RGB', (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("times.ttf", 20)
    except IOError:
        try:
            font = ImageFont.truetype("Times New Roman.ttf", 20)
        except IOError:
            font = ImageFont.load_default()

    for i, arr in enumerate(arrays):
        r = i // cols
        c = i % cols
        
        if is_error and eabs is not None:
            vvmin, vvmax = -eabs, eabs
        else:
            vvmin, vvmax = vmin, vmax

        rgba = array_to_rgba(arr, cmap_name=cmap, vmin=vvmin, vmax=vvmax)
        img = Image.fromarray(rgba)
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        img = img.resize((cell_w, cell_h), Image.LANCZOS)
        
        # === 计算坐标: 加上间隙 GAP ===
        x_base = c * (cell_w + GAP)
        y_base = r * (cell_h + TITLE_H + GAP)
        # ============================
        
        # 绘制标题
        if labels and i < len(labels):
            raw_tag = labels[i]
            display_text = get_display_name(raw_tag) 
            draw.text((x_base + 5, y_base + 5), display_text, fill="black", font=font)

        # 粘贴图片
        canvas.paste(img, (x_base, y_base + TITLE_H))

        # 给图片画边框 (BBox)
        rect_coords = [
            x_base, 
            y_base + TITLE_H, 
            x_base + cell_w - 1, 
            y_base + TITLE_H + cell_h - 1
        ]
        draw.rectangle(rect_coords, outline="black", width=1)
        
    return canvas


def add_colorbar_to_image(pil_img, cmap_name, vmin, vmax, is_error=False, eabs=None):
    fig_w_inch = 1.0  
    fig_h_inch = pil_img.height / 100.0
    
    fig = plt.figure(figsize=(fig_w_inch, fig_h_inch), dpi=100)
    ax = fig.add_axes([0.1, 0.05, 0.25, 0.9]) 

    cmap = get_cmap_safe(cmap_name)
    if is_error and eabs is not None:
        norm = Normalize(vmin=-eabs, vmax=eabs)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    cb = matplotlib.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')
    
    for l in cb.ax.yaxis.get_ticklabels():
        l.set_family("serif")
        l.set_size(10)
        
    fig.canvas.draw()
    
    rgba_buffer = fig.canvas.buffer_rgba()
    cbar_arr = np.asarray(rgba_buffer)
    
    if cbar_arr.ndim == 3 and cbar_arr.shape[2] == 4:
        cbar_arr = cbar_arr[:, :, :3]
    
    plt.close(fig)
    
    cbar_img = Image.fromarray(cbar_arr)

    new_width = pil_img.width + cbar_img.width
    new_img = Image.new('RGB', (new_width, pil_img.height), (255, 255, 255))
    new_img.paste(pil_img, (0, 0))
    new_img.paste(cbar_img, (pil_img.width, 0))
    return new_img


def draw_timestamp(img, day):
    # 1. 定义底部“页脚”高度
    FOOTER_HEIGHT = 60  
    
    # 2. 创建新画布：宽度不变，高度增加
    new_width = img.width
    new_height = img.height + FOOTER_HEIGHT
    new_img = Image.new('RGB', (new_width, new_height), (255, 255, 255))
    
    # 3. 将原图贴在顶部
    new_img.paste(img, (0, 0))
    
    draw = ImageDraw.Draw(new_img)
    
    # 4. 设置字号
    FONT_SIZE = 32
    try:
        font = ImageFont.truetype("times.ttf", FONT_SIZE)
    except IOError:
        try:
            font = ImageFont.truetype("Times New Roman.ttf", FONT_SIZE)
        except IOError:
            font = ImageFont.load_default()
            
    text = f"Day {day}"
    
    # 5. 计算文字位置 (在新增加的底部区域居中)
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except AttributeError:
        text_w, text_h = draw.textsize(text, font=font)

    # 横向居中
    x_pos = (new_width - text_w) // 2
    # 纵向：位于原图高度 + 页脚高度的一半，再微调
    y_pos = img.height + (FOOTER_HEIGHT - text_h) // 2 - 5
    
    draw.text((x_pos, y_pos), text, fill="black", font=font)
    
    return new_img


def build_day_frames_for_kernels(tags, pack, day_range):
    H = int(pack['H']); W = int(pack['W'])
    
    per_kernel_preds = {}
    per_kernel_errs = {}
    all_vals = []
    all_eabs = []

    for tag in tags:
        pred_pack = load_pred(tag)
        yhat = choose_prediction_array(pred_pack)
        Yte = None
        if hasattr(pred_pack, 'keys'):
            for k in ['Yte', 'Yte_temp', 'Yte_elev', 'Yte_y', 'Yte_Y']:
                if k in pred_pack:
                    Yte = pred_pack[k]
                    break
        if Yte is None: Yte = pred_pack.get('Yte', pack['y_test'])
        
        yhat_np = np.asarray(yhat) if not isinstance(yhat, torch.Tensor) else yhat.cpu().numpy()
        Yte_np = np.asarray(Yte) if not isinstance(Yte, torch.Tensor) else Yte.cpu().numpy()

        p_list = []
        e_list = []
        for day in day_range:
            is_day = (pack['X_test'][:,2] == day)
            if not np.any(is_day):
                p = np.full((H,W), np.nan, dtype=np.float32)
                e = np.full((H,W), np.nan, dtype=np.float32)
            else:
                lat = pack['X_test'][is_day,0]
                lon = pack['X_test'][is_day,1]
                
                def to_idx(val, vmin, vmax, N):
                    return np.clip(((val - vmin) / (vmax - vmin) * (N - 1) + 0.5).astype(int), 0, N - 1)
                
                lat_idx = to_idx(lat, 35, 40, H)
                lon_idx = to_idx(lon, -115, -105, W)
                
                p = np.full((H,W), np.nan, dtype=np.float32)
                yhat_day = yhat_np[is_day]
                Yte_day = Yte_np[is_day]
                
                p[lat_idx, lon_idx] = yhat_day
                e_temp = p - np.full((H,W), np.nan, dtype=np.float32)
                e_temp[lat_idx, lon_idx] = yhat_day - Yte_day

                # Flip UD
                p = np.flipud(p)
                e_temp = np.flipud(e_temp)
                e = e_temp

            p_list.append(p)
            e_list.append(e)
            
            if not np.all(np.isnan(p)): all_vals.append(p[~np.isnan(p)])
            if not np.all(np.isnan(e)): all_eabs.append(np.nanmax(np.abs(e)))

        per_kernel_preds[tag] = p_list
        per_kernel_errs[tag] = e_list

    if all_vals:
        merged = np.concatenate(all_vals)
        vmin = float(np.nanpercentile(merged,2))
        vmax = float(np.nanpercentile(merged,98))
    else:
        vmin, vmax = 0.0, 1.0
    global_eabs = float(np.nanmax(all_eabs)) if all_eabs else 1e-6

    pred_day_frames = []
    err_day_frames = []
    
    print(f"Building {len(day_range)} frames with colorbars...")
    
    for i, day in enumerate(day_range):
        arrays_pred = [per_kernel_preds[tag][i] for tag in tags]
        arrays_err = [per_kernel_errs[tag][i] for tag in tags]
        
        # 1. 生成网格 (Make Grid Image now supports Gaps)
        pred_img = make_grid_image(arrays_pred, GRID_COLS, CELL_W, CELL_H, cmap='plasma', vmin=vmin, vmax=vmax, is_error=False, labels=tags)
        err_img = make_grid_image(arrays_err, GRID_COLS, CELL_W, CELL_H, cmap='coolwarm', is_error=True, eabs=global_eabs, labels=tags)
        
        # 2. 加色带
        pred_img_with_cbar = add_colorbar_to_image(pred_img, 'plasma', vmin, vmax)
        err_img_with_cbar = add_colorbar_to_image(err_img, 'coolwarm', 0, 0, is_error=True, eabs=global_eabs)

        # 3. 打上时间戳 (在新增加的底部)
        pred_img_with_cbar = draw_timestamp(pred_img_with_cbar, day)
        err_img_with_cbar = draw_timestamp(err_img_with_cbar, day)

        pred_day_frames.append(pred_img_with_cbar)
        err_day_frames.append(err_img_with_cbar)
        
    return pred_day_frames, err_day_frames, vmin, vmax, global_eabs


def save_gif_from_pil_images(img_list, out_path, duration=0.6):
    if not img_list: return
    arrs = [np.asarray(im) for im in img_list]
    imageio.mimsave(out_path, arrs, duration=duration)


def save_day31_composite(pred_frames, err_frames, day_index, out_dir=FIG_DIR):
    if day_index >= len(pred_frames): return None, None
    pred_img = pred_frames[day_index]
    err_img = err_frames[day_index]
    pth1 = os.path.join(out_dir, f'all_pred_day{day_index+1:02d}.png')
    pth2 = os.path.join(out_dir, f'all_err_day{day_index+1:02d}.png')
    pred_img.save(pth1)
    err_img.save(pth2)
    print('Saved Day31 composites:', pth1, pth2)
    return pth1, pth2


def save_single_kernel_day31_images(tags, pack, vmin, vmax, global_eabs, day=31):
    H = int(pack['H']); W = int(pack['W'])
    
    try:
        font = ImageFont.truetype("times.ttf", 24)
    except:
        try: font = ImageFont.truetype("Times New Roman.ttf", 24)
        except: font = ImageFont.load_default()

    for tag in tags:
        pred_pack = load_pred(tag)
        yhat = choose_prediction_array(pred_pack)
        Yte = pred_pack.get('Yte', pack['y_test'])
        
        yhat_np = np.asarray(yhat) if not isinstance(yhat, torch.Tensor) else yhat.cpu().numpy()
        Yte_np = np.asarray(Yte) if not isinstance(Yte, torch.Tensor) else Yte.cpu().numpy()
        
        is_day = (pack['X_test'][:,2] == day)
        
        pred_day = np.full((H,W), np.nan, dtype=np.float32)
        true_day = np.full((H,W), np.nan, dtype=np.float32)

        if np.any(is_day):
            lat = pack['X_test'][is_day,0]
            lon = pack['X_test'][is_day,1]
            def to_idx(val, vmin, vmax, N):
                return np.clip(((val - vmin) / (vmax - vmin) * (N - 1) + 0.5).astype(int), 0, N - 1)
            lat_idx = to_idx(lat, 35, 40, H)
            lon_idx = to_idx(lon, -115, -105, W)
            
            pred_day[lat_idx, lon_idx] = yhat_np[is_day]
            true_day[lat_idx, lon_idx] = Yte_np[is_day]

        # 翻转
        pred_day = np.flipud(pred_day)
        true_day = np.flipud(true_day)
        err_day = pred_day - true_day

        pred_rgb = array_to_rgba(pred_day, cmap_name='plasma', vmin=vmin, vmax=vmax)
        err_rgb = array_to_rgba(err_day, cmap_name='coolwarm', vmin=-global_eabs, vmax=global_eabs)
        
        pred_im = Image.fromarray(pred_rgb).convert('RGB').resize((CELL_W*2, CELL_H*2), Image.LANCZOS)
        err_im = Image.fromarray(err_rgb).convert('RGB').resize((CELL_W*2, CELL_H*2), Image.LANCZOS)
        
        pred_im = add_colorbar_to_image(pred_im, 'plasma', vmin, vmax)
        err_im = add_colorbar_to_image(err_im, 'coolwarm', 0, 0, is_error=True, eabs=global_eabs)

        display_name = get_display_name(tag)
        
        d1 = ImageDraw.Draw(pred_im)
        d1.text((10,10), f"{display_name} (Pred)", fill="black", stroke_fill="white", stroke_width=2, font=font)
        # 添加边框
        d1.rectangle([0, 0, pred_im.width-1, pred_im.height-1], outline="black", width=1)

        d2 = ImageDraw.Draw(err_im)
        d2.text((10,10), f"{display_name} (Err)", fill="black", stroke_fill="white", stroke_width=2, font=font)
        # 添加边框
        d2.rectangle([0, 0, err_im.width-1, err_im.height-1], outline="black", width=1)
        
        # === 间距逻辑 GAP ===
        GAP = 20
        w = pred_im.width + GAP + err_im.width
        h = max(pred_im.height, err_im.height)
        canvas = Image.new('RGB', (w, h), (255,255,255))
        
        canvas.paste(pred_im, (0,0))
        canvas.paste(err_im, (pred_im.width + GAP, 0))
        # ==================
        
        out_p = os.path.join(FIG_DIR, f'{tag}_day{day}_comparison.png')
        canvas.save(out_p)


def main():
    tags = discover_kernel_tags()
    if len(tags) == 0:
        print('No pred_*.pt files found or none matched list. Exiting.')
        return
    
    print('Using kernel tags (Ordered):')
    for t in tags:
        print(f" - {t} -> {get_display_name(t)}")

    if not os.path.exists(DATA_NPZ):
        print("Data NPZ not found.")
        return
        
    pack = np.load(DATA_NPZ)
    day_range = list(range(1, int(pack['T'])+1))

    pred_frames, err_frames, vmin, vmax, global_eabs = build_day_frames_for_kernels(tags, pack, day_range)

    pred_gif = os.path.join(FIG_DIR, 'all_kernels_pred_days1-31.gif')
    save_gif_from_pil_images(pred_frames, pred_gif, duration=GIF_DURATION)
    print('Saved GIF:', pred_gif)

    err_gif = os.path.join(FIG_DIR, 'all_kernels_err_days1-31.gif')
    save_gif_from_pil_images(err_frames, err_gif, duration=GIF_DURATION)
    print('Saved GIF:', err_gif)

    day31_idx = len(day_range) - 1
    save_day31_composite(pred_frames, err_frames, day31_idx)
    
    print("Generating Day 31 comparison images for all kernels...")
    save_single_kernel_day31_images(tags, pack, vmin, vmax, global_eabs, day=31)

    print('\nDone.')

if __name__ == '__main__':
    main()