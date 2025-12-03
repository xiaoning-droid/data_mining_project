import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ------------------ paths & config ------------------
RESULTS_DIR = r"C:\Users\Xiaonongzi\Desktop\CIVE 650\Project\Results"
DATA_NPZ    = os.path.join(RESULTS_DIR, "data_prep.npz")

# 这里填你想分析的 kernel tag，比如 "spacextime" 或 "spacextime_sharedkernel_temp_elev"
KERNEL_TAG  = "linear"  # 修改为你想分析的模型标签

# 是否随机抽样一部分点画图（点太多时避免图被刷成一团）
SUBSAMPLE = True
SUBSAMPLE_N = 10000  # 抽样点数


# ------------------ helper: load prediction ------------------
def load_pred(tag: str):
    pth = os.path.join(RESULTS_DIR, f"pred_{tag}.pt")
    if not os.path.exists(pth):
        raise FileNotFoundError(pth)
    try:
        return torch.load(pth, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(pth, map_location="cpu")
    except RuntimeError:
        return torch.load(pth, map_location="cpu")


# ------------------ main plotting function ------------------
def plot_uncertainty_vs_error(tag: str):
    # 1) load prediction pack
    pred_pack = load_pred(tag)

    # 2) get yhat, y_true (Yte) and predictive std (ystd)
    # keys can be adjusted depending on your file structure
    if "yhat" in pred_pack:
        yhat = pred_pack["yhat"]
    elif "y_hat" in pred_pack:
        yhat = pred_pack["y_hat"]
    else:
        raise KeyError("Cannot find 'yhat' in pred file.")

    if "Yte" in pred_pack:
        y_true = pred_pack["Yte"]
    else:
        # fallback: use data_prep.npz
        pack = np.load(DATA_NPZ)
        y_true = torch.from_numpy(pack["y_test"])

    if "ystd" not in pred_pack:
        raise KeyError("Cannot find 'ystd' in pred file; "
                       "need predictive std to plot uncertainty vs. error.")

    ystd = pred_pack["ystd"]

    # 3) convert to numpy arrays
    yhat  = yhat.detach().cpu().numpy().ravel()
    y_true = y_true.detach().cpu().numpy().ravel() if isinstance(y_true, torch.Tensor) else np.ravel(y_true)
    ystd  = ystd.detach().cpu().numpy().ravel()

    # 4) compute absolute error
    abs_err = np.abs(yhat - y_true)

    # 5) clean NaNs / infs
    mask = np.isfinite(abs_err) & np.isfinite(ystd)
    abs_err = abs_err[mask]
    ystd    = ystd[mask]

    # optional subsampling for plotting
    if SUBSAMPLE and abs_err.size > SUBSAMPLE_N:
        idx = np.random.choice(abs_err.size, size=SUBSAMPLE_N, replace=False)
        abs_err_plot = abs_err[idx]
        ystd_plot    = ystd[idx]
    else:
        abs_err_plot = abs_err
        ystd_plot    = ystd

    # 6) compute correlation between uncertainty and error
    if abs_err.size > 2:
        r, pval = pearsonr(ystd, abs_err)
        corr_text = f"Pearson r = {r:.3f}, p = {pval:.2e}"
    else:
        corr_text = "Not enough points for correlation."

    # 7) make scatter plot
    plt.figure(figsize=(6, 5))
    plt.scatter(ystd_plot, abs_err_plot, s=6, alpha=0.25)
    plt.xlabel("Predictive Uncertainty (std)")
    plt.ylabel("Absolute Error |y - ŷ|")
    plt.title(f"{tag} kernel: Uncertainty vs. Error\n" + corr_text)
    plt.grid(True, linestyle="--", alpha=0.3)

    # optional: use log scale if tail is heavy
    # plt.yscale("log")

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, f"scatter_uncert_vs_error_{tag}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print("Saved scatter plot to:", out_path)


if __name__ == "__main__":
    plot_uncertainty_vs_error(KERNEL_TAG)

    # 在 compute abs_err 之后加：
    rmse = np.sqrt(np.mean((yhat - y_true)**2))
    mae  = np.mean(abs_err)
    print("RMSE from this script:", rmse)
    print("MAE  from this script:", mae)
