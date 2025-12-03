import os
import sys
import subprocess
import json
from typing import List, Tuple

# === 修改这里：与你的 gp_train_svgp.py 一致的路径 ===
PROJECT_DIR = r"C:\Users\Xiaonongzi\Desktop\CIVE 650\Project"
CODE_DIR    = os.path.join(PROJECT_DIR, "Code")
RESULTS_DIR = os.path.join(PROJECT_DIR, "Results")

# === 选择要做超参数搜索的 kernel（代表性三个） ===
KERNELS = [
    "rbf",
    "matern52",
    "spaceXtime+linear",   # 注意：这里用的是和你平时命令行一样的写法
]

# === 选择要尝试的诱导点数量 M ===
M_VALUES = [200, 400, 800, 1200]

# === 训练的一些固定参数，可以根据需要改 ===
EPOCHS = 30
BATCH  = 8192
JITTER = 1e-3

# === 这里复制一份简单的 canonicalize + sanitize，用来推 tag 和 metrics 文件名 ===
import re

def _canonicalize_kernel_name(name: str):
    s = name.strip().lower()
    s = s.replace(' ', '')
    s = s.replace('(', '').replace(')', '')
    s = s.replace('/', '_')
    s = s.replace('exponential', 'exp').replace('matern12', 'exp')
    s = s.replace('times', '*').replace('x', '*')

    if 'spacextime' in s or ('space' in s and 'time' in s):
        return 'spacextime+linear', {}

    if 'rq_like' in s or 'rqlike' in s:
        return 'rq_like', {}

    if 'periodic' in s and 'time' in s and ('*' in s or 'periodictime' in s):
        return 'periodic*time', {}

    if 'rbf' in s and '+linear' in s:
        return 'rbf+linear', {}

    if 'rbf' in s and 'pluslinear' in s:
        return 'rbf+linear', {}

    if ('exp' in s or 'matern0.5' in s) and ('+linear' in s or 'pluslinear' in s):
        return 'exp+linear', {}

    if s in ('rbf', 'rbfkernel'):
        return 'rbf', {}
    if 'matern32' in s or 'matern3' in s:
        return 'matern32', {}
    if 'matern52' in s or 'matern5' in s:
        return 'matern52', {}
    if s == 'linear' or s.endswith('linear'):
        return 'linear', {}

    raise ValueError(f"Unknown or unsupported kernel name: {name}")


def sanitize(s: str) -> str:
    return s.replace('*', 'x').replace('+', 'plus').replace(' ', '_').replace('/', '_')


def kernel_to_tag(kernel_name: str) -> str:
    """与训练脚本中生成 pred_*.pt / metrics_*.json 的 tag 规则保持一致。"""
    key, params = _canonicalize_kernel_name(kernel_name)
    tag = sanitize(key)
    return tag


def run_single_train(kernel: str, m: int) -> None:
    """
    调用 gp_train_svgp.py 训练单个 (kernel, M) 组合。
    """
    cmd = [
        sys.executable,                 # 等价于 python.exe
        os.path.join(CODE_DIR, "gp_train_svgp.py"),
        "--kernel", kernel,
        "--m", str(m),
        "--epochs", str(EPOCHS),
        "--batch", str(BATCH),
        "--jitter", str(JITTER),
        "--no-hyper",         # 不保存 raw_hyperparams，让 metrics 文件更干净
    ]

    print("=" * 80)
    print(f"Running: kernel={kernel}, M={m}")
    print("Command:", " ".join(cmd))
    print("=" * 80)
    # check=True：如果训练脚本出错，这里会报错并停止，方便你排查
    subprocess.run(cmd, check=True)


def load_metrics_for_kernel(kernel: str) -> dict:
    """
    读取指定 kernel 最近一次训练产生的 metrics_*.json 文件。
    注意：当前版本 metrics 文件名里不区分 M，
    但 metrics 内容内部有 "m" 字段记录诱导点数量。
    """
    tag = kernel_to_tag(kernel)
    metrics_path = os.path.join(RESULTS_DIR, f"metrics_{tag}.json")
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    # 确保当前工作目录是 CODE_DIR，这样相对路径更安全
    os.chdir(CODE_DIR)
    print("Working directory:", os.getcwd())
    print("Project dir:", PROJECT_DIR)
    print("Results dir:", RESULTS_DIR)
    print()

    summary: List[Tuple[str, int, float, float, float, float]] = []
    #           kernel, M, rmse, r2, crps, train_time

    for k in KERNELS:
        for m in M_VALUES:
            # 1) 训练
            run_single_train(k, m)

            # 2) 读 metrics
            try:
                metrics = load_metrics_for_kernel(k)
            except Exception as e:
                print(f"[Warning] Failed to load metrics for kernel={k}, M={m}: {e}")
                continue

            rmse = metrics.get("rmse", float("nan"))
            r2   = metrics.get("r2", float("nan"))
            crps = metrics.get("crps", float("nan"))
            train_time = metrics.get("train_time_sec", float("nan"))
            m_recorded = metrics.get("m", None)

            print(f">> Metrics for kernel={k}, M={m} (recorded m={m_recorded}):")
            print(f"   RMSE = {rmse:.4f}, R2 = {r2:.4f}, CRPS = {crps:.4f}, time = {train_time:.1f}s")
            print()

            summary.append((k, m_recorded, rmse, r2, crps, train_time))

    # 3) 打印一个简洁的汇总表（方便 copy 到报告或手动整理）
    if summary:
        print("\n" + "#" * 80)
        print("SUMMARY TABLE (Kernel, M, RMSE, R2, CRPS, TrainTime[s])")
        print("#" * 80)
        for (k, m, rmse, r2, crps, t) in summary:
            print(f"{k:20s}  M={m:4d}  RMSE={rmse:8.4f}  R2={r2:8.4f}  CRPS={crps:8.4f}  time={t:7.1f}s")
    else:
        print("No metrics collected. Please check if training or metrics loading failed.")


if __name__ == "__main__":
    main()
