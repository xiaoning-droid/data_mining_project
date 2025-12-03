import os
import time
import json
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# ===== 0. 路径设置 =====
# 注意：data_prep.npz 在 Figures 里，不是在 Results 里
DATA_PATH = r"C:\Users\Xiaonongzi\Desktop\CIVE 650\Course project\Figures\data_prep.npz"
OUT_DIR   = r"C:\Users\Xiaonongzi\Desktop\CIVE 650\Project\Results"
os.makedirs(OUT_DIR, exist_ok=True)

# ===== 1. 读取数据 =====
data = np.load(DATA_PATH)
X_train = data["X_train"]
y_train = data["y_train"]
X_test  = data["X_test"]
y_test  = data["y_test"]

print("Train shape:", X_train.shape, y_train.shape)
print("Test  shape:", X_test.shape,  y_test.shape)

# ===== 2. 想要尝试的树数 和 种子 =====
n_list   = [100, 200]      # 不同的森林树数
seed_list = [0, 1, 2]          # 不同的 random_state

# ===== 3. 双重循环：对每个 (树数, 种子) 组合都跑一遍 =====
for n_tree in n_list:
    for seed in seed_list:
        print(f"\n=== n_estimators = {n_tree}, random_state = {seed} ===")

        t0 = time.perf_counter()

        # 3.1 创建模型
        rf = RandomForestRegressor(
            n_estimators=n_tree,
            random_state=seed,
            n_jobs=-1
        )

        print("Begin to train Random Forest Regressor")
        rf.fit(X_train, y_train)
        print("Training completed.")

        # 3.2 预测
        y_pred = rf.predict(X_test)

        # 如果是用户指定的配置 (n=100, seed=2)，保存预测为 pred_n100_seed2.pt 供后续绘图使用
        if n_tree == 100 and seed == 2:
            out_pt = os.path.join(OUT_DIR, f"pred_n{n_tree}_seed{seed}.pt")
            # ystd: placeholder zeros (no predictive std from RF)
            ystd = np.zeros_like(y_pred, dtype=np.float32)
            meta = {
                'method': 'random_forest',
                'n_estimators': n_tree,
                'random_state': seed,
                'created_time': float(time.time())
            }
            torch.save({
                'yhat': torch.from_numpy(y_pred.astype(np.float32)),
                'Yte': torch.from_numpy(y_test.astype(np.float32)),
                'ystd': torch.from_numpy(ystd),
                'meta': meta,
            }, out_pt)
            print("Saved predictions to:", out_pt)

        # 3.3 计算各种指标
        mse  = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        mae  = float(mean_absolute_error(y_test, y_pred))
        mpae = float(np.mean(np.abs(y_test - y_pred) / (np.abs(y_test) + 1e-6)) * 100.0)
        r2   = float(r2_score(y_test, y_pred))
        elapsed = float(time.perf_counter() - t0)

        # 3.4 把指标装到一个字典里
        metrics = {
            "n_estimators": n_tree,
            "random_state": seed,
            "RMSE": rmse,
            "MAE": mae,
            "MPAE": mpae,
            "R2": r2,
            "Time": elapsed
        }

        # 3.5 生成文件名：metrics_n{树数}_seed{种子}.json
        filename = f"metrics_n{n_tree}_seed{seed}.json"
        if not filename.startswith("metrics_"):
            filename = "metrics_" + filename
        if not filename.endswith(".json"):
            filename = filename + ".json"

        save_path = os.path.join(OUT_DIR, filename)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print("Saved metrics to:", save_path)

        # 3.6 控制台也打印一下
        print(f"R2   = {r2:.4f}")
        print(f"RMSE = {rmse:.4f}")
        print(f"MAE  = {mae:.4f}")
        print(f"MPAE = {mpae:.2f}%")
        print(f"Time = {elapsed:.3f} s")
