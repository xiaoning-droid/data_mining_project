import os
import time
import json
import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

DATA_PATH = r"C:\Users\Xiaonongzi\Desktop\CIVE 650\Course project\Figures\data_prep.npz"
OUT_DIR   = r"C:\Users\Xiaonongzi\Desktop\CIVE 650\Project\Results"
os.makedirs(OUT_DIR, exist_ok=True)

data = np.load(DATA_PATH)
X_train = data["X_train"]
y_train = data["y_train"]
X_test  = data["X_test"]
y_test  = data["y_test"]

print("Train shape:", X_train.shape, y_train.shape)
print("Test  shape:", X_test.shape,  y_test.shape)

n_list   = [100, 200]      
seed_list = [0, 1, 2]       

for n_tree in n_list:
    for seed in seed_list:
        print(f"\n=== n_estimators = {n_tree}, random_state = {seed} ===")

        t0 = time.perf_counter()

        rf = RandomForestRegressor(
            n_estimators=n_tree,
            random_state=seed,
            n_jobs=-1
        )

        print("Begin to train Random Forest Regressor")
        rf.fit(X_train, y_train)
        print("Training completed.")

        y_pred = rf.predict(X_test)

        if n_tree == 100 and seed == 2:
            out_pt = os.path.join(OUT_DIR, f"pred_n{n_tree}_seed{seed}.pt")
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
        mse  = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        mae  = float(mean_absolute_error(y_test, y_pred))
        mpae = float(np.mean(np.abs(y_test - y_pred) / (np.abs(y_test) + 1e-6)) * 100.0)
        r2   = float(r2_score(y_test, y_pred))
        elapsed = float(time.perf_counter() - t0)

        metrics = {
            "n_estimators": n_tree,
            "random_state": seed,
            "RMSE": rmse,
            "MAE": mae,
            "MPAE": mpae,
            "R2": r2,
            "Time": elapsed
        }

        filename = f"metrics_n{n_tree}_seed{seed}.json"
        if not filename.startswith("metrics_"):
            filename = "metrics_" + filename
        if not filename.endswith(".json"):
            filename = filename + ".json"

        save_path = os.path.join(OUT_DIR, filename)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print("Saved metrics to:", save_path)
        print(f"R2   = {r2:.4f}")
        print(f"RMSE = {rmse:.4f}")
        print(f"MAE  = {mae:.4f}")
        print(f"MPAE = {mpae:.2f}%")
        print(f"Time = {elapsed:.3f} s")
