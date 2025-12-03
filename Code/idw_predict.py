import os
import time
import numpy as np
import torch

DATA_NPZ = r"C:\Users\Xiaonongzi\Desktop\CIVE 650\Project\Results\data_prep.npz"
RESULTS_DIR = r"C:\Users\Xiaonongzi\Desktop\CIVE 650\Project\Results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def idw_interp(xy_train, z_train, xy_query, power=2, eps=1e-6):
    """
    Inverse Distance Weighting (IDW) 插值，对空间 (lat, lon) 做插值
    """
    xy_train = np.asarray(xy_train, dtype=float)
    z_train = np.asarray(z_train, dtype=float)
    xy_query = np.asarray(xy_query, dtype=float)

    M = xy_query.shape[0]
    z_query = np.empty(M, dtype=float)

    diff = xy_query[:, None, :] - xy_train[None, :, :]  
    dist = np.sqrt(np.sum(diff ** 2, axis=2)) + eps     

    for i in range(M):
        di = dist[i]  
        zero_mask = di < eps * 10
        if np.any(zero_mask):
            z_query[i] = z_train[zero_mask][0]
        else:
            w = 1.0 / (di ** power)
            w /= w.sum()
            z_query[i] = np.dot(w, z_train)

    return z_query


def main():
    pack = np.load(DATA_NPZ)
    X_train = pack["X_train"]    
    y_train = pack["y_train"]    
    X_test  = pack["X_test"]     
    y_test  = pack["y_test"]    
    H = int(pack["H"])
    W = int(pack["W"])
    T = int(pack["T"])

    print("Loaded:", DATA_NPZ)
    print("X_train:", X_train.shape, "X_test:", X_test.shape)
    print("H, W, T =", H, W, T)

    start_time = time.time()
    yhat = np.empty_like(y_test)

    for day in range(1, T + 1):
        mask_tr = (X_train[:, 2] == day)
        mask_te = (X_test[:, 2] == day)

        n_tr = np.sum(mask_tr)
        n_te = np.sum(mask_te)
        if n_te == 0:
            continue 
        if n_tr == 0:
            print(f"Day {day}: no training points, filling with global mean")
            yhat[mask_te] = y_train.mean()
            continue

        xy_tr = X_train[mask_tr][:, :2]  
        z_tr  = y_train[mask_tr]       
        xy_te = X_test[mask_te][:, :2]   

        print(f"Day {day:2d}: train={n_tr:5d}, test={n_te:5d} -> IDW...")
        yhat_day = idw_interp(xy_tr, z_tr, xy_te, power=2)
        yhat[mask_te] = yhat_day

    mse = np.mean((yhat - y_test) ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(yhat - y_test)))

    ss_res = np.sum((y_test - yhat) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2) + 1e-12 
    r2 = float(1.0 - ss_res / ss_tot)

    eps = 1e-6
    denom = np.maximum(np.abs(y_test), eps)
    mape = float(np.mean(np.abs((y_test - yhat) / denom)) * 100.0)

    train_time_s = float(time.time() - start_time)

    print(f"IDW overall RMSE = {rmse:.4f}, MAE = {mae:.4f}, R2 = {r2:.4f}, MAPE = {mape:.4f}%")
    print(f"IDW total runtime (s) = {train_time_s:.2f}")

    out_pt = os.path.join(RESULTS_DIR, "pred_idw.pt")
    ystd = np.zeros_like(yhat, dtype=np.float32)
    meta = {
        'method': 'idw',
        'created_time': float(time.time())
    }

    torch.save(
        {
            "yhat": torch.from_numpy(yhat.astype(np.float32)),
            "Yte":  torch.from_numpy(y_test.astype(np.float32)),
            "ystd": torch.from_numpy(ystd),
            "mape": mape,
            "train_time_s": train_time_s,
            "meta": meta,
        },
        out_pt,
    )
    print("Saved IDW predictions to:", out_pt)


if __name__ == "__main__":
    main()
