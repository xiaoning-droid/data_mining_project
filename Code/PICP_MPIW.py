import os
import numpy as np
import torch
import pandas as pd

RESULTS_DIR = r"C:\Users\Xiaonongzi\Desktop\CIVE 650\Project\Results"
DATA_NPZ = os.path.join(RESULTS_DIR, "data_prep.npz")
Z_SCORE = 1.96 

TARGET_ORDER = [
    'n100_seed2',                        # Random Forest
    'idw',                               # IDW
    'linear',                            # Linear
    'matern32',                          # Matern32
    'matern52',                          # Matern52
    'rbf',                               # RBF
    'rq_like',                           # RQ Like
    'rbfpluslinear',                     # RBF + Linear
    'periodicxtime',                     # Periodic x Time
    'spacextime',                        # Space x Time
    'matern52_elev',                     # Matern52 (Elev)
    'rbf_elev',                          # RBF (Elev)
    'spacextime_elev',                   # SpacexTime (Elev)
    'matern52_sharedkernel_temp_elev',   # Matern52 (Multi)
    'rbf_sharedkernel_temp_elev',        # RBF (Multi)
    'spacextime_sharedkernel_temp_elev'  # SpacexTime (Multi)
]

def load_pred(tag):
    pth = os.path.join(RESULTS_DIR, f'pred_{tag}.pt')
    try:
        d = torch.load(pth, map_location='cpu', weights_only=False)
        return d
    except Exception as e:
        print(f"Error loading {tag}: {e}")
        return None

def get_data_arrays(pred_pack, pack):
    yhat = None
    for k in ['yhat', 'mean', 'pred', 'mu', 'y_pred', 'temp_hat']:
        if k in pred_pack: 
            yhat = pred_pack[k]
            break
            
    ystd = None
    for k in ['ystd', 'std', 'sigma', 'uncertainty', 'y_std', 'temp_std']:
        if k in pred_pack: 
            ystd = pred_pack[k]
            break
            
    yte = None
    if 'Yte_temp' in pred_pack:
        yte = pred_pack['Yte_temp']
    elif 'Yte' in pred_pack:
        yte = pred_pack['Yte']
    else:
        yte = pack['y_test']

    def to_np(x):
        if x is None: return None
        if isinstance(x, torch.Tensor): return x.cpu().numpy()
        return np.asarray(x)
        
    return to_np(yhat), to_np(ystd), to_np(yte)

def calculate_metrics(tag, y_pred, y_std, y_true):
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if y_std is not None:
        mask = mask & ~np.isnan(y_std)
    
    y_pred_clean = y_pred[mask]
    y_true_clean = y_true[mask]
    
    rmse = np.sqrt(np.mean((y_pred_clean - y_true_clean)**2))
    
    if y_std is None:
        return {
            "Model": tag, 
            "RMSE": rmse, 
            "PICP": np.nan, 
            "MPIW": np.nan,
            "Has_UQ": False
        }
    
    y_std_clean = y_std[mask]

    lower = y_pred_clean - Z_SCORE * y_std_clean
    upper = y_pred_clean + Z_SCORE * y_std_clean
    

    is_in_interval = (y_true_clean >= lower) & (y_true_clean <= upper)
    picp = np.mean(is_in_interval)
    
    width = upper - lower
    mpiw = np.mean(width)
    
    return {
        "Model": tag, 
        "RMSE": rmse, 
        "PICP": picp, 
        "MPIW": mpiw,
        "Has_UQ": True
    }

def main():
    if not os.path.exists(DATA_NPZ):
        print(f"Data file not found: {DATA_NPZ}")
        return
    pack = np.load(DATA_NPZ)
    
    results = []
    print(f"{'Model':<35} | {'RMSE':<8} | {'PICP (95%)':<10} | {'MPIW':<10}")
    print("-" * 75)
    
    for tag in TARGET_ORDER:
        pred_pack = load_pred(tag)
        if not pred_pack:
            continue
            
        y_pred, y_std, y_true = get_data_arrays(pred_pack, pack)
        
        if y_pred is None:
            print(f"Skipping {tag} (No prediction data)")
            continue
            
        m = calculate_metrics(tag, y_pred, y_std, y_true)
        results.append(m)
        
        picp_str = f"{m['PICP']:.4f}" if m['Has_UQ'] else "N/A"
        mpiw_str = f"{m['MPIW']:.4f}" if m['Has_UQ'] else "N/A"
        print(f"{tag:<35} | {m['RMSE']:.4f}   | {picp_str:<10} | {mpiw_str:<10}")

    df = pd.DataFrame(results)
    df = df[['Model', 'RMSE', 'PICP', 'MPIW']]
    output_path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    df.to_csv(output_path, index=False)
    print(f"\nSummary saved to: {output_path}")

if __name__ == '__main__':
    main()