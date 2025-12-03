import os, json, time, math, argparse, numpy as np, torch, gpytorch
import matplotlib.pyplot as plt
import shutil
from datetime import datetime
import re

OUT_DIR = r"C:\Users\Xiaonongzi\Desktop\CIVE 650\Project\Results"
os.makedirs(OUT_DIR, exist_ok=True)
DATA_NPZ = os.path.join(OUT_DIR, "data_prep.npz")
ELEV_NPY = r"C:\Users\Xiaonongzi\Desktop\CIVE 650\Project\course_project_data\data\elev_grid_100x200.npy"

def get_elevation_vector(X_raw, elev_grid, H, W):
    rows = np.clip(X_raw[:, 0].astype(int), 0, H - 1)
    cols = np.clip(X_raw[:, 1].astype(int), 0, W - 1)
    elev_vals = elev_grid[rows, cols].astype(np.float32)
    return elev_vals

def _canonicalize_kernel_name(name: str):
    s = name.strip().lower()
    s = s.replace(' ', '')
    s = s.replace('(', '').replace(')', '')
    s = s.replace('/', '')
    s = s.replace('×', 'x')
    s = s.replace('times', 'x')

    # RBF
    if 'rbf' in s or 'rbfkernel' in s:
        return 'rbf', {}

    # Matern 5/2
    if 'matern52' in s or 'matern5/2' in s or s == 'matern5' or s == 'matern':
        return 'matern52', {}

    # Space × Time
    if 'spacextime' in s:
        return 'spacextime', {}
    if ('space' in s) and ('time' in s):
        return 'spacextime', {}

    raise ValueError(f"Unknown or unsupported kernel name (only RBF, Matern52, SpaceXTime allowed): {name}")


def make_kernel(name: str, D: int):

    key, params = _canonicalize_kernel_name(name)

    if key == 'rbf':
        return gpytorch.kernels.RBFKernel(ard_num_dims=D)

    if key == 'matern52':
        return gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=D)

    if key == 'spacextime':
        if D < 3:
            raise ValueError("SpaceXTime kernel expects at least 3 input dims: row, col, time")

        SPACE = gpytorch.kernels.MaternKernel(
            nu=1.5, ard_num_dims=2, active_dims=[0, 1]
        )
        TIME  = gpytorch.kernels.RBFKernel(
            ard_num_dims=1, active_dims=[2]
        )
        return SPACE * TIME

    raise ValueError(f"Unknown canonical kernel key: {key}")


# ----------------- model utils -----------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    try:
        devname = torch.cuda.get_device_name(0)
    except Exception:
        devname = 'cuda'
    print(f"Using GPU for training: {devname}")
    torch.backends.cudnn.benchmark = True
else:
    print("CUDA not available — training will run on CPU")


class SVGPModel(gpytorch.models.ApproximateGP):

    def __init__(self, inducing_points, base_kernel):
        qdist  = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        qstrat = gpytorch.variational.VariationalStrategy(
            self, inducing_points, qdist, learn_inducing_locations=True
        )
        super().__init__(qstrat)

        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


def gaussian_crps(mu, sigma, y, eps=1e-6):
    sigma = torch.clamp(sigma, min=eps)
    z = (y - mu) / sigma
    phi = torch.exp(-0.5 * z**2) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1 + torch.erf(z / math.sqrt(2.0)))
    return torch.mean(sigma * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi)))


def make_Z(Xn, M):
    idx = torch.randperm(Xn.size(0))[:M]
    return Xn[idx].clone().to(device)


def sanitize(s):
    return s.replace('*', 'x').replace('+', 'plus').replace(' ', '_').replace('/', '_')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", type=str, required=True,
                    help="One of: RBF, Matern52, SpaceXTime")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--m", type=int, default=800)
    ap.add_argument("--jitter", type=float, default=1e-3)
    ap.add_argument("--maxn", type=int, default=120000)
    ap.add_argument("--no-hyper", action='store_true')
    ap.add_argument("--backup", action='store_true')
    args = ap.parse_args()

    pack = np.load(DATA_NPZ)
    X_train = pack["X_train"]; y_train_temp = pack["y_train"]
    X_test  = pack["X_test"];  y_test_temp  = pack["y_test"]
    Ytemp_mean  = pack["Y_mean"].item(); Ytemp_std = pack["Y_std"].item()
    H = int(pack["H"]); W = int(pack["W"]); T = int(pack["T"])

    elev_grid = np.load(ELEV_NPY)   
    elev_train = get_elevation_vector(X_train, elev_grid, H, W)  
    elev_test  = get_elevation_vector(X_test,  elev_grid, H, W)  

    X_mean = X_train.mean(axis=0, keepdims=True)
    X_std  = X_train.std(axis=0, keepdims=True) + 1e-8

    Xtr = torch.from_numpy(((X_train - X_mean) / X_std).astype(np.float32))
    Xte = torch.from_numpy(((X_test  - X_mean) / X_std).astype(np.float32))


    ytr_temp_n = ((y_train_temp - Ytemp_mean) / Ytemp_std).astype(np.float32)
    yte_temp   = y_test_temp.astype(np.float32)
    Yelev_mean = float(elev_train.mean())
    Yelev_std  = float(elev_train.std() + 1e-8)
    ytr_elev_n = ((elev_train - Yelev_mean) / Yelev_std).astype(np.float32)
    yte_elev   = elev_test.astype(np.float32)

    Ytr_temp = torch.from_numpy(ytr_temp_n).reshape(-1)  # (N,)
    Ytr_elev = torch.from_numpy(ytr_elev_n).reshape(-1)  # (N,)

    Yte_temp = torch.from_numpy(yte_temp.astype(np.float32)).reshape(-1)
    Yte_elev = torch.from_numpy(yte_elev.astype(np.float32)).reshape(-1)

    if Xtr.shape[0] > args.maxn:
        idx = torch.randperm(Xtr.shape[0])[:args.maxn]
        Xtr, Ytr_temp, Ytr_elev = Xtr[idx], Ytr_temp[idx], Ytr_elev[idx]

    D = Xtr.shape[1]

    base_kernel_temp = make_kernel(args.kernel, D).to(device)
    base_kernel_elev = make_kernel(args.kernel, D).to(device)

    Z = make_Z(Xtr, args.m)   

    model_temp = SVGPModel(Z, base_kernel_temp).to(device)
    model_elev = SVGPModel(Z, base_kernel_elev).to(device)

    key, _ = _canonicalize_kernel_name(args.kernel)
    if key in ("rbf", "matern52"):     
        model_elev.covar_module.base_kernel.raw_lengthscale = \
            model_temp.covar_module.base_kernel.raw_lengthscale
        model_elev.covar_module.raw_outputscale = \
            model_temp.covar_module.raw_outputscale
        print("[INFO] Shared kernel hyperparameters (lengthscale & outputscale) between temp and elev.")
    else:
        print("[INFO] SpaceXTime kernel: keep temp/elev kernel hyperparameters independent.")

    lik_temp = gpytorch.likelihoods.GaussianLikelihood().to(device)
    lik_elev = gpytorch.likelihoods.GaussianLikelihood().to(device)

    mll_temp = gpytorch.mlls.VariationalELBO(lik_temp, model_temp, num_data=Xtr.size(0))
    mll_elev = gpytorch.mlls.VariationalELBO(lik_elev, model_elev, num_data=Xtr.size(0))

    temp_params = list(model_temp.parameters())
    elev_params_all = list(model_elev.parameters())

    if key in ("rbf", "matern52"):
        shared_param_ids = set()
        for p in model_temp.covar_module.base_kernel.parameters():
            shared_param_ids.add(id(p))
        shared_param_ids.add(id(model_temp.covar_module.raw_outputscale))

        elev_params = [p for p in elev_params_all if id(p) not in shared_param_ids]

        print(f"[INFO] Filtered {len(elev_params_all) - len(elev_params)} shared params from elev model for optimizer.")
    else:
        elev_params = elev_params_all

    opt = torch.optim.Adam(
        [
            {'params': temp_params},
            {'params': lik_temp.parameters()},
            {'params': elev_params},
            {'params': lik_elev.parameters()},
        ],
        lr=0.01
    )

    use_pin_memory = True if device.type == 'cuda' else False
    num_workers = 0 if os.name == 'nt' else min(4, max(0, (os.cpu_count() or 1) - 1))
    if use_pin_memory:
        Xtr = Xtr.pin_memory()
        Ytr_temp = Ytr_temp.pin_memory()
        Ytr_elev = Ytr_elev.pin_memory()

    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr, Ytr_temp, Ytr_elev),
        batch_size=args.batch, shuffle=True,
        pin_memory=use_pin_memory, num_workers=num_workers
    )

    # ---------- Training ----------
    model_temp.train(); lik_temp.train()
    model_elev.train(); lik_elev.train()
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tot = 0.0
        for xb, yb_t, yb_e in dl:
            xb   = xb.to(device, non_blocking=True)
            yb_t = yb_t.to(device, non_blocking=True)
            yb_e = yb_e.to(device, non_blocking=True)

            opt.zero_grad()
            with gpytorch.settings.cholesky_jitter(args.jitter):
                out_t = model_temp(xb)
                out_e = model_elev(xb)
                loss_t = -mll_temp(out_t, yb_t)
                loss_e = -mll_elev(out_e, yb_e)
                loss = loss_t + loss_e
            loss.backward(); opt.step()
            tot += loss.item() * xb.size(0)
        print(f"[{args.kernel}] epoch {ep:02d}/{args.epochs} joint ELBO={tot/len(dl.dataset):.4f}")
    train_time = time.time() - t0
    print(f"Train time: {train_time:.1f}s")

    # ---------- Prediction ----------
    model_temp.eval(); lik_temp.eval()
    model_elev.eval(); lik_elev.eval()
    mu_t_list = []; std_t_list = []
    mu_e_list = []; std_e_list = []

    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(args.jitter):
        bs = args.batch
        for i in range(0, Xte.size(0), bs):
            xb = Xte[i:i+bs].to(device)
            p_t = lik_temp(model_temp(xb))
            p_e = lik_elev(model_elev(xb))
            mu_t_list.append(p_t.mean.detach().cpu())
            std_t_list.append(p_t.stddev.detach().cpu())
            mu_e_list.append(p_e.mean.detach().cpu())
            std_e_list.append(p_e.stddev.detach().cpu())

    mu_t_n  = torch.cat(mu_t_list, 0)
    std_t_n = torch.cat(std_t_list, 0)
    mu_e_n  = torch.cat(mu_e_list, 0)
    std_e_n = torch.cat(std_e_list, 0)

    temp_hat  = mu_t_n * Ytemp_std + Ytemp_mean
    temp_std  = std_t_n * Ytemp_std
    elev_hat  = mu_e_n * Yelev_std + Yelev_mean
    elev_std  = std_e_n * Yelev_std

    rmse_temp = torch.sqrt(torch.mean((temp_hat - Yte_temp)**2)).item()
    mae_temp  = torch.mean(torch.abs(temp_hat - Yte_temp)).item()
    mape_temp = (torch.mean(torch.abs((temp_hat - Yte_temp) / (Yte_temp + 1e-6))).item()) * 100.0
    ss_res_t  = torch.sum((Yte_temp - temp_hat)**2)
    ss_tot_t  = torch.sum((Yte_temp - torch.mean(Yte_temp))**2) + 1e-12
    r2_temp   = (1 - ss_res_t / ss_tot_t).item()
    crps_temp = gaussian_crps(temp_hat, torch.clamp(temp_std, min=1e-6), Yte_temp).item()

    rmse_elev = torch.sqrt(torch.mean((elev_hat - Yte_elev)**2)).item()
    mae_elev  = torch.mean(torch.abs(elev_hat - Yte_elev)).item()
    ss_res_e  = torch.sum((Yte_elev - elev_hat)**2)
    ss_tot_e  = torch.sum((Yte_elev - torch.mean(Yte_elev))**2) + 1e-12
    r2_elev   = (1 - ss_res_e / ss_tot_e).item()

    try:
        key2, params = _canonicalize_kernel_name(args.kernel)
    except Exception:
        key2 = args.kernel
        params = {}
    tag = sanitize(key2) + "_sharedkernel_temp_elev"

    pred_path    = os.path.join(OUT_DIR, f"pred_{tag}.pt")
    metrics_path = os.path.join(OUT_DIR, f"metrics_{tag}.json")

    if args.backup:
        for p in (pred_path, metrics_path):
            if os.path.exists(p):
                bak = p + ".bak." + datetime.now().strftime('%Y%m%d%H%M%S')
                try:
                    shutil.copy2(p, bak)
                    print(f"Backed up {p} -> {bak}")
                except Exception as e:
                    print(f"Warning: failed to backup {p}: {e}")

    torch.save({
        "temp_hat": temp_hat, "temp_std": temp_std, "Yte_temp": Yte_temp,
        "elev_hat": elev_hat, "elev_std": elev_std, "Yte_elev": Yte_elev
    }, pred_path)

    metrics = {
        "requested_kernel": args.kernel,
        "kernel": key2,
        # temperature metrics
        "temp_rmse": rmse_temp,
        "temp_mae": mae_temp,
        "temp_mape": mape_temp,
        "temp_r2": r2_temp,
        "temp_crps": crps_temp,
        # elevation metrics
        "elev_rmse": rmse_elev,
        "elev_mae": mae_elev,
        "elev_r2": r2_elev,
        # meta
        "epochs": args.epochs,
        "batch": args.batch,
        "m": args.m,
        "jitter": args.jitter,
        "train_time_sec": train_time,
        "input_dim": D,
        "num_tasks": 2,
        "use_elevation_as_output": True,
        "shared_kernel_hyperparams": key in ("rbf", "matern52"),
    }

    if not args.no_hyper:
        raw_hyp = {}
        for name, param in model_temp.named_parameters():
            if ("lengthscale" in name) or ("outputscale" in name):
                raw_hyp[f"temp_model.{name}"] = param.detach().cpu().view(-1).tolist()
        for name, param in lik_temp.named_parameters():
            if "noise" in name:
                raw_hyp[f"temp_likelihood.{name}"] = float(param.detach().cpu().item())
        for name, param in model_elev.named_parameters():
            if ("lengthscale" in name) or ("outputscale" in name):
                raw_hyp[f"elev_model.{name}"] = param.detach().cpu().view(-1).tolist()
        for name, param in lik_elev.named_parameters():
            if "noise" in name:
                raw_hyp[f"elev_likelihood.{name}"] = float(param.detach().cpu().item())
        print("DEBUG raw hyperparams:", raw_hyp)
        metrics["raw_hyperparams"] = raw_hyp

    print("=== Temperature Metrics ===")
    print(f"RMSE  = {rmse_temp:.4f}")
    print(f"MAE   = {mae_temp:.4f}")
    print(f"MAPE  = {mape_temp:.2f}%")
    print(f"R^2   = {r2_temp:.4f}")
    print(f"CRPS  = {crps_temp:.4f}")
    print("=== Elevation Metrics ===")
    print(f"RMSE  = {rmse_elev:.4f}")
    print(f"MAE   = {mae_elev:.4f}")
    print(f"R^2   = {r2_elev:.4f}")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("Saved predictions & metrics for", args.kernel, "as", tag)
    print("Metrics path:", metrics_path)


if __name__ == "__main__":
    main()
