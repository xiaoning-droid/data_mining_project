import os, json, time, math, argparse, numpy as np, torch, gpytorch
import matplotlib.pyplot as plt  
import shutil
from datetime import datetime
import re

OUT_DIR = r"C:\Users\Xiaonongzi\Desktop\CIVE 650\Project\Results"
os.makedirs(OUT_DIR, exist_ok=True)
DATA_NPZ = os.path.join(OUT_DIR, "data_prep.npz")

# ----------------- kernels Design -----------------
def _canonicalize_kernel_name(name: str):
    """
      'linear', 'rbf', 'matern32', 'matern52', 'rq_like',
      'rbf+linear', 'exp+linear', 'periodic*time', 'spacextime'
    """
    s = name.strip().lower()
    s = s.replace(' ', '')
    s = s.replace('(', '').replace(')', '')
    s = s.replace('/', '_')
    s = s.replace('exponential', 'exp').replace('matern12', 'exp')
    s = s.replace('times', '*').replace('x', '*')

    # ---- 8.space-time  ----
    if 'spacextime' in s or ('space' in s and 'time' in s):
        if '+linear' in s or 'pluslinear' in s:
            return 'spacextime+linear', {}
        else:
            return 'spacextime', {}
        
    # ---- 5.multi-scale RQ-like ----
    if 'rq_like' in s or 'rqlike' in s:
        return 'rq_like', {}

    # ---- 7.periodic × time ----
    if 'periodic' in s and 'time' in s and ('*' in s or 'periodictime' in s):
        return 'periodic*time', {}

    # ---- 6.RBF + Linear ----
    if 'rbf' in s and '+linear' in s:
        return 'rbf+linear', {}
    if 'rbf' in s and 'pluslinear' in s:
        return 'rbf+linear', {}

    # ---- Exp + Linear（Matern ν=1/2 + Linear）----
    if ('exp' in s or 'matern0.5' in s) and ('+linear' in s or 'pluslinear' in s):
        return 'exp+linear', {}

    if 'multiscale' in s or 'matern_mix' in s:
        return 'multiscale', {}   # 新增 multiscale kernel

    # ---- Basic kernels ----
        #------  1. linear  ----
    if s == 'linear' or s.endswith('linear'):
        return 'linear', {}
    #------  2. RBF  ----
    if s in ('rbf', 'rbfkernel'):
        return 'rbf', {}
     #------  3. matern32  ----
    if 'matern32' in s or 'matern3' in s:
        return 'matern32', {}
    #------  4. matern52  ----
    if 'matern52' in s or 'matern5' in s:
        return 'matern52', {}


    raise ValueError(f"Unknown or unsupported kernel name: {name}")
    

    
def make_kernel(name: str, D: int):
    """
      - 'linear'
      - 'rbf'
      - 'matern32'
      - 'matern52'
      - 'rq_like'
      - 'rbf+linear'
      - 'periodic*time'
        - 'spacextime'
    """
    key, params = _canonicalize_kernel_name(name)
    if key == 'linear':
        return gpytorch.kernels.LinearKernel(ard_num_dims=D)
    
    if key == 'rbf':
        return gpytorch.kernels.RBFKernel(ard_num_dims=D)

    if key == 'matern32':
        return gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=D)

    if key == 'matern52':
        return gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=D)

    if key == 'periodic*time':
        return gpytorch.kernels.PeriodicKernel(active_dims=[2]) * gpytorch.kernels.RBFKernel(ard_num_dims=D)

    if key == 'rbf+linear':
        return gpytorch.kernels.RBFKernel(ard_num_dims=D) + gpytorch.kernels.LinearKernel(ard_num_dims=D)

    if key == 'rq_like':
        ks = [gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=D)) for _ in range(3)]
        k = ks[0]
        for ki in ks[1:]:
            k = k + ki
        return k
    
    if key == 'spacextime':
        # space：Matern32，time：RBF
        SPACE = gpytorch.kernels.MaternKernel(nu=1.5, ard_num_dims=2, active_dims=[0, 1])
        TIME  = gpytorch.kernels.RBFKernel(ard_num_dims=1, active_dims=[2])
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
        qstrat = gpytorch.variational.VariationalStrategy(self, inducing_points, qdist, learn_inducing_locations=True)
        super().__init__(qstrat)
        self.mean_module  = gpytorch.means.ConstantMean()
        base_scaled = gpytorch.kernels.ScaleKernel(base_kernel)
        try:
            noise_k = gpytorch.kernels.WhiteNoiseKernel(noise=1e-6)
            self.covar_module = base_scaled + noise_k
        except Exception:
            self.covar_module = base_scaled

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
                    help="One of: RBF, Matern32, Matern52, Linear, RQ_Like, RBF+Linear, Exp+Linear, Periodic*Time, SpaceXTime+Linear")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--m", type=int, default=800)
    ap.add_argument("--jitter", type=float, default=1e-3)
    ap.add_argument("--maxn", type=int, default=120000)
    ap.add_argument("--no-hyper", action='store_true')
    ap.add_argument("--backup", action='store_true')
    args = ap.parse_args()

    pack = np.load(DATA_NPZ)
    X_train = pack["X_train"]; y_train = pack["y_train"]
    X_test  = pack["X_test"];  y_test  = pack["y_test"]
    X_mean  = pack["X_mean"];  X_std   = pack["X_std"]
    Y_mean  = pack["Y_mean"].item(); Y_std = pack["Y_std"].item()
    H = int(pack["H"]); W = int(pack["W"]); T = int(pack["T"])

    # normalize
    Xtr = torch.from_numpy(((X_train - X_mean) / X_std).astype(np.float32))
    Xte = torch.from_numpy(((X_test  - X_mean) / X_std).astype(np.float32))
    Ytr = torch.from_numpy(((y_train - Y_mean) / Y_std).astype(np.float32)).reshape(-1)
    Yte = torch.from_numpy(y_test.astype(np.float32)).reshape(-1)

    # optional downsample
    if Xtr.shape[0] > args.maxn:
        idx = torch.randperm(Xtr.shape[0])[:args.maxn]
        Xtr, Ytr = Xtr[idx], Ytr[idx]

    D = Xtr.shape[1]
    base_kernel = make_kernel(args.kernel, D).to(device)
    model = SVGPModel(make_Z(Xtr, args.m), base_kernel).to(device)
    lik = gpytorch.likelihoods.GaussianLikelihood().to(device)

    mll = gpytorch.mlls.VariationalELBO(lik, model, num_data=Xtr.size(0))
    opt = torch.optim.Adam(
        [{'params': model.parameters()}, {'params': lik.parameters()}],
        lr=0.01
    )

    use_pin_memory = True if device.type == 'cuda' else False
    num_workers = 0 if os.name == 'nt' else min(4, max(0, (os.cpu_count() or 1) - 1))
    if use_pin_memory:
        Xtr = Xtr.pin_memory()
        Ytr = Ytr.pin_memory()

    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr, Ytr),
        batch_size=args.batch, shuffle=True,
        pin_memory=use_pin_memory, num_workers=num_workers
    )

    # ---------- Training ----------
    model.train(); lik.train()
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        tot = 0.0
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            opt.zero_grad()
            with gpytorch.settings.cholesky_jitter(args.jitter):
                out = model(xb)
                loss = -mll(out, yb)
            loss.backward(); opt.step()
            tot += loss.item() * xb.size(0)
        print(f"[{args.kernel}] epoch {ep:02d}/{args.epochs} ELBO={tot/len(dl.dataset):.4f}")
    train_time = time.time() - t0
    print(f"Train time: {train_time:.1f}s")

    # ---------- Prediction ----------
    model.eval(); lik.eval()
    mu_list = []; std_list = []
    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(args.jitter):
        bs = args.batch
        for i in range(0, Xte.size(0), bs):
            xb = Xte[i:i+bs].to(device)
            p = lik(model(xb))
            mu_list.append(p.mean.detach().cpu())
            std_list.append(p.stddev.detach().cpu())
    mu_n  = torch.cat(mu_list, 0)
    std_n = torch.cat(std_list, 0)
    yhat  = mu_n * Y_std + Y_mean
    ystd  = std_n * Y_std

    # ---------- Metrics ----------
    rmse = torch.sqrt(torch.mean((yhat - Yte)**2)).item()
    mae  = torch.mean(torch.abs(yhat - Yte)).item()
    mape = (torch.mean(torch.abs((yhat - Yte) / (Yte + 1e-6))).item()) * 100.0
    ss_res = torch.sum((Yte - yhat)**2)
    ss_tot = torch.sum((Yte - torch.mean(Yte))**2) + 1e-12
    r2   = (1 - ss_res / ss_tot).item()
    crps = gaussian_crps(yhat, torch.clamp(ystd, min=1e-6), Yte).item()

    # ---------- Save ----------
    try:
        key, params = _canonicalize_kernel_name(args.kernel)
    except Exception:
        key = args.kernel
        params = {}
    tag = sanitize(key)

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

    torch.save({"yhat": yhat, "ystd": ystd, "Yte": Yte}, pred_path)

    metrics = {
        "requested_kernel": args.kernel,
        "kernel": key,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2": r2,
        "crps": crps,
        "epochs": args.epochs,
        "batch": args.batch,
        "m": args.m,
        "jitter": args.jitter,
        "train_time_sec": train_time,
    }

    if not args.no_hyper:
        raw_hyp = {}
        for name, param in model.named_parameters():
            if ("lengthscale" in name) or ("outputscale" in name):
                raw_hyp[f"model.{name}"] = param.detach().cpu().view(-1).tolist()
        for name, param in lik.named_parameters():
            if "noise" in name:
                try:
                    raw_hyp[f"likelihood.{name}"] = float(param.detach().cpu().item())
                except Exception:
                    raw_hyp[f"likelihood.{name}"] = float(param.detach().cpu())
        print("DEBUG raw hyperparams:", raw_hyp)
        metrics["raw_hyperparams"] = raw_hyp

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("Saved predictions & metrics for", args.kernel, "as", tag)
    print("Metrics path:", metrics_path)


if __name__ == "__main__":
    main()
