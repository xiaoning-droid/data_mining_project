
import os
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import gpytorch

PROJECT_DIR = r"C:\Users\Xiaonongzi\Desktop\CIVE 650\Project"
DATA_PATH = os.path.join(PROJECT_DIR, "Results", "data_prep.npz")
OUT_DIR = os.path.join(PROJECT_DIR, "Results")
os.makedirs(OUT_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


class DeepFeatureExtractor(nn.Module):
    def __init__(self, input_dim, feature_dim=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, feature_dim),
        )

    def forward(self, x):
        return self.net(x)


class VariationalDKLModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, feature_extractor, num_mixtures=4):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        self.feature_extractor = feature_extractor
        feature_dim = inducing_points.size(1)
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_covar = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=num_mixtures, ard_num_dims=feature_dim)
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_covar)

    def forward(self, x_projected):
        mean_x = self.mean_module(x_projected)
        covar_x = self.covar_module(x_projected)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--feature-dim", type=int, default=16)
    ap.add_argument("--num-mixtures", type=int, default=4)
    args = ap.parse_args()

    pack = np.load(DATA_PATH)
    X_train = pack["X_train"]
    y_train = pack["y_train"].astype(np.float32)
    X_test = pack["X_test"]
    y_test = pack["y_test"].astype(np.float32)

    Xtr = torch.from_numpy(X_train).float()
    Xte = torch.from_numpy(X_test).float()
    Ytr = torch.from_numpy(y_train).float().reshape(-1)
    Yte = torch.from_numpy(y_test).float().reshape(-1)

    N = Xtr.size(0)
    print(f"Loaded data: N_train={N}, N_test={Xte.size(0)}")

    y_mean = Ytr.mean()
    y_std = Ytr.std()
    Ytr_norm = (Ytr - y_mean) / (y_std + 1e-8)

    feature_extractor = DeepFeatureExtractor(input_dim=Xtr.size(1), feature_dim=args.feature_dim).to(device)

    init_idx = torch.randperm(N)[: args.m]
    with torch.no_grad():
        inducing_init = feature_extractor(Xtr[init_idx].to(device)).detach().cpu()

    inducing_points = inducing_init.clone().to(device)

    model = VariationalDKLModel(inducing_points, feature_extractor, num_mixtures=args.num_mixtures).to(device)
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=N)
    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': args.lr},
        {'params': model.covar_module.parameters(), 'lr': args.lr},
        {'params': model.mean_module.parameters(), 'lr': args.lr},
        {'params': likelihood.parameters(), 'lr': args.lr},
    ], lr=args.lr)
    ds = torch.utils.data.TensorDataset(Xtr, Ytr_norm)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

    model.train(); likelihood.train()
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        total_nll = 0.0
        for xb, yb in dl:
            xb = xb.to(device); yb = yb.to(device)
            xb_proj = model.feature_extractor(xb)
            optimizer.zero_grad()
            output = model(xb_proj)
            loss = -mll(output, yb)
            loss.backward()
            optimizer.step()
            total_nll += loss.item() * xb.size(0)
        avg = total_nll / N
        print(f"Epoch {ep}/{args.epochs} - AvgNegELBO={avg:.6f}")

    train_time = time.time() - t0
    print(f"Training finished in {train_time:.1f}s")

    model.eval(); likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        mu_list = []
        std_list = []
        bs = args.batch
        for i in range(0, Xte.size(0), bs):
            xb = Xte[i:i+bs].to(device)
            xb_proj = model.feature_extractor(xb)
            p = likelihood(model(xb_proj))
            mu_list.append(p.mean.detach().cpu())
            std_list.append(p.stddev.detach().cpu())

    mu_n = torch.cat(mu_list, 0)
    std_n = torch.cat(std_list, 0)

    yhat = mu_n * (y_std + 1e-8) + y_mean
    ystd = std_n * (y_std + 1e-8)

    mse = torch.mean((yhat - Yte)**2).item()
    rmse = float(np.sqrt(mse))
    mae = float(torch.mean(torch.abs(yhat - Yte)).item())

    metrics = {
        'model': 'dkl_svgp',
        'm': args.m,
        'epochs': args.epochs,
        'batch': args.batch,
        'rmse': rmse,
        'mae': mae,
        'train_time_sec': train_time,
    }

    save_path = os.path.join(OUT_DIR, 'dkl_svgp_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'likelihood_state_dict': likelihood.state_dict(),
        'y_mean': y_mean.cpu(),
        'y_std': y_std.cpu(),
        'metrics': metrics,
    }, save_path)
    print('Saved model to:', save_path)
    metrics_path = os.path.join(OUT_DIR, 'metrics_dkl_svgp.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    print('Saved metrics to:', metrics_path)

    # plotting
    try:
        import matplotlib.pyplot as plt
        y_true = Yte.numpy()
        y_pred = yhat.numpy()
        plt.figure(figsize=(6,6))
        plt.scatter(y_true, y_pred, alpha=0.4)
        mn = min(y_true.min(), y_pred.min())
        mx = max(y_true.max(), y_pred.max())
        plt.plot([mn,mx],[mn,mx],'k--',linewidth=1)
        plt.xlabel('True y'); plt.ylabel('Pred y')
        plt.title(f'DKL SVGP - RMSE={rmse:.4f}')
        out_plot = os.path.join(OUT_DIR, 'dkl_svgp_y_true_vs_pred.png')
        plt.tight_layout(); plt.savefig(out_plot, dpi=200); plt.close()
        print('Saved plot to:', out_plot)
    except Exception as e:
        print('Plotting failed:', e)


if __name__ == '__main__':
    main()
