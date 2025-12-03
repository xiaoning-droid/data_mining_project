import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress

BASE = r"C:\Users\Xiaonongzi\Desktop\CIVE 650\Course project"
MAT_PATH   = os.path.join(BASE, "course_project_data", "MODIS_Aug.mat")
ELEV_PATH  = os.path.join(BASE, "course_project_data", "data", "elev_grid_100x200.npy")
OUT_FIG    = os.path.join(BASE, "Figures", "Fig6_temp_vs_elev_day1_full.png")
os.makedirs(os.path.dirname(OUT_FIG), exist_ok=True)

NODATA_ELEV = -1000.0   
DAY_IDX = 0          

elev = np.load(ELEV_PATH) 
print("Elevation shape:", elev.shape, "min/max:", elev.min(), elev.max())

data  = sio.loadmat(MAT_PATH)
train = data["training_tensor"].astype(np.float32)  
test  = data["test_tensor"].astype(np.float32)     
H, W, T = train.shape
print("Train/Test shape:", train.shape, test.shape)

full = np.where(train != 0, train, test)   

temp_day = full[:, :, DAY_IDX].astype(float)  
temp_day[temp_day == 0] = np.nan

temp_day = temp_day - 273.15

print("Temp Day1 valid range (°C):", np.nanmin(temp_day), np.nanmax(temp_day))

mask_elev = elev > NODATA_ELEV
mask_temp = ~np.isnan(temp_day)
mask = mask_elev & mask_temp

elev_flat = elev[mask]
temp_flat = temp_day[mask]
print("有效样本数:", len(elev_flat))

r, p = pearsonr(elev_flat, temp_flat)
slope, intercept, _, _, _ = linregress(elev_flat, temp_flat)
lapse_rate = slope * 1000 
print("=======================================")
print("Day 1 Temperature vs Elevation (FULL)")
print("=======================================")
print(f"Pearson r      = {r:.4f}")
print(f"p-value        = {p:.4e}")
print(f"T = {slope:.5f} * Elev + {intercept:.2f}")
print(f"Lapse rate     = {lapse_rate:.2f} °C/km")
print("=======================================")

plt.figure(figsize=(6, 6))
plt.scatter(elev_flat, temp_flat, s=5, alpha=0.25, label="Data")

x_line = np.linspace(elev_flat.min(), elev_flat.max(), 300)
plt.plot(x_line, slope * x_line + intercept, "r", label="Regression")

plt.xlabel("Elevation (m)")
plt.ylabel("Temperature (°C)")
plt.title("Day 1 Temperature vs Elevation (Full MODIS LST)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)
plt.close()
print("Saved figure:", OUT_FIG)
