import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress

ELEV_PATH = r"C:\Users\Xiaonongzi\Desktop\CIVE 650\Project\course_project_data\data\elev_grid_100x200.npy"
MAT_PATH  = r"C:\Users\Xiaonongzi\Desktop\CIVE 650\Project\course_project_data\MODIS_Aug.mat"

elev = np.load(ELEV_PATH)
mat  = sio.loadmat(MAT_PATH)
train = mat["training_tensor"].astype(float)   
test = mat["test_tensor"].astype(float)        
full = train + test    

day = list(range(31))   # [0~30]
temp_mean = np.nanmean(full[:, :, day], axis=2)   

temp_mean[temp_mean == 0] = np.nan
temp_C = temp_mean 
mask = (~np.isnan(temp_C)) & (elev > -1000)
e = elev[mask]
t = temp_C[mask]

r, _ = pearsonr(e, t)
slope, intercept, *_ = linregress(e, t)

print("effective points：", len(e))
print("r =", r)
print("T = %.5f * Elev + %.2f" % (slope, intercept))
print("Lapse rate (°C/km) ≈", slope * 1000)

plt.scatter(e, t, s=3, alpha=0.3)
x = np.linspace(e.min(), e.max(), 200)
plt.plot(x, slope * x + intercept, "r")

plt.xlabel("Elevation (m)")
plt.ylabel("Temperature ")
plt.title("AVE Temperature vs Elevation")
plt.tight_layout()
plt.show()
