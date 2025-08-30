import torch
import matplotlib.pyplot as plt
from net import DNN
from matplotlib import gridspec
import pandas as pd
from scipy.interpolate import griddata
import numpy as np
from matplotlib.ticker import MaxNLocator
from train32 import PINN
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 参数设置
x_step = 0.1
t_step = 0.1
x_min = 0
x_max = 35
t_min = 0
t_max = 72

model = PINN()
model.net.load_state_dict(torch.load(f'model_inverse32.param'))
print(model.lambda_m)
print(model.lambda_x)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
file_path = 'data3real.csv'
df = pd.read_csv(file_path, header=0)
x_real = df.iloc[:, 0].values
t_real = df.iloc[:, 1].values
c_real = df.iloc[:, 2].values

# 定义网格
xc = torch.arange(x_min, x_max + x_step, x_step, dtype=torch.float32)
tc = torch.arange(t_min, t_max + t_step, t_step, dtype=torch.float32)
xx, tt = torch.meshgrid(xc, tc)

xt = np.concatenate([xx.reshape(-1, 1), tt.reshape(-1, 1)], axis=1)
xt = torch.tensor(xt, dtype=torch.float32).to(device)

with torch.no_grad():
    c_pred = model.net(xt)[:, 0:1]

# 重塑为 3D 数组
c_pred = c_pred.reshape(len(xc), len(tc)).detach().cpu().numpy()
c_real_grid = griddata((x_real, t_real), c_real, (xx.cpu().numpy(), tt.cpu().numpy()), method='linear')

# 计算误差
error = np.log10(np.abs(c_pred - c_real_grid))
relative_l2_error = np.linalg.norm(c_real_grid - c_pred) / np.linalg.norm(c_real_grid)
print(f'Relative L2 Error: {relative_l2_error:.4f}')

# 第一个图：云图
plt.figure(figsize=(15, 5))

gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])

# PINN云图
ax1 = plt.subplot(gs[0])
contour_pred = ax1.contourf(xx.cpu().numpy(), tt.cpu().numpy(), c_pred, levels=100, cmap='viridis')
ax1.set_title("(a) Predicted by PINN", fontsize=15, fontname='Times New Roman')
ax1.set_xlabel("x (km)", fontsize=15, fontname='Times New Roman')
ax1.set_ylabel("time (h)", fontsize=15, fontname='Times New Roman')
plt.colorbar(contour_pred, ax=ax1)

# CFD云图
ax2 = plt.subplot(gs[1])
contour_real = ax2.contourf(xx.cpu().numpy(), tt.cpu().numpy(), c_real_grid, levels=100, cmap='viridis')
ax2.set_title("(b) Calculated by CFD", fontsize=15, fontname='Times New Roman')
ax2.set_xlabel("x (km)", fontsize=15, fontname='Times New Roman')
ax2.set_ylabel("time (h)", fontsize=15, fontname='Times New Roman')
plt.colorbar(contour_real, ax=ax2)

# 误差云图
ax3 = plt.subplot(gs[2])
contour_error = ax3.contourf(xx.cpu().numpy(), tt.cpu().numpy(), error, levels=100, cmap='viridis')
ax3.set_title(r"(c) log$_{10}$ |C$_{pinn}$ - C$_{CFD}$|", fontsize=15, fontname='Times New Roman')
ax3.set_xlabel("x (km)", fontsize=15, fontname='Times New Roman')
ax3.set_ylabel("time (h)", fontsize=15, fontname='Times New Roman')
plt.colorbar(contour_error, ax=ax3)

plt.tight_layout()
plt.savefig("cloud_plots.tiff", dpi=600)
plt.show()

# 第二个图：浓度分布曲线，分为三小图
plt.figure(figsize=(15, 5))
gs2 = gridspec.GridSpec(1, 3)

for i, t_val in enumerate([24, 48, 72]):
    ax = plt.subplot(gs2[i])
    ax.plot(xc.cpu().numpy(), c_real_grid[:, int(t_val / t_step)], label='CFD', color='red', linestyle='-')
    ax.plot(xc.cpu().numpy(), c_pred[:, int(t_val / t_step)], label='PINN', color='blue', linestyle='--')
    ax.set_title(f'time = {t_val}h', fontsize=12, fontname='Times New Roman')
    ax.set_xlabel('x (km)', fontsize=12, fontname='Times New Roman')
    ax.set_ylabel(r'Concentration flux(g/m)', fontsize=12, fontname='Times New Roman')
    ax.legend(loc='upper left', fontsize=10, frameon=False, prop={'family': 'Times New Roman'})
    ax.set_facecolor('white')  # 设置背景为白色
    ax.grid(False)  # 不显示网格

plt.tight_layout()
plt.savefig("concentration_distribution_different_time.tiff", dpi=600)
plt.show()
# 第二个图：浓度分布曲线，分为三小图
plt.figure(figsize=(15, 5))
gs3 = gridspec.GridSpec(1, 3)

for i, x_val in enumerate([10, 20, 30]):
    ax = plt.subplot(gs3[i])
    ax.plot(tc.cpu().numpy(), c_real_grid[int(x_val / x_step), :], label=f'CFD ', color='red', linestyle='-')
    ax.plot(tc.cpu().numpy(), c_pred[int(x_val / x_step), :], label=f'PINN ', color='blue', linestyle='--')
    ax.set_title(f'Section = {x_val}km', fontsize=12, fontname='Times New Roman')
    ax.set_xlabel('time (h)', fontsize=12, fontname='Times New Roman')
    ax.set_ylabel(r'Concentration flux(g/m)', fontsize=12, fontname='Times New Roman')
    ax.legend(loc='upper left', fontsize=10, frameon=False, prop={'family': 'Times New Roman'})
    ax.set_facecolor('white')  # 设置背景为白色
    ax.grid(False)  # 不显示网格

plt.tight_layout()
plt.savefig("concentration_distribution_different_section.tiff", dpi=600)
plt.show()
#第三个图：不同断面的浓度分布图
plt.figure(figsize=(10, 5))
for x_val in [10, 20, 30]:

    plt.plot(tc.cpu().numpy(), c_real_grid[int(x_val / x_step), :], label=f'CFD x={x_val}m', color='red', linestyle='-')
    plt.plot(tc.cpu().numpy(), c_pred[int(x_val / x_step), :], label=f'PINN x={x_val}m', color='blue', linestyle='--')

plt.title('Concentration Distribution at Different Sections', fontsize=15, fontname='Times New Roman')
plt.xlabel('t (s)', fontsize=15, fontname='Times New Roman')
plt.ylabel(r'C (g/m$^{-3}$)', fontsize=15, fontname='Times New Roman')
plt.legend(loc='upper left', fontsize=10, frameon=False, prop={'family': 'Times New Roman'})
plt.grid(False)  # 不显示网格
plt.savefig("sectional_concentration.tiff", dpi=600)
plt.show()