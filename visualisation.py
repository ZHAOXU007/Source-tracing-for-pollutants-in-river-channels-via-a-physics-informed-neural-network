
import torch
import matplotlib.pyplot as plt
from net import DNN
from matplotlib import gridspec
import pandas as pd
from scipy.interpolate import griddata
import numpy as np
from main import PINN
import os
from matplotlib.ticker import MaxNLocator
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
x_step = 0.1
t_step = 0.1
x_min = 0
x_max = 20
t_min = 0
t_max = 10

model = PINN()
model.net.load_state_dict(torch.load(f'model.param'))

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

# 对 c_real 进行插值
c_real_grid = griddata((x_real, t_real), c_real, (xx.cpu().numpy(), tt.cpu().numpy()), method='linear')

# 计算误差
error = np.log10(np.abs(c_pred - c_real_grid))
relative_l2_error = np.linalg.norm(c_real_grid - c_pred) / np.linalg.norm(c_real_grid)

# 打印相对 L2 误差
print(f'Relative L2 Error: {relative_l2_error:.4f}')
# 创建一个 2x3 的网格布局
plt.rc('font', family='Times New Roman')  # 设置字体为 serif，大小为 12
#plt.rcParams['font.family'] = 'Times New Roman'
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1])

# 第一行的云图误差图
ax_error = plt.subplot(gs[0, 2])
contour_error = ax_error.contourf(xx.cpu().numpy(), tt.cpu().numpy(), error, levels=100, cmap='viridis')
ax_error.set_title(r"(c)log$_{10}$ (C$_{pinn}$ - C$_{CFD}$)", fontsize=15)
ax_error.set_xlabel("x(m)", fontsize=15)
ax_error.set_ylabel("time(s)", fontsize=15)
cbar3=plt.colorbar(contour_error, ax=ax_error)
cbar3.locator = MaxNLocator(integer=True)
cbar3.update_ticks()
# 第一行的预测图
ax3 = plt.subplot(gs[0, 0])
contour_pred2 = ax3.contourf(xx.cpu().numpy(), tt.cpu().numpy(), c_pred, levels=100, cmap='viridis')
ax3.set_title("(a)Predicted by PINN", fontsize=15)
ax3.set_xlabel("x(m)", fontsize=15)
ax3.set_ylabel("time(s)", fontsize=15)
cbar1=plt.colorbar(contour_pred2, ax=ax3)
cbar1.locator = MaxNLocator(integer=True)
cbar1.update_ticks()
# 第一行的真实值图
ax2 = plt.subplot(gs[0, 1])
contour_real = ax2.contourf(xx.cpu().numpy(), tt.cpu().numpy(), c_real_grid, levels=100, cmap='viridis')
ax2.set_title("(b)Calculated by CFD", fontsize=15)
ax2.set_xlabel("x(m)", fontsize=15)
ax2.set_ylabel("time(s)", fontsize=15)
cbar2=plt.colorbar(contour_real, ax=ax2)
cbar2.locator = MaxNLocator(integer=True)
cbar2.update_ticks()
# 第二行的曲线图（每个图占 1/2）
gs2 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, :], wspace=0.2)

ax_curve1 = plt.subplot(gs2[0, 0])  # 第一列
ax_curve2 = plt.subplot(gs2[0, 1])  # 第二列

# 绘制 CFD 曲线
ax_curve1.plot(xc.cpu().numpy(), c_real_grid[:, int(t_max / t_step * 0.2)], label='CFD', color='red', linestyle='-', alpha=1.0)
ax_curve1.plot(xc.cpu().numpy(), c_real_grid[:, int(t_max / t_step * 0.4)], color='red', linestyle='-', alpha=1.0)
ax_curve1.plot(xc.cpu().numpy(), c_real_grid[:, int(t_max / t_step * 0.6)], color='red', linestyle='-', alpha=1.0)
ax_curve1.plot(xc.cpu().numpy(), c_real_grid[:, int(t_max / t_step * 0.8)], color='red', linestyle='-', alpha=1.0)
ax_curve1.plot(xc.cpu().numpy(), c_real_grid[:, int(t_max / t_step * 1.0)], color='red', linestyle='-', alpha=1.0)
# 绘制 PINN 曲线
ax_curve1.plot(xc.cpu().numpy(), c_pred[:, int(t_max / t_step * 0.2)], label='PINN', color='blue', linestyle='--', alpha=1.0)
ax_curve1.plot(xc.cpu().numpy(), c_pred[:, int(t_max / t_step * 0.4)], color='blue', linestyle='--', alpha=1.0)
ax_curve1.plot(xc.cpu().numpy(), c_pred[:, int(t_max / t_step * 0.6)], color='blue', linestyle='--', alpha=1.0)
ax_curve1.plot(xc.cpu().numpy(), c_pred[:, int(t_max / t_step * 0.8)], color='blue', linestyle='--', alpha=1.0)
ax_curve1.plot(xc.cpu().numpy(), c_pred[:, int(t_max / t_step * 1.0)], color='blue', linestyle='--', alpha=1.0)

# 添加标记和标签
y_value_to_mark = 0.25
ax_curve1.text(6.5, y_value_to_mark, f't={t_max * 0.2:.1f}s', color='black', fontsize=11, ha='center', va='center')
ax_curve1.text(10.5, y_value_to_mark, f't={t_max * 0.4:.1f}s', color='black', fontsize=11, ha='center', va='center')
ax_curve1.text(14.5, y_value_to_mark, f't={t_max * 0.6:.1f}s', color='black', fontsize=11, ha='center', va='center')
ax_curve1.text(18.3, 0.25, f't={t_max * 0.8:.1f}s', color='black', fontsize=11, ha='center', va='center')
ax_curve1.text(19, 0.5, f't={t_max * 1.0:.1f}s', color='black', fontsize=11, ha='center', va='center')
ax_curve1.set_title('(d)Concentration flux changes at different times', fontsize=15 )
ax_curve1.set_xlabel('x(m)', fontsize=15)
ax_curve1.set_ylabel(r"Concentration flux(g/m)", fontsize=15)
ax_curve1.legend(loc='upper left', fontsize=8, frameon=False)

# 第二行的曲线图
ax_curve2.plot(tc.cpu().numpy(), c_real_grid[int(x_max / x_step * 0.2), :], label='CFD', color='red', linestyle='-', alpha=1.0)
ax_curve2.plot(tc.cpu().numpy(), c_real_grid[int(x_max / x_step * 0.4), :], color='red', linestyle='-', alpha=1.0)
ax_curve2.plot(tc.cpu().numpy(), c_real_grid[int(x_max / x_step * 0.6), :], color='red', linestyle='-', alpha=1.0)
ax_curve2.plot(tc.cpu().numpy(), c_real_grid[int(x_max / x_step * 0.8), :], color='red', linestyle='-', alpha=1.0)
ax_curve2.plot(tc.cpu().numpy(), c_real_grid[int(x_max / x_step * 1.0), :], color='red', linestyle='-', alpha=1.0)

ax_curve2.plot(tc.cpu().numpy(), c_pred[int(x_max / x_step * 0.2), :], label='PINN', color='blue', linestyle='--', alpha=1.0)
ax_curve2.plot(tc.cpu().numpy(), c_pred[int(x_max / x_step * 0.4), :], color='blue', linestyle='--', alpha=1.0)
ax_curve2.plot(tc.cpu().numpy(), c_pred[int(x_max / x_step * 0.6), :], color='blue', linestyle='--', alpha=1.0)
ax_curve2.plot(tc.cpu().numpy(), c_pred[int(x_max / x_step * 0.8), :], color='blue', linestyle='--', alpha=1.0)
ax_curve2.plot(tc.cpu().numpy(), c_pred[int(x_max / x_step * 1.0), :], color='blue', linestyle='--', alpha=1.0)

y_value_to_mark = 0.25

ax_curve2.text(0.67, 0.25, f'Section_{x_max * 0.2:.1f}m', color='black', fontsize=11, ha='center', va='center')
ax_curve2.text(2.87, 0.25, f'Section_{x_max * 0.4:.1f}m', color='black', fontsize=11, ha='center', va='center')
ax_curve2.text(4.7, 0.25 ,f'Section_{x_max * 0.6:.1f}m', color='black', fontsize=11, ha='center', va='center')
ax_curve2.text(6.7, 0.25, f'Section_{x_max * 0.8:.1f}m', color='black', fontsize=11, ha='center', va='center')
ax_curve2.text(8.7, 0.25, f'Section_{x_max * 1.0:.1f}m', color='black', fontsize=11, ha='center', va='center')
ax_curve2.set_title('(e)Concentration flux changes at different sections', fontsize=15 )
ax_curve2.set_xlabel('time(s)', fontsize=15)
ax_curve2.set_ylabel(r"Concentration flux(g/m)", fontsize=15)
ax_curve2.legend(loc='upper left', fontsize=8, frameon=False)
# 调整布局
plt.savefig("sine_wave.png", dpi=600)
plt.tight_layout()
plt.show(block=True)

# 保存数据
#save_1D_data(tc, c_pred, c_real_grid)
#print('文件输出完成')