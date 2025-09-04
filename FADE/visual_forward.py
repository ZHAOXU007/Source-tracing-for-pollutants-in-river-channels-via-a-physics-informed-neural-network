import torch
import matplotlib.pyplot as plt
from net import DNN
import pandas as pd
import numpy as np
from main_forward import PINN
import os

# 设置环境变量防止冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 参数设置
x_step = 0.01
t_step = 0.01
x_min=0
x_max=10
t_min=0
t_max=1

# 加载训练好的模型
model = PINN()
model.net.load_state_dict(torch.load('model1.5.param'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义网格
xc = torch.arange(x_min, x_max + x_step, x_step, dtype=torch.float32)
tc = torch.arange(t_min, t_max + t_step, t_step, dtype=torch.float32)
xx, tt = torch.meshgrid(xc, tc)

xt = np.concatenate([xx.reshape(-1, 1), tt.reshape(-1, 1)], axis=1)
xt = torch.tensor(xt, dtype=torch.float32).to(device)

# 预测浓度
with torch.no_grad():
    c_pred = model.net(xt)[:, 0:1]

# 重塑为2D数组 (x, t)
c_pred = c_pred.reshape(len(xc), len(tc)).detach().cpu().numpy()

# 选择要保存的时间点
time_points = [ 0.2, 0.4, 0.6, 0.8, 1.0]
time_indices = [int(t / t_step) for t in time_points]

# # 创建DataFrame保存数据
data_dict = {'x': xc.cpu().numpy()}
for i, t in zip(time_indices, time_points):
    data_dict[f't={t:.1f}s'] = c_pred[:, i]

df = pd.DataFrame(data_dict)

# 保存到Excel
excel_filename = 'concentration_profiles.xlsx'
df.to_excel(excel_filename, index=False)
print(f'浓度数据已保存到 {excel_filename}')

# 可视化
plt.figure(figsize=(10, 6))
for t in time_points:
    idx = int(t / t_step)
    plt.plot(xc, c_pred[:, idx], label=f't={t:.1f}s')

plt.xlabel('x')
plt.ylabel('Concentration')
plt.title('Concentration Profiles at Different Times')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

