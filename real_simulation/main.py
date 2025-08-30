import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

sys.path.append(".")
import numpy as np
import torch
from torch.autograd import grad
from net import DNN
from scipy.io import loadmat
import pandas as pd
import os

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# 定义尺度因子
time_scale = 3600  # 秒到小时
space_scale = 1000  # 米到千米
torch.manual_seed(2025)
np.random.seed(2025)
# 调整上下界
x_min, x_max = 0, 35
t_min, t_max = 0 , 72

x_step = 0.1
t_step = 0.1

# 定义上界和下界
ub = np.array([x_max, t_max])
lb = np.array([x_min, t_min])

class PINN:
    def __init__(self):
        self.diffusion_x = 1.0*3600/1000/1000

        # self.lambda_m = torch.tensor([2.0], requires_grad=True).to(device)
        # self.lambda_x = torch.tensor([2.0], requires_grad=True).to(device)
        self.lambda_m = (torch.rand(1) * 5).to(device)  # [0, 10] 均匀分布
        self.lambda_x = (torch.rand(1) * 10).to(device)  # [0, 20] 均匀分布

        self.noise = 0.02
        self.lambda_m = torch.nn.Parameter(self.lambda_m)
        self.lambda_x = torch.nn.Parameter(self.lambda_x)

        self.recorded_data = []  # 用于存储记录的数据
        self.net = DNN(dim_in=2, dim_out=1, n_layer=6, n_node=50, ub=ub, lb=lb).to(
            device
        )

        self.net.register_parameter("lambda_m", self.lambda_m)
        self.net.register_parameter("lambda_x", self.lambda_x)

        self.adam = torch.optim.Adam(self.net.parameters())
        self.lbfgs = torch.optim.LBFGS(
            self.net.parameters(),
            lr=1.0,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )
        self.iter = 0
        self.get_training_data()



    def get_training_data(self):
        """生成各类训练数据"""
        # 初始条件数据 (t=0)
        init_x = np.arange(x_min, x_max + x_step, x_step).reshape(-1, 1)
        init_t = np.zeros_like(init_x)
        self.init_xt = torch.tensor(np.concatenate([init_x, init_t], axis=1), dtype=torch.float32).to(device)
        self.init_c = torch.zeros_like(self.init_xt[:, 0:1]).to(device)

        # 边界条件数据 (x=0)
        bound_t = np.arange(t_min, t_max + t_step, t_step).reshape(-1, 1)
        bound_x = np.zeros_like(bound_t)
        self.bound_xt = torch.tensor(np.concatenate([bound_x, bound_t], axis=1), dtype=torch.float32).to(device)
        self.bound_c = torch.zeros_like(self.bound_xt[:, 0:1]).to(device)

        # PDE配点
        speed = pd.read_csv("./inputdata2.csv")
        self.pde_points = torch.tensor(np.column_stack([speed["x"], speed["t"]]), dtype=torch.float32).to(device)
        self.pde_u = torch.tensor(speed["u"].values.reshape(-1, 1), dtype=torch.float32).to(device)
        self.pde_dux = torch.tensor(speed["dux"].values.reshape(-1, 1), dtype=torch.float32).to(device)
        # x = np.arange(x_min, x_max + x_step, x_step)
        # t = np.arange(t_min, t_max + t_step, t_step)
        # X, T = np.meshgrid(x, t)
        # self.pde_points = torch.tensor(np.stack([X.flatten(), T.flatten()], axis=1), dtype=torch.float32).to(device)

        # # 观测数据
        data = pd.read_csv("./data32.csv")
        # obs_xi = data["xi"].values.reshape(-1, 1)  # 取出 xi 列
        # obs_ti = data["ti"].values.reshape(-1, 1)  # 取出 ti 列
        # obs_ci = data["ci"].values.reshape(-1, 1)  # 取出 ci 列
        # noise_ci = obs_ci + self.noise * np.std(obs_ci) * np.random.randn(*obs_ci.shape)
        # output_filename = f'noise_data_with_{self.noise}.xlsx'
        # output_df = pd.DataFrame({
        #     'xi': obs_xi.flatten(),
        #     'ti': obs_ti.flatten(),
        #     'ci': obs_ci.flatten(),
        #     'noise_ci': noise_ci.flatten()
        # })
        # output_df.to_excel(output_filename, index=False)
        # os.system('pause')

        self.obs_xt = torch.tensor(np.column_stack([data["xi"], data["ti"]]), dtype=torch.float32).to(device)
        self.obs_c = torch.tensor(data["ci"].values.reshape(-1, 1), dtype=torch.float32).to(device)




    def gradients(self, outputs, inputs):
        return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)


    def loss_IC(self):
        """初始条件损失"""
        c_init = self.net(self.init_xt)
        return torch.mean(torch.square(c_init - self.init_c))

    def loss_BC(self):
        """边界条件损失"""
        c_bound = self.net(self.bound_xt)
        return torch.mean(torch.square(c_bound - self.bound_c))

    def loss_PDE(self):
        """PDE约束损失"""
        self.pde_points.requires_grad = True
        c = self.net(self.pde_points)

        # 计算各阶导数
        dc_grad = self.gradients(c, self.pde_points)[0]
        dc_dt = dc_grad[:, 1:2]
        dc_dx = dc_grad[:, 0:1]

        d2c_dx2 = self.gradients(dc_dx, self.pde_points)[0][:, 0:1]

        # 计算流速
        u= self.pde_u
        du_dx = self.pde_dux

        # 源项
        epsilon = torch.tensor(0.2, dtype=torch.float32).to(device)
        S = self.lambda_m / (torch.sqrt(2 * epsilon ** 2 * torch.pi)) * torch.exp(
            -((self.pde_points[:, 0:1] - self.lambda_x) ** 2) / (2 * epsilon ** 2)
        )
        # PDE残差
        pde = dc_dt + u * dc_dx +c * du_dx - self.diffusion_x * d2c_dx2 - S
        return torch.mean(torch.square(pde))

    def loss_obs(self):
        """观测数据损失"""
        c_pred = self.net(self.obs_xt)
        return torch.mean(torch.square(c_pred - self.obs_c))
    def plot_predict_now(self,epoch):

        xc = torch.arange(x_min, x_max + x_step, x_step, dtype=torch.float32)  # x 方向
        tc = torch.arange(t_min, t_max + t_step, t_step, dtype=torch.float32)  # t 方向
        # 创建网格
        xx, tt = torch.meshgrid(xc, tc, indexing='ij')  # 使用 'ij' 索
        # 将网格数据合并为输入
        xt = torch.cat([xx.reshape(-1, 1), tt.reshape(-1, 1)], dim=1).to(device)


        with torch.no_grad():
            c_pred = self.net(xt)

        # 重塑为 3D 数组
        c_pred = c_pred.reshape(len(xc), len(tc)).detach().cpu().numpy()
        # 创建图形和子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        # 左侧云图
        cax1 = ax1.contourf(xx.numpy(), tt.numpy(), c_pred, levels=50, cmap='viridis')  # 使用转置后的 c_pred
        ax1.set_title(f'Contour plot of predictions at epoch {epoch}')
        ax1.set_xlabel('x (m)')
        ax1.set_ylabel('t (s)')
        cb1 = fig.colorbar(cax1, ax=ax1)
        cb1.set_label('Predicted values')

        # 右侧曲线图
        time_indices = [0.2, 0.4, 0.6, 0.8, 1.0]
        for t in time_indices:
            time_index = int(t_max / t_step * t)
            if time_index < c_pred.shape[0]:  # 确保索引在范围内
                ax2.plot(xc.numpy(), c_pred[:, time_index], label=f'h_Predicted_at_t={t_max * t}s')

        ax2.set_xlabel('x (m)')
        ax2.set_ylabel('Predicted values')
        ax2.set_title('Predictions at different times')
        ax2.legend()

        # 保存图像
        path = './output/with_Source'
        os.makedirs(path, exist_ok=True)
        plt.savefig(os.path.join(path, f'contour_curve_epoch-{epoch}.png'))
        plt.close('all')

    def closure(self):
        self.adam.zero_grad()
        self.lbfgs.zero_grad()

        loss_ic = self.loss_IC()
        loss_bc = self.loss_BC()
        loss_pde = self.loss_PDE()
        loss_obs = self.loss_obs()
        loss = loss_ic + loss_bc + loss_pde + loss_obs

        loss.backward()


        print(
            f"\r{self.iter} loss : {loss.item():.3e},m : {self.lambda_m.item():.5f}, x : {self.lambda_x.item():.5f}",
            end="",
        )
        if self.iter % 500 == 0:
            self.plot_predict_now(self.iter)
            print("")

        # 记录 mse_data, mse_pde 和参数
        self.recorded_data.append({
            'Iteration': self.iter,
            'MSE_Data': loss_obs.item(),
            'MSE_PDE': loss_pde.item(),
            'Lambda_m': self.lambda_m.item(),
            'Lambda_x': self.lambda_x.item(),
        })
        self.iter += 1
        return loss

if __name__ == "__main__":
    pinn = PINN()
    for i in range(10000):
        pinn.closure()
        pinn.adam.step()
    pinn.lbfgs.step(pinn.closure)
    # torch.save(pinn.net.state_dict(), ".weight.pt")
    torch.save(pinn.net.state_dict(), f'model_inverse32.param')
    # 将记录的数据保存到 Excel
    df = pd.DataFrame(pinn.recorded_data)
    df.to_excel("training_log32.xlsx", index=False)