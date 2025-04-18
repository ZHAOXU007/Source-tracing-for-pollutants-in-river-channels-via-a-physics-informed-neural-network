import sys

sys.path.append(".")
import numpy as np
import torch
from torch.autograd import grad
from net import DNN
from scipy.io import loadmat
import pandas as pd
import os
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

x_step = 0.1
t_step = 0.1
x_min=0
x_max=20
t_min=0
t_max=15

# 定义上界和下界
ub = np.array([x_max, t_max])
lb = np.array([x_min, t_min])

class PINN:
    def __init__(self):
        self.diffusion_x = 0.1

        self.lambda_m = torch.tensor([1.0], requires_grad=True).to(device)
        self.lambda_x = torch.tensor([1.0], requires_grad=True).to(device)

        self.lambda_m = torch.nn.Parameter(self.lambda_m)
        self.lambda_x = torch.nn.Parameter(self.lambda_x)

        self.recorded_data = []  # 用于存储记录的数据
        self.net = DNN(dim_in=2, dim_out=1, n_layer=6, n_node=20, ub=ub, lb=lb).to(
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
        x = np.arange(x_min, x_max + x_step, x_step)
        t = np.arange(t_min, t_max + t_step, t_step)
        X, T = np.meshgrid(x, t)
        self.pde_points = torch.tensor(np.stack([X.flatten(), T.flatten()], axis=1), dtype=torch.float32).to(device)

        # 观测数据
        data = pd.read_csv("./data3.csv")
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
        u = 2.0 + torch.sin((self.pde_points[:, 1:2] / 5 ) * torch.pi)
        du_dx = self.gradients(u, self.pde_points)[0][:, 0:1]
        # 源项
        epsilon = torch.tensor(0.2, dtype=torch.float32).to(device)
        S = self.lambda_m / (torch.sqrt(2 * epsilon ** 2 * torch.pi)) * torch.exp(
            -((self.pde_points[:, 0:1] - self.lambda_x) ** 2) / (2 * epsilon ** 2)
        )

        # PDE残差
        pde = dc_dt + u * dc_dx+c*du_dx - self.diffusion_x * d2c_dx2 - S
        return torch.mean(torch.square(pde))


    def loss_obs(self):
        """观测数据损失"""
        c_pred = self.net(self.obs_xt)
        return torch.mean(torch.square(c_pred - self.obs_c))


    def closure(self):
        self.adam.zero_grad()
        self.lbfgs.zero_grad()

        loss_ic = self.loss_IC()
        loss_bc = self.loss_BC()
        loss_pde = self.loss_PDE()
        loss_obs = self.loss_obs()
        loss = loss_ic + loss_bc + loss_pde+loss_obs

        loss.backward()

        self.iter += 1
        print(
            f"\r{self.iter} loss : {loss.item():.3e},m : {self.lambda_m.item():.5f}, x : {self.lambda_x.item():.5f}",
            end="",
        )
        if self.iter % 500 == 0:
            print("")

        # 记录 mse_data, mse_pde 和参数
        self.recorded_data.append({
            'Iteration': self.iter,
            'MSE_Data': loss_obs.item(),
            'MSE_PDE': loss_pde.item(),
            'Lambda_m': self.lambda_m.item(),
            'Lambda_x': self.lambda_x.item(),

        })
        return loss

if __name__ == "__main__":
    pinn = PINN()
    for i in range(10000):
        pinn.closure()
        pinn.adam.step()
    pinn.lbfgs.step(pinn.closure)
    #torch.save(pinn.net.state_dict(), ".weight.pt")
    torch.save(pinn.net.state_dict(), f'model.param')
    # 将记录的数据保存到 Excel
    df = pd.DataFrame(pinn.recorded_data)
    df.to_excel("training_log.xlsx", index=False)