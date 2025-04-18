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
t_max=10
#  Noisy Data Preparation
noise = 0.2


dt = 0.1
dx = 0.1
# 定义上界和下界
ub = np.array([x_max, t_max])
lb = np.array([x_min, t_min])
def getData():
    #init data
    init_x = np.arange(x_min, x_max + x_step, x_step).reshape(-1, 1)
    init_t = np.zeros_like(init_x)
    init_c = 0.0 * np.ones_like(init_x)
    init_xt = np.concatenate([init_x, init_t], axis=1)
    # left boundary data
    left_t = np.arange(t_min, t_max + t_step, t_step).reshape(-1, 1)
    left_x = x_min * np.ones_like(left_t)
    left_c = 0.0 * np.ones_like(left_t)
    left_xt = np.concatenate([left_x, left_t], axis=1)
    # observed data
    data = pd.read_csv("./data3.csv")  # 替换为您的 CSV 文件路径
    obs_xi = data["xi"].values.reshape(-1, 1)  # 取出 xi 列
    obs_ti = data["ti"].values.reshape(-1, 1)  # 取出 ti 列
    obs_ci = data["ci"].values.reshape(-1, 1)  # 取出 ci 列
    noise_ci= obs_ci + noise * np.std(obs_ci) * np.random.randn(*obs_ci.shape)
    output_filename = f'noise_data_with_{noise}.xlsx'
    output_df = pd.DataFrame({
        'xi': obs_xi.flatten(),
        'ti': obs_ti.flatten(),
        'ci': obs_ci.flatten(),
        'noise_ci': noise_ci.flatten()
    })
    output_df.to_excel(output_filename, index=False)
    obs_xt = np.concatenate([obs_xi, obs_ti], axis=1)

    #PDE训练数据

    PDE_x = np.arange(x_min, x_max + x_step, x_step)
    PDE_t = np.arange(t_min, t_max + t_step, t_step)
    input_data = np.array(np.meshgrid(PDE_x, PDE_t)).T.reshape(-1, 2)


    # all obs data (includ boundary and init)
    data_xt=np.concatenate((init_xt, left_xt, obs_xt), axis=0)
    data_c = np.concatenate((init_c, left_c, noise_ci), axis=0)


    # convert to tensor
    data_xt = torch.tensor(data_xt, dtype=torch.float32).to(device)
    data_c = torch.tensor(data_c, dtype=torch.float32).to(device)
    input_data = torch.tensor(input_data, dtype=torch.float32).to(device)

    return data_xt, data_c,input_data

data_xt, data_c,input_data = getData()
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
            lr=0.1,
            max_iter=50000,
            max_eval=50000,
            history_size=50,
            tolerance_grad=1e-5,
            tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )
        self.iter = 0

    def gradients(self, outputs, inputs):
        return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)

    def predict(self, xt):
        if xt.requires_grad == False:
            xt = xt.clone()
            xt.requires_grad = True
        c = self.net(xt)
        return c



    def loss_PDE(self, input_data):
        input_data = input_data.clone()
        input_data.requires_grad = True

        param_m = self.lambda_m
        param_x = self.lambda_x

        c = self.predict(input_data)

        c_out = self.gradients(c, input_data)[0]
        # os.system('pause')
        dc_dx = c_out[:, 0:1]
        dc_dt = c_out[:, 1:2]
        d2c_dx2 = self.gradients(dc_dx, input_data)[0][:, 0:1]
        epsilon = torch.tensor(0.2, dtype=torch.float32).to(device)
        # 源项 S1 在时间 [t1, t1 + dt1] 释放
        S1 = param_m / (torch.sqrt(2 * epsilon ** 2 * torch.pi)) * torch.exp(
            -((input_data[:, 0:1] - param_x) ** 2) / (2 * epsilon ** 2))
        # loss_fn = torch.nn.MSELoss()
        pred_pde = dc_dt + (2.0 + torch.sin(
            input_data[:, 1:2] / 5 * torch.pi)) * dc_dx - self.diffusion_x * d2c_dx2 - S1  # - S2

        mse_PDE = torch.mean(torch.square(pred_pde))
        return mse_PDE

    def loss_data(self, xt):
        c_data = self.predict(xt)
        mse_data = torch.mean(torch.square(c_data - data_c))
        return mse_data

    def closure(self):
        self.adam.zero_grad()
        self.lbfgs.zero_grad()

        mse_data = self.loss_data(data_xt)
        mse_pde = self.loss_PDE(input_data)
        loss = mse_data + mse_pde

        loss.backward()

        self.iter += 1
        print(
            f"\r{self.iter} loss : {loss.item():.3e} l1 : {self.lambda_m.item():.5f}, l2 : {self.lambda_x.item():.5f}",
            end="",
        )
        if self.iter % 500 == 0:
            print("")

        # 记录 mse_data, mse_pde 和参数
        self.recorded_data.append({
            'Iteration': self.iter,
            'MSE_Data': mse_data.item(),
            'MSE_PDE': mse_pde.item(),
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
    torch.save(pinn.net.state_dict(), f'model_with_{noise}noise.param')
    # 将记录的数据保存到 Excel
    df = pd.DataFrame(pinn.recorded_data)
    df.to_excel(f'training_with_{noise}_noise_log.xlsx', index=False)