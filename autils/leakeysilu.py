import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# 创建一个x值的范围
x = np.linspace(-10, 10, 1000)

# 将numpy数组转换为PyTorch张量
x_tensor = torch.from_numpy(x).float()

# 计算LeakyReLU和SiLU函数的值
leakyrelu_y = F.leaky_relu(x_tensor, negative_slope=0.01).numpy()
silu_y = F.silu(x_tensor).numpy()

# 绘制图形
plt.figure(figsize=(8, 6))

# 绘制LeakyReLU函数图像
plt.plot(x, leakyrelu_y, label='LeakyReLU')

# 绘制SiLU函数图像
plt.plot(x, silu_y, label='SiLU (Swish)')

# 设置图例、标题和坐标轴标签
plt.legend()
# plt.title('Comparison of LeakyReLU and SiLU Activation Functions')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)

# 显示图形
plt.show()