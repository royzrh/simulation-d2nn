import torch
import numpy as np
import matplotlib.pyplot as plt

# =====================================
# 1. 基本物理参数
# =====================================

N = 256                  # 采样点数
wavelength = 532e-9       # 波长 532 nm
dx = 4e-6                 # 空间采样间隔 4 μm
z = 0.02                  # 传播距离 20 mm

device = torch.device("cpu")

# =====================================
# 2. 构建二维空间坐标
# =====================================

x = (torch.arange(N, device=device) - N // 2) * dx
X, Y = torch.meshgrid(x, x, indexing="ij")

# =====================================
# 3. 生成高斯光束
# =====================================

w0 = 50e-6                # 光束束腰半径 50 μm

U0 = torch.exp(-(X**2 + Y**2) / w0**2)

# 转换为复数光场
U0 = U0.to(torch.complex64)

# =====================================
# 4. 定义角谱传播函数
# =====================================

def angular_spectrum(U, wavelength, dx, z):

    N = U.shape[0]

    # 空间频率
    fx = torch.fft.fftfreq(N, d=dx)
    FX, FY = torch.meshgrid(fx, fx, indexing="ij")

    # 波数
    k = 2 * torch.pi / wavelength

    # 纵向波矢平方
    kz_squared = (
        k**2
        - (2 * torch.pi * FX)**2
        - (2 * torch.pi * FY)**2
    )

    # 本例采样条件下无倏逝波分量
    kz = torch.sqrt(torch.clamp(kz_squared, min=0))

    # 传播传递函数
    H = torch.exp(1j * kz * z)

    # 傅里叶变换
    U_fft = torch.fft.fft2(U)

    # 频域传播
    U_fft_z = U_fft * H

    # 逆傅里叶变换
    Uz = torch.fft.ifft2(U_fft_z)

    return Uz

# =====================================
# 5. 计算传播后的光场
# =====================================

Uz = angular_spectrum(U0, wavelength, dx, z)

# 计算光强
I0 = torch.abs(U0)**2
Iz = torch.abs(Uz)**2

# =====================================
# 6. 可视化
# =====================================

extent = [
    x[0].item() * 1e3,
    x[-1].item() * 1e3,
    x[0].item() * 1e3,
    x[-1].item() * 1e3
]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(I0.numpy(), cmap="inferno", extent=extent)
plt.title("Input Intensity")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(Iz.numpy(), cmap="inferno", extent=extent)
plt.title("After Propagation")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.colorbar()

plt.tight_layout()
plt.savefig("step1_result.png", dpi=300)
plt.show()

print("Propagation simulation completed.")