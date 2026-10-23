import torch
import numpy as np
import matplotlib

# 使用非交互式后端，确保图片可以正常保存
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. 基本物理参数
# ============================================================

N = 256                    # 采样点数

wavelength = 532e-9         # 波长：532 nm
dx = 4e-6                  # 空间采样间隔：4 μm

z = 0.02                   # 传播距离：20 mm
f = 0.02                   # 相位透镜焦距：20 mm

w0 = 150e-6                # 高斯光束束腰半径：150 μm

device = torch.device("cpu")

print("=" * 50)
print("Step 2: Phase Modulation Simulation")
print("=" * 50)


# ============================================================
# 2. 构建二维空间坐标
# ============================================================

x = (
    torch.arange(N, dtype=torch.float64, device=device)
    - N // 2
) * dx

Y, X = torch.meshgrid(x, x, indexing="ij")


# ============================================================
# 3. 构建初始高斯光束
# ============================================================

U0 = torch.exp(
    -(X**2 + Y**2) / w0**2
)

# 转换为复数光场
U0 = U0.to(torch.complex128)

print("[1/5] Initial Gaussian beam generated.")


# ============================================================
# 4. 构建相位调制层
# ============================================================

# 理想薄透镜相位分布
phase = (
    -np.pi
    * (X**2 + Y**2)
    / (wavelength * f)
)

# 相位透射函数
phase_mask = torch.exp(1j * phase)

# 光场经过相位调制层
U1 = U0 * phase_mask

print("[2/5] Phase modulation layer generated.")


# ============================================================
# 5. 定义角谱传播函数
# ============================================================

def angular_spectrum(U, wavelength, dx, z):

    """
    使用角谱法计算光场自由空间传播

    Parameters
    ----------
    U : torch.Tensor
        输入复振幅

    wavelength : float
        光波长（m）

    dx : float
        空间采样间隔（m）

    z : float
        传播距离（m）

    Returns
    -------
    Uz : torch.Tensor
        传播后的复振幅
    """

    ny, nx = U.shape

    # 空间频率坐标
    fx = torch.fft.fftfreq(
        nx, d=dx, device=U.device
    )

    fy = torch.fft.fftfreq(
        ny, d=dx, device=U.device
    )

    FY, FX = torch.meshgrid(
        fy, fx, indexing="ij"
    )

    # 波数
    k = 2 * np.pi / wavelength

    # 纵向波矢平方
    kz_squared = (
        k**2
        - (2 * np.pi * FX)**2
        - (2 * np.pi * FY)**2
    )

    # 转换为复数，正确处理倏逝波
    kz = torch.sqrt(
        kz_squared.to(torch.complex128)
    )

    # 传播传递函数
    H = torch.exp(1j * kz * z)

    # 傅里叶变换
    U_fft = torch.fft.fft2(U)

    # 频域传播
    U_fft_z = U_fft * H

    # 傅里叶逆变换
    Uz = torch.fft.ifft2(U_fft_z)

    return Uz


# ============================================================
# 6. 计算两种传播情况
# ============================================================

# 情况一：自由空间传播
U_free = angular_spectrum(
    U0, wavelength, dx, z
)

# 情况二：经过相位调制层后传播
U_focus = angular_spectrum(
    U1, wavelength, dx, z
)

print("[3/5] Optical propagation completed.")


# ============================================================
# 7. 计算光强和相位
# ============================================================

# 初始光强
I0 = torch.abs(U0)**2

# 自由传播后的光强
I_free = torch.abs(U_free)**2

# 相位调制后的光强
I_focus = torch.abs(U_focus)**2

# 相位分布：[-π, π]
phase_display = torch.angle(phase_mask)

print("[4/5] Intensity distributions calculated.")


# ============================================================
# 8. 数据转换
# ============================================================

I0_np = I0.detach().cpu().numpy()

I_free_np = I_free.detach().cpu().numpy()

I_focus_np = I_focus.detach().cpu().numpy()

phase_np = phase_display.detach().cpu().numpy()

# 空间坐标转换为 mm
extent = [
    x[0].item() * 1e3,
    x[-1].item() * 1e3,
    x[0].item() * 1e3,
    x[-1].item() * 1e3
]


# ============================================================
# 9. 绘制结果
# ============================================================

fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 10)
)

# ------------------------------------------------------------
# 图1：初始高斯光束
# ------------------------------------------------------------

im0 = axes[0, 0].imshow(
    I0_np,
    cmap="inferno",
    extent=extent,
    origin="lower"
)

axes[0, 0].set_title(
    "Input Gaussian Beam",
    fontsize=13
)

plt.colorbar(
    im0,
    ax=axes[0, 0],
    label="Intensity"
)


# ------------------------------------------------------------
# 图2：相位调制层
# ------------------------------------------------------------

im1 = axes[0, 1].imshow(
    phase_np,
    cmap="twilight",
    extent=extent,
    origin="lower",
    vmin=-np.pi,
    vmax=np.pi
)

axes[0, 1].set_title(
    "Phase Modulation Layer",
    fontsize=13
)

plt.colorbar(
    im1,
    ax=axes[0, 1],
    label="Phase (rad)"
)


# ------------------------------------------------------------
# 图3：无相位调制
# ------------------------------------------------------------

im2 = axes[1, 0].imshow(
    I_free_np,
    cmap="inferno",
    extent=extent,
    origin="lower"
)

axes[1, 0].set_title(
    "Without Phase Modulation",
    fontsize=13
)

plt.colorbar(
    im2,
    ax=axes[1, 0],
    label="Intensity"
)


# ------------------------------------------------------------
# 图4：经过相位调制
# ------------------------------------------------------------

im3 = axes[1, 1].imshow(
    I_focus_np,
    cmap="inferno",
    extent=extent,
    origin="lower"
)

axes[1, 1].set_title(
    "With Phase Modulation",
    fontsize=13
)

plt.colorbar(
    im3,
    ax=axes[1, 1],
    label="Intensity"
)


# ------------------------------------------------------------
# 统一设置坐标
# ------------------------------------------------------------

for ax in axes.flat:

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")

    ax.set_aspect("equal")


plt.tight_layout()


# ============================================================
# 10. 保存图片
# ============================================================

# 获取当前脚本所在目录
output_dir = Path(__file__).resolve().parent

# 自动创建结果目录
result_dir = output_dir / "results"
result_dir.mkdir(parents=True, exist_ok=True)

# 图片保存路径
output_path = result_dir / "step2_result.png"

# 保存图片
plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("[5/5] Figure saved successfully.")


# ============================================================
# 11. 输出数值结果
# ============================================================

initial_peak = I0.max().item()

free_peak = I_free.max().item()

focus_peak = I_focus.max().item()

enhancement = focus_peak / free_peak

power_initial = I0.sum().item() * dx**2

power_focus = I_focus.sum().item() * dx**2


print()
print("=" * 50)
print("Simulation Results")
print("=" * 50)

print(f"Initial peak intensity : {initial_peak:.6f}")

print(f"Free peak intensity    : {free_peak:.6f}")

print(f"Focused peak intensity : {focus_peak:.6f}")

print(f"Intensity enhancement  : {enhancement:.4f}")

print()

print(f"Initial total power    : {power_initial:.6e}")

print(f"Focused total power    : {power_focus:.6e}")

print()

print("=" * 50)
print("Output File")
print("=" * 50)

print(output_path)

print()
print("Simulation completed successfully!")