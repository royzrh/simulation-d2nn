import math
from pathlib import Path

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. 原论文光学参数
# ============================================================

C = 299792458.0
FREQUENCY = 0.4e12

WAVELENGTH = C / FREQUENCY

N = 200
DX = 400e-6
Z = 30e-3

# 测试用高斯光束束腰
W0 = 3e-3

DTYPE = torch.float64
DEVICE = torch.device("cpu")

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "results" / "step6_1"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Step 6.1 - Angular Spectrum Validation")
print("=" * 60)

print(f"Frequency: {FREQUENCY / 1e12:.3f} THz")
print(f"Wavelength: {WAVELENGTH * 1e6:.3f} um")
print(f"Sampling interval: {DX * 1e6:.1f} um")
print(f"Grid size: {N} x {N}")
print(f"Propagation distance: {Z * 1e3:.1f} mm")


# ============================================================
# 2. 构建二维坐标
# ============================================================

x = (
    torch.arange(N, dtype=DTYPE, device=DEVICE)
    - (N - 1) / 2
) * DX

Y, X = torch.meshgrid(
    x, x, indexing="ij"
)


# ============================================================
# 3. 构建高斯输入光场
# ============================================================

def gaussian_beam(x_center=0.0, w0=W0):

    amplitude = torch.exp(
        -(
            (X - x_center)**2 + Y**2
        ) / w0**2
    )

    return amplitude.to(torch.complex128)


U0 = gaussian_beam()

# 额外构建一个靠近窗口边缘的光束
# 用于检查FFT周期边界影响
U_edge = gaussian_beam(
    x_center=33e-3,
    w0=4e-3
)

print("\n[1/5] Input fields generated.")


# ============================================================
# 4. 角谱传播函数
# ============================================================

def angular_spectrum(U, z, padding_factor=2):

    """
    Angular Spectrum Method (ASM)

    参数：
        U：输入复振幅 [N, N]
        z：传播距离
        padding_factor：零填充倍数

    返回：
        Uz_crop：原始窗口内的输出光场
        Uz_full：完整计算窗口内的输出光场
    """

    ny, nx = U.shape

    assert ny == nx
    assert padding_factor >= 1

    n = nx
    m = n * padding_factor

    pad_left = (m - n) // 2
    pad_right = m - n - pad_left

    # --------------------------------------------------------
    # 零填充
    # --------------------------------------------------------

    U_pad = F.pad(
        U,
        (
            pad_left,
            pad_right,
            pad_left,
            pad_right
        )
    )

    # --------------------------------------------------------
    # 空间频率
    # --------------------------------------------------------

    freq = torch.fft.fftfreq(
        m,
        d=DX,
        device=U.device,
        dtype=DTYPE
    )

    FY, FX = torch.meshgrid(
        freq, freq, indexing="ij"
    )

    # --------------------------------------------------------
    # 纵向波矢
    # --------------------------------------------------------

    k = 2 * math.pi / WAVELENGTH

    argument = (
        1.0
        - (WAVELENGTH * FX)**2
        - (WAVELENGTH * FY)**2
    )

    # 使用复数平方根处理倏逝波
    kz = k * torch.sqrt(
        argument.to(torch.complex128)
    )

    # --------------------------------------------------------
    # 传播传递函数
    # --------------------------------------------------------

    H = torch.exp(
        1j * kz * z
    )

    # --------------------------------------------------------
    # FFT -> 传播 -> IFFT
    # --------------------------------------------------------

    spectrum = torch.fft.fft2(U_pad)

    Uz_full = torch.fft.ifft2(
        spectrum * H
    )

    # --------------------------------------------------------
    # 裁剪回原始窗口
    # --------------------------------------------------------

    Uz_crop = Uz_full[
        pad_left:pad_left + n,
        pad_left:pad_left + n
    ]

    return Uz_crop, Uz_full


print("[2/5] ASM propagator initialized.")


# ============================================================
# 5. 定义光功率和误差计算
# ============================================================

def optical_power(U):

    return (
        torch.abs(U)**2
    ).sum() * DX**2


def relative_field_error(U, U_ref):

    numerator = torch.linalg.vector_norm(
        U - U_ref
    )

    denominator = torch.linalg.vector_norm(
        U_ref
    )

    return (
        numerator / denominator
    ).item()


# ============================================================
# 6. 检验一：零距离传播
# ============================================================

U_zero, _ = angular_spectrum(
    U0,
    z=0.0,
    padding_factor=2
)

zero_error = relative_field_error(
    U_zero,
    U0
)

print("\n" + "=" * 60)
print("Test 1: Zero-distance propagation")
print("=" * 60)

print(f"Relative error: {zero_error:.6e}")


# ============================================================
# 7. 检验二：纯相位调制的功率守恒
# ============================================================

phase = (
    0.8 * torch.sin(
        2 * math.pi * X / 12e-3
    )
)

phase_mask = torch.exp(
    1j * phase
)

U_phase = U0 * phase_mask

P_input = optical_power(U0).item()
P_phase = optical_power(U_phase).item()

phase_power_error = abs(
    P_phase - P_input
) / P_input

print("\n" + "=" * 60)
print("Test 2: Phase modulation power")
print("=" * 60)

print(f"Input power: {P_input:.8e}")
print(f"After phase mask: {P_phase:.8e}")
print(f"Relative error: {phase_power_error:.6e}")


# ============================================================
# 8. 检验三：传播后的功率变化
# ============================================================

Uz, Uz_full = angular_spectrum(
    U0,
    z=Z,
    padding_factor=2
)

P_full = optical_power(Uz_full).item()
P_crop = optical_power(Uz).item()

full_ratio = P_full / P_input
crop_ratio = P_crop / P_input

print("\n" + "=" * 60)
print("Test 3: Propagation power")
print("=" * 60)

print(f"Input power: {P_input:.8e}")
print(f"Full-grid power: {P_full:.8e}")
print(f"Cropped-grid power: {P_crop:.8e}")

print(f"Full-grid ratio: {full_ratio:.8f}")
print(f"Cropped-grid ratio: {crop_ratio:.8f}")

print(
    f"Power outside original window: "
    f"{(P_full - P_crop) / P_input:.6e}"
)


# ============================================================
# 9. 检验四：零填充收敛性
# ============================================================

# 使用靠近边界的高斯光束
# 更容易观察FFT周期回绕问题

padding_results = {}

for factor in [1, 2, 3, 4]:

    U_result, _ = angular_spectrum(
        U_edge,
        z=Z,
        padding_factor=factor
    )

    padding_results[factor] = U_result

# 以4倍零填充结果为参考
U_reference = padding_results[4]

print("\n" + "=" * 60)
print("Test 4: Padding convergence")
print("=" * 60)

for factor in [1, 2, 3]:

    error = relative_field_error(
        padding_results[factor],
        U_reference
    )

    print(
        f"Padding x{factor} vs x4: "
        f"{error:.6e}"
    )


# ============================================================
# 10. 检验五：PyTorch自动求导
# ============================================================

# 定义一个可训练的相位倾斜系数
theta = torch.tensor(
    0.35,
    dtype=DTYPE,
    requires_grad=True
)


def center_of_mass(theta_value):

    """
    使用线性相位倾斜测试梯度。

    返回输出光强的x方向质心，单位mm。
    """

    tilted_field = U0 * torch.exp(
        1j * theta_value * X / W0
    )

    U_out, _ = angular_spectrum(
        tilted_field,
        z=Z,
        padding_factor=2
    )

    intensity = torch.abs(U_out)**2

    x_mm = X * 1e3

    centroid = (
        (intensity * x_mm).sum()
        / intensity.sum()
    )

    return centroid


# 自动微分
loss = center_of_mass(theta)

gradient_auto = torch.autograd.grad(
    loss,
    theta
)[0].item()

# 有限差分
epsilon = 1e-4

with torch.no_grad():

    theta_plus = torch.tensor(
        theta.item() + epsilon,
        dtype=DTYPE
    )

    theta_minus = torch.tensor(
        theta.item() - epsilon,
        dtype=DTYPE
    )

    loss_plus = center_of_mass(
        theta_plus
    ).item()

    loss_minus = center_of_mass(
        theta_minus
    ).item()

gradient_fd = (
    loss_plus - loss_minus
) / (2 * epsilon)

gradient_error = abs(
    gradient_auto - gradient_fd
) / max(abs(gradient_fd), 1e-12)

print("\n" + "=" * 60)
print("Test 5: Gradient validation")
print("=" * 60)

print(f"Autograd: {gradient_auto:.8e}")
print(f"Finite difference: {gradient_fd:.8e}")
print(f"Relative error: {gradient_error:.6e}")


# ============================================================
# 11. 绘制结果
# ============================================================

extent = [
    -N * DX * 1e3 / 2,
    N * DX * 1e3 / 2,
    -N * DX * 1e3 / 2,
    N * DX * 1e3 / 2
]

I0 = torch.abs(U0)**2
Iz = torch.abs(Uz)**2

I_edge_1 = (
    torch.abs(padding_results[1])**2
)

I_edge_4 = (
    torch.abs(padding_results[4])**2
)

fig, axes = plt.subplots(
    2, 2,
    figsize=(12, 10)
)

images = [
    I0,
    Iz,
    I_edge_1,
    I_edge_4
]

titles = [
    "Input Gaussian Beam",
    "After 30 mm Propagation",
    "Edge Beam: Padding x1",
    "Edge Beam: Padding x4"
]

for ax, data, title in zip(
    axes.flat,
    images,
    titles
):

    im = ax.imshow(
        data.detach().cpu().numpy(),
        cmap="inferno",
        extent=extent,
        origin="lower"
    )

    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")

    fig.colorbar(
        im,
        ax=ax
    )

plt.tight_layout()

output_path = (
    OUTPUT_DIR / "asm_validation.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("\n[3/5] Validation figures generated.")


# ============================================================
# 12. 输出验证结果
# ============================================================

print("\n" + "=" * 60)
print("Validation Summary")
print("=" * 60)

print(
    f"Zero-distance error: "
    f"{zero_error:.3e}"
)

print(
    f"Phase power error: "
    f"{phase_power_error:.3e}"
)

print(
    f"Full-grid power ratio: "
    f"{full_ratio:.8f}"
)

print(
    f"Cropped-grid power ratio: "
    f"{crop_ratio:.8f}"
)

print(
    f"Gradient relative error: "
    f"{gradient_error:.3e}"
)

print("\n[4/5] Numerical validation completed.")

print("\nOutput figure:")
print(output_path)

print("\n[5/5] Simulation completed successfully!")