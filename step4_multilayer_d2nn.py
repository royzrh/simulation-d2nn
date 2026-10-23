import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# ============================================================
# 1. 基础设置
# ============================================================

torch.manual_seed(42)

device = torch.device("cpu")
dtype = torch.float32

N = 128
num_layers = 5

wavelength = 532e-9
dx = 4e-6
z = 0.003
w0 = 55e-6

epochs = 180
learning_rate = 0.08

target_radius = 24e-6

# 使用两倍尺寸进行零填充传播
padding_factor = 2

# 结果保存目录
output_dir = Path(__file__).resolve().parent / "results"
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Step 4: Five-Layer Diffractive Neural Network")
print("=" * 60)


# ============================================================
# 2. 构建空间坐标
# ============================================================

x = (
    torch.arange(N, dtype=dtype, device=device)
    - N // 2
) * dx

Y, X = torch.meshgrid(x, x, indexing="ij")


# ============================================================
# 3. 构建两个输入光场
# ============================================================

def gaussian_beam(x_center, y_center):

    U = torch.exp(
        -(
            (X - x_center)**2
            + (Y - y_center)**2
        ) / w0**2
    )

    return U.to(torch.complex64)


# 输入0：左侧高斯光束
U_left = gaussian_beam(
    -75e-6,
    0
)

# 输入1：右侧高斯光束
U_right = gaussian_beam(
    75e-6,
    0
)

# 将两个输入组成一个batch
U_inputs = torch.stack(
    [U_left, U_right],
    dim=0
)

# 每个输入的总光强
input_powers = (
    torch.abs(U_inputs)**2
).sum(dim=(-2, -1))

print("[1/6] Input optical fields created.")


# ============================================================
# 4. 定义输出探测区域
# ============================================================

def circular_detector(x_center, y_center):

    mask = (
        (X - x_center)**2
        + (Y - y_center)**2
        <= target_radius**2
    )

    return mask.to(dtype)


# 探测器0：下方
detector_0 = circular_detector(
    0,
    -75e-6
)

# 探测器1：上方
detector_1 = circular_detector(
    0,
    75e-6
)

detector_masks = torch.stack(
    [detector_0, detector_1],
    dim=0
)

print("[2/6] Output detectors initialized.")


# ============================================================
# 5. 构建自由空间传播算子
# ============================================================

# 传播计算使用2倍尺寸的网格
M = N * padding_factor

fx = torch.fft.fftfreq(
    M,
    d=dx,
    dtype=torch.float64,
    device=device
)

FY, FX = torch.meshgrid(
    fx, fx, indexing="ij"
)

k = 2 * math.pi / wavelength

# 传播波与倏逝波的统一表达
sqrt_argument = (
    1
    - (wavelength * FX)**2
    - (wavelength * FY)**2
)

sqrt_argument = sqrt_argument.to(
    torch.complex128
)

# 移除不影响光强的整体传播相位
H = torch.exp(
    1j
    * k
    * z
    * (torch.sqrt(sqrt_argument) - 1)
)

H = H.to(torch.complex64)


def angular_spectrum(U):

    """
    角谱法自由空间传播。

    使用零填充减小FFT周期边界带来的误差。
    传播后裁剪回原始计算区域。
    """

    pad = (M - N) // 2

    # 零填充
    U_pad = F.pad(
        U,
        (pad, pad, pad, pad)
    )

    # 频域传播
    U_fft = torch.fft.fft2(U_pad)

    U_fft_z = U_fft * H

    Uz_pad = torch.fft.ifft2(U_fft_z)

    # 裁剪回原来的计算区域
    Uz = Uz_pad[
        ...,
        pad:pad + N,
        pad:pad + N
    ]

    return Uz


print("[3/6] Propagation operator initialized.")


# ============================================================
# 6. 定义五层衍射神经网络
# ============================================================

class D2NN(nn.Module):

    def __init__(self, size, layers):

        super().__init__()

        self.num_layers = layers

        # 创建五个独立的可训练相位层
        self.phases = nn.ParameterList()

        for i in range(layers):

            phase = (
                2
                * math.pi
                * torch.rand(
                    size,
                    size,
                    dtype=dtype,
                    device=device
                )
            )

            self.phases.append(
                nn.Parameter(phase)
            )

    def forward(self, U):

        # 光依次通过五层衍射网络
        for phase in self.phases:

            # 相位调制
            phase_mask = torch.exp(
                1j * phase
            )

            U = U * phase_mask

            # 传播到下一层或输出平面
            U = angular_spectrum(U)

        # 输出光强
        intensity = torch.abs(U)**2

        return intensity


# 实例化网络
model = D2NN(
    size=N,
    layers=num_layers
).to(device)

# 优化器
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)

print("[4/6] Five-layer D2NN created.")

print(
    "Trainable parameters:",
    sum(p.numel() for p in model.parameters())
)


# ============================================================
# 7. 计算两个探测器接收的光能量
# ============================================================

def detector_efficiencies(intensity):

    """
    intensity:
        [batch, N, N]

    返回:
        [batch, 2]

    每个元素表示对应探测器接收到的
    光功率占输入光功率的比例。
    """

    detector_powers = (
        intensity[:, None, :, :]
        * detector_masks[None, :, :, :]
    ).sum(dim=(-2, -1))

    efficiencies = (
        detector_powers
        / input_powers[:, None]
    )

    return efficiencies


# ============================================================
# 8. 训练前测试
# ============================================================

with torch.no_grad():

    I_before = model(U_inputs)

    efficiency_before = detector_efficiencies(
        I_before
    )

print("[5/6] Initial network evaluated.")

print()
print("Initial detector efficiencies:")
print(efficiency_before)
print()


# ============================================================
# 9. 定义训练目标
# ============================================================

# 输入0应进入探测器0
# 输入1应进入探测器1

# 效率矩阵:
#
#              Detector 0    Detector 1
#
# Input 0          a              b
#
# Input 1          c              d
#
# 希望 a 和 d 尽可能大
# 希望 b 和 c 尽可能小


def calculate_loss(efficiencies):

    # 正确探测区域的效率
    correct = torch.stack([
        efficiencies[0, 0],
        efficiencies[1, 1]
    ])

    # 错误探测区域的效率
    wrong = torch.stack([
        efficiencies[0, 1],
        efficiencies[1, 0]
    ])

    # 最大化正确探测区域中的光功率
    # 同时抑制错误探测区域的光功率
    loss = (
        1.0
        - correct.mean()
        + 0.3 * wrong.mean()
    )

    return loss, correct, wrong


# ============================================================
# 10. 联合训练五层网络
# ============================================================

loss_history = []

correct_0_history = []
correct_1_history = []

wrong_0_history = []
wrong_1_history = []

print("=" * 60)
print("Training Started")
print("=" * 60)

for epoch in range(1, epochs + 1):

    optimizer.zero_grad()

    # 五层网络前向传播
    I_output = model(U_inputs)

    # 计算探测效率
    efficiencies = detector_efficiencies(
        I_output
    )

    # 计算损失
    loss, correct, wrong = calculate_loss(
        efficiencies
    )

    # 反向传播
    loss.backward()

    # 同时更新五层相位
    optimizer.step()

    # 保存训练记录
    loss_history.append(
        loss.item()
    )

    correct_0_history.append(
        correct[0].item()
    )

    correct_1_history.append(
        correct[1].item()
    )

    wrong_0_history.append(
        wrong[0].item()
    )

    wrong_1_history.append(
        wrong[1].item()
    )

    if epoch == 1 or epoch % 20 == 0:

        print(
            f"Epoch [{epoch:3d}/{epochs}] "
            f"Loss={loss.item():.6f} "
            f"Class0={correct[0].item():.4f} "
            f"Class1={correct[1].item():.4f}",
            flush=True
        )


print("=" * 60)
print("Training Completed")
print("=" * 60)


# ============================================================
# 11. 获取训练后的结果
# ============================================================

with torch.no_grad():

    I_after = model(U_inputs)

    efficiency_after = detector_efficiencies(
        I_after
    )

print("[6/6] Final network evaluated.")


# ============================================================
# 12. 数据转换
# ============================================================

I_input_np = (
    torch.abs(U_inputs)**2
).detach().cpu().numpy()

I_before_np = (
    I_before.detach().cpu().numpy()
)

I_after_np = (
    I_after.detach().cpu().numpy()
)

# 绘图坐标，单位mm
half_width = N * dx * 1e3 / 2

extent = [
    -half_width,
    half_width,
    -half_width,
    half_width
]


# ============================================================
# 13. 绘制输入输出光强
# ============================================================

fig, axes = plt.subplots(
    2, 3,
    figsize=(14, 9)
)

titles = [
    "Input Optical Field",
    "Before Training",
    "After Training"
]

for i in range(2):

    images = [
        I_input_np[i],
        I_before_np[i],
        I_after_np[i]
    ]

    for j in range(3):

        ax = axes[i, j]

        im = ax.imshow(
            images[j],
            cmap="inferno",
            extent=extent,
            origin="lower"
        )

        ax.set_title(
            f"Class {i}: {titles[j]}"
        )

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

        ax.set_aspect("equal")

        fig.colorbar(im, ax=ax)

        # 在输出图中标注对应的目标探测器
        if j > 0:

            y_target = (
                -75e-6 if i == 0 else 75e-6
            )

            circle = Circle(
                (0, y_target * 1e3),
                target_radius * 1e3,
                fill=False,
                color="white",
                linewidth=1.2,
                linestyle="--"
            )

            ax.add_patch(circle)


plt.tight_layout()

output_image = output_dir / "step4_outputs.png"

plt.savefig(
    output_image,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 14. 绘制五层训练后的相位分布
# ============================================================

fig, axes = plt.subplots(
    1, num_layers,
    figsize=(18, 4)
)

for i, phase in enumerate(model.phases):

    phase_display = torch.angle(
        torch.exp(1j * phase.detach())
    ).cpu().numpy()

    im = axes[i].imshow(
        phase_display,
        cmap="twilight",
        extent=extent,
        origin="lower",
        vmin=-math.pi,
        vmax=math.pi
    )

    axes[i].set_title(
        f"Diffractive Layer {i + 1}"
    )

    axes[i].set_xlabel("x (mm)")
    axes[i].set_ylabel("y (mm)")

    axes[i].set_aspect("equal")

    fig.colorbar(im, ax=axes[i])


plt.tight_layout()

phase_image = output_dir / "step4_phases.png"

plt.savefig(
    phase_image,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 15. 绘制训练曲线
# ============================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 4)
)

epoch_axis = range(1, epochs + 1)

# 正确探测效率
axes[0].plot(
    epoch_axis,
    correct_0_history,
    label="Class 0"
)

axes[0].plot(
    epoch_axis,
    correct_1_history,
    label="Class 1"
)

axes[0].set_title(
    "Correct Detector Efficiency"
)

axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Efficiency")

axes[0].set_ylim(0, 1.05)

axes[0].legend()
axes[0].grid(True, alpha=0.3)


# 损失函数
axes[1].plot(
    epoch_axis,
    loss_history
)

axes[1].set_title(
    "Training Loss"
)

axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")

axes[1].grid(True, alpha=0.3)


plt.tight_layout()

training_image = output_dir / "step4_training.png"

plt.savefig(
    training_image,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 16. 保存模型
# ============================================================

model_path = output_dir / "step4_model.pt"

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "num_layers": num_layers,
        "N": N,
        "wavelength": wavelength,
        "dx": dx,
        "z": z,
        "epochs": epochs
    },
    model_path
)


# ============================================================
# 17. 输出最终结果
# ============================================================

print()
print("=" * 60)
print("Final Results")
print("=" * 60)

print()
print("Detector efficiency matrix:")
print(efficiency_after)

print()
print(
    "Class 0 correct efficiency:",
    efficiency_after[0, 0].item()
)

print(
    "Class 1 correct efficiency:",
    efficiency_after[1, 1].item()
)

print()
print("Output files:")

print(output_image)
print(phase_image)
print(training_image)
print(model_path)

print()
print("Simulation completed successfully!")