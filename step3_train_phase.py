import math
from pathlib import Path

import torch
import torch.nn as nn
import matplotlib

# 使用非交互后端，直接保存图片
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. 基本参数
# ============================================================

torch.manual_seed(42)

N = 256

wavelength = 532e-9       # 波长 532 nm
dx = 4e-6                 # 空间采样间隔 4 μm
z = 0.02                  # 传播距离 20 mm
w0 = 150e-6               # 高斯光束束腰半径 150 μm

target_radius = 30e-6     # 目标聚焦区域半径 30 μm

epochs = 200              # 训练次数
learning_rate = 0.08      # Adam 学习率

# 使用 CPU，保证 macOS 上的复数运算兼容性
device = torch.device("cpu")

# 使用双精度
dtype = torch.float64

print("=" * 60)
print("Step 3: Trainable Diffractive Phase Layer")
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
# 3. 构建输入高斯光束
# ============================================================

U_input = torch.exp(
    -(X**2 + Y**2) / w0**2
).to(torch.complex128)

I_input = torch.abs(U_input)**2

# 输入总光强求和
input_power = I_input.sum()

print("[1/5] Gaussian input beam initialized.")


# ============================================================
# 4. 定义目标聚焦区域
# ============================================================

# 中心半径30 μm的圆形区域
target_mask = (
    X**2 + Y**2 <= target_radius**2
).to(dtype)

print("[2/5] Target region initialized.")


# ============================================================
# 5. 构建角谱传播算子
# ============================================================

fx = torch.fft.fftfreq(
    N,
    d=dx,
    dtype=dtype,
    device=device
)

FY, FX = torch.meshgrid(
    fx, fx, indexing="ij"
)

k = 2 * math.pi / wavelength

# 纵向波矢
kz_squared = (
    1
    - (wavelength * FX)**2
    - (wavelength * FY)**2
)

# 使用复数平方根，允许处理倏逝波
kz = k * torch.sqrt(
    kz_squared.to(torch.complex128)
)

# 传播传递函数
H = torch.exp(1j * kz * z)


def propagate(U):
    """
    使用角谱法计算自由空间传播。
    """

    U_fft = torch.fft.fft2(U)

    Uz = torch.fft.ifft2(U_fft * H)

    return Uz


print("[3/5] Angular spectrum propagator initialized.")


# ============================================================
# 6. 定义可训练衍射层
# ============================================================

class DiffractiveLayer(nn.Module):

    def __init__(self, size):

        super().__init__()

        # 初始化随机相位
        initial_phase = (
            2 * math.pi
            * torch.rand(
                size,
                size,
                dtype=dtype,
                device=device
            )
        )

        # 注册为可训练参数
        self.phase = nn.Parameter(initial_phase)

    def forward(self, U):

        # 纯相位调制
        phase_mask = torch.exp(
            1j * self.phase
        )

        # 光场经过衍射层
        U_modulated = U * phase_mask

        return U_modulated


# 创建衍射层
layer = DiffractiveLayer(N)

# Adam优化器
optimizer = torch.optim.Adam(
    layer.parameters(),
    lr=learning_rate
)

print("[4/5] Trainable phase layer initialized.")


# ============================================================
# 7. 定义完整光学系统
# ============================================================

def optical_system(U):

    # 第一步：相位调制
    U1 = layer(U)

    # 第二步：自由空间传播
    U2 = propagate(U1)

    # 第三步：计算输出光强
    I_out = torch.abs(U2)**2

    return I_out


# ============================================================
# 8. 定义聚焦效率
# ============================================================

def calculate_efficiency(I):

    # 目标区域内的光强求和
    target_power = torch.sum(
        I * target_mask
    )

    # 聚焦效率
    efficiency = target_power / input_power

    return efficiency


# ============================================================
# 9. 保存训练前的结果
# ============================================================

with torch.no_grad():

    initial_phase = layer.phase.detach().clone()

    I_before = optical_system(U_input)

    initial_efficiency = calculate_efficiency(
        I_before
    ).item()

    I_before = I_before.detach().clone()


print()
print(f"Initial efficiency: {initial_efficiency:.6f}")
print()


# ============================================================
# 10. 开始训练
# ============================================================

loss_history = []
efficiency_history = []

print("=" * 60)
print("Training Started")
print("=" * 60)

for epoch in range(1, epochs + 1):

    # 清空上一轮梯度
    optimizer.zero_grad()

    # 前向传播
    I_output = optical_system(U_input)

    # 计算聚焦效率
    efficiency = calculate_efficiency(I_output)

    # 损失函数
    loss = 1.0 - efficiency

    # 反向传播
    loss.backward()

    # Adam优化相位
    optimizer.step()

    # 记录训练数据
    loss_history.append(loss.item())

    efficiency_history.append(efficiency.item())

    # 每20轮打印一次
    if epoch == 1 or epoch % 20 == 0:

        print(
            f"Epoch [{epoch:3d}/{epochs}] "
            f"Loss: {loss.item():.6f} "
            f"Efficiency: {efficiency.item():.6f}",
            flush=True
        )


print("=" * 60)
print("Training Completed")
print("=" * 60)


# ============================================================
# 11. 获取训练后的结果
# ============================================================

with torch.no_grad():

    I_after = optical_system(U_input)

    final_efficiency = calculate_efficiency(
        I_after
    ).item()

    final_phase = layer.phase.detach().clone()

    output_power = I_after.sum().item()


# ============================================================
# 12. 转换为NumPy数据
# ============================================================

I_input_np = I_input.detach().cpu().numpy()

I_before_np = I_before.detach().cpu().numpy()

I_after_np = I_after.detach().cpu().numpy()

# 将相位限制到 [-π, π]
initial_phase_np = torch.angle(
    torch.exp(1j * initial_phase)
).cpu().numpy()

final_phase_np = torch.angle(
    torch.exp(1j * final_phase)
).cpu().numpy()

# 空间坐标转换为mm
extent = [
    x[0].item() * 1e3,
    x[-1].item() * 1e3,
    x[0].item() * 1e3,
    x[-1].item() * 1e3
]


# ============================================================
# 13. 绘制训练结果
# ============================================================

fig, axes = plt.subplots(
    2, 3,
    figsize=(16, 10)
)


def show_image(ax, data, title, cmap="inferno",
               vmin=None, vmax=None):

    im = ax.imshow(
        data,
        cmap=cmap,
        extent=extent,
        origin="lower",
        vmin=vmin,
        vmax=vmax
    )

    ax.set_title(title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal")

    plt.colorbar(im, ax=ax)


# ------------------------------------------------------------
# 第一行：训练前
# ------------------------------------------------------------

show_image(
    axes[0, 0],
    I_input_np,
    "Input Gaussian Beam"
)

show_image(
    axes[0, 1],
    initial_phase_np,
    "Initial Random Phase",
    cmap="twilight",
    vmin=-math.pi,
    vmax=math.pi
)

show_image(
    axes[0, 2],
    I_before_np,
    "Output Before Training"
)


# ------------------------------------------------------------
# 第二行：训练后
# ------------------------------------------------------------

show_image(
    axes[1, 0],
    final_phase_np,
    "Trained Phase Distribution",
    cmap="twilight",
    vmin=-math.pi,
    vmax=math.pi
)

show_image(
    axes[1, 1],
    I_after_np,
    "Output After Training"
)

# 标注目标区域
circle = plt.Circle(
    (0, 0),
    target_radius * 1e3,
    fill=False,
    color="white",
    linewidth=1.2,
    linestyle="--"
)

axes[1, 1].add_patch(circle)


# ------------------------------------------------------------
# 训练曲线
# ------------------------------------------------------------

axes[1, 2].plot(
    range(1, epochs + 1),
    efficiency_history,
    linewidth=2,
    label="Focusing Efficiency"
)

axes[1, 2].set_title(
    "Training Efficiency"
)

axes[1, 2].set_xlabel("Epoch")

axes[1, 2].set_ylabel("Efficiency")

axes[1, 2].set_ylim(0, 1.05)

axes[1, 2].grid(
    True,
    linestyle="--",
    alpha=0.4
)

axes[1, 2].legend()


# ============================================================
# 14. 保存图片
# ============================================================

output_dir = (
    Path(__file__).resolve().parent
    / "results"
)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

image_path = output_dir / "step3_training.png"

plt.tight_layout()

plt.savefig(
    image_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 15. 保存训练后的相位参数
# ============================================================

model_path = output_dir / "step3_trained_phase.pt"

torch.save(
    layer.state_dict(),
    model_path
)


# ============================================================
# 16. 输出最终结果
# ============================================================

print()
print("=" * 60)
print("Final Results")
print("=" * 60)

print(
    f"Initial efficiency : "
    f"{initial_efficiency:.6f}"
)

print(
    f"Final efficiency   : "
    f"{final_efficiency:.6f}"
)

print(
    f"Efficiency gain    : "
    f"{final_efficiency / max(initial_efficiency, 1e-12):.2f}x"
)

print(
    f"Power conservation : "
    f"{output_power / input_power.item():.6f}"
)

print()
print("Figure saved to:")
print(image_path)

print()
print("Model saved to:")
print(model_path)

print()
print("Simulation completed successfully!")