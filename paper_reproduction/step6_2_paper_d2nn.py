import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. 原论文物理参数
# ============================================================

SEED = 42
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")

# 太赫兹光学参数
C = 299792458.0
FREQUENCY = 0.4e12
WAVELENGTH = C / FREQUENCY

# 五层衍射神经网络
NUM_LAYERS = 5

# 每层200×200个衍射神经元
N = 200

# 单个衍射神经元的物理尺寸
NEURON_SIZE = 400e-6

# 相邻衍射层传播距离
LAYER_DISTANCE = 30e-3

# 最后一层到探测平面的距离
OUTPUT_DISTANCE = 30e-3

# 输入光场到第一层的距离
# 本阶段将输入定义在第一层入射平面
# 原论文的完整入射几何将在后续核对
INPUT_DISTANCE = 0.0

# 每个衍射神经元边长上的数值采样点数
# 首次运行建议使用1
# 后续可改为2进行采样敏感性测试
OVERSAMPLE = 3

# FFT零填充倍数
PAD_FACTOR = 2

# 数值采样间隔
DX = NEURON_SIZE / OVERSAMPLE

# 有效光学网格尺寸
ACTIVE_N = N * OVERSAMPLE

# 完整传播计算网格尺寸
GRID_N = ACTIVE_N * PAD_FACTOR

# 两侧填充宽度
PAD = (GRID_N - ACTIVE_N) // 2

# 演示用高斯光束束腰
# 这不是论文实验中入射光源的完整模型
BEAM_WAIST = 10e-3

# 输出目录
ROOT = Path(__file__).resolve().parent

OUTPUT_DIR = ROOT / "results" / "step6_2"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 65)
print("Step 6.2: Paper-Based Five-Layer D2NN")
print("=" * 65)

print(f"Frequency: {FREQUENCY / 1e12:.3f} THz")
print(f"Wavelength: {WAVELENGTH * 1e6:.3f} um")
print(f"Number of layers: {NUM_LAYERS}")
print(f"Neurons per layer: {N} x {N}")

print(f"Neuron size: {NEURON_SIZE * 1e6:.1f} um")
print(f"Numerical sampling: {DX * 1e6:.1f} um")

print(f"Active grid: {ACTIVE_N} x {ACTIVE_N}")
print(f"FFT grid: {GRID_N} x {GRID_N}")

print(f"Layer distance: {LAYER_DISTANCE * 1e3:.1f} mm")


# ============================================================
# 2. 构建完整传播网格
# ============================================================

x = (
    torch.arange(
        GRID_N,
        dtype=torch.float64,
        device=DEVICE
    )
    - GRID_N // 2
) * DX

Y, X = torch.meshgrid(
    x,
    x,
    indexing="ij"
)


# ============================================================
# 3. 构建有限尺寸衍射层孔径
# ============================================================

# 有效衍射层区域为80 mm × 80 mm
active_aperture = torch.ones(
    ACTIVE_N,
    ACTIVE_N,
    dtype=torch.float32,
    device=DEVICE
)

# 填充到完整计算网格
aperture = F.pad(
    active_aperture,
    (PAD, PAD, PAD, PAD)
)

# 有效区域切片
active_slice = slice(
    PAD,
    PAD + ACTIVE_N
)

print("\n[1/6] Optical aperture initialized.")


# ============================================================
# 4. 构建角谱传播传递函数
# ============================================================

freq = torch.fft.fftfreq(
    GRID_N,
    d=DX,
    dtype=torch.float64,
    device=DEVICE
)

FY, FX = torch.meshgrid(
    freq,
    freq,
    indexing="ij"
)

k = 2 * math.pi / WAVELENGTH

# 角谱传播根号项
argument = (
    1.0
    - (WAVELENGTH * FX)**2
    - (WAVELENGTH * FY)**2
)

# 复数平方根，兼容传播波与倏逝波
kz_relative = (
    torch.sqrt(
        argument.to(torch.complex128)
    )
    - 1.0
)


def build_transfer_function(distance):

    """
    构建角谱传播传递函数。

    移除不影响光强的整体传播相位。

    对正向传播，倏逝波分量自然衰减。
    """

    H = torch.exp(
        1j * k * distance * kz_relative
    )

    return H.to(torch.complex64)


H_LAYER = build_transfer_function(
    LAYER_DISTANCE
)

H_OUTPUT = build_transfer_function(
    OUTPUT_DISTANCE
)

H_INPUT = (
    build_transfer_function(INPUT_DISTANCE)
    if INPUT_DISTANCE > 0
    else None
)

print("[2/6] Angular spectrum transfer functions ready.")


# ============================================================
# 5. 定义五层D2NN模型
# ============================================================

class PaperD2NN(nn.Module):

    def __init__(self):

        super().__init__()

        # 原始可训练参数
        # 五层，每层200×200
        self.raw_phases = nn.Parameter(
            0.7 * torch.randn(
                NUM_LAYERS,
                N,
                N,
                dtype=torch.float32,
                device=DEVICE
            )
        )

        # 保存传播算子
        self.register_buffer(
            "H_layer",
            H_LAYER
        )

        self.register_buffer(
            "H_output",
            H_OUTPUT
        )

        if H_INPUT is not None:
            self.register_buffer(
                "H_input",
                H_INPUT
            )
        else:
            self.H_input = None

        # 孔径
        self.register_buffer(
            "aperture",
            aperture
        )

    def get_phase(self, layer_index):

        """
        将原始参数映射到0～π。

        返回200×200物理神经元相位矩阵。
        """

        raw = self.raw_phases[layer_index]

        phase = math.pi * torch.sigmoid(raw)

        return phase

    def get_upsampled_phase(self, layer_index):

        """
        将物理神经元相位映射到更细的数值网格。

        每个物理神经元内部保持相同相位。
        """

        phase = self.get_phase(layer_index)

        phase = phase.repeat_interleave(
            OVERSAMPLE,
            dim=0
        )

        phase = phase.repeat_interleave(
            OVERSAMPLE,
            dim=1
        )

        # 填充到完整传播网格
        phase_full = F.pad(
            phase,
            (PAD, PAD, PAD, PAD)
        )

        return phase_full

    def propagate(self, U, H):

        """
        自由空间传播。

        输入：
            [B, GRID_N, GRID_N]
        """

        spectrum = torch.fft.fft2(
            U,
            dim=(-2, -1)
        )

        Uz = torch.fft.ifft2(
            spectrum * H,
            dim=(-2, -1)
        )

        return Uz

    def forward(self, U, record=False):

        """
        五层D2NN前向传播。

        U:
            [B, GRID_N, GRID_N]

        record=True时记录各层光强和功率。
        """

        snapshots = []
        power_records = []

        if record:

            snapshots.append(
                torch.abs(
                    U[0, active_slice, active_slice]
                ).square().detach().cpu()
            )

        # 输入到第一层
        if self.H_input is not None:

            U = self.propagate(
                U,
                self.H_input
            )

        # ====================================================
        # 五层衍射传播
        # ====================================================

        for i in range(NUM_LAYERS):

            # 调制前完整计算窗口内的光功率
            if record:

                power_before = (
                    torch.abs(U)**2
                ).sum().item() * DX**2

            # 获取当前层相位
            phase = self.get_upsampled_phase(i)

            # 相位透射函数
            transmission = torch.exp(
                1j * phase
            )

            # 有限孔径 + 相位调制
            U = (
                U
                * self.aperture
                * transmission
            )

            if record:

                power_after = (
                    torch.abs(U)**2
                ).sum().item() * DX**2

            # 层间传播
            # 第五层传播至探测平面
            if i < NUM_LAYERS - 1:

                U = self.propagate(
                    U,
                    self.H_layer
                )

            else:

                U = self.propagate(
                    U,
                    self.H_output
                )

            if record:

                power_propagated = (
                    torch.abs(U)**2
                ).sum().item() * DX**2

                power_records.append({
                    "layer": i + 1,
                    "before": power_before,
                    "after_mask": power_after,
                    "after_propagation": power_propagated
                })

                snapshots.append(
                    torch.abs(
                        U[0, active_slice, active_slice]
                    ).square().detach().cpu()
                )

        if record:

            return U, snapshots, power_records

        return U


# ============================================================
# 6. 初始化五层网络
# ============================================================

model = PaperD2NN().to(DEVICE)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

print("[3/6] Five-layer D2NN initialized.")

print(
    "Trainable parameters:",
    parameter_count
)


# ============================================================
# 7. 构建演示输入光束
# ============================================================

amplitude = torch.exp(
    -(X**2 + Y**2) / BEAM_WAIST**2
)

# 有限输入孔径
amplitude = amplitude * aperture

# 转换成复振幅
U_input = amplitude.to(
    torch.complex64
)

# 增加batch维度
U_input = U_input.unsqueeze(0)

input_power = (
    torch.abs(U_input)**2
).sum().item() * DX**2

print("[4/6] Gaussian input field generated.")


# ============================================================
# 8. 执行五层网络前向传播
# ============================================================

with torch.no_grad():

    U_output, snapshots, power_records = model(
        U_input,
        record=True
    )

I_output = torch.abs(U_output)**2

output_power = I_output.sum().item() * DX**2

window_efficiency = (
    output_power / input_power
)

print("[5/6] Five-layer forward propagation completed.")


# ============================================================
# 9. 检查相位范围
# ============================================================

print()
print("=" * 65)
print("Phase Range Validation")
print("=" * 65)

for i in range(NUM_LAYERS):

    phase = model.get_phase(i).detach()

    phase_min = phase.min().item()
    phase_max = phase.max().item()

    print(
        f"Layer {i+1}: "
        f"min={phase_min:.6f}, "
        f"max={phase_max:.6f}"
    )

    assert phase_min >= 0
    assert phase_max <= math.pi


# ============================================================
# 10. 检查各层光功率
# ============================================================

print()
print("=" * 65)
print("Optical Power Budget")
print("=" * 65)

print(f"Input power: {input_power:.8e}")

for record in power_records:

    print(
        f"Layer {record['layer']}: "
        f"Before={record['before']:.6e}, "
        f"After Mask={record['after_mask']:.6e}, "
        f"After Propagation={record['after_propagation']:.6e}"
    )

print()
print(f"Output power: {output_power:.8e}")

print(
    f"Output/Input power ratio: "
    f"{window_efficiency:.6f}"
)


# ============================================================
# 11. 检查五层网络的梯度
# ============================================================

# 定义一个演示用目标区域
# 中心位于x=8 mm、y=0
# 半径为4 mm
# 该区域仅用于梯度测试，并非论文探测器布局

target_mask = (
    (X - 8e-3)**2 + Y**2
    <= (4e-3)**2
).float()

model.zero_grad()

# 重新执行前向传播，保留计算图
U_grad = model(U_input)

I_grad = torch.abs(U_grad)**2

target_power = (
    I_grad * target_mask
).sum() * DX**2

# 目标：提高指定区域中的光功率
loss = -target_power / input_power

loss.backward()

grad = model.raw_phases.grad

print()
print("=" * 65)
print("Gradient Validation")
print("=" * 65)

for i in range(NUM_LAYERS):

    grad_norm = torch.linalg.vector_norm(
        grad[i]
    ).item()

    print(
        f"Layer {i+1} gradient norm: "
        f"{grad_norm:.8e}"
    )

assert torch.isfinite(grad).all()

print("All phase gradients are finite.")


# ============================================================
# 12. 绘制五层相位分布
# ============================================================

fig, axes = plt.subplots(
    1,
    NUM_LAYERS,
    figsize=(18, 4)
)

for i in range(NUM_LAYERS):

    phase = (
        model.get_phase(i)
        .detach()
        .cpu()
        .numpy()
    )

    im = axes[i].imshow(
        phase,
        cmap="twilight",
        vmin=0,
        vmax=math.pi,
        extent=[-40, 40, -40, 40],
        origin="lower"
    )

    axes[i].set_title(
        f"Diffractive Layer {i+1}"
    )

    axes[i].set_xlabel("x (mm)")
    axes[i].set_ylabel("y (mm)")

fig.colorbar(
    im,
    ax=axes.tolist(),
    shrink=0.7,
    label="Phase (rad)"
)

plt.savefig(
    OUTPUT_DIR / "phase_layers.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 13. 绘制逐层传播的光强
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(14, 9)
)

titles = [
    "Input Optical Field",
    "After Layer 1",
    "After Layer 2",
    "After Layer 3",
    "After Layer 4",
    "Output After Layer 5"
]

for ax, intensity, title in zip(
    axes.flat,
    snapshots,
    titles
):

    im = ax.imshow(
        intensity.numpy(),
        cmap="inferno",
        extent=[-40, 40, -40, 40],
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

plt.savefig(
    OUTPUT_DIR / "field_propagation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 14. 绘制光功率变化
# ============================================================

power_values = [input_power]

for record in power_records:

    power_values.append(
        record["after_propagation"]
    )

normalized_power = [
    p / input_power
    for p in power_values
]

fig, ax = plt.subplots(
    figsize=(9, 5)
)

ax.plot(
    range(NUM_LAYERS + 1),
    normalized_power,
    marker="o",
    linewidth=2
)

ax.set_xticks(
    range(NUM_LAYERS + 1)
)

ax.set_xticklabels([
    "Input",
    "Layer 1",
    "Layer 2",
    "Layer 3",
    "Layer 4",
    "Output"
])

ax.set_xlabel("Propagation Stage")
ax.set_ylabel("Normalized Optical Power")

ax.set_title(
    "Optical Power Through Five-Layer D2NN"
)

ax.grid(
    True,
    alpha=0.3
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "power_budget.png",
    dpi=300
)

plt.close(fig)


# ============================================================
# 15. 保存模型参数
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),

        "frequency": FREQUENCY,
        "wavelength": WAVELENGTH,

        "num_layers": NUM_LAYERS,

        "neurons_per_layer": N,
        "neuron_size": NEURON_SIZE,

        "layer_distance": LAYER_DISTANCE,
        "output_distance": OUTPUT_DISTANCE,
        "input_distance": INPUT_DISTANCE,

        "oversample": OVERSAMPLE,
        "padding_factor": PAD_FACTOR,

        "parameter_count": parameter_count
    },
    OUTPUT_DIR / "paper_d2nn_initial.pt"
)


# ============================================================
# 16. 完成
# ============================================================

print()
print("=" * 65)
print("Final Summary")
print("=" * 65)

print(f"Trainable parameters: {parameter_count}")

print(
    f"Output power ratio: "
    f"{window_efficiency:.6f}"
)

print(
    f"Gradient test loss: "
    f"{loss.item():.8e}"
)

print()
print("Results saved to:")
print(OUTPUT_DIR)

print()
print("[6/6] Step 6.2 completed successfully!")