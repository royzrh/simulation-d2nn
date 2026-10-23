import math
import csv
import json
import copy
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Circle


# ============================================================
# 1. 实验配置
# ============================================================

# ============================================================
# 1. 实验配置
# ============================================================

SEED = 42
torch.manual_seed(SEED)

TRAIN_MODE = "baseline"

DEVICE = torch.device("cpu")

# 光学系统
N = 128
NUM_LAYERS = 5

WAVELENGTH = 532e-9
DX = 4e-6
Z = 0.003

PAD_FACTOR = 2

# 训练参数
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.005

TRAIN_LIMIT = 30000
VAL_LIMIT = 5000
TEST_LIMIT = 10000

NUM_CLASSES = 10

CLASS9_WEIGHT = 1.5
MARGIN = 0.3
MARGIN_WEIGHT = 0.1

EFFICIENCY_WEIGHT = 0.05

EPS = 1e-8

# 项目路径
ROOT = Path(__file__).resolve().parent

# 每次运行创建独立目录，避免覆盖
RUN_ID = datetime.now().strftime(
    "%Y%m%d_%H%M%S_%f"
)

RESULT_DIR = (
    ROOT
    / "results"
    / "step5"
    / f"{TRAIN_MODE}_{RUN_ID}"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

print("=" * 65)
print("D2NN MNIST Classification")
print("=" * 65)

print("Training mode:", TRAIN_MODE)
print("Device:", DEVICE)
print("Number of layers:", NUM_LAYERS)
print("Epochs:", EPOCHS)
print("Output:", RESULT_DIR)


# ============================================================
# 2. 保存实验配置
# ============================================================

config = {
    "seed": SEED,
    "train_mode": TRAIN_MODE,
    "device": str(DEVICE),

    "N": N,
    "num_layers": NUM_LAYERS,

    "wavelength": WAVELENGTH,
    "dx": DX,
    "z": Z,

    "pad_factor": PAD_FACTOR,

    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,

    "train_limit": TRAIN_LIMIT,
    "val_limit": VAL_LIMIT,
    "test_limit": TEST_LIMIT,

    "class9_weight": CLASS9_WEIGHT,
    "margin": MARGIN,
    "margin_weight": MARGIN_WEIGHT,
    "efficiency_weight": EFFICIENCY_WEIGHT
}

with open(
    RESULT_DIR / "config.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        config,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# 3. 加载MNIST数据
# ============================================================

transform = transforms.ToTensor()

train_full = datasets.MNIST(
    root=str(ROOT / "data"),
    train=True,
    download=True,
    transform=transform
)

test_full = datasets.MNIST(
    root=str(ROOT / "data"),
    train=False,
    download=True,
    transform=transform
)

assert TRAIN_LIMIT > 0
assert VAL_LIMIT > 0
assert TEST_LIMIT > 0

assert TRAIN_LIMIT + VAL_LIMIT <= len(train_full)
assert TEST_LIMIT <= len(test_full)

# 固定数据划分
split_generator = torch.Generator().manual_seed(
    SEED
)

train_indices = torch.randperm(
    len(train_full),
    generator=split_generator
).tolist()

train_set = Subset(
    train_full,
    train_indices[:TRAIN_LIMIT]
)

val_set = Subset(
    train_full,
    train_indices[
        TRAIN_LIMIT:TRAIN_LIMIT + VAL_LIMIT
    ]
)

# 官方测试集独立于训练、验证数据
test_set = Subset(
    test_full,
    list(range(TEST_LIMIT))
)

# 单独设置DataLoader随机数生成器
loader_generator = torch.Generator().manual_seed(
    SEED + 1
)

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    generator=loader_generator
)

val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print()
print("[1/6] MNIST loaded")

print("Training:", len(train_set))
print("Validation:", len(val_set))
print("Testing:", len(test_set))


# ============================================================
# 4. 图像编码为光场
# ============================================================

def image_to_field(images):

    """
    MNIST图像转换为输入光场。

    输入：
        [B, 1, 28, 28]

    输出：
        [B, 128, 128]
    """

    # 图像放大到80×80
    images = F.interpolate(
        images,
        size=(80, 80),
        mode="bilinear",
        align_corners=False
    )

    # 居中填充到128×128
    images = F.pad(
        images,
        (24, 24, 24, 24)
    )

    amplitude = images[:, 0]

    # 输入光功率归一化
    power = amplitude.square().sum(
        dim=(-2, -1),
        keepdim=True
    )

    amplitude = amplitude / torch.sqrt(
        power.clamp_min(EPS)
    )

    return amplitude.to(torch.complex64)


print("[2/6] Input optical encoding ready")


# ============================================================
# 5. 定义十个光学探测器
# ============================================================

yy, xx = torch.meshgrid(
    torch.arange(N),
    torch.arange(N),
    indexing="ij"
)

detector_positions = []

# 128×128网格下的探测器位置
for cy in [34, 94]:

    for cx in [20, 42, 64, 86, 108]:

        detector_positions.append(
            (cx, cy)
        )

# 探测器半径
DETECTOR_RADIUS = 6

detector_masks = []

for cx, cy in detector_positions:

    mask = (
        (xx - cx)**2
        + (yy - cy)**2
        <= DETECTOR_RADIUS**2
    )

    detector_masks.append(
        mask.float()
    )

detector_masks = torch.stack(
    detector_masks,
    dim=0
).to(DEVICE)

print("[3/6] Optical detectors ready")


# ============================================================
# 6. 构建角谱传播算子
# ============================================================

M = N * PAD_FACTOR

PAD = (M - N) // 2

freq = torch.fft.fftfreq(
    M,
    d=DX,
    dtype=torch.float64
)

FY, FX = torch.meshgrid(
    freq,
    freq,
    indexing="ij"
)

k = 2 * math.pi / WAVELENGTH

argument = (
    1.0
    - (WAVELENGTH * FX)**2
    - (WAVELENGTH * FY)**2
)

# 当前采样条件下不包含倏逝波频率
kz_relative = torch.sqrt(
    torch.clamp(argument, min=0.0)
) - 1.0

# 去除不影响光强的整体传播相位
H = torch.exp(
    1j * k * Z * kz_relative
).to(torch.complex64)


def angular_spectrum(U, H):

    """
    带零填充的角谱传播。
    """

    U_pad = F.pad(
        U,
        (PAD, PAD, PAD, PAD)
    )

    U_fft = torch.fft.fft2(U_pad)

    Uz_pad = torch.fft.ifft2(
        U_fft * H
    )

    # 裁剪回有效窗口
    Uz = Uz_pad[
        ...,
        PAD:PAD + N,
        PAD:PAD + N
    ]

    return Uz


print("[4/6] Angular spectrum ready")


# ============================================================
# 7. 五层D2NN模型
# ============================================================

class D2NN(nn.Module):

    def __init__(self):

        super().__init__()

        # 五个独立的可训练相位层
        initial_phases = (
            2 * math.pi
            * torch.rand(
                NUM_LAYERS,
                N,
                N
            )
        )

        self.phases = nn.Parameter(
            initial_phases
        )

        self.register_buffer(
            "H",
            H
        )

        self.register_buffer(
            "detectors",
            detector_masks
        )

    def forward(self, U):

        for phase in self.phases:

            # 相位调制
            phase_mask = torch.exp(
                1j * phase
            )

            U = U * phase_mask

            # 传播至下一平面
            U = angular_spectrum(
                U,
                self.H
            )

        # 输出光强
        intensity = torch.abs(U)**2

        return intensity

    def detect(self, intensity):

        # 对十个探测器进行光强积分
        powers = torch.einsum(
            "bxy,cxy->bc",
            intensity,
            self.detectors
        )

        return powers


model = D2NN().to(DEVICE)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)

print("[5/6] D2NN model ready")

print(
    "Trainable parameters:",
    parameter_count
)


# ============================================================
# 8. 损失函数
# ============================================================

def calculate_loss(powers, labels):

    """
    基础损失：
        分类交叉熵
        + 正确探测器效率损失

    Targeted模式额外包含：
        数字9加权分类
        + 数字9与4、7的判别间隔损失
        + 数字4、7相对于9的反向约束
    """

    logits = torch.log(
        powers.clamp_min(EPS)
    )

    # --------------------------------------------------------
    # 8.1 分类损失
    # --------------------------------------------------------

    weights = None

    if TRAIN_MODE == "targeted":

        weights = torch.ones(
            NUM_CLASSES,
            device=powers.device
        )

        weights[9] = CLASS9_WEIGHT

    classification_loss = F.cross_entropy(
        logits,
        labels,
        weight=weights
    )

    # --------------------------------------------------------
    # 8.2 正确探测效率损失
    # --------------------------------------------------------

    correct_power = powers.gather(
        1,
        labels.unsqueeze(1)
    ).squeeze(1)

    efficiency_loss = -torch.log(
        correct_power.clamp_min(EPS)
    ).mean()

    loss = (
        classification_loss
        + EFFICIENCY_WEIGHT * efficiency_loss
    )

    # --------------------------------------------------------
    # 8.3 数字9专项优化
    # --------------------------------------------------------

    if TRAIN_MODE == "targeted":

        margin_sum = logits.sum() * 0.0

        # ===== 数字9与4、7的区分 =====

        mask_9 = labels == 9

        if mask_9.any():

            logits_9 = logits[mask_9]

            score_9 = logits_9[:, 9]

            rival_score = torch.maximum(
                logits_9[:, 4],
                logits_9[:, 7]
            )

            loss_9 = F.relu(
                MARGIN
                + rival_score
                - score_9
            )

            margin_sum = (
                margin_sum
                + loss_9.sum()
            )

        # ===== 避免4、7被过度预测成9 =====

        mask_47 = (
            (labels == 4)
            | (labels == 7)
        )

        if mask_47.any():

            logits_47 = logits[mask_47]

            labels_47 = labels[mask_47]

            true_score = logits_47.gather(
                1,
                labels_47.unsqueeze(1)
            ).squeeze(1)

            score_9 = logits_47[:, 9]

            loss_47 = F.relu(
                MARGIN
                + score_9
                - true_score
            )

            margin_sum = (
                margin_sum
                + loss_47.sum()
            )

        # 按整个batch平均
        margin_loss = (
            margin_sum / labels.numel()
        )

        loss = (
            loss
            + MARGIN_WEIGHT * margin_loss
        )

    return loss


# ============================================================
# 9. 模型评价函数
# ============================================================

@torch.no_grad()
def evaluate(loader, collect_predictions=False):

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # 光学指标
    total_correct_power = 0.0
    total_window_power = 0.0

    confusion = torch.zeros(
        NUM_CLASSES,
        NUM_CLASSES,
        dtype=torch.int64
    )

    all_labels = []
    all_predictions = []

    for images, labels in loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        U = image_to_field(images)

        intensity = model(U)

        powers = model.detect(intensity)

        loss = calculate_loss(
            powers,
            labels
        )

        predictions = powers.argmax(dim=1)

        batch_size = labels.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

        # 正确探测器光功率
        correct_power = powers.gather(
            1,
            labels.unsqueeze(1)
        ).squeeze(1)

        total_correct_power += (
            correct_power.sum().item()
        )

        # 输出窗口光功率
        window_power = intensity.sum(
            dim=(-2, -1)
        )

        total_window_power += (
            window_power.sum().item()
        )

        # 混淆矩阵
        indices = (
            labels * NUM_CLASSES
            + predictions
        ).cpu()

        confusion += torch.bincount(
            indices,
            minlength=NUM_CLASSES**2
        ).reshape(
            NUM_CLASSES,
            NUM_CLASSES
        )

        if collect_predictions:

            all_labels.append(
                labels.cpu()
            )

            all_predictions.append(
                predictions.cpu()
            )

    result = {
        "loss": total_loss / total_samples,

        "accuracy": (
            total_correct / total_samples
        ),

        # 输入光场已经归一化为单位光功率
        "correct_efficiency": (
            total_correct_power / total_samples
        ),

        "window_efficiency": (
            total_window_power / total_samples
        ),

        "confusion": confusion
    }

    if collect_predictions:

        result["labels"] = torch.cat(
            all_labels
        )

        result["predictions"] = torch.cat(
            all_predictions
        )

    return result


# ============================================================
# 10. 正式训练
# ============================================================

history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}

best_val_acc = -1.0
best_val_loss = float("inf")

best_epoch = 0
best_state = None

print()
print("=" * 65)
print("Training Started")
print("=" * 65)

for epoch in range(1, EPOCHS + 1):

    model.train()

    running_loss = 0.0
    running_correct = 0
    running_samples = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        U = image_to_field(images)

        optimizer.zero_grad()

        # 前向传播
        intensity = model(U)

        powers = model.detect(intensity)

        # 损失函数
        loss = calculate_loss(
            powers,
            labels
        )

        # 反向传播
        loss.backward()

        # 更新五层相位
        optimizer.step()

        predictions = powers.argmax(dim=1)

        batch_size = labels.size(0)

        running_loss += (
            loss.item() * batch_size
        )

        running_correct += (
            predictions == labels
        ).sum().item()

        running_samples += batch_size

    # --------------------------------------------------------
    # 训练指标
    # --------------------------------------------------------

    train_loss = (
        running_loss / running_samples
    )

    train_acc = (
        running_correct / running_samples
    )

    # --------------------------------------------------------
    # 验证集
    # --------------------------------------------------------

    val_result = evaluate(
        val_loader
    )

    val_loss = val_result["loss"]
    val_acc = val_result["accuracy"]

    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_acc)

    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_acc)

    # --------------------------------------------------------
    # 保存最佳模型
    # --------------------------------------------------------

    better = (
        val_acc > best_val_acc + 1e-12
        or (
            abs(val_acc - best_val_acc) <= 1e-12
            and val_loss < best_val_loss
        )
    )

    if better:

        best_val_acc = val_acc
        best_val_loss = val_loss

        best_epoch = epoch

        best_state = copy.deepcopy(
            model.state_dict()
        )

    print(
        f"Epoch [{epoch:02d}/{EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc*100:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc*100:.2f}%",
        flush=True
    )


print("=" * 65)
print("Training Completed")
print("=" * 65)


# ============================================================
# 11. 加载最佳模型
# ============================================================

model.load_state_dict(
    best_state
)

model.eval()

# 最终测试只进行一次
test_result = evaluate(
    test_loader,
    collect_predictions=True
)

test_acc = test_result["accuracy"]

cm = test_result["confusion"].numpy()

labels_all = test_result["labels"]
predictions_all = test_result["predictions"]


# ============================================================
# 12. 计算各类别召回率
# ============================================================

class_recall = {}

for digit in range(NUM_CLASSES):

    total = cm[digit].sum()

    correct = cm[digit, digit]

    recall = (
        correct / total
        if total > 0
        else 0.0
    )

    class_recall[digit] = float(recall)


# ============================================================
# 13. 数字9误分类分析
# ============================================================

digit9_total = int(
    cm[9].sum()
)

digit9_correct = int(
    cm[9, 9]
)

digit9_to_4 = int(
    cm[9, 4]
)

digit9_to_7 = int(
    cm[9, 7]
)

digit4_to_9 = int(
    cm[4, 9]
)

digit7_to_9 = int(
    cm[7, 9]
)

print()
print("=" * 65)
print("Final Test Results")
print("=" * 65)

print(
    f"Best Epoch: {best_epoch}"
)

print(
    f"Best Validation Accuracy: "
    f"{best_val_acc*100:.2f}%"
)

print(
    f"Test Accuracy: "
    f"{test_acc*100:.2f}%"
)

print()
print("Per-class recall:")

for digit, recall in class_recall.items():

    print(
        f"Digit {digit}: "
        f"{recall*100:.2f}%"
    )

print()
print("Digit 9 Analysis")

print(
    "Total digit 9 samples:",
    digit9_total
)

print(
    "Correct digit 9 predictions:",
    digit9_correct
)

print(
    "Digit 9 recall:",
    f"{class_recall[9]*100:.2f}%"
)

print(
    "9 -> 4 errors:",
    digit9_to_4
)

print(
    "9 -> 7 errors:",
    digit9_to_7
)

print(
    "4 -> 9 errors:",
    digit4_to_9
)

print(
    "7 -> 9 errors:",
    digit7_to_9
)

print()
print("Optical performance:")

print(
    "Mean correct detector efficiency:",
    f"{test_result['correct_efficiency']*100:.2f}%"
)

print(
    "Mean output window efficiency:",
    f"{test_result['window_efficiency']*100:.2f}%"
)


# ============================================================
# 14. 保存训练记录CSV
# ============================================================

csv_path = RESULT_DIR / "training_history.csv"

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "epoch",
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy"
    ])

    for i in range(EPOCHS):

        writer.writerow([
            i + 1,
            history["train_loss"][i],
            history["train_accuracy"][i],
            history["val_loss"][i],
            history["val_accuracy"][i]
        ])


# ============================================================
# 15. 绘制训练曲线
# ============================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 4)
)

epoch_axis = range(
    1,
    EPOCHS + 1
)

axes[0].plot(
    epoch_axis,
    history["train_loss"],
    label="Training Loss"
)

axes[0].plot(
    epoch_axis,
    history["val_loss"],
    label="Validation Loss"
)

axes[0].set_title(
    "Training Loss"
)

axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")

axes[0].legend()
axes[0].grid(True, alpha=0.3)


axes[1].plot(
    epoch_axis,
    history["train_accuracy"],
    label="Training Accuracy"
)

axes[1].plot(
    epoch_axis,
    history["val_accuracy"],
    label="Validation Accuracy"
)

axes[1].set_title(
    "Classification Accuracy"
)

axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")

axes[1].set_ylim(0, 1)

axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "training_curves.png",
    dpi=300
)

plt.close(fig)


# ============================================================
# 16. 绘制混淆矩阵
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 8)
)

im = ax.imshow(
    cm,
    cmap="Blues"
)

ax.set_title(
    f"MNIST Confusion Matrix | "
    f"Accuracy {test_acc*100:.2f}%"
)

ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")

ax.set_xticks(range(10))
ax.set_yticks(range(10))

threshold = cm.max() / 2

for i in range(10):

    for j in range(10):

        ax.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=8,
            color=(
                "white"
                if cm[i, j] > threshold
                else "black"
            )
        )

fig.colorbar(
    im,
    ax=ax
)

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close(fig)


# ============================================================
# 17. 绘制各类别召回率
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 5)
)

recall_values = [
    class_recall[i] * 100
    for i in range(10)
]

bars = ax.bar(
    range(10),
    recall_values
)

ax.set_title(
    "MNIST Per-Class Recall"
)

ax.set_xlabel("Digit")
ax.set_ylabel("Recall (%)")

ax.set_xticks(range(10))
ax.set_ylim(0, 105)

for bar, value in zip(
    bars,
    recall_values
):

    ax.text(
        bar.get_x() + bar.get_width()/2,
        value + 1,
        f"{value:.1f}%",
        ha="center",
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "per_class_recall.png",
    dpi=300
)

plt.close(fig)


# ============================================================
# 18. 绘制五层相位分布
# ============================================================

fig, axes = plt.subplots(
    1,
    NUM_LAYERS,
    figsize=(18, 4)
)

for i in range(NUM_LAYERS):

    phase = model.phases[i].detach()

    wrapped_phase = torch.angle(
        torch.exp(1j * phase)
    ).cpu().numpy()

    im = axes[i].imshow(
        wrapped_phase,
        cmap="twilight",
        vmin=-math.pi,
        vmax=math.pi
    )

    axes[i].set_title(
        f"Diffractive Layer {i+1}"
    )

    axes[i].axis("off")

fig.colorbar(
    im,
    ax=axes.tolist(),
    shrink=0.7,
    label="Phase (rad)"
)

plt.savefig(
    RESULT_DIR / "phase_layers.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# 19. 自动选择有代表性的测试样本
# ============================================================

def find_example(true_digit, pred_digit, used):

    """
    在测试集中查找指定的真实类别和预测类别。
    """

    mask = (
        (labels_all == true_digit)
        & (predictions_all == pred_digit)
    )

    indices = torch.where(mask)[0].tolist()

    for idx in indices:

        if idx not in used:

            return idx

    return None


selected_indices = []
used_indices = set()

# 优先展示数字9相关的分类情况
priority_cases = [
    (9, 4),
    (9, 7),
    (9, 9),
    (4, 9),
    (7, 9),
    (8, 8)
]

for true_digit, pred_digit in priority_cases:

    idx = find_example(
        true_digit,
        pred_digit,
        used_indices
    )

    if idx is not None:

        selected_indices.append(idx)
        used_indices.add(idx)

# 如果某种错误不存在，使用其他样本补足
if len(selected_indices) < 6:

    for idx in range(len(test_set)):

        if idx not in used_indices:

            selected_indices.append(idx)
            used_indices.add(idx)

        if len(selected_indices) >= 6:
            break


# ============================================================
# 20. 绘制分类示例
# ============================================================

example_images = []

for idx in selected_indices:

    image, _ = test_set[idx]

    example_images.append(image)

example_images = torch.stack(
    example_images
).to(DEVICE)

with torch.no_grad():

    U = image_to_field(
        example_images
    )

    example_intensity = model(U)

    example_powers = model.detect(
        example_intensity
    )

rows = len(selected_indices)

fig, axes = plt.subplots(
    rows,
    2,
    figsize=(9, rows * 3)
)

# imshow坐标边界与采样点中心一致
extent = [
    (-N/2 - 0.5) * DX * 1e3,
    (N/2 - 0.5) * DX * 1e3,
    (-N/2 - 0.5) * DX * 1e3,
    (N/2 - 0.5) * DX * 1e3
]

for row, idx in enumerate(selected_indices):

    true_digit = labels_all[idx].item()

    pred_digit = predictions_all[idx].item()

    # --------------------------------------------------------
    # 输入图像
    # --------------------------------------------------------

    axes[row, 0].imshow(
        example_images[row, 0].cpu().numpy(),
        cmap="gray"
    )

    axes[row, 0].set_title(
        f"Input Digit: {true_digit}"
    )

    axes[row, 0].axis("off")

    # --------------------------------------------------------
    # 输出光强
    # --------------------------------------------------------

    intensity_np = (
        example_intensity[row]
        .cpu()
        .numpy()
    )

    axes[row, 1].imshow(
        intensity_np,
        cmap="inferno",
        extent=extent,
        origin="lower"
    )

    axes[row, 1].set_title(
        f"True: {true_digit} | "
        f"Predicted: {pred_digit}"
    )

    # 标注十个探测器
    for digit, (cx, cy) in enumerate(
        detector_positions
    ):

        px = (
            cx - N // 2
        ) * DX * 1e3

        py = (
            cy - N // 2
        ) * DX * 1e3

        circle = Circle(
            (px, py),
            DETECTOR_RADIUS * DX * 1e3,
            fill=False,
            edgecolor="white",
            linewidth=0.8
        )

        axes[row, 1].add_patch(circle)

        axes[row, 1].text(
            px,
            py,
            str(digit),
            color="white",
            fontsize=7,
            ha="center",
            va="center"
        )

    axes[row, 1].set_xlabel("x (mm)")
    axes[row, 1].set_ylabel("y (mm)")

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "classification_examples.png",
    dpi=300
)

plt.close(fig)


# ============================================================
# 21. 保存测试指标
# ============================================================

metrics = {
    "train_mode": TRAIN_MODE,

    "best_epoch": best_epoch,

    "best_validation_accuracy": best_val_acc,

    "test_accuracy": test_acc,

    "test_loss": test_result["loss"],

    "correct_detector_efficiency": (
        test_result["correct_efficiency"]
    ),

    "window_efficiency": (
        test_result["window_efficiency"]
    ),

    "class_recall": class_recall,

    "digit9_total": digit9_total,

    "digit9_correct": digit9_correct,

    "digit9_to_4": digit9_to_4,

    "digit9_to_7": digit9_to_7,

    "digit4_to_9": digit4_to_9,

    "digit7_to_9": digit7_to_9,

    "confusion_matrix": cm.tolist()
}

with open(
    RESULT_DIR / "metrics.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        metrics,
        f,
        indent=4,
        ensure_ascii=False
    )


# ============================================================
# 22. 保存最佳模型
# ============================================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "config": config,
        "metrics": metrics,
        "detector_positions": detector_positions
    },
    RESULT_DIR / "mnist_d2nn_best.pt"
)


# ============================================================
# 23. 输出全部文件
# ============================================================

print()
print("=" * 65)
print("Output Files")
print("=" * 65)

for file in sorted(RESULT_DIR.iterdir()):

    print(file.name)

print()
print("Results directory:")
print(RESULT_DIR)

print()
print("[6/6] Simulation completed successfully!")