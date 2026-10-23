"""Step 7.1 - controlled ablation of MNIST encoding and detector geometry.

Variants:
baseline: old 2x5/6 mm/grayscale Step 6.3 configuration
geometry: 3-4-3/4.8 mm/grayscale
aligned: 3-4-3/4.8 mm/binarized input at adjustable 0.5 threshold

Detector numbering and width are supported by Lin et al. (Science 2018,
Fig. 3A; Supplementary Fig. S1); exact center spacing, input pixel mapping,
binarization threshold, input plane distance and loss coefficients remain
provisional research choices. Retains old objective for controlled comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# ============================= 1. CONFIG =============================
PARSER = argparse.ArgumentParser(description="Step 7.1 geometry / input ablations")
PARSER.add_argument(
    "--variant", choices=["baseline", "geometry", "aligned"],
    default="geometry",
    help="baseline: old; geometry: new detectors; aligned: new detectors + binarization"
)
ARGS = PARSER.parse_args()
VARIANT = ARGS.variant

SEED = 42
torch.manual_seed(SEED)

MODE = "smoke"        # "smoke" first; "full" only after validation
OVERSAMPLE = 1         # 1 = fast 400 um grid, 2 = finer 200 um grid
PAD_FACTOR = 2
DEVICE = torch.device("cpu")  # Mac MPS complex FFT support is version-dependent

C = 299_792_458.0
FREQUENCY = 0.4e12
WAVELENGTH = C / FREQUENCY
N_LAYERS = 5
N_NEURONS = 200
NEURON_SIZE = 400e-6
LAYER_DISTANCE = 30e-3
OUTPUT_DISTANCE = 30e-3

# IMPORTANT: only detector numbering and detector width are source-grounded.
# Positions below are a reasonable 3-4-3 approximation, NOT measured paper centers.
INPUT_PIXEL_SIZE = NEURON_SIZE
INPUT_TO_FIRST_LAYER = 0.0  # still provisional
BINARIZE_THRESHOLD = 0.5    # still provisional: paper says binarized, not threshold

if VARIANT == "baseline":
    DETECTOR_SIDE = 6e-3
    DETECTOR_CENTERS = [
        (cx, cy)
        for cy in (-10e-3, 10e-3)
        for cx in (-24e-3, -12e-3, 0.0, 12e-3, 24e-3)
    ]
else:
    DETECTOR_SIDE = 4.8e-3  # Supplementary Fig. S1
    DETECTOR_CENTERS = [
        (cx_mm * 1e-3, cy_mm * 1e-3)
        for cy_mm, xs_mm in [
            (12.0, (-12.0, 0.0, 12.0)),           # classes 0,1,2
            (0.0, (-18.0, -6.0, 6.0, 18.0)),     # classes 3,4,5,6
            (-12.0, (-12.0, 0.0, 12.0)),         # classes 7,8,9
        ]
        for cx_mm in xs_mm
    ]
BINARIZE_INPUT = VARIANT == "aligned"
EFFICIENCY_WEIGHT = 0.5         # retain Step 6.3 loss weights for fair ablation
OUTSIDE_WEIGHT = 0.05
LR = 5e-3
EPS = 1e-12

if MODE == "smoke":
    TRAIN_PER_CLASS, VAL_PER_CLASS, TEST_PER_CLASS = 20, 10, 10
    EPOCHS, BATCH_SIZE = 2, 2
elif MODE == "full":
    EPOCHS, BATCH_SIZE = 10, 8
else:
    raise ValueError("MODE must be 'smoke' or 'full'")

assert OVERSAMPLE >= 1 and PAD_FACTOR >= 1
SAMPLES = N_NEURONS * OVERSAMPLE
DX = NEURON_SIZE / OVERSAMPLE
FFT_N = SAMPLES * PAD_FACTOR
PAD_LEFT = (FFT_N - SAMPLES) // 2
PAD_RIGHT = FFT_N - SAMPLES - PAD_LEFT
ACTIVE_SLICE = slice(PAD_LEFT, PAD_LEFT + SAMPLES)

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "data"  # ../data/MNIST/raw/  (Step 5 data)
RUN_NAME = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
OUT_DIR = ROOT / "results" / "step7_1" / f"{VARIANT}_{MODE}_os{OVERSAMPLE}_{RUN_NAME}"
OUT_DIR.mkdir(parents=True, exist_ok=False)

print("=" * 67)
print("Step 7.1 - MNIST input/detector ablation")
print(f"Variant: {VARIANT}; binary input: {BINARIZE_INPUT}; detector width: {DETECTOR_SIDE*1e3:.1f} mm")
print("=" * 67)
print(f"Mode: {MODE}; device: {DEVICE}; layers: {N_LAYERS}")
print(f"Frequency: {FREQUENCY/1e12:.3f} THz; wavelength: {WAVELENGTH*1e6:.3f} um")
print(f"Physical pixels/layer: {N_NEURONS} x {N_NEURONS}; parameters: {N_LAYERS*N_NEURONS**2:,}")
print(f"Oversample: {OVERSAMPLE}; active grid: {SAMPLES} x {SAMPLES}; FFT: {FFT_N} x {FFT_N}")
print(f"Dataset root: {DATA_ROOT}")
print(f"Results: {OUT_DIR}", flush=True)


# ========================== 2. DATA LOADING ==========================
transform = transforms.ToTensor()  # previously missing: must be defined before MNIST
try:
    train_full = datasets.MNIST(root=str(DATA_ROOT), train=True, download=False, transform=transform)
    test_full = datasets.MNIST(root=str(DATA_ROOT), train=False, download=False, transform=transform)
except RuntimeError as exc:
    raise RuntimeError(
        f"Cannot find local MNIST in {DATA_ROOT / 'MNIST' / 'raw'}. "
        "Ensure Step 5's four uncompressed IDX files exist."
    ) from exc


def choose_balanced(dataset: datasets.MNIST, count_per_class: int, seed: int) -> list[int]:
    """Deterministically choose count_per_class disjoint indexes for each digit."""
    labels = torch.as_tensor(dataset.targets)
    generator = torch.Generator().manual_seed(seed)
    output: list[int] = []
    for digit in range(10):
        choices = torch.nonzero(labels == digit, as_tuple=False).flatten()
        if len(choices) < count_per_class:
            raise ValueError(f"Digit {digit} has only {len(choices)} samples")
        choices = choices[torch.randperm(len(choices), generator=generator)[:count_per_class]]
        output.extend(choices.tolist())
    return output


if MODE == "smoke":
    selected = choose_balanced(train_full, TRAIN_PER_CLASS + VAL_PER_CLASS, SEED)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for digit in range(10):
        group = selected[digit * (TRAIN_PER_CLASS + VAL_PER_CLASS):
                         (digit + 1) * (TRAIN_PER_CLASS + VAL_PER_CLASS)]
        train_idx.extend(group[:TRAIN_PER_CLASS])
        val_idx.extend(group[TRAIN_PER_CLASS:])
    test_idx = choose_balanced(test_full, TEST_PER_CLASS, SEED + 1)
else:
    shuffled = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(SEED)).tolist()
    train_idx = shuffled[:55_000]
    val_idx = shuffled[55_000:60_000]
    test_idx = list(range(len(test_full)))

train_set = Subset(train_full, train_idx)
val_set = Subset(train_full, val_idx)
test_set = Subset(test_full, test_idx)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
                          generator=torch.Generator().manual_seed(SEED + 3))
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"[1/7] MNIST loaded: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}", flush=True)


# ====================== 3. INPUT AMPLITUDE ENCODING ===================
def encode_mnist(images: torch.Tensor) -> torch.Tensor:
    """[B,1,28,28] -> complex [B,FFT_N,FFT_N], normalized to unit discrete |U|^2 sum.

    This prototype maps a 28x28 image to a centered 11.2 mm square on the
    first diffractive layer's incident plane. Exact paper geometry is TBD.
    """
    # Paper uses amplitude-encoded, upsampled, binarized digit inputs.
    # Threshold is a documented experimental choice; baseline remains grayscale.
    amps = images[:, 0]
    if BINARIZE_INPUT:
        amps = (amps >= BINARIZE_THRESHOLD).to(amps.dtype)
    amps = amps.repeat_interleave(OVERSAMPLE, -2).repeat_interleave(OVERSAMPLE, -1)
    digit_n = 28 * OVERSAMPLE
    top = (SAMPLES - digit_n) // 2
    bottom = SAMPLES - digit_n - top
    amps = F.pad(amps, (top, bottom, top, bottom))
    amps = F.pad(amps, (PAD_LEFT, PAD_RIGHT, PAD_LEFT, PAD_RIGHT))
    energy = amps.square().sum(dim=(-2, -1), keepdim=True)
    return (amps / torch.sqrt(energy.clamp_min(EPS))).to(torch.complex64)


# Show a sample's original greyscale and encoded amplitude as an audit.
sample_img, sample_label = test_set[0]
with torch.no_grad():
    sample_u = encode_mnist(sample_img.unsqueeze(0))
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].imshow(sample_img[0].numpy(), cmap="gray", vmin=0, vmax=1)
axes[0].set_title(f"Original MNIST (digit {sample_label})")
axes[1].imshow(sample_u[0, ACTIVE_SLICE, ACTIVE_SLICE].abs().numpy(),
               cmap="gray", interpolation="nearest", extent=(-40, 40, -40, 40))
axes[1].set_title(f"Encoded: {'binary' if BINARIZE_INPUT else 'grayscale'}")
for ax in axes:
    ax.axis("off")
fig.tight_layout()
fig.savefig(OUT_DIR / "input_encoding.png", dpi=200)
plt.close(fig)


# ======================== 4. OPTICAL GEOMETRY ========================
x = (torch.arange(FFT_N, dtype=torch.float64) - FFT_N // 2) * DX
Y, X = torch.meshgrid(x, x, indexing="ij")

aperture = torch.zeros((FFT_N, FFT_N), dtype=torch.float32)
aperture[ACTIVE_SLICE, ACTIVE_SLICE] = 1.0

detectors = []
for cx, cy in DETECTOR_CENTERS:
    if VARIANT == "baseline":
        mask = (((X - cx).abs() <= DETECTOR_SIDE / 2).logical_and(
                (Y - cy).abs() <= DETECTOR_SIDE / 2).to(torch.float32))
    else:
        # Use exactly 12 * OVERSAMPLE grid cells for 4.8 mm,
        # avoiding ambiguous <= endpoints that could yield 13 cells.
        side_n = round(DETECTOR_SIDE / DX)
        assert abs(side_n * DX - DETECTOR_SIDE) < 1e-10
        x_center = round(cx / DX) + FFT_N // 2
        y_center = round(cy / DX) + FFT_N // 2
        x0, y0 = x_center - side_n // 2, y_center - side_n // 2
        assert 0 <= x0 <= FFT_N - side_n and 0 <= y0 <= FFT_N - side_n
        mask = torch.zeros((FFT_N, FFT_N), dtype=torch.float32)
        mask[y0:y0+side_n, x0:x0+side_n] = 1.
        assert mask.sum().item() == side_n ** 2
    detectors.append(mask)
detectors = torch.stack(detectors, dim=0)
assert detectors.shape[0] == 10
assert float(detectors.sum(dim=0).max()) == 1.0, "Detector masks overlap"
assert bool((detectors.sum(dim=(-2, -1)) > 0).all()), "Empty detector"

# Save audit visualization before time-consuming MNIST training.
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(detectors.sum(dim=0)[ACTIVE_SLICE, ACTIVE_SLICE].numpy(),
          cmap="gray_r", origin="lower", extent=(-40, 40, -40, 40), vmin=0, vmax=1)
for digit, (cx, cy) in enumerate(DETECTOR_CENTERS):
    ax.text(cx*1e3, cy*1e3, str(digit),
            ha="center", va="center", fontsize=10, color="white",
            bbox={"facecolor": "black", "alpha": .65, "pad": 1.0})
ax.set(title=f"{VARIANT}: detector layout (side={DETECTOR_SIDE*1e3:.1f} mm)",
       xlabel="x (mm)", ylabel="y (mm)", xlim=(-40, 40), ylim=(-40, 40))
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(OUT_DIR / "detector_geometry.png", dpi=200)
plt.close(fig)

freq = torch.fft.fftfreq(FFT_N, d=DX, dtype=torch.float64)
FY, FX = torch.meshgrid(freq, freq, indexing="ij")
k = 2.0 * math.pi / WAVELENGTH
kz = k * torch.sqrt((1.0 - (WAVELENGTH*FX)**2 - (WAVELENGTH*FY)**2).to(torch.complex128))


def transfer_function(distance: float) -> torch.Tensor:
    # Remove the physically irrelevant common exp(i k z) phase.
    return torch.exp(1j * (kz - k) * distance).to(torch.complex64)

H_LAYER = transfer_function(LAYER_DISTANCE)
H_OUTPUT = transfer_function(OUTPUT_DISTANCE)
print("[2/7] ASM, finite aperture and ten square detectors ready", flush=True)


# ============================= 5. MODEL ==============================
class D2NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_phases = nn.Parameter(0.7 * torch.randn(N_LAYERS, N_NEURONS, N_NEURONS))
        self.register_buffer("H_layer", H_LAYER, persistent=False)
        self.register_buffer("H_output", H_OUTPUT, persistent=False)
        self.register_buffer("aperture", aperture, persistent=False)
        self.register_buffer("detectors", detectors, persistent=False)

    def phase(self, layer: int) -> torch.Tensor:
        return math.pi * torch.sigmoid(self.raw_phases[layer])

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        for layer in range(N_LAYERS):
            ph = self.phase(layer)
            ph = ph.repeat_interleave(OVERSAMPLE, -2).repeat_interleave(OVERSAMPLE, -1)
            ph = F.pad(ph, (PAD_LEFT, PAD_RIGHT, PAD_LEFT, PAD_RIGHT))
            u = u * self.aperture * torch.exp(1j * ph)
            transfer = self.H_output if layer == N_LAYERS - 1 else self.H_layer
            u = torch.fft.ifft2(torch.fft.fft2(u, dim=(-2, -1)) * transfer, dim=(-2, -1))
        return u.abs().square()

    def detect(self, intensity: torch.Tensor) -> torch.Tensor:
        return intensity.flatten(1) @ self.detectors.flatten(1).T


model = D2NN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
print(f"[3/7] Trainable phase parameters: {model.raw_phases.numel():,}", flush=True)


# ============================= 6. LOSS ===============================
def objective(powers: torch.Tensor, intensity: torch.Tensor,
              labels: torch.Tensor, input_powers: torch.Tensor):
    """Provisional differentiable objective; NOT the paper's exact released code."""
    chosen = powers.gather(1, labels[:, None]).squeeze(1)
    ce = F.cross_entropy(torch.log(powers.clamp_min(EPS)), labels)
    correct_efficiency = chosen / input_powers.clamp_min(EPS)
    all_output = intensity.sum(dim=(-2, -1)).clamp_min(EPS)
    outside_fraction = (1 - powers.sum(dim=1) / all_output).clamp(0, 1)
    loss = (ce + EFFICIENCY_WEIGHT * (1 - correct_efficiency).mean()
            + OUTSIDE_WEIGHT * outside_fraction.mean())
    return loss, correct_efficiency.detach(), (all_output / input_powers.clamp_min(EPS)).detach()


@torch.no_grad()
def evaluate(loader: DataLoader) -> dict:
    model.eval()
    count = 0
    num_correct = 0
    loss_sum = correct_eff_sum = window_eff_sum = 0.0
    cm = torch.zeros((10, 10), dtype=torch.int64)
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        u = encode_mnist(images)
        input_p = u.abs().square().sum(dim=(-2, -1))
        intensity = model(u)
        power = model.detect(intensity)
        loss, correct_eff, window_eff = objective(power, intensity, labels, input_p)
        pred = power.argmax(dim=1)
        batch = labels.numel()
        count += batch
        num_correct += int((pred == labels).sum().item())
        loss_sum += loss.item() * batch
        correct_eff_sum += correct_eff.sum().item()
        window_eff_sum += window_eff.sum().item()
        cm += torch.bincount((labels * 10 + pred).cpu(), minlength=100).reshape(10, 10)
    return {"loss": loss_sum / count,
            "accuracy": num_correct / count,
            "correct_efficiency": correct_eff_sum / count,
            "window_efficiency": window_eff_sum / count,
            "confusion": cm}


# ============================ 7. TRAINING ============================
history: list[dict] = []
best_val_accuracy = -1.0
best_epoch = 0
best_path = OUT_DIR / "best_model.pt"
print(f"[4/7] Training: {EPOCHS} epochs, batch size {BATCH_SIZE}", flush=True)

for epoch in range(1, EPOCHS + 1):
    model.train()
    count = num_correct = 0
    loss_sum = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        u = encode_mnist(images)
        input_p = u.abs().square().sum(dim=(-2, -1))
        optimizer.zero_grad(set_to_none=True)
        intensity = model(u)
        powers = model.detect(intensity)
        loss, _, _ = objective(powers, intensity, labels, input_p)
        if not torch.isfinite(loss).item():
            raise RuntimeError(f"Nonfinite training loss at epoch {epoch}")
        loss.backward()
        optimizer.step()
        preds = powers.argmax(dim=1)
        batch = labels.numel()
        count += batch
        num_correct += int((preds == labels).sum().item())
        loss_sum += float(loss.item()) * batch

    train_loss, train_acc = loss_sum / count, num_correct / count
    val = evaluate(val_loader)
    history.append({"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_acc,
                    "val_loss": val["loss"], "val_accuracy": val["accuracy"]})
    print(f"Epoch {epoch:02d}/{EPOCHS} | loss {train_loss:.4f} | "
          f"train {train_acc:.2%} | val {val['accuracy']:.2%}", flush=True)
    if val["accuracy"] > best_val_accuracy:
        best_val_accuracy = val["accuracy"]
        best_epoch = epoch
        torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                    "validation_accuracy": best_val_accuracy}, best_path)

print("[5/7] Training complete; evaluating saved best checkpoint", flush=True)
checkpoint = torch.load(best_path, map_location=DEVICE, weights_only=True)
model.load_state_dict(checkpoint["model_state_dict"])
test = evaluate(test_loader)
cm_np = test["confusion"].cpu().numpy()


# ============================ 8. FIGURES =============================
with (OUT_DIR / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
    writer.writeheader()
    writer.writerows(history)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
epochs = [h["epoch"] for h in history]
for field, axis, ylabel in (("loss", axes[0], "Loss"), ("accuracy", axes[1], "Accuracy")):
    axis.plot(epochs, [h["train_" + field] for h in history], marker="o", label="Train")
    axis.plot(epochs, [h["val_" + field] for h in history], marker="o", label="Validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.set_xticks(epochs)
    axis.legend()
    axis.grid(alpha=0.25)
    if field == "accuracy":
        axis.set_ylim(0, 1)
plt.tight_layout()
fig.savefig(OUT_DIR / "training_curves.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm_np, cmap="Blues")
ax.set(title=f"MNIST test confusion matrix ({test['accuracy']:.2%})",
       xlabel="Predicted digit", ylabel="True digit",
       xticks=list(range(10)), yticks=list(range(10)))
cutoff = cm_np.max() / 2
for row in range(10):
    for col in range(10):
        ax.text(col, row, str(cm_np[row, col]), ha="center", va="center",
                fontsize=8, color="white" if cm_np[row, col] > cutoff else "black")
fig.colorbar(im, ax=ax)
plt.tight_layout()
fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=200)
plt.close(fig)

fig, axes = plt.subplots(1, N_LAYERS, figsize=(16, 4))
for layer, ax in enumerate(axes):
    ph = model.phase(layer).detach().cpu().numpy()
    im = ax.imshow(ph, cmap="twilight", origin="lower", vmin=0, vmax=math.pi,
                   extent=[-40, 40, -40, 40])
    ax.set_title(f"Layer {layer + 1}")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
fig.colorbar(im, ax=list(axes), shrink=0.7, label="Phase (rad)")
fig.savefig(OUT_DIR / "phase_layers.png", dpi=200, bbox_inches="tight")
plt.close(fig)

wanted = [1, 4, 7, 9]
example_images: list[torch.Tensor] = []
example_labels: list[int] = []
for digit in wanted:
    for idx in range(len(test_set)):
        img, label = test_set[idx]
        if int(label) == digit:
            example_images.append(img)
            example_labels.append(digit)
            break
assert len(example_images) == len(wanted), "Failed to find test examples"
example_batch = torch.stack(example_images).to(DEVICE)
with torch.no_grad():
    example_I = model(encode_mnist(example_batch))
    example_pred = model.detect(example_I).argmax(dim=1).tolist()
fig, axes = plt.subplots(len(wanted), 2, figsize=(9, 12))
for i, (digit, predicted) in enumerate(zip(example_labels, example_pred)):
    axes[i, 0].imshow(example_batch[i, 0].cpu(), cmap="gray")
    axes[i, 0].set_title(f"Input {digit}")
    axes[i, 0].axis("off")
    axes[i, 1].imshow(example_I[i, ACTIVE_SLICE, ACTIVE_SLICE].cpu(), cmap="inferno",
                      extent=[-40, 40, -40, 40], origin="lower")
    axes[i, 1].set_title(f"True {digit}, predicted {predicted}")
    for label, (cx, cy) in enumerate(DETECTOR_CENTERS):
        cx_mm, cy_mm = cx * 1e3, cy * 1e3
        side_mm = DETECTOR_SIDE * 1e3
        axes[i, 1].add_patch(Rectangle((cx_mm - side_mm/2, cy_mm - side_mm/2),
                                       side_mm, side_mm, fill=False, edgecolor="white", lw=0.7))
        axes[i, 1].text(cx_mm, cy_mm, str(label), color="white", fontsize=7,
                        ha="center", va="center")
    axes[i, 1].set_xlabel("x (mm)")
    axes[i, 1].set_ylabel("y (mm)")
plt.tight_layout()
fig.savefig(OUT_DIR / "classification_examples.png", dpi=200)
plt.close(fig)

# =========================== 9. FINAL METRICS =========================
recall = {str(d): float(cm_np[d, d] / max(1, cm_np[d].sum())) for d in range(10)}
metrics = {
    "mode": MODE,
    "variant": VARIANT,
    "binarize_input": BINARIZE_INPUT,
    "binarize_threshold": BINARIZE_THRESHOLD if BINARIZE_INPUT else None,
    "detector_arrangement": "2x5" if VARIANT == "baseline" else "3-4-3",
    "seed": SEED,
    "test_accuracy": test["accuracy"],
    "best_epoch": best_epoch,
    "best_validation_accuracy": best_val_accuracy,
    "correct_detector_efficiency": test["correct_efficiency"],
    "output_window_efficiency": test["window_efficiency"],
    "class_recall": recall,
    "confusion_matrix": cm_np.tolist(),
    "physical": {
        "frequency_hz": FREQUENCY, "wavelength_m": WAVELENGTH,
        "layers": N_LAYERS, "neurons_per_layer_side": N_NEURONS,
        "neuron_size_m": NEURON_SIZE, "layer_gap_m": LAYER_DISTANCE,
        "output_gap_m": OUTPUT_DISTANCE,
    },
    "sampling": {"oversample": OVERSAMPLE, "pad_factor": PAD_FACTOR,
                 "dx_m": DX, "fft_grid": FFT_N},
    "training": {"train_size": len(train_set), "val_size": len(val_set),
                 "test_size": len(test_set), "epochs": EPOCHS, "batch_size": BATCH_SIZE,
                 "lr": LR, "efficiency_weight": EFFICIENCY_WEIGHT,
                 "outside_weight": OUTSIDE_WEIGHT},
    "provisional_geometry": {
        "input_pixel_size_m": INPUT_PIXEL_SIZE,
        "input_to_first_layer_m": INPUT_TO_FIRST_LAYER,
        "detector_side_m": DETECTOR_SIDE,
        "detector_centers_m": DETECTOR_CENTERS,
    },
     "note": "Fig 3A 3-4-3 and SM Fig S1 4.8 mm grounded; detector spacing, input pixel size, threshold, input-plane distance and exact loss remain provisional.",
}
with (OUT_DIR / "metrics.json").open("w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print("[6/7] Figures, checkpoint and JSON saved", flush=True)
print("=" * 67)
print(f"Best epoch: {best_epoch}; validation accuracy: {best_val_accuracy:.2%}")
print(f"Test accuracy: {test['accuracy']:.2%}")
print(f"Correct-detector / input-light fraction: {test['correct_efficiency']:.2%}")
print(f"Output-window / input-light fraction: {test['window_efficiency']:.2%}")
print(f"Saved to: {OUT_DIR}")
print("[7/7] Step 7.1 finished", flush=True)
