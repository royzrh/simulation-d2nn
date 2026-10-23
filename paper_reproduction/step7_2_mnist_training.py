#!/usr/bin/env python3
"""Step 7.2 - medium-scale MNIST training for phase-only five-layer THz D2NN.

Reuses the optical and training assumptions of Step 7.1 `aligned`:
- 0.4 THz, 200x200 phase neurons/layer, 5 layers, 30mm propagation gaps,
- 3-4-3 square detector ordering, 4.8mm width,
- binary amplitude MNIST using a provisional 0.5 threshold,
- provisional centers, first-plane distance, input pixel scaling and loss.

Modes: --stage smoke (200/100/100, 2 epochs), pilot (1000/200/1000, 5),
       medium (5000/1000/1000, 10).

`--resume <.../last_checkpoint.pt>` resumes the *same* experiment and restores
optimizer state. The last checkpoint is written after every completed epoch.

No data download: expects ../data/MNIST/raw (relative to this script directory).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
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


# -------------------------- Experiment setup --------------------------
PARSER = argparse.ArgumentParser(description="Step 7.2 five-layer MNIST training")
PARSER.add_argument("--stage", choices=("smoke", "pilot", "medium"), default="pilot")
PARSER.add_argument("--epochs", type=int, default=None, help="override stage epochs")
PARSER.add_argument("--batch-size", type=int, default=None)
PARSER.add_argument("--oversample", type=int, default=1)
PARSER.add_argument("--pad-factor", type=int, default=2)
PARSER.add_argument("--lr", type=float, default=0.005)
PARSER.add_argument("--efficiency-weight", type=float, default=0.5)
PARSER.add_argument("--outside-weight", type=float, default=0.05)
PARSER.add_argument("--resume", type=Path, default=None,
                    help="path to last_checkpoint.pt from a previous run")
PARSER.add_argument("--seed", type=int, default=42)
PARSER.add_argument("--data-root", type=Path, default=None,
                    help="MNIST parent directory; default: ../data")
PARSER.add_argument("--results-root", type=Path, default=None,
                    help="parent directory for generated experiment folders")
ARGS = PARSER.parse_args()

STAGES = {
    "smoke": {"train_each": 20, "val_each": 10, "test_each": 10,
              "epochs": 2, "batch_size": 2},
    "pilot": {"train_each": 100, "val_each": 20, "test_each": 100,
              "epochs": 5, "batch_size": 8},
    "medium": {"train_each": 500, "val_each": 100, "test_each": 100,
               "epochs": 10, "batch_size": 8},
}
STAGE = ARGS.stage
SPLIT = STAGES[STAGE]
EPOCHS = ARGS.epochs if ARGS.epochs is not None else SPLIT["epochs"]
BATCH_SIZE = ARGS.batch_size if ARGS.batch_size is not None else SPLIT["batch_size"]
assert EPOCHS > 0 and BATCH_SIZE > 0 and ARGS.oversample >= 1 and ARGS.pad_factor >= 2
assert ARGS.lr > 0 and ARGS.efficiency_weight >= 0 and ARGS.outside_weight >= 0
SEED = ARGS.seed
torch.manual_seed(SEED)
DEVICE = torch.device("cpu")  # complex FFT on Apple Silicon MPS is not generally reliable

# ----------- THz physical parameters; align with previous steps -----------
C = 299_792_458.0
FREQUENCY = 0.4e12
WAVELENGTH = C / FREQUENCY
N_LAYERS = 5
N_NEURONS = 200
NEURON_SIZE = 400e-6
LAYER_DISTANCE = 30e-3
OUTPUT_DISTANCE = 30e-3
PHASE_MAX = math.pi

# ----------- Provisional choices: investigate before paper comparison ------
BINARIZE_THRESHOLD = 0.5
INPUT_PIXEL_SIZE = NEURON_SIZE
INPUT_TO_FIRST_LAYER = 0.0
DETECTOR_SIDE = 4.8e-3  # Supplementary Fig. S1
DETECTOR_CENTERS = [
    (x_mm * 1e-3, y_mm * 1e-3)
    for y_mm, xs_mm in [
        (12.0, (-12.0, 0.0, 12.0)),
        (0.0, (-18.0, -6.0, 6.0, 18.0)),
        (-12.0, (-12.0, 0.0, 12.0)),
    ]
    for x_mm in xs_mm
]
# The exact detector centers, pixel scaling, initial gap, binarization threshold
# and loss formula/weights are NOT fully specified in the inspected paper text.
EPS = 1e-12

OVERSAMPLE = ARGS.oversample
PAD_FACTOR = ARGS.pad_factor
ACTIVE_N = N_NEURONS * OVERSAMPLE
DX = NEURON_SIZE / OVERSAMPLE
FFT_N = ACTIVE_N * PAD_FACTOR
PAD_LEFT = (FFT_N - ACTIVE_N) // 2
PAD_RIGHT = FFT_N - ACTIVE_N - PAD_LEFT
ACTIVE_SLICE = slice(PAD_LEFT, PAD_LEFT + ACTIVE_N)

ROOT = Path(__file__).resolve().parent
DATA_ROOT = (ARGS.data_root or (ROOT.parent / "data")).expanduser().resolve()
RESULTS_ROOT = (ARGS.results_root or (ROOT / "results" / "step7_2")).expanduser().resolve()

# All of these (except epochs, which may be increased) must match on resume.
RUN_CONFIG = {
    "stage": STAGE, "seed": SEED, "train_each": SPLIT["train_each"],
    "val_each": SPLIT["val_each"], "test_each": SPLIT["test_each"],
    "batch_size": BATCH_SIZE, "oversample": OVERSAMPLE,
    "pad_factor": PAD_FACTOR, "lr": ARGS.lr,
    "efficiency_weight": ARGS.efficiency_weight,
    "outside_weight": ARGS.outside_weight,
    "binarize_threshold": BINARIZE_THRESHOLD,
    "detector_side": DETECTOR_SIDE,
    "detector_centers": DETECTOR_CENTERS,
    "data_root": str(DATA_ROOT),
}

if ARGS.resume is None:
    RUN_NAME = f"{STAGE}_os{OVERSAMPLE}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    OUT_DIR = RESULTS_ROOT / RUN_NAME
    OUT_DIR.mkdir(parents=True, exist_ok=False)
else:
    RESUME_FILE = ARGS.resume.expanduser().resolve()
    if not RESUME_FILE.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {RESUME_FILE}")
    OUT_DIR = RESUME_FILE.parent
BEST_PATH = OUT_DIR / "best_model.pt"
LAST_PATH = OUT_DIR / "last_checkpoint.pt"

print("=" * 72, flush=True)
print("Step 7.2 - THz five-layer D2NN | MNIST medium-scale training", flush=True)
print(f"Stage={STAGE}, epochs={EPOCHS}, batch={BATCH_SIZE}, device={DEVICE}", flush=True)
print(f"Training/validation/test sizes: {10*SPLIT['train_each']}/"
      f"{10*SPLIT['val_each']}/{10*SPLIT['test_each']}", flush=True)
print(f"Wavelength={WAVELENGTH * 1e6:.3f} um; trainable params=200,000", flush=True)
print(f"Oversample={OVERSAMPLE}, dx={DX*1e6:.1f} um; FFT={FFT_N}x{FFT_N}", flush=True)
print(f"MNIST root: {DATA_ROOT}; results: {OUT_DIR}", flush=True)
print("=" * 72, flush=True)
if OVERSAMPLE >= 2:
    print("WARNING: high-resolution FFT backprop may consume significant CPU RAM.", flush=True)


# ------------------------ Dataset, fixed nested splits --------------------
train_mnist = datasets.MNIST(root=str(DATA_ROOT), train=True, download=False,
                            transform=transforms.ToTensor())
test_mnist = datasets.MNIST(root=str(DATA_ROOT), train=False, download=False,
                           transform=transforms.ToTensor())


def balanced_indices(labels: torch.Tensor, n_each: int, seed: int) -> list[list[int]]:
    g = torch.Generator().manual_seed(seed)
    groups = []
    for label in range(10):
        candidates = torch.where(torch.as_tensor(labels) == label)[0]
        if len(candidates) < n_each:
            raise ValueError(f"Insufficient examples for digit {label}")
        selected = candidates[torch.randperm(len(candidates), generator=g)[:n_each]]
        groups.append([int(i) for i in selected])
    return groups


# For fair comparisons, all stages draw from the SAME per-class shuffled pools.
# Pilot training is a subset of medium training; validation/test also nested.
train_groups = balanced_indices(train_mnist.targets, 600, SEED)
test_groups = balanced_indices(test_mnist.targets, 100, SEED + 1)
train_idx = [v for group in train_groups for v in group[:SPLIT["train_each"]]]
val_idx = [v for group in train_groups for v in group[500:500 + SPLIT["val_each"]]]
test_idx = [v for group in test_groups for v in group[:SPLIT["test_each"]]]
assert set(train_idx).isdisjoint(val_idx), "Train/validation overlap"
train_set = Subset(train_mnist, train_idx)
val_set = Subset(train_mnist, val_idx)
test_set = Subset(test_mnist, test_idx)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


def train_loader_for_epoch(epoch: int):
    # Epoch-specific permutation means resume reproduces the same next epoch.
    g = torch.Generator().manual_seed(SEED + 1000 + epoch)
    return DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=0, generator=g)


print("[1/7] Local MNIST loaded and non-overlapping splits prepared", flush=True)


def encode_mnist(images: torch.Tensor) -> torch.Tensor:
    """Binary amplitude on input plane; normalized to unit discrete intensity sum."""
    amp = (images[:, 0] >= BINARIZE_THRESHOLD).float()
    amp = amp.repeat_interleave(OVERSAMPLE, -2).repeat_interleave(OVERSAMPLE, -1)
    size = 28 * OVERSAMPLE
    top = (ACTIVE_N - size) // 2
    bottom = ACTIVE_N - size - top
    amp = F.pad(amp, (top, bottom, top, bottom))
    amp = F.pad(amp, (PAD_LEFT, PAD_RIGHT, PAD_LEFT, PAD_RIGHT))
    total = amp.square().sum(dim=(-2, -1), keepdim=True).clamp_min(EPS)
    return (amp / total.sqrt()).to(torch.complex64)


# -------------------------- Optics / detectors ---------------------------
x = (torch.arange(FFT_N, dtype=torch.float64) - FFT_N // 2) * DX
Y, X = torch.meshgrid(x, x, indexing="ij")
mask_aperture = torch.zeros(FFT_N, FFT_N, dtype=torch.float32)
mask_aperture[ACTIVE_SLICE, ACTIVE_SLICE] = 1.0
side_px = round(DETECTOR_SIDE / DX)
if abs(side_px * DX - DETECTOR_SIDE) > 1e-9:
    raise ValueError("Detector side must be an exact multiple of dx")
det_masks = []
for cx, cy in DETECTOR_CENTERS:
    ix = round(cx / DX) + FFT_N // 2
    iy = round(cy / DX) + FFT_N // 2
    x0, y0 = ix - side_px // 2, iy - side_px // 2
    if not (0 <= x0 <= FFT_N - side_px and 0 <= y0 <= FFT_N - side_px):
        raise ValueError("Detector outside computational window")
    m = torch.zeros((FFT_N, FFT_N), dtype=torch.float32)
    m[y0:y0+side_px, x0:x0+side_px] = 1
    det_masks.append(m)
det_masks = torch.stack(det_masks)
assert det_masks.shape[0] == 10
assert det_masks.sum(dim=0).max().item() == 1, "Overlapping detectors"

freq = torch.fft.fftfreq(FFT_N, d=DX, dtype=torch.float64)
FY, FX = torch.meshgrid(freq, freq, indexing="ij")
k = 2 * math.pi / WAVELENGTH
kz = k * torch.sqrt((1 - (WAVELENGTH*FX)**2 - (WAVELENGTH*FY)**2).to(torch.complex128))


def make_transfer(distance):
    # Remove global exp(i k z); includes decay of evanescent spectral components.
    return torch.exp(1j * (kz - k) * distance).to(torch.complex64)


H_LAYER = make_transfer(LAYER_DISTANCE)
H_OUTPUT = make_transfer(OUTPUT_DISTANCE)


class D2NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_phases = nn.Parameter(0.7 * torch.randn(N_LAYERS, N_NEURONS, N_NEURONS))
        self.register_buffer("H_layer", H_LAYER, persistent=False)
        self.register_buffer("H_output", H_OUTPUT, persistent=False)
        self.register_buffer("aperture", mask_aperture, persistent=False)
        self.register_buffer("detectors", det_masks, persistent=False)

    def phase(self, layer):
        return PHASE_MAX * torch.sigmoid(self.raw_phases[layer])

    def forward(self, u):
        for i in range(N_LAYERS):
            p = self.phase(i)
            if OVERSAMPLE > 1:
                p = p.repeat_interleave(OVERSAMPLE, -2).repeat_interleave(OVERSAMPLE, -1)
            p = F.pad(p, (PAD_LEFT, PAD_RIGHT, PAD_LEFT, PAD_RIGHT))
            u = u * self.aperture * torch.exp(1j * p)
            H = self.H_output if i == N_LAYERS - 1 else self.H_layer
            u = torch.fft.ifft2(torch.fft.fft2(u, dim=(-2, -1)) * H, dim=(-2, -1))
        return u.abs().square()

    def detect(self, intensity):
        return intensity.flatten(1) @ self.detectors.flatten(1).T


model = D2NN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=ARGS.lr)
assert model.raw_phases.numel() == 200_000
print("[2/7] Five-layer ASM network and 3-4-3 detectors prepared", flush=True)


def objective(powers, intensity, labels, input_p):
    """Provisional Step 7.1 loss: cross-entropy + efficiency + outside penalty."""
    correct = powers.gather(1, labels[:, None]).squeeze(1)
    all_detector_p = powers.sum(dim=1)
    total_output_p = intensity.sum(dim=(-2, -1)).clamp_min(EPS)
    correct_eff = correct / input_p.clamp_min(EPS)
    detector_capture = all_detector_p / input_p.clamp_min(EPS)
    window_eff = total_output_p / input_p.clamp_min(EPS)
    # Fractions use discrete intensity sums; not absolute Poynting flux.
    outside_fraction = (1 - all_detector_p / total_output_p).clamp(0, 1)
    ce = F.cross_entropy(torch.log(powers.clamp_min(EPS)), labels)
    loss = (ce + ARGS.efficiency_weight * (1 - correct_eff).mean()
            + ARGS.outside_weight * outside_fraction.mean())
    return loss, correct_eff, detector_capture, window_eff


@torch.no_grad()
def evaluate(loader):
    model.eval()
    sums = {k: 0.0 for k in ("loss", "correct_efficiency", "detector_capture", "window_efficiency")}
    n = n_correct = 0
    cm = torch.zeros(10, 10, dtype=torch.int64)
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        u = encode_mnist(images)
        in_p = u.abs().square().sum(dim=(-2, -1))
        intensity = model(u)
        powers = model.detect(intensity)
        loss, correct_eff, detector_cap, window_eff = objective(powers, intensity, labels, in_p)
        pred = powers.argmax(1)
        b = len(labels)
        n += b
        n_correct += int((pred == labels).sum().item())
        sums["loss"] += loss.item() * b
        sums["correct_efficiency"] += correct_eff.sum().item()
        sums["detector_capture"] += detector_cap.sum().item()
        sums["window_efficiency"] += window_eff.sum().item()
        cm += torch.bincount((labels * 10 + pred).cpu(), minlength=100).reshape(10, 10)
    return {**{k: v / n for k, v in sums.items()},
            "accuracy": n_correct / n, "confusion": cm}


# ----------------------------- Resume / history -------------------------
history = []
start_epoch = 1
best_acc = -1.0
best_val_loss = float("inf")
best_epoch = 0
if ARGS.resume is not None:
    # We create these checkpoints ourselves; weights_only=True accepts tensor,
    # optimizer dictionaries, lists, floats, ints, strings, etc.
    ckpt = torch.load(RESUME_FILE, map_location="cpu", weights_only=True)
    if ckpt["run_config"] != RUN_CONFIG:
        differing = [k for k in RUN_CONFIG if ckpt["run_config"].get(k) != RUN_CONFIG[k]]
        raise ValueError("Resume configuration differs: " + ", ".join(differing))
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    history = ckpt["history"]
    start_epoch = ckpt["epoch"] + 1
    best_acc = ckpt["best_val_accuracy"]
    best_val_loss = ckpt["best_val_loss"]
    best_epoch = ckpt["best_epoch"]
    torch.set_rng_state(ckpt["rng_state"])
    print(f"[3/7] Resumed at epoch {start_epoch} from {RESUME_FILE}", flush=True)
else:
    print("[3/7] New training run", flush=True)

if start_epoch > EPOCHS:
    print(f"Checkpoint already at epoch {start_epoch - 1}; no extra epochs requested.", flush=True)


def save_checkpoint(epoch):
    payload = {
        "epoch": epoch,
        "run_config": RUN_CONFIG,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_accuracy": best_acc,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "history": history,
        "rng_state": torch.get_rng_state(),
    }
    tmp = LAST_PATH.with_suffix(".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, LAST_PATH)


def persist_history():
    if not history:
        return
    with (OUT_DIR / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


# ------------------------------ Training --------------------------------
print(f"[4/7] Training: epoch {start_epoch} ... {EPOCHS}", flush=True)
try:
    for epoch in range(start_epoch, EPOCHS + 1):
        t0 = time.perf_counter()
        model.train()
        n = n_correct = 0
        sum_loss = 0.0
        train_ceff = train_deff = train_weff = 0.0
        for batch_i, (images, labels) in enumerate(train_loader_for_epoch(epoch), start=1):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            u = encode_mnist(images)
            in_p = u.abs().square().sum(dim=(-2, -1))
            optimizer.zero_grad(set_to_none=True)
            intensity = model(u)
            powers = model.detect(intensity)
            loss, ceff, deff, weff = objective(powers, intensity, labels, in_p)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Nonfinite loss at epoch={epoch}, batch={batch_i}")
            loss.backward()
            optimizer.step()
            n_batch = labels.numel()
            n += n_batch
            sum_loss += loss.item() * n_batch
            n_correct += int((powers.argmax(1) == labels).sum().item())
            train_ceff += ceff.detach().sum().item()
            train_deff += deff.detach().sum().item()
            train_weff += weff.detach().sum().item()
            if batch_i == 1 or batch_i % 50 == 0:
                print(f"  epoch={epoch} batch={batch_i:4d} samples={n:5d}/{len(train_set)} "
                      f"loss={sum_loss/n:.4f} elapsed={time.perf_counter()-t0:.1f}s", flush=True)

        val = evaluate(val_loader)
        elapsed = time.perf_counter() - t0
        entry = {
            "epoch": epoch, "train_loss": sum_loss / n,
            "train_accuracy": n_correct / n,
            "train_correct_efficiency": train_ceff / n,
            "train_detector_capture": train_deff / n,
            "train_window_efficiency": train_weff / n,
            "val_loss": val["loss"], "val_accuracy": val["accuracy"],
            "val_correct_efficiency": val["correct_efficiency"],
            "val_detector_capture": val["detector_capture"],
            "val_window_efficiency": val["window_efficiency"],
            "epoch_seconds": elapsed,
        }
        history.append(entry)
        # Save model selected only by validation results (not test set).
        if (val["accuracy"] > best_acc or
            (val["accuracy"] == best_acc and val["loss"] < best_val_loss)):
            best_acc, best_val_loss, best_epoch = val["accuracy"], val["loss"], epoch
            best_tmp = BEST_PATH.with_suffix(".tmp")
            torch.save({"model_state_dict": model.state_dict(),
                        "epoch": epoch, "validation_accuracy": best_acc,
                        "run_config": RUN_CONFIG}, best_tmp)
            os.replace(best_tmp, BEST_PATH)
        save_checkpoint(epoch)
        persist_history()
        print(f"Epoch {epoch:02d}/{EPOCHS}: loss={entry['train_loss']:.4f}, "
              f"train={entry['train_accuracy']:.2%}, val={val['accuracy']:.2%}, "
              f"val correct={val['correct_efficiency']:.2%}, "
              f"window={val['window_efficiency']:.2%}, elapsed={elapsed:.1f}s", flush=True)
except KeyboardInterrupt:
    print(f"\nInterrupted. Last completed epoch checkpoint: {LAST_PATH}", flush=True)
    raise SystemExit(130)

print("[5/7] Loading best validation checkpoint and evaluating test set", flush=True)
ckpt_best = torch.load(BEST_PATH, map_location="cpu", weights_only=True)
model.load_state_dict(ckpt_best["model_state_dict"])
test = evaluate(test_loader)
cm = test["confusion"].numpy()

# ----------------------------- Plots ------------------------------------
epoch_axis = [x["epoch"] for x in history]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, metric, title in [
    (axes[0, 0], "loss", "Loss"),
    (axes[0, 1], "accuracy", "Accuracy"),
    (axes[1, 0], "correct_efficiency", "Correct-detector / input-light fraction"),
    (axes[1, 1], "window_efficiency", "Output-window / input-light fraction"),
]:
    ax.plot(epoch_axis, [h[f"train_{metric}"] for h in history], marker="o", label="Train (online)")
    ax.plot(epoch_axis, [h[f"val_{metric}"] for h in history], marker="o", label="Validation")
    ax.set(title=title, xlabel="Epoch")
    if metric == "accuracy":
        ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "training_curves.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(epoch_axis, [h["epoch_seconds"] for h in history], marker="o")
ax.set(xlabel="Epoch", ylabel="Seconds incl. validation", title="Epoch runtime")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(OUT_DIR / "epoch_times.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cm, cmap="Blues")
ax.set(title=f"Test confusion matrix ({test['accuracy']:.2%})",
       xlabel="Predicted digit", ylabel="True digit",
       xticks=list(range(10)), yticks=list(range(10)))
for row in range(10):
    for col in range(10):
        ax.text(col, row, str(cm[row, col]), ha="center", va="center", fontsize=8,
                color="white" if cm[row, col] > cm.max() / 2 else "black")
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=200)
plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(det_masks.sum(0)[ACTIVE_SLICE, ACTIVE_SLICE].numpy(),
          cmap="gray_r", origin="lower", extent=(-40, 40, -40, 40), vmin=0, vmax=1)
for digit, (cx, cy) in enumerate(DETECTOR_CENTERS):
    ax.text(cx * 1e3, cy * 1e3, str(digit), ha="center", va="center", color="white",
            fontsize=9, bbox=dict(facecolor="black", alpha=0.7, pad=0.6))
ax.set(title="3-4-3 detectors (4.8mm width; approximate centers)",
       xlabel="x (mm)", ylabel="y (mm)")
fig.tight_layout()
fig.savefig(OUT_DIR / "detector_geometry.png", dpi=200)
plt.close(fig)

fig, axes = plt.subplots(1, N_LAYERS, figsize=(16, 4))
for l, ax in enumerate(axes):
    img = ax.imshow(model.phase(l).detach().numpy(), cmap="twilight", origin="lower",
                    vmin=0, vmax=math.pi, extent=(-40, 40, -40, 40))
    ax.set(title=f"Layer {l+1}", xlabel="x (mm)", ylabel="y (mm)")
fig.colorbar(img, ax=list(axes), shrink=0.7, label="Phase (rad)")
fig.savefig(OUT_DIR / "phase_layers.png", dpi=200, bbox_inches="tight")
plt.close(fig)

wanted = [1, 4, 7, 9]
examples = []
for digit in wanted:
    for j in range(len(test_set)):
        image, label = test_set[j]
        if int(label) == digit:
            examples.append((image, digit))
            break
if len(examples) == len(wanted):
    input_imgs = torch.stack([img for img, _ in examples])
    with torch.no_grad():
        output_intensity = model(encode_mnist(input_imgs))
        predictions = model.detect(output_intensity).argmax(dim=1).tolist()
    fig, axes = plt.subplots(len(wanted), 2, figsize=(9, 12))
    for i, (image, digit) in enumerate(examples):
        axes[i, 0].imshow(image[0], cmap="gray")
        axes[i, 0].set_title(f"Input {digit}")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(output_intensity[i, ACTIVE_SLICE, ACTIVE_SLICE],
                          cmap="inferno", origin="lower", extent=(-40, 40, -40, 40))
        axes[i, 1].set_title(f"True {digit}, predicted {predictions[i]}")
        for label, (cx, cy) in enumerate(DETECTOR_CENTERS):
            cxmm, cymm, side = cx * 1e3, cy * 1e3, DETECTOR_SIDE * 1e3
            axes[i, 1].add_patch(Rectangle((cxmm-side/2, cymm-side/2), side, side,
                                            fill=False, edgecolor="white", linewidth=0.7))
            axes[i, 1].text(cxmm, cymm, str(label), ha="center", va="center", fontsize=7, color="white")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "classification_examples.png", dpi=200)
    plt.close(fig)

metrics = {
    "stage": STAGE, "best_epoch": best_epoch, "best_validation_accuracy": best_acc,
    "test_accuracy": test["accuracy"], "test_loss": test["loss"],
    "correct_detector_efficiency": test["correct_efficiency"],
    "detector_capture_efficiency": test["detector_capture"],
    "output_window_efficiency": test["window_efficiency"],
    "confusion_matrix": cm.tolist(),
    "class_recall": {str(d): cm[d, d] / max(int(cm[d].sum()), 1) for d in range(10)},
    "epochs_completed": len(history),
    "total_epoch_seconds": sum(h["epoch_seconds"] for h in history),
    "run_config": RUN_CONFIG,
    "physical": {"frequency_hz": FREQUENCY, "wavelength_m": WAVELENGTH,
                 "layer_count": N_LAYERS, "neurons_per_layer": N_NEURONS**2,
                 "neuron_pitch_m": NEURON_SIZE, "layer_gap_m": LAYER_DISTANCE,
                 "output_gap_m": OUTPUT_DISTANCE},
    "sampling": {"dx_m": DX, "fft_grid": FFT_N},
    "notes": "Pilot reproduction. Detector center spacing, input scaling/distance, "
             "0.5 threshold and loss coefficients are provisional; "
             "oversample=1 is not proven spatially converged."
}
with (OUT_DIR / "metrics.json").open("w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2, ensure_ascii=False)
print("[6/7] Saved plots, metrics, and model checkpoints", flush=True)
print("=" * 72)
print(f"Best validation: {best_acc:.2%} (epoch {best_epoch})")
print(f"Test accuracy: {test['accuracy']:.2%}")
print(f"Correct-detector / input-light: {test['correct_efficiency']:.2%}")
print(f"All detectors / input-light: {test['detector_capture']:.2%}")
print(f"Output-window / input-light: {test['window_efficiency']:.2%}")
print(f"Epoch wall-time sum: {metrics['total_epoch_seconds']:.1f} s")
print(f"Saved in: {OUT_DIR}")
print("[7/7] Step 7.2 completed", flush=True)
