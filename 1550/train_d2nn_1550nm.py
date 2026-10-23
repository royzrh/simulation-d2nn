#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 7.4: CUDA MNIST training of a five-layer 1550 nm D2NN.
Retains Step 7.2 full-grid ASM propagation with per-layer apertures.
Two research objectives:
baseline:
    Cross-entropy + linear efficiency + outside fraction.
joint:
    Cross-entropy + log(correct efficiency)
    + log(detector capture) + outside fraction.
Neither objective is claimed to reproduce the original paper's
exact loss formula.
Geometry, digit scaling, threshold and input gap remain provisional.
GPU:
    --device cuda --oversample 4 --batch-size 8
Full dataset:
    55000 training
    5000 validation
    10000 testing
No test metrics are used for model selection.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
# ============================================================
# 1. Physical parameters
# ============================================================
C = 299_792_458.0
WAVELENGTH = 1550e-9
FREQUENCY = C / WAVELENGTH
NLAYERS = 5
NEURONS = 200
PITCH = 400e-6
GAP = 30e-3
DETECTOR_SIDE = 4.8e-3
EPS = 1e-12
# Provisional detector coordinates.
# Consistent with previous Step 7.2 experiments.
DETECTOR_CENTERS = [
    (x * 1e-3, y * 1e-3)
    for y, xs in (
        (12., (-12., 0., 12.)),
        (0., (-18., -6., 6., 18.)),
        (-12., (-12., 0., 12.)),
    )
    for x in xs
]
# ============================================================
# 2. Command-line arguments
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="D2NN 5-layer GPU MNIST full training"
    )
    p.add_argument(
        "--stage",
        choices=["smoke", "pilot", "full"],
        default="smoke"
    )
    p.add_argument(
        "--loss",
        choices=["baseline", "joint"],
        default="baseline"
    )
    p.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto"
    )
    p.add_argument(
        "--oversample",
        type=int,
        default=4
    )
    p.add_argument(
        "--pad-factor",
        type=int,
        default=2
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None
    )
    p.add_argument(
        "--lr",
        type=float,
        default=0.005
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5
    )
    p.add_argument(
        "--input-pixel-scale",
        type=int,
        default=1,
        help="Number of physical neurons per MNIST pixel"
    )
    p.add_argument(
        "--input-gap-mm",
        type=float,
        default=0.,
        help="Input to first layer distance in millimeters"
    )
    p.add_argument("--wavelength-nm", type=float, default=1550.,
                   help="Vacuum wavelength in nanometers")
    p.add_argument("--pitch-um", type=float, default=400.,
                   help="Physical neuron pitch in micrometers")
    p.add_argument("--gap-mm", type=float, default=30.,
                   help="Interlayer and final output distance in millimeters")
    p.add_argument("--detector-side-mm", type=float, default=4.8,
                   help="Square detector width in millimeters")
    p.add_argument("--detector-coordinate-scale", type=float, default=1.,
                   help="Scale the original detector center coordinates")
    p.add_argument(
        "--num-workers",
        type=int,
        default=0
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=200
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=None
    )
    p.add_argument(
        "--results-root",
        type=Path,
        default=None
    )
    p.add_argument(
        "--resume",
        type=Path,
        default=None
    )
    p.add_argument(
        "--train-count",
        type=int,
        default=None
    )
    p.add_argument(
        "--val-count",
        type=int,
        default=None
    )
    p.add_argument(
        "--test-count",
        type=int,
        default=None
    )
    a = p.parse_args()
    if a.epochs is None:
        a.epochs = {
            "smoke": 1,
            "pilot": 3,
            "full": 10
        }[a.stage]
    for value in (
        a.epochs,
        a.batch_size,
        a.oversample,
        a.pad_factor,
        a.input_pixel_scale
    ):
        if value < 1:
            p.error(
                "epochs/batch/oversample/pad/"
                "input pixel scale must be >= 1"
            )
    if a.pad_factor < 2:
        p.error("pad-factor must be >= 2")
    if a.input_pixel_scale * 28 > NEURONS:
        p.error("Input exceeds 200x200 aperture")
    if (
        a.lr <= 0
        or not (0 <= a.threshold <= 1)
        or a.input_gap_mm < 0
    ):
        p.error("Invalid training or optical parameters")
    return a
# ============================================================
# 3. Optical D2NN
# ============================================================
class OpticalD2NN(nn.Module):
    def __init__(
        self,
        over: int,
        pad_factor: int,
        input_gap_mm: float
    ):
        super().__init__()
        self.over = over
        self.active_n = NEURONS * over
        self.dx = PITCH / over
        self.fft_n = self.active_n * pad_factor
        self.left = (
            self.fft_n - self.active_n
        ) // 2
        self.right = (
            self.fft_n - self.active_n - self.left
        )
        self.s = slice(
            self.left,
            self.left + self.active_n
        )
        # Five trainable phase masks.
        self.raw_phases = nn.Parameter(
            0.7 * torch.randn(
                NLAYERS,
                NEURONS,
                NEURONS
            )
        )
        # Active optical aperture.
        aperture = torch.zeros(
            self.fft_n,
            self.fft_n,
            dtype=torch.float32
        )
        aperture[self.s, self.s] = 1.
        self.register_buffer(
            "aperture",
            aperture,
            persistent=False
        )
        # Detector masks.
        detectors = []
        side_px = round(
            DETECTOR_SIDE / self.dx
        )
        if abs(
            side_px * self.dx - DETECTOR_SIDE
        ) > 1e-9:
            raise ValueError(
                "Detector width must be an integer "
                "number of simulation pixels"
            )
        for cx, cy in DETECTOR_CENTERS:
            ix = (
                round(cx / self.dx)
                + self.fft_n // 2
            )
            iy = (
                round(cy / self.dx)
                + self.fft_n // 2
            )
            x0 = ix - side_px // 2
            y0 = iy - side_px // 2
            if (
                min(x0, y0) < 0
                or x0 + side_px > self.fft_n
                or y0 + side_px > self.fft_n
            ):
                raise ValueError(
                    "Detector outside FFT grid"
                )
            mask = torch.zeros(
                self.fft_n,
                self.fft_n,
                dtype=torch.float32
            )
            mask[
                y0:y0 + side_px,
                x0:x0 + side_px
            ] = 1.
            detectors.append(mask)
        det = torch.stack(detectors)
        if det.sum(dim=0).max().item() > 1:
            raise ValueError(
                "Detector overlaps found"
            )
        self.register_buffer(
            "detectors",
            det,
            persistent=False
        )
        # ASM transfer functions.
        # Use the same propagation convention as Step 7.2.
        f = torch.fft.fftfreq(
            self.fft_n,
            d=self.dx,
            dtype=torch.float64
        )
        fy, fx = torch.meshgrid(
            f,
            f,
            indexing="ij"
        )
        k = 2 * math.pi / WAVELENGTH
        kz = k * torch.sqrt(
            (
                1.
                - (WAVELENGTH * fx) ** 2
                - (WAVELENGTH * fy) ** 2
            ).to(torch.complex128)
        )
        def transfer(z):
            return torch.exp(
                1j * (kz - k) * z
            ).to(torch.complex64)
        self.register_buffer(
            "H_layer",
            transfer(GAP),
            persistent=False
        )
        self.register_buffer(
            "H_output",
            transfer(GAP),
            persistent=False
        )
        self.register_buffer(
            "H_input",
            (
                transfer(input_gap_mm * 1e-3)
                if input_gap_mm
                else torch.ones_like(self.H_layer)
            ),
            persistent=False
        )
        self.has_input_gap = input_gap_mm > 0
    # --------------------------------------------------------
    # MNIST amplitude encoding
    # --------------------------------------------------------
    def encode(
        self,
        imgs: torch.Tensor,
        threshold: float,
        scale: int
    ) -> torch.Tensor:
        amp = (
            imgs[:, 0] >= threshold
        ).to(torch.float32)
        rep = self.over * scale
        amp = amp.repeat_interleave(
            rep,
            -2
        ).repeat_interleave(
            rep,
            -1
        )
        side = amp.shape[-1]
        a = (
            self.active_n - side
        ) // 2
        b = self.active_n - side - a
        amp = F.pad(
            amp,
            (a, b, a, b)
        )
        amp = F.pad(
            amp,
            (
                self.left,
                self.right,
                self.left,
                self.right
            )
        )
        total = amp.square().sum(
            (-2, -1),
            keepdim=True
        ).clamp_min(EPS)
        amp = amp / total.sqrt()
        return amp.to(torch.complex64)
    # --------------------------------------------------------
    # Full-grid optical propagation
    # --------------------------------------------------------
    def forward(self, field: torch.Tensor):
        if self.has_input_gap:
            field = torch.fft.ifft2(
                torch.fft.fft2(field)
                * self.H_input
            )
        for layer in range(NLAYERS):
            phi = math.pi * torch.sigmoid(
                self.raw_phases[layer]
            )
            if self.over > 1:
                phi = phi.repeat_interleave(
                    self.over,
                    -2
                ).repeat_interleave(
                    self.over,
                    -1
                )
            phi = F.pad(
                phi,
                (
                    self.left,
                    self.right,
                    self.left,
                    self.right
                )
            )
            transmission = torch.exp(
                1j * phi
            )
            field = (
                field
                * self.aperture
                * transmission
            )
            H = (
                self.H_output
                if layer == NLAYERS - 1
                else self.H_layer
            )
            spectrum = torch.fft.fft2(
                field
            )
            field = torch.fft.ifft2(
                spectrum * H
            )
        intensity = field.abs().square()
        powers = (
            intensity.flatten(1)
            @ self.detectors.flatten(1).T
        )
        return intensity, powers
# ============================================================
# 4. Loss functions
# ============================================================
def compute_loss(
    method,
    power,
    intensity,
    labels,
    input_power,
    active_slice
):
    correct = power.gather(
        1,
        labels[:, None]
    ).squeeze(1)
    capture = power.sum(dim=1)
    total = intensity.sum(
        (-2, -1)
    ).clamp_min(EPS)
    active = intensity[
        :,
        active_slice,
        active_slice
    ].sum((-2, -1))
    inp = input_power.clamp_min(EPS)
    ce = F.cross_entropy(
        power.clamp_min(EPS).log(),
        labels
    )
    correct_eff = correct / inp
    detector_eff = capture / inp
    fft_eff = total / inp
    active_eff = active / inp
    outside = (
        1 - capture / total
    ).clamp(0, 1)
    if method == "baseline":
        loss = (
            ce
            + 0.5 * (1 - correct_eff).mean()
            + 0.05 * outside.mean()
        )
    else:
        loss = (
            ce
            - 0.05 * correct_eff.clamp_min(
                EPS
            ).log().mean()
            - 0.05 * detector_eff.clamp_min(
                EPS
            ).log().mean()
            + 0.05 * outside.mean()
        )
    stats = {
        "loss": loss,
        "ce": ce,
        "accuracy": (
            power.argmax(1) == labels
        ).float().mean(),
        "correct_efficiency": (
            correct_eff.mean()
        ),
        "detector_capture": (
            detector_eff.mean()
        ),
        "correct_share_of_detectors": (
            correct
            / capture.clamp_min(EPS)
        ).mean(),
        "fft_window_efficiency": (
            fft_eff.mean()
        ),
        "active_80mm_efficiency": (
            active_eff.mean()
        )
    }
    return loss, stats
def to_numpy_metrics(metrics: dict) -> dict:
    return {
        k: float(v.detach().item())
        for k, v in metrics.items()
    }
# ============================================================
# 5. DataLoader
# ============================================================
def make_loader(
    ds,
    indices,
    batch_size,
    workers,
    shuffle,
    generator_seed,
    pin
):
    generator = torch.Generator().manual_seed(
        generator_seed
    )
    return DataLoader(
        Subset(ds, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=(
            generator if shuffle else None
        ),
        num_workers=workers,
        pin_memory=pin,
        persistent_workers=workers > 0
    )
# ============================================================
# 6. Training and evaluation
# ============================================================
def run_epoch(
    model,
    loader,
    device,
    threshold,
    input_scale,
    method,
    optimizer=None,
    log_every=0,
    epoch=0
):
    training = optimizer is not None
    model.train(training)
    names = (
        "loss",
        "ce",
        "correct_efficiency",
        "detector_capture",
        "correct_share_of_detectors",
        "fft_window_efficiency",
        "active_80mm_efficiency"
    )
    totals = {
        k: 0.0 for k in names
    }
    n = 0
    right = 0
    confusion = torch.zeros(
        (10, 10),
        dtype=torch.int64
    )
    start = time.perf_counter()
    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )
    with context:
        for idx, (images, labels) in enumerate(
            loader,
            start=1
        ):
            images = images.to(
                device,
                non_blocking=True
            )
            labels = labels.to(
                device,
                non_blocking=True
            )
            inputs = model.encode(
                images,
                threshold,
                input_scale
            )
            input_p = inputs.abs().square().sum(
                (-2, -1)
            )
            if training:
                optimizer.zero_grad(
                    set_to_none=True
                )
            intensity, powers = model(inputs)
            loss, measures = compute_loss(
                method,
                powers,
                intensity,
                labels,
                input_p,
                model.s
            )
            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"Nonfinite loss at "
                    f"epoch {epoch}, batch {idx}"
                )
            if training:
                loss.backward()
                gradients = model.raw_phases.grad
                if not torch.isfinite(
                    gradients
                ).all().item():
                    raise FloatingPointError(
                        "Nonfinite phase gradient"
                    )
                optimizer.step()
            b = len(labels)
            n += b
            right += int(
                (
                    powers.argmax(1) == labels
                ).sum().item()
            )
            scalar = to_numpy_metrics(
                measures
            )
            for k in totals:
                totals[k] += (
                    scalar[k] * b
                )
            pred = powers.argmax(1)
            c = torch.bincount(
                (
                    labels * 10 + pred
                ).cpu(),
                minlength=100
            ).view(10, 10)
            confusion += c
            if (
                training
                and log_every
                and (
                    idx == 1
                    or idx % log_every == 0
                )
            ):
                print(
                    f"  epoch={epoch:02d} "
                    f"batch={idx:5d}/{len(loader)} "
                    f"samples={n:5d} "
                    f"loss={totals['loss']/n:.4f} "
                    f"elapsed={time.perf_counter()-start:.1f}s",
                    flush=True
                )
    out = {
        k: v / n
        for k, v in totals.items()
    }
    out["accuracy"] = right / n
    out["n"] = n
    out["confusion"] = confusion
    return out
def json_metrics(m):
    return {
        k: (
            v.tolist()
            if isinstance(v, torch.Tensor)
            else v
        )
        for k, v in m.items()
    }
# ============================================================
# 7. Checkpoint and history
# ============================================================
def save_atomic(obj, destination: Path):
    temporary = destination.with_suffix(
        ".tmp"
    )
    torch.save(
        obj,
        temporary
    )
    os.replace(
        temporary,
        destination
    )
def save_history(history, path):
    if not history:
        return
    with path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(history[0])
        )
        writer.writeheader()
        writer.writerows(history)
# ============================================================
# 8. Result plots
# ============================================================
def final_plots(
    history,
    cm,
    out: Path,
    model
):
    epochs = [
        h["epoch"]
        for h in history
    ]
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4.5)
    )
    for ax, key in (
        (axes[0], "loss"),
        (axes[1], "accuracy")
    ):
        ax.plot(
            epochs,
            [
                h["train_" + key]
                for h in history
            ],
            marker="o",
            label="Train"
        )
        ax.plot(
            epochs,
            [
                h["val_" + key]
                for h in history
            ],
            marker="o",
            label="Validation"
        )
        ax.set(
            xlabel="Epoch",
            title=key.capitalize()
        )
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    fig.savefig(
        out / "training_curves.png",
        dpi=180
    )
    plt.close(fig)
    # Confusion matrix
    fig, ax = plt.subplots(
        figsize=(7, 6)
    )
    c = np.asarray(cm)
    ax.imshow(
        c,
        cmap="Blues"
    )
    for i in range(10):
        for j in range(10):
            ax.text(
                j,
                i,
                str(c[i, j]),
                ha="center",
                va="center",
                fontsize=8,
                color=(
                    "white"
                    if c[i, j] > c.max() / 2
                    else "black"
                )
            )
    ax.set(
        xlabel="Predicted",
        ylabel="Ground truth",
        title="Final evaluation confusion matrix",
        xticks=list(range(10)),
        yticks=list(range(10))
    )
    fig.tight_layout()
    fig.savefig(
        out / "confusion_matrix.png",
        dpi=180
    )
    plt.close(fig)
    # Five phase masks
    fig, axes = plt.subplots(
        1,
        5,
        figsize=(18, 4)
    )
    phases = (
        math.pi
        * torch.sigmoid(
            model.raw_phases.detach()
        )
    ).cpu().numpy()
    for j, ax in enumerate(axes):
        ax.imshow(
            phases[j],
            cmap="twilight",
            vmin=0,
            vmax=math.pi,
            origin="lower"
        )
        ax.set(
            title=f"Layer {j + 1}"
        )
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(
        out / "phase_layers.png",
        dpi=160
    )
    plt.close(fig)
# ============================================================
# 9. Main program
# ============================================================
def main():
    a = parse_args()
    # Geometry is explicit: changing wavelength alone does not rescale a device.
    global WAVELENGTH, FREQUENCY, PITCH, GAP, DETECTOR_SIDE, DETECTOR_CENTERS
    physical_values = (a.wavelength_nm, a.pitch_um, a.gap_mm,
                       a.detector_side_mm, a.detector_coordinate_scale)
    if not all(math.isfinite(v) and v > 0 for v in physical_values):
        raise ValueError("Wavelength and geometry must be positive finite values")
    if not math.isfinite(a.input_gap_mm) or a.input_gap_mm < 0:
        raise ValueError("Input gap must be a nonnegative finite value")
    WAVELENGTH = a.wavelength_nm * 1e-9
    FREQUENCY = C / WAVELENGTH
    PITCH = a.pitch_um * 1e-6
    GAP = a.gap_mm * 1e-3
    DETECTOR_SIDE = a.detector_side_mm * 1e-3
    DETECTOR_CENTERS = [
        (x * 1e-3 * a.detector_coordinate_scale,
         y * 1e-3 * a.detector_coordinate_scale)
        for y, xs in ((12., (-12., 0., 12.)),
                      (0., (-18., -6., 6., 18.)),
                      (-12., (-12., 0., 12.)))
        for x in xs
    ]
    print(f"[PHYSICS] wavelength={a.wavelength_nm:g} nm; "
          f"pitch={a.pitch_um:g} um; aperture={NEURONS * PITCH * 1e3:g} mm; "
          f"gap/output={a.gap_mm:g} mm; input gap={a.input_gap_mm:g} mm; "
          f"detector side={a.detector_side_mm:g} mm; "
          f"detector coordinate scale={a.detector_coordinate_scale:g}", flush=True)
    here = Path(__file__).resolve().parent
    data_root = (
        a.data_root
        or here.parent / "data"
    ).expanduser().resolve()
    results_root = (
        a.results_root
        or here / "results" / "step7_4"
    ).expanduser().resolve()
    # Reproducibility
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    # Device selection
    if (
        a.device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable"
        )
    if (
        a.device == "cuda"
        or (
            a.device == "auto"
            and torch.cuda.is_available()
        )
    ):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # Stage sizes
    counts = {
        "smoke": (
            40,
            20,
            20
        ),
        "pilot": (
            1000,
            200,
            1000
        ),
        "full": (
            55000,
            5000,
            10000
        )
    }
    tn, vn, xn = counts[a.stage]
    tn = (
        a.train_count
        if a.train_count is not None
        else tn
    )
    vn = (
        a.val_count
        if a.val_count is not None
        else vn
    )
    xn = (
        a.test_count
        if a.test_count is not None
        else xn
    )
    if min(tn, vn, xn) <= 0:
        raise ValueError(
            "Dataset counts must be positive"
        )
    # MNIST loading
    train_mnist = datasets.MNIST(
        root=str(data_root),
        train=True,
        download=False,
        transform=transforms.ToTensor()
    )
    test_mnist = datasets.MNIST(
        root=str(data_root),
        train=False,
        download=False,
        transform=transforms.ToTensor()
    )
    if (
        len(train_mnist) < tn + vn
        or len(test_mnist) < xn
    ):
        raise RuntimeError(
            "MNIST dataset is incomplete"
        )
    # Reproducible train / validation split
    gen = torch.Generator().manual_seed(
        a.seed
    )
    all_idx = torch.randperm(
        len(train_mnist),
        generator=gen
    )
    train_idx = all_idx[:tn].tolist()
    val_idx = all_idx[
        tn:tn + vn
    ].tolist()
    test_idx = list(range(xn))
    split_id = hashlib.sha256(
        all_idx.numpy().tobytes()
    ).hexdigest()[:16]
    # Experiment configuration
    cfg = {
        "wavelength_m": WAVELENGTH,
        "frequency_hz": FREQUENCY,
        "pitch_m": PITCH,
        "interlayer_gap_m": GAP,
        "output_gap_m": GAP,
        "layers": NLAYERS,
        "neurons_per_side": NEURONS,
        "aperture_side_m": NEURONS * PITCH,
        "detector_coordinate_scale": a.detector_coordinate_scale,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "stage": a.stage,
        "loss": a.loss,
        "oversample": a.oversample,
        "pad_factor": a.pad_factor,
        "batch_size": a.batch_size,
        "lr": a.lr,
        "seed": a.seed,
        "threshold": a.threshold,
        "input_pixel_scale": (
            a.input_pixel_scale
        ),
        "input_gap_mm": a.input_gap_mm,
        "train_count": tn,
        "val_count": vn,
        "test_count": xn,
        "split_id": split_id,
        "data_root": str(data_root),
        "geometry": (
            "step7_2_full_fft_aperture_each_layer"
        ),
        "detector_centers": (
            DETECTOR_CENTERS
        ),
        "detector_side_m": (
            DETECTOR_SIDE
        ),
        "source_paper_reproduction": (
            "partial, geometry/loss/encoding provisional"
        )
    }
    # Output directory
    if a.resume:
        checkpoint_path = (
            a.resume.expanduser().resolve()
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                checkpoint_path
            )
        out = checkpoint_path.parent
    else:
        out = results_root / (
            f"{a.stage}_{a.loss}_"
            f"os{a.oversample}_"
            f"seed{a.seed}_"
            f"{datetime.now():%Y%m%d_%H%M%S}"
        )
        out.mkdir(
            parents=True,
            exist_ok=False
        )
    best_path = out / "best_model.pt"
    last_path = out / "last_checkpoint.pt"
    # Check resume configuration
    if a.resume:
        preliminary = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )
        if preliminary["config"] != cfg:
            diffs = [
                k for k in cfg
                if preliminary["config"].get(k)
                != cfg[k]
            ]
            raise ValueError(
                "Resume configuration mismatch: "
                + ", ".join(diffs)
            )
        del preliminary
    else:
        with (
            out / "config.json"
        ).open(
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                cfg,
                f,
                indent=2,
                ensure_ascii=False
            )
    # DataLoaders
    pin = device.type == "cuda"
    valid = make_loader(
        train_mnist,
        val_idx,
        a.batch_size,
        a.num_workers,
        False,
        0,
        pin
    )
    test = (
        make_loader(
            test_mnist,
            test_idx,
            a.batch_size,
            a.num_workers,
            False,
            0,
            pin
        )
        if a.stage == "full"
        else None
    )
    # Initialize optical network
    model = OpticalD2NN(
        a.oversample,
        a.pad_factor,
        a.input_gap_mm
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=a.lr
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    # Training state
    hist = []
    best_val_acc = -1.
    best_val_loss = float("inf")
    best_epoch = 0
    start_epoch = 1
    # Resume previous experiment
    if a.resume:
        cp = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )
        model.load_state_dict(
            cp["model_state_dict"]
        )
        optimizer.load_state_dict(
            cp["optimizer_state_dict"]
        )
        hist = cp["history"]
        best_val_acc = cp["best_val_acc"]
        best_val_loss = cp["best_val_loss"]
        best_epoch = cp["best_epoch"]
        start_epoch = cp["epoch"] + 1
        torch.set_rng_state(
            cp["cpu_rng_state"]
        )
        if (
            device.type == "cuda"
            and cp["cuda_rng_state"] is not None
        ):
            torch.cuda.set_rng_state(
                cp["cuda_rng_state"]
            )
    # Experiment description
    print("=" * 72, flush=True)
    print(
        f"Step 7.4 | "
        f"stage={a.stage} "
        f"loss={a.loss} "
        f"OS={a.oversample} "
        f"batch={a.batch_size}",
        flush=True
    )
    print(
        f"Device={device} "
        f"PyTorch={torch.__version__} "
        f"data={data_root}",
        flush=True
    )
    print(
        f"Grid={model.fft_n}x{model.fft_n}, "
        f"neurons={model.raw_phases.numel()}, "
        f"split={split_id}",
        flush=True
    )
    print(
        f"Train={tn} val={vn} test={xn} "
        f"epochs={a.epochs}; "
        f"output={out}",
        flush=True
    )
    print(
        "WARNING: detector centers, input size, "
        "threshold and input gap are provisional.",
        flush=True
    )
    print(
        "WARNING: neither loss is proven "
        "to match the exact paper objective.",
        flush=True
    )
    print("=" * 72, flush=True)
    start_time = time.perf_counter()
    # ========================================================
    # Training loop
    # ========================================================
    for epoch in range(
        start_epoch,
        a.epochs + 1
    ):
        e0 = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        train_loader = make_loader(
            train_mnist,
            train_idx,
            a.batch_size,
            a.num_workers,
            True,
            a.seed + 1000 + epoch,
            pin
        )
        tr = run_epoch(
            model,
            train_loader,
            device,
            a.threshold,
            a.input_pixel_scale,
            a.loss,
            optimizer=optimizer,
            log_every=a.log_every,
            epoch=epoch
        )
        va = run_epoch(
            model,
            valid,
            device,
            a.threshold,
            a.input_pixel_scale,
            a.loss
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (
            time.perf_counter() - e0
        )
        peak = (
            torch.cuda.max_memory_allocated()
            / 1024 ** 3
            if device.type == "cuda"
            else 0
        )
        row = {
            "epoch": epoch,
            **{
                "train_" + k: v
                for k, v in tr.items()
                if k not in (
                    "confusion",
                    "n"
                )
            },
            **{
                "val_" + k: v
                for k, v in va.items()
                if k not in (
                    "confusion",
                    "n"
                )
            },
            "seconds": elapsed,
            "cuda_peak_allocated_gb": peak
        }
        hist.append(row)
        # Select best model by validation set only
        if (
            va["accuracy"] > best_val_acc
            or (
                va["accuracy"] == best_val_acc
                and va["loss"] < best_val_loss
            )
        ):
            best_val_acc = va["accuracy"]
            best_val_loss = va["loss"]
            best_epoch = epoch
            save_atomic(
                {
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "config": cfg,
                    "epoch": epoch,
                    "validation_accuracy": (
                        best_val_acc
                    )
                },
                best_path
            )
        # Save resume checkpoint
        cp = {
            "epoch": epoch,
            "config": cfg,
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "history": hist,
            "best_val_acc": (
                best_val_acc
            ),
            "best_val_loss": (
                best_val_loss
            ),
            "best_epoch": best_epoch,
            "cpu_rng_state": (
                torch.get_rng_state()
            ),
            "cuda_rng_state": (
                torch.cuda.get_rng_state()
                if device.type == "cuda"
                else None
            )
        }
        save_atomic(
            cp,
            last_path
        )
        save_history(
            hist,
            out / "training_history.csv"
        )
        print(
            f"Epoch {epoch}/{a.epochs} "
            f"train={tr['accuracy']:.2%} "
            f"val={va['accuracy']:.2%} "
            f"best={best_val_acc:.2%}@{best_epoch} "
            f"sec={elapsed:.1f} "
            f"peak={peak:.2f}GB",
            flush=True
        )
    # ========================================================
    # Final evaluation
    # ========================================================
    if not best_path.is_file():
        raise RuntimeError(
            f"Missing best checkpoint: {best_path}"
        )
    best = torch.load(
        best_path,
        map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(
        best["model_state_dict"]
    )
    if a.stage == "full":
        print(
            "[FINAL] Evaluating official "
            "10,000-example MNIST test set.",
            flush=True
        )
        metrics = run_epoch(
            model,
            test,
            device,
            a.threshold,
            a.input_pixel_scale,
            a.loss
        )
        metric_key = "test"
        metric_file = "test_metrics.json"
    else:
        print(
            "[FINAL] Validation-only evaluation. "
            "Official test set remains untouched.",
            flush=True
        )
        metrics = run_epoch(
            model,
            valid,
            device,
            a.threshold,
            a.input_pixel_scale,
            a.loss
        )
        metric_key = "validation"
        metric_file = "validation_metrics.json"
    payload = {
        "config": cfg,
        "best_epoch": best_epoch,
        "best_validation_accuracy": (
            best_val_acc
        ),
        metric_key: json_metrics(
            metrics
        ),
        "elapsed_seconds": (
            time.perf_counter()
            - start_time
        ),
        "notes": [
            "Official test set is evaluated "
            "only after a FULL run.",
            "Intensity integrals are not "
            "calibrated Poynting flux.",
            "FFT and 80mm active-window "
            "efficiencies are recorded separately.",
            "Paper geometry, input encoding "
            "and exact loss are provisional."
        ]
    }
    with (
        out / metric_file
    ).open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            payload,
            f,
            indent=2,
            ensure_ascii=False
        )
    # Save figures
    final_plots(
        hist,
        metrics["confusion"].numpy(),
        out,
        model
    )
    print(
        f"[DONE] {metric_key} "
        f"accuracy={metrics['accuracy']:.2%}; "
        f"correct detector="
        f"{metrics['correct_efficiency']:.4%}; "
        f"all detectors="
        f"{metrics['detector_capture']:.4%}",
        flush=True
    )
    print(
        f"Best epoch={best_epoch}",
        flush=True
    )
    print(
        f"Results saved in: {out}",
        flush=True
    )
if __name__ == "__main__":
    main()
