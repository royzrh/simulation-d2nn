
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fashion-MNIST Phase-Input D2NN Reproduction

Based on:
    step7_4_full_gpu_training.py

Features:
- Five-layer phase-only D2NN
- Fashion-MNIST phase encoding (0 to 2pi)
- Angular spectrum propagation
- Independent training and result directories
- Smoke / Pilot / Full training
- Validation-based best model selection
- Checkpoint resume
- Optical efficiency evaluation
- Training curves and confusion matrix

Paper alignment:
Lin et al., Science 2018

Some geometry, detector positions and exact loss
parameters still require verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time

from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from torch.nn import functional as F
from torchvision import datasets, transforms

from step7_4_full_gpu_training import (
    OpticalD2NN,
    DETECTOR_CENTERS,
    DETECTOR_SIDE,
    EPS,
    make_loader,
    run_epoch,
    save_atomic,
    save_history,
    final_plots,
)


# ============================================================
# 1. Command-line arguments
# ============================================================

def parse_args():

    p = argparse.ArgumentParser(
        description="Fashion-MNIST Phase D2NN"
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

    p.add_argument("--oversample", type=int, default=4)
    p.add_argument("--pad-factor", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--input-pixel-scale",
        type=int,
        default=1
    )

    p.add_argument(
        "--input-gap-mm",
        type=float,
        default=0.0
    )

    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--log-every", type=int, default=200)

    p.add_argument("--train-count", type=int, default=None)
    p.add_argument("--val-count", type=int, default=None)
    p.add_argument("--test-count", type=int, default=None)

    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--results-root", type=Path, default=None)

    p.add_argument(
        "--resume",
        type=Path,
        default=None
    )

    a = p.parse_args()

    if a.epochs is None:
        a.epochs = {
            "smoke": 1,
            "pilot": 3,
            "full": 10
        }[a.stage]

    vals = [
        a.oversample,
        a.pad_factor,
        a.batch_size,
        a.epochs,
        a.input_pixel_scale
    ]

    if (
        min(vals) < 1
        or a.pad_factor < 2
        or a.lr <= 0
        or a.input_gap_mm < 0
    ):
        p.error("Invalid parameters")

    if a.input_pixel_scale * 28 > 200:
        p.error("Input exceeds 200x200 aperture")

    if a.num_workers < 0 or a.log_every < 0:
        p.error("Invalid worker/log settings")

    return a


# ============================================================
# 2. Fashion-MNIST optical model
# ============================================================

class FashionOpticalD2NN(OpticalD2NN):

    """
    Reuse the original five-layer optical network.

    Replace MNIST amplitude encoding with
    Fashion-MNIST phase encoding.

    Grayscale 0..1 -> optical phase 0..2pi.

    Inside the image footprint:
        constant amplitude, variable phase

    Outside the image footprint:
        zero transmission
    """

    def encode(
        self,
        imgs: torch.Tensor,
        threshold: float,
        scale: int
    ) -> torch.Tensor:

        # Fashion-MNIST must preserve grayscale.
        # Threshold is unused.
        del threshold

        phase = (
            imgs[:, 0]
            .float()
            .clamp(0.0, 1.0)
            * (2.0 * math.pi)
        )

        repeat = self.over * scale

        phase = phase.repeat_interleave(
            repeat,
            dim=-2
        ).repeat_interleave(
            repeat,
            dim=-1
        )

        amplitude = torch.ones_like(phase)

        side = phase.shape[-1]

        if side > self.active_n:
            raise ValueError(
                "Input image exceeds optical aperture"
            )

        before = (self.active_n - side) // 2
        after = self.active_n - side - before

        padding = (
            before,
            after,
            before,
            after
        )

        phase = F.pad(phase, padding)
        amplitude = F.pad(amplitude, padding)

        fft_padding = (
            self.left,
            self.right,
            self.left,
            self.right
        )

        phase = F.pad(phase, fft_padding)
        amplitude = F.pad(amplitude, fft_padding)

        # Complex input optical field.
        field = torch.polar(
            amplitude,
            phase
        )

        # Normalize input optical power.
        power = (
            field.abs()
            .square()
            .sum((-2, -1), keepdim=True)
            .clamp_min(EPS)
        )

        return field / power.sqrt()


# ============================================================
# 3. Result serialization
# ============================================================

def clean_metrics(metrics):

    result = {}

    for key, value in metrics.items():

        if isinstance(value, torch.Tensor):
            result[key] = (
                value.detach().cpu().tolist()
            )

        elif isinstance(
            value,
            (np.integer, np.floating)
        ):
            result[key] = value.item()

        else:
            result[key] = value

    return result


def write_json(path: Path, obj):

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    temp.write_text(
        json.dumps(
            obj,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    temp.replace(path)


# ============================================================
# 4. Main training program
# ============================================================

def main():

    a = parse_args()

    base = Path(__file__).resolve().parent

    data_root = (
        a.data_root or base.parent / "data"
    ).expanduser().resolve()

    results_root = (
        a.results_root
        or base / "results" / "fashion_mnist"
    ).expanduser().resolve()

    results_root.mkdir(
        parents=True,
        exist_ok=True
    )

    # Reproducibility
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)

    if (
        a.device == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA requested but unavailable"
        )

    device = torch.device(
        "cuda"
        if (
            a.device == "cuda"
            or (
                a.device == "auto"
                and torch.cuda.is_available()
            )
        )
        else "cpu"
    )

    # Dataset sizes
    sizes = {
        "smoke": (40, 20, 20),
        "pilot": (1000, 200, 1000),
        "full": (55000, 5000, 10000)
    }

    tn, vn, xn = sizes[a.stage]

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

    if (
        min(tn, vn, xn) <= 0
        or tn + vn > 60000
        or xn > 10000
    ):
        raise ValueError(
            "Invalid dataset subset sizes"
        )

    # Load Fashion-MNIST
    train_ds = datasets.FashionMNIST(
        root=str(data_root),
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    test_ds = datasets.FashionMNIST(
        root=str(data_root),
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )

    if (
        len(train_ds) != 60000
        or len(test_ds) != 10000
    ):
        raise RuntimeError(
            "Unexpected Fashion-MNIST dataset size"
        )

    # Deterministic train/validation split
    permutation = torch.randperm(
        60000,
        generator=(
            torch.Generator().manual_seed(a.seed)
        )
    )

    train_indices = permutation[:tn].tolist()

    val_indices = permutation[
        tn:tn + vn
    ].tolist()

    test_indices = list(range(xn))

    split_id = hashlib.sha256(
        permutation.numpy().tobytes()
    ).hexdigest()[:16]

    # Experiment configuration
    config = {
        "dataset": "FashionMNIST",
        "input_encoding": "phase_0_to_2pi",
        "image_support": "constant_amplitude_inside_zero_outside",
        "stage": a.stage,
        "loss": a.loss,
        "oversample": a.oversample,
        "pad_factor": a.pad_factor,
        "batch_size": a.batch_size,
        "epochs": a.epochs,
        "lr": a.lr,
        "seed": a.seed,
        "input_pixel_scale": a.input_pixel_scale,
        "input_gap_mm": a.input_gap_mm,
        "train_count": tn,
        "val_count": vn,
        "test_count": xn,
        "split_id": split_id,
        "data_root": str(data_root),
        "detector_centers": DETECTOR_CENTERS,
        "detector_side_m": DETECTOR_SIDE,
        "geometry": "step7_4_full_fft",
        "paper_alignment": "partial_unverified"
    }

    # Independent experiment directory
    if a.resume:

        cp_path = (
            a.resume.expanduser().resolve()
        )

        if (
            not cp_path.is_file()
            or cp_path.name != "last_checkpoint.pt"
        ):
            raise FileNotFoundError(
                "Expected Fashion last_checkpoint.pt"
            )

        out = cp_path.parent

        old = torch.load(
            cp_path,
            map_location="cpu",
            weights_only=True
        )

        if old["config"] != config:

            differing = [
                k for k, v in config.items()
                if old["config"].get(k) != v
            ]

            raise ValueError(
                f"Resume configuration mismatch: {differing}"
            )

    else:

        geometry_label = (
            f"s{a.input_pixel_scale}_"
            f"z{a.input_gap_mm:g}"
        )

        out = results_root / (
            f"{a.stage}_{a.loss}_"
            f"{geometry_label}_"
            f"os{a.oversample}_"
            f"seed{a.seed}_"
            f"{datetime.now():%Y%m%d_%H%M%S_%f}"
        )

        out.mkdir(
            parents=True,
            exist_ok=False
        )

        write_json(
            out / "config.json",
            config
        )

        old = None

    best_path = out / "best_model.pt"
    last_path = out / "last_checkpoint.pt"

    pin = device.type == "cuda"

    # DataLoaders
    validation = make_loader(
        train_ds,
        val_indices,
        a.batch_size,
        a.num_workers,
        False,
        0,
        pin
    )

    test_loader = (
        make_loader(
            test_ds,
            test_indices,
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
    model = FashionOpticalD2NN(
        a.oversample,
        a.pad_factor,
        a.input_gap_mm
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=a.lr
    )

    history = []

    best_acc = -1.0
    best_loss = float("inf")
    best_epoch = 0
    start_epoch = 1

    # Resume
    if old is not None:

        model.load_state_dict(
            old["model_state_dict"]
        )

        optimizer.load_state_dict(
            old["optimizer_state_dict"]
        )

        history = old["history"]

        best_acc = old["best_val_acc"]
        best_loss = old["best_val_loss"]
        best_epoch = old["best_epoch"]
        start_epoch = old["epoch"] + 1

        torch.set_rng_state(
            old["cpu_rng_state"]
        )

        if (
            device.type == "cuda"
            and old.get("cuda_rng_state") is not None
        ):
            torch.cuda.set_rng_state(
                old["cuda_rng_state"]
            )

        del old

    # Training information
    print("=" * 72, flush=True)

    print(
        "Fashion-MNIST PHASE-input D2NN",
        a.stage,
        a.loss,
        flush=True
    )

    print(
        f"Device={device}, "
        f"FFT grid={model.fft_n}x{model.fft_n}, "
        f"OS={a.oversample}",
        flush=True
    )

    print(
        f"Samples train/val/test={tn}/{vn}/{xn}, "
        f"epochs={a.epochs}, "
        f"batch={a.batch_size}",
        flush=True
    )

    print(
        f"Input phase=[0,2pi], "
        f"scale={a.input_pixel_scale}, "
        f"gap={a.input_gap_mm} mm",
        flush=True
    )

    print(
        "Results:",
        out,
        flush=True
    )

    print(
        "WARNING: Some paper parameters "
        "remain unverified.",
        flush=True
    )

    if device.type == "cuda":

        free, total = torch.cuda.mem_get_info()

        print(
            f"GPU free/total: "
            f"{free / 2**30:.2f}/"
            f"{total / 2**30:.2f} GiB",
            flush=True
        )

    print("=" * 72, flush=True)

    # ========================================================
    # 5. Training loop
    # ========================================================

    for epoch in range(
        start_epoch,
        a.epochs + 1
    ):

        t0 = time.perf_counter()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        train_loader = make_loader(
            train_ds,
            train_indices,
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
            0.0,
            a.input_pixel_scale,
            a.loss,
            optimizer=optimizer,
            log_every=a.log_every,
            epoch=epoch
        )

        va = run_epoch(
            model,
            validation,
            device,
            0.0,
            a.input_pixel_scale,
            a.loss
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        peak_gb = (
            torch.cuda.max_memory_allocated() / 2**30
            if device.type == "cuda"
            else 0.0
        )

        row = {
            "epoch": epoch,
            "seconds": time.perf_counter() - t0,
            "cuda_peak_allocated_gb": peak_gb
        }

        row.update({
            "train_" + k: v
            for k, v in tr.items()
            if k not in {"confusion", "n"}
        })

        row.update({
            "val_" + k: v
            for k, v in va.items()
            if k not in {"confusion", "n"}
        })

        history.append(row)

        # Best model chosen by validation only
        if (
            va["accuracy"] > best_acc
            or (
                va["accuracy"] == best_acc
                and va["loss"] < best_loss
            )
        ):

            best_acc = va["accuracy"]
            best_loss = va["loss"]
            best_epoch = epoch

            save_atomic(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "epoch": epoch,
                    "validation_accuracy": best_acc
                },
                best_path
            )

        # Full checkpoint
        save_atomic(
            {
                "epoch": epoch,
                "config": config,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "best_val_acc": best_acc,
                "best_val_loss": best_loss,
                "best_epoch": best_epoch,
                "cpu_rng_state": torch.get_rng_state(),
                "cuda_rng_state": (
                    torch.cuda.get_rng_state()
                    if device.type == "cuda"
                    else None
                )
            },
            last_path
        )

        save_history(
            history,
            out / "training_history.csv"
        )

        print(
            f"Epoch {epoch}/{a.epochs} "
            f"train={tr['accuracy']:.2%} "
            f"val={va['accuracy']:.2%} "
            f"best={best_acc:.2%}@{best_epoch} "
            f"time={row['seconds']:.1f}s "
            f"GPUpeak={peak_gb:.2f}GiB",
            flush=True
        )

    # ========================================================
    # 6. Final evaluation
    # ========================================================

    best = torch.load(
        best_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(
        best["model_state_dict"]
    )

    final_validation = run_epoch(
        model,
        validation,
        device,
        0.0,
        a.input_pixel_scale,
        a.loss
    )

    summary = {
        "paper_comparison_accuracy": 0.8113,
        "paper_comparison_valid_only_for_matching_setup": True,
        "method_warning": (
            "Exact paper loss and some geometry/"
            "detector positions not verified"
        ),
        "best_epoch": best_epoch,
        "validation": clean_metrics(final_validation),
        "official_test_used": a.stage == "full"
    }

    if test_loader is not None:

        official_test = run_epoch(
            model,
            test_loader,
            device,
            0.0,
            a.input_pixel_scale,
            a.loss
        )

        summary["official_test"] = clean_metrics(
            official_test
        )

        write_json(
            out / "test_metrics.json",
            summary["official_test"]
        )

        figure_cm = (
            official_test["confusion"].cpu().numpy()
        )

        print(
            f"Official Fashion test accuracy: "
            f"{official_test['accuracy']:.2%}",
            flush=True
        )

    else:

        figure_cm = (
            final_validation["confusion"].cpu().numpy()
        )

    # Save figures
    final_plots(
        history,
        figure_cm,
        out,
        model
    )

    write_json(
        out / "summary.json",
        summary
    )

    print(
        "FINISHED. Results:",
        out,
        flush=True
    )


if __name__ == "__main__":
    main()
