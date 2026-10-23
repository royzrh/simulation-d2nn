# -*- coding: utf-8 -*-

"""
Step 7.6
D2NN MNIST Optical Geometry Validation

Purpose
-------
1. Validate MNIST physical input size.
2. Validate input-plane to first-layer distance.
3. Compare validation accuracy and optical efficiency.
4. Preserve paper-alignment information.
5. Use a fixed MNIST 55k/5k master split.
6. Save checkpoints and resume interrupted experiments.

Hardware:
    Windows 11
    NVIDIA RTX 4060 Laptop GPU
    Python 3.12
    PyTorch CUDA

This script imports the Step 7.4 optical model.

No official MNIST test samples are used.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import random
import time

from pathlib import Path

import numpy as np
import torch

from torch.utils.data import Subset, DataLoader
from torchvision import datasets, transforms

from step7_4_full_gpu_training import (
    OpticalD2NN,
    make_loader,
    run_epoch,
    save_atomic,
    save_history,
    final_plots,
)


# ============================================================
# 1. Global configuration
# ============================================================

HERE = Path(__file__).resolve().parent

DATA_ROOT = HERE.parent / "data"

RESULTS_ROOT = (
    HERE / "results" / "step7_6"
)

SEED = 42

OVERSAMPLE = 4
PAD_FACTOR = 2

BATCH_SIZE = 8

EPOCHS = 10

LEARNING_RATE = 0.005

TRAIN_COUNT = 5000
VAL_COUNT = 1000

THRESHOLD = 0.5

LOSS_METHOD = "baseline"

LOG_EVERY = 200

# Physical neuron pitch: 0.4 mm
PITCH_MM = 0.4

# Original MNIST image: 28 x 28
MNIST_PIXELS = 28

# Fixed 60k MNIST master split
MASTER_TRAIN_COUNT = 55000
MASTER_VAL_COUNT = 5000


# ============================================================
# 2. Geometry candidates
# ============================================================

GEOMETRY_CONFIGS = [
    {
        "name": "A_original",
        "scale": 1,
        "gap_mm": 0.0,
    },
    {
        "name": "B_input_propagation",
        "scale": 1,
        "gap_mm": 30.0,
    },
    {
        "name": "C_larger_input",
        "scale": 2,
        "gap_mm": 0.0,
    },
    {
        "name": "D_larger_input_propagation",
        "scale": 3,
        "gap_mm": 30.0,
    },
]


# ============================================================
# 3. Reproducibility
# ============================================================

def set_seed(seed: int):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False


# ============================================================
# 4. Fixed MNIST data split
# ============================================================

def prepare_dataset():

    train_dataset = datasets.MNIST(
        root=str(DATA_ROOT),
        train=True,
        download=False,
        transform=transforms.ToTensor(),
    )

    if len(train_dataset) != 60000:
        raise RuntimeError(
            "Expected 60000 MNIST training images"
        )

    generator = torch.Generator().manual_seed(SEED)

    permutation = torch.randperm(
        len(train_dataset),
        generator=generator
    )

    # Fixed master split
    master_train = permutation[:55000]
    master_val = permutation[55000:60000]

    # Nested subsets
    train_indices = master_train[:TRAIN_COUNT].tolist()
    val_indices = master_val[:VAL_COUNT].tolist()

    assert len(train_indices) == TRAIN_COUNT
    assert len(val_indices) == VAL_COUNT

    assert set(train_indices).isdisjoint(
        set(val_indices)
    )

    split_id = hashlib.sha256(
        permutation.numpy().tobytes()
    ).hexdigest()[:16]

    train_hash = hashlib.sha256(
        np.asarray(
            train_indices,
            dtype=np.int64
        ).tobytes()
    ).hexdigest()[:16]

    val_hash = hashlib.sha256(
        np.asarray(
            val_indices,
            dtype=np.int64
        ).tobytes()
    ).hexdigest()[:16]

    print("MNIST master split:", split_id)
    print("Training subset hash:", train_hash)
    print("Validation subset hash:", val_hash)

    return (
        train_dataset,
        train_indices,
        val_indices,
        split_id,
        train_hash,
        val_hash,
    )


# ============================================================
# 5. Paper-alignment audit
# ============================================================

def paper_alignment(scale, gap_mm):

    width_mm = (
        MNIST_PIXELS
        * PITCH_MM
        * scale
    )

    return {
        "paper": (
            "Lin et al., Science 2018, "
            "All-optical machine learning using "
            "diffractive deep neural networks"
        ),

        "confirmed_parameters": {
            "frequency_THz": 0.4,
            "layers": 5,
            "neurons_per_layer": [200, 200],
            "neuron_pitch_um": 400,
            "aperture_mm": [80, 80],
            "interlayer_gap_mm": 30,
            "last_layer_to_output_mm": 30,
            "phase_range": [0, "pi"],
            "propagation_method": "ASM",
            "detector_count": 10,
            "detector_side_mm": 4.8,
            "optimizer": "Adam",
            "paper_batch_size": 8,
            "paper_epochs": 10,
        },

        "current_geometry": {
            "mnist_input_width_mm": width_mm,
            "mnist_input_pixel_scale": scale,
            "input_to_first_layer_mm": gap_mm,
        },

        "geometry_alignment": {
            "input_width_status": (
                "UNVERIFIED: Exact digit width "
                "not established from paper"
            ),
            "input_gap_status": (
                "UNVERIFIED: Exact five-layer "
                "input-to-first-layer distance "
                "not established"
            ),
            "interpretation": (
                "Geometry screening candidate, "
                "not confirmed original-paper geometry"
            ),
        },

        "additional_unverified_settings": {
            "detector_center_coordinates": True,
            "exact_classifier_loss": True,
            "binarization_threshold": True,
            "learning_rate": True,
            "phase_initialization": True,
            "fft_padding": True,
            "sampling_oversample": True,
        },

        "training_note": (
            "Uses 5000/1000 MNIST subsets for "
            "geometry selection. Accuracy cannot "
            "be directly compared to the paper's "
            "10000-image test accuracy."
        ),
    }


# ============================================================
# 6. Save JSON helper
# ============================================================

def write_json(path, data):

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# 7. Model and DataLoader initialization
# ============================================================

def create_model(scale, gap_mm, device):

    model = OpticalD2NN(
        over=OVERSAMPLE,
        pad_factor=PAD_FACTOR,
        input_gap_mm=gap_mm,
    )

    return model.to(device)


def create_validation_loader(dataset, indices, device):

    return make_loader(
        ds=dataset,
        indices=indices,
        batch_size=BATCH_SIZE,
        workers=0,
        shuffle=False,
        generator_seed=0,
        pin=(device.type == "cuda"),
    )


def create_train_loader(
    dataset,
    indices,
    device,
    epoch
):

    # Same shuffle seed per epoch across all geometries.
    return make_loader(
        ds=dataset,
        indices=indices,
        batch_size=BATCH_SIZE,
        workers=0,
        shuffle=True,
        generator_seed=SEED + 1000 + epoch,
        pin=(device.type == "cuda"),
    )


# ============================================================
# 8. Geometry experiment
# ============================================================

def run_geometry(
    config,
    dataset,
    train_indices,
    val_indices,
    split_id,
    train_hash,
    val_hash,
    device
):

    name = config["name"]
    scale = int(config["scale"])
    gap_mm = float(config["gap_mm"])

    width_mm = (
        MNIST_PIXELS * PITCH_MM * scale
    )

    out = RESULTS_ROOT / name
    out.mkdir(parents=True, exist_ok=True)

    metrics_path = out / "validation_metrics.json"
    best_path = out / "best_model.pt"
    last_path = out / "last_checkpoint.pt"
    history_path = out / "training_history.csv"

    experiment_config = {
        "name": name,
        "scale": scale,
        "width_mm": width_mm,
        "gap_mm": gap_mm,
        "seed": SEED,
        "oversample": OVERSAMPLE,
        "pad_factor": PAD_FACTOR,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "train_count": TRAIN_COUNT,
        "val_count": VAL_COUNT,
        "threshold": THRESHOLD,
        "loss": LOSS_METHOD,
        "split_id": split_id,
        "train_hash": train_hash,
        "val_hash": val_hash,
        "geometry_version": "fixed_master_split_v1",
    }

    config_path = out / "config.json"

    # Prevent accidental reuse of incompatible results.
    if config_path.exists():

        old_config = json.loads(
            config_path.read_text(encoding="utf-8")
        )

        if old_config != experiment_config:
            raise RuntimeError(
                f"Existing experiment config differs: {out}"
            )

    else:
        write_json(
            config_path,
            experiment_config
        )

    write_json(
        out / "paper_alignment.json",
        paper_alignment(scale, gap_mm)
    )

    # Skip completed and compatible experiment.
    if metrics_path.is_file():

        report = json.loads(
            metrics_path.read_text(encoding="utf-8")
        )

        print(f"[SKIP] Completed: {name}", flush=True)

        return report["summary"]

    print()
    print("=" * 72)
    print("GEOMETRY EXPERIMENT:", name)
    print("=" * 72)

    print(f"Input width: {width_mm:.1f} mm")
    print(f"Input gap: {gap_mm:.1f} mm")
    print(f"OS: {OVERSAMPLE}")
    print(f"Batch: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Device: {device}")
    print(f"Results: {out}")
    print("=" * 72, flush=True)

    # Ensure identical initial random phase parameters.
    set_seed(SEED)

    model = create_model(
        scale,
        gap_mm,
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    val_loader = create_validation_loader(
        dataset,
        val_indices,
        device
    )

    history = []

    best_acc = -1.0
    best_loss = float("inf")
    best_epoch = 0
    start_epoch = 1

    # ========================================================
    # Resume
    # ========================================================

    if last_path.exists():

        print("[RESUME] Loading checkpoint...", flush=True)

        checkpoint = torch.load(
            last_path,
            map_location="cpu",
            weights_only=True
        )

        if checkpoint["config"] != experiment_config:
            raise RuntimeError(
                "Checkpoint configuration mismatch"
            )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        history = checkpoint["history"]

        best_acc = checkpoint["best_acc"]
        best_loss = checkpoint["best_loss"]
        best_epoch = checkpoint["best_epoch"]

        start_epoch = checkpoint["epoch"] + 1

        if "cpu_rng_state" in checkpoint:
            torch.set_rng_state(
                checkpoint["cpu_rng_state"]
            )

        if (
            device.type == "cuda"
            and checkpoint.get("cuda_rng_states") is not None
        ):
            torch.cuda.set_rng_state_all(
                checkpoint["cuda_rng_states"]
            )

        print(
            f"[RESUME] Start epoch: {start_epoch}",
            flush=True
        )

    # ========================================================
    # Train
    # ========================================================

    for epoch in range(start_epoch, EPOCHS + 1):

        epoch_start = time.perf_counter()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        train_loader = create_train_loader(
            dataset,
            train_indices,
            device,
            epoch
        )

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            threshold=THRESHOLD,
            input_scale=scale,
            method=LOSS_METHOD,
            optimizer=optimizer,
            log_every=LOG_EVERY,
            epoch=epoch,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            threshold=THRESHOLD,
            input_scale=scale,
            method=LOSS_METHOD,
        )

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = (
            time.perf_counter() - epoch_start
        )

        peak_gb = (
            torch.cuda.max_memory_allocated()
            / 1024 ** 3
            if device.type == "cuda"
            else 0.0
        )

        # Training history.
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_accuracy": val_metrics["accuracy"],
            "train_correct_efficiency": (
                train_metrics["correct_efficiency"]
            ),
            "val_correct_efficiency": (
                val_metrics["correct_efficiency"]
            ),
            "train_detector_capture": (
                train_metrics["detector_capture"]
            ),
            "val_detector_capture": (
                val_metrics["detector_capture"]
            ),
            "seconds": elapsed,
            "cuda_peak_allocated_gb": peak_gb,
        }

        history.append(row)

        # Select best model using validation only.
        improved = (
            val_metrics["accuracy"] > best_acc
            or (
                val_metrics["accuracy"] == best_acc
                and val_metrics["loss"] < best_loss
            )
        )

        if improved:

            best_acc = val_metrics["accuracy"]
            best_loss = val_metrics["loss"]
            best_epoch = epoch

            save_atomic(
                {
                    "model_state_dict": model.state_dict(),
                    "config": experiment_config,
                    "epoch": epoch,
                    "validation_accuracy": best_acc,
                },
                best_path
            )

        # Full resume checkpoint.
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "best_acc": best_acc,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "config": experiment_config,
            "cpu_rng_state": torch.get_rng_state(),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all()
                if device.type == "cuda"
                else None
            ),
        }

        save_atomic(
            checkpoint,
            last_path
        )

        save_history(
            history,
            history_path
        )

        print(
            f"[{name}] "
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train={train_metrics['accuracy']:.2%} | "
            f"Val={val_metrics['accuracy']:.2%} | "
            f"Best={best_acc:.2%}@{best_epoch} | "
            f"Time={elapsed:.1f}s | "
            f"VRAM={peak_gb:.2f}GB",
            flush=True
        )

    # ========================================================
    # Final validation using best model
    # ========================================================

    if not best_path.exists():
        raise RuntimeError(
            f"Missing best model: {best_path}"
        )

    best = torch.load(
        best_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(
        best["model_state_dict"]
    )

    final_val = run_epoch(
        model=model,
        loader=val_loader,
        device=device,
        threshold=THRESHOLD,
        input_scale=scale,
        method=LOSS_METHOD,
    )

    # Record only validation results.
    summary = {
        "name": name,
        "scale": scale,
        "input_width_mm": width_mm,
        "input_gap_mm": gap_mm,
        "best_epoch": best_epoch,
        "val_accuracy": float(
            final_val["accuracy"]
        ),
        "correct_efficiency": float(
            final_val["correct_efficiency"]
        ),
        "detector_capture": float(
            final_val["detector_capture"]
        ),
        "correct_share_of_detectors": float(
            final_val["correct_share_of_detectors"]
        ),
        "best_val_loss": float(
            final_val["loss"]
        ),
        "total_epoch_seconds": float(
            sum(h["seconds"] for h in history)
        ),
        "result_directory": str(out),
    }

    report = {
        "experiment_config": experiment_config,
        "summary": summary,
        "official_test_used": False,
        "paper_geometry_verified": False,
    }

    # Save figures before marking run complete.
    final_plots(
        history,
        final_val["confusion"].numpy(),
        out,
        model
    )

    write_json(
        metrics_path,
        report
    )

    print()
    print(f"[DONE] {name}")
    print(
        f"Validation accuracy: "
        f"{summary['val_accuracy']:.2%}"
    )
    print(
        f"Correct detector efficiency: "
        f"{summary['correct_efficiency']:.4%}"
    )
    print(
        f"Detector capture: "
        f"{summary['detector_capture']:.4%}"
    )
    print(
        f"Best epoch: {best_epoch}"
    )
    print()

    del model
    del optimizer
    del train_loader
    del val_loader

    gc.collect()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return summary


# ============================================================
# 9. Summary CSV
# ============================================================

def save_summary(rows):

    output = (
        RESULTS_ROOT
        / "geometry_validation.csv"
    )

    if not rows:
        return

    fieldnames = [
        "name",
        "scale",
        "input_width_mm",
        "input_gap_mm",
        "best_epoch",
        "val_accuracy",
        "correct_efficiency",
        "detector_capture",
        "correct_share_of_detectors",
        "best_val_loss",
        "total_epoch_seconds",
        "result_directory",
    ]

    with output.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)

    print(
        "[SUMMARY UPDATED]",
        output,
        flush=True
    )


# ============================================================
# 10. Main
# ============================================================

def main():

    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    set_seed(SEED)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    if device.type != "cuda":
        raise RuntimeError(
            "CUDA not available. "
            "Check NVIDIA driver and PyTorch."
        )

    print("=" * 72)
    print("STEP 7.6 - D2NN OPTICAL GEOMETRY VALIDATION")
    print("=" * 72)

    print("PyTorch:", torch.__version__)
    print("Device:", torch.cuda.get_device_name(0))
    print("GPU memory:", round(
        torch.cuda.get_device_properties(0).total_memory
        / 1024 ** 3,
        2
    ), "GB")

    print("Train:", TRAIN_COUNT)
    print("Validation:", VAL_COUNT)
    print("Epochs:", EPOCHS)
    print("Oversample:", OVERSAMPLE)
    print("Batch size:", BATCH_SIZE)

    print("=" * 72)
    print(
        "NOTE: These are candidate geometries, "
        "not confirmed original-paper geometries."
    )
    print("=" * 72)

    (
        dataset,
        train_indices,
        val_indices,
        split_id,
        train_hash,
        val_hash,
    ) = prepare_dataset()

    results = []

    started = time.perf_counter()

    for config in GEOMETRY_CONFIGS:

        result = run_geometry(
            config=config,
            dataset=dataset,
            train_indices=train_indices,
            val_indices=val_indices,
            split_id=split_id,
            train_hash=train_hash,
            val_hash=val_hash,
            device=device
        )

        results.append(result)

        # Save after every experiment.
        save_summary(results)

    print()
    print("=" * 72)
    print("FINAL GEOMETRY COMPARISON")
    print("=" * 72)

    ranked = sorted(
        results,
        key=lambda x: x["val_accuracy"],
        reverse=True
    )

    for rank, r in enumerate(ranked, start=1):

        print(
            f"{rank}. {r['name']} | "
            f"Width={r['input_width_mm']:.1f} mm | "
            f"Gap={r['input_gap_mm']:.0f} mm | "
            f"Acc={r['val_accuracy']:.2%} | "
            f"CorrectEff="
            f"{r['correct_efficiency']:.4%} | "
            f"BestEpoch={r['best_epoch']}"
        )

    total_seconds = (
        time.perf_counter() - started
    )

    print()
    print(
        "Total wall time:",
        round(total_seconds / 3600, 3),
        "hours"
    )

    print(
        "Summary:",
        RESULTS_ROOT / "geometry_validation.csv"
    )

    print()
    print("STEP 7.6 COMPLETED")


if __name__ == "__main__":
    main()