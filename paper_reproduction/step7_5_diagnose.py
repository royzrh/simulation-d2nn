#!/usr/bin/env python3
# Step 7.5: paired validation diagnostics.

import argparse
import csv
import json
import math

from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from step7_4_full_gpu_training import (
    OpticalD2NN,
    DETECTOR_CENTERS,
    DETECTOR_SIDE,
)

ROOT = Path(__file__).resolve().parent


def find_checkpoint(loss, supplied):

    if supplied:
        path = Path(supplied).expanduser().resolve()

        if not path.is_file():
            raise FileNotFoundError(path)

        return path

    candidates = list(
        (ROOT / "results" / "step7_4").glob(
            f"full_{loss}_os4_seed42_*/best_model.pt"
        )
    )

    if not candidates:
        raise FileNotFoundError(
            f"Cannot find {loss} best_model.pt"
        )

    return max(
        candidates,
        key=lambda p: p.stat().st_mtime
    )


def load_model(path, device):

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    cfg = checkpoint["config"]

    model = OpticalD2NN(
        int(cfg["oversample"]),
        int(cfg["pad_factor"]),
        float(cfg["input_gap_mm"])
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model = model.to(device).eval()

    return model, cfg


@torch.inference_mode()
def evaluate_model(model, cfg, loader, device, name):

    labels_all = []
    predictions_all = []
    powers_all = []

    for step, (images, labels) in enumerate(
        loader, start=1
    ):

        field = model.encode(
            images.to(device),
            float(cfg["threshold"]),
            int(cfg["input_pixel_scale"])
        )

        _, powers = model(field)

        predictions_all.append(
            powers.argmax(1).cpu().numpy()
        )

        labels_all.append(labels.numpy())

        powers_all.append(
            powers.cpu().numpy()
        )

        if step == 1 or step % 100 == 0:

            print(
                f"[{name}] "
                f"Batch {step}/{len(loader)}",
                flush=True
            )

    return (
        np.concatenate(labels_all),
        np.concatenate(predictions_all),
        np.concatenate(powers_all)
    )


def calculate_metrics(labels, predictions, powers):

    n = len(labels)

    correct_power = powers[
        np.arange(n),
        labels
    ]

    total_detector_power = powers.sum(axis=1)

    normalized = powers / np.maximum(
        total_detector_power[:, None],
        1e-12
    )

    wrong_power = normalized.copy()

    wrong_power[
        np.arange(n),
        labels
    ] = -np.inf

    margin = (
        normalized[np.arange(n), labels]
        - wrong_power.max(axis=1)
    )

    confusion = np.zeros(
        (10, 10),
        dtype=np.int64
    )

    np.add.at(
        confusion,
        (labels, predictions),
        1
    )

    response = np.stack([
        normalized[labels == digit].mean(axis=0)
        if np.any(labels == digit)
        else np.zeros(10)
        for digit in range(10)
    ])

    return {
        "correct": predictions == labels,
        "correct_eff": correct_power,
        "detector_eff": total_detector_power,
        "normalized_power": normalized,
        "margin": margin,
        "confusion": confusion,
        "response": response,
    }


def create_figures(
    baseline,
    joint,
    labels,
    models,
    cfg,
    dataset,
    original_indices,
    output,
    device
):

    # 1. Detector response matrices

    fig, axes = plt.subplots(
        1, 2, figsize=(11, 4.8)
    )

    maximum = max(
        baseline["response"].max(),
        joint["response"].max()
    )

    for ax, data, title in zip(
        axes,
        [baseline, joint],
        ["Baseline", "Joint"]
    ):

        im = ax.imshow(
            data["response"],
            cmap="gray_r",
            vmin=0,
            vmax=maximum
        )

        ax.set(
            title=title,
            xlabel="Detector index",
            ylabel="True digit",
            xticks=range(10),
            yticks=range(10)
        )

    fig.colorbar(
        im,
        ax=list(axes),
        shrink=0.72,
        label="Mean normalized detector power"
    )

    fig.savefig(
        output / "mean_detector_response.png",
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    # 2. Per-class recall

    counts = np.bincount(
        labels,
        minlength=10
    )

    recall_base = (
        np.diag(baseline["confusion"])
        / np.maximum(counts, 1)
    )

    recall_joint = (
        np.diag(joint["confusion"])
        / np.maximum(counts, 1)
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))

    x = np.arange(10)

    ax.bar(
        x - 0.19,
        recall_base * 100,
        width=0.38,
        label="Baseline",
        color="0.3"
    )

    ax.bar(
        x + 0.19,
        recall_joint * 100,
        width=0.38,
        label="Joint",
        color="0.75",
        edgecolor="black"
    )

    ax.set(
        xlabel="True digit",
        ylabel="Recall (%)",
        xticks=range(10),
        ylim=(0, 105)
    )

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        output / "per_class_recall.png",
        dpi=180
    )

    plt.close(fig)

    # 3. Phase comparison

    fig, axes = plt.subplots(
        2, 5, figsize=(16, 6)
    )

    for row, (name, model) in enumerate(models):

        phases = (
            math.pi
            * torch.sigmoid(
                model.raw_phases.detach()
            )
        ).cpu().numpy()

        for layer in range(5):

            ax = axes[row, layer]

            ax.imshow(
                phases[layer],
                cmap="gray",
                vmin=0,
                vmax=math.pi,
                origin="lower",
                extent=(-40, 40, -40, 40)
            )

            ax.set(
                title=f"{name} - Layer {layer + 1}",
                xticks=[],
                yticks=[]
            )

    fig.tight_layout()

    fig.savefig(
        output / "phase_comparison.png",
        dpi=180
    )

    plt.close(fig)

    # 4. Select paired misclassification examples

    base_correct = baseline["correct"]
    joint_correct = joint["correct"]

    base_wins = np.flatnonzero(
        base_correct & ~joint_correct
    )

    joint_wins = np.flatnonzero(
        ~base_correct & joint_correct
    )

    selected = []

    for digit in [2, 7, 8]:

        candidates = base_wins[
            labels[base_wins] == digit
        ]

        if len(candidates):
            selected.append(int(candidates[0]))

    for digit in [4, 9]:

        candidates = joint_wins[
            labels[joint_wins] == digit
        ]

        if len(candidates):
            selected.append(int(candidates[0]))

    if not selected:
        selected = [
            int(i) for i in base_wins[:3]
        ]

    # 5. Visualize optical output intensity

    for number, pos in enumerate(
        selected[:5], start=1
    ):

        image, true_label = dataset[pos]

        fields = []

        for _, model in models:

            with torch.inference_mode():

                optical_input = model.encode(
                    image[None].to(device),
                    float(cfg["threshold"]),
                    int(cfg["input_pixel_scale"])
                )

                intensity, _ = model(
                    optical_input
                )

                active = intensity[
                    :,
                    model.s,
                    model.s
                ].unsqueeze(1)

                # Visualization downsampling only.
                active = F.avg_pool2d(
                    active,
                    kernel_size=int(
                        cfg["oversample"]
                    )
                )

                fields.append(
                    active[0, 0].cpu().numpy()
                )

        fig, axes = plt.subplots(
            1, 3, figsize=(14, 4.5)
        )

        axes[0].imshow(
            image[0].numpy(),
            cmap="gray",
            origin="lower"
        )

        axes[0].set_title(
            f"MNIST #{original_indices[pos]}\n"
            f"True digit: {int(true_label)}"
        )

        axes[0].axis("off")

        vmax = max(
            np.percentile(f, 99.9)
            for f in fields
        )

        vmax = max(vmax, 1e-12)

        for ax, field, title, data in zip(
            axes[1:],
            fields,
            ["Baseline", "Joint"],
            [baseline, joint]
        ):

            # Same intensity scale for both models.
            ax.imshow(
                np.log1p(
                    1000 * field / vmax
                ),
                cmap="gray_r",
                origin="lower",
                extent=(-40, 40, -40, 40),
                vmin=0,
                vmax=np.log1p(1000)
            )

            for digit, (cx, cy) in enumerate(
                DETECTOR_CENTERS
            ):

                side = DETECTOR_SIDE * 1000

                ax.add_patch(
                    Rectangle(
                        (
                            cx * 1000 - side / 2,
                            cy * 1000 - side / 2
                        ),
                        side,
                        side,
                        fill=False,
                        edgecolor="red",
                        linewidth=0.6
                    )
                )

                ax.text(
                    cx * 1000,
                    cy * 1000,
                    str(digit),
                    color="red",
                    fontsize=6,
                    ha="center",
                    va="center"
                )

            predicted = int(
                data["normalized_power"][pos].argmax()
            )

            efficiency = (
                data["correct_eff"][pos] * 100
            )

            ax.set(
                title=(
                    f"{title}: predicted {predicted}\n"
                    f"Correct efficiency: {efficiency:.3f}%"
                ),
                xlabel="x (mm)",
                ylabel="y (mm)"
            )

        fig.tight_layout()

        fig.savefig(
            output
            / f"example_{number}_digit{int(true_label)}.png",
            dpi=180
        )

        plt.close(fig)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--baseline",
        type=str,
        default=None
    )

    parser.add_argument(
        "--joint",
        type=str,
        default=None
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8
    )

    parser.add_argument(
        "--max-samples",
        type=int,
        default=5000
    )

    args = parser.parse_args()

    if not 1 <= args.max_samples <= 5000:
        raise ValueError(
            "max-samples must be between 1 and 5000"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    models = []
    configs = []
    paths = []

    for name, supplied in [
        ("Baseline", args.baseline),
        ("Joint", args.joint)
    ]:

        path = find_checkpoint(
            name.lower(),
            supplied
        )

        model, cfg = load_model(
            path, device
        )

        if cfg["loss"] != name.lower():
            raise ValueError(
                f"{name} checkpoint loss mismatch"
            )

        models.append((name, model))
        configs.append(cfg)
        paths.append(str(path))

        print(
            f"{name}: {path}",
            flush=True
        )

    # Verify matching experimental settings.

    required = [
        "stage",
        "oversample",
        "pad_factor",
        "seed",
        "threshold",
        "input_pixel_scale",
        "input_gap_mm",
        "train_count",
        "val_count",
        "split_id",
        "geometry",
        "detector_centers"
    ]

    for key in required:

        if configs[0][key] != configs[1][key]:

            raise ValueError(
                f"Experiment configuration mismatch: {key}"
            )

    cfg = configs[0]

    if (
        cfg["stage"] != "full"
        or cfg["val_count"] != 5000
    ):
        raise ValueError(
            "Expected full-training checkpoints"
        )

    # Reconstruct original validation split.

    trainset = datasets.MNIST(
        root=str(cfg["data_root"]),
        train=True,
        download=False,
        transform=transforms.ToTensor()
    )

    generator = torch.Generator().manual_seed(
        int(cfg["seed"])
    )

    permutation = torch.randperm(
        len(trainset),
        generator=generator
    )

    start = int(cfg["train_count"])

    original_indices = permutation[
        start:start + args.max_samples
    ].tolist()

    validation_set = Subset(
        trainset,
        original_indices
    )

    loader = DataLoader(
        validation_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    # Evaluate both models on the same examples.

    results = []

    for name, model in models:

        results.append(
            evaluate_model(
                model,
                cfg,
                loader,
                device,
                name
            )
        )

    labels, pred_base, powers_base = results[0]

    labels_joint, pred_joint, powers_joint = results[1]

    if not np.array_equal(
        labels,
        labels_joint
    ):
        raise RuntimeError(
            "Validation sample order mismatch"
        )

    baseline = calculate_metrics(
        labels,
        pred_base,
        powers_base
    )

    joint = calculate_metrics(
        labels,
        pred_joint,
        powers_joint
    )

    correct_base = baseline["correct"]
    correct_joint = joint["correct"]

    both_correct = int(
        np.sum(correct_base & correct_joint)
    )

    both_wrong = int(
        np.sum(~correct_base & ~correct_joint)
    )

    baseline_only = int(
        np.sum(correct_base & ~correct_joint)
    )

    joint_only = int(
        np.sum(~correct_base & correct_joint)
    )

    # Exact paired classification test.

    try:
        from scipy.stats import binomtest

        discordant = baseline_only + joint_only

        p_value = (
            float(
                binomtest(
                    min(baseline_only, joint_only),
                    n=discordant,
                    p=0.5
                ).pvalue
            )
            if discordant
            else 1.0
        )

    except ImportError:
        p_value = None

    # Create output directory.

    output = (
        ROOT
        / "results"
        / "step7_5"
        / datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    output.mkdir(
        parents=True,
        exist_ok=False
    )

    # Save paired predictions.

    with (
        output / "paired_validation_predictions.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "validation_position",
            "mnist_train_index",
            "true_label",
            "baseline_pred",
            "joint_pred",
            "baseline_correct",
            "joint_correct",
            "baseline_correct_eff",
            "joint_correct_eff",
            "baseline_detector_eff",
            "joint_detector_eff",
            "baseline_margin",
            "joint_margin"
        ])

        for i in range(len(labels)):

            writer.writerow([
                i,
                original_indices[i],
                int(labels[i]),
                int(pred_base[i]),
                int(pred_joint[i]),
                int(correct_base[i]),
                int(correct_joint[i]),
                float(baseline["correct_eff"][i]),
                float(joint["correct_eff"][i]),
                float(baseline["detector_eff"][i]),
                float(joint["detector_eff"][i]),
                float(baseline["margin"][i]),
                float(joint["margin"][i])
            ])

    # Generate structured analysis.

    ratio = (
        joint["correct_eff"].mean()
        / max(
            baseline["correct_eff"].mean(),
            1e-12
        )
    )

    report = {
        "dataset": "MNIST validation set",
        "samples": len(labels),
        "checkpoints": {
            "baseline": paths[0],
            "joint": paths[1]
        },
        "baseline_accuracy": float(
            correct_base.mean()
        ),
        "joint_accuracy": float(
            correct_joint.mean()
        ),
        "baseline_correct_efficiency": float(
            baseline["correct_eff"].mean()
        ),
        "joint_correct_efficiency": float(
            joint["correct_eff"].mean()
        ),
        "correct_efficiency_ratio": float(ratio),
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "baseline_only_correct": baseline_only,
        "joint_only_correct": joint_only,
        "mcnemar_exact_p": p_value,
        "baseline_confusion": (
            baseline["confusion"].tolist()
        ),
        "joint_confusion": (
            joint["confusion"].tolist()
        ),
        "baseline_detector_response": (
            baseline["response"].tolist()
        ),
        "joint_detector_response": (
            joint["response"].tolist()
        )
    }

    with (
        output / "diagnosis.json"
    ).open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
            ensure_ascii=False
        )

    create_figures(
        baseline,
        joint,
        labels,
        models,
        cfg,
        validation_set,
        original_indices,
        output,
        device
    )

    print("=" * 65)

    print(
        "Baseline validation accuracy:",
        f"{correct_base.mean():.2%}"
    )

    print(
        "Joint validation accuracy:",
        f"{correct_joint.mean():.2%}"
    )

    print(
        "Correct-detector efficiency gain:",
        f"{ratio:.2f}x"
    )

    print(
        "Baseline-only correct:",
        baseline_only
    )

    print(
        "Joint-only correct:",
        joint_only
    )

    print(
        "McNemar exact p-value:",
        p_value
    )

    print(
        "Results saved:",
        output
    )


if __name__ == "__main__":
    main()