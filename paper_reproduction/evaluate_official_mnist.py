#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate Step 7.4 five-layer D2NN checkpoints on official MNIST 10k test split.

Place beside step7_4_full_gpu_training.py in paper_reproduction and run:
    python -u paper_reproduction/evaluate_official_mnist.py --batch-size 2

No model training, checkpoint modification, or test-driven hyperparameter tuning.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from step7_4_full_gpu_training import OpticalD2NN, EPS  # noqa: E402

EXPECTED_RUNS = {
    "baseline": "full_baseline_os4_seed42_20261009_231658",
    "joint": "full_joint_os4_seed42_20261010_030826",
}
PAPER_ACCURACY = 0.9175


def args():
    p = argparse.ArgumentParser(description="Independent official MNIST-10k test of existing D2NN models")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--checkpoint-baseline", type=Path, default=None)
    p.add_argument("--checkpoint-joint", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--log-every", type=int, default=250)
    o = p.parse_args()
    if o.batch_size < 1:
        p.error("batch-size must be >= 1")
    if o.log_every < 0:
        p.error("log-every must be >= 0")
    return o


def choose_checkpoint(kind: str, explicit: Path | None):
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{kind} checkpoint does not exist: {path}")
        return path
    run_root = HERE / "results" / "step7_4"
    preferred = run_root / EXPECTED_RUNS[kind] / "best_model.pt"
    if preferred.is_file():
        return preferred
    candidates = sorted(run_root.glob(f"full_{kind}_os4_seed42_*/best_model.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"Cannot locate {kind} best_model.pt under {run_root}; provide --checkpoint-{kind}")
    print(f"[WARNING] Exact {kind} run missing; using latest match: {candidates[0]}", flush=True)
    return candidates[0]


def data_root_for(checkpoint_cfg: dict, override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    original = checkpoint_cfg.get("data_root")
    if original and Path(original).exists():
        return Path(original)
    return HERE.parent / "data"


def evaluate(kind, checkpoint_path, options, device, output_dir):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt or "config" not in ckpt:
        raise ValueError(f"Unsupported checkpoint (missing model_state_dict/config): {checkpoint_path}")
    cfg = ckpt["config"]
    if cfg.get("stage") != "full" or cfg.get("train_count") != 55000 or cfg.get("val_count") != 5000:
        raise ValueError(f"Checkpoint does not declare original full 55k/5k split: {checkpoint_path}")
    if cfg.get("loss") != kind:
        raise ValueError(f"Expected {kind} loss but checkpoint declares {cfg.get('loss')!r}")
    over = int(cfg["oversample"])
    pad = int(cfg["pad_factor"])
    gap = float(cfg["input_gap_mm"])
    threshold = float(cfg["threshold"])
    scale = int(cfg["input_pixel_scale"])
    print(f"\n[{kind.upper()}] checkpoint: {checkpoint_path}")
    print(f"Geometry: OS={over}, pad={pad}, input gap={gap} mm, scale={scale}, threshold={threshold}")

    model = OpticalD2NN(over, pad, gap).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    data_root = data_root_for(cfg, options.data_root)
    test_data = datasets.MNIST(root=str(data_root), train=False, download=True, transform=transforms.ToTensor())
    if len(test_data) != 10000:
        raise AssertionError(f"Expected exactly 10000 official test examples, got {len(test_data)}")
    loader = DataLoader(test_data, batch_size=options.batch_size, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")

    cm = torch.zeros((10, 10), dtype=torch.int64)
    per_class_correct = torch.zeros(10, dtype=torch.int64)
    per_class_count = torch.zeros(10, dtype=torch.int64)
    rows = []
    n_correct = 0
    sum_corr_eff = 0.0
    sum_det_eff = 0.0
    sum_corr_share = 0.0
    sum_pred_margin = 0.0
    sum_true_margin = 0.0
    t0 = time.perf_counter()

    with torch.inference_mode():
        for batch_index, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            inputs = model.encode(imgs, threshold, scale)
            input_p = inputs.abs().square().sum((-2, -1)).clamp_min(EPS)
            intensity, powers = model(inputs)
            if not torch.isfinite(powers).all().item():
                raise FloatingPointError(f"Non-finite detector powers at batch {batch_index}")
            pred = powers.argmax(dim=1)
            top2 = powers.topk(2, dim=1).values
            correct_power = powers.gather(dim=1, index=labels.unsqueeze(1)).squeeze(1)
            other_power = powers.masked_fill(torch.nn.functional.one_hot(labels, 10).bool(), float("-inf")).max(dim=1).values
            all_det = powers.sum(dim=1).clamp_min(EPS)
            correct_eff = correct_power / input_p
            detector_eff = all_det / input_p
            correct_share = correct_power / all_det
            predicted_margin = (top2[:, 0] - top2[:, 1]) / all_det
            true_margin = (correct_power - other_power) / all_det
            n_correct += (pred == labels).sum().item()
            sum_corr_eff += correct_eff.sum().item()
            sum_det_eff += detector_eff.sum().item()
            sum_corr_share += correct_share.sum().item()
            sum_pred_margin += predicted_margin.sum().item()
            sum_true_margin += true_margin.sum().item()
            cm += torch.bincount((labels * 10 + pred).cpu(), minlength=100).reshape(10, 10)
            per_class_count += torch.bincount(labels.cpu(), minlength=10)
            per_class_correct += torch.bincount(labels[pred == labels].cpu(), minlength=10)
            base_index = batch_index * options.batch_size
            cols = [x.detach().cpu().tolist() for x in (labels, pred, correct_eff, detector_eff, correct_share, predicted_margin, true_margin)]
            for j, vals in enumerate(zip(*cols)):
                y, yhat, ceff, deff, share, p_margin, t_margin = vals
                rows.append((base_index + j, y, yhat, int(y == yhat), ceff, deff, share, p_margin, t_margin))
            if options.log_every and ((batch_index + 1) % options.log_every == 0 or len(rows) == 10000):
                print(f"[{kind}] tested {len(rows):5d}/10000 | accuracy so far={n_correct / len(rows):.2%} | {time.perf_counter()-t0:.1f}s", flush=True)
            del intensity

    n = len(rows)
    assert n == 10000, f"Expected all 10000 MNIST test examples; evaluated {n}"
    assert sum(cm.flatten()).item() == 10000, "Confusion matrix sum != 10000"
    assert len({r[0] for r in rows}) == 10000, "Duplicate sample index"
    result = {
        "dataset": "official MNIST test split, train=False",
        "samples": n,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_validation_accuracy": ckpt.get("validation_accuracy"),
        "config": cfg,
        "test_accuracy": n_correct / n,
        "test_correct": n_correct,
        "paper_target_accuracy": PAPER_ACCURACY,
        "difference_from_paper_percentage_points": 100 * (n_correct / n - PAPER_ACCURACY),
        "correct_detector_efficiency": sum_corr_eff / n,
        "all_detector_efficiency": sum_det_eff / n,
        "correct_share_of_all_detectors": sum_corr_share / n,
        "predicted_top_two_margin_normalized_by_all_detectors": sum_pred_margin / n,
        "true_vs_best_other_margin_normalized_by_all_detectors": sum_true_margin / n,
        "per_class": {str(i): {"count": int(per_class_count[i]), "correct": int(per_class_correct[i]), "accuracy": (per_class_correct[i] / per_class_count[i]).item() if per_class_count[i] else None} for i in range(10)},
        "confusion": cm.tolist(),
        "notes": ["No training or optimizer step performed during this evaluation.", "Uses the model.encode and OpticalD2NN.forward methods imported from original Step7.4 script.", "Optical efficiencies are discrete simulated intensity sums, not calibrated laboratory optical power."]
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"{kind}_predictions_10000.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(("mnist_test_index", "true_label", "predicted_label", "correct", "correct_detector_efficiency", "all_detector_efficiency", "true_detector_share", "predicted_top_two_margin", "true_vs_best_other_margin"))
        w.writerows(rows)
    with (output_dir / f"{kind}_official_test.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[{kind.upper()}] OFFICIAL TEST: {result['test_accuracy']:.2%} ({n_correct}/10000); difference from paper: {result['difference_from_paper_percentage_points']:+.2f} pp", flush=True)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    o = args()
    if o.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device("cuda" if (o.device == "cuda" or (o.device == "auto" and torch.cuda.is_available())) else "cpu")
    output_dir = o.output_dir or (HERE / "results" / "official_mnist_test" / datetime.now().strftime("%Y%m%d_%H%M%S"))
    output_dir = output_dir.expanduser().resolve()
    checkpoints = {
        "baseline": choose_checkpoint("baseline", o.checkpoint_baseline),
        "joint": choose_checkpoint("joint", o.checkpoint_joint)
    }
    print(f"DEVICE: {device}; TEST SET: 10000 official MNIST test images; BATCH SIZE: {o.batch_size}", flush=True)
    print(f"OUTPUT: {output_dir}", flush=True)
    result = {kind: evaluate(kind, p, o, device, output_dir) for kind, p in checkpoints.items()}
    compact = {k: {"accuracy": v["test_accuracy"], "correct": v["test_correct"], "delta_vs_paper_pp": v["difference_from_paper_percentage_points"]} for k, v in result.items()}
    (output_dir / "comparison.json").write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n[DONE] baseline={:.2%} joint={:.2%}; results saved: {}".format(result["baseline"]["test_accuracy"], result["joint"]["test_accuracy"], output_dir), flush=True)


if __name__ == "__main__":
    main()
