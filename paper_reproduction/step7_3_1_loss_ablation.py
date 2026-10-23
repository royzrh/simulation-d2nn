#!/usr/bin/env python3
"""Step 7.3.1 | Five-layer THz D2NN loss-function finetuning ablation.

Compared methods, each starting from the SAME Step7.2 MEDIUM best_model.pt:
  baseline: CE + 0.5*(1-correct_efficiency) + 0.05*outside_fraction
  eff:      CE + 0.05*(-log(correct_efficiency)) + 0.05*outside_fraction
  joint:    CE + 0.05*(-log(correct_efficiency))
                + 0.05*(-log(detector_capture)) + 0.05*outside_fraction

The two 0.05 weights for log terms are EXPERIMENTAL values, NOT paper facts.

Each variant has identical initial optics, split and epoch-specific data permutations;
optimizer starts fresh for each variant. Model selection uses validation only:
maximum correct-detector efficiency subject to validation accuracy no lower than
(initial Medium validation accuracy minus --allowed-val-drop). If no fine-tuned
epoch is eligible, epoch 0 (unchanged Medium model) wins. The test set is
measured only ONCE per variant, after model selection. Never select a variant
based on the held-out test set.

The input scaling, detector centers, classification loss, 0.5 binary threshold,
first-plane gap, scalar power proxy, and oversample=1 are PROVISIONAL choices.
See README in output folder for interpretation.
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
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

C = 299_792_458.0
FREQ = 0.4e12
WL = C / FREQ
NLAYERS = 5
NEURONS = 200
PITCH = 400e-6
GAP = 30e-3
DETECTOR_SIDE = 4.8e-3
EPS = 1e-12
DETECTORS = [
    (x * 1e-3, y * 1e-3)
    for y, xs in ((12, (-12, 0, 12)), (0, (-18, -6, 6, 18)), (-12, (-12, 0, 12)))
    for x in xs
]

STAGES = {
    "smoke": {"train_each": 20, "val_each": 10, "test_each": 10, "epochs": 1},
    "pilot": {"train_each": 100, "val_each": 100, "test_each": 100, "epochs": 3},
    "medium": {"train_each": 500, "val_each": 100, "test_each": 100, "epochs": 3},
}
METHODS = ("baseline", "eff", "joint")


def cli():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=None, help="Step7.2 medium/.../best_model.pt; auto-search if omitted")
    p.add_argument("--stage", choices=STAGES, default="pilot")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--lambda-eff", type=float, default=0.05)
    p.add_argument("--lambda-collect", type=float, default=0.05)
    p.add_argument("--allowed-val-drop", type=float, default=0.01,
                   help="max allowed absolute validation accuracy drop, e.g. 0.01 is 1 percentage point")
    p.add_argument("--oversample", type=int, default=1)
    p.add_argument("--pad-factor", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--torch-threads", type=int, default=None)
    args = p.parse_args()
    if args.epochs is not None and args.epochs < 1:
        p.error("--epochs must be >=1")
    if args.batch_size < 1 or args.lr <= 0 or args.oversample < 1 or args.pad_factor < 2:
        p.error("Invalid batch-size/lr/oversample/pad-factor")
    if args.lambda_eff < 0 or args.lambda_collect < 0 or not 0 <= args.allowed_val_drop <= 1:
        p.error("Invalid loss weights or allowed-val-drop")
    if len(args.methods) != len(set(args.methods)):
        p.error("--methods contains duplicates")
    return args


def get_checkpoint(arg, root):
    if arg is not None:
        path = arg.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {path}")
        return path
    options = list((root / "results" / "step7_2").glob("medium*/best_model.pt"))
    if not options:
        raise FileNotFoundError(
            "Cannot find a Step7.2 Medium checkpoint. Pass the exact path, e.g.\n"
            'python step7_3_1_loss_ablation.py --checkpoint "results/step7_2/medium_os1_.../best_model.pt"'
        )
    return max(options, key=lambda q: q.stat().st_mtime).resolve()


def check_checkpoint(ckpt, args):
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError("Expected a Step7.2 best_model.pt containing 'model_state_dict'")
    state = ckpt["model_state_dict"]
    if tuple(state.get("raw_phases", torch.empty(0)).shape) != (5, 200, 200):
        raise ValueError("Checkpoint does not contain a five-layer 200x200 phase tensor")
    cfg = ckpt.get("run_config")
    if not isinstance(cfg, dict) or cfg.get("stage") != "medium":
        raise ValueError("Checkpoint lacks verified Step7.2 Medium metadata (wrong model?)")
    checks = {
        "seed": args.seed, "oversample": args.oversample,
        "pad_factor": args.pad_factor, "binarize_threshold": 0.5,
        "detector_side": DETECTOR_SIDE, "detector_centers": DETECTORS,
    }
    for key, expected in checks.items():
        if cfg.get(key) != expected:
            raise ValueError(f"Checkpoint {key}={cfg.get(key)!r} differs from requested {expected!r}")
    return state, cfg


def balanced_indices(labels, n_each, seed):
    g = torch.Generator().manual_seed(seed)
    groups = []
    labels = torch.as_tensor(labels)
    for digit in range(10):
        inds = torch.where(labels == digit)[0]
        if len(inds) < n_each:
            raise ValueError(f"Dataset has too few samples for digit {digit}")
        groups.append(inds[torch.randperm(len(inds), generator=g)[:n_each]].tolist())
    return groups


class OpticalD2NN(nn.Module):
    """Same phase/aperture/ASM/detector discretization as step7_2_mnist_training.py."""
    def __init__(self, over, pad_factor):
        super().__init__()
        self.over = over
        self.active = NEURONS * over
        self.dx = PITCH / over
        self.grid = self.active * pad_factor
        self.left = (self.grid - self.active) // 2
        self.right = self.grid - self.active - self.left
        self.raw_phases = nn.Parameter(torch.zeros(NLAYERS, NEURONS, NEURONS))

        ap = torch.zeros(self.grid, self.grid, dtype=torch.float32)
        sl = slice(self.left, self.left + self.active)
        ap[sl, sl] = 1
        self.register_buffer("aperture", ap, persistent=False)

        side_px = round(DETECTOR_SIDE / self.dx)
        if abs(side_px * self.dx - DETECTOR_SIDE) > 1e-9:
            raise ValueError("Detector width must be exactly expressible on numerical sampling")
        dets = []
        for cx, cy in DETECTORS:
            ix = round(cx / self.dx) + self.grid // 2
            iy = round(cy / self.dx) + self.grid // 2
            x0, y0 = ix-side_px//2, iy-side_px//2
            if not (0 <= x0 <= self.grid-side_px and 0 <= y0 <= self.grid-side_px):
                raise ValueError("Detector outside numerical FFT window")
            m = torch.zeros((self.grid, self.grid), dtype=torch.float32)
            m[y0:y0+side_px, x0:x0+side_px] = 1
            dets.append(m)
        detector_masks = torch.stack(dets)
        if detector_masks.sum(0).max().item() > 1:
            raise ValueError("Detectors overlap")
        self.register_buffer("detectors", detector_masks, persistent=False)
        freq = torch.fft.fftfreq(self.grid, d=self.dx, dtype=torch.float64)
        fy, fx = torch.meshgrid(freq, freq, indexing="ij")
        k = 2*math.pi/WL
        kz = k * torch.sqrt((1 - (WL*fx)**2 - (WL*fy)**2).to(torch.complex128))
        transfer = torch.exp(1j*(kz-k)*GAP).to(torch.complex64)
        self.register_buffer("transfer", transfer, persistent=False)

    def encode(self, images):
        a = (images[:, 0] >= 0.5).float()
        if self.over > 1:
            a = a.repeat_interleave(self.over, -2).repeat_interleave(self.over, -1)
        size = 28 * self.over
        top = (self.active-size)//2
        a = F.pad(a, (top, self.active-size-top, top, self.active-size-top))
        a = F.pad(a, (self.left, self.right, self.left, self.right))
        return (a / a.square().sum((-2, -1), keepdim=True).clamp_min(EPS).sqrt()).to(torch.complex64)

    def forward(self, u):
        for layer in range(NLAYERS):
            phase = math.pi * torch.sigmoid(self.raw_phases[layer])
            if self.over > 1:
                phase = phase.repeat_interleave(self.over, -2).repeat_interleave(self.over, -1)
            phase = F.pad(phase, (self.left, self.right, self.left, self.right))
            u = u * self.aperture * torch.exp(1j * phase)
            u = torch.fft.ifft2(torch.fft.fft2(u, dim=(-2,-1)) * self.transfer, dim=(-2,-1))
        intensity = u.abs().square()
        powers = intensity.flatten(1) @ self.detectors.flatten(1).T
        return intensity, powers


def loss_and_stats(method, power, intensity, labels, input_power, eff_weight, collect_weight):
    correct = power.gather(1, labels[:, None]).squeeze(1)
    all_det = power.sum(1)
    output = intensity.sum(dim=(-2, -1)).clamp_min(EPS)
    correct_eff = correct / input_power.clamp_min(EPS)
    collect_eff = all_det / input_power.clamp_min(EPS)
    win_eff = output / input_power.clamp_min(EPS)
    correct_share = correct / all_det.clamp_min(EPS)
    outside = (1 - all_det / output).clamp(0, 1)
    ce = F.cross_entropy(power.clamp_min(EPS).log(), labels)
    eff_log_loss = -correct_eff.clamp_min(EPS).log().mean()
    collect_log_loss = -collect_eff.clamp_min(EPS).log().mean()
    if method == "baseline":
        eff_term = 0.5*(1-correct_eff).mean()
        collect_term = ce.new_zeros(())
    else:
        eff_term = eff_weight * eff_log_loss
        collect_term = collect_weight * collect_log_loss if method == "joint" else ce.new_zeros(())
    outside_term = 0.05 * outside.mean()
    loss = ce + eff_term + collect_term + outside_term
    stats = {
        "loss": loss, "ce": ce, "efficiency_loss_unweighted": eff_log_loss,
        "collection_loss_unweighted": collect_log_loss,
        "efficiency_term": eff_term, "collection_term": collect_term,
        "outside_term": outside_term,
        "accuracy": (power.argmax(1) == labels).float().mean(),
        "correct_efficiency": correct_eff.mean(),
        "detector_capture": collect_eff.mean(),
        "window_efficiency": win_eff.mean(),
        "correct_share_of_detectors": correct_share.mean(),
    }
    return loss, stats


def evaluate(model, loader, method, eff_weight, collect_weight):
    model.eval()
    totals = {}
    n = correct = 0
    confusion = torch.zeros((10, 10), dtype=torch.int64)
    with torch.no_grad():
        for images, labels in loader:
            u = model.encode(images)
            input_p = u.abs().square().sum((-2, -1))
            intensity, powers = model(u)
            _, metrics = loss_and_stats(method, powers, intensity, labels, input_p, eff_weight, collect_weight)
            b = len(labels)
            for k, v in metrics.items():
                totals[k] = totals.get(k, 0.) + float(v.detach().item())*b
            pred = powers.argmax(1)
            correct += int((pred == labels).sum())
            n += b
            confusion += torch.bincount((labels*10 + pred).cpu(), minlength=100).reshape(10, 10)
    if n == 0:
        raise ValueError("Empty dataset")
    return {**{k: value/n for k,value in totals.items()}, "accuracy":correct/n,
            "confusion":confusion}


def train_epoch(model, opt, loader, method, eff_weight, collect_weight, epoch, total_epochs):
    model.train()
    totals = {}
    n = 0
    t0 = time.perf_counter()
    for i, (images, labels) in enumerate(loader, start=1):
        u = model.encode(images)
        input_p = u.abs().square().sum((-2, -1))
        opt.zero_grad(set_to_none=True)
        intensity, powers = model(u)
        loss, metrics = loss_and_stats(method, powers, intensity, labels, input_p, eff_weight, collect_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Loss is NaN/Inf at epoch {epoch}, batch {i}")
        loss.backward()
        opt.step()
        b = len(labels)
        n += b
        for k, v in metrics.items():
            totals[k] = totals.get(k, 0.) + float(v.detach().item())*b
        if i == 1 or i % 50 == 0:
            print(f"    epoch {epoch}/{total_epochs} batch {i} images {n}/{len(loader.dataset)} "
                  f"loss {totals['loss']/n:.3f} elapsed {time.perf_counter()-t0:.1f}s", flush=True)
    return {k:v/n for k,v in totals.items()}, time.perf_counter()-t0


def strip_confusion(result):
    return {k:v for k,v in result.items() if k != "confusion"}


def csv_write(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_plots(output_dir, histories, results):
    keys = [("val_accuracy", "Validation classification accuracy"),
            ("val_correct_efficiency", "Correct detector / input-light fraction"),
            ("val_detector_capture", "All detectors / input-light fraction"),
            ("val_window_efficiency", "Output window / input-light fraction")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (metric, title) in zip(axes.flat, keys):
        for method, hist in histories.items():
            ax.plot([h["epoch"] for h in hist], [h[metric] for h in hist], marker="o", label=method)
        ax.set(xlabel="Fine-tuning epoch (0 = original Medium model)", title=title)
        ax.grid(alpha=0.3)
        ax.legend()
        if metric == "val_accuracy":
            ax.set_ylim(0,1)
    fig.tight_layout()
    fig.savefig(output_dir / "val_comparison.png", dpi=190)
    plt.close(fig)

    names = list(results)
    fig, axes = plt.subplots(1, 3, figsize=(14,4))
    for ax, (metric, title) in zip(axes, [("accuracy", "Test accuracy"),
                                          ("correct_efficiency", "Test correct-detector fraction"),
                                          ("detector_capture", "Test detector capture fraction")]):
        ax.bar(names, [results[m][metric] for m in names])
        ax.set(title=title)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "test_summary.png", dpi=190)
    plt.close(fig)

    for method, res in results.items():
        cm = np.array(res["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(7, 6))
        im = ax.imshow(cm, cmap="Blues")
        ax.set(xlabel="Predicted digit",ylabel="True digit",title=f"{method}: {res['accuracy']:.2%} test accuracy",
               xticks=range(10),yticks=range(10))
        for r in range(10):
            for c in range(10):
                ax.text(c,r,str(cm[r,c]),ha="center",va="center",fontsize=8,
                        color="white" if cm[r,c] > cm.max()/2 else "black")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(output_dir / method / "confusion_matrix.png", dpi=170)
        plt.close(fig)


def main():
    args = cli()
    root = Path(__file__).resolve().parent
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    torch.manual_seed(args.seed)
    checkpoint_path = get_checkpoint(args.checkpoint, root)
    init_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    initial_state, source_config = check_checkpoint(init_ckpt, args)
    initial_state = {k:v.detach().clone() for k,v in initial_state.items()}
    stage = STAGES[args.stage]
    epochs = args.epochs if args.epochs is not None else stage["epochs"]
    data_root = (args.data_root or root.parent/"data").expanduser().resolve()
    out_root = (args.output_root or root/"results"/"step7_3_1").expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_dir = out_root / f"ablation_{args.stage}_os{args.oversample}_{stamp}"
    out_dir.mkdir(exist_ok=False)

    print("="*75)
    print("Step 7.3.1 | fair finetuning ablation from SAME Step7.2 Medium weights")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Stage={args.stage}, train={stage['train_each']*10}, validation={stage['val_each']*10}, "
          f"test={stage['test_each']*10}, epochs={epochs}, batch={args.batch_size}")
    print(f"Methods={','.join(args.methods)} lr={args.lr} over={args.oversample} pad={args.pad_factor}")
    print(f"Results={out_dir}")
    print("="*75, flush=True)

    train_mnist = datasets.MNIST(root=str(data_root),train=True,download=False,transform=transforms.ToTensor())
    test_mnist = datasets.MNIST(root=str(data_root),train=False,download=False,transform=transforms.ToTensor())
    train_groups = balanced_indices(train_mnist.targets, 600, args.seed)
    test_groups = balanced_indices(test_mnist.targets, 100, args.seed+1)
    train_idx = [idx for g in train_groups for idx in g[:stage["train_each"]]]
    val_idx = [idx for g in train_groups for idx in g[500:500+stage["val_each"]]]
    test_idx = [idx for g in test_groups for idx in g[:stage["test_each"]]]
    assert set(train_idx).isdisjoint(val_idx)
    train_set = Subset(train_mnist, train_idx)
    val_set = Subset(train_mnist, val_idx)
    test_set = Subset(test_mnist, test_idx)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,shuffle=False,num_workers=0)
    test_loader = DataLoader(test_set,batch_size=args.batch_size,shuffle=False,num_workers=0)

    def loader_for_epoch(epoch):
        rng = torch.Generator().manual_seed(args.seed + 1000 + epoch)
        return DataLoader(train_set,batch_size=args.batch_size,shuffle=True,num_workers=0,generator=rng)

    config = {
        "source_checkpoint":str(checkpoint_path), "source_checkpoint_epoch":init_ckpt.get("epoch"),
        "source_validation_accuracy":init_ckpt.get("validation_accuracy"),
        "source_run_config":source_config,"stage":args.stage,"epochs":epochs,
        "batch_size":args.batch_size,"seed":args.seed,"lr":args.lr,"oversample":args.oversample,
        "pad_factor":args.pad_factor,"lambda_eff":args.lambda_eff,"lambda_collect":args.lambda_collect,
        "allowed_val_drop":args.allowed_val_drop,"methods":args.methods,
        "train_size":len(train_set),"val_size":len(val_set),"test_size":len(test_set),
        "input": "binary at threshold 0.5, normalized discrete intensity sum",
        "detectors": "3-4-3; width 4.8mm; provisional centers",
        "note":"Numerical scalar proxy; no claim of converged optics or exact paper loss.",
    }
    with (out_dir/"config.json").open("w",encoding="utf-8") as f:
        json.dump(config,f,ensure_ascii=False,indent=2)

    histories, results = {}, {}
    for method in args.methods:
        print("\n"+"="*65+f"\nMETHOD = {method}\n"+"="*65,flush=True)
        method_dir = out_dir/method
        method_dir.mkdir()
        # Enforce exactly the same initial state AND a freshly initialized Adam.
        model = OpticalD2NN(args.oversample,args.pad_factor)
        model.load_state_dict(initial_state,strict=True)
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
        initial_val = evaluate(model,val_loader,method,args.lambda_eff,args.lambda_collect)
        initial_acc = initial_val["accuracy"]
        min_acc = initial_acc - args.allowed_val_drop
        best_epoch, best_score = 0, initial_val["correct_efficiency"]
        best_val = strip_confusion(initial_val)
        selected_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
        history = [{"epoch":0,"train_loss":None,"train_accuracy":None,"epoch_seconds":0.,
                    **{f"val_{k}":v for k,v in strip_confusion(initial_val).items()},
                    "meets_accuracy_floor":True}]
        print(f"  initial validation: accuracy={initial_acc:.2%} "
              f"correct efficiency={initial_val['correct_efficiency']:.3%} "
              f"all-detector={initial_val['detector_capture']:.3%}", flush=True)

        for epoch in range(1,epochs+1):
            train_stats, elapsed = train_epoch(model,optimizer,loader_for_epoch(epoch),method,
                                               args.lambda_eff,args.lambda_collect,epoch,epochs)
            val = evaluate(model,val_loader,method,args.lambda_eff,args.lambda_collect)
            eligible = val["accuracy"] + 1e-12 >= min_acc
            if eligible and (val["correct_efficiency"] > best_score + 1e-12 or
                             (abs(val["correct_efficiency"]-best_score) <= 1e-12 and val["accuracy"] > best_val["accuracy"])):
                best_epoch = epoch
                best_score = val["correct_efficiency"]
                best_val = strip_confusion(val)
                selected_state = {k:v.detach().clone() for k,v in model.state_dict().items()}
            row = {"epoch":epoch,"train_loss":train_stats["loss"],
                   "train_accuracy":train_stats["accuracy"],"epoch_seconds":elapsed,
                   **{f"val_{k}":v for k,v in strip_confusion(val).items()},
                   "meets_accuracy_floor":eligible}
            history.append(row)
            csv_write(method_dir/"training_history.csv",history)
            print(f"  epoch {epoch}/{epochs}: train={train_stats['accuracy']:.2%}, "
                  f"val={val['accuracy']:.2%}, correct efficiency={val['correct_efficiency']:.3%}, "
                  f"det capture={val['detector_capture']:.3%}, selected={best_epoch} "
                  f"elapsed={elapsed:.1f}s",flush=True)
            # Durable partial results in case of interruption.
            torch.save({"model_state_dict":selected_state,"epoch":best_epoch,
                        "validation_metrics":best_val,"source_checkpoint":str(checkpoint_path),
                        "method":method}, method_dir/"best_model.pt")
        if best_epoch == 0:
            torch.save({"model_state_dict":selected_state,"epoch":0,
                        "validation_metrics":best_val,"source_checkpoint":str(checkpoint_path),
                        "method":method}, method_dir/"best_model.pt")
        model.load_state_dict(selected_state)
        # Test only after freezing the validation-chosen weights; never use test to pick settings.
        test = evaluate(model,test_loader,method,args.lambda_eff,args.lambda_collect)
        result = {**strip_confusion(test),"confusion_matrix":test["confusion"].tolist(),
                  "selected_epoch":best_epoch,"selected_val_metrics":best_val,
                  "accuracy_floor":min_acc}
        results[method] = result
        histories[method] = history
        with (method_dir/"test_metrics.json").open("w",encoding="utf-8") as f:
            json.dump(result,f,indent=2)
        print(f"  TEST once: accuracy={test['accuracy']:.2%}, "
              f"correct efficiency={test['correct_efficiency']:.3%}, "
              f"det capture={test['detector_capture']:.3%}, "
              f"window={test['window_efficiency']:.2%}",flush=True)
        del model, optimizer

    render_plots(out_dir,histories,results)
    summary = {"configuration":config,"test_metrics_by_method":results,
               "warning":"Test outcomes are descriptive, not for selecting hyperparameters."}
    with (out_dir/"summary.json").open("w",encoding="utf-8") as f:
        json.dump(summary,f,indent=2,ensure_ascii=False)
    with (out_dir/"README.txt").open("w",encoding="utf-8") as f:
        f.write("Step 7.3.1 from identical Medium weights, fixed split/shuffle and NEW Adam per method.\n"
                "Model selected by maximum validation correct-detector fraction while validation accuracy "
                f"must remain at least initial validation accuracy minus {args.allowed_val_drop:.1%}.\n"
                "Epoch 0 = untouched Medium checkpoint; it can be selected if tuning fails.\n"
                "Final test set is evaluated ONCE per method after model selection; do not tune based on it.\n"
                "Values are discrete scalar optical intensity ratios, NOT validated physical Poynting flux.\n"
                "Input pixel scaling, exact detector coordinates, and loss form remain approximations.\n"
                "Baseline weights and the original Medium results are never modified.\n")
    print("\n"+"="*75)
    for method, stats in results.items():
        print(f"{method:>9}: test={stats['accuracy']:.2%} "
              f"correct={stats['correct_efficiency']:.3%} "
              f"capture={stats['detector_capture']:.3%} best_epoch={stats['selected_epoch']}")
    print(f"Saved comparisons: {out_dir}",flush=True)


if __name__ == "__main__":
    main()
