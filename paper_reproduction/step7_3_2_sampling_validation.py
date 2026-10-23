#!/usr/bin/env python3
"""Step 7.3.2 | fixed-phase D2NN ASM oversampling convergence check.

Works with step7_3_1_loss_ablation.py in the SAME folder. Reuses the EXACT
five-layer geometry, phase mapping, binary-MNIST encoding and detector masks.
No training and no optimizer: the 200,000 phase parameters are fixed.

Examples:
  python -u step7_3_2_sampling_validation.py --stage smoke
  python -u step7_3_2_sampling_validation.py --stage pilot
  python -u step7_3_2_sampling_validation.py --stage full
  python -u step7_3_2_sampling_validation.py --run-dir 'results/step7_3_1/ablation_pilot_os1_...' --stage pilot
  python -u step7_3_2_sampling_validation.py --baseline '/path/baseline/best_model.pt' --joint '/path/joint/best_model.pt' --stage full

Sampling stages: smoke=20 (2/class), pilot=100 (10/class), full=1000
(100/class, matching Step7.3.1), all=entire 10000 MNIST test set.
Only the sampling precision varies; phase weights and physical geometry fixed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

try:
    from step7_3_1_loss_ablation import (
        OpticalD2NN, NEURONS, NLAYERS, PITCH, GAP, WL, DETECTORS,
        DETECTOR_SIDE, balanced_indices, EPS,
    )
except ImportError as e:
    raise SystemExit(
        "Cannot import step7_3_1_loss_ablation.py. Place this file in the "
        "same paper_reproduction directory as Step 7.3.1. Original error: " + str(e)
    ) from e

ROOT = Path(__file__).resolve().parent
STAGES = {"smoke": 2, "pilot": 10, "full": 100, "all": -1}
DEFAULT_OVERSAMPLES = (1, 2, 3)
MODELS = ("baseline", "joint")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, default=None,
                   help="Step7.3.1 ablation_... output folder (auto-discovers latest non-smoke run if omitted)")
    p.add_argument("--baseline", type=Path, default=None, help="explicit Baseline best_model.pt")
    p.add_argument("--joint", type=Path, default=None, help="explicit Joint best_model.pt")
    p.add_argument("--stage", choices=STAGES, default="pilot", help="pilot: 100 test images; full: 1000")
    p.add_argument("--oversamples", nargs="+", type=int, default=list(DEFAULT_OVERSAMPLES),
                   help="e.g. --oversamples 1 2 3; highest used as field reference")
    p.add_argument("--batch-size", type=int, default=1,
                   help="batch size for CPU FFT; 1 or 2 recommended on MacBook Air")
    p.add_argument("--pad-factor", type=int, default=2, help="same as Step7.2/7.3.1; default 2")
    p.add_argument("--seed", type=int, default=None, help="default from source Step7.3.1 config, normally 42")
    p.add_argument("--data-root", type=Path, default=None, help="default ../data (offline MNIST)")
    p.add_argument("--output-root", type=Path, default=None, help="default results/step7_3_2")
    p.add_argument("--torch-threads", type=int, default=None, help="limit PyTorch CPU threads")
    p.add_argument("--progress-every", type=int, default=10)
    args = p.parse_args()
    if args.batch_size < 1 or args.pad_factor < 2 or args.progress_every < 1:
        p.error("batch-size must be >=1, pad-factor >=2, progress-every >=1")
    if any(x < 1 for x in args.oversamples) or len(set(args.oversamples)) != len(args.oversamples):
        p.error("oversamples must be unique positive integers")
    args.oversamples.sort()
    if len(args.oversamples) < 2:
        p.error("Need at least two oversampling levels for a convergence comparison")
    if max(args.oversamples) > 4:
        p.error("Oversampling >4 not supported by default to avoid excessive CPU/RAM usage")
    if (args.baseline is None) != (args.joint is None):
        p.error("--baseline and --joint must be supplied together")
    if args.torch_threads is not None and args.torch_threads < 1:
        p.error("--torch-threads must be >=1")
    return args


def abs_path(path: Path) -> Path:
    return path.expanduser().resolve()


def locate_models(args):
    if args.baseline and args.joint:
        paths = {"baseline": abs_path(args.baseline), "joint": abs_path(args.joint)}
        run = paths["joint"].parent.parent
    else:
        if args.run_dir:
            run = abs_path(args.run_dir)
        else:
            parent = ROOT / "results" / "step7_3_1"
            candidates = [p for p in parent.glob("ablation_*")
                          if (p/"baseline"/"best_model.pt").is_file()
                          and (p/"joint"/"best_model.pt").is_file()]
            # Prefer an actual pilot/medium experiment over a later smoke test.
            normal = [p for p in candidates if "smoke" not in p.name]
            if normal:
                candidates = normal
            if not candidates:
                raise FileNotFoundError(
                    "No Step7.3.1 Baseline/Joint checkpoint pair found.\n"
                    "Pass --run-dir 'results/step7_3_1/ablation_pilot_...'\n"
                    "or --baseline '/.../baseline/best_model.pt' --joint '/.../joint/best_model.pt'."
                )
            run = max(candidates, key=lambda q: q.stat().st_mtime)
        paths = {name: run / name / "best_model.pt" for name in MODELS}
    for label, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} checkpoint: {path}")
    return run, paths


def load_config(run):
    f = run / "config.json"
    if not f.is_file():
        return {}
    cfg = json.loads(f.read_text(encoding="utf-8"))
    return cfg


def load_states(paths, pad_factor, config):
    states, infos = {}, {}
    src_list = []
    if config:
        for key, expected in (("oversample", 1), ("pad_factor", pad_factor),
                              ("binarize_threshold", None)):
            if key == "binarize_threshold":
                continue
            if key in config and config[key] != expected:
                raise ValueError(f"Step7.3.1 config {key}={config[key]!r}, expected {expected!r}")
        src = config.get("source_run_config") or {}
        if src:
            checks = {"detector_side": DETECTOR_SIDE, "detector_centers": DETECTORS,
                      "binarize_threshold": 0.5, "pad_factor": pad_factor,
                      "oversample": 1}
            for k, expected in checks.items():
                if k in src and (src[k] != [list(v) for v in expected] if k == "detector_centers" else src[k] != expected):
                    raise ValueError(f"Source optical config mismatch: {k} {src[k]!r} != {expected!r}")
    for name, path in paths.items():
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            raise ValueError(f"Expected Step7.3.1 best_model.pt: {path}")
        state = ckpt["model_state_dict"]
        if list(state) != ["raw_phases"] or tuple(state["raw_phases"].shape) != (NLAYERS, NEURONS, NEURONS):
            raise ValueError(f"Invalid 5x200x200 phase tensor: {path}")
        if not torch.isfinite(state["raw_phases"]).all():
            raise ValueError(f"Checkpoint contains NaN or Inf: {path}")
        if "method" in ckpt and ckpt["method"] != name:
            raise ValueError(f"Wrong method checkpoint: expected {name}, got {ckpt['method']}")
        if "source_checkpoint" in ckpt:
            src_list.append(str(ckpt["source_checkpoint"]))
        states[name] = {"raw_phases": state["raw_phases"].detach().cpu().clone()}
        infos[name] = {"path": str(path.resolve()), "selected_epoch": ckpt.get("epoch"),
                       "source_checkpoint": ckpt.get("source_checkpoint")}
    if len(src_list) == 2 and src_list[0] != src_list[1]:
        raise ValueError("Baseline/Joint checkpoints originated from DIFFERENT medium runs")
    return states, infos


def forward_with_field(model: OpticalD2NN, u: torch.Tensor):
    """Exact forward operations from Step7.3.1, plus return complex Uout."""
    for layer in range(NLAYERS):
        phase = math.pi * torch.sigmoid(model.raw_phases[layer])
        if model.over > 1:
            phase = phase.repeat_interleave(model.over, -2).repeat_interleave(model.over, -1)
        phase = F.pad(phase, (model.left, model.right, model.left, model.right))
        u = u * model.aperture * torch.exp(1j * phase)
        u = torch.fft.ifft2(torch.fft.fft2(u, dim=(-2, -1)) * model.transfer, dim=(-2, -1))
    intensity = u.abs().square()
    power = intensity.flatten(1) @ model.detectors.flatten(1).T
    return u, intensity, power


def calculate_metrics(field, intensity, power, input_power, labels):
    cls = power.argmax(1)
    true_power = power.gather(1, labels[:, None]).squeeze(1)
    total_det = power.sum(1)
    total_window = intensity.sum((-2, -1))
    return {
        "predictions": cls,
        "correct": (cls == labels),
        "correct_efficiency": true_power / input_power.clamp_min(EPS),
        "detector_capture": total_det / input_power.clamp_min(EPS),
        "window_efficiency": total_window / input_power.clamp_min(EPS),
        "correct_share_of_detectors": true_power / total_det.clamp_min(EPS),
    }


def active_field(field, model):
    """Return complex output only over the 80mm physical aperture region.

    Each input U is normalized to discrete sum(|U|²)=1 (Step7.3.1 encoding).
    Since the physical sample area dx² changes with over, the corresponding
    physically-normalized complex amplitude is U / dx, i.e. U * over / PITCH.
    Multiplying all levels by PITCH gives equivalent dimensionless U * over.
    This conversion is REQUIRED before comparing different oversampling levels.
    """
    s = slice(model.left, model.left+model.active)
    return (field[:, s, s] * model.over).contiguous()


def resize_complex(field, shape):
    if tuple(field.shape[-2:]) == tuple(shape):
        return field
    real = F.interpolate(field.real[:, None], size=shape, mode="bilinear", align_corners=False)[:, 0]
    imag = F.interpolate(field.imag[:, None], size=shape, mode="bilinear", align_corners=False)[:, 0]
    return torch.complex(real, imag)


def pairwise_errors(coarse, reference):
    """Relative complex L2 after phase alignment and matched physical gridding.

    interpolation uses a common 80mm FOV, cell-centered align_corners=False;
    it can add interpolation error to the sampling discrepancy.
    """
    other = resize_complex(coarse, reference.shape[-2:])
    # Remove physically irrelevant single global phase separately for each image.
    corr = (other.conj() * reference).sum((-2, -1))
    angle = corr / corr.abs().clamp_min(EPS)
    aligned = other * angle[:, None, None]
    ref_norm = reference.abs().square().sum((-2, -1)).clamp_min(EPS).sqrt()
    field_error = (aligned-reference).abs().square().sum((-2, -1)).sqrt() / ref_norm
    i_ref = reference.abs().square()
    i_coarse = other.abs().square()
    intensity_error = ((i_coarse-i_ref).square().sum((-2, -1)).sqrt() /
                       i_ref.square().sum((-2, -1)).clamp_min(EPS).sqrt())
    return field_error, intensity_error


def select_dataset(root, stage, seed):
    ds = datasets.MNIST(root=str(root), train=False, download=False,
                        transform=transforms.ToTensor())
    if stage == "all":
        return ds, list(range(len(ds)))
    groups = balanced_indices(ds.targets, 100, seed+1)  # exactly Step7.3.1 test split
    n = STAGES[stage]
    indices = [index for group in groups for index in group[:n]]
    return Subset(ds, indices), indices


def save_csv(path, data):
    if not data:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(data[0]))
        wr.writeheader()
        wr.writerows(data)


def plot_metrics(folder, rows, overs):
    titles = [
        ("accuracy", "Classification accuracy", 100),
        ("correct_efficiency", "Correct detector / input (%)", 100),
        ("detector_capture", "All detectors / input (%)", 100),
        ("window_efficiency", "Output window / input (%)", 100),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (key, title, scalar) in zip(axes.flat, titles):
        for name in MODELS:
            rr = sorted((r for r in rows if r["model"] == name), key=lambda z: z["oversample"])
            ax.plot([r["oversample"] for r in rr],
                    [r[key]*scalar for r in rr], "o-", label=name)
        ax.set(xlabel="OVERSAMPLE", ylabel="%", title=title, xticks=overs)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Step 7.3.2 | fixed weights, changing only ASM sampling", fontsize=14)
    fig.tight_layout()
    fig.savefig(folder/"sampling_comparison.png", dpi=180)
    plt.close(fig)


def plot_field_error(folder, field_summary, overs):
    ref = max(overs)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for model in MODELS:
        rows = [r for r in field_summary if r["model"] == model and r["reference_os"] == ref]
        rows = sorted(rows, key=lambda r: r["sample_os"])
        for ax, key, title in zip(axs,
                                  ("complex_field_rel_l2", "intensity_rel_l2"),
                                  ("Complex field rel L2 (global phase aligned)",
                                   "Intensity rel L2")):
            ax.plot([r["sample_os"] for r in rows], [r[key] for r in rows], "o-", label=model)
            ax.set(xlabel=f"OVERSAMPLE compared to reference {ref}", title=title, xticks=overs)
            ax.grid(alpha=0.3)
            ax.legend()
    fig.suptitle("80mm physical output FOV; bilinear registration to fine grid")
    fig.tight_layout()
    fig.savefig(folder/"field_error_vs_sampling.png",dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12,4.5))
    for model in MODELS:
        rr = [r for r in field_summary if r["model"] == model and r["sample_os"] != r["reference_os"]]
        rr = sorted(rr,key=lambda r:(r["sample_os"],r["reference_os"]))
        xs = [f"{r['sample_os']}→{r['reference_os']}" for r in rr]
        for a,k,label in zip(ax,("complex_field_rel_l2","intensity_rel_l2"),
                             ("Complex field comparison","Intensity comparison")):
            a.plot(xs,[r[k] for r in rr],"o-",label=model)
            a.set(xlabel="OVERSAMPLE grid comparison",ylabel="Relative L2",title=label)
            a.grid(alpha=0.3)
            a.legend()
    fig.suptitle("Pairwise sampling discrepancies (not a proof of convergence)")
    fig.tight_layout()
    fig.savefig(folder/"pairwise_field_error.png",dpi=180)
    plt.close(fig)


def plot_example(folder, snapshots, overs):
    if set(snapshots) != set(MODELS):
        return
    fig, axs = plt.subplots(len(MODELS),len(overs),figsize=(4*len(overs),7),squeeze=False)
    for i,name in enumerate(MODELS):
        for j,over in enumerate(overs):
            z = snapshots[name][over]
            arr = z["intensity"]
            im = axs[i,j].imshow(arr,origin="lower",extent=(-40,40,-40,40),cmap="inferno")
            axs[i,j].set(title=f"{name} os={over}  (digit={z['digit']})",xlabel="mm",ylabel="mm")
            fig.colorbar(im,ax=axs[i,j],fraction=0.046,pad=0.04)
    fig.suptitle("Example output intensity over same 80mm aperture (normalized physical units)")
    fig.tight_layout()
    fig.savefig(folder/"example_fields.png",dpi=130)
    plt.close(fig)


def main():
    args = parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    run, paths = locate_models(args)
    config = load_config(run)
    states, info = load_states(paths, args.pad_factor, config)
    seed = config.get("seed", 42) if args.seed is None else args.seed
    if config and args.seed is not None and seed != config.get("seed", seed):
        print("WARNING: seed differs from Step7.3.1; test subset will be different.",flush=True)
    test_root = abs_path(args.data_root or ROOT.parent/"data")
    dataset, subset_idx = select_dataset(test_root, args.stage, seed)
    data_loader = DataLoader(dataset,batch_size=args.batch_size,shuffle=False,num_workers=0)
    output_root = abs_path(args.output_root or ROOT/"results"/"step7_3_2")
    output_root.mkdir(parents=True,exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder = output_root/f"sampling_{args.stage}_{stamp}"
    folder.mkdir(exist_ok=False)
    overs = args.oversamples
    assert min(overs) >= 1
    print("="*79)
    print("Step 7.3.2 | fixed-phase, CPU ASM numerical sampling validation")
    print(f"Step7.3.1 folder: {run}")
    for n,p in paths.items():
        print(f"  {n}: {p}")
    print(f"Dataset: {test_root}; stage={args.stage}, n={len(dataset)}, seed={seed}")
    print(f"Oversamples={overs}; effective dx (um)={[round(PITCH*1e6/o,3) for o in overs]}")
    print(f"FFT sizes={[NEURONS*o*args.pad_factor for o in overs]}; batch={args.batch_size}; pad_factor={args.pad_factor}")
    print(f"OUTPUT: {folder}")
    print("Phases are FIXED. No optimizer. No training.")
    print("="*79,flush=True)
    setup = {"models":info,"ablation_run_dir":str(run),"stage":args.stage,
             "samples":len(dataset),"seed":seed,"oversamples":overs,
             "pad_factor":args.pad_factor,"batch_size":args.batch_size,
             "data_root":str(test_root),"physical":{
                 "wavelength_m":WL,"frequency_hz":299_792_458.0/WL,
                 "neurons_per_layer":NEURONS,"layers":NLAYERS,"pitch_m":PITCH,
                 "layer_gap_m":GAP,"detector_side_m":DETECTOR_SIDE,
                 "detector_centers_m":DETECTORS},
             "field_comparison": "80mm active output FOV, U*oversample to undo per-grid discrete input normalization, bilinear interpolation with align_corners=False, minimize constant global phase offset",
             "warnings": [
                 "Fixed phase masks, but sampled ASM and sampled detector integrals can differ.",
                 "FOV complex errors contain interpolation and registration errors; not direct experimental errors.",
                 "OS=3 is only the finest grid used, not an exact physical reference.",
                 "Propagation model and geometry inherited from Step 7.3.1 are provisional; scalar integrated intensity is not calibrated physical Poynting flux.",
                 "Do not select hyperparameters by repeatedly inspecting the test set." ]}
    (folder/"config.json").write_text(json.dumps(setup,indent=2,ensure_ascii=False),encoding="utf-8")

    detail_rows, error_rows, summaries, errors_summary = [], [], [], []
    sample_preds = {name:{} for name in MODELS}
    snapshots = {}
    tic = time.perf_counter()
    for name in MODELS:
        print(f"\nMODEL: {name}",flush=True)
        snapshots[name] = {}
        state = states[name]
        models = {}
        for over in overs:
            mdl = OpticalD2NN(over,args.pad_factor).eval()
            mdl.load_state_dict(state,strict=True)
            for par in mdl.parameters():
                par.requires_grad_(False)
            models[over] = mdl
        counts = {over:{"correct":0,"correct_efficiency":0.0,"detector_capture":0.0,
                        "window_efficiency":0.0,"correct_share_of_detectors":0.0,"n":0}
                  for over in overs}
        accum = {(a,b):{"complex_field_rel_l2":0.0,"intensity_rel_l2":0.0,"n":0}
                 for i,a in enumerate(overs) for b in overs[i+1:]}
        sample_counter = 0
        with torch.inference_mode():
            for images, labels in data_loader:
                bsz=len(labels)
                fields = {}
                for over in overs:
                    mdl = models[over]
                    u = mdl.encode(images)
                    in_p = u.abs().square().sum((-2,-1))
                    field,intensity,powers = forward_with_field(mdl,u)
                    stats = calculate_metrics(field,intensity,powers,in_p,labels)
                    stats_cpu = {k:v.cpu() for k,v in stats.items()}
                    acc = counts[over]
                    acc["n"] += bsz
                    acc["correct"] += int(stats_cpu["correct"].sum())
                    for key in ("correct_efficiency","detector_capture","window_efficiency","correct_share_of_detectors"):
                        acc[key] += float(stats_cpu[key].double().sum())
                    for j in range(bsz):
                        item = {"model":name,"oversample":over,"image_id":subset_idx[sample_counter+j],
                                "true_digit":int(labels[j]),"prediction":int(stats_cpu["predictions"][j]),
                                "correct":int(stats_cpu["correct"][j])}
                        for key in ("correct_efficiency","detector_capture","window_efficiency","correct_share_of_detectors"):
                            item[key] = float(stats_cpu[key][j])
                        detail_rows.append(item)
                    # Retain active complex fields for cross-resolution comparisons only.
                    fields[over] = active_field(field,mdl)
                    if sample_counter == 0:
                        w = fields[over][0]
                        # Convert to common units; values comparable at different sampling.
                        snapshots[name][over] = {"intensity":w.abs().square().cpu().numpy(),
                                                  "digit":int(labels[0])}
                    del u,field,intensity,powers
                for i, low in enumerate(overs):
                    for high in overs[i+1:]:
                        ferr,ierr = pairwise_errors(fields[low],fields[high])
                        key=(low,high)
                        a=accum[key]
                        a["complex_field_rel_l2"] += float(ferr.double().sum())
                        a["intensity_rel_l2"] += float(ierr.double().sum())
                        a["n"] += bsz
                        for j in range(bsz):
                            error_rows.append({"model":name,"image_id":subset_idx[sample_counter+j],
                                               "true_digit":int(labels[j]),
                                               "sample_os":low,"reference_os":high,
                                               "complex_field_rel_l2":float(ferr[j]),
                                               "intensity_rel_l2":float(ierr[j])})
                del fields
                sample_counter += bsz
                if sample_counter % args.progress_every < bsz or sample_counter == len(dataset):
                    print(f"  {name} {sample_counter}/{len(dataset)} images, elapsed={time.perf_counter()-tic:.1f}s",flush=True)
        for over in overs:
            a=counts[over]
            row={"model":name,"oversample":over,"images":a["n"],"accuracy":a["correct"]/a["n"]}
            for k in ("correct_efficiency","detector_capture","window_efficiency","correct_share_of_detectors"):
                row[k]=a[k]/a["n"]
            summaries.append(row)
            sample_preds[name][over]={r["image_id"]:r["prediction"] for r in detail_rows
                                       if r["model"]==name and r["oversample"]==over}
            print(f"  os={over}: accuracy={row['accuracy']:.2%}, correct-eff={row['correct_efficiency']:.4%}, "
                  f"all-det={row['detector_capture']:.4%}, window={row['window_efficiency']:.2%}",flush=True)
        for (low,high),a in accum.items():
            row={"model":name,"sample_os":low,"reference_os":high,
                 "images":a["n"],"complex_field_rel_l2":a["complex_field_rel_l2"]/a["n"],
                 "intensity_rel_l2":a["intensity_rel_l2"]/a["n"]}
            errors_summary.append(row)
            print(f"  field error os{low}→os{high}: complex={row['complex_field_rel_l2']:.4f}, "
                  f"intensity={row['intensity_rel_l2']:.4f}",flush=True)
        del models
        save_csv(folder/"sample_metrics.csv",detail_rows)
        save_csv(folder/"sample_field_errors.csv",error_rows)

    agreement=[]
    for name in MODELS:
        ref=max(overs)
        r=sample_preds[name][ref]
        for over in overs:
            p=sample_preds[name][over]
            agreement.append({"model":name,"oversample":over,"reference_os":ref,
                              "prediction_agreement_fraction":sum(p[i]==r[i] for i in r)/len(r)})
    advantages=[]
    for over in overs:
        b=next(r for r in summaries if r["model"]=="baseline" and r["oversample"]==over)
        j=next(r for r in summaries if r["model"]=="joint" and r["oversample"]==over)
        advantages.append({"oversample":over,
                           "accuracy_difference_pp":100*(j["accuracy"]-b["accuracy"]),
                           "correct_efficiency_relative_gain_pct":100*(j["correct_efficiency"]/max(b["correct_efficiency"],EPS)-1),
                           "detector_capture_relative_gain_pct":100*(j["detector_capture"]/max(b["detector_capture"],EPS)-1),
                           "window_efficiency_relative_gain_pct":100*(j["window_efficiency"]/max(b["window_efficiency"],EPS)-1)})
    save_csv(folder/"sampling_metrics.csv",summaries)
    save_csv(folder/"field_error_summary.csv",errors_summary)
    save_csv(folder/"model_advantage_by_sampling.csv",advantages)
    save_csv(folder/"prediction_agreement.csv",agreement)
    plot_metrics(folder,summaries,overs)
    plot_field_error(folder,errors_summary,overs)
    plot_example(folder,snapshots,overs)
    result={"configuration":setup,"sampling_metrics":summaries,
            "field_error_summary":errors_summary,"prediction_agreement":agreement,
            "joint_minus_baseline":advantages,"runtime_seconds":time.perf_counter()-tic}
    (folder/"summary.json").write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    (folder/"README.txt").write_text(
        "Step 7.3.2 fixed-phase ASM oversampling sensitivity test.\n"
        "No training; both checkpoints are validation-selected Step7.3.1 weights.\n"
        "Output metrics: sums of discrete |U|^2 as in Step7.3.1; normalization by input power cancels grid cell area.\n"
        "For complex-field comparisons, use U*oversample, crop same physical 80mm FOV,\n"
        "interpolate real/imag with bilinear align_corners=False, and remove one global phase per image.\n"
        "Complex/intensity errors include grid registration/interpolation differences.\n"
        "The physically meaningful conclusion is sampling sensitivity, not certification of converged optical accuracy.\n"
        "Only 1,2,3 grid levels cannot guarantee convergence.\n"
        "All model geometry and data preprocessing inherited from Step7.3.1 remain provisional.\n"
        "If Joint advantage disappears or classification collapses at higher sampling,\n"
        "revisit numerical propagation and retrain at a better numerical resolution.\n",
        encoding="utf-8")
    print("\n"+"="*79)
    for r in advantages:
        print(f"os{r['oversample']}: Joint–Baseline accuracy={r['accuracy_difference_pp']:+.2f}pp, "
              f"correct-eff relative gain={r['correct_efficiency_relative_gain_pct']:+.2f}%, "
              f"capture relative gain={r['detector_capture_relative_gain_pct']:+.2f}%")
    print(f"FINISHED ({time.perf_counter()-tic:.1f}s): {folder}",flush=True)


if __name__ == "__main__":
    main()
