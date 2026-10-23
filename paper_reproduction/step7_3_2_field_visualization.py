#!/usr/bin/env python3
"""Step 7.3.2 supplement | visualize output fields and diffractive phase masks.

No training. The code uses the EXACT OpticalD2NN model/checkpoint loader from
step7_3_2_sampling_validation.py, which must be in the SAME folder together
with step7_3_1_loss_ablation.py.

Examples:
  python -u step7_3_2_field_visualization.py
  python -u step7_3_2_field_visualization.py --oversample 3 --digits 0 1 4 7 9
  python -u step7_3_2_field_visualization.py --oversample 4 --examples-per-digit 2
  python -u step7_3_2_field_visualization.py --run-dir results/step7_3_1/ablation_pilot_os1_...
  python -u step7_3_2_field_visualization.py --baseline /.../baseline/best_model.pt --joint /.../joint/best_model.pt

The images are deterministic: first N entries of each digit's balanced
Step7.3.1 test subset (seed + 1). Both networks see identical inputs.
Scientific caveats: output intensity is a numerical scalar proxy (not calibrated
optical power), detectors use provisional center locations, and visualized
images are not a proof of model convergence or physical optical performance.
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
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch
from torchvision import datasets, transforms

try:
    from step7_3_2_sampling_validation import (
        ROOT, MODELS, abs_path, locate_models, load_config, load_states,
        OpticalD2NN, DETECTORS, DETECTOR_SIDE, NEURONS, NLAYERS, PITCH,
        balanced_indices, forward_with_field, active_field,
    )
except ImportError as exc:
    raise SystemExit(
        "Place this script alongside step7_3_2_sampling_validation.py and "
        "step7_3_1_loss_ablation.py. Original error: " + str(exc)
    ) from exc


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Step 7.3.1 run dir, otherwise auto-select latest non-smoke experiment")
    parser.add_argument("--baseline", type=Path, default=None, help="explicit baseline/best_model.pt")
    parser.add_argument("--joint", type=Path, default=None, help="explicit joint/best_model.pt")
    parser.add_argument("--oversample", type=int, default=3, help="display resolution: 1, 2, 3 or 4")
    parser.add_argument("--pad-factor", type=int, default=2,
                        help="FFT zero padding factor, must match training run (normally 2)")
    parser.add_argument("--digits", nargs="+", type=int, default=[0, 1, 4, 7, 9],
                        help="MNIST digit labels; defaults to 0 1 4 7 9")
    parser.add_argument("--examples-per-digit", type=int, default=1,
                        help="deterministic first N examples from same Step 7.3.1 test split")
    parser.add_argument("--seed", type=int, default=None,
                        help="normally taken from ablation config; must match previous data partition")
    parser.add_argument("--data-root", type=Path, default=None, help="offline MNIST folder (default ../data)")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="default results/step7_3_2/field_visualization_...")
    parser.add_argument("--torch-threads", type=int, default=None)
    args = parser.parse_args()
    if args.oversample not in (1, 2, 3, 4):
        parser.error("--oversample must be 1, 2, 3 or 4")
    if args.pad_factor < 2:
        parser.error("--pad-factor must be >=2")
    if not 1 <= args.examples_per_digit <= 20:
        parser.error("--examples-per-digit must be 1..20")
    if len(args.digits) != len(set(args.digits)) or not all(0 <= x <= 9 for x in args.digits):
        parser.error("--digits must be unique integers from 0..9")
    if (args.baseline is None) != (args.joint is None):
        parser.error("--baseline and --joint must be given together")
    if args.torch_threads is not None and args.torch_threads < 1:
        parser.error("--torch-threads must be >=1")
    return args


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def annotate_detector_boxes(ax, correct_digit, label_color="white"):
    """Use the EXACT detector centers and widths from training."""
    half = DETECTOR_SIDE * 1000 / 2
    for digit, (cx, cy) in enumerate(DETECTORS):
        x, y = cx * 1000, cy * 1000
        color = "lime" if digit == correct_digit else label_color
        ax.add_patch(Rectangle((x-half, y-half), 2*half, 2*half,
                               fill=False, linewidth=1.1 if digit != correct_digit else 2.0,
                               edgecolor=color))
        ax.text(x, y, str(digit), ha="center", va="center", fontsize=8,
                color=color, fontweight="bold")


def visualize_phases(folder, states):
    """Draw exact trained phase masks and wrapped Joint-Baseline phase difference."""
    phases = {
        name: (math.pi * torch.sigmoid(states[name]["raw_phases"])).cpu().numpy()
        for name in MODELS
    }
    difference = np.angle(np.exp(1j * (phases["joint"] - phases["baseline"])))
    fig, axes = plt.subplots(3, NLAYERS, figsize=(17, 9), constrained_layout=True)
    extent = (-40, 40, -40, 40)
    for layer in range(NLAYERS):
        for row, key in enumerate(MODELS):
            arr = phases[key][layer]
            im = axes[row, layer].imshow(arr, origin="lower", extent=extent,
                                         vmin=0, vmax=math.pi, cmap="twilight")
            axes[row, layer].set_title(f"{key}: layer {layer+1}", fontsize=10)
        axes[2, layer].imshow(difference[layer], origin="lower", extent=extent,
                              vmin=-math.pi, vmax=math.pi, cmap="coolwarm")
        axes[2, layer].set_title(f"wrapped Joint-Baseline: layer {layer+1}", fontsize=10)
        for row in range(3):
            axes[row, layer].set_xlabel("x (mm)")
            if layer == 0:
                axes[row, layer].set_ylabel("y (mm)")
    fig.colorbar(im, ax=list(axes[:2, :].flat), shrink=0.68, label="phase (rad)")
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(-math.pi, math.pi), cmap="coolwarm")
    fig.colorbar(sm, ax=list(axes[2, :].flat), shrink=0.68, label="wrapped phase difference (rad)")
    fig.suptitle("Fixed trained phase masks: same physical 80-mm aperture", fontsize=13)
    fig.savefig(folder / "phase_layers_comparison.png", dpi=130)
    plt.close(fig)
    return float(np.sqrt(np.mean(difference ** 2))), float(np.max(np.abs(difference)))


def load_images(root, seed, digits, n_each):
    ds = datasets.MNIST(root=str(root), train=False, download=False,
                        transform=transforms.ToTensor())
    # Identical deterministic test indices as Step 7.3.1 and Step 7.3.2.
    groups = balanced_indices(ds.targets, 100, seed + 1)
    samples = []
    for digit in digits:
        for local_number, ix in enumerate(groups[digit][:n_each], start=1):
            image, target = ds[ix]
            assert int(target) == digit
            samples.append({"index": int(ix), "label": int(target), "ordinal": local_number,
                            "image": image})
    return samples


def get_sample_snapshots(samples, states, over, padding):
    output = {}
    for name in MODELS:
        print(f"Computing fields: {name}, oversample={over}", flush=True)
        model = OpticalD2NN(over, padding).eval()
        model.load_state_dict(states[name], strict=True)
        model.requires_grad_(False)
        with torch.inference_mode():
            for item in samples:
                images = item["image"].unsqueeze(0)
                u = model.encode(images)
                in_power = float(u.abs().square().sum().item())
                f, intensity, powers = forward_with_field(model, u)
                p = powers[0].cpu().numpy().astype(np.float64)
                active = active_field(f, model)
                physical_intensity = active.abs().square()[0].cpu().numpy().astype(np.float32)
                # Since active_field=U*over, sum(I_physical)/over^2 is the
                # discretized output-window/input-light fraction, consistently
                # normalized across sampling rates.
                predicted = int(p.argmax())
                true = item["label"]
                result = {
                    "label": true,
                    "prediction": predicted,
                    "correct": int(predicted == true),
                    "powers": p,
                    "intensity": physical_intensity,
                    "input_power": in_power,
                    "correct_efficiency": float(p[true] / in_power),
                    "detector_capture": float(p.sum() / in_power),
                    "window_efficiency": float(intensity.sum().item() / in_power),
                    "correct_share": float(p[true] / max(p.sum(), 1e-12)),
                }
                output[name, item["index"]] = result
                print(f"  idx={item['index']} digit={true} pred={predicted} "
                      f"correct-eff={100*result['correct_efficiency']:.4f}%", flush=True)
        # Release large detector masks before constructing the next model.
        del model
    return output


def visualize_one(folder, item, output, over):
    idx = item["index"]
    digit = item["label"]
    base = output["baseline", idx]
    joint = output["joint", idx]
    ib = base["intensity"]
    ij = joint["intensity"]
    # Shared log intensity scale; no independent normalization between models.
    max_i = float(max(ib.max(), ij.max()))
    floor = max(max_i * 1e-6, 1e-20)
    log_b = np.log10(ib + floor)
    log_j = np.log10(ij + floor)
    vmin, vmax = math.log10(floor), math.log10(max_i + floor)
    diff = ij - ib
    spread = float(np.quantile(np.abs(diff), 0.997))
    if spread < 1e-20:
        spread = max_i * 1e-3 + 1e-20

    fig = plt.figure(figsize=(17, 9), constrained_layout=True)
    axes = fig.subplots(2, 3)
    input_arr = (item["image"][0].numpy() >= 0.5).astype(np.float32)
    axes[0, 0].imshow(input_arr, cmap="gray", vmin=0, vmax=1, origin="upper")
    axes[0, 0].set(title=f"Binary MNIST, digit {digit} (test id={idx})", xlabel="image pixel (top-down)", ylabel="image pixel")

    for ax, name, obj, transformed in ((axes[0, 1], "Baseline", base, log_b),
                                        (axes[0, 2], "Joint", joint, log_j)):
        im = ax.imshow(transformed, extent=(-40, 40, -40, 40), origin="lower",
                       vmin=vmin, vmax=vmax, cmap="magma", interpolation="nearest")
        annotate_detector_boxes(ax, digit)
        ax.set(xlabel="x (mm)", ylabel="y (mm)",
               title=f"{name}  pred={obj['prediction']}  "
                     f"correct-eff={obj['correct_efficiency']*100:.4f}%")
        ax.set_aspect("equal")
    fig.colorbar(im, ax=[axes[0,1], axes[0,2]], shrink=0.72,
                 label="log10(output intensity, matched normalization)")

    inds = np.arange(10)
    width = 0.38
    axes[1, 0].bar(inds-width/2, 100*base["powers"]/base["input_power"],
                   width, label="Baseline")
    axes[1, 0].bar(inds+width/2, 100*joint["powers"]/joint["input_power"],
                   width, label="Joint")
    axes[1, 0].axvline(digit, linestyle="--", linewidth=1)
    axes[1, 0].set(title="Per-detector collected intensity", xlabel="Detector class",
                   ylabel="% of input integrated intensity", xticks=range(10))
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(axis="y", alpha=0.3)

    norm = TwoSlopeNorm(vcenter=0, vmin=-spread, vmax=spread)
    imdiff = axes[1,1].imshow(diff, extent=(-40,40,-40,40), origin="lower",
                              cmap="coolwarm", norm=norm, interpolation="nearest")
    annotate_detector_boxes(axes[1,1], digit, label_color="black")
    axes[1,1].set(title="Joint - Baseline output intensity",xlabel="x (mm)",ylabel="y (mm)")
    axes[1,1].set_aspect("equal")
    fig.colorbar(imdiff, ax=axes[1,1], shrink=0.72)

    def fmt(name, row):
        return (f"{name}\n"
                f"predicted: {row['prediction']} {'OK' if row['correct'] else 'ERROR'}\n"
                f"correct detector: {100*row['correct_efficiency']:.5f}%\n"
                f"all detectors: {100*row['detector_capture']:.5f}%\n"
                f"output window: {100*row['window_efficiency']:.3f}%\n"
                f"correct/all (per image): {100*row['correct_share']:.2f}%")
    delta_pct = (joint["correct_efficiency"] / base["correct_efficiency"] - 1)*100 \
        if base["correct_efficiency"] > 0 else float("nan")
    txt = fmt("Baseline", base) + "\n\n" + fmt("Joint", joint) + \
          f"\n\nCorrect-detector relative gain: {delta_pct:+.2f}%"
    axes[1,2].axis("off")
    axes[1,2].text(0.04, 0.98, txt, va="top", family="monospace", fontsize=10,
                   transform=axes[1,2].transAxes)
    fig.suptitle(f"D2NN output fields (OS={over}) | digit={digit}, sample #{item['ordinal']} | fixed phases",
                 fontsize=14)
    filename = f"digit_{digit}_example_{item['ordinal']}_id_{idx}.png"
    fig.savefig(folder/filename, dpi=150)
    plt.close(fig)
    return filename


def plot_detector_matrix(folder, samples, output):
    # Each sample is one column; rows are detector classes, equal normalized color scale.
    fig, axs = plt.subplots(2, 1, figsize=(max(9, len(samples)*0.75), 8),
                            constrained_layout=True)
    all_values = [output[name, item["index"]]["powers"] /
                  output[name, item["index"]]["input_power"] * 100
                  for name in MODELS for item in samples]
    hi = max(float(a.max()) for a in all_values) if all_values else 1.0
    for ax, name in zip(axs, MODELS):
        arr = np.stack([output[name, item["index"]]["powers"] /
                        output[name, item["index"]]["input_power"] * 100
                        for item in samples], axis=1)
        im = ax.imshow(arr, origin="lower", aspect="auto", cmap="viridis",
                       vmin=0, vmax=hi)
        ax.set(title=f"{name}: detector responses (% of input)", ylabel="Detector class",
               yticks=range(10), xticks=np.arange(len(samples)),
               xticklabels=[f"{item['label']}|{item['index']}" for item in samples])
        ax.set_xlabel("True class | MNIST test image ID")
        for col, item in enumerate(samples):
            ax.scatter(col, item["label"], marker="s", edgecolor="white",
                       facecolor="none", s=75, linewidth=1.2)
    fig.colorbar(im, ax=axs, shrink=0.75, label="% of normalized input-light integral")
    fig.savefig(folder/"detector_response_matrix.png", dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    run, paths = locate_models(args)
    config = load_config(run)
    states, source = load_states(paths, args.pad_factor, config)
    seed = config.get("seed", 42) if args.seed is None else args.seed
    if args.seed is not None and seed != config.get("seed", seed):
        print("WARNING: seed overrides Step 7.3.1; examples may differ from prior tests.", flush=True)
    data_root = abs_path(args.data_root or ROOT.parent / "data")
    samples = load_images(data_root, seed, args.digits, args.examples_per_digit)
    output_root = abs_path(args.output_root or ROOT / "results" / "step7_3_2")
    folder = output_root / f"field_visualization_os{args.oversample}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    folder.mkdir(parents=True, exist_ok=False)
    print("="*72, flush=True)
    print("Step 7.3.2 | trained phase masks and output field visualization", flush=True)
    print("Model run:",run, flush=True)
    print("MNIST:",data_root,"digit labels:",args.digits, flush=True)
    print(f"OS={args.oversample}, dx={1e6*PITCH/args.oversample:.3f} um, "
          f"FFT={NEURONS*args.oversample*args.pad_factor}^2", flush=True)
    print("Output:",folder, flush=True)
    print("No training or optimizer; physical model and phase masks unchanged", flush=True)
    phase_rms, phase_max = visualize_phases(folder, states)
    snapshots = get_sample_snapshots(samples, states, args.oversample, args.pad_factor)
    rows = []
    files = []
    for item in samples:
        filepath = visualize_one(folder, item, snapshots, args.oversample)
        files.append(filepath)
        for name in MODELS:
            data = snapshots[name, item["index"]]
            row = {"model":name,"digit":item["label"],"mnist_test_image_id":item["index"],
                   "example_number":item["ordinal"],"oversample":args.oversample,
                   "prediction":data["prediction"],"correct":data["correct"],
                   "correct_efficiency":data["correct_efficiency"],
                   "detector_capture":data["detector_capture"],
                   "window_efficiency":data["window_efficiency"],
                   "correct_share_of_detectors":data["correct_share"]}
            for d,p in enumerate(data["powers"]):
                row[f"detector_{d}_power"] = float(p/data["input_power"])
            rows.append(row)
    save_csv(folder/"example_metrics.csv", rows)
    plot_detector_matrix(folder, samples, snapshots)
    summaries = {}
    for name in MODELS:
        rr = [r for r in rows if r["model"]==name]
        summaries[name] = {
            "n":len(rr),"accuracy":sum(r["correct"] for r in rr)/len(rr),
            "correct_efficiency":float(np.mean([r["correct_efficiency"] for r in rr])),
            "detector_capture":float(np.mean([r["detector_capture"] for r in rr])),
            "window_efficiency":float(np.mean([r["window_efficiency"] for r in rr])),
        }
    report = {
        "configuration":{"source_models":source,"run":str(run),"seed":seed,
                         "digits":args.digits,"examples_per_digit":args.examples_per_digit,
                         "oversample":args.oversample,"pad_factor":args.pad_factor,
                         "dx_m":PITCH/args.oversample,"data_root":str(data_root)},
        "phase_statistics":{"wrapped_joint_baseline_rms_rad":phase_rms,
                            "wrapped_joint_baseline_max_abs_rad":phase_max},
        "subset_metrics":summaries,
        "plot_files":["phase_layers_comparison.png","detector_response_matrix.png"]+files,
        "caveats":[
            "Images are illustrative examples, not representative aggregate statistics.",
            "The displayed intensity scale is shared between both models at the same oversampling level.",
            "Fractions are discretized squared-amplitude integrals, not calibrated real optical power.",
            "Input images are displayed top-down for readability; output field arrays follow the coordinate convention of the existing detector masks.",
            "Detector centers, input encoding and propagation model remain provisional.",
            "Test examples must not be used for loss hyperparameter selection.",
        ]
    }
    (folder/"summary.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    (folder/"README.txt").write_text(
        "Step 7.3.2 visual diagnostics: fixed trained phase masks (no training).\n"
        "The output maps cover the physical 80x80 mm aperture (not full padded FFT).\n"
        "The detector rectangles use exact configured width/centers; correct detector highlighted.\n"
        "Intensity U_active*OS squared is on a consistent relative physical scale\n"
        "for a fixed OS and across models. Use scalar-integral metrics for quantitation.\n"
        "WARNING: visible phase difference and power maps alone do not establish\n"
        "full ASM convergence or real optical transmission efficiency.\n",
        encoding="utf-8")
    print("="*72)
    print("Done. Generated",len(files),"digit field plots and two overview plots")
    for name, m in summaries.items():
        print(f"  {name}: {m['accuracy']*100:.1f}% accuracy on {m['n']} displayed examples; "
              f"correct-eff {m['correct_efficiency']*100:.4f}%")
    print("Saved to",folder)


if __name__=="__main__":
    main()
