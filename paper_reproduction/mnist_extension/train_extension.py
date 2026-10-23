"""Reproduce Fig. S1 MNIST layers/modulation ablation (paper alignment partial)."""
from __future__ import annotations
import argparse
import hashlib
from datetime import datetime
from pathlib import Path
import torch
from d2nn_model import FlexibleD2NN
from engine import seed_all, device_from_arg, split_indices
from runner import fit

HERE = Path(__file__).resolve().parent
PAPER_ROOT = HERE.parent

def parse_args():
    p = argparse.ArgumentParser(description='MNIST Fig. S1 D2NN extension')
    p.add_argument('--stage', choices=['smoke','pilot','full'], default='smoke')
    p.add_argument('--layers', type=int, choices=[1,5], required=True)
    p.add_argument('--modulation', choices=['phase','complex'], required=True)
    p.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    p.add_argument('--oversample', type=int, default=4)
    p.add_argument('--pad-factor', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=.005)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threshold', type=float, default=.5)
    p.add_argument('--input-scale', type=int, default=1)
    p.add_argument('--input-gap-mm', type=float, default=30.)
    p.add_argument('--detector-side-mm', type=float, default=4.8)
    p.add_argument('--loss', choices=['baseline','joint'], default='baseline')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--log-every', type=int, default=200)
    p.add_argument('--data-root', type=Path, default=PAPER_ROOT.parent / 'data')
    p.add_argument('--results-root', type=Path, default=HERE / 'results' / 'figS1')
    p.add_argument('--resume', type=Path, default=None,
                   help='Path to the exact last_checkpoint.pt of this experiment')
    args = p.parse_args()
    if (args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or
        args.oversample < 1 or args.pad_factor < 2 or
        args.input_scale < 1 or 28*args.input_scale > 200 or
        args.input_gap_mm < 0 or not 0 <= args.threshold <= 1 or
        args.workers < 0 or args.log_every < 0):
        p.error('Invalid optical/training arguments')
    args.data_root = args.data_root.expanduser().resolve()
    args.results_root = args.results_root.expanduser().resolve()
    if args.resume is not None:
        args.resume = args.resume.expanduser().resolve()
        if args.resume.name != 'last_checkpoint.pt' or not args.resume.is_file():
            p.error('--resume must point to existing last_checkpoint.pt')
    return args

def main():
    a = parse_args()
    seed_all(a.seed)
    device = device_from_arg(a.device)
    _, _, split_id = split_indices(a.seed, a.stage)
    cfg = dict(experiment='FigS1_layers_modulation', stage=a.stage,
         layers=a.layers, modulation=a.modulation,
         over=a.oversample, pad_factor=a.pad_factor,
         batch_size=a.batch_size, epochs=a.epochs, lr=a.lr, seed=a.seed,
         threshold=a.threshold, input_scale=a.input_scale,
         input_gap_mm=a.input_gap_mm,
         detector_side_mm=a.detector_side_mm, loss=a.loss,
         data_root=str(a.data_root), split_id=split_id,
         physics='full_grid_ASM_0.4THz_400um_30mm_interlayer_and_output',
         original_paper_input_geometry_confirmed=False,
         detector_coordinates='provisional_step7_4')
    model = FlexibleD2NN(n_layers=a.layers, modulation=a.modulation,
          over=a.oversample, pad_factor=a.pad_factor,
          input_gap_mm=a.input_gap_mm,
          detector_side_mm=a.detector_side_mm).to(device)
    if a.resume is not None:
        out = a.resume.parent
    else:
        tag = f'n{a.layers}_{a.modulation}_{a.stage}_s{a.input_scale}_z{a.input_gap_mm:g}'
        out = a.results_root / (tag + '_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    print('WARNING: Fig. S1 replication is PARTIAL until the original geometry and '
          'loss are verified.', flush=True)
    fit(model, a, cfg, out)

if __name__ == '__main__':
    main()
