"""Fig. S2-inspired 5+2 trainable patch, using a frozen COMPLEX 5-layer base.

This is explicitly not a strict reproduction until exact physical patch placement
and detector coordinates are verified from the original paper.
"""
from __future__ import annotations
import argparse
from datetime import datetime
from pathlib import Path
import torch
from d2nn_model import FlexibleD2NN
from engine import seed_all, device_from_arg, split_indices
from runner import fit

HERE = Path(__file__).resolve().parent
PAPER_ROOT = HERE.parent

def main():
    p = argparse.ArgumentParser(description='MNIST Fig. S2-inspired 5+2 optical patch')
    p.add_argument('--base-checkpoint', type=Path, required=True,
                   help='best_model.pt from a completed FULL FigS1 N=5 complex run')
    p.add_argument('--stage', choices=['smoke','pilot','full'], default='smoke')
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
    p.add_argument('--detector-side-mm', type=float, default=.8)
    p.add_argument('--loss', choices=['baseline','joint'], default='baseline')
    p.add_argument('--workers', type=int, default=0)
    p.add_argument('--log-every', type=int, default=200)
    p.add_argument('--data-root', type=Path, default=PAPER_ROOT.parent / 'data')
    p.add_argument('--results-root', type=Path, default=HERE / 'results' / 'figS2')
    p.add_argument('--resume', type=Path, default=None)
    a = p.parse_args()
    if (a.epochs < 1 or a.batch_size < 1 or a.lr <= 0 or a.oversample < 1 or
        a.pad_factor < 2 or a.input_scale < 1 or 28*a.input_scale > 200 or
        a.input_gap_mm < 0 or a.detector_side_mm <= 0 or a.workers < 0 or
        a.log_every < 0 or not 0 <= a.threshold <= 1):
        p.error('Invalid arguments')
    a.base_checkpoint = a.base_checkpoint.expanduser().resolve()
    if not a.base_checkpoint.is_file():
        p.error('No base checkpoint found')
    a.data_root = a.data_root.expanduser().resolve()
    a.results_root = a.results_root.expanduser().resolve()
    if a.resume is not None:
        a.resume = a.resume.expanduser().resolve()
        if a.resume.name != 'last_checkpoint.pt' or not a.resume.is_file():
            p.error('--resume must refer to an existing last_checkpoint.pt')
    base = torch.load(a.base_checkpoint, map_location='cpu', weights_only=True)
    weights, base_config = base['model_state_dict'], base['config']
    if not {'raw_phases','raw_amplitudes'}.issubset(weights):
        p.error('Paper Fig. S2 uses a COMPLEX-valued five-layer base. '
                'This checkpoint is phase-only or incompatible.')
    if weights['raw_phases'].shape != (5, 200, 200) or weights['raw_amplitudes'].shape != (5,200,200):
        p.error('Expected a 5-layer 200x200 COMPLEX base checkpoint')
    # The only supported patch base is a full training run using this extension.
    if (base_config.get('experiment') != 'FigS1_layers_modulation' or
        base_config.get('stage') != 'full' or
        base_config.get('modulation') != 'complex' or
        base_config.get('layers') != 5):
        p.error('Base must come from a FULL FigS1 complex 5-layer experiment')
    expected = {'seed': a.seed, 'over': a.oversample,
                'pad_factor': a.pad_factor, 'input_gap_mm': a.input_gap_mm,
                'input_scale': a.input_scale, 'threshold': a.threshold,
                'loss': a.loss}
    wrong = [key for key, val in expected.items() if base_config.get(key) != val]
    if wrong:
        p.error('Patch/base configuration mismatch: ' + ', '.join(wrong))
    if abs(float(base_config.get('detector_side_mm', -1)) - a.detector_side_mm) > 1e-9:
        p.error('Base/Patch detector width mismatch')
    seed_all(a.seed)
    device = device_from_arg(a.device)
    _, _, split_id = split_indices(a.seed, a.stage)
    if base_config.get('split_id') != split_id:
        p.error('Five-layer base uses a different 55k/5k master split')
    cfg = dict(experiment='FigS2_5plus2_frozen_complex', stage=a.stage,
        layers=7, base_layers=5, modulation='complex', base_path=str(a.base_checkpoint),
        base_sha256=__import__('hashlib').sha256(a.base_checkpoint.read_bytes()).hexdigest(),
        over=a.oversample, pad_factor=a.pad_factor, batch_size=a.batch_size,
        epochs=a.epochs, lr=a.lr, seed=a.seed,
        input_scale=a.input_scale, threshold=a.threshold,
        input_gap_mm=a.input_gap_mm, detector_side_mm=a.detector_side_mm,
        loss=a.loss, data_root=str(a.data_root), split_id=split_id,
        patch_interlayer_distance_mm=10.,
        patch_placement_exact_paper_match_unverified=True,
        detector_coordinates='provisional_step7_4')
    model = FlexibleD2NN(n_layers=7, modulation='complex',
        over=a.oversample, pad_factor=a.pad_factor,
        input_gap_mm=a.input_gap_mm,
        detector_side_mm=a.detector_side_mm)
    with torch.no_grad():
        model.raw_phases[:5].copy_(weights['raw_phases'])
        model.raw_amplitudes[:5].copy_(weights['raw_amplitudes'])
    model = model.to(device)
    _handles = model.freeze_prefix(5)
    if a.resume is not None:
        out = a.resume.parent
    else:
        out = a.results_root / ('patch7_complex_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    print('WARNING: Fig. S2 patch placement and exact original loss remain unverified.',
          flush=True)
    fit(model, a, cfg, out, frozen_prefix=5)

if __name__ == '__main__':
    main()
