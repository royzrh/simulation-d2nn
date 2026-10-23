"""Evaluate a finished extension model, reporting validation metrics only by default."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from d2nn_model import FlexibleD2NN
from engine import device_from_arg, dataset_load, split_indices, loader, run_epoch, save_json, jsonable

def main():
    p = argparse.ArgumentParser(description='Evaluate saved MNIST extension checkpoint')
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--device', default='auto', choices=['auto','cuda','cpu'])
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--dataset', choices=['validation','test'], default='validation')
    args = p.parse_args()
    if args.batch_size < 1:
        p.error('batch-size must be positive')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    cfg = ckpt['config']
    if args.dataset == 'test' and cfg['stage'] != 'full':
        p.error('Do not evaluate official test with smoke or pilot checkpoints')
    stage = cfg['stage']
    device = device_from_arg(args.device)
    train_ds, test_ds = dataset_load(Path(cfg['data_root']))
    _, vi, actual_split_id = split_indices(cfg['seed'], stage)
    if actual_split_id != cfg['split_id']:
        raise RuntimeError('Dataset split ID mismatch')
    dl = loader(train_ds, vi, args.batch_size, False, 0, device) if args.dataset == 'validation' else (
        loader(test_ds, list(range(10000)), args.batch_size, False, 0, device))
    model = FlexibleD2NN(n_layers=cfg['layers'], modulation=cfg['modulation'],
        over=cfg['over'], pad_factor=cfg['pad_factor'],
        input_gap_mm=cfg['input_gap_mm'],
        detector_side_mm=cfg['detector_side_mm']).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    metrics = run_epoch(model, dl, device, cfg['input_scale'],
        cfg['threshold'], cfg['loss'])
    output = args.checkpoint.parent / f'evaluation_{args.dataset}.json'
    save_json(output, {'source_checkpoint': str(args.checkpoint),
                       'metric_dataset': args.dataset, 'metrics': jsonable(metrics)})
    print(f'{args.dataset} accuracy={metrics["accuracy"]:.2%} '
          f'correct_eff={metrics["correct_efficiency"]:.4%}\nSaved: {output}')

if __name__ == '__main__':
    main()
