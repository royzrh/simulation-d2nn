"""Shared train, validation, checkpoint resume, and final official test protocol."""
from __future__ import annotations
from pathlib import Path
import time
import torch
from engine import (dataset_load, split_indices, loader, run_epoch, save_json,
                    atomic_torch_save, save_history, checkpoint_state,
                    restore_state, save_plots, jsonable)


def fit(model, args, config, out: Path, frozen_prefix=0):
    out.mkdir(parents=True, exist_ok=True)
    last = out / 'last_checkpoint.pt'
    best = out / 'best_model.pt'
    cfg_file = out / 'config.json'
    if cfg_file.exists():
        import json
        old_cfg = json.loads(cfg_file.read_text(encoding='utf-8'))
        if old_cfg != config:
            raise ValueError('Existing output directory contains different config')
    else:
        save_json(cfg_file, config)

    device = next(model.parameters()).device
    train_ds, test_ds = dataset_load(args.data_root)
    train_idx, val_idx, split_id = split_indices(args.seed, args.stage)
    if split_id != config['split_id']:
        raise RuntimeError('Master split mismatch')
    val_dl = loader(train_ds, val_idx, args.batch_size, False, 0, device, args.workers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    history, best_acc, best_loss, best_epoch, first_epoch = [], -1., float('inf'), 0, 1
    if args.resume is not None:
        cp = torch.load(last, map_location='cpu', weights_only=True)
        cp = restore_state(cp, model, optimizer, config, device)
        history = cp['history']
        best_acc = cp['best_val_acc']; best_loss = cp['best_val_loss']
        best_epoch = cp['best_epoch']; first_epoch = cp['epoch'] + 1
        print(f'[RESUME] Continue at epoch {first_epoch}', flush=True)

    snapshots = {}
    if frozen_prefix:
        for key in ('raw_phases', 'raw_amplitudes'):
            if hasattr(model, key):
                snapshots[key] = getattr(model, key).detach()[:frozen_prefix].clone()

    started = time.perf_counter()
    print(f'[START] {config["experiment"]} device={device} out={out}', flush=True)
    for epoch in range(first_epoch, args.epochs + 1):
        st = time.perf_counter()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        train_dl = loader(train_ds, train_idx, args.batch_size, True,
                          args.seed + 1000 + epoch, device, args.workers)
        train = run_epoch(model, train_dl, device, args.input_scale,
                          args.threshold, args.loss, optimizer,
                          args.log_every, epoch)
        val = run_epoch(model, val_dl, device, args.input_scale,
                        args.threshold, args.loss)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        for key, snapshot in snapshots.items():
            if not torch.equal(getattr(model, key).detach()[:frozen_prefix], snapshot):
                raise RuntimeError(f'Frozen parameter {key} changed!')
        seconds = time.perf_counter() - st
        row = {'epoch': epoch}
        row.update({'train_' + k: v for k, v in train.items()
                    if k not in ('confusion', 'n')})
        row.update({'val_' + k: v for k, v in val.items()
                    if k not in ('confusion', 'n')})
        row['seconds'] = seconds
        row['cuda_peak_allocated_gb'] = (
            torch.cuda.max_memory_allocated() / 2**30 if device.type == 'cuda' else 0.)
        history.append(row)
        if val['accuracy'] > best_acc or (val['accuracy'] == best_acc and
                                          val['loss'] < best_loss):
            best_acc, best_loss, best_epoch = val['accuracy'], val['loss'], epoch
            atomic_torch_save(dict(model_state_dict=model.state_dict(),
                          config=config, epoch=epoch,
                          validation_accuracy=best_acc), best)
        atomic_torch_save(checkpoint_state(model, optimizer, history,
                  best_acc, best_loss, best_epoch, epoch, config), last)
        save_history(out / 'training_history.csv', history)
        print(f'[Epoch {epoch}/{args.epochs}] '
              f'train={train["accuracy"]:.2%} val={val["accuracy"]:.2%} '
              f'best={best_acc:.2%}@{best_epoch} sec={seconds:.1f} '
              f'peak={row["cuda_peak_allocated_gb"]:.2f} GiB', flush=True)

    cp = torch.load(best, map_location='cpu', weights_only=True)
    model.load_state_dict(cp['model_state_dict'])
    val = run_epoch(model, val_dl, device, args.input_scale,
                    args.threshold, args.loss)
    report = {'config': config, 'best_epoch': best_epoch,
              'best_validation_accuracy': best_acc,
              'validation': jsonable(val), 'official_test_used': args.stage == 'full',
              'paper_geometry_fully_verified': False,
              'elapsed_seconds_this_invocation': time.perf_counter() - started}
    if args.stage == 'full':
        test_dl = loader(test_ds, list(range(10000)), args.batch_size,
                         False, 0, device, args.workers)
        print('[FINAL] Evaluate official MNIST test set once using best validation checkpoint',
              flush=True)
        test = run_epoch(model, test_dl, device, args.input_scale,
                         args.threshold, args.loss)
        report['test'] = jsonable(test)
        save_json(out / 'test_metrics.json', report)
        confusion = test['confusion']
    else:
        save_json(out / 'validation_metrics.json', report)
        confusion = val['confusion']
    save_plots(model, history, confusion, out)
    print(f'[DONE] validation={val["accuracy"]:.2%} '
          f'correct_eff={val["correct_efficiency"]:.4%} out={out}', flush=True)
    return report
