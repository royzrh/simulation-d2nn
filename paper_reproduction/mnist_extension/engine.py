"""Shared, self-contained experiment engine for the MNIST extension scripts."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from d2nn_model import EPS


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def device_from_arg(which):
    if which == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    return torch.device('cuda' if (which == 'cuda' or
        (which == 'auto' and torch.cuda.is_available())) else 'cpu')


def dataset_load(root: Path):
    train = datasets.MNIST(str(root), train=True, download=False,
                           transform=transforms.ToTensor())
    test = datasets.MNIST(str(root), train=False, download=False,
                          transform=transforms.ToTensor())
    if len(train) != 60000 or len(test) != 10000:
        raise RuntimeError('Incomplete local MNIST dataset')
    return train, test


def split_indices(seed, stage):
    # Stable master split: changing pilot sample counts does not change the validation pool.
    if stage == 'smoke':
        train_n, val_n = 40, 20
    elif stage == 'pilot':
        train_n, val_n = 1000, 200
    elif stage == 'full':
        train_n, val_n = 55000, 5000
    else:
        raise ValueError(stage)
    p = torch.randperm(60000, generator=torch.Generator().manual_seed(seed))
    split_id = hashlib.sha256(p.numpy().tobytes()).hexdigest()[:16]
    return p[:train_n].tolist(), p[55000:55000 + val_n].tolist(), split_id


def loader(dataset, indices, batch_size, shuffle, seed, device, workers=0):
    return DataLoader(Subset(dataset, indices), batch_size=batch_size,
        shuffle=shuffle, generator=torch.Generator().manual_seed(seed) if shuffle else None,
        num_workers=workers, pin_memory=device.type == 'cuda',
        persistent_workers=workers > 0)


def optical_loss(powers, intensity, labels, input_power, active_slice, mode):
    correct = powers.gather(1, labels[:, None]).squeeze(1)
    capture = powers.sum(1)
    total = intensity.sum((-2, -1)).clamp_min(EPS)
    active = intensity[:, active_slice, active_slice].sum((-2, -1))
    inp = input_power.clamp_min(EPS)
    correct_eff = correct / inp
    detector_eff = capture / inp
    outside = (1 - capture / total).clamp(0, 1)
    ce = F.cross_entropy(powers.clamp_min(EPS).log(), labels)
    if mode == 'baseline':
        loss = ce + 0.5 * (1 - correct_eff).mean() + 0.05 * outside.mean()
    elif mode == 'joint':
        loss = (ce - 0.05 * correct_eff.clamp_min(EPS).log().mean()
                - 0.05 * detector_eff.clamp_min(EPS).log().mean()
                + 0.05 * outside.mean())
    else:
        raise ValueError(mode)
    return loss, {
        'loss': loss.detach(), 'ce': ce.detach(),
        'correct_efficiency': correct_eff.mean().detach(),
        'detector_capture': detector_eff.mean().detach(),
        'correct_share_of_detectors': (correct / capture.clamp_min(EPS)).mean().detach(),
        'fft_window_efficiency': (total / inp).mean().detach(),
        'active_80mm_efficiency': (active / inp).mean().detach(),
    }


def run_epoch(model, dl, device, scale, threshold, mode, optimizer=None,
              log_every=0, epoch=0):
    training = optimizer is not None
    model.train(training)
    names = ['loss', 'ce', 'correct_efficiency', 'detector_capture',
             'correct_share_of_detectors', 'fft_window_efficiency', 'active_80mm_efficiency']
    totals = {k: 0. for k in names}
    total_count, right = 0, 0
    confusion = torch.zeros((10, 10), dtype=torch.int64)
    t0 = time.perf_counter()
    with torch.enable_grad() if training else torch.no_grad():
        for batch, (images, labels) in enumerate(dl, 1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            fields = model.encode(images, threshold=threshold, scale=scale)
            in_power = fields.abs().square().sum((-2, -1))
            if training:
                optimizer.zero_grad(set_to_none=True)
            intensity, powers = model(fields)
            loss, measures = optical_loss(powers, intensity, labels, in_power, model.s, mode)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f'Nonfinite loss at epoch={epoch} batch={batch}')
            if training:
                loss.backward()
                for name, param in model.named_parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all().item():
                        raise FloatingPointError(f'Nonfinite gradient in {name}')
                optimizer.step()
            b = len(labels)
            total_count += b
            predictions = powers.argmax(1)
            right += (predictions == labels).sum().item()
            confusion += torch.bincount((labels * 10 + predictions).cpu(),
                                        minlength=100).reshape(10, 10)
            for k, v in measures.items():
                totals[k] += v.item() * b
            if training and log_every and (batch == 1 or batch % log_every == 0):
                print(f'  epoch={epoch} batch={batch}/{len(dl)} '
                      f'loss={totals["loss"]/total_count:.4f} '
                      f't={time.perf_counter()-t0:.1f}s', flush=True)
    if total_count == 0:
        raise RuntimeError('Empty dataloader')
    out = {k: v / total_count for k, v in totals.items()}
    out.update({'accuracy': right / total_count, 'n': total_count,
                'confusion': confusion})
    return out


def jsonable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(i) for i in obj]
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    return obj


def save_json(path, data):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(jsonable(data), indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(temp, path)


def atomic_torch_save(data, dest):
    dest = Path(dest)
    temp = dest.with_name(dest.name + '.tmp')
    torch.save(data, temp)
    os.replace(temp, dest)


def save_history(path, history):
    if not history:
        return
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def checkpoint_state(model, optimizer, history, best_acc, best_loss, best_epoch,
                     epoch, config):
    return dict(epoch=epoch, config=config, model_state_dict=model.state_dict(),
         optimizer_state_dict=optimizer.state_dict(), history=history,
         best_val_acc=best_acc, best_val_loss=best_loss, best_epoch=best_epoch,
         cpu_rng_state=torch.get_rng_state(),
         cuda_rng_state=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_state(checkpoint, model, optimizer, config, device):
    if checkpoint['config'] != config:
        differing = sorted({*config, *checkpoint['config']})
        differing = [k for k in differing if checkpoint['config'].get(k) != config.get(k)]
        raise ValueError('Resume config mismatch: ' + ', '.join(differing))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    torch.set_rng_state(checkpoint['cpu_rng_state'])
    if device.type == 'cuda' and checkpoint.get('cuda_rng_state') is not None:
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
    return checkpoint


def save_plots(model, history, confusion, path):
    path = Path(path)
    if history:
        fig, axs = plt.subplots(1, 2, figsize=(11, 4))
        epochs = [r['epoch'] for r in history]
        for ax, metric in zip(axs, ('loss', 'accuracy')):
            ax.plot(epochs, [r['train_' + metric] for r in history], '-o', label='Train')
            ax.plot(epochs, [r['val_' + metric] for r in history], '-o', label='Validation')
            ax.set(xlabel='Epoch', ylabel=metric, title=f'{metric.capitalize()} vs Epoch')
            ax.legend()
            ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(path / 'training_curves.png', dpi=160)
        plt.close(fig)
    cm = np.asarray(confusion)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap='Blues')
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=7,
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    ax.set(xlabel='Predicted', ylabel='True', title='MNIST confusion matrix')
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    fig.tight_layout(); fig.savefig(path / 'confusion_matrix.png', dpi=160)
    plt.close(fig)
    phase = (math.pi * torch.sigmoid(model.raw_phases.detach())).cpu().numpy()
    fig, axs = plt.subplots(1, model.n_layers,
                            figsize=(3.1 * model.n_layers, 3.2), squeeze=False)
    for l, ax in enumerate(axs[0]):
        ax.imshow(phase[l], cmap='twilight', origin='lower', vmin=0, vmax=math.pi)
        ax.set(title=f'L{l+1}'); ax.axis('off')
    fig.tight_layout(); fig.savefig(path / 'phase_layers.png', dpi=150)
    plt.close(fig)
    if hasattr(model, 'raw_amplitudes'):
        amps = torch.sigmoid(model.raw_amplitudes.detach()).cpu().numpy()
        fig, axs = plt.subplots(1, model.n_layers,
                                figsize=(3.1 * model.n_layers, 3.2), squeeze=False)
        for l, ax in enumerate(axs[0]):
            ax.imshow(amps[l], cmap='gray', origin='lower', vmin=0, vmax=1)
            ax.set(title=f'L{l+1}'); ax.axis('off')
        fig.tight_layout(); fig.savefig(path / 'amplitude_layers.png', dpi=150)
        plt.close(fig)
