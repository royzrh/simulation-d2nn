"""Fast CPU-only optical sanity checks; safe while other processes use the GPU."""
from __future__ import annotations
import torch
from d2nn_model import FlexibleD2NN
from engine import optical_loss

def main():
    torch.set_num_threads(2)
    torch.manual_seed(42)
    for layers, modulation in ((1,'phase'), (1,'complex'), (5,'complex')):
        m = FlexibleD2NN(n_layers=layers, modulation=modulation,
                         over=1, pad_factor=2, input_gap_mm=30,
                         detector_side_mm=4.8)
        img = torch.rand(1,1,28,28)
        field = m.encode(img, threshold=.5, scale=1)
        intensity, powers = m(field)
        assert powers.shape == (1,10)
        assert torch.isfinite(intensity).all() and torch.isfinite(powers).all()
        loss, metrics = optical_loss(powers, intensity, torch.tensor([3]),
                                   field.abs().square().sum((-2,-1)), m.s,
                                   'baseline')
        loss.backward()
        assert torch.isfinite(m.raw_phases.grad).all()
        if modulation == 'complex':
            assert torch.isfinite(m.raw_amplitudes.grad).all()
        print(f'[PASS] {layers}-layer {modulation}: output={tuple(powers.shape)}, '
              f'loss={loss.item():.4f}', flush=True)
    # Patch test: masked gradient truly freezes both amplitude and phase.
    m = FlexibleD2NN(n_layers=7, modulation='complex', over=1,
                     pad_factor=2, input_gap_mm=30., detector_side_mm=.8)
    hook_handles = m.freeze_prefix(5)
    initial_phase = m.raw_phases.detach()[:5].clone()
    initial_amp = m.raw_amplitudes.detach()[:5].clone()
    opt = torch.optim.Adam(m.parameters(), lr=.005, weight_decay=0)
    field = m.encode(torch.rand(1,1,28,28), .5, 1)
    intensity, powers = m(field)
    loss, _ = optical_loss(powers, intensity, torch.tensor([3]),
                           field.abs().square().sum((-2,-1)), m.s, 'baseline')
    opt.zero_grad(set_to_none=True)
    loss.backward(); opt.step()
    assert torch.equal(m.raw_phases.detach()[:5], initial_phase)
    assert torch.equal(m.raw_amplitudes.detach()[:5], initial_amp)
    assert m.raw_phases.grad[5:].abs().sum().item() > 0
    print('[PASS] 7-layer patch: old 5 layers unchanged, added layers trainable')
    print('ALL CPU CHECKS PASSED (no GPU used)')

if __name__ == '__main__':
    main()
