"""Configurable full-grid ASM D2NN, compatible with Step 7.4 phase weights.

Research caveat: the five-layer input gap and exact detector centers are not
confirmed original-paper values. Default centers reproduce Step 7.4 only.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F

C = 299_792_458.0
FREQUENCY_HZ = 0.4e12
WAVELENGTH_M = C / FREQUENCY_HZ
NEURONS = 200
PITCH_M = 400e-6
GAP_M = 30e-3
EPS = 1e-12
# Coordinates taken from the user's Step 7.4 implementation (PROVISIONAL).
DETECTOR_CENTERS_MM = [
    (-12., 12.), (0., 12.), (12., 12.),
    (-18., 0.), (-6., 0.), (6., 0.), (18., 0.),
    (-12., -12.), (0., -12.), (12., -12.),
]

class FlexibleD2NN(nn.Module):
    def __init__(self, n_layers=5, modulation='phase', over=4, pad_factor=2,
                 input_gap_mm=30.0, detector_side_mm=4.8,
                 detector_centers_mm=None):
        super().__init__()
        if n_layers < 1 or over < 1 or pad_factor < 2:
            raise ValueError('n_layers>=1, over>=1, pad_factor>=2 required')
        if modulation not in {'phase', 'complex'}:
            raise ValueError('modulation must be phase or complex')
        if input_gap_mm < 0 or detector_side_mm <= 0:
            raise ValueError('Invalid geometry')
        self.n_layers = int(n_layers)
        self.modulation = modulation
        self.over = int(over)
        self.active_n = NEURONS * self.over
        self.fft_n = self.active_n * int(pad_factor)
        self.dx = PITCH_M / self.over
        self.left = (self.fft_n - self.active_n) // 2
        self.right = self.fft_n - self.active_n - self.left
        self.s = slice(self.left, self.left + self.active_n)
        self.input_gap_mm = float(input_gap_mm)
        self.detector_side_mm = float(detector_side_mm)
        self.detector_centers_mm = list(detector_centers_mm or DETECTOR_CENTERS_MM)

        # Keep the original Step 7.4 parameter name, shape, and initialization.
        self.raw_phases = nn.Parameter(0.7 * torch.randn(n_layers, NEURONS, NEURONS))
        if modulation == 'complex':
            # 0 <= amplitude <= 1; near-transparent initialization.
            # The original paper's exact amplitude parameterization is unknown.
            self.raw_amplitudes = nn.Parameter(
                torch.full((n_layers, NEURONS, NEURONS), 4.0, dtype=torch.float32)
            )

        aperture = torch.zeros((self.fft_n, self.fft_n), dtype=torch.float32)
        aperture[self.s, self.s] = 1.0
        self.register_buffer('aperture', aperture, persistent=False)
        side_px = round(detector_side_mm * 1e-3 / self.dx)
        if abs(side_px * self.dx - detector_side_mm * 1e-3) > 1e-9:
            raise ValueError('Detector width must be a multiple of simulation pixel pitch')
        masks = torch.zeros((10, self.fft_n, self.fft_n), dtype=torch.float32)
        if len(self.detector_centers_mm) != 10:
            raise ValueError('Exactly 10 detector centers required')
        for d, (cx_mm, cy_mm) in enumerate(self.detector_centers_mm):
            ix = round(cx_mm * 1e-3 / self.dx) + self.fft_n // 2
            iy = round(cy_mm * 1e-3 / self.dx) + self.fft_n // 2
            x0, y0 = ix - side_px // 2, iy - side_px // 2
            if min(x0, y0) < 0 or max(x0 + side_px, y0 + side_px) > self.fft_n:
                raise ValueError('Detector lies outside FFT grid')
            masks[d, y0:y0+side_px, x0:x0+side_px] = 1.
        if masks.sum(dim=0).max().item() > 1:
            raise ValueError('Detector regions overlap')
        self.register_buffer('detectors', masks, persistent=False)

        freqs = torch.fft.fftfreq(self.fft_n, d=self.dx, dtype=torch.float64)
        fy, fx = torch.meshgrid(freqs, freqs, indexing='ij')
        k = 2 * math.pi / WAVELENGTH_M
        kz = k * torch.sqrt((1. - (WAVELENGTH_M * fx) ** 2
                                 - (WAVELENGTH_M * fy) ** 2).to(torch.complex128))
        def asm(z_m):
            return torch.exp(1j * (kz - k) * z_m).to(torch.complex64)
        self.register_buffer('H_layer', asm(GAP_M), persistent=False)
        self.register_buffer('H_output', asm(GAP_M), persistent=False)
        self.register_buffer('H_patch', asm(10e-3), persistent=False)
        self.register_buffer('H_input', asm(input_gap_mm * 1e-3)
                             if input_gap_mm else torch.ones_like(self.H_layer),
                             persistent=False)
        self.has_input_gap = bool(input_gap_mm)

    def encode(self, images: torch.Tensor, threshold=0.5, scale=1) -> torch.Tensor:
        if not (1 <= scale <= NEURONS // 28):
            raise ValueError('scale must fit 200 neurons')
        amp = (images[:, 0] >= threshold).float()
        rep = self.over * scale
        amp = amp.repeat_interleave(rep, -2).repeat_interleave(rep, -1)
        side = amp.shape[-1]
        a = (self.active_n - side) // 2
        b = self.active_n - side - a
        amp = F.pad(amp, (a, b, a, b))
        amp = F.pad(amp, (self.left, self.right, self.left, self.right))
        amp = amp / amp.square().sum((-2, -1), keepdim=True).clamp_min(EPS).sqrt()
        return amp.to(torch.complex64)

    def _transmission(self, layer):
        phi = math.pi * torch.sigmoid(self.raw_phases[layer])
        if self.modulation == 'complex':
            amp = torch.sigmoid(self.raw_amplitudes[layer])
        else:
            amp = None
        if self.over != 1:
            phi = phi.repeat_interleave(self.over, -2).repeat_interleave(self.over, -1)
            if amp is not None:
                amp = amp.repeat_interleave(self.over, -2).repeat_interleave(self.over, -1)
        phi = F.pad(phi, (self.left, self.right, self.left, self.right))
        transmission = torch.exp(1j * phi)
        if amp is not None:
            amp = F.pad(amp, (self.left, self.right, self.left, self.right))
            transmission = transmission * amp
        return transmission

    def forward(self, field: torch.Tensor):
        if self.has_input_gap:
            field = torch.fft.ifft2(torch.fft.fft2(field) * self.H_input)
        for layer in range(self.n_layers):
            field = field * self.aperture * self._transmission(layer)
            if self.n_layers == 7 and layer >= 4:
                transfer = self.H_patch
            else:
                transfer = self.H_output if layer == self.n_layers - 1 else self.H_layer
            field = torch.fft.ifft2(torch.fft.fft2(field) * transfer)
        intensity = field.abs().square()
        powers = intensity.flatten(1) @ self.detectors.flatten(1).T
        return intensity, powers

    def freeze_prefix(self, n_fixed=5):
        """Freeze first n_fixed masks through gradient hooks (Adam weight decay MUST be zero).

        The prefix is checked against an immutable snapshot after every epoch.
        """
        if not 0 < n_fixed < self.n_layers:
            raise ValueError('n_fixed must be between 1 and n_layers-1')
        handles = []
        for name in ('raw_phases', 'raw_amplitudes'):
            if not hasattr(self, name):
                continue
            p = getattr(self, name)
            mask = torch.ones_like(p)
            mask[:n_fixed] = 0.
            handles.append(p.register_hook(lambda grad, mask=mask: grad * mask))
        return handles
