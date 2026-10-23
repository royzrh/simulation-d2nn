
"""
Step 7.4 - RTX 4060 GPU Benchmark

五层相位型 D2NN
200 x 200 physical neurons per layer
Frequency: 0.4 THz
Layer spacing: 30 mm
Phase range: [0, pi]
Angular Spectrum Method (ASM)

Tests:
    OS=2, Batch=1
    OS=2, Batch=2
    OS=3, Batch=1

Only synthetic inputs are used for performance testing.
"""

import gc
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Parameters
# ============================================================

DEVICE = "cuda"

N = 200
NUM_LAYERS = 5

FREQUENCY = 0.4e12
C = 299792458.0
WAVELENGTH = C / FREQUENCY

PITCH = 400e-6
LAYER_DISTANCE = 0.03

PAD_FACTOR = 2
DETECTOR_SIDE = 4.8e-3

TEST_CONFIGS = [
    (4, 4),
    (4, 8),
    (4, 12),
    (4, 16),
]

WARMUP_STEPS = 5
MEASURE_STEPS = 50

DETECTOR_CENTERS = [
    (-0.012,  0.012),
    ( 0.000,  0.012),
    ( 0.012,  0.012),

    (-0.018,  0.000),
    (-0.006,  0.000),
    ( 0.006,  0.000),
    ( 0.018,  0.000),

    (-0.012, -0.012),
    ( 0.000, -0.012),
    ( 0.012, -0.012),
]


# ============================================================
# 2. ASM transfer function
# ============================================================

def make_transfer_function(grid_size, dx, z, device):

    fx = torch.fft.fftfreq(
        grid_size, d=dx, device=device
    )

    fy = torch.fft.fftfreq(
        grid_size, d=dx, device=device
    )

    FY, FX = torch.meshgrid(
        fy, fx, indexing="ij"
    )

    k = 2.0 * math.pi / WAVELENGTH

    kx = 2.0 * math.pi * FX
    ky = 2.0 * math.pi * FY

    kz_square = k**2 - kx**2 - ky**2

    # Propagating components
    kz_real = torch.sqrt(
        torch.clamp(kz_square, min=0.0)
    )

    # Evanescent components
    kz_imag = torch.sqrt(
        torch.clamp(-kz_square, min=0.0)
    )

    amplitude = torch.exp(-kz_imag * z)

    phase = kz_real * z

    H = torch.polar(amplitude, phase)

    return H.to(torch.complex64)


# ============================================================
# 3. Detector masks
# ============================================================

def make_detector_masks(size, dx, device):

    coords = (
        torch.arange(
            size, device=device, dtype=torch.float32
        ) - size / 2 + 0.5
    ) * dx

    YY, XX = torch.meshgrid(
        coords, coords, indexing="ij"
    )

    masks = []

    for cx, cy in DETECTOR_CENTERS:

        mask = (
            (torch.abs(XX - cx) <= DETECTOR_SIDE / 2)
            &
            (torch.abs(YY - cy) <= DETECTOR_SIDE / 2)
        )

        masks.append(mask.float())

    return torch.stack(masks, dim=0)


# ============================================================
# 4. Five-layer D2NN
# ============================================================

class D2NN(nn.Module):

    def __init__(self, oversample):

        super().__init__()

        self.oversample = oversample

        self.size = N * oversample

        self.dx = PITCH / oversample

        self.fft_size = self.size * PAD_FACTOR

        self.padding = (
            self.fft_size - self.size
        ) // 2

        # Five trainable phase masks
        self.phase_raw = nn.Parameter(
            torch.randn(
                NUM_LAYERS,
                N,
                N,
                dtype=torch.float32
            ) * 0.1
        )

        H = make_transfer_function(
            self.fft_size,
            self.dx,
            LAYER_DISTANCE,
            DEVICE
        )

        self.register_buffer("H", H)

        detector_masks = make_detector_masks(
            self.size,
            self.dx,
            DEVICE
        )

        self.register_buffer(
            "detector_masks",
            detector_masks
        )

    def propagate(self, field):

        p = self.padding

        # Zero-padding
        field_pad = F.pad(
            field,
            (p, p, p, p)
        )

        spectrum = torch.fft.fft2(field_pad)

        propagated = torch.fft.ifft2(
            spectrum * self.H
        )

        # Crop to active 80 mm aperture
        return propagated[
            ...,
            p:p + self.size,
            p:p + self.size
        ]

    def forward(self, input_field):

        field = input_field

        # Phase constraint: 0 to pi
        phases = math.pi * torch.sigmoid(
            self.phase_raw
        )

        # Nearest-neighbor physical neuron representation
        phases = phases.repeat_interleave(
            self.oversample, dim=-2
        ).repeat_interleave(
            self.oversample, dim=-1
        )

        # Five diffractive layers
        for layer in range(NUM_LAYERS):

            transmission = torch.exp(
                1j * phases[layer]
            )

            field = field * transmission

            field = self.propagate(field)

        intensity = torch.abs(field) ** 2

        powers = torch.einsum(
            "bxy,dxy->bd",
            intensity,
            self.detector_masks
        )

        return powers


# ============================================================
# 5. Synthetic input
# ============================================================

def make_input(batch_size, oversample):

    # Simple binary patterns, not real MNIST
    images = torch.randint(
        low=0,
        high=2,
        size=(batch_size, 28, 28),
        device=DEVICE
    ).float()

    field = torch.zeros(
        batch_size,
        N,
        N,
        device=DEVICE
    )

    start = (N - 28) // 2

    field[
        :,
        start:start + 28,
        start:start + 28
    ] = images

    field = field.repeat_interleave(
        oversample, dim=-2
    ).repeat_interleave(
        oversample, dim=-1
    )

    # Normalize discrete input intensity
    norm = torch.sqrt(
        (field ** 2).sum(
            dim=(-2, -1),
            keepdim=True
        ).clamp_min(1e-12)
    )

    field = field / norm

    return field.to(torch.complex64)


# ============================================================
# 6. One training step
# ============================================================

def training_step(model, optimizer, inputs, labels):

    optimizer.zero_grad(set_to_none=True)

    powers = model(inputs)

    logits = torch.log(
        powers.clamp_min(1e-12)
    )

    loss_ce = F.cross_entropy(
        logits, labels
    )

    correct = powers.gather(
        1,
        labels.unsqueeze(1)
    ).squeeze(1)

    # A simple efficiency-aware training objective
    loss_eff = -torch.log(
        correct.clamp_min(1e-12)
    ).mean()

    loss = loss_ce + 0.05 * loss_eff

    loss.backward()

    if not torch.isfinite(
        model.phase_raw.grad
    ).all():
        raise RuntimeError(
            "Non-finite phase gradient detected"
        )

    optimizer.step()

    return float(loss.detach().item())


# ============================================================
# 7. GPU benchmark
# ============================================================

def benchmark_case(oversample, batch_size):

    print()
    print("=" * 65)
    print(
        f"OVERSAMPLE={oversample} | Batch={batch_size}"
    )
    print("=" * 65)

    model = D2NN(oversample).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    inputs = make_input(
        batch_size, oversample
    )

    labels = torch.randint(
        0, 10,
        (batch_size,),
        device=DEVICE
    )

    # Warmup
    print("Warming up GPU ...")

    for _ in range(WARMUP_STEPS):
        training_step(
            model, optimizer, inputs, labels
        )

    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    times = []
    losses = []

    for step in range(MEASURE_STEPS):

        torch.cuda.synchronize()

        start = time.perf_counter()

        loss = training_step(
            model, optimizer, inputs, labels
        )

        torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

        times.append(elapsed)
        losses.append(loss)

        print(
            f"  Step {step + 1}: "
            f"{elapsed:.3f}s | Loss={loss:.5f}"
        )

    peak_allocated = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    peak_reserved = (
        torch.cuda.max_memory_reserved()
        / 1024**3
    )

    avg_time = sum(times) / len(times)

    print()
    print(f"Average step time : {avg_time:.3f} s")
    print(f"Peak allocated    : {peak_allocated:.3f} GB")
    print(f"Peak reserved     : {peak_reserved:.3f} GB")

    return {
        "os": oversample,
        "batch": batch_size,
        "time": avg_time,
        "allocated": peak_allocated,
        "reserved": peak_reserved,
        "seconds_per_image": avg_time / batch_size,
    }


# ============================================================
# 8. Main
# ============================================================

def main():

    print("=" * 65)
    print("Step 7.4 - D2NN RTX 4060 CUDA Benchmark")
    print("=" * 65)

    print("PyTorch version:", torch.__version__)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Install CUDA-enabled PyTorch."
        )

    print("GPU:", torch.cuda.get_device_name(0))

    properties = torch.cuda.get_device_properties(0)

    print(
        "Total VRAM:",
        round(properties.total_memory / 1024**3, 2),
        "GB"
    )

    print("Physical neurons:", N, "x", N)
    print("Layers:", NUM_LAYERS)
    print("Trainable parameters:", NUM_LAYERS * N * N)

    results = []

    for oversample, batch_size in TEST_CONFIGS:

        gc.collect()
        torch.cuda.empty_cache()

        try:
            result = benchmark_case(
                oversample, batch_size
            )

            results.append(result)

        except torch.cuda.OutOfMemoryError:

            print(
                f"[OOM] OS={oversample}, "
                f"Batch={batch_size}"
            )

        finally:
            gc.collect()
            torch.cuda.empty_cache()

    print()
    print("=" * 65)
    print("FINAL GPU BENCHMARK RESULTS")
    print("=" * 65)

    for r in results:

        print(
            f"OS={r['os']} "
            f"Batch={r['batch']} | "
            f"Time={r['time']:.3f}s | "
            f"Peak VRAM={r['allocated']:.3f} GB | "
            f"Per image={r['seconds_per_image']:.3f}s"
        )

    print()
    print("Benchmark finished.")


if __name__ == "__main__":
    main()
