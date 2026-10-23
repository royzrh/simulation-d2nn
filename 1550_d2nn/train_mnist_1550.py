#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""1550 nm MNIST D2NN：单入口完成训练、验证、测试、绘图及加工/误差评估。

只依赖 numpy、matplotlib、torch；不依赖旧项目代码或 torchvision。
本文件交付时仅经过人工静态审阅，未执行、未训练、未验证数值结果。
未来运行示例（不是本次已经执行的命令）：
    python train_mnist_1550.py --download

参考：Lin et al., Science 361, 1004 (2018), doi:10.1126/science.aat8084。
训练组织与baseline/joint目标参考用户指定step7_4_full_gpu_training.py；
其400µm单元与零输入距离未直接继承，1550nm几何另行推导。
原文参数和工程调整的边界见同目录 REPRODUCTION_PLAN.md。
所有长度配置均带单位后缀；物理计算统一转换成 m，积分面积为 m^2。
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import math
import os
import platform
import random
import struct
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # 无需图形桌面；一次运行自动保存所有图。
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger("d2nn1550")
CLASS_CENTERS_MM = (
    (-0.60, -0.60), (0.00, -0.60), (0.60, -0.60),
    (-0.90, 0.00), (-0.30, 0.00), (0.30, 0.00), (0.90, 0.00),
    (-0.60, 0.60), (0.00, 0.60), (0.60, 0.60),
)  # (x,y)，图像y向下为正。中心坐标是本方案选择，并非原文精确坐标。


@dataclass
class Config:
    """几何、训练和验收配置；默认值是待训练的工程设计，不是已验证结果。"""

    wavelength_nm: float = 1550.0
    layers: int = 5
    neurons: int = 200
    pitch_um: float = 20.0
    input_neurons: int = 112
    binary_threshold: float = 0.5
    beam_waist_mm: float = 3.0  # 强度1/e^2半径，场振幅为exp(-r^2/w^2)。
    input_gap_mm: float = 40.0
    layer_gap_mm: float = 40.0
    output_gap_mm: float = 40.0
    substrate_mm: float = 0.5  # 最小厚度；出口薄浮雕在此参考面上附加。
    detector_width_um: float = 240.0
    phase_range_pi: float = 2.0
    phase_levels: int = 8
    surface_reflectance: float = 0.005  # 每面功率反射率；AR设计假定，需实测。
    absorption_per_m: float = 0.0  # 强度吸收系数，默认忽略体吸收，需实测替换。
    oversample: int = 4
    padding: int = 2
    epochs: int = 30
    qat_epochs: int = 10
    batch_size: int = 8
    eval_batch_size: int = 2
    learning_rate: float = 0.005  # 继承指定参考脚本的起点，不随波长机械缩放。
    phase_init_std: float = 0.7
    qat_lr_factor: float = 0.2  # 默认量化阶段初始学习率0.001。
    loss_mode: str = "joint"  # 效率导向的工程目标；不宣称优于baseline。
    efficiency_weight: float = 0.05
    capture_weight: float = 0.05
    leakage_weight: float = 0.05
    baseline_efficiency_weight: float = 0.5
    seed: int = 42
    workers: int = 0  # Windows默认无多进程，避免反复初始化。
    device: str = "auto"
    gradient_checkpoint: bool = True
    robust_samples: int = 200
    robust_trials: int = 3
    lateral_sigma_um: float = 3.0
    axial_sigma_um: float = 100.0
    height_sigma_nm: float = 30.0
    convergence_samples: int = 256
    convergence_oversample: int = 6
    convergence_padding: int = 3
    convergence_power_tolerance: float = 0.02
    convergence_accuracy_tolerance: float = 0.01
    download: bool = False
    data_dir: str = str(ROOT / "data" / "MNIST" / "raw")
    output_root: str = str(ROOT / "runs")
    # 只用于用户未来主动调小的调试；非零会在报告中标为非完整复现。
    train_limit: int = 0
    validation_limit: int = 0
    test_limit: int = 0
    skip_robustness: bool = False
    skip_convergence: bool = False

    @property
    def wavelength_m(self) -> float:
        return self.wavelength_nm * 1e-9

    @property
    def dx_m(self) -> float:
        return self.pitch_um * 1e-6 / self.oversample

    @property
    def active_samples(self) -> int:
        return self.neurons * self.oversample

    @property
    def grid_samples(self) -> int:
        return self.active_samples * self.padding

    @property
    def aperture_m(self) -> float:
        return self.neurons * self.pitch_um * 1e-6

    def validate(self) -> None:
        if self.layers < 1 or self.neurons < 28 or self.neurons % 2:
            raise ValueError("layers须为正整数，neurons须为不小于28的偶数。")
        if not (28 <= self.input_neurons <= self.neurons):
            raise ValueError("输入区域必须在有效孔径内。")
        if self.input_neurons % 2 or (self.neurons - self.input_neurons) % 2:
            raise ValueError("input_neurons须与neurons具有可对称填充的偶数尺寸。")
        if self.pitch_um <= 0 or self.wavelength_nm <= 0 or self.beam_waist_mm <= 0:
            raise ValueError("波长、像元尺寸和束腰须大于0。")
        if min(self.input_gap_mm, self.layer_gap_mm, self.output_gap_mm) <= 0:
            raise ValueError("输入、层间和输出空气距离都必须严格大于0。")
        if self.substrate_mm < 0 or self.absorption_per_m < 0:
            raise ValueError("基板厚度和吸收系数不得为负。")
        if self.phase_range_pi not in (1.0, 2.0):
            raise ValueError("仅支持0..π或0..2π相位范围。")
        if self.phase_levels < 2 or self.phase_levels & (self.phase_levels - 1):
            raise ValueError("phase_levels须为2的幂，便于二值光刻掩模导出。")
        if self.oversample < 2 or self.padding < 2:
            raise ValueError("最低oversample=2、padding=2；这仍不代表已经收敛。")
        if not 0 <= self.surface_reflectance < 1:
            raise ValueError("surface_reflectance是单面功率反射率，须在[0,1)。")
        if not 0 < self.binary_threshold < 1:
            raise ValueError("二值化阈值须在(0,1)内。")
        if self.detector_width_um <= 0:
            raise ValueError("探测区域宽度须大于0。")
        half_width = self.detector_width_um * 1e-6 / 2
        centers = np.asarray(CLASS_CENTERS_MM) * 1e-3
        if np.max(np.abs(centers)) + half_width > self.aperture_m / 2:
            raise ValueError("当前固定探测区布局超出有效孔径；更改几何时同步设计ROI。")
        for i, c in enumerate(centers):
            if np.any(np.all(np.abs(centers[:i] - c) < 2 * half_width, axis=1)):
                raise ValueError("探测区域发生重叠。")
        counts = (self.epochs, self.qat_epochs, self.batch_size, self.eval_batch_size,
                  self.robust_samples, self.robust_trials, self.convergence_samples)
        if min(counts) <= 0:
            raise ValueError("epochs、batch及评估样本/抽样次数必须为正。")
        if self.workers < 0 or min(self.train_limit, self.validation_limit, self.test_limit) < 0:
            raise ValueError("workers与样本上限不得为负。")
        if self.learning_rate <= 0 or min(self.efficiency_weight, self.capture_weight,
                                         self.leakage_weight, self.baseline_efficiency_weight) < 0:
            raise ValueError("学习率须为正，损失权重不得为负。")
        if self.phase_init_std <= 0 or not 0 < self.qat_lr_factor <= 1:
            raise ValueError("相位参数初始化标准差须为正，QAT学习率比例须在(0,1]。")
        if self.loss_mode not in ("baseline", "joint"):
            raise ValueError("loss_mode仅支持baseline或joint。")
        if min(self.lateral_sigma_um, self.axial_sigma_um, self.height_sigma_nm) < 0:
            raise ValueError("误差标准差不得为负。")
        if self.convergence_oversample <= self.oversample or self.convergence_padding <= self.padding:
            raise ValueError("收敛参考采样及padding须严格大于基准设置。")
        # 输出和新下载始终留在用户指定的1550_d2nn目录中。
        if not Path(self.output_root).resolve().is_relative_to(ROOT):
            raise ValueError("output_root必须位于1550_d2nn内。")
        if self.download and not Path(self.data_dir).resolve().is_relative_to(ROOT):
            raise ValueError("自动下载的data_dir必须位于1550_d2nn内。")


def silica_index(wavelength_m: float) -> float:
    """Malitson (1965) 熔石英三项Sellmeier；λ单位µm，非特定玻璃批次标定。

    doi:10.1364/JOSA.55.001205。实际加工应以采购批次n、膜层和温度标定为准。
    """
    wavelength_um = wavelength_m * 1e6
    if not 0.21 <= wavelength_um <= 3.71:
        raise ValueError("波长超出本Sellmeier参数采用的范围0.21..3.71µm。")
    b = (0.6961663, 0.4079426, 0.8974794)
    c = (0.0684043**2, 0.1162414**2, 9.896161**2)
    l2 = wavelength_um**2
    return math.sqrt(1.0 + sum(bi * l2 / (l2 - ci) for bi, ci in zip(b, c)))


def json_write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # 固定随机种子不等于跨硬件逐位相同；关闭cuDNN自动算法搜索。
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def read_idx(data_dir: Path, filename: str, allow_download: bool) -> np.ndarray:
    """读IDX或IDX.gz；只在显式--download时联网，并核对MNIST公开MD5。"""
    raw = data_dir / filename.removesuffix(".gz")
    packed = data_dir / filename
    if not raw.exists() and not packed.exists():
        if not allow_download:
            raise FileNotFoundError(
                f"缺少 {raw.name} 或 {packed.name}。放入data/MNIST/raw，或下次运行加--download。")
        data_dir.mkdir(parents=True, exist_ok=True)
        part = packed.with_suffix(packed.suffix + ".part")
        url = "https://storage.googleapis.com/cvdf-datasets/mnist/" + filename
        LOG.info("下载MNIST: %s", filename)
        try:
            with urllib.request.urlopen(url, timeout=120) as response, part.open("wb") as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
            digest = hashlib.md5(part.read_bytes()).hexdigest()  # 数据集兼容校验，非安全用途。
            if digest != MNIST_FILES[filename]:
                raise ValueError(f"MNIST下载校验失败: {filename}")
            part.replace(packed)
        finally:
            if part.exists():
                part.unlink()
    if raw.exists():
        payload = raw.read_bytes()
    else:
        if hashlib.md5(packed.read_bytes()).hexdigest() != MNIST_FILES[filename]:
            raise ValueError(f"压缩MNIST文件校验失败: {packed}")
        with gzip.open(packed, "rb") as handle:
            payload = handle.read()
    if len(payload) < 8:
        raise ValueError("IDX文件头不完整。")
    if payload[:3] != b"\x00\x00\x08" or payload[3] not in (1, 3):
        raise ValueError("仅支持MNIST uint8 IDX。")
    dims = payload[3]
    shape = struct.unpack(">" + "I" * dims, payload[4:4 + 4 * dims])
    values = np.frombuffer(payload, dtype=np.uint8, offset=4 + 4 * dims)
    if values.size != math.prod(shape):
        raise ValueError("IDX形状和文件长度不一致。")
    return values.reshape(shape).copy()


class Digits(Dataset):
    """只保留原始28×28数字；光学缩放在模型内完成，避免预存大复场。"""

    def __init__(self, images: np.ndarray, labels: np.ndarray, indices: np.ndarray):
        self.images = torch.from_numpy(images[indices].copy())
        self.labels = torch.from_numpy(labels[indices].astype(np.int64))
        self.indices = torch.from_numpy(indices.astype(np.int64))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.images[index].float().unsqueeze(0) / 255.0, self.labels[index], self.indices[index]


def make_datasets(cfg: Config, run_dir: Path) -> tuple[Digits, Digits, Digits]:
    source = Path(cfg.data_dir).resolve()
    # 允许只读复用已有原始MNIST；不调用旧项目代码，也不向旧data写入。
    shared = ROOT.parent / "data" / "MNIST" / "raw"
    if not cfg.download and not source.exists() and shared.exists():
        source = shared
    images = read_idx(source, "train-images-idx3-ubyte.gz", cfg.download)
    labels = read_idx(source, "train-labels-idx1-ubyte.gz", cfg.download)
    test_images = read_idx(source, "t10k-images-idx3-ubyte.gz", cfg.download)
    test_labels = read_idx(source, "t10k-labels-idx1-ubyte.gz", cfg.download)
    if images.shape != (60000, 28, 28) or labels.shape != (60000,):
        raise ValueError("须使用官方60000张MNIST训练集。")
    if test_images.shape != (10000, 28, 28) or test_labels.shape != (10000,):
        raise ValueError("须使用官方10000张MNIST测试集。")
    if np.any(labels > 9) or np.any(test_labels > 9):
        raise ValueError("MNIST标签超出0..9。")
    rng = np.random.default_rng(cfg.seed)
    # 各类固定抽500张验证，其余55000张训练；不读取测试集来调参。
    validation = np.concatenate([rng.permutation(np.flatnonzero(labels == c))[:500] for c in range(10)])
    training = np.setdiff1d(np.arange(60000), validation)
    training = rng.permutation(training)
    validation = rng.permutation(validation)
    testing = np.arange(10000)
    if cfg.train_limit:
        training = training[:cfg.train_limit]
    if cfg.validation_limit:
        validation = validation[:cfg.validation_limit]
    if cfg.test_limit:
        testing = testing[:cfg.test_limit]
    np.savez_compressed(run_dir / "dataset_split.npz", train=training, validation=validation, test=testing)
    json_write(run_dir / "data_source.json", {"read_from": str(source), "split_seed": cfg.seed})
    return Digits(images, labels, training), Digits(images, labels, validation), Digits(test_images, test_labels, testing)


def make_loader(dataset: Dataset, cfg: Config, training: bool = False) -> DataLoader:
    return DataLoader(dataset, batch_size=cfg.batch_size if training else cfg.eval_batch_size,
                      shuffle=training, num_workers=cfg.workers, drop_last=False,
                      pin_memory=torch.cuda.is_available(),
                      generator=torch.Generator().manual_seed(cfg.seed))


def angular_kernel(cfg: Config, distance_m: float, index: float, device: torch.device) -> Tensor:
    """带限角谱传播；频率使用fftfreq原顺序，不进行fftshift。

    去掉无关全局相位。用有理化公式避免kz-k0在单精度中的相消。
    以每轴相位梯度不超过计算窗半宽作带限，抑制传播核欠采样。
    该带限会舍弃频谱，绝不能代替扩大窗口/加密网格的收敛检查。
    """
    if distance_m < 0:
        raise ValueError("本脚本只允许向前传播。")
    n_grid = cfg.grid_samples
    if distance_m == 0:
        return torch.ones((n_grid, n_grid), dtype=torch.complex64, device=device)
    f = torch.fft.fftfreq(n_grid, d=cfg.dx_m, device=device, dtype=torch.float64)
    fy, fx = torch.meshgrid(f, f, indexing="ij")
    f2 = fx.square() + fy.square()
    k_cycles = index / cfg.wavelength_m
    propagating = f2 < k_cycles**2
    kz = torch.sqrt((k_cycles**2 - f2).clamp_min(1.0))
    width = n_grid * cfg.dx_m
    sampled = ((distance_m * fx.abs() / kz <= width / 2)
               & (distance_m * fy.abs() / kz <= width / 2))
    phase = -2 * math.pi * distance_m * f2 / (kz + k_cycles)
    return (torch.polar(torch.ones_like(phase), phase) * (propagating & sampled)).to(torch.complex64)


def propagate(field: Tensor, kernel: Tensor) -> Tensor:
    # 统一unitary FFT，使离散能量在未带限/未截光的传播中守恒。
    return torch.fft.ifft2(torch.fft.fft2(field, norm="ortho") * kernel, norm="ortho")


def center_embed(array: Tensor, size: int) -> Tensor:
    pad = size - array.shape[-1]
    if pad < 0 or pad % 2:
        raise ValueError("中心嵌入尺寸必须可对称补零。")
    return F.pad(array, (pad // 2,) * 4)


class D2NN(nn.Module):
    """二值幅度输入 -> 非零空气距离 -> 五片含基板DOE -> 探测能量积分。"""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.n_silica = silica_index(cfg.wavelength_m)
        self.raw_phase = nn.Parameter(torch.randn(cfg.layers, cfg.neurons, cfg.neurons) * cfg.phase_init_std)
        fine = cfg.active_samples
        size = cfg.grid_samples
        aperture = center_embed(torch.ones(fine, fine), size)
        self.register_buffer("aperture", aperture)
        xy = (torch.arange(size, dtype=torch.float64) - (size - 1) / 2) * cfg.dx_m
        yy, xx = torch.meshgrid(xy, xy, indexing="ij")
        self.register_buffer("illumination", torch.exp(-(xx.square() + yy.square()) /
                                                      (cfg.beam_waist_mm * 1e-3)**2).float())
        # ROI边缘与细网格不整除时按面积重叠积分，保证不同OS的实际ROI相同。
        roi = []
        half = cfg.detector_width_um * 1e-6 / 2
        for cx_mm, cy_mm in CLASS_CENTERS_MM:
            cx, cy = cx_mm * 1e-3, cy_mm * 1e-3
            wx = ((torch.minimum(xy + cfg.dx_m / 2, xy.new_tensor(cx + half))
                   - torch.maximum(xy - cfg.dx_m / 2, xy.new_tensor(cx - half))) / cfg.dx_m).clamp(0, 1)
            wy = ((torch.minimum(xy + cfg.dx_m / 2, xy.new_tensor(cy + half))
                   - torch.maximum(xy - cfg.dx_m / 2, xy.new_tensor(cy - half))) / cfg.dx_m).clamp(0, 1)
            roi.append((wy[:, None] * wx[None, :]).float())
        self.register_buffer("detector_masks", torch.stack(roi))
        border = torch.ones(size, size)
        margin = max(1, size // 10)
        border[margin:-margin, margin:-margin] = 0
        self.register_buffer("border_mask", border)
        cpu = torch.device("cpu")
        self.register_buffer("h_input", angular_kernel(cfg, cfg.input_gap_mm * 1e-3, 1.0, cpu))
        self.register_buffer("h_gap", angular_kernel(cfg, cfg.layer_gap_mm * 1e-3, 1.0, cpu))
        self.register_buffer("h_output", angular_kernel(cfg, cfg.output_gap_mm * 1e-3, 1.0, cpu))
        self.register_buffer("h_glass", angular_kernel(cfg, cfg.substrate_mm * 1e-3, self.n_silica, cpu))
        # 每片两面：功率(1-R)^2；对应场振幅(1-R)。不含多次相干反射。
        self.field_transmission = ((1 - cfg.surface_reflectance)
                                   * math.exp(-cfg.absorption_per_m * cfg.substrate_mm * 1e-3 / 2))

    def phase(self, quantized: bool, straight_through: bool = False) -> Tensor:
        phase = self.cfg.phase_range_pi * math.pi * torch.sigmoid(self.raw_phase)
        if not quantized:
            return phase
        if self.cfg.phase_range_pi == 2.0:
            step = 2 * math.pi / self.cfg.phase_levels
            q = torch.round(phase / step).remainder(self.cfg.phase_levels) * step
        else:
            step = math.pi / (self.cfg.phase_levels - 1)
            q = torch.round(phase / step).clamp(0, self.cfg.phase_levels - 1) * step
        # QAT前向是真正量化相位；反向使用连续相位梯度。
        return phase + (q - phase).detach() if straight_through else q

    def input_field(self, images: Tensor) -> Tensor:
        binary = (F.interpolate(images, size=(self.cfg.input_neurons,) * 2,
                                mode="nearest") >= self.cfg.binary_threshold).float()[:, 0]
        coarse = center_embed(binary, self.cfg.neurons)
        fine = coarse.repeat_interleave(self.cfg.oversample, -2).repeat_interleave(self.cfg.oversample, -1)
        field = center_embed(fine, self.cfg.grid_samples) * self.illumination
        # 不做逐图归一化；Pin始终指实际掩模后的输入能量。
        return field.to(torch.complex64)

    def layer_step(self, field: Tensor, phase: Tensor, h_air: Tensor,
                   shift: tuple[float, float]) -> Tensor:
        field = propagate(field, h_air)
        if shift == (0.0, 0.0):
            shifted_aperture = self.aperture
            phase_fine = phase.repeat_interleave(self.cfg.oversample, -2).repeat_interleave(self.cfg.oversample, -1)
            transmission = center_embed(torch.polar(torch.ones_like(phase_fine), phase_fine),
                                         self.cfg.grid_samples)
        else:
            # 在真实偏移后的物理坐标查询分段常数像元，禁止对复透过率做
            # 双线性插值：不同相位相加会造成不存在的吸收，污染容差结果。
            coordinate = (torch.arange(self.cfg.grid_samples, device=phase.device,
                                       dtype=torch.float64) - (self.cfg.grid_samples - 1) / 2) * self.cfg.dx_m
            ix = torch.floor((coordinate - shift[0] + self.cfg.aperture_m / 2)
                             / (self.cfg.pitch_um * 1e-6)).long()
            iy = torch.floor((coordinate - shift[1] + self.cfg.aperture_m / 2)
                             / (self.cfg.pitch_um * 1e-6)).long()
            valid_x = (ix >= 0) & (ix < self.cfg.neurons)
            valid_y = (iy >= 0) & (iy < self.cfg.neurons)
            shifted_aperture = (valid_y[:, None] & valid_x[None, :]).float()
            shifted_phase = phase[iy.clamp(0, self.cfg.neurons - 1)[:, None],
                                  ix.clamp(0, self.cfg.neurons - 1)[None, :]]
            transmission = torch.polar(torch.ones_like(shifted_phase), shifted_phase) * shifted_aperture
            # 仍是网格上的点采样，位移小于dx时边界响应有离散误差。
            # 最终机械容差应在更细网格复核；当前评估仅作初步筛查。
        # 前后均有4mm遮光口径；若实验只保留一个光阑，模型须相应调整。
        field = propagate(field * shifted_aperture, self.h_glass)
        return field * transmission * self.field_transmission

    def forward(self, images: Tensor, quantized: bool = False,
                assembly: dict[str, Any] | None = None) -> dict[str, Tensor]:
        field = self.input_field(images)
        input_power = field.abs().square().sum((-2, -1)) * self.cfg.dx_m**2
        if bool((input_power <= 0).any()):
            raise ValueError("二值输入有空白图像，请检查阈值及原始数据。")
        phases = self.phase(quantized, straight_through=self.training and quantized)
        if assembly is not None:
            phases = phases + assembly["phase_error"]
        for layer in range(self.cfg.layers):
            h_air = self.h_input if layer == 0 else self.h_gap
            shift = (0.0, 0.0)
            if assembly is not None:
                h_air = assembly["air_kernels"][layer]
                shift = assembly["shifts"][layer]
            if self.training and self.cfg.gradient_checkpoint:
                from torch.utils.checkpoint import checkpoint

                # 闭包参数以默认值绑定；反向重算时不能引用循环的最终layer。
                def step(f: Tensor, p: Tensor, kernel: Tensor = h_air,
                         offset: tuple[float, float] = shift) -> Tensor:
                    return self.layer_step(f, p, kernel, offset)

                field = checkpoint(step, field, phases[layer], use_reentrant=False)
            else:
                field = self.layer_step(field, phases[layer], h_air, shift)
        h_out = self.h_output if assembly is None else assembly["air_kernels"][-1]
        intensity = propagate(field, h_out).abs().square()
        detector_power = torch.einsum("bhw,khw->bk", intensity, self.detector_masks) * self.cfg.dx_m**2
        total_power = intensity.sum((-2, -1)) * self.cfg.dx_m**2
        border_power = (intensity * self.border_mask).sum((-2, -1)) * self.cfg.dx_m**2
        return {"powers": detector_power, "input_power": input_power,
                "output_power": total_power, "border_power": border_power, "intensity": intensity}


def optical_loss(result: dict[str, Tensor], labels: Tensor, cfg: Config) -> Tensor:
    """复用指定参考脚本的工程目标；原论文未披露唯一分类损失公式。"""
    powers = result["powers"]
    target = powers.gather(1, labels[:, None])[:, 0]
    detected = powers.sum(1).clamp_min(1e-20)
    fraction = (target / detected).clamp(1e-8, 1.0)
    efficiency = (target / result["input_power"].clamp_min(1e-20)).clamp(1e-8, 1.0)
    leakage = (1 - detected / result["output_power"].clamp_min(1e-20)).clamp(0, 1)
    ce = -fraction.log()  # 等价于对log(Pk)作交叉熵；无需softmax强度温度系数。
    if cfg.loss_mode == "baseline":
        return (ce + cfg.baseline_efficiency_weight * (1 - efficiency)
                + cfg.leakage_weight * leakage).mean()
    capture_efficiency = (detected / result["input_power"].clamp_min(1e-20)).clamp(1e-8, 1.0)
    return (ce - cfg.efficiency_weight * efficiency.log()
            - cfg.capture_weight * capture_efficiency.log() + cfg.leakage_weight * leakage).mean()


def checkpoint_save(path: Path, model: D2NN, cfg: Config, epoch: int,
                    validation: dict[str, float], quantized: bool) -> None:
    # 不保存大传播核；checkpoints只含少量训练参数及重建所需配置。
    torch.save({"raw_phase": model.raw_phase.detach().cpu(), "config": asdict(cfg),
                "epoch": epoch, "validation": validation, "quantized": quantized}, path)


def checkpoint_load(path: Path, model: D2NN) -> dict[str, Any]:
    # 仅加载本次运行自身产生的checkpoint。
    payload = torch.load(path, map_location="cpu", weights_only=True)
    with torch.no_grad():
        model.raw_phase.copy_(payload["raw_phase"].to(model.raw_phase.device))
    return payload


@torch.no_grad()
def validation_metrics(model: D2NN, loader: DataLoader, cfg: Config, quantized: bool) -> dict[str, float]:
    model.eval()
    count = correct = 0
    loss_sum = efficiency_sum = 0.0
    device = model.raw_phase.device
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        result = model(images, quantized)
        batch = len(labels)
        loss_sum += float(optical_loss(result, labels, cfg)) * batch
        correct += int((result["powers"].argmax(1) == labels).sum())
        target = result["powers"].gather(1, labels[:, None])[:, 0]
        efficiency_sum += float((target / result["input_power"]).sum())
        count += batch
    return {"loss": loss_sum / count, "accuracy": correct / count,
            "target_efficiency": efficiency_sum / count}


def train_stage(model: D2NN, train_loader: DataLoader, val_loader: DataLoader,
                cfg: Config, run_dir: Path, history: list[dict[str, Any]], quantized: bool) -> None:
    stage = "quantized" if quantized else "continuous"
    count_epochs = cfg.qat_epochs if quantized else cfg.epochs
    lr = cfg.learning_rate * (cfg.qat_lr_factor if quantized else 1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=count_epochs, eta_min=lr * 0.1)
    checkpoint = run_dir / f"best_{stage}.pt"
    best = (-1.0, -math.inf)
    device = model.raw_phase.device
    for epoch in range(1, count_epochs + 1):
        started = time.perf_counter()
        model.train()
        total = correct = 0
        loss_sum = 0.0
        for batch_index, (images, labels, _) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            result = model(images, quantized)
            loss = optical_loss(result, labels, cfg)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("训练损失非有限值；停止并保留已有文件。")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, error_if_nonfinite=True)
            optimizer.step()
            batch = len(labels)
            total += batch
            correct += int((result["powers"].detach().argmax(1) == labels).sum())
            loss_sum += float(loss.detach()) * batch
            if (batch_index + 1) % 100 == 0:
                LOG.info("%s epoch %d/%d, batch %d/%d, loss %.5f", stage, epoch,
                         count_epochs, batch_index + 1, len(train_loader), loss_sum / total)
        val = validation_metrics(model, val_loader, cfg, quantized)
        row = {"stage": stage, "epoch": epoch, "global_epoch": len(history) + 1,
               "learning_rate": optimizer.param_groups[0]["lr"],
               "train_loss": loss_sum / total, "train_accuracy": correct / total,
               "validation_loss": val["loss"], "validation_accuracy": val["accuracy"],
               "validation_target_efficiency": val["target_efficiency"],
               "seconds": time.perf_counter() - started}
        history.append(row)
        csv_write(run_dir / "history.csv", history)
        score = (val["accuracy"], -val["loss"])
        if score > best:
            best = score
            checkpoint_save(checkpoint, model, cfg, epoch, val, quantized)
        scheduler.step()
        LOG.info("%s epoch %d: train %.4f, val %.4f, val target efficiency %.4f",
                 stage, epoch, correct / total, val["accuracy"], val["target_efficiency"])
    checkpoint_load(checkpoint, model)  # 下一阶段/最终测试使用验证集选出的最佳模型。


@torch.no_grad()
def evaluate(model: D2NN, loader: DataLoader, quantized: bool,
             assembly: dict[str, Any] | None = None, keep_examples: bool = False) -> dict[str, Any]:
    model.eval()
    cfg = model.cfg
    device = model.raw_phase.device
    arrays: dict[str, list[np.ndarray]] = {k: [] for k in ("powers", "input_power", "output_power", "border_power")}
    labels_all, indices_all = [], []
    examples: dict[int, dict[str, Any]] = {}
    error_examples: list[dict[str, Any]] = []
    for batch_index, (images, labels, indices) in enumerate(loader):
        result = model(images.to(device), quantized, assembly)
        for name in arrays:
            arrays[name].append(result[name].cpu().numpy())
        labels_all.append(labels.numpy())
        indices_all.append(indices.numpy())
        if keep_examples:
            start = (cfg.grid_samples - cfg.active_samples) // 2
            stop = start + cfg.active_samples
            for i, label in enumerate(labels.tolist()):
                prediction = int(result["powers"][i].argmax())
                failed = prediction != label and len(error_examples) < 10
                if label not in examples or failed:
                    item = {"image": images[i, 0].numpy(), "index": int(indices[i]), "label": label,
                            "intensity": result["intensity"][i, start:stop, start:stop].cpu().numpy(),
                            "powers": result["powers"][i].cpu().numpy(), "prediction": prediction}
                    if label not in examples:  # 每类按原测试顺序第一张，不按分类成功筛选。
                        examples[label] = item
                    if failed:
                        error_examples.append(item)
        if (batch_index + 1) % 250 == 0:
            LOG.info("评估 %d/%d batches", batch_index + 1, len(loader))
    values = {k: np.concatenate(v) for k, v in arrays.items()}
    labels = np.concatenate(labels_all)
    indices = np.concatenate(indices_all)
    powers = values["powers"]
    predictions = powers.argmax(1)
    detected = powers.sum(1)
    target = powers[np.arange(len(labels)), labels]
    norm = np.maximum(detected, 1e-20)
    p_in = np.maximum(values["input_power"], 1e-20)
    p_out = np.maximum(values["output_power"], 1e-20)
    confusion = np.zeros((10, 10), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)  # 行为真值，列为预测。
    class_counts = confusion.sum(1)
    recall = np.divide(np.diag(confusion), class_counts, out=np.zeros(10), where=class_counts > 0)
    # 所有百分比明确分母；这里报告逐样本比值的平均，而非能量先求和。
    metrics = {"samples": len(labels), "accuracy": float(np.mean(predictions == labels)),
               "mean_target_efficiency_to_input": float(np.mean(target / p_in)),
               "mean_all_detectors_efficiency_to_input": float(np.mean(detected / p_in)),
               "mean_target_fraction_within_detectors": float(np.mean(target / norm)),
               "mean_detector_fraction_of_output_window": float(np.mean(detected / p_out)),
               "mean_output_window_efficiency_to_input": float(np.mean(values["output_power"] / p_in)),
               "mean_border_fraction_of_output_window": float(np.mean(values["border_power"] / p_out)),
               "per_class_recall": recall.tolist(), "confusion_true_rows": confusion.tolist()}
    return {"metrics": metrics, "values": values, "labels": labels, "indices": indices,
            "predictions": predictions, "examples": examples, "error_examples": error_examples}


def predictions_save(path: Path, result: dict[str, Any]) -> None:
    v = result["values"]
    rows = []
    for i, (index, label, prediction) in enumerate(zip(result["indices"], result["labels"], result["predictions"])):
        row = {"official_test_index": int(index), "true_label": int(label), "prediction": int(prediction),
               "input_power_relative_m2": float(v["input_power"][i]),
               "output_window_power_relative_m2": float(v["output_power"][i])}
        row.update({f"detector_{k}_power_relative_m2": float(v["powers"][i, k]) for k in range(10)})
        rows.append(row)
    csv_write(path, rows)


def figure_save(fig: Any, path: Path) -> None:
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def detector_outlines(ax: Any, cfg: Config) -> None:
    side = cfg.detector_width_um / 1000
    for label, (x, y) in enumerate(CLASS_CENTERS_MM):
        ax.add_patch(Rectangle((x - side / 2, y - side / 2), side, side,
                               fill=False, edgecolor="cyan", linewidth=0.8))
        ax.text(x, y - side / 2 - 0.05, str(label), color="cyan", ha="center", fontsize=7)


def make_figures(cfg: Config, run_dir: Path, history: list[dict[str, Any]],
                 result: dict[str, Any], phases: np.ndarray, continuous_phases: np.ndarray) -> None:
    folder = run_dir / "figures"
    folder.mkdir()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    x = [r["global_epoch"] for r in history]
    axes[0].plot(x, [r["train_loss"] for r in history], label="train")
    axes[0].plot(x, [r["validation_loss"] for r in history], label="validation")
    axes[0].set_ylabel("Loss")
    axes[1].plot(x, [r["train_accuracy"] for r in history], label="train")
    axes[1].plot(x, [r["validation_accuracy"] for r in history], label="validation")
    axes[1].set_ylabel("Accuracy")
    axes[2].plot(x, [r["validation_target_efficiency"] for r in history])
    axes[2].set_ylabel("Target power / input power")
    for ax in axes:
        ax.axvline(cfg.epochs + 0.5, color="gray", linestyle="--", label="QAT start")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.tight_layout()
    figure_save(fig, folder / "training_curves.png")
    confusion = np.array(result["metrics"]["confusion_true_rows"])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(confusion, cmap="Blues")
    for y in range(10):
        for x0 in range(10):
            ax.text(x0, y, str(confusion[y, x0]), ha="center", va="center", fontsize=7,
                    color="white" if confusion[y, x0] > confusion.max() / 2 else "black")
    ax.set(xticks=range(10), yticks=range(10), xlabel="Predicted digit", ylabel="True digit",
           title=f"{cfg.phase_levels}-level test accuracy: {result['metrics']['accuracy']:.2%}")
    fig.colorbar(im, ax=ax)
    figure_save(fig, folder / "confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(10), result["metrics"]["per_class_recall"])
    ax.set(xticks=range(10), ylim=(0, 1), xlabel="Digit", ylabel="Recall")
    figure_save(fig, folder / "per_class_recall.png")
    detected = result["values"]["powers"]
    normalized = detected / np.maximum(detected.sum(1, keepdims=True), 1e-20)
    energy = np.stack([normalized[result["labels"] == c].mean(0)
                       if np.any(result["labels"] == c) else np.zeros(10) for c in range(10)])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(energy, vmin=0, vmax=1, cmap="magma")
    ax.set(xticks=range(10), yticks=range(10), xlabel="Detector", ylabel="True digit",
           title="Mean detector power fraction (row normalized)")
    fig.colorbar(im, ax=ax)
    figure_save(fig, folder / "detector_energy_matrix.png")
    width = cfg.aperture_m * 1e3
    extent = (-width / 2, width / 2, width / 2, -width / 2)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set(xlim=(-width / 2, width / 2), ylim=(width / 2, -width / 2),
           xlabel="x (mm)", ylabel="y (mm)", title="3/4/3 detector geometry")
    ax.set_facecolor("black")
    ax.set_aspect("equal")
    detector_outlines(ax, cfg)
    figure_save(fig, folder / "detector_geometry.png")
    fig, axes = plt.subplots(cfg.layers, 2, figsize=(8, 3.1 * cfg.layers), squeeze=False)
    height_um = phases * cfg.wavelength_m / (2 * math.pi * (silica_index(cfg.wavelength_m) - 1)) * 1e6
    for layer in range(cfg.layers):
        im = axes[layer, 0].imshow(phases[layer], cmap="twilight", vmin=0,
                                   vmax=cfg.phase_range_pi * math.pi, extent=extent)
        axes[layer, 0].set_title(f"Layer {layer + 1}: phase (rad)")
        fig.colorbar(im, ax=axes[layer, 0])
        im = axes[layer, 1].imshow(height_um[layer], cmap="viridis", extent=extent)
        axes[layer, 1].set_title(f"Layer {layer + 1}: relief height (um)")
        fig.colorbar(im, ax=axes[layer, 1])
    fig.tight_layout()
    figure_save(fig, folder / "phase_and_height.png")
    fig, axes = plt.subplots(1, cfg.layers, figsize=(3.2 * cfg.layers, 3.5), squeeze=False)
    for layer in range(cfg.layers):
        im = axes[0, layer].imshow(continuous_phases[layer], cmap="twilight", vmin=0,
                                    vmax=cfg.phase_range_pi * math.pi, extent=extent)
        axes[0, layer].set_title(f"Continuous layer {layer + 1}")
        fig.colorbar(im, ax=axes[0, layer], fraction=0.046)
    figure_save(fig, folder / "continuous_phase_layers.png")
    phase_step = (2 * math.pi / cfg.phase_levels if cfg.phase_range_pi == 2.0
                  else math.pi / (cfg.phase_levels - 1))
    fig, axes = plt.subplots(1, cfg.layers, figsize=(3.2 * cfg.layers, 3.5), squeeze=False)
    for layer in range(cfg.layers):
        codes = np.rint(phases[layer] / phase_step).astype(np.int64)
        axes[0, layer].bar(range(cfg.phase_levels), np.bincount(codes.ravel(), minlength=cfg.phase_levels))
        axes[0, layer].set(title=f"Layer {layer + 1} level counts", xlabel="Phase code")
    fig.tight_layout()
    figure_save(fig, folder / "phase_level_histogram.png")
    examples = sorted(result["examples"].items())
    if examples:
        fig, axes = plt.subplots(len(examples), 3, figsize=(11, 2.8 * len(examples)), squeeze=False)
        for row, (label, item) in enumerate(examples):
            axes[row, 0].imshow(item["image"], cmap="gray", vmin=0, vmax=1)
            axes[row, 0].set_title(f"Test index {item['index']}: true {label}")
            axes[row, 0].axis("off")
            intensity = item["intensity"]
            axes[row, 1].imshow(intensity / max(float(intensity.max()), 1e-20),
                                 cmap="inferno", extent=extent, vmin=0, vmax=1)
            detector_outlines(axes[row, 1], cfg)
            axes[row, 1].set_title(f"Output / own maximum; predicted {item['prediction']}")
            powers = item["powers"] / max(float(item["powers"].sum()), 1e-20)
            axes[row, 2].bar(range(10), powers)
            axes[row, 2].set(xticks=range(10), ylim=(0, 1), ylabel="Power / all ROI power")
        fig.tight_layout()
        figure_save(fig, folder / "classification_examples.png")
    errors = result["error_examples"]
    if errors:
        fig, axes = plt.subplots(len(errors), 2, figsize=(7, 2.6 * len(errors)), squeeze=False)
        for row, item in enumerate(errors):
            axes[row, 0].imshow(item["image"], cmap="gray", vmin=0, vmax=1)
            axes[row, 0].set_title(f"Index {item['index']}, true {item['label']}, predicted {item['prediction']}")
            axes[row, 0].axis("off")
            intensity = item["intensity"]
            axes[row, 1].imshow(intensity / max(float(intensity.max()), 1e-20),
                                 cmap="inferno", extent=extent, vmin=0, vmax=1)
            detector_outlines(axes[row, 1], cfg)
        fig.tight_layout()
        figure_save(fig, folder / "first_classification_errors.png")
    else:
        (folder / "no_errors_in_evaluated_set.txt").write_text(
            "本次实际评估集合没有误分类；未制造失败案例图。", encoding="utf-8")
    fig, ax = plt.subplots(figsize=(12, 3))
    z_mm = cfg.input_gap_mm
    ax.axvline(0, color="black")
    ax.text(0, 0.75, "Input", ha="center")
    for layer in range(cfg.layers):
        ax.axvspan(z_mm, z_mm + cfg.substrate_mm, color="skyblue", alpha=0.9)
        z_mm += cfg.substrate_mm
        ax.axvline(z_mm, color="navy")
        ax.text(z_mm, 0.75, f"L{layer + 1}", ha="center")
        z_mm += cfg.layer_gap_mm if layer + 1 < cfg.layers else cfg.output_gap_mm
    ax.axvline(z_mm, color="darkred")
    ax.text(z_mm, 0.75, "Detector", ha="center")
    ax.set(xlabel="Axial coordinate (mm), substrate not to scale", yticks=[], ylim=(0, 1),
           title=f"Total nominal length {z_mm:.3f} mm; all air gaps > 0")
    figure_save(fig, folder / "optical_layout.png")


def mask_svg(path: Path, bits: np.ndarray, pitch_um: float, title: str) -> None:
    """SVG为1:1物理尺寸，黑色区域表示该道需要刻蚀，须由加工厂确认胶极性。

    坐标：左上原点，x向右、y向下；不在有效口径内擅加对准标记。
    实际GDS、套刻标记、工艺补偿和版图DRC应由加工方完成。
    """
    height, width = bits.shape
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width * pitch_um / 1000:.9f}mm" '
             f'height="{height * pitch_um / 1000:.9f}mm" viewBox="0 0 {width * pitch_um} {height * pitch_um}">',
             f"<title>{title}</title>", '<g fill="black" stroke="none">']
    # 同行连续区域合并为矩形，减少文件体积；所有边界仍位于20µm单元格上。
    for row in range(height):
        padded = np.pad(bits[row].astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts, stops = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
        for left, right in zip(starts, stops):
            parts.append(f'<rect x="{left * pitch_um:.6f}" y="{row * pitch_um:.6f}" '
                         f'width="{(right - left) * pitch_um:.6f}" height="{pitch_um:.6f}"/>')
    parts.extend(["</g>", "</svg>"])
    path.write_text("\n".join(parts), encoding="utf-8")


def fabrication_export(cfg: Config, run_dir: Path, phases: np.ndarray) -> dict[str, Any]:
    folder = run_dir / "fabrication"
    folder.mkdir()
    index = silica_index(cfg.wavelength_m)
    h_per_rad = cfg.wavelength_m / (2 * math.pi * (index - 1))
    phase_step = (2 * math.pi / cfg.phase_levels if cfg.phase_range_pi == 2.0
                  else math.pi / (cfg.phase_levels - 1))
    height_step_m = phase_step * h_per_rad
    codes = np.rint(phases / phase_step).astype(np.uint16)
    max_height = (cfg.phase_levels - 1) * height_step_m
    height_m = codes * height_step_m
    # 向下刻蚀是负相位。反码深度与正浮雕模型仅差空间常数相位。
    etch_codes = cfg.phase_levels - 1 - codes
    etch_depth_m = etch_codes * height_step_m
    np.savez_compressed(folder / "all_layers.npz", phase_rad=phases, relief_height_m=height_m,
                        etch_depth_m=etch_depth_m, phase_codes=codes, etch_codes=etch_codes,
                        pitch_m=cfg.pitch_um * 1e-6)
    n_bits = int(math.log2(cfg.phase_levels))
    for layer in range(cfg.layers):
        prefix = folder / f"layer_{layer + 1:02d}"
        np.savetxt(str(prefix) + "_phase_rad.csv", phases[layer], delimiter=",", fmt="%.9g")
        np.savetxt(str(prefix) + "_height_um.csv", height_m[layer] * 1e6, delimiter=",", fmt="%.9g")
        np.savetxt(str(prefix) + "_etch_um.csv", etch_depth_m[layer] * 1e6, delimiter=",", fmt="%.9g")
        for bit in range(n_bits):
            depth_um = height_step_m * (1 << bit) * 1e6
            mask_svg(folder / f"layer_{layer + 1:02d}_etch_bit_{bit}.svg",
                     ((etch_codes[layer] >> bit) & 1).astype(bool), cfg.pitch_um,
                     f"Layer {layer + 1}, bit {bit}, etch {depth_um:.6f} um; black = etch")
    metadata = {"wavelength_nm": cfg.wavelength_nm, "n_silica_design": index,
                "phase_range_pi": cfg.phase_range_pi, "levels": cfg.phase_levels,
                "pitch_um": cfg.pitch_um, "height_step_um": height_step_m * 1e6,
                "max_relief_um": max_height * 1e6,
                "etch_pass_depths_um": [height_step_m * (1 << bit) * 1e6 for bit in range(n_bits)],
                "minimum_remaining_substrate_mm": cfg.substrate_mm,
                "nominal_unetched_thickness_mm": cfg.substrate_mm + max_height * 1e3,
                "coordinate_convention": "top-left origin; x right, y down; preserve beam-view orientation",
                "mask_polarity": "black = requested etched area; resist polarity must be confirmed",
                "release_status": "DESIGN_EXPORT_ONLY: convergence and measured process approval required"}
    json_write(folder / "fabrication_metadata.json", metadata)
    return metadata


def input_masks_export(cfg: Config, run_dir: Path, dataset: Digits) -> None:
    """每类前五张测试数字的实体输入版图；不按数值分类是否成功筛选。

    和模型使用完全相同的nearest放大、二值化及补零。黑色代表不透光膜，
    白色/未绘制区代表透明笔画；SVG边界外的载片必须继续挡光。
    """
    folder = run_dir / "fabrication" / "input_masks"
    folder.mkdir()
    rows = []
    labels = dataset.labels.numpy()
    for label in range(10):
        for position in np.flatnonzero(labels == label)[:5]:
            image, _, original_index = dataset[int(position)]
            binary = (F.interpolate(image[None], size=(cfg.input_neurons,) * 2,
                                    mode="nearest")[0, 0] >= cfg.binary_threshold)
            padded = center_embed(binary.float(), cfg.neurons).numpy().astype(bool)
            filename = f"digit_{label}_test_{int(original_index):05d}_opaque.svg"
            mask_svg(folder / filename, ~padded, cfg.pitch_um,
                     f"MNIST {int(original_index)}, true {label}; black=opaque; unpainted=transmitting")
            rows.append({"file": filename, "true_label": label,
                         "official_test_index": int(original_index),
                         "selection": "first five of each class, without filtering by predictions"})
    csv_write(folder / "input_mask_manifest.csv", rows)
    (folder / "POLARITY.txt").write_text(
        "输入版图极性：黑色是不透光区，未绘制部分是透光笔画。\n"
        "有效SVG尺寸为整个4mm口径（随配置改变）；载片外框也须遮光。\n"
        "这与DOE刻蚀掩模的黑色=刻蚀含义不同。图案位于输入载片下游出射面。\n"
        "所有样本按真值类别与原始测试顺序选择，包含可能被模型误分类的样本。\n",
        encoding="utf-8")


def fixed_subset(dataset: Digits, samples: int, seed: int) -> Digits:
    # 每类近似等额抽样，无成功/失败筛选，且始终固定同一批样本。
    rng = np.random.default_rng(seed)
    labels = dataset.labels.numpy()
    groups = [rng.permutation(np.flatnonzero(labels == c)).tolist() for c in range(10)]
    selected: list[int] = []
    while len(selected) < min(samples, len(dataset)):
        added = False
        for group in groups:
            if group and len(selected) < min(samples, len(dataset)):
                selected.append(group.pop())
                added = True
        if not added:
            break
    local = np.array(selected)
    # 构建时保持原official index，不把子集序号冒充数据集编号。
    subset = Digits(dataset.images.numpy(), dataset.labels.numpy(), local)
    subset.indices = dataset.indices[torch.as_tensor(local)]
    return subset


def make_assembly(model: D2NN, seed: int, lateral: bool, axial: bool, height: bool) -> dict[str, Any]:
    """一套固定加工/装配误差；整个数据子集共用，模拟静态实体装置。

    高度噪声为独立像元最终高度误差，非逐曝光误差；轴向噪声按各空气间隔定义。
    """
    cfg = model.cfg
    rng = np.random.default_rng(seed)
    # 无条件抽取各类误差，开关只决定是否施加，保证同seed的单项/组合配对。
    shifts = rng.normal(0, cfg.lateral_sigma_um * 1e-6, size=(cfg.layers, 2))
    if not lateral:
        shifts.fill(0.0)
    gaps = np.array([cfg.input_gap_mm] + [cfg.layer_gap_mm] * (cfg.layers - 1)
                    + [cfg.output_gap_mm]) * 1e-3
    gap_errors = rng.normal(0, cfg.axial_sigma_um * 1e-6, len(gaps))
    if axial:
        gaps += gap_errors
    if np.any(gaps <= 0):
        raise ValueError("误差抽样产生非正空气间隔，须减小误差或增大设计距离。")
    height_error = rng.normal(0, cfg.height_sigma_nm * 1e-9,
                              size=(cfg.layers, cfg.neurons, cfg.neurons))
    if not height:
        height_error.fill(0.0)
    phase_error = 2 * math.pi * (model.n_silica - 1) / cfg.wavelength_m * height_error
    device = model.raw_phase.device
    return {"shifts": [tuple(map(float, xy)) for xy in shifts],
            "air_gaps_m": gaps.tolist(),
            "air_kernels": [angular_kernel(cfg, float(z), 1.0, device) for z in gaps],
            "phase_error": torch.as_tensor(phase_error, dtype=torch.float32, device=device)}


def robustness_analysis(model: D2NN, dataset: Digits, cfg: Config, run_dir: Path) -> dict[str, Any]:
    subset = fixed_subset(dataset, cfg.robust_samples, cfg.seed + 1000)
    loader = make_loader(subset, cfg)
    nominal = evaluate(model, loader, True)["metrics"]
    rows: list[dict[str, Any]] = []
    summaries = {}
    for case, use_lateral, use_axial, use_height in (
        ("lateral", True, False, False), ("axial", False, True, False),
        ("height", False, False, True), ("combined", True, True, True),
    ):
        accuracies = []
        for trial in range(cfg.robust_trials):
            seed = cfg.seed + 2000 + trial  # 各case共用相关随机种子便于比较。
            assembly = make_assembly(model, seed, use_lateral, use_axial, use_height)
            metric = evaluate(model, loader, True, assembly)["metrics"]
            row = {"case": case, "trial": trial, "seed": seed,
                   "samples": len(subset), "accuracy": metric["accuracy"],
                   "nominal_subset_accuracy": nominal["accuracy"],
                   "accuracy_drop": nominal["accuracy"] - metric["accuracy"],
                   "mean_target_efficiency_to_input": metric["mean_target_efficiency_to_input"],
                   "lateral_sigma_um": cfg.lateral_sigma_um if use_lateral else 0,
                   "air_gap_sigma_um": cfg.axial_sigma_um if use_axial else 0,
                   "height_sigma_nm": cfg.height_sigma_nm if use_height else 0,
                   "sampled_air_gaps_mm": json.dumps([z * 1e3 for z in assembly["air_gaps_m"]]),
                   "sampled_shifts_um": json.dumps([[x * 1e6, y * 1e6] for x, y in assembly["shifts"]])}
            rows.append(row)
            accuracies.append(metric["accuracy"])
            del assembly
            LOG.info("容差 %s trial %d: %.4f", case, trial + 1, metric["accuracy"])
        summaries[case] = {"mean_accuracy": float(np.mean(accuracies)),
                           "std_between_assemblies": float(np.std(accuracies)),
                           "minimum_accuracy": min(accuracies)}
    csv_write(run_dir / "robustness.csv", rows)
    json_write(run_dir / "robustness_subset_indices.json", subset.indices.tolist())
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(summaries)
    ax.bar(names, [summaries[k]["mean_accuracy"] for k in names],
           yerr=[summaries[k]["std_between_assemblies"] for k in names], capsize=5)
    ax.axhline(nominal["accuracy"], linestyle="--", color="black", label="Nominal same subset")
    ax.set(ylim=(0, 1), ylabel="Accuracy", title=f"Static assembly errors, n={len(subset)}, {cfg.robust_trials} trials")
    ax.legend()
    figure_save(fig, run_dir / "figures" / "robustness.png")
    return {"samples": len(subset), "trials": cfg.robust_trials, "nominal": nominal,
            "cases": summaries, "spatial_sampling_um": cfg.dx_m * 1e6,
            "interpretation": "small fixed subset; point-sampled shifted pixel edges; refine grid before mechanical tolerance claims"}


def convergence_analysis(model: D2NN, dataset: Digits, cfg: Config, run_dir: Path) -> dict[str, Any]:
    """同一制造相位在不同网格上重新前向，不重训、不把重训结果当数值收敛。"""
    subset = fixed_subset(dataset, cfg.convergence_samples, cfg.seed + 3000)
    loader = make_loader(subset, cfg)
    baseline = evaluate(model, loader, True)
    rows = []

    def comparison_row(name: str, base: dict[str, Any], result: dict[str, Any],
                       reference_cfg: Config, assembly_name: str) -> dict[str, Any]:
        # 相对于同一掩模后输入能量的实际效率，保留绝对收光变化。
        base_eff = base["values"]["powers"] / np.maximum(base["values"]["input_power"][:, None], 1e-20)
        efficiency = result["values"]["powers"] / np.maximum(result["values"]["input_power"][:, None], 1e-20)
        relative_l1 = float(np.mean(np.sum(np.abs(efficiency - base_eff), axis=1)
                                   / np.maximum(base_eff.sum(1), 1e-12)))
        agreement = float(np.mean(result["predictions"] == base["predictions"]))
        accuracy_difference = abs(result["metrics"]["accuracy"] - base["metrics"]["accuracy"])
        passed = (relative_l1 <= cfg.convergence_power_tolerance
                  and accuracy_difference <= cfg.convergence_accuracy_tolerance
                  and 1 - agreement <= cfg.convergence_accuracy_tolerance)
        return {"comparison": name, "assembly": assembly_name,
                "oversample": reference_cfg.oversample, "padding": reference_cfg.padding,
                "grid_samples": reference_cfg.grid_samples, "samples": len(subset),
                "baseline_accuracy": base["metrics"]["accuracy"],
                "reference_accuracy": result["metrics"]["accuracy"],
                "absolute_accuracy_difference": accuracy_difference, "prediction_agreement": agreement,
                "mean_relative_detector_efficiency_l1": relative_l1,
                "baseline_border_fraction": base["metrics"]["mean_border_fraction_of_output_window"],
                "reference_border_fraction": result["metrics"]["mean_border_fraction_of_output_window"],
                "passed_configured_thresholds": passed}

    configs = (("finer_sampling", replace(cfg, oversample=cfg.convergence_oversample)),
               ("larger_window", replace(cfg, padding=cfg.convergence_padding)))
    device = model.raw_phase.device
    for name, reference_cfg in configs:
        reference = D2NN(reference_cfg).to(device)
        with torch.no_grad():
            reference.raw_phase.copy_(model.raw_phase)
        result = evaluate(reference, make_loader(subset, reference_cfg), True)
        rows.append(comparison_row(name, baseline, result, reference_cfg, "nominal"))
        if name == "finer_sampling":
            # 名义模型的收敛不能代表横移边界也收敛；用完全相同实体误差再比较。
            assembly_seed = cfg.seed + 2000
            base_assembly = make_assembly(model, assembly_seed, True, True, True)
            base_perturbed = evaluate(model, loader, True, base_assembly)
            del base_assembly
            reference_assembly = make_assembly(reference, assembly_seed, True, True, True)
            reference_perturbed = evaluate(reference, make_loader(subset, reference_cfg), True, reference_assembly)
            rows.append(comparison_row("shifted_finer_sampling", base_perturbed, reference_perturbed,
                                       reference_cfg, f"combined_fixed_seed_{assembly_seed}"))
            del reference_assembly, base_perturbed, reference_perturbed
        LOG.info("网格比较 %s: %s", name, rows[-1])
        del reference, result
        if device.type == "cuda":
            torch.cuda.empty_cache()
    csv_write(run_dir / "convergence.csv", rows)
    json_write(run_dir / "convergence_subset_indices.json", subset.indices.tolist())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([r["comparison"] for r in rows], [r["mean_relative_detector_efficiency_l1"] for r in rows])
    ax.axhline(cfg.convergence_power_tolerance, color="red", linestyle="--", label="Configured tolerance")
    ax.set(ylabel="Mean relative L1 of detector efficiencies", title="Fixed phase masks; no retraining")
    ax.legend()
    figure_save(fig, run_dir / "figures" / "numerical_convergence.png")
    return {"samples": len(subset), "all_comparisons_passed": all(r["passed_configured_thresholds"] for r in rows),
            "rows": rows, "limitation": "subset-based engineering screen; not a proof of continuum convergence"}


def run_report(cfg: Config, run_dir: Path, metrics: dict[str, Any], fabrication: dict[str, Any]) -> None:
    final = metrics["quantized_test"]
    convergence = metrics["convergence"]
    status = ("通过所设子集数值筛查，仍须材料/工艺及装调实测确认"
              if convergence and convergence["all_comparisons_passed"] else "未通过或未完成数值收敛筛查，不应据此直接加工")
    full = not any((cfg.train_limit, cfg.validation_limit, cfg.test_limit))
    text = f"""# 本次运行报告

这是脚本运行后由实际计算写出的报告，不是交付时预设的实验结果。

- 数据范围：{'完整55k/5k/10k' if full else '用户限制了样本数，属于调试运行'}。
- 连续相位测试准确率：{metrics['continuous_test']['accuracy']:.4%}。
- 最佳验证集{cfg.phase_levels}级量化模型测试准确率：{final['accuracy']:.4%}，测试样本{final['samples']}。
- 平均目标区/掩模后输入能量：{final['mean_target_efficiency_to_input']:.4%}。
- 平均全部探测区/掩模后输入能量：{final['mean_all_detectors_efficiency_to_input']:.4%}。
- 折射率设计值：{fabrication['n_silica_design']:.8f}；最大浮雕高度{fabrication['max_relief_um']:.6f} µm。
- 收敛状态：{status}。
- 原论文91.75%来自750µm波长系统，不能保证迁移后保持该数值。

模型选择只使用验证集。连续与量化模型均在相应训练完成后才进行测试；
容差与网格子集仅用于固定模型诊断，不将测试诊断结果反向用于选择checkpoint。
若依据这些诊断调整超参数，应将最终无偏性能留给新的独立实验数据。

模型采用单程标量传播，包含空气、均匀基板、薄浮雕、前后有限孔径和常数表面损耗。
未包含多次相干反射、偏振/矢量边缘散射、AR膜层细节、表面粗糙度及相机噪声。
默认每面0.5%反射只是采购设计假定；实际浮雕镀膜与材料批次必须标定。
输出功率为相对输入场单位下的面积积分，并非瓦特；绝对功率需要实验标定。

完整数据、各类召回与误差诊断见metrics.json；逐图结果见predictions.csv。
fabrication中的SVG只表示理想刻蚀区域，未包含套刻标记、工艺偏置或厂家版图审查。
"""
    (run_dir / "run_report.md").write_text(text, encoding="utf-8")


def parse_args() -> Config:
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 同一个脚本通过参数调整，无需修改多个阶段脚本。
    integer_options = ("neurons", "input_neurons", "layers", "phase_levels", "oversample", "padding",
                       "epochs", "qat_epochs", "batch_size", "eval_batch_size", "seed", "workers",
                       "robust_samples", "robust_trials", "convergence_samples", "convergence_oversample",
                       "convergence_padding", "train_limit", "validation_limit", "test_limit")
    float_options = ("wavelength_nm", "pitch_um", "binary_threshold", "beam_waist_mm", "input_gap_mm",
                     "layer_gap_mm", "output_gap_mm", "substrate_mm", "detector_width_um", "phase_range_pi",
                     "surface_reflectance", "absorption_per_m", "learning_rate", "phase_init_std",
                     "qat_lr_factor", "efficiency_weight", "capture_weight", "leakage_weight",
                     "baseline_efficiency_weight", "lateral_sigma_um", "axial_sigma_um", "height_sigma_nm",
                     "convergence_power_tolerance", "convergence_accuracy_tolerance")
    for name in integer_options:
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=getattr(defaults, name))
    for name in float_options:
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=getattr(defaults, name))
    parser.add_argument("--data-dir", default=defaults.data_dir)
    parser.add_argument("--loss-mode", choices=("baseline", "joint"), default=defaults.loss_mode)
    parser.add_argument("--output-root", default=defaults.output_root)
    parser.add_argument("--device", default="auto", help="auto、cpu、cuda或cuda:0等torch设备")
    parser.add_argument("--download", action="store_true", help="允许下载缺失MNIST，下载位置限本目录")
    parser.add_argument("--skip-robustness", action="store_true", help="跳过容差诊断，报告会记录")
    parser.add_argument("--skip-convergence", action="store_true", help="跳过收敛诊断，不能据此直接加工")
    parser.add_argument("--no-gradient-checkpoint", action="store_true", help="关闭省显存重计算")
    args = vars(parser.parse_args())
    args["gradient_checkpoint"] = not args.pop("no_gradient_checkpoint")
    cfg = Config(**args)
    cfg.validate()
    return cfg


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(cfg.output_root).resolve() / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(run_dir / "run.log", encoding="utf-8")])
    json_write(run_dir / "config.json", asdict(cfg))
    device = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available()
                          else "cpu" if cfg.device == "auto" else cfg.device)
    json_write(run_dir / "environment.json", {"python": sys.version, "torch": str(torch.__version__),
               "numpy": np.__version__, "platform": platform.platform(), "device": str(device),
               "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
               "pid": os.getpid()})
    LOG.info("输出目录: %s", run_dir)
    LOG.info("1550nm方案：dx=%.3fum, grid=%dx%d, 有效口径%.3fmm",
             cfg.dx_m * 1e6, cfg.grid_samples, cfg.grid_samples, cfg.aperture_m * 1e3)
    try:
        train, validation, test = make_datasets(cfg, run_dir)
        LOG.info("数据量 train=%d validation=%d test=%d", len(train), len(validation), len(test))
        train_loader, val_loader, test_loader = (make_loader(train, cfg, True),
                                                make_loader(validation, cfg), make_loader(test, cfg))
        model = D2NN(cfg).to(device)
        history: list[dict[str, Any]] = []
        train_stage(model, train_loader, val_loader, cfg, run_dir, history, quantized=False)
        # 连续阶段只选验证checkpoint，暂不读取测试表现来决定QAT训练。
        continuous_raw = model.raw_phase.detach().cpu().clone()
        train_stage(model, train_loader, val_loader, cfg, run_dir, history, quantized=True)
        quantized_raw = model.raw_phase.detach().cpu().clone()
        final_test = evaluate(model, test_loader, True, keep_examples=True)
        predictions_save(run_dir / "predictions.csv", final_test)
        with torch.no_grad():
            phases = model.phase(True).cpu().numpy()
            model.raw_phase.copy_(continuous_raw.to(device))
            continuous_phases = model.phase(False).cpu().numpy()
        np.savez_compressed(run_dir / "trained_phases.npz", continuous_phase_rad=continuous_phases,
                            final_quantized_phase_rad=phases)
        continuous_test = evaluate(model, test_loader, False)
        predictions_save(run_dir / "continuous_predictions.csv", continuous_test)
        with torch.no_grad():
            model.raw_phase.copy_(quantized_raw.to(device))
        make_figures(cfg, run_dir, history, final_test, phases, continuous_phases)
        fabrication = fabrication_export(cfg, run_dir, phases)
        input_masks_export(cfg, run_dir, test)
        # 先保存主结果；后续高分辨率诊断若显存不足，已完成结果仍可查看。
        metrics: dict[str, Any] = {"execution_status": "main_workflow_complete_diagnostics_pending",
                                  "continuous_test": continuous_test["metrics"],
                                  "quantized_test": final_test["metrics"],
                                  "robustness": None, "convergence": None,
                                  "fabrication_release": False}
        json_write(run_dir / "metrics.json", metrics)
        if not cfg.skip_robustness:
            metrics["robustness"] = robustness_analysis(model, test, cfg, run_dir)
            json_write(run_dir / "metrics.json", metrics)
        if not cfg.skip_convergence:
            metrics["convergence"] = convergence_analysis(model, test, cfg, run_dir)
        metrics["execution_status"] = "completed"
        metrics["fabrication_release"] = False  # 数值子集通过仍不能替代工艺与实体装调验收。
        json_write(run_dir / "metrics.json", metrics)
        run_report(cfg, run_dir, metrics, fabrication)
        LOG.info("完整流程结束。量化测试准确率 %.4f，结果 %s", final_test["metrics"]["accuracy"], run_dir)
    except Exception as exc:
        json_write(run_dir / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        LOG.exception("本次运行中断；已有日志及checkpoint保留。")
        raise


if __name__ == "__main__":
    # Windows DataLoader保护；导入文件不会开始下载、训练或出图。
    main()
