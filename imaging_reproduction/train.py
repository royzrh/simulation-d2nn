"""成像复现的完整训练代码：光学模型、数据读取、训练与验证集中在此文件。

修改层数、波长、传播距离、学习率、损失函数或训练循环，均可在本文件中完成。
结构按以下顺序排列：一、光学模型；二、数据读取；三、训练与检查点保存。

新训练默认使用实验室波长 1550 nm；续训默认继承检查点波长。
--wavelength-nm 可显式指定波长，仍需符合项目的波长控制设置。
--name 指定实验名称，结果保存到 results/<名称>/；--epochs 是总轮数。
--resume 恢复原实验；--continue-from 在新目录延长训练。
训练不会使用测试集选模；完整评估和仿真图由 experiments/ 中的流程调用独立工具生成。
查看所有参数：python train.py --help。仅导入本模块不会启动训练。
"""


# 一、光学模型：物理参数、相位层、角谱传播和评价指标。
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Geometry:
    neurons: int = 300             # 每层横、纵方向各 300 个物理像元，并非像元总数。
    pitch_m: float = 0.0003        # 物理像元间距，单位米；口径边长 = neurons × pitch_m。
    frequency_hz: float = 0.4e12   # 论文参考频率；新训练在 main() 中默认覆盖为 1550 nm，旧模型评估保留原定义。
    layers: int = 5
    input_mm: float = 4.0         # 输入面到第一层相位板的距离。
    interlayer_mm: float = 4.0    # 相邻两层相位板之间的距离。
    output_mm: float = 7.0        # 最后一层相位板到探测面的距离。
    oversample: int = 2           # 每个物理像元沿每个方向细分的计算网格数量。
    pad_factor: float = 2         # FFT 网格相对有效口径的扩展倍数，用于减小周期边界影响。
    init_std: float = 0.7         # 未映射相位参数的初始标准差，不直接等于相位弧度标准差。

    def distances_mm(self):
        # N 层相位板共有 N+1 段传播：输入段、N−1 个层间段、输出段。
        return [self.input_mm] + [self.interlayer_mm] * (self.layers-1) + [self.output_mm]


class ImagingD2NN(nn.Module):
    def __init__(self, geometry: Geometry):
        super().__init__()
        self.g = geometry
        g = geometry
        if min(g.neurons, g.layers, g.oversample) < 1 or g.pad_factor <= 1:
            raise ValueError('Positive dimensions and pad_factor>1 required; padding needs convergence verification')
        if min(g.distances_mm()) < 0:
            raise ValueError('Negative forward distance is not supported')
        # 有效口径用更细的网格计算；增加采样数不会改变物理口径或像元尺寸。
        self.n = g.neurons * g.oversample
        self.fft_n = round(self.n * g.pad_factor)
        self.dx = g.pitch_m / g.oversample
        # 将有效口径放在 FFT 网格中央；左右补零数量允许相差一个采样点。
        self.left = (self.fft_n-self.n)//2
        self.right = self.fft_n-self.n-self.left
        self.s = slice(self.left,self.left+self.n)
        # 优化器更新的是 raw_phases；实际透射相位在 mask() 中映射到 0～2π。
        self.raw_phases = nn.Parameter(g.init_std * torch.randn(g.layers,g.neurons,g.neurons))
        aperture = torch.zeros(self.fft_n,self.fft_n)
        aperture[self.s,self.s] = 1
        # buffer 随模型迁移到 CPU/GPU，但不参与梯度优化；这些派生量无需写入权重。
        self.register_buffer('aperture',aperture,persistent=False)
        # 空间频率单位为 1/m，顺序与 fft2 输出一致，因此传播时不需要 fftshift。
        freq = torch.fft.fftfreq(self.fft_n,d=self.dx,dtype=torch.float64)
        fy,fx = torch.meshgrid(freq,freq,indexing='ij')
        wavelength = 299792458.0/g.frequency_hz
        k=2*math.pi/wavelength
        # 角谱法纵向波数；采用复数平方根，使超出传播带宽的分量按倏逝波衰减。
        kz=k*torch.sqrt((1-(wavelength*fx)**2-(wavelength*fy)**2).to(torch.complex128))
        # 去掉统一的载波相位 kz 中的 k，不影响此处的强度读数，并减小大相位数值误差。
        self.register_buffer('kz_relative', (kz-k).to(torch.complex64),persistent=False)
        # 预先计算每段传播的传递函数 H；距离由毫米换算成米。
        for i,z in enumerate(g.distances_mm()):
            self.register_buffer(f'H_{i}',torch.exp(1j*(kz-k)*z*1e-3).to(torch.complex64),persistent=False)

    def embed(self, amplitude):
        if amplitude.shape[-2:] != (self.g.neurons,self.g.neurons):
            raise ValueError('Input must be at the physical neuron resolution')
        # 每个物理像元在细网格上保持常值；这里细分网格，不是增加可训练像元。
        a=amplitude.repeat_interleave(self.g.oversample,-2).repeat_interleave(self.g.oversample,-1)
        # 输入默认相位为零，转为复数光场后在口径外补零。
        return F.pad(a,(self.left,self.right,self.left,self.right)).to(torch.complex64)

    def propagate(self,field,z_mm=None,index=None):
        # 按 index 取缓存的固定距离传递函数，或为指定 z_mm 临时计算传递函数。
        h=getattr(self,f'H_{index}') if index is not None else torch.exp(1j*self.kz_relative*z_mm*1e-3)
        # 先将光场变换到空间频率域，乘传播传递函数，再变换回实空间。
        return torch.fft.ifft2(torch.fft.fft2(field)*h)

    def mask(self,i,absorption=False,shift_mm=(0.,0.),height_error=None):
        # 折射率和吸收系数只核对过论文太赫兹条件；1550 nm 不允许直接使用这些材料误差分支。
        if (absorption or height_error is not None) and not math.isclose(self.g.frequency_hz,0.4e12,rel_tol=1e-12):
            raise ValueError('Material absorption/height-error parameters are verified only at 0.4 THz; supply wavelength-specific material data before using these branches.')
        # sigmoid 限制相位范围；理想训练只调相位，透射振幅在有效像元内为 1。
        phase=2*math.pi*torch.sigmoid(self.raw_phases[i])
        # 用论文折射率把相位换算为板厚，供材料误差分支使用；不代表已确定 1550 nm 板厚。
        height=phase*(299792458/self.g.frequency_hz)/(2*math.pi*(1.7227-1))
        if height_error is not None:
            height=height+height_error.to(height)
            phase=height*2*math.pi*(1.7227-1)/(299792458/self.g.frequency_hz)
        # 复数透射系数 exp(iφ) 乘在入射光场上，使其相位增加 φ。
        transmission=torch.exp(1j*phase)
        if absorption:
            # 补充材料第 3 页的太赫兹材料参数：功率衰减系数 520.7177 /m，基底厚度 0.5 mm。
            # 这里作用于光场幅度，所以功率吸收系数前有 1/2。
            transmission=transmission*torch.exp(-0.5*520.7177*(height+0.0005))
        transmission=transmission.repeat_interleave(self.g.oversample,-2).repeat_interleave(self.g.oversample,-1)
        transmission=F.pad(transmission,(self.left,self.right,self.left,self.right))
        if any(shift_mm):
            # 按亚像元尺度计算横向位移，相位板范围之外的透射率为零。
            ty,tx=(2*v*1e-3/(self.dx*self.fft_n) for v in shift_mm)
            theta=transmission.real.new_tensor([[[1,0,-tx],[0,1,-ty]]])
            # 插值工具不直接处理复数，将实部、虚部分为两个通道插值后再组合。
            channels=torch.stack([transmission.real,transmission.imag])[None]
            grid=F.affine_grid(theta,channels.shape,align_corners=False)
            shifted=F.grid_sample(channels,grid,align_corners=False,padding_mode='zeros')
            transmission=torch.complex(shifted[0,0],shifted[0,1])
        return transmission

    def field(self, amplitude, absorption=False, shifts=None, height_errors=None, trace=False):
        # 输入先传播到第一层相位板，而不是直接将输入贴在第一层上。
        u=self.propagate(self.embed(amplitude),index=0)
        fields=[u] if trace else None
        for i in range(self.g.layers):
            # 训练时可用重计算换显存；trace 记录场分布时保留普通传播路径。
            if self.training and torch.is_grad_enabled() and getattr(self,'activation_checkpointing',False) and not trace:
                from torch.utils.checkpoint import checkpoint
                u=checkpoint(self.layer_step,u,i,absorption,shifts[i] if shifts else (0.,0.),
                             height_errors[i] if height_errors is not None else None,use_reentrant=False)
                continue
            # 每层先调制，再传播至下一层；最后一次传播到输出探测面。
            u=u*self.mask(i,absorption,shifts[i] if shifts else (0.,0.),height_errors[i] if height_errors is not None else None)
            if trace: fields.append(u)
            u=self.propagate(u,index=i+1)
        # trace 额外保留探测面光场，供相位图和沿传播方向的场分布分析使用。
        if trace: fields.append(u)
        return (u,fields) if trace else u

    def layer_step(self,u,index,absorption,shift,height_error):
        return self.propagate(u*self.mask(index,absorption,shift,height_error),index=index+1)

    def forward(self,amplitude,**kwargs):
        u=self.field(amplitude,**kwargs)
        # 对子像元强度积分得到物理探测像元读数，不能直接平均光场幅度。
        # 裁去补零边界，以复振幅模平方得到强度；输出尺寸最终恢复为物理像元数。
        intensity=u[:,self.s,self.s].abs().square()
        return F.avg_pool2d(intensity[:,None],self.g.oversample)[:,0]

    def free_space(self,amplitude):
        # 自由空间基线：使用相同输入和总距离，但移除所有相位板。
        u=self.propagate(self.embed(amplitude),z_mm=sum(self.g.distances_mm()))
        return F.avg_pool2d(u[:,None,self.s,self.s].abs().square(),self.g.oversample)[:,0]


def ssim(x,y):
    """Wang et al. Gaussian-window SSIM, data_range=1; per image, valid pixels.

    This is a documented implementation choice; original SSIM code/settings unknown.
    """
    if x.ndim==3: x=x[:,None]; y=y[:,None]
    # SSIM 使用 11×11、标准差 1.5 的归一化高斯窗口；这里不是训练损失。
    coord=torch.arange(11,device=x.device,dtype=x.dtype)-5
    w=torch.exp(-coord.square()/(2*1.5**2)); w=w/w.sum()
    kernel=(w[:,None]*w[None,:])[None,None]
    conv=lambda a:F.conv2d(a,kernel)
    # 计算局部均值、方差及协方差；不补边卷积，只统计完整窗口覆盖的位置。
    mx,my=conv(x),conv(y)
    vx,vy=conv(x*x)-mx*mx,conv(y*y)-my*my
    cov=conv(x*y)-mx*my
    score=((2*mx*my+.01**2)*(2*cov+.03**2))/((mx*mx+my*my+.01**2)*(vx+vy+.03**2))
    return score.mean((1,2,3))


# 二、数据读取：数据来源标识、固定划分和幅度输入。
import hashlib
import json
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageOps
import torch

HERE=Path(__file__).resolve().parent


def save_json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
    tmp.replace(path)


def prepare(root,source,seed=42,size=300):
    root=Path(root).resolve()
    files=sorted(p for p in root.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'})
    # 只在数据准备时固定打乱一次；后续训练读取清单，不重新划分训练、验证和测试集。
    random.Random(seed).shuffle(files)
    chosen=[]; hashes=set(); rejected=[]
    for p in files:
        try:
            digest=hashlib.sha256(p.read_bytes()).hexdigest()
            # 按文件内容哈希去重，避免相同图片仅因文件名不同而被重复选入。
            if digest in hashes: continue
            with Image.open(p) as im:
                # 论文未完整公开缩放方式和灰度的幅度约定，此处作为明确记录的实现选择。
                a=np.asarray(ImageOps.exif_transpose(im).convert('L').resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32)/255
            chosen.append((p,digest,a)); hashes.add(digest)
        except Exception as e:
            rejected.append({'path':str(p),'error':str(e)})
        if len(chosen)==2000: break
    if len(chosen)<2000: raise ValueError(f'Need 2000 distinct decodable ImageNet images; got {len(chosen)}')
    out=HERE/'data'; out.mkdir(exist_ok=True)
    manifest={'source':source,'source_root':str(root),'seed':seed,'size':size,
              'preprocessing':'EXIF orientation; PIL L grayscale; bilinear resize; [0,1] amplitude',
              'split_counts':{'train':1500,'validation':200,'test':300},'images':[], 'rejected':rejected,
              'original_authors_image_indices_available':False}
    for i,(p,digest,a) in enumerate(chosen):
        split='train' if i<1500 else 'validation' if i<1700 else 'test'
        manifest['images'].append({'path':str(p),'sha256':digest,'split':split})
    ident=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    manifest['manifest_sha256']=ident
    target=out/'manifest.json'
    if target.exists() and json.loads(target.read_text(encoding='utf-8'))!=manifest:
        raise ValueError('Existing data manifest differs; preserve it and use a separate experiment directory')
    save_json(target,manifest)
    np.save(out/'amplitudes.npy',np.stack([x[2] for x in chosen]))
    return manifest


class ImagingDataset(torch.utils.data.Dataset):
    def __init__(self,split,input_encoding="amplitude_gray",data_dir=None):
        if input_encoding not in {"amplitude_gray","intensity_gray"}:raise ValueError(input_encoding)
        self.input_encoding=input_encoding
        self.data_dir=Path(data_dir).resolve() if data_dir is not None else HERE/'data'
        self.manifest=json.loads((self.data_dir/'manifest.json').read_text(encoding='utf-8'))
        # 重算清单内容哈希，防止数据划分或来源被改动后仍沿用原实验标识。
        canonical={key:value for key,value in self.manifest.items() if key!='manifest_sha256'}
        actual=hashlib.sha256(json.dumps(canonical,sort_keys=True).encode()).hexdigest()
        if actual!=self.manifest['manifest_sha256']:
            raise ValueError('Data manifest content no longer matches its recorded identity')
        # 内存映射按需读取数组，不将整个大样本图像缓存一次性复制到内存。
        self.a=np.load(self.data_dir/'amplitudes.npy',mmap_mode='r')
        if self.a.shape!=(len(self.manifest['images']),self.manifest['size'],self.manifest['size']):
            raise ValueError('Image cache shape differs from the data manifest')
        # 一个数据集实例只暴露指定划分的索引，验证集不会参与优化器更新。
        self.indices=[i for i,v in enumerate(self.manifest['images']) if v['split']==split]
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        # 复制当前图像，避免可写张量直接引用只读的内存映射数组。
        gray=torch.from_numpy(self.a[self.indices[i]].copy())
        # 若灰度被解释为强度，输入光场幅度取其平方根；否则灰度直接作为幅度。
        return gray.sqrt() if self.input_encoding=="intensity_gray" else gray


# 三、训练流程：参数解析、损失计算、梯度累积、验证与断点保存。
import argparse
from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch


# 输入 a 是光场幅度。默认目标强度为 |a|²；另一选项直接以 a 为目标，用于定义敏感性对照。
def target_of(a,convention):return a.square() if convention=='amplitude_squared' else a


# 先写临时文件，再替换正式检查点，降低写入中断导致原检查点损坏的风险。
def atomic_save(obj,path):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp'); torch.save(obj,tmp); tmp.replace(path)


# optimizer 存在时训练，否则仅评估；free=True 用于相同输入的自由空间传播对照。
def epoch(model,loader,device,target,optimizer=None,free=False,micro_batch_size=None):
    model.train(optimizer is not None)
    sums={'mse':0.,'ssim_intensity':0.,'ssim_amplitude':0.,'capture_ratio':0.}; n=0
    started=time.perf_counter()
    # 验证阶段关闭梯度，不更新相位参数；训练和验证共用相同的指标计算方式。
    with torch.set_grad_enabled(optimizer is not None):
        for batch_index,batch in enumerate(loader,1):
            if optimizer is not None:optimizer.zero_grad(set_to_none=True)
            # 将一个有效 batch 分为多个微批次，全部反向传播后才更新一次参数。
            for a in batch.split(micro_batch_size or len(batch)):
                a=a.to(device); y=target_of(a,target)
                pred=model.free_space(a) if free else model(a)
                # 对当前微批次的全部像素计算原始强度 MSE，不归一化每张预测图。
                loss=(pred-y).square().mean()
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                if optimizer is not None:
                    # 按微批次实际样本数加权，确保末尾不足整批时也等价于整批平均梯度。
                    (loss*(len(a)/len(batch))).backward()
                b=len(a); n+=b
                sums['mse']+=loss.item()*b
                # 同时记录强度 SSIM、幅度 SSIM 和收集功率比例；指标不会替代训练损失。
                sums['ssim_intensity']+=ssim(pred,y).sum().item()
                sums['ssim_amplitude']+=ssim(pred.clamp_min(0).sqrt(),y.sqrt()).sum().item()
                sums['capture_ratio']+=(pred.sum((-2,-1))/a.square().sum((-2,-1)).clamp_min(1e-12)).sum().item()
            if optimizer is not None:
                if not torch.isfinite(model.raw_phases.grad).all():raise FloatingPointError('Nonfinite phase gradient')
                # 每个有效 batch 仅执行一次 Adam 更新，微批次数量不改变有效 batch 大小。
                optimizer.step()
            if micro_batch_size and (batch_index==1 or batch_index%10==0 or batch_index==len(loader)):
                print(f"  {'train' if optimizer is not None else 'evaluation'} batch {batch_index}/{len(loader)} "
                      f"MSE={sums['mse']/n:.6g} elapsed={time.perf_counter()-started:.1f}s",flush=True)
    return {k:v/n for k,v in sums.items()}|{'n':n}


def main():
    # 命令行参数分别控制实验名称、几何、数值采样、训练设置和恢复方式。
    p=argparse.ArgumentParser(); p.add_argument('--name',default='baseline'); p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--layers',type=int,default=5);p.add_argument('--input-mm',type=float,default=4)
    p.add_argument('--gap-mm',type=float,default=4);p.add_argument('--output-mm',type=float,default=7)
    p.add_argument('--oversample',type=int,default=2);p.add_argument('--pad-factor',type=float,default=2)
    p.add_argument('--lr',type=float,default=.005);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--init-std',type=float,default=None)
    p.add_argument('--input-encoding',choices=['amplitude_gray','intensity_gray'],default=None)
    p.add_argument('--batch-size',type=int,default=4);p.add_argument('--device',default='cuda')
    p.add_argument('--micro-batch-size',type=int,help='Accumulate gradients across chunks; one optimizer step per batch-size images')
    p.add_argument('--activation-checkpointing',action='store_true',help='Recompute optical layers during backward to bound GPU memory')
    p.add_argument('--target',choices=['amplitude_squared','gray_intensity'],default='amplitude_squared')
    p.add_argument('--resume',type=Path);p.add_argument('--continue-from',type=Path,
                   help='Continue a shorter completed run into a NEW directory with a larger total epoch budget')
    p.add_argument('--limit-train',type=int);p.add_argument('--limit-val',type=int)
    p.add_argument('--data-dir',type=Path,help='Separate immutable dataset; default remains the original 2000-image set')
    p.add_argument('--wavelength-nm',type=float,help='Physical vacuum wavelength; new runs default to 1550 nm, resumed runs inherit checkpoint geometry')
    p.add_argument('--pitch-um',type=float,help='Physical mask pixel pitch, not numerical sampling pitch')
    a=p.parse_args()
    if a.resume and a.continue_from:p.error('--resume and --continue-from are mutually exclusive')
    # 恢复时优先继承检查点中的波长和物理像元尺寸；显式命令行值仍会接受配置一致性检查。
    reference_checkpoint=a.resume or a.continue_from
    # 新训练默认对齐实验室 1550 nm；保留 Geometry 的论文参考默认值以兼容既有模型评估。
    # 续训不套用新默认值，必须继承原模型频率，避免悄悄改变同一实验的物理条件。
    physical={} if reference_checkpoint else {'frequency_hz':299792458.0/(1550e-9)}
    if reference_checkpoint:
        saved_geometry=torch.load(reference_checkpoint,map_location='cpu',weights_only=True)['config']['geometry']
        physical={key:saved_geometry[key] for key in ('frequency_hz','pitch_m')}
    # 波长由 nm 换算为 m，再通过真空光速换为模型使用的频率；像元由 μm 换算为 m。
    if a.wavelength_nm is not None:
        if not np.isfinite(a.wavelength_nm) or a.wavelength_nm<=0:p.error('Positive finite wavelength required')
        physical['frequency_hz']=299792458.0/(a.wavelength_nm*1e-9)
    if a.pitch_um is not None:
        if not np.isfinite(a.pitch_um) or a.pitch_um<=0:p.error('Positive finite physical pitch required')
        physical['pitch_m']=a.pitch_um*1e-6
    # 读取既有队列控制设置；等待条件未解除时不会开始相应训练。
    control=HERE/'misc'/'queue_control.json'
    if control.exists():
        policy=json.loads(control.read_text(encoding='utf-8-sig'))
        requested_nm=299792458.0/physical.get('frequency_hz',Geometry().frequency_hz)*1e9
        desired_nm=policy.get('target_wavelength_nm')
        incompatible=desired_nm is not None and not np.isclose(requested_nm,desired_nm,rtol=1e-9)
        # 配置冲突直接报告错误；不在这里无限等待，且不创建结果目录或启动优化器。
        if policy.get('hold_incompatible_new_training',False) and incompatible:
            p.error(f'Wavelength mismatch: requested {requested_nm:g} nm; project requires {desired_nm:g} nm. '
                    'For a NEW experiment, use --wavelength-nm 1550. '
                    'Do not change the wavelength of an existing checkpoint to bypass this check.')
    # 大样本实验续训时继承其独立数据目录，避免误用默认 2000 张数据。
    if a.data_dir is None and reference_checkpoint:
        saved_data=torch.load(reference_checkpoint,map_location='cpu',weights_only=True)['config'].get('data_dir')
        if saved_data:a.data_dir=Path(saved_data)
    if min(a.epochs,a.batch_size,a.lr)<=0: p.error('Positive training settings required')
    if a.micro_batch_size is not None and not 1<=a.micro_batch_size<=a.batch_size:p.error('micro-batch-size must be within 1..batch-size')
    # 读取既有队列控制设置；等待条件未解除时不会开始相应训练。
    control=HERE/'misc'/'queue_control.json'
    if a.name.startswith(('layers_','gap_')) and control.exists():
        announced=False
        while json.loads(control.read_text(encoding='utf-8')).get('hold_architecture_training',False):
            if not announced:
                print('ARCHITECTURE QUEUE: waiting for validation-only initialization review; no optimizer steps started.',flush=True)
                announced=True
            time.sleep(2)
    # 初始化和输入编码优先继承原实验；新实验使用已记录的默认选择。
    if a.init_std is None:
        defaults=HERE/'misc'/'selected_defaults.json'
        if reference_checkpoint:
            a.init_std=torch.load(reference_checkpoint,map_location='cpu',weights_only=True)['config']['geometry']['init_std']
        else:
            a.init_std=json.loads(defaults.read_text(encoding='utf-8')).get('init_std',.7) if defaults.exists() else .7
    if not 0<=a.init_std<=1:p.error('init-std must be within the disclosed 0..1 range')
    if a.input_encoding is None:
        defaults=HERE/'misc'/'selected_defaults.json'
        if reference_checkpoint:
            a.input_encoding=torch.load(reference_checkpoint,map_location='cpu',weights_only=True)['config'].get('input_encoding','amplitude_gray')
        else:
            a.input_encoding=json.loads(defaults.read_text(encoding='utf-8')).get('input_encoding','amplitude_gray') if defaults.exists() else 'amplitude_gray'
    # 固定随机种子；每轮数据打乱另用 seed + epoch，便于恢复后保持轮次顺序。
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if a.device=='cpu':torch.set_num_threads(2)
    torch.backends.cudnn.benchmark=False
    device=torch.device(a.device)
    # 物理像元尺寸与 oversample 分开：前者属于器件几何，后者只是数值传播采样。
    g=Geometry(layers=a.layers,input_mm=a.input_mm,interlayer_mm=a.gap_mm,output_mm=a.output_mm,oversample=a.oversample,pad_factor=a.pad_factor,init_std=a.init_std,**physical)
    # 训练只读取 train 和 validation 划分，不用测试集挑选模型或训练参数。
    train,valid=ImagingDataset('train',a.input_encoding,a.data_dir),ImagingDataset('validation',a.input_encoding,a.data_dir)
    # 保存完整实验约定和数据清单标识；直接比较原始输出强度，不对输出拟合缩放。
    config={'geometry':asdict(g),'epochs':a.epochs,'batch_size':a.batch_size,'lr':a.lr,'seed':a.seed,
            'loss':'raw_output_intensity_MSE','target':a.target,'output_rescaling':'none',
            'manifest_sha256':train.manifest['manifest_sha256'], 'limit_train':a.limit_train,'limit_val':a.limit_val,
            'status':'partial_until_conventions_and_numerics_verified'}
    if a.input_encoding!='amplitude_gray':config['input_encoding']=a.input_encoding
    if a.data_dir is not None:config['data_dir']=str(a.data_dir.resolve())
    if a.micro_batch_size is not None:config['micro_batch_size']=a.micro_batch_size
    if a.activation_checkpointing:config['activation_checkpointing']=True
    # continue-from 仅允许延长总轮数并换到空目录，其余实验配置必须一致。
    parent_state=None
    if a.continue_from:
        parent_path=a.continue_from.resolve()
        parent_state=torch.load(parent_path,map_location='cpu',weights_only=True)
        if a.epochs<=parent_state['epoch']:p.error('Continuation must increase the total epoch budget')
        excluded={'epochs','continued_from','parent_checkpoint_sha256'}
        old={k:v for k,v in parent_state['config'].items() if k not in excluded}
        new={k:v for k,v in config.items() if k not in excluded}
        if old!=new:raise ValueError('Only epoch budget and destination may change for --continue-from')
        config['continued_from']=str(parent_path)
        config['parent_checkpoint_sha256']=hashlib.sha256(parent_path.read_bytes()).hexdigest()
    elif a.resume:
        saved_config=torch.load(a.resume,map_location='cpu',weights_only=True)['config']
        for key in ['continued_from','parent_checkpoint_sha256']:
            if key in saved_config:config[key]=saved_config[key]
    # 同名结果目录已有配置时必须匹配，并要求显式恢复，防止覆盖既有实验。
    out=HERE/'results'/a.name; out.mkdir(parents=True,exist_ok=True)
    if a.continue_from and any(out.iterdir()):raise ValueError('--continue-from requires a NEW empty destination')
    if (out/'config.json').exists():
        if json.loads((out/'config.json').read_text(encoding='utf-8'))!=config:raise ValueError('Output config mismatch')
        if a.resume is None:raise ValueError('Existing experiment: use --resume or a new --name')
    save_json(out/'config.json',config)
    # 创建相位模型和 Adam；激活检查点在反向时重算中间光场以节省显存。
    model=ImagingD2NN(g).to(device);opt=torch.optim.Adam(model.parameters(),lr=a.lr)
    model.activation_checkpointing=a.activation_checkpointing
    history=[];best=float('inf');start=1
    # 恢复模型、Adam 状态、历史、最佳验证值和已保存的 PyTorch 随机状态。
    if a.resume or parent_state is not None:
        c=parent_state if parent_state is not None else torch.load(a.resume,map_location='cpu',weights_only=True)
        if a.resume and c['config']!=config:raise ValueError('Resume config mismatch')
        model.load_state_dict(c['model']); opt.load_state_dict(c['optimizer']);history=c['history'];best=c['best'];start=c['epoch']+1
        # 这里恢复的是检查点中已保存的 CPU 随机状态，不应解读为恢复全部 CUDA 随机状态。
        torch.set_rng_state(c['rng'])
        if parent_state is not None:
            parent_best=torch.load(a.continue_from.parent/'best_model.pt',map_location='cpu',weights_only=True)
            if parent_best['config']!=parent_state['config']:raise ValueError('Parent best/last config mismatch')
            parent_best['config']=config
            atomic_save(parent_best,out/'best_model.pt')
            print(f'CONTINUE: preserving parent history/optimizer; next epoch {start}, total budget {a.epochs}',flush=True)
    # 样本数量限制仅供小规模调试；其取值已经写入配置，不能冒充完整训练。
    if a.limit_train:train=torch.utils.data.Subset(train,range(min(a.limit_train,len(train))))
    if a.limit_val:valid=torch.utils.data.Subset(valid,range(min(a.limit_val,len(valid))))
    val_dl=torch.utils.data.DataLoader(valid,batch_size=a.batch_size)
    print('START',out,'device',device,'geometry',asdict(g),'input_encoding',a.input_encoding,flush=True)
    # 每轮依次训练和验证；终端实时输出批次进度及本轮指标。
    for e in range(start,a.epochs+1):
        t=time.perf_counter()
        # 每轮顺序不同，但由种子和轮次确定；恢复到同一轮可重建该轮的打乱顺序。
        dl=torch.utils.data.DataLoader(train,batch_size=a.batch_size,shuffle=True,generator=torch.Generator().manual_seed(a.seed+e))
        tr=epoch(model,dl,device,a.target,opt,micro_batch_size=a.micro_batch_size);va=epoch(model,val_dl,device,a.target,micro_batch_size=a.micro_batch_size)
        row={'epoch':e,'seconds':time.perf_counter()-t}|{'train_'+k:v for k,v in tr.items()}|{'val_'+k:v for k,v in va.items()}
        history.append(row)
        # best_model.pt 只按验证 MSE 更新，最终测试需使用这个已选定的模型。
        if va['mse']<best:
            best=va['mse'];atomic_save({'model':model.state_dict(),'config':config,'epoch':e,'validation':va},out/'best_model.pt')
        # last_checkpoint.pt 每轮更新，包含优化器等恢复信息，与最佳模型各司其职。
        atomic_save({'model':model.state_dict(),'config':config,'epoch':e,'optimizer':opt.state_dict(),'history':history,'best':best,'rng':torch.get_rng_state()},out/'last_checkpoint.pt')
        # 覆盖写入包含全部已完成轮次的历史，便于绘制曲线和核查续训记录。
        with (out/'training_history.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(history)
        print(f"epoch {e}/{a.epochs} val_MSE={va['mse']:.6g} amp_SSIM={va['ssim_amplitude']:.4f} seconds={row['seconds']:.1f}",flush=True)
    # 仅在指定轮数全部结束后写入完成标记；这不代表测试和仿真图已经生成。
    save_json(out/'training_complete.json',{'epochs':a.epochs,'best_validation_mse':best,'test_used':False})


if __name__=='__main__':main()
