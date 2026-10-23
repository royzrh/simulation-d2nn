"""Four compact scientific figure plates per experiment; incremental export queue.

Existing full evaluations are reused. Diagnostic/pilot examples use validation
images, never test images. Optical weights and all datasets are read-only.
"""

# Allow this tool to run directly while keeping project imports rooted correctly.
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _BootstrapPath
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from train import HERE,ImagingDataset,save_json
from imaging.evaluation.evaluate import load,patterns,grid
from train import target_of

FILES=['training_and_phase.png','validation_imaging.png','resolution_and_pinhole.png','defocus.png']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finished(root):
    marker=root/'figures'/'manifest.json'
    if marker.exists():
        saved=json.loads(marker.read_text(encoding='utf-8'))
        if saved.get('renderer_version')==2 and saved['checkpoint_sha256']==sha(root/'best_model.pt') and all((root/'figures'/p).is_file() for p in FILES):return True
    evaluation=root/'evaluation'
    existing=['training_curves.png','phase_masks.png','test_000.png','resolution_chart_both_axes.png','pinhole_3mm.png']
    return all((evaluation/p).is_file() and (evaluation/p).stat().st_mtime>=(root/'best_model.pt').stat().st_mtime for p in existing)


def export(checkpoint_path,device='cuda',model=None,checkpoint=None):
    checkpoint_path=Path(checkpoint_path);root=checkpoint_path.parent
    if finished(root):return
    if model is None:model,checkpoint=load(checkpoint_path,device)
    m,c=model,checkpoint;out=root/'figures';out.mkdir(exist_ok=True)
    with (root/'training_history.csv').open(newline='',encoding='utf-8') as f:history=list(csv.DictReader(f))
    phases=(2*np.pi*torch.sigmoid(m.raw_phases.detach())).cpu().numpy();n=len(phases)
    fig=plt.figure(figsize=(14,3*(1+math.ceil(n/5))))
    layout=fig.add_gridspec(1+math.ceil(n/5),6,width_ratios=[1,1,1,1,1,.08])
    for col,key,ylabel in [(0,'mse','Intensity MSE'),(2,'ssim_amplitude','Amplitude SSIM')]:
        ax=fig.add_subplot(layout[0,col:col+2])
        for split in ['train','val']:
            ax.plot([int(r['epoch']) for r in history],[float(r[split+'_'+key]) for r in history],label=split)
        if key=='mse':ax.set_yscale('log')
        ax.axvline(c['epoch'],color='gray',linestyle='--');ax.set_xlabel('Epoch');ax.set_ylabel(ylabel);ax.legend(fontsize=8)
    info=fig.add_subplot(layout[0,4]);info.axis('off')
    info.text(0,.9,f"{n} phase masks\nWavelength {299792458/m.g.frequency_hz*1e9:g} nm\nTotal {sum(m.g.distances_mm()):g} mm\nOS {m.g.oversample}\nSelected epoch {c['epoch']}\nObserved {len(history)} epochs",va='top',fontsize=10)
    if (root/'diagnostic_status.json').exists():
        info.text(0,.05,'Stopped diagnostic\nSampling not accepted',color='darkred',fontsize=9)
    for i,phase in enumerate(phases):
        ax=fig.add_subplot(layout[1+i//5,i%5]);image=ax.imshow(phase,cmap='twilight',vmin=0,vmax=2*np.pi,extent=[-45,45,45,-45])
        ax.set_title(f'L{i+1} phase (rad)',fontsize=9);ax.set_xlabel('x (mm)');ax.set_ylabel('y (mm)')
    cax=fig.add_subplot(layout[1:,5]);bar=fig.colorbar(image,cax=cax,ticks=[0,np.pi,2*np.pi]);bar.ax.set_yticklabels(['0','pi','2pi']);bar.set_label('Phase (rad)')
    fig.tight_layout();fig.savefig(out/FILES[0],dpi=130,bbox_inches='tight');plt.close(fig)
    ds=ImagingDataset('validation',c['config'].get('input_encoding','amplitude_gray'),c['config'].get('data_dir'))
    if ds.manifest['manifest_sha256']!=c['config']['manifest_sha256']:raise ValueError('Preview dataset mismatch')
    count=min(4,c['config'].get('limit_val') or len(ds));examples=[]
    with torch.no_grad():
        for i in range(count):
            a=ds[i][None].to(device);target=target_of(a,c['config']['target'])
            examples.append([target[0].sqrt().cpu().numpy(),m(a)[0].sqrt().cpu().numpy(),m.free_space(a)[0].sqrt().cpu().numpy()])
        vmax=max(1.,max(float(a.max()) for row in examples for a in row))
        fig,axes=plt.subplots(count,3,figsize=(9,3*count),squeeze=False)
        for i,row in enumerate(examples):
            for j,array in enumerate(row):
                image=axes[i,j].imshow(array,cmap='gray',vmin=0,vmax=vmax,extent=[-45,45,45,-45])
                if i==0:axes[i,j].set_title(['Target amplitude','D2NN amplitude','Free-space amplitude'][j])
                if j==0:axes[i,j].set_ylabel(f'Validation {i+1}\ny (mm)')
                if i==count-1:axes[i,j].set_xlabel('x (mm)')
        fig.suptitle('Fixed validation examples; common raw amplitude scale; no fitted gain')
        fig.tight_layout(rect=[0,0,.96,.97]);bar=fig.add_axes([.965,.15,.015,.65]);fig.colorbar(image,cax=bar,label='a.u.')
        fig.savefig(out/FILES[1],dpi=120,bbox_inches='tight');plt.close(fig)
        if root.name.startswith('layers_n1_total30_'):
            control=HERE/'results'/root.name.replace('layers_n1_','layers_n0_',1)/'figures';control.mkdir(parents=True,exist_ok=True)
            fig,axes=plt.subplots(count,2,figsize=(6,3*count),squeeze=False)
            for i,row in enumerate(examples):
                for j,array in enumerate([row[0],row[2]]):
                    axes[i,j].imshow(array,cmap='gray',vmin=0,vmax=vmax,extent=[-45,45,45,-45])
                    if i==0:axes[i,j].set_title(['Input amplitude','Free space: 30 mm'][j])
                    axes[i,j].set_xlabel('x (mm)')
            fig.suptitle('N=0 control: no trained masks; fixed validation examples')
            fig.tight_layout();fig.savefig(control/'free_space_control.png',dpi=120);plt.close(fig)
        pp=patterns(m);chart=pp['resolution_chart_both_axes'][None]
        ideal=m(chart);free=m.free_space(chart)
        fig,axes=plt.subplots(4,3,figsize=(12,13))
        for ax,array,title in zip(axes[0],[chart[0],ideal[0].sqrt(),free[0].sqrt()],['Input chart','D2NN amplitude','Free-space amplitude']):
            ax.imshow(array.cpu().numpy(),cmap='gray',vmin=0,vmax=1,extent=[-45,45,45,-45]);ax.set_title(title);ax.set_xlabel('x (mm)')
        contrasts=[[],[]]
        for k,w in enumerate([4.2,3.6,3.,2.4,1.8,1.2]):
            cx=-23 if k%2==0 else 23;cy=-25+(k//2)*25
            for col,orientation in enumerate(['vertical','horizontal']):
                pcx=cx-8.5 if col==0 else cx+10.5
                cross=np.linspace(-3*w,3*w,241);along=np.linspace(-6 if col==0 else -4,6 if col==0 else 4,41)
                coords=[[(pcx+t,cy+d) if col==0 else (pcx+d,cy+t) for t in cross] for d in along]
                loc=ideal.new_tensor(coords)[None]/45
                values=[F.grid_sample(a[:,None],loc,align_corners=False)[0,0].mean(0).cpu().numpy() for a in [chart.square(),ideal,free]]
                peak=values[1][[40,120,200]].mean();valley=values[1][[80,160]].mean()
                contrasts[col].append(float((peak-valley)/max(peak+valley,1e-12)))
                if k==5:
                    for value,label in zip(values,['Target','D2NN','Free space']):axes[1,col].plot(cross,value,label=label)
                    axes[1,col].set_title('1.2 mm, '+orientation);axes[1,col].set_xlabel('Across bars (mm)');axes[1,col].set_ylabel('Raw intensity');axes[1,col].legend(fontsize=7)
        for values,label in zip(contrasts,['Vertical','Horizontal']):axes[1,2].plot([4.2,3.6,3.,2.4,1.8,1.2],values,'o-',label=label)
        axes[1,2].axhline(0,color='gray',linewidth=.6);axes[1,2].set_xlabel('Linewidth (mm)');axes[1,2].set_ylabel('Measured modulation');axes[1,2].legend(fontsize=7)
        xaxis=(np.arange(m.g.neurons)-(m.g.neurons-1)/2)*m.g.pitch_m*1000
        for col,diameter in enumerate([1,2,3]):
            a=pp[f'pinhole_{diameter}mm'][None];p=m(a);q=m.free_space(a)
            axes[2,col].imshow(p[0].cpu().numpy(),cmap='hot',vmin=0,vmax=1,extent=[-45,45,45,-45]);axes[2,col].set_xlim(-10,10);axes[2,col].set_ylim(10,-10);axes[2,col].set_title(f'{diameter} mm pinhole: D2NN intensity')
            for value,label in zip([a.square(),p,q],['Target','D2NN','Free space']):axes[3,col].plot(xaxis,value[0,m.g.neurons//2].cpu().numpy(),label=label)
            axes[3,col].set_xlim(-10,10);axes[3,col].set_xlabel('x (mm)');axes[3,col].set_ylabel('Raw intensity');axes[3,col].legend(fontsize=7)
        fig.tight_layout();fig.savefig(out/FILES[2],dpi=130);plt.close(fig)
        offsets=[0,3,6,9,12,24];saved=m.H_0.clone();outputs=[]
        try:
            for offset in offsets:
                m.H_0.copy_(torch.exp(1j*m.kz_relative*(m.g.input_mm+offset)*1e-3))
                a=pp['pinhole_3mm'][None];p=m(a)
                u=m.propagate(m.embed(a),z_mm=sum(m.g.distances_mm())+offset)
                q=F.avg_pool2d(u[:,None,m.s,m.s].abs().square(),m.g.oversample)[:,0]
                outputs.append([p[0].cpu().numpy(),q[0].cpu().numpy()])
        finally:m.H_0.copy_(saved)
        fig,axes=plt.subplots(2,6,figsize=(15,5))
        for col,(offset,arrays) in enumerate(zip(offsets,outputs)):
            for row,array in enumerate(arrays):
                axes[row,col].imshow(array,cmap='hot',vmin=0,vmax=1,extent=[-45,45,45,-45]);axes[row,col].set_xlim(-12,12);axes[row,col].set_ylim(12,-12)
                axes[row,col].set_title(f'Input +{offset} mm',fontsize=9)
                if col==0:axes[row,col].set_ylabel(['D2NN','Free space'][row]+'\ny (mm)')
                if row==1:axes[row,col].set_xlabel('x (mm)')
        fig.suptitle('3 mm pinhole; uniform illumination; raw intensity; no fabrication errors')
        fig.tight_layout();fig.savefig(out/FILES[3],dpi=130);plt.close(fig)
    for name in FILES:
        with Image.open(out/name) as im:im.verify()
    save_json(out/'manifest.json',{'renderer_version':2,'checkpoint_sha256':sha(checkpoint_path),'dataset_manifest_sha256':ds.manifest['manifest_sha256'],
        'images':FILES,'examples_split':'validation','example_indices':list(range(count)),
        'output_rescaling':'none','scientific_acceptance_established_by_images':False})
    print('FOUR FIGURE PLATES WRITTEN',root.name,flush=True)


def refresh_training_preview(root):
    """Refresh existing CPU-only phase/curve plates after a saved epoch changes."""
    best=root/'best_model.pt';last=root/'last_checkpoint.pt'
    if (root/'training_complete.json').exists() or not best.exists() or not last.exists():return
    latest=max(best.stat().st_mtime_ns,last.stat().st_mtime_ns)
    images=[root/'evaluation'/name for name in ['phase_masks.png','training_curves.png']]
    if all(path.exists() and path.stat().st_mtime_ns>=latest for path in images):return
    print('CPU TRAINING PREVIEW',root.name,flush=True)
    result=subprocess.run([sys.executable,'-B','-u',str(HERE/'imaging/evaluation/evaluate.py'),
        '--checkpoint',str(best),'--device','cpu','--artifacts-only'],cwd=HERE)
    if result.returncode:
        print('Preview not updated; checkpoint may be changing. Will retry on the next watch cycle.',flush=True)


def catalog():
    lines=['# 仿真图像索引','','每个实验优先保留少量综合图；正式评估图和S6/S8原图保留在对应实验目录。',
           '诊断、短训练和结构扫描的综合预览使用验证集，不能当作盲测成绩。','']
    for root in sorted((HERE/'results').iterdir()):
        if not root.is_dir():continue
        has_preview=any((root/'evaluation'/name).exists() for name in ['high_resolution_validation.png','phase_masks.png','training_curves.png'])
        if not (root/'training_complete.json').exists() and not (root/'diagnostic_status.json').exists() and not (root/'figures'/'manifest.json').exists() and not has_preview and not (root.name.startswith('layers_n0_') and (root/'figures'/'free_space_control.png').exists()):continue
        lines+=['## '+root.name,'']
        if (root/'diagnostic_status.json').exists():lines+=['已因数值采样误差停止，仅为诊断图，不作为正式成像验收结果。','']
        elif not (root/'training_complete.json').exists() and ((root/'figures'/'manifest.json').exists() or has_preview):
            lines+=['训练进行中的预览；尚非最终测试结果。','']
        paths=list((root/'figures').glob('*.png'))
        if not paths:paths=list((root/'evaluation').glob('*.png'))
        else:paths+=list((root/'evaluation').glob('S6*.png'))
        if not paths:lines+=['图像待生成。',''];continue
        for path in sorted(paths):lines.append(f'- [{path.stem}]({path.relative_to(HERE/"results").as_posix()})')
        s8=root/'evaluation'/'S8_measurement_conditions_v2'
        if s8.exists():lines+=['',f'S8条件图保存在 `{s8.relative_to(HERE/"results").as_posix()}`。']
        lines.append('')
    (HERE/'results'/'图像索引.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path);p.add_argument('--watch',action='store_true');a=p.parse_args()
    if a.checkpoint:export(a.checkpoint);catalog();return
    while True:
        # These visible workflows already evaluate and export sequentially;
        # concurrent fine-grid exporters could exhaust the laptop GPU memory.
        managed={'wavelength1550_paper_geometry_os16','imagenet_train10000_1550nm_os16'}
        roots=[r for r in (HERE/'results').iterdir() if r.is_dir() and r.name not in managed and not r.name.endswith('_1550nm_os16') and (r/'training_complete.json').exists() and (r/'best_model.pt').exists()]
        priority={'layers_n10_total30_os2':0,'uniform_init_full50':1}
        roots.sort(key=lambda r:(priority.get(r.name,2),r.name))
        for root in roots:
            if finished(root):continue
            child=subprocess.Popen([sys.executable,'-B','-u',str(HERE/'imaging/evaluation/figures.py'),'--checkpoint',str(root/'best_model.pt')],cwd=HERE)
            print('EXPORT',root.name,'PID',child.pid,flush=True)
            save_json(HERE/'misc'/'figure_export_state.json',{'run':root.name,'child_pid':child.pid,'status':'exporting'})
            if child.wait()!=0:
                save_json(HERE/'misc'/'figure_export_state.json',{'run':root.name,'status':'failed'});raise RuntimeError('Figure export failed: '+root.name)
        for root in sorted((HERE/'results').iterdir()):
            if root.is_dir() and (root.name in managed or root.name.endswith('_1550nm_os16')):
                refresh_training_preview(root)
        catalog()
        save_json(HERE/'misc'/'figure_export_state.json',{'status':'current_completed_experiments_have_figures','checked_runs':len(roots)})
        if not a.watch:return
        time.sleep(20)


if __name__=='__main__':main()
