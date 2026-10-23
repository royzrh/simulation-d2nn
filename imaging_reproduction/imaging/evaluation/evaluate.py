"""S6/S7/S8 numerical measurements; never labels simulations as THz experiments."""

# Allow this tool to run directly while keeping project imports rooted correctly.
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _BootstrapPath
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
from dataclasses import asdict
import csv
import hashlib
import io
import json
import math
import subprocess
import sys
import textwrap
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image,ImageDraw,ImageFont
from train import Geometry,ImagingD2NN,ssim
from train import target_of,epoch
from train import HERE,ImagingDataset,save_json


def load(path,device,over=None,pad=None):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    g=Geometry(**ck['config']['geometry'])
    if over:g.oversample=over
    if pad:g.pad_factor=pad
    m=ImagingD2NN(g).to(device);m.load_state_dict(ck['model']);m.eval()
    m.checkpoint_epoch=ck['epoch']
    return m,ck


def grid(model):
    x=(torch.arange(model.g.neurons,device=next(model.parameters()).device)-(model.g.neurons-1)/2)*model.g.pitch_m*1000
    return torch.meshgrid(x,x,indexing='ij')



def fwhm_mm(profile,spacing_mm):
    """Connected half-height width around the principal peak, interpolated."""
    v=np.asarray(profile);peak=int(np.argmax(v));half=v[peak]/2
    if v[peak]<=0:return None
    left=peak;right=peak
    while left>0 and v[left-1]>=half:left-=1
    while right<len(v)-1 and v[right+1]>=half:right+=1
    if left==0 or right==len(v)-1:return None
    xl=(left-1)+(half-v[left-1])/(v[left]-v[left-1])
    xr=right+(v[right]-half)/(v[right]-v[right+1])
    return float((xr-xl)*spacing_mm)


def patterns(model):
    y,x=grid(model); patterns={}
    for diameter in [1.,2.,3.]:patterns[f'pinhole_{diameter:g}mm']=((x*x+y*y)<(diameter/2)**2).float()
    a=torch.zeros_like(x)
    for cx in [-9.,0.,9.]:
        a[model.g.neurons//2,model.g.neurons//2+round(cx/(model.g.pitch_m*1000))]=1
    patterns['three_points']=a
    chart=torch.zeros_like(x)
    for i,w in enumerate([4.2,3.6,3.,2.4,1.8,1.2]):
        cx=-23 if i%2==0 else 23;cy=-25+(i//2)*25
        for shift in [-2,0,2]:
            if i%2==0:bar=((x-(cx+shift*w)).abs()<w/2)&((y-cy).abs()<9)
            else:bar=((y-(cy+shift*w)).abs()<w/2)&((x-cx).abs()<9)
            chart[bar]=1
    patterns['resolution_chart']=chart
    # Each published linewidth must be checked along both axes. Placement is a
    # disclosed reconstruction choice; bar widths remain in physical millimeters.
    paired=torch.zeros_like(x)
    for i,w in enumerate([4.2,3.6,3.,2.4,1.8,1.2]):
        cx=-23 if i%2==0 else 23;cy=-25+(i//2)*25
        for offset in [-2,0,2]:
            vertical=((x-(cx-8.5+offset*w)).abs()<w/2)&((y-cy).abs()<9)
            horizontal=((y-(cy+offset*w)).abs()<w/2)&((x-(cx+10.5)).abs()<6)
            paired[vertical|horizontal]=1
    patterns['resolution_chart_both_axes']=paired
    for letter in 'UCLA':
        im=Image.new('L',(300,300));draw=ImageDraw.Draw(im)
        try:font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',170)
        except OSError:font=ImageFont.load_default(size=170)
        box=draw.textbbox((0,0),letter,font=font)
        draw.text(((300-box[2]+box[0])/2-box[0],(300-box[3]+box[1])/2-box[1]),letter,font=font,fill=255)
        patterns['letter_'+letter]=torch.from_numpy(np.asarray(im,dtype=np.float32).copy()/255).to(x.device)
    return patterns


def comparison(path,arrays,titles):
    fig,axs=plt.subplots(1,len(arrays),figsize=(3*len(arrays),3),squeeze=False)
    for ax,a,title in zip(axs[0],arrays,titles):
        ar=a.detach().cpu().numpy() if isinstance(a,torch.Tensor) else a
        ax.imshow(ar,cmap='gray',vmin=0,vmax=1,extent=[-45,45,45,-45]);ax.set_title(textwrap.fill(title,30),fontsize=10);ax.set_xlabel('x (mm)')
    fig.tight_layout();fig.savefig(path,dpi=150,bbox_inches='tight');plt.close(fig)



def model_artifacts(model,checkpoint,out):
    raw=model.raw_phases.detach() if model is not None else checkpoint['model']['raw_phases']
    geometry=model.g if model is not None else Geometry(**checkpoint['config']['geometry'])
    phases=(2*np.pi*torch.sigmoid(raw)).cpu().numpy()
    half=geometry.neurons*geometry.pitch_m*500
    status='Validation-selected model' if (out.parent/'training_complete.json').exists() else 'Training preview'
    description=f"{status}; wavelength {299792458/geometry.frequency_hz*1e9:g} nm; OS{geometry.oversample}; selected epoch {checkpoint['epoch']}"
    np.save(out/'trained_phases_rad.npy',phases)
    fig,axes=plt.subplots(1,len(phases),figsize=(3*len(phases),3),squeeze=False)
    for i,(ax,phase) in enumerate(zip(axes[0],phases)):
        image=ax.imshow(phase,cmap='twilight',vmin=0,vmax=2*np.pi,extent=[-half,half,half,-half])
        ax.set_title(f'L{i+1} phase');ax.set_xlabel('x (mm)')
    fig.colorbar(image,ax=axes.ravel().tolist(),label='radians')
    fig.suptitle(description)
    fig.savefig(out/'phase_masks.png',dpi=150,bbox_inches='tight');plt.close(fig)
    history_path=out.parent/'training_history.csv'
    if checkpoint.get('history') or history_path.exists():
        if checkpoint.get('history'):
            history={key:np.array([row[key] for row in checkpoint['history']]) for key in checkpoint['history'][0]}
        else:history=np.genfromtxt(history_path,delimiter=',',names=True)
        single_point=np.size(history['epoch'])==1
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for key,ax in [('mse',axes[0]),('ssim_amplitude',axes[1])]:
            for split in ['train','val']:ax.plot(history['epoch'],history[split+'_'+key],label=split,marker='o' if single_point else None,zorder=3)
            ax.axvline(checkpoint['epoch'],color='gray',linestyle='--',label='selected epoch')
            if single_point:
                observed=float(np.atleast_1d(history['epoch'])[0])
                ax.set_xlim(observed-.5,observed+.5);ax.set_xticks([observed])
            ax.set_xlabel('epoch');ax.set_ylabel(key);ax.legend()
        fig.suptitle(description)
        fig.tight_layout();fig.savefig(out/'training_curves.png',dpi=150);plt.close(fig)


def natural(model,ck,out,device):
    data=ImagingDataset('test',ck['config'].get('input_encoding','amplitude_gray'),ck['config'].get('data_dir'))
    if ck['config']['manifest_sha256']!=data.manifest['manifest_sha256']:raise ValueError('Dataset mismatch')
    dl=torch.utils.data.DataLoader(data,batch_size=1 if model.g.oversample>=8 else 4)
    progress={'micro_batch_size':1} if model.g.oversample>=8 else {}
    print('TEST: evaluating D2NN on all held-out images',flush=True)
    network_metrics=epoch(model,dl,device,ck['config']['target'],**progress)
    print('TEST: evaluating matched free-space baseline',flush=True)
    free_metrics=epoch(model,dl,device,ck['config']['target'],free=True,**progress)
    summary={'network':network_metrics,
             'free_space':free_metrics,
             'first_100_test_images':[], 'sampling':asdict(model.g)}
    records=[]
    with torch.no_grad():
        for i,a in enumerate(data):
            a=a[None].to(device);y=target_of(a,ck['config']['target']);pred=model(a);free=model.free_space(a)
            row={'index':i,'mse':(pred-y).square().mean().item(),
                 'ssim_amplitude':ssim(pred.clamp_min(0).sqrt(),y.sqrt()).item(),
                 'ssim_intensity':ssim(pred,y).item(),
                 'free_ssim_amplitude':ssim(free.clamp_min(0).sqrt(),y.sqrt()).item()}
            records.append(row)
            if i<6:comparison(out/f'test_{i:03}.png',[a[0],pred[0].sqrt(),free[0].sqrt()],['Input amplitude','D2NN amplitude','Free-space amplitude'])
            if i==0 or (i+1)%10==0 or i+1==len(data):
                print(f'TEST: paired per-image measurements {i+1}/{len(data)}',flush=True)
    summary['ssim_amplitude_mean_first100']=float(np.mean([r['ssim_amplitude'] for r in records[:100]]))
    summary['ssim_intensity_mean_first100']=float(np.mean([r['ssim_intensity'] for r in records[:100]]))
    summary['test_indices']='All 300 held-out images; first 100 fixed by immutable manifest for S7A-style average'
    summary.pop('first_100_test_images')
    with (out/'per_image.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    save_json(out/'test_metrics.json',summary)


def synthetic(model,out,pattern_names=None):
    metrics={};rng=np.random.default_rng(42)
    paper_material=math.isclose(model.g.frequency_hz,0.4e12,rel_tol=1e-12)
    shifts=rng.uniform(-.1,.1,(model.g.layers,2)).tolist()
    with torch.no_grad():
        for name,a in patterns(model).items():
            if pattern_names and name not in pattern_names:continue
            print('SYNTHETIC: evaluating and exporting',name,flush=True)
            a=a[None];ideal=model(a);free=model.free_space(a)
            lossy=model(a,absorption=True,shifts=shifts) if paper_material else None
            metrics[name]={'ssim_amplitude':ssim(ideal.sqrt(),a).item(),
                           'mse_intensity':(ideal-a.square()).square().mean().item(),
                           'lossy_shifted_ssim_amplitude':ssim(lossy.sqrt(),a).item() if lossy is not None else None,
                           'free_space_ssim_amplitude':ssim(free.sqrt(),a).item()}
            if not paper_material:
                metrics[name]['material_error_status']='Not evaluated: wavelength-specific hardware/material data unavailable'
            if name.startswith('pinhole_'):
                yy,xx=grid(model)
                within=(xx.square()+yy.square())<=3.0**2
                power=ideal[0].sum().clamp_min(1e-20)
                metrics[name]['fraction_power_within_3mm_radius']=(ideal[0][within].sum()/power).item()
                metrics[name]['capture_ratio']=(power/a.square().sum().clamp_min(1e-20)).item()
                metrics[name]['background_dominates_full_frame_ssim']=True
                for kind,signal in [('amplitude',ideal[0].sqrt()),('intensity',ideal[0])]:
                    line=signal[model.g.neurons//2].cpu().numpy()
                    metrics[name]['fwhm_'+kind+'_mm']=fwhm_mm(line,model.g.pitch_m*1000)
            if name in {'resolution_chart','resolution_chart_both_axes'}:
                contrast=[]
                from torch.nn.functional import grid_sample
                for k,w in enumerate([4.2,3.6,3.,2.4,1.8,1.2]):
                    cx=-23 if k%2==0 else 23;cy=-25+(k//2)*25
                    orientations=['vertical','horizontal'] if name.endswith('both_axes') else ['vertical' if k%2==0 else 'horizontal']
                    for orientation in orientations:
                        pcx=cx-8.5 if name.endswith('both_axes') and orientation=='vertical' else cx+10.5 if name.endswith('both_axes') else cx
                        # Average along each bar, so a single speckle peak cannot
                        # be mistaken for resolution of an entire stripe group.
                        half_length=6. if orientation=='vertical' else 4.
                        along=np.linspace(-half_length,half_length,41)
                        coords=[[(pcx+offset*w,cy+d) if orientation=='vertical' else (pcx+d,cy+offset*w) for offset in [-2,-1,0,1,2]] for d in along]
                        loc=ideal.new_tensor(coords)[None]/45
                        values=grid_sample(ideal[:,None],loc,align_corners=False)[0,0].mean(0)
                        peak=values[[0,2,4]].mean();valley=values[[1,3]].mean()
                        contrast.append({'linewidth_mm':w,'orientation':orientation,'profile_averaged_length_mm':2*half_length,'peak_mean':peak.item(),'valley_mean':valley.item(),
                                         'modulation':((peak-valley)/(peak+valley).clamp_min(1e-12)).item()})
                metrics[name]['line_modulations']=contrast
            arrays=[a[0],ideal[0].sqrt()];titles=['Input','Ideal D2NN']
            saved={'input':a.cpu().numpy(),'ideal':ideal.cpu().numpy(),'free':free.cpu().numpy()}
            if lossy is not None:
                arrays.append(lossy[0].sqrt());titles.append('Absorption + shift');saved['absorption_shift']=lossy.cpu().numpy()
            arrays.append(free[0].sqrt());titles.append('Free-space')
            comparison(out/f'{name}.png',arrays,titles)
            np.savez_compressed(out/f'{name}.npz',**saved)
            if name=='resolution_chart_both_axes':
                from imaging.evaluation.resolution_profiles import plot_profiles
                plot_profiles(out/f'{name}.npz')
            if name=='three_points':longitudinal(model,a,out)
        # Numerically probe axial defocus; S8 is a physical experiment, not reproduced hardware.
        original=model.H_0.clone();a=patterns(model)['pinhole_3mm'][None]
        for offset in [-3.,0.,3.,6.,9.,12.]:
            model.H_0.copy_(torch.exp(1j*model.kz_relative*(model.g.input_mm+offset)*1e-3))
            p=model(a);metrics[f'axial_offset_{offset:g}mm']={'ssim_amplitude':ssim(p.sqrt(),a).item()}
        model.H_0.copy_(original)
        # Undocumented off-axis positions are recorded, not claimed to match P1-P3 exactly.
        yy,xx=grid(model)
        for cx,cy in [(0.,0.),(10.,0.),(-10.,0.),(0.,10.)]:
            image=(((xx-cx)**2+(yy-cy)**2)<1.5**2).float()[None]
            pred=model(image)
            metrics[f'offaxis_{cx:g}_{cy:g}mm']={'ssim_amplitude':ssim(pred.sqrt(),image).item()}
    metric_name='synthetic_metrics.json' if pattern_names is None else 'synthetic_metrics_selected_patterns.json'
    save_json(out/metric_name,{'measurements':metrics,'error_seed':42,'shifts_mm':shifts,
        'surface_reconstruction_included':False,'hardware_experiment_reproduced':False,
        'note':'Absorption/shift only; does not impersonate the original Poisson surface reconstruction error. S8 patterns are numerical analogues.'})


def longitudinal(model,a,out):
    print('S6 complex fields:',model.g.layers,'layers, physical-grid samples',model.n,flush=True)
    # Sample propagation between all masks at 0.5mm increments; identical free-space length.
    u=model.embed(a);z=0.;samples=[];zs=[];mask_planes=[]
    for i,dist in enumerate(model.g.distances_mm()):
        for local in np.linspace(0,dist,max(2,int(dist/.5)+1),endpoint=False):
            f=model.propagate(u,z_mm=float(local))
            samples.append(f[0,model.fft_n//2,model.s].detach().cpu().numpy().copy());zs.append(z+local)
        u=model.propagate(u,index=i)
        z+=dist
        if i<model.g.layers:
            mask_planes.append(z);u=u*model.mask(i)
    samples.append(u[0,model.fft_n//2,model.s].cpu().numpy().copy());zs.append(z)
    free=[];start=model.embed(a)
    for zi in zs:
        f=model.propagate(start,z_mm=float(zi));free.append(f[0,model.fft_n//2,model.s].cpu().numpy().copy())
    network=np.stack(samples);free=np.stack(free)
    carrier=np.exp(1j*(2*np.pi/(299792458/model.g.frequency_hz))*np.asarray(zs)[:,None]*1e-3)
    network_physical=network*carrier;free_physical=free*carrier
    np.savez_compressed(out/'S6_longitudinal.npz',z_mm=zs,network_relative=network,free_relative=free,network=network_physical,free=free_physical,mask_z_mm=mask_planes)
    # S6A/B: per-plane complex fields in addition to S6C/D longitudinal sections.
    _,plane_fields=model.field(a,trace=True)
    plane_z=[model.g.input_mm]+mask_planes+[z]
    fig,axs=plt.subplots(4,len(plane_fields),figsize=(2.5*len(plane_fields),9))
    plane_data=[];display_stride=max(1,math.ceil(model.n/600))
    half=model.g.neurons*model.g.pitch_m*500
    preview='Training preview' if not (out.parent/'training_complete.json').exists() else 'Numerical model'
    description=f"{preview}; {model.g.layers} layers; wavelength {299792458/model.g.frequency_hz*1e9:g} nm; epoch {getattr(model,'checkpoint_epoch',-1)}"
    plane_labels=['Before L1']+[f'After L{i+1}' for i in range(model.g.layers)]+['Detector']
    for k,(f,zi) in enumerate(zip(plane_fields,plane_z)):
        f=f[0,model.s,model.s].cpu().numpy()
        ff=model.propagate(start,z_mm=float(zi))[0,model.s,model.s].cpu().numpy()
        factor=np.exp(1j*(2*np.pi/(299792458/model.g.frequency_hz))*zi*1e-3)
        # Store the propagation's actual complex64 precision. Full spatial
        # sampling is retained; only arrays sent to Matplotlib are subsampled.
        f=(f*factor).astype(np.complex64);ff=(ff*factor).astype(np.complex64)
        plane_data.append(f)
        plane_fields[k]=None
        shown=f[::display_stride,::display_stride].copy();shown_free=ff[::display_stride,::display_stride].copy()
        for row,arr in enumerate([np.abs(shown),np.mod(np.angle(shown),2*np.pi),np.abs(shown_free),np.mod(np.angle(shown_free),2*np.pi)]):
            axs[row,k].imshow(arr,cmap='gray' if row%2==0 else 'twilight',extent=[-half,half,half,-half])
            axs[row,k].set_title(f'{plane_labels[k]}\nz={zi:g}mm');axs[row,k].set_xticks([]);axs[row,k].set_yticks([])
        print('S6 plane',k+1,'/',len(plane_fields),'collected at full spatial resolution',flush=True)
    for ax,label in zip(axs[:,0],['D2NN amplitude (a.u.)','D2NN phase','Free amplitude (a.u.)','Free phase']):ax.set_ylabel(label)
    fig.suptitle(description+f'; display stride {display_stride}, raw fields retain full grid')
    fig.tight_layout();fig.savefig(out/'S6_plane_fields.png',dpi=130);plt.close(fig)
    np.savez_compressed(out/'S6_plane_fields.npz',network=np.stack(plane_data),z_mm=plane_z,
        display_stride=display_stride,physical_grid_spacing_mm=model.g.pitch_m*1000/model.g.oversample,
        wavelength_nm=299792458/model.g.frequency_hz*1e9,checkpoint_epoch=getattr(model,'checkpoint_epoch',-1),
        input_amplitude=a.detach().cpu().numpy())
    print('S6 full-grid plane archive written',flush=True)
    del plane_data,plane_fields,_,f,ff,shown,shown_free,u,start
    x_samples=(np.arange(model.n)-(model.n-1)/2)*model.g.pitch_m*1000/model.g.oversample
    plot_longitudinal_sections(network_physical,free_physical,zs,x_samples,mask_planes,z,half,description,out)


def plot_longitudinal_sections(network_physical,free_physical,zs,x_samples,mask_planes,z,half,description,out):
    """Render saved cross-sections at their actual, nonuniform z coordinates."""
    fig,axs=plt.subplots(2,2,figsize=(11,7))
    for row,arr in enumerate([network_physical,free_physical]):
        for col,(data,title) in enumerate([(np.abs(arr),'Amplitude'),(np.mod(np.angle(arr),2*np.pi),'Sampled phase (0 to 2pi)')]):
            mesh=axs[row,col].pcolormesh(zs,x_samples,data.T,shading='nearest',cmap='gray' if col==0 else 'twilight')
            if col:mesh.set_clim(0,2*np.pi)
            axs[row,col].set_xlim(0,z);axs[row,col].set_ylim(-half,half)
            axs[row,col].set_title(('D2NN ' if row==0 else 'Free-space ')+title)
            axs[row,col].set_xlabel('z (mm)');axs[row,col].set_ylabel('x (mm)')
    fig.suptitle(description)
    fig.tight_layout();fig.savefig(out/'S6_longitudinal.png',dpi=150);plt.close(fig)
    # A linear global scale hides weak diffracted point-source fields. Preserve it,
    # and add a clearly labeled logarithmic amplitude view with the same reference.
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    peak=max(np.abs(network_physical).max(),np.abs(free_physical).max())
    for ax,arr,title in zip(axes,[network_physical,free_physical],['D2NN','Free space']):
        db=20*np.log10(np.maximum(np.abs(arr)/max(peak,1e-20),1e-4))
        image=ax.pcolormesh(zs,x_samples,db.T,shading='nearest',vmin=-80,vmax=0,cmap='viridis')
        ax.set_xlim(0,z);ax.set_ylim(-min(20,half),min(20,half));ax.set_title(title+' amplitude (dB, common reference)')
        ax.set_xlabel('z (mm)');ax.set_ylabel('x (mm)')
        if title=='D2NN':
            for k,zmask in enumerate(mask_planes):
                ax.axvline(zmask,color='white',alpha=.35,linewidth=.6)
                ax.text(zmask,19,f'L{k+1}',rotation=90,color='white',fontsize=7,va='top')
    fig.colorbar(image,ax=axes.ravel().tolist(),label='20 log10(amplitude / common maximum)')
    fig.suptitle(description)
    fig.savefig(out/'S6_log_amplitude.png',dpi=150,bbox_inches='tight');plt.close(fig)


def streamed_cpu_intensity(geometry,raw_phases,amplitude,free=False,cache_transfers=True):
    """Independent ideal ASM using SciPy FFT, row-block masks and cached distances.

    No optimizer or material model. Avoid allocating all padded layer masks or
    six identical propagation buffers when checking finer grids on CPU.
    """
    from scipy.fft import fft2,ifft2,fftfreq
    g=geometry;over=g.oversample;n=g.neurons*over;size=round(n*g.pad_factor)
    left=(size-n)//2;s=slice(left,left+n)
    source=np.asarray(amplitude,dtype=np.float32)
    if source.shape!=(g.neurons,g.neurons):raise ValueError('Physical input shape mismatch')
    u=np.zeros((size,size),dtype=np.complex64)
    for row in range(0,g.neurons,16):
        end=min(row+16,g.neurons)
        u[left+row*over:left+end*over,s]=source[row:end].repeat(over,0).repeat(over,1)
    wavelength=299792458/g.frequency_hz;k=2*np.pi/wavelength
    squared=(wavelength*fftfreq(size,d=g.pitch_m/over))**2
    transfers={}
    def transfer_rows(row,distance):
        q=squared[row:row+128,None]+squared[None,:]
        root=np.sqrt((1-q).astype(np.complex128))
        relative=-k*q/(1+root)
        return np.exp(1j*relative*distance*1e-3).astype(np.complex64)
    def propagate(field,distance):
        if cache_transfers and distance not in transfers:
            h=np.empty((size,size),dtype=np.complex64)
            for row in range(0,size,128):
                h[row:row+128]=transfer_rows(row,distance)
            transfers[distance]=h
        transformed=fft2(field,overwrite_x=True,workers=2)
        if cache_transfers:transformed*=transfers[distance]
        else:
            for row in range(0,size,128):
                transformed[row:row+128]*=transfer_rows(row,distance)
        return ifft2(transformed,overwrite_x=True,workers=2)
    if free:u=propagate(u,sum(g.distances_mm()))
    else:
        phases=(2*math.pi*torch.sigmoid(torch.as_tensor(raw_phases))).numpy()
        u=propagate(u,g.input_mm)
        for index in range(g.layers):
            u[:left]=0;u[left+n:]=0;u[:,:left]=0;u[:,left+n:]=0
            for row in range(0,g.neurons,16):
                end=min(row+16,g.neurons)
                phase=phases[index,row:end].repeat(over,0).repeat(over,1)
                u[left+row*over:left+end*over,s]*=np.exp(1j*phase)
            u=propagate(u,g.distances_mm()[index+1])
    intensity=np.abs(u[s,s]);np.square(intensity,out=intensity)
    result=intensity.reshape(g.neurons,over,g.neurons,over).mean((1,3),dtype=np.float64).astype(np.float32)
    if not np.isfinite(result).all():raise FloatingPointError('Nonfinite independent CPU propagation')
    return result


def high_resolution_audit(path,out):
    """Two fixed validation inputs, frozen weights, independent OS8/16/32 grids."""
    torch.set_num_threads(2)
    snapshot=Path(path).read_bytes();ck=torch.load(io.BytesIO(snapshot),map_location='cpu',weights_only=True)
    config=ck['config'];g=Geometry(**config['geometry'])
    ds=ImagingDataset('validation',config.get('input_encoding','amplitude_gray'),config.get('data_dir'))
    if ds.manifest['manifest_sha256']!=config['manifest_sha256']:raise ValueError('Audit dataset mismatch')
    inputs=np.stack([ds[i].numpy() for i in range(2)]);raw=ck['model']['raw_phases'].numpy()
    predictions={};rows=[]
    for over in [8,16,32]:
        geometry=Geometry(**asdict(g));geometry.oversample=over;geometry.pad_factor=16/15
        predictions[over]=np.stack([streamed_cpu_intensity(geometry,raw,image) for image in inputs])
        print('INDEPENDENT CPU GRID',over,'checkpoint epoch',ck['epoch'],flush=True)
    # Check the independent implementation against the existing Torch model on
    # the same grid, not just against other outputs of the new implementation.
    geometry=Geometry(**asdict(g));geometry.oversample=8;geometry.pad_factor=16/15
    model=ImagingD2NN(geometry);model.load_state_dict(ck['model']);model.eval()
    with torch.no_grad():torch_prediction=model(torch.from_numpy(inputs[:1]))[0].numpy()
    backend_difference=float(np.max(np.abs(torch_prediction-predictions[8][0])))
    if backend_difference>2e-5:raise ValueError(f'Independent backend disagrees with Torch: {backend_difference}')
    del model
    geometry.oversample=32
    free=np.stack([streamed_cpu_intensity(geometry,raw,image,free=True) for image in inputs])
    target=target_of(torch.from_numpy(inputs),config['target']);reference=predictions[32]
    for over,prediction in predictions.items():
        p=torch.from_numpy(prediction)
        rows.append({'oversample':over,'pad_factor':16/15,
            'relative_intensity_l2_vs_os32':float(np.linalg.norm(prediction-reference)/max(np.linalg.norm(reference),1e-20)),
            'validation_intensity_mse':float(np.mean((prediction-target.numpy())**2)),
            'validation_amplitude_ssim':ssim(p.sqrt(),target.sqrt()).tolist()})
    half=g.neurons*g.pitch_m*500
    fig,axes=plt.subplots(2,4,figsize=(12,6))
    arrays=[target[0].sqrt().numpy()]+[predictions[over][0]**.5 for over in [8,16,32]]
    for ax,array,title in zip(axes[0],arrays,['Target','OS8','OS16','OS32 finite reference']):
        ax.imshow(array,cmap='gray',vmin=0,vmax=max(float(a.max()) for a in arrays),extent=[-half,half,half,-half]);ax.set_title(title)
    differences=[np.abs(predictions[over][0]-reference[0]) for over in [8,16,32]]
    for ax,array,title in zip(axes[1,:3],differences,['OS8 difference','OS16 difference','OS32 difference']):
        im=ax.imshow(array,cmap='magma',vmin=0,vmax=max(max(float(a.max()) for a in differences),1e-12),extent=[-half,half,half,-half]);ax.set_title(title)
    axes[1,3].imshow(free[0]**.5,cmap='gray',vmin=0,vmax=max(float(a.max()) for a in arrays),extent=[-half,half,half,-half]);axes[1,3].set_title('Free space OS32')
    fig.suptitle(f'Independent CPU ASM; validation indices 0/1; fixed epoch {ck["epoch"]}; no gain fitting')
    fig.tight_layout(rect=[0,0,.96,.95]);bar=fig.add_axes([.965,.12,.01,.3]);fig.colorbar(im,cax=bar)
    fig.savefig(out/'high_resolution_validation.png',dpi=140,bbox_inches='tight');plt.close(fig)
    np.savez_compressed(out/'high_resolution_validation.npz',input=inputs,raw_phases=raw,
        geometry_json=json.dumps(asdict(g)),prediction_os8=predictions[8],prediction_os16=predictions[16],prediction_os32=reference,free_os32=free)
    save_json(out/'high_resolution_validation.json',{'checkpoint_sha256':hashlib.sha256(snapshot).hexdigest(),
        'checkpoint_epoch':ck['epoch'],'dataset_manifest_sha256':ds.manifest['manifest_sha256'],
        'validation_indices':[0,1],'reference_is_finite_resolution':True,'rows':rows,
        'independent_backend':'SciPy FFT complex64; row-block phase screens; physical-pixel intensity integration',
        'backend_reference':'https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft2.html',
        'torch_vs_scipy_os8_max_absolute_intensity_difference':backend_difference,
        'free_space_os32_amplitude_ssim':ssim(torch.from_numpy(free).sqrt(),target.sqrt()).tolist(),
        'free_space_os32_intensity_mse':float(np.mean((free-target.numpy())**2)),
        'network_os32_intensity_mse_per_image':np.mean((reference-target.numpy())**2,axis=(1,2)).tolist(),
        'free_space_os32_intensity_mse_per_image':np.mean((free-target.numpy())**2,axis=(1,2)).tolist(),
        'scope_warning':'Two fixed validation images are a numerical diagnostic, not a complete-dataset convergence proof.'})
    print('INDEPENDENT CPU AUDIT COMPLETE',rows,flush=True)


def extend_resolution_audit(path,out):
    """Extend the saved two-image diagnostic to OS48 without caching large H arrays."""
    torch.set_num_threads(2)
    snapshot=Path(path).read_bytes();ck=torch.load(io.BytesIO(snapshot),map_location='cpu',weights_only=True)
    audit_path=out/'high_resolution_validation.json'
    audit=json.loads(audit_path.read_text(encoding='utf-8'))
    if audit['checkpoint_sha256']!=hashlib.sha256(snapshot).hexdigest():
        raise ValueError('Extended audit requires the same frozen checkpoint as the saved audit')
    with np.load(out/'high_resolution_validation.npz') as saved:
        arrays={key:saved[key] for key in saved.files}
    if not np.array_equal(arrays['raw_phases'],ck['model']['raw_phases'].numpy()):
        raise ValueError('Saved raw phases disagree with the frozen checkpoint')
    ds=ImagingDataset('validation',ck['config'].get('input_encoding','amplitude_gray'),ck['config'].get('data_dir'))
    if ds.manifest['manifest_sha256']!=audit['dataset_manifest_sha256']:
        raise ValueError('Extended audit dataset mismatch')
    expected=np.stack([ds[i].numpy() for i in audit['validation_indices']])
    if not np.array_equal(expected,arrays['input']):raise ValueError('Saved validation inputs disagree with dataset')
    geometry=Geometry(**ck['config']['geometry']);geometry.oversample=48;geometry.pad_factor=16/15
    for free,key in [(False,'prediction_os48'),(True,'free_os48')]:
        results=[]
        for index,amplitude in enumerate(arrays['input']):
            print('EXTENDED CPU OS48',key,'image',index+1,'/',len(expected),flush=True)
            results.append(streamed_cpu_intensity(geometry,arrays['raw_phases'],amplitude,free=free,cache_transfers=False))
        arrays[key]=np.stack(results)
    target=target_of(torch.from_numpy(arrays['input']),ck['config']['target'])
    reference=arrays['prediction_os48'];rows=[]
    for over in [8,16,32,48]:
        pred=arrays[f'prediction_os{over}']
        rows.append({'oversample':over,
            'relative_intensity_l2_vs_os48':float(np.linalg.norm(pred-reference)/max(np.linalg.norm(reference),1e-20)),
            'validation_intensity_mse':float(np.mean((pred-target.numpy())**2)),
            'validation_amplitude_ssim':ssim(torch.from_numpy(pred).sqrt(),target.sqrt()).tolist()})
    audit['extended_reference']={'oversample':48,'pad_factor':16/15,'reference_is_finite_resolution':True,
        'transfer_storage':'recomputed in 128-row blocks, no cached full-grid transfer arrays',
        'rows':rows,'free_space_intensity_mse':float(np.mean((arrays['free_os48']-target.numpy())**2)),
        'free_space_amplitude_ssim':ssim(torch.from_numpy(arrays['free_os48']).sqrt(),target.sqrt()).tolist(),
        'scope_warning':'Same two fixed validation images; OS48 is another finite diagnostic, not proof of full-data convergence.'}
    half=geometry.neurons*geometry.pitch_m*500
    fig,axes=plt.subplots(3,5,figsize=(15,9))
    keys=['input','prediction_os16','prediction_os32','prediction_os48','free_os48']
    titles=['Target','OS16 network','OS32 network','OS48 network','OS48 free space']
    for index in range(2):
        for ax,key,title in zip(axes[index],keys,titles):
            image=target[index].sqrt().numpy() if key=='input' else np.sqrt(arrays[key][index])
            ax.imshow(image,cmap='gray',vmin=0,vmax=1,extent=[-half,half,half,-half])
            ax.set_title(f'Validation {audit["validation_indices"][index]}: {title}',fontsize=9)
    differences=[(over,index,np.abs(arrays[f'prediction_os{over}'][index]-reference[index])) for index in range(2) for over in [16,32]]
    scale=max(max(float(value.max()) for _,_,value in differences),1e-12)
    for ax,(over,index,value) in zip(axes[2,:4],differences):
        im=ax.imshow(value,cmap='magma',vmin=0,vmax=scale,extent=[-half,half,half,-half])
        ax.set_title(f'Validation {index}: |OS{over} - OS48| intensity',fontsize=9)
    axes[2,4].axis('off');fig.colorbar(im,ax=axes[2,4],fraction=.6,label='Absolute intensity difference')
    fig.suptitle(f'Independent CPU ASM; epoch {ck["epoch"]}; same two validation inputs; OS48 finite reference; no gain fitting')
    fig.tight_layout();fig.savefig(out/'high_resolution_validation.png',dpi=140);plt.close(fig)
    np.savez_compressed(out/'high_resolution_validation.npz',**arrays)
    save_json(audit_path,audit)
    print('EXTENDED CPU OS48 AUDIT COMPLETE',rows,flush=True)


def convergence(path,device,out):
    rows=[]
    snapshot=Path(path).read_bytes()
    ckmeta=torch.load(io.BytesIO(snapshot),map_location='cpu',weights_only=True)
    ds=ImagingDataset('validation',ckmeta['config'].get('input_encoding','amplitude_gray'),ckmeta['config'].get('data_dir'));a=torch.stack([ds[i] for i in range(20)]).to(device)
    short_wave=299792458.0/ckmeta['config']['geometry']['frequency_hz']<1e-4
    reference_os=16 if short_wave else 4
    reference_pad=16/15 if short_wave else 2
    settings=[(1,2),(2,2),(3,2),(4,2)]+([(8,2),(8,16/15),(16,16/15)] if short_wave else [])+[(2,3)]
    reference=None
    reference_free=None
    for over,pad in settings:
        geometry=Geometry(**ckmeta['config']['geometry']);geometry.oversample=over;geometry.pad_factor=pad
        m=ImagingD2NN(geometry).to(device);m.load_state_dict(ckmeta['model']);m.eval()
        # Bound evaluation memory while the architecture-training queue runs.
        with torch.no_grad():pred=torch.cat([m(batch).cpu() for batch in a.split(1 if short_wave else 2)])
        rows.append({'oversample':over,'pad_factor':pad,'mean_power':pred.sum((-2,-1)).mean().item()})
        if over==reference_os and pad==reference_pad:
            reference=pred
            with torch.no_grad():reference_free=torch.cat([m.free_space(batch).cpu() for batch in a.split(1)])
        rows[-1]['pred']=pred
        print('CONVERGENCE',over,pad,'checkpoint epoch',ckmeta['epoch'],flush=True)
        del m
        if device.type=='cuda':torch.cuda.empty_cache()
    fig,axes=plt.subplots(2,len(rows),figsize=(3*len(rows),6),squeeze=False)
    difference_max=max(float((row['pred'][0]-reference[0]).abs().max()) for row in rows)
    amplitude_max=max(float(row['pred'][0].sqrt().max()) for row in rows)
    half_width=ckmeta['config']['geometry']['neurons']*ckmeta['config']['geometry']['pitch_m']*500
    target=target_of(a,ckmeta['config']['target']).cpu()
    for j,row in enumerate(rows):
        pred=row.pop('pred');row['intensity_relative_l2_vs_os4_pad2']=((pred-reference).square().sum()/reference.square().sum().clamp_min(1e-20)).sqrt().item()
        row['validation_intensity_mse']=(pred-target).square().mean().item()
        row['validation_amplitude_ssim']=ssim(pred.clamp_min(0).sqrt(),target.sqrt()).mean().item()
        if short_wave:row['intensity_relative_l2_vs_reference']=row.pop('intensity_relative_l2_vs_os4_pad2')
        axes[0,j].imshow(pred[0].sqrt(),cmap='gray',vmin=0,vmax=amplitude_max,extent=[-half_width,half_width,half_width,-half_width])
        axes[0,j].set_title(f"OS{row['oversample']}, pad{row['pad_factor']}")
        im=axes[1,j].imshow((pred[0]-reference[0]).abs(),cmap='magma',vmin=0,vmax=max(difference_max,1e-12),extent=[-half_width,half_width,half_width,-half_width])
        axes[1,j].set_xlabel('x (mm)')
    axes[0,0].set_ylabel('Raw amplitude / y (mm)');axes[1,0].set_ylabel('Absolute intensity difference / y (mm)')
    fig.suptitle(f'Fixed validation image; same epoch {ckmeta["epoch"]}; finite reference OS{reference_os}, pad{reference_pad:g}')
    fig.tight_layout(rect=[0,0,.96,.95]);bar=fig.add_axes([.965,.12,.01,.3]);fig.colorbar(im,cax=bar)
    fig.savefig(out/'numerical_convergence.png',dpi=130,bbox_inches='tight');plt.close(fig)
    save_json(out/'numerical_convergence.json',{'split':'first20 validation','rows':rows,'reference_is_finite_resolution':True,
        'reference_oversample':reference_os,'reference_pad_factor':reference_pad,'checkpoint_epoch':ckmeta['epoch'],
        'reference_free_space_validation':{'intensity_mse':(reference_free-target).square().mean().item(),
            'amplitude_ssim':ssim(reference_free.clamp_min(0).sqrt(),target.sqrt()).mean().item()},
        'checkpoint_sha256':hashlib.sha256(snapshot).hexdigest(),'dataset_manifest_sha256':ds.manifest['manifest_sha256']})


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True,type=Path);p.add_argument('--device',default='cuda')
    p.add_argument('--convergence',action='store_true');p.add_argument('--synthetic-only',action='store_true')
    p.add_argument('--patterns',nargs='+',help='Only regenerate named synthetic patterns; requires --synthetic-only')
    p.add_argument('--convergence-only',action='store_true',help='Validation-only sampling audit of one immutable checkpoint snapshot')
    p.add_argument('--high-resolution-audit-only',action='store_true',help='Independent CPU validation propagation at OS8/16/32')
    p.add_argument('--extended-resolution-audit-only',action='store_true',help='Extend an existing matching CPU audit to OS48 without cached transfer arrays')
    p.add_argument('--artifacts-only',action='store_true',help='CPU-only phase/learning-curve preview; no propagation, test evaluation or training')
    p.add_argument('--eval-oversample',type=int,help='Evaluation grid; default OS8 for short wavelengths, checkpoint grid otherwise')
    a=p.parse_args();device=torch.device(a.device)
    if a.patterns and not a.synthetic_only:p.error('--patterns requires --synthetic-only')
    out=a.checkpoint.parent/'evaluation';out.mkdir(parents=True,exist_ok=True)
    if a.artifacts_only:
        torch.set_num_threads(2)
        snapshot=Path(a.checkpoint).read_bytes()
        ck=torch.load(io.BytesIO(snapshot),map_location='cpu',weights_only=True)
        last=a.checkpoint.parent/'last_checkpoint.pt'
        if last.exists():
            history=torch.load(last,map_location='cpu',weights_only=True)
            if history['config']!=ck['config']:raise ValueError('Preview checkpoint/history configuration mismatch')
            if history['epoch']<ck['epoch']:raise ValueError('Checkpoint update is in progress; retry the preview after the epoch finishes saving')
            ck['history']=history['history']
        model_artifacts(None,ck,out)
        print('PHASE AND LEARNING-CURVE PREVIEW',ck['epoch'],out,flush=True);return
    if a.high_resolution_audit_only:
        high_resolution_audit(a.checkpoint,out);return
    if a.extended_resolution_audit_only:
        extend_resolution_audit(a.checkpoint,out);return
    if a.convergence_only:
        convergence(a.checkpoint,device,out);return
    metadata=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    evaluation_os=a.eval_oversample
    if evaluation_os is None and 299792458/metadata['config']['geometry']['frequency_hz']<1e-4:
        evaluation_os=max(8,metadata['config']['geometry']['oversample'])
    model,ck=load(a.checkpoint,device,over=evaluation_os)
    effective_eval_os=model.g.oversample
    print('EVALUATION GRID',model.g.oversample,'training grid',ck['config']['geometry']['oversample'],flush=True)
    model_artifacts(model,ck,out)
    if not a.synthetic_only:natural(model,ck,out,device)
    synthetic(model,out,a.patterns)
    del model
    if a.convergence:convergence(a.checkpoint,device,out)
    if a.convergence and not a.synthetic_only and 299792458/ck['config']['geometry']['frequency_hz']<1e-4:
        if device.type=='cuda':torch.cuda.empty_cache()
        high_resolution_audit(a.checkpoint,out)
        extend_resolution_audit(a.checkpoint,out)
        print('S8: starting published illumination/scan numerical analogues; hardware beam parameters remain unconfirmed.',flush=True)
        subprocess.run([sys.executable,'-B','-u',str(HERE/'imaging/evaluation/s8_measurement.py'),
            '--checkpoint',str(a.checkpoint.resolve()),'--device',a.device,'--oversample',str(effective_eval_os)],
            cwd=HERE,check=True)
    print('EVALUATION COMPLETE',out,flush=True)
    control=HERE/'misc'/'queue_control.json'
    if a.convergence and a.checkpoint.parent.name=='baseline_os2' and control.exists():
        import time
        announced=False
        while json.loads(control.read_text(encoding='utf-8')).get('hold_after_baseline',False):
            if not announced:
                print('BASELINE REVIEW: evaluation saved. Waiting for validation-only initialization comparison before architecture scans.',flush=True)
                announced=True
            time.sleep(2)


if __name__=='__main__':main()
