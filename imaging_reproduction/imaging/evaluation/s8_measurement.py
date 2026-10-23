"""S8 numerical analogues with published illumination and detector scan spacing.

Supplement p4: ~20 mm beam FWHM, ~0.6 mm scan steps. Beam FWHM's
amplitude/intensity convention and detector aperture are not specified here;
both beam conventions are evaluated, with point samples (no invented aperture).
No training, data mutation, output gain fitting, or experimental claims.
"""

# Allow this tool to run directly while keeping project imports rooted correctly.
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _BootstrapPath
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import math
import numpy as np
import torch
from torch.nn.functional import grid_sample
from imaging.evaluation.evaluate import load,grid,patterns,fwhm_mm
from train import save_json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_probe(path,sampled,coords,title):
    """Show raw comparisons, a shared-scale ROI, signed errors and a line profile."""
    coords=np.asarray(coords);step=float(np.median(np.diff(coords)))
    extent=[coords[0]-step/2,coords[-1]+step/2,coords[-1]+step/2,coords[0]-step/2]
    target=sampled['target'];peak=float(target.max())
    iy,ix=np.where(target>max(peak*.01,1e-20))
    if not len(ix):iy,ix=np.indices(target.shape).reshape(2,-1)
    margin=max(2*step,.15*max(np.ptp(coords[ix]),np.ptp(coords[iy])))
    xlim=(float(coords[ix.min()]-margin),float(coords[ix.max()]+margin))
    ylim=(float(coords[iy.max()]+margin),float(coords[iy.min()]-margin))
    keys=['target','D2NN','free_segments','free_29p5mm']
    labels=['Target','D2NN','Free (segment sum)','Free (29.5 mm)']
    fig,axes=plt.subplots(3,4,figsize=(14,10),layout='constrained')
    roi_peak=max(float(sampled[key].max()) for key in keys)
    for col,(key,label) in enumerate(zip(keys,labels)):
        for row,vmax in [(0,1.),(1,max(roi_peak,1e-20))]:
            ax=axes[row,col]
            im=ax.imshow(sampled[key],cmap='hot',vmin=0,vmax=vmax,extent=extent,interpolation='nearest')
            ax.set_title(label if row==0 else label+' — object ROI',fontsize=10)
            ax.set_xlabel('x (mm)');ax.set_ylabel('y (mm)')
            if row==0:ax.set_xlim(-22.5,22.5);ax.set_ylim(22.5,-22.5)
            else:ax.set_xlim(*xlim);ax.set_ylim(*ylim)
        if col==3:
            fig.colorbar(axes[0,col].images[0],ax=axes[0,:].tolist(),label='Raw intensity; fixed 0–1 scale',shrink=.8)
            fig.colorbar(im,ax=axes[1,:].tolist(),label='Raw intensity; one shared ROI scale',shrink=.8)
    errors=[sampled[key]-target for key in keys[1:]]
    bound=max(max(float(np.max(np.abs(value))) for value in errors),1e-20)
    for ax,label,value in zip(axes[2,:3],labels[1:],errors):
        im=ax.imshow(value,cmap='RdBu_r',vmin=-bound,vmax=bound,extent=extent,interpolation='nearest')
        ax.set_xlim(*xlim);ax.set_ylim(*ylim);ax.set_xlabel('x (mm)');ax.set_ylabel('y (mm)')
        ax.set_title(f'{label} − target\nmax |difference|={np.max(np.abs(value)):.3g}',fontsize=9)
    fig.colorbar(im,ax=axes[2,:3].tolist(),label='Signed raw intensity difference',shrink=.8)
    weights=target.sum(0);cx=float(np.dot(coords,weights)/max(float(weights.sum()),1e-20))
    weights=target.sum(1);cy=float(np.dot(coords,weights)/max(float(weights.sum()),1e-20))
    vertical='horizontal_bars' in path.stem
    center=int(np.argmin(np.abs(coords-(cx if vertical else cy))))
    ax=axes[2,3]
    for key,label,style in zip(keys,labels,['k-','r--','b-.','g:']):
        profile=sampled[key][:,center] if vertical else sampled[key][center,:]
        ax.plot(coords,profile,style,label=label,linewidth=1.2)
    ax.set_xlim(*(tuple(reversed(ylim)) if vertical else xlim))
    ax.set_xlabel(('y' if vertical else 'x')+' (mm)');ax.set_ylabel('Raw intensity')
    ax.set_title(f'Profile at {"x" if vertical else "y"}={coords[center]:g} mm',fontsize=9)
    ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.suptitle(title+'\nSaved scan spacing 0.6 mm; shared scales; no per-image gain fitting',fontsize=10)
    fig.savefig(path,dpi=140);plt.close(fig)


def replot_saved(out):
    metadata_path=out/'measurement_metrics.json'
    metadata=json.loads(metadata_path.read_text(encoding='utf-8'))
    paths=sorted(out.glob('*.npz'))
    if len(paths)!=len(metadata['records']):raise ValueError('Saved S8 conditions/raw output counts disagree')
    for path in paths:
        with np.load(path) as raw:
            sampled={key:raw[key] for key in ['target','D2NN','free_segments','free_29p5mm']}
            plot_probe(path.with_suffix('.png'),sampled,raw['x_mm'],
                f'{path.stem}; wavelength {metadata["wavelength_nm"]:g} nm; epoch {metadata["selected_epoch"]}')
        print('S8 REPLOTTED',path.stem,flush=True)
    metadata['visualization']={'layout':'full field, object ROI, signed differences, raw line profile',
        'full_field_scale':[0,1],'roi_scale':'one shared scale for all four arrays within each condition',
        'differences':'output minus target; shared symmetric color scale; raw intensities unchanged',
        'scan_spacing_mm':.6,'replot_does_not_improve_spatial_sampling':True}
    save_json(metadata_path,metadata)


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--device',default='cuda');p.add_argument('--oversample',type=int)
    p.add_argument('--replot-only',action='store_true',help='Re-render saved raw S8 arrays on CPU; no propagation or training')
    a=p.parse_args()
    if a.replot_only:
        replot_saved(a.checkpoint.parent/'evaluation'/'S8_measurement_conditions_v2');return
    model,ck=load(a.checkpoint,a.device,over=a.oversample)
    # Preserve the earlier generic-letter / amplitude-display evaluation.
    out=a.checkpoint.parent/'evaluation'/'S8_measurement_conditions_v2';out.mkdir(parents=True,exist_ok=True)
    y,x=grid(model);r2=x*x+y*y
    coords=torch.arange(-44.4,44.4001,.6,device=x.device)
    sy,sx=torch.meshgrid(coords,coords,indexing='ij')
    locations=torch.stack([sx,sy],-1)[None]/45
    def sample(intensity):
        return grid_sample(intensity[:,None],locations,align_corners=False)[0,0]
    def probe(name,obj,beam,offset,convention,center_mm=(0.,0.)):
        original=model.H_0.clone()
        try:
            model.H_0.copy_(torch.exp(1j*model.kz_relative*(model.g.input_mm+offset)*1e-3))
            amplitude=(obj*beam)[None]
            pred=model(amplitude)
            uf_segments=model.propagate(model.embed(amplitude),z_mm=sum(model.g.distances_mm())+offset)
            free=torch.nn.functional.avg_pool2d(uf_segments[:,None,model.s,model.s].abs().square(),model.g.oversample)[:,0]
            # S8 caption gives 29.5 mm, whereas explicit segments sum to 27 mm.
            uf=model.propagate(model.embed(amplitude),z_mm=29.5+offset)
            free295=torch.nn.functional.avg_pool2d(uf[:,None,model.s,model.s].abs().square(),model.g.oversample)[:,0]
        finally:model.H_0.copy_(original)
        sampled={key:sample(value).cpu().numpy() for key,value in
                 [('target',amplitude.square()),('D2NN',pred),('free_segments',free),('free_29p5mm',free295)]}
        stem=f'{convention}_{name}_offset{offset:g}mm'
        np.savez_compressed(out/(stem+'.npz'),x_mm=coords.cpu().numpy(),**sampled)
        plot_probe(out/(stem+'.png'),sampled,coords.cpu().numpy(),
            f'{convention}: {name}, input offset +{offset:g} mm; wavelength {299792458/model.g.frequency_hz*1e9:g} nm')
        center=int((coords-center_mm[1]).abs().argmin())
        record={'pattern':name,'input_offset_mm':offset,'beam_fwhm_convention':convention,
                'capture_ratio':float(pred.sum()/amplitude.square().sum().clamp_min(1e-20)),
                'object_center_mm':list(center_mm),'profile_y_mm':float(coords[center])}
        if name.startswith('pinhole_'):
            record['sampled_amplitude_fwhm_x_mm']=fwhm_mm(np.sqrt(np.maximum(sampled['D2NN'][center],0)),.6)
            record['sampled_intensity_fwhm_x_mm']=fwhm_mm(sampled['D2NN'][center],.6)
        print('S8 EVALUATED',stem,flush=True)
        return record
    records=[]
    with torch.no_grad():
        test_patterns={name:obj for name,obj in patterns(model).items() if name.startswith('pinhole_')}
        for name,obj in patterns(model).items():
            if not name.startswith('letter_'):continue
            # S8A's 2mm scale bars suggest about 9mm letter heights; the exact
            # input masks are unavailable. Preserve font geometry, change only
            # the explicitly disclosed physical bounding-box height.
            iy,ix=torch.where(obj>0)
            cropped=obj[iy.min():iy.max()+1,ix.min():ix.max()+1]
            nh=round(9./(model.g.pitch_m*1000));nw=round(cropped.shape[1]/cropped.shape[0]*nh)
            small=torch.nn.functional.interpolate(cropped[None,None],size=(nh,nw),mode='bilinear',align_corners=False)[0,0]
            canvas=torch.zeros_like(obj);oy=(canvas.shape[0]-nh)//2;ox=(canvas.shape[1]-nw)//2
            canvas[oy:oy+nh,ox:ox+nw]=small
            test_patterns[name+'_height9mm']=canvas
        for w in [1.8,2.4,3.,3.6]:
            bars=torch.zeros_like(x)
            for shift in [-2,0,2]:bars[((y-shift*w).abs()<w/2)&(x.abs()<6)]=1
            test_patterns[f'horizontal_bars_{w:g}mm']=bars
        for convention,factor in [('intensity_FWHM20mm',2.),('amplitude_FWHM20mm',4.)]:
            beam=torch.exp(-factor*math.log(2)*r2/20**2)
            for name,obj in test_patterns.items():
                records.append(probe(name,obj,beam,0,convention))
            obj=patterns(model)['pinhole_3mm']
            for offset in [3.,6.,9.,12.,16.,20.,24.]:records.append(probe('pinhole_3mm',obj,beam,offset,convention))
            for cx,cy in [(-15.,-15.),(15.,-15.),(-15.,15.)]:
                obj=(((x-cx)**2+(y-cy)**2)<1.5**2).float()
                for offset in [0.,12.]:
                    records.append(probe(f'pinhole_3mm_x{cx:g}_y{cy:g}',obj,beam,offset,convention,(cx,cy)))
    save_json(out/'measurement_metrics.json',{
        'paper_source':'Supplement p4 illumination and scan step; p30 S8 caption',
        'checkpoint':str(a.checkpoint.resolve()),'selected_epoch':ck['epoch'],
        'wavelength_nm':299792458/model.g.frequency_hz*1e9,'sampling':asdict(model.g),
        'laboratory_1550nm_beam_measured':False,
        'beam_fwhm_mm':20,'scan_spacing_mm':.6,'detector_aperture_model':'point sample, aperture not supplied',
        'fwhm_definition_unspecified_both_conventions_reported':True,
        'pattern_geometry_is_reconstruction_choice':True,'illumination_curvature':'plane phase; original wavefront unavailable',
        'display_quantity':'raw intensity (S8 figure labels Int. a.u.)','display_limits':[0,1],
        'letter_height_mm':9,'letter_scale_basis':'Approximate from S8A 2mm scale bars; font/input masks not supplied',
        'bars_linewidth_mm':[1.8,2.4,3.,3.6],'bar_length_mm':12,'bar_length_is_assumption':True,
        'offaxis_locations_mm':[[-15,-15],[15,-15],[-15,15]],'offaxis_coordinates_are_assumptions':True,
        'display_field_mm':45,'source_field_of_view_mm':45,
        'absorption_reconstruction_misalignment_included':False,
        'physical_experiment_reproduced':False,'output_rescaling':'none','records':records})
    from imaging.evaluation.analyze_s8_positions import analyze_positions
    analyze_positions(out)
    print('S8 measurement-condition numerical comparison complete',out,flush=True)


if __name__=='__main__':main()
