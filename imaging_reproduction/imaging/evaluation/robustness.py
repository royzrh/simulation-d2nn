"""S7C joint reconstruction/absorption/misalignment evaluated on unseen images."""

# Allow this tool to run directly while keeping project imports rooted correctly.
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _BootstrapPath
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from train import ssim
from imaging.evaluation.evaluate import load,patterns,comparison
from train import ImagingDataset,save_json


def validation_ablation(m,c,errors,root,checkpoint_path):
    """Separate error mechanisms on validation; no model/scale fitting."""
    ds=ImagingDataset('validation',c['config'].get('input_encoding','amplitude_gray'),c['config'].get('data_dir'))
    if ds.manifest['manifest_sha256']!=c['config']['manifest_sha256']:raise ValueError('Validation dataset identity mismatch')
    dl=torch.utils.data.DataLoader(ds,batch_size=4)
    shifts=np.random.default_rng(42).uniform(-.1,.1,(m.g.layers,2)).tolist()
    cases={'ideal':{},'absorption_only':{'absorption':True},
           'surface_only':{'height_errors':errors},'shift_only':{'shifts':shifts},
           'joint':{'absorption':True,'shifts':shifts,'height_errors':errors}}
    rows=[];chart=patterns(m)['resolution_chart_both_axes'][None];chart_outputs={}
    with torch.no_grad():
        for name,kwargs in cases.items():
            totals={'mse':0.,'ssim_amplitude':0.,'capture_ratio':0.};n=0
            for image in dl:
                image=image.cuda();target=image.square() if c['config']['target']=='amplitude_squared' else image
                pred=m(image,**kwargs);batch=len(image);n+=batch
                totals['mse']+=(pred-target).square().mean().item()*batch
                totals['ssim_amplitude']+=ssim(pred.sqrt(),target.sqrt()).sum().item()
                totals['capture_ratio']+=(pred.sum((-2,-1))/image.square().sum((-2,-1)).clamp_min(1e-12)).sum().item()
            row={'case':name,'n':n}|{k:v/n for k,v in totals.items()};rows.append(row)
            chart_outputs[name]=m(chart,**kwargs)[0].cpu().numpy()
            print('VALIDATION ABLATION',row,flush=True)
    out=root/'evaluation'
    np.savez_compressed(out/'S7C_error_ablation.npz',input=chart[0].cpu().numpy(),**chart_outputs)
    comparison(out/'S7C_error_ablation.png',[chart[0]]+[np.sqrt(v) for v in chart_outputs.values()],
               ['Input']+list(chart_outputs))
    save_json(out/'error_ablation_validation.json',{'split':'all200 validation','rows':rows,
        'output_rescaling':'none','seed':42,'shifts_mm':shifts,
        'absorption_power_coefficient_per_m':520.7177,'substrate_m':.0005,'index':1.7227,
        'height_errors_sha256':hashlib.sha256((root/'surface_reconstruction'/'height_errors_m.npy').read_bytes()).hexdigest(),
        'checkpoint_sha256':hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        'original_reconstruction_settings_unpublished':True})


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--ablation-only',action='store_true');a=p.parse_args()
    m,c=load(a.checkpoint,'cuda');root=a.checkpoint.parent
    errors=np.load(root/'surface_reconstruction'/'height_errors_m.npy')
    summary=json.loads((root/'surface_reconstruction'/'summary.json').read_text(encoding='utf-8'))
    if summary.get('checkpoint_sha256')!=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest():
        raise ValueError('Re-run reconstruct.py to verify checkpoint/surface provenance before evaluation')
    if summary.get('height_errors_sha256')!=hashlib.sha256((root/'surface_reconstruction'/'height_errors_m.npy').read_bytes()).hexdigest():
        raise ValueError('Height-error file differs from verified reconstruction')
    if errors.shape!=(m.g.layers,m.g.neurons,m.g.neurons):raise ValueError('Height error shape mismatch')
    if not np.isfinite(errors).all():raise ValueError('Uncovered surface rays: reconstruction cannot be evaluated silently')
    errors=torch.from_numpy(errors).float().cuda()
    if a.ablation_only:
        validation_ablation(m,c,errors,root,a.checkpoint);return
    ds=ImagingDataset('test',c['config'].get('input_encoding','amplitude_gray'),c['config'].get('data_dir'));dl=torch.utils.data.DataLoader(ds,batch_size=4)
    if ds.manifest['manifest_sha256']!=c['config']['manifest_sha256']:raise ValueError('Test dataset identity mismatch')
    records=[]
    with torch.no_grad():
        for seed in [42,43,44]:
            shifts=np.random.default_rng(seed).uniform(-.1,.1,(m.g.layers,2)).tolist()
            totals={'mse':0.,'ssim_amplitude':0.,'capture_ratio':0.};n=0
            for image in dl:
                image=image.cuda();target=image.square() if c['config']['target']=='amplitude_squared' else image
                pred=m(image,absorption=True,shifts=shifts,height_errors=errors)
                totals['mse']+=(pred-target).square().mean().item()*len(image)
                totals['ssim_amplitude']+=ssim(pred.sqrt(),target.sqrt()).sum().item()
                totals['capture_ratio']+=(pred.sum((-2,-1))/image.square().sum((-2,-1)).clamp_min(1e-12)).sum().item();n+=len(image)
            records.append({'seed':seed,'n':n,'shifts_mm':shifts}|{k:v/n for k,v in totals.items()})
        chart=patterns(m)['resolution_chart_both_axes'][None]
        ideal=m(chart);error=m(chart,absorption=True,shifts=shifts,height_errors=errors);free=m.free_space(chart)
        comparison(root/'evaluation'/'S7C_full_error.png',[chart[0],ideal[0].sqrt(),error[0].sqrt(),free[0].sqrt()],['Input','Ideal','Poisson + loss + shift','Free-space'])
        np.savez_compressed(root/'evaluation'/'S7C_full_error_data.npz',input=chart[0].cpu().numpy(),
                            ideal=ideal[0].cpu().numpy(),joint=error[0].cpu().numpy(),free=free[0].cpu().numpy())
    save_json(root/'evaluation'/'full_error_metrics.json',{'runs':records,
        'mean_ssim_amplitude':float(np.mean([r['ssim_amplitude'] for r in records])),
        'checkpoint_sha256':summary['checkpoint_sha256'],'height_errors_sha256':summary['height_errors_sha256'],
        'manifest_sha256':ds.manifest['manifest_sha256'],'chart_seed':44,
        'chart_pattern':'resolution_chart_both_axes','output_rescaling':'none',
        'poisson_reconstruction':str(root/'surface_reconstruction'/'summary.json'),
        'original_reconstruction_settings_unpublished':True})
    print('FULL ERROR EVALUATION COMPLETE',records,flush=True)


if __name__=='__main__':main()
