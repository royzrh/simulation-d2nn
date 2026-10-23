"""Fixed-mask sampling sensitivity of misalignment and joint-error evaluation."""

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
from imaging.evaluation.evaluate import load
from train import ImagingDataset,save_json
from train import ssim


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--samples',type=int,default=8);a=p.parse_args()
    root=a.checkpoint.parent
    summary=json.loads((root/'surface_reconstruction'/'summary.json').read_text())
    assert summary['checkpoint_sha256']==hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    errorfile=root/'surface_reconstruction'/'height_errors_m.npy'
    assert summary['height_errors_sha256']==hashlib.sha256(errorfile.read_bytes()).hexdigest()
    errors=torch.from_numpy(np.load(errorfile)).float().cuda()
    rows=[];predictions={}
    for over in [2,3,4]:
        m,ck=load(a.checkpoint,'cuda',over=over)
        ds=ImagingDataset('validation',ck['config'].get('input_encoding','amplitude_gray'),ck['config'].get('data_dir'))
        shifts=np.random.default_rng(42).uniform(-.1,.1,(m.g.layers,2)).tolist()
        cases={'ideal':{},'shift_only':{'shifts':shifts},
               'joint':{'absorption':True,'shifts':shifts,'height_errors':errors}}
        with torch.no_grad():
            for case,kwargs in cases.items():
                result=[];inputs=[]
                for start in range(0,a.samples,2):
                    image=torch.stack([ds[i] for i in range(start,min(start+2,a.samples))]).cuda()
                    result.append(m(image,**kwargs).cpu());inputs.append(image.cpu())
                pred=torch.cat(result);image=torch.cat(inputs)
                target=image.square() if ck['config']['target']=='amplitude_squared' else image
                predictions[over,case]=pred
                row={'oversample':over,'case':case,'n':len(pred),
                     'ssim_amplitude':float(ssim(pred.sqrt(),target.sqrt()).mean()),
                     'capture_ratio':float((pred.sum((-2,-1))/image.square().sum((-2,-1))).mean())}
                rows.append(row);print('ERROR CONVERGENCE',row,flush=True)
        del m
        torch.cuda.empty_cache()
    for row in rows:
        pred=predictions[row['oversample'],row['case']];ref=predictions[4,row['case']]
        row['intensity_relative_l2_vs_os4']=float(((pred-ref).square().sum()/ref.square().sum()).sqrt())
    save_json(root/'evaluation'/'error_numerical_convergence.json',{
        'split':f'first{a.samples} validation','checkpoint_sha256':summary['checkpoint_sha256'],
        'height_errors_sha256':summary['height_errors_sha256'],'rows':rows,'seed':42,'shifts_mm':shifts,
        'reference_is_finite_resolution':True,'shift_interpolation':'bilinear complex transmission',
        'physical_phase_parameters_and_errors_unchanged':True})


if __name__=='__main__':main()
