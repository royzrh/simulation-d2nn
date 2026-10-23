"""Validation and fixed 100 held-out images for S7A architecture curves."""

# Allow this tool to run directly while keeping project imports rooted correctly.
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _BootstrapPath
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import hashlib
from dataclasses import asdict
from pathlib import Path
import torch
from imaging.evaluation.evaluate import load
from train import epoch
from train import ImagingDataset,save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--sampling-audit',action='store_true');a=p.parse_args()
    m,c=load(a.checkpoint,'cuda')
    encoding=c['config'].get('input_encoding','amplitude_gray')
    va=ImagingDataset('validation',encoding,c['config'].get('data_dir'));test=ImagingDataset('test',encoding,c['config'].get('data_dir'))
    batch=1 if m.g.oversample>=8 else 4
    progress={'micro_batch_size':1} if batch==1 else {}
    summary={'geometry':asdict(m.g),'selected_epoch':c['epoch'],
             'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
             'dataset_manifest_sha256':c['config']['manifest_sha256'],
             'validation':epoch(m,torch.utils.data.DataLoader(va,batch_size=batch),'cuda',c['config']['target'],**progress),
             'test100':epoch(m,torch.utils.data.DataLoader(torch.utils.data.Subset(test,range(100)),batch_size=batch),'cuda',c['config']['target'],**progress),
             'geometry_segment_rule_is_assumption':True,
             'do_not_select_architecture_on_test100':True}
    save_json(a.checkpoint.parent/'sweep_metrics.json',summary);print(summary,flush=True)
    if a.checkpoint.parent.name.startswith('layers_n10_total30_'):
        # The original S6 diagram labels L1..L10. Use the 10-mask imaging model,
        # not the selected five-mask main imager, for its principal comparison.
        from imaging.evaluation.evaluate import patterns,longitudinal
        out=a.checkpoint.parent/'evaluation';out.mkdir(exist_ok=True)
        with torch.no_grad():longitudinal(m,patterns(m)['three_points'][None],out)
        save_json(out/'S6_geometry_note.json',{'paper_diagram_layers':10,
            'source':'supplement p27, L1-L10 visible in panel C',
            'geometry':asdict(m.g),'segment_distances_explicit_in_S6':False,
            'assumption':'Use S7A 30mm total-distance ten-layer architecture; exact segment allocation undocumented.'})
    if a.checkpoint.parent.name.startswith('layers_n1_total30_'):
        # S7A includes the no-mask (N=0) control at the same 30mm distance.
        free_g=asdict(m.g);free_g.update(layers=0,input_mm=30.,interlayer_mm=0.,output_mm=0.)
        control={'geometry':free_g,'selected_epoch':None,'evaluation_mode':'free_space_no_masks',
                 'validation':epoch(m,torch.utils.data.DataLoader(va,batch_size=batch),'cuda',c['config']['target'],free=True,**progress),
                 'test100':epoch(m,torch.utils.data.DataLoader(torch.utils.data.Subset(test,range(100)),batch_size=batch),'cuda',c['config']['target'],free=True,**progress)}
        free_name=a.checkpoint.parent.name.replace('layers_n1_','layers_n0_',1)
        save_json(a.checkpoint.parent.parent/free_name/'sweep_metrics.json',control)
    del m;torch.cuda.empty_cache()
    if a.sampling_audit:
        from imaging.evaluation.evaluate import high_resolution_audit
        out=a.checkpoint.parent/'evaluation';out.mkdir(exist_ok=True)
        high_resolution_audit(a.checkpoint,out)


if __name__=='__main__':main()
