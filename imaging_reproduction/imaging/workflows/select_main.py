"""Select among completed equal-budget main imagers using validation MSE only."""

# Allow this tool to run directly while keeping project imports rooted correctly.
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _BootstrapPath
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import csv
import hashlib
import json
from pathlib import Path
import torch
from train import HERE,ImagingDataset,save_json


def main():
    candidates=['baseline_os2','uniform_init_full50','lr010_init01_full50']
    rows=[];reference=None
    manifest=ImagingDataset('validation').manifest['manifest_sha256']
    for name in candidates:
        root=HERE/'results'/name
        if not (root/'training_complete.json').is_file():
            raise RuntimeError('Candidate still incomplete; selection not written: '+name)
        complete=json.loads((root/'training_complete.json').read_text(encoding='utf-8'))
        config=json.loads((root/'config.json').read_text(encoding='utf-8'))
        with (root/'training_history.csv').open(encoding='utf-8',newline='') as f:history=list(csv.DictReader(f))
        if complete['epochs']!=50 or config['epochs']!=50 or [int(r['epoch']) for r in history]!=list(range(1,51)):
            raise ValueError('All candidates must complete the same 50-epoch budget: '+name)
        if config['limit_train'] or config['limit_val'] or config['manifest_sha256']!=manifest:
            raise ValueError('Limited or mismatched dataset: '+name)
        if any(int(r['train_n'])!=1500 or int(r['val_n'])!=200 for r in history):
            raise ValueError('Incomplete per-epoch data counts: '+name)
        comparable={k:v for k,v in config.items() if k not in
                    {'lr','geometry','continued_from','parent_checkpoint_sha256'}}
        comparable['geometry']={k:v for k,v in config['geometry'].items() if k!='init_std'}
        if reference is None:reference=comparable
        if comparable!=reference:raise ValueError('Candidate differences exceed learning rate / initialization: '+name)
        path=root/'best_model.pt';best=torch.load(path,map_location='cpu',weights_only=True)
        minimum=min(history,key=lambda r:float(r['val_mse']))
        if best['config']!=config or best['epoch']!=int(minimum['epoch']):
            raise ValueError('Best checkpoint/config does not match validation-selected history: '+name)
        mse=best['validation']['mse']
        if abs(mse-float(minimum['val_mse']))>1e-12 or abs(mse-complete['best_validation_mse'])>1e-12:
            raise ValueError('Validation metric inconsistency: '+name)
        if not all(torch.isfinite(t).all() for t in best['model'].values()):
            raise ValueError('Nonfinite model: '+name)
        rows.append({'run':name,'checkpoint':str(path.resolve()),
                     'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                     'selected_epoch':best['epoch'],'validation':best['validation'],
                     'learning_rate':config['lr'],'init_std':config['geometry']['init_std']})
    selected=min(rows,key=lambda r:r['validation']['mse'])
    result={'criterion':'minimum_validation_raw_intensity_MSE','test_used_for_selection':False,
            'epochs_per_candidate':50,'manifest_sha256':manifest,'candidates':rows,'selected':selected,
            'completion_of_full_reproduction_established':False}
    output=HERE/'results'/'main_model_selection.json'
    if output.exists() and json.loads(output.read_text(encoding='utf-8'))!=result:
        raise ValueError('Preserve existing selection; investigate changed candidates before replacing it')
    save_json(output,result)
    print('VALIDATION-SELECTED MAIN MODEL',selected,flush=True)


if __name__=='__main__':main()
