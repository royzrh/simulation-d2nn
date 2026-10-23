"""Paired held-out comparison for the p10 larger training-sample claim."""

# Allow this tool to run directly while keeping project imports rooted correctly.
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _BootstrapPath
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import csv
import argparse
import json
from pathlib import Path
import numpy as np
from train import HERE,ImagingDataset,save_json


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--control-run')
    parser.add_argument('--large-run',default='imagenet_train10000_main50')
    options=parser.parse_args()
    control=HERE/'results'/options.control_run if options.control_run else Path(read(HERE/'results'/'main_model_selection.json')['selected']['checkpoint']).parent
    large=HERE/'results'/options.large_run
    a,b=read(control/'config.json'),read(large/'config.json')
    excluded={'data_dir','manifest_sha256','continued_from','parent_checkpoint_sha256'}
    if {k:v for k,v in a.items() if k not in excluded}!={k:v for k,v in b.items() if k not in excluded}:
        raise ValueError('Training settings differ beyond dataset size/identity')
    for root,count in [(control,1500),(large,10000)]:
        with (root/'training_history.csv').open(newline='',encoding='utf-8') as f:history=list(csv.DictReader(f))
        if len(history)!=50 or any(int(row['train_n'])!=count or int(row['val_n'])!=200 for row in history):
            raise ValueError('Training budget/data-count mismatch')
    first=ImagingDataset('test',a.get('input_encoding','amplitude_gray'),a.get('data_dir'))
    second=ImagingDataset('test',b.get('input_encoding','amplitude_gray'),b.get('data_dir'))
    rows1=[r for r in first.manifest['images'] if r['split']=='test']
    rows2=[r for r in second.manifest['images'] if r['split']=='test']
    if rows1!=rows2 or len(first)!=300:raise ValueError('Held-out comparison is not paired')
    records=[]
    for root in [control,large]:
        with (root/'evaluation'/'per_image.csv').open(newline='',encoding='utf-8') as f:records.append(list(csv.DictReader(f)))
    if any([int(r['index']) for r in rows]!=list(range(300)) for rows in records):raise ValueError('Per-image pairing mismatch')
    rng=np.random.default_rng(42);indices=rng.integers(0,300,size=(2000,300));deltas={}
    for metric in ['mse','ssim_amplitude','ssim_intensity']:
        # per_image.csv includes amplitude SSIM and MSE; intensity mean remains
        # available in test_metrics.json but has no per-image record here.
        if metric not in records[0][0] or metric not in records[1][0]:continue
        differences=np.array([float(y[metric])-float(x[metric]) for x,y in zip(*records)])
        ci=np.quantile(differences[indices].mean(1),[.025,.975])
        deltas[metric]={'mean_large_minus_control':float(differences.mean()),'paired_bootstrap_95_percent_ci':ci.tolist()}
    save_json(HERE/'results'/'data_size_comparison.json',{
        'source':'Supplement p10 motivates the data-size comparison; wavelength adaptation is explicitly separate from the original THz experiment',
        'wavelength_nm':299792458/a['geometry']['frequency_hz']*1e9,
        'control_run':control.name,'large_run':large.name,'training_counts':[1500,10000],
        'epochs_per_run':50,'validation_count':200,'paired_test_count':300,
        'control_test':read(control/'evaluation'/'test_metrics.json')['network'],
        'large_test':read(large/'evaluation'/'test_metrics.json')['network'],'paired_deltas':deltas,
        'split_choice':'10000 actual training images; original 200/300 held-out images retained. Larger-run split not specified by authors.',
        'paper_equivalence_threshold_provided':False,
        'warning':'A confidence interval containing zero does not establish equivalence; no numerical equivalence threshold was published.'})
    print('DATA SIZE COMPARISON',deltas,flush=True)


if __name__=='__main__':main()
