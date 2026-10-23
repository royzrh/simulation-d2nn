"""Position checks from saved intensities; no image rescaling or retraining."""

# Allow this tool to run directly while keeping project imports rooted correctly.
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _BootstrapPath
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path
import numpy as np
from train import save_json


def analyze_positions(folder):
    folder=Path(folder)
    metadata=json.loads((folder/'measurement_metrics.json').read_text(encoding='utf-8'))
    rows=[]
    for record in metadata['records']:
        if record['input_offset_mm']!=0 or not record['pattern'].startswith('pinhole_3mm'):continue
        stem=f"{record['beam_fwhm_convention']}_{record['pattern']}_offset0mm"
        with np.load(folder/(stem+'.npz')) as arrays:
            axis=arrays['x_mm'];yy,xx=np.meshgrid(axis,axis,indexing='ij')
            target=arrays['target'];target_power=target.sum()
            target_xy=np.array([(xx*target).sum()/target_power,(yy*target).sum()/target_power])
            for key in ['D2NN','free_segments','free_29p5mm']:
                intensity=arrays[key];power=float(intensity.sum())
                centroid=np.array([(xx*intensity).sum()/power,(yy*intensity).sum()/power])
                peak_iy,peak_ix=np.unravel_index(intensity.argmax(),intensity.shape)
                peak=np.array([axis[peak_ix],axis[peak_iy]])
                near=(xx-target_xy[0])**2+(yy-target_xy[1])**2<=3**2
                rows.append({'beam':record['beam_fwhm_convention'],'pattern':record['pattern'],'mode':key,
                             'target_centroid_mm':target_xy.tolist(),'output_centroid_mm':centroid.tolist(),
                             'output_peak_mm':peak.tolist(),'centroid_error_mm':float(np.linalg.norm(centroid-target_xy)),
                             'peak_error_mm':float(np.linalg.norm(peak-target_xy)),
                             'power_fraction_within_3mm_of_target':float(intensity[near].sum()/power)})
    fits=[]
    for beam in sorted({r['beam'] for r in rows}):
        for mode in ['D2NN','free_segments','free_29p5mm']:
            rr=[r for r in rows if r['beam']==beam and r['mode']==mode]
            inputs=np.array([r['target_centroid_mm'] for r in rr]);design=np.c_[inputs,np.ones(len(inputs))]
            if len(rr)!=4 or np.linalg.matrix_rank(design)!=3:raise ValueError('Four noncollinear positions required')
            observed=np.array([r['output_centroid_mm'] for r in rr])
            coefficients=np.linalg.lstsq(design,observed,rcond=None)[0]
            fits.append({'beam':beam,'mode':mode,'n_positions':len(rr),
                         'centroid_mapping_matrix':coefficients[:2].T.tolist(),
                         'offset_mm':coefficients[2].tolist(),
                         'fit_rms_residual_mm':float(np.sqrt(np.mean((design@coefficients-observed)**2))),
                         'mean_fraction_power_within_target_3mm':float(np.mean([r['power_fraction_within_3mm_of_target'] for r in rr]))})
    save_json(folder/'position_analysis.json',{'source_requirement':'Main paper p3, unit-magnification amplitude imaging',
        'rows':rows,'centroid_fits':fits,'output_rescaling':'none',
        'power_centroid_domain':'Entire saved ~90mm detector scan, no target-centered fitting window',
        'warning':'Centroid mapping near identity is insufficient to establish imaging: free diffraction can preserve centroids while blurring images.',
        'offaxis_positions_are_reconstruction_choices':True,'physical_experiment_reproduced':False})
    print('S8 POSITION ANALYSIS',fits,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path)
    analyze_positions(parser.parse_args().folder)
