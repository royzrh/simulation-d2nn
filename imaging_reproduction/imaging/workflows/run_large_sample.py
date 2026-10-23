"""Visible-terminal queue for the supplementary p10 larger-sample check."""

# Allow this tool to run directly while keeping project imports rooted correctly.
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _BootstrapPath
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import csv
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from train import HERE,ImagingDataset,save_json

SCRIPT_PATHS = {'optics.py': 'train.py', 'data_io.py': 'train.py', 'train.py': 'train.py', 'check_physics.py': 'imaging/validation/check_physics.py', 'acquire_imagenet.py': 'imaging/datasets/acquire_imagenet.py', 'prepare_large_sample.py': 'imaging/datasets/prepare_large_sample.py', 'evaluate.py': 'imaging/evaluation/evaluate.py', 's8_measurement.py': 'imaging/evaluation/s8_measurement.py', 'resolution_profiles.py': 'imaging/evaluation/resolution_profiles.py', 'reconstruct.py': 'imaging/evaluation/reconstruct.py', 'robustness.py': 'imaging/evaluation/robustness.py', 'analyze_s8_positions.py': 'imaging/evaluation/analyze_s8_positions.py', 'compare_large_sample.py': 'imaging/evaluation/compare_large_sample.py', 'error_convergence.py': 'imaging/evaluation/error_convergence.py', 'measure_sweep.py': 'imaging/evaluation/measure_sweep.py', 'figures.py': 'imaging/evaluation/figures.py', 'report.py': 'imaging/evaluation/report.py', 'select_main.py': 'imaging/workflows/select_main.py', 'run_large_sample.py': 'imaging/workflows/run_large_sample.py', 'run_suite.py': 'imaging/workflows/run_suite.py'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def run(script,args,label):
    command=[sys.executable,'-B','-u',str(HERE/SCRIPT_PATHS[script]),*map(str,args)]
    child=subprocess.Popen(command,cwd=HERE)
    save_json(HERE/'misc'/'large_sample_process.json',{'pid':child.pid,'label':label,'command':command})
    print('LARGE-SAMPLE STAGE',label,'PID',child.pid,flush=True)
    if child.wait()!=0:raise RuntimeError('Larger-sample stage failed: '+label)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--control-run',default='wavelength1550_paper_geometry_os16')
    parser.add_argument('--large-run',default='imagenet_train10000_1550nm_os16')
    options=parser.parse_args()
    deadline=time.monotonic()+24*3600;last=0
    control=HERE/'results'/options.control_run
    out=HERE/'results'/options.large_run
    data_dir=HERE/'data_imagenet_train10000'
    try:
        save_json(HERE/'misc'/'large_sample_status.json',{'exit_code':None,'status':'waiting_for_prerequisites'})
        while True:
            preparation=HERE/'misc'/'prepare_large_sample_status.json'
            evaluation=HERE/'misc'/'wavelength1550_status.json'
            if preparation.exists() and read(preparation)['exit_code'] not in (None,0):raise RuntimeError('Additional ImageNet acquisition/preparation failed')
            if evaluation.exists() and read(evaluation)['exit_code'] not in (None,0):raise RuntimeError('1550 nm main-model workflow failed')
            if (data_dir/'data_ready.json').exists() and evaluation.exists() and read(evaluation)['exit_code']==0 and (control/'training_complete.json').exists() and (control/'evaluation'/'test_metrics.json').exists():break
            if time.monotonic()>deadline:raise TimeoutError('Waiting for separate data/main-model prerequisites timed out')
            if time.monotonic()-last>50:
                print('Waiting: separate 10000-image data and completed 1550 nm fine-grid control evaluation.',flush=True);last=time.monotonic()
            time.sleep(10)
        checkpoint=control/'best_model.pt'
        control_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        config=read(checkpoint.parent/'config.json');g=config['geometry']
        train=ImagingDataset('train',config.get('input_encoding','amplitude_gray'),data_dir)
        if len(train)!=10000:raise ValueError('Exactly 10000 training images required')
        for split,count in [('validation',200),('test',300)]:
            old=ImagingDataset(split,config.get('input_encoding','amplitude_gray'))
            new=ImagingDataset(split,config.get('input_encoding','amplitude_gray'),data_dir)
            old_rows=[r for r in old.manifest['images'] if r['split']==split]
            new_rows=[r for r in new.manifest['images'] if r['split']==split]
            if len(new)!=count or old_rows!=new_rows:raise ValueError('Held-out image identities/order differ')
        args=['--name',out.name,'--epochs','50','--batch-size',config['batch_size'],
              '--layers',g['layers'],'--input-mm',g['input_mm'],'--gap-mm',g['interlayer_mm'],
              '--output-mm',g['output_mm'],'--oversample',g['oversample'],'--pad-factor',g['pad_factor'],
              '--init-std',g['init_std'],'--lr',config['lr'],'--seed',config['seed'],
              '--input-encoding',config.get('input_encoding','amplitude_gray'),'--target',config['target'],
              '--data-dir',data_dir]
        args+=['--wavelength-nm',format(299792458/g['frequency_hz']*1e9,'.12g'),
               '--pitch-um',format(g['pitch_m']*1e6,'.12g')]
        if config.get('micro_batch_size'):args+=['--micro-batch-size',config['micro_batch_size']]
        if config.get('activation_checkpointing'):args+=['--activation-checkpointing']
        if not (out/'training_complete.json').exists():
            if (out/'last_checkpoint.pt').exists():args+=['--resume',out/'last_checkpoint.pt']
            save_json(HERE/'misc'/'large_sample_status.json',{'exit_code':None,'status':'training_10000_images','run':out.name})
            run('train.py',args,'train_10000_images_50epochs')
        with (out/'training_history.csv').open(newline='',encoding='utf-8') as f:history=list(csv.DictReader(f))
        if [int(r['epoch']) for r in history]!=list(range(1,51)) or any(int(r['train_n'])!=10000 or int(r['val_n'])!=200 for r in history):
            raise ValueError('Larger-sample training budget/data counts not fulfilled')
        save_json(HERE/'misc'/'large_sample_status.json',{'exit_code':None,'status':'evaluating_10000_image_model','run':out.name})
        run('evaluate.py',['--checkpoint',out/'best_model.pt','--convergence'],'evaluate_10000_image_model')
        run('figures.py',['--checkpoint',out/'best_model.pt'],'index_10000_image_figures')
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=control_sha:raise ValueError('Control checkpoint changed during larger-sample training')
        run('compare_large_sample.py',['--control-run',control.name,'--large-run',out.name],'paired_data_size_comparison')
        run('report.py',[],'report')
        save_json(HERE/'misc'/'large_sample_status.json',{'exit_code':0,'finished_at_unix':time.time(),'full_goal_complete':False})
    except Exception as error:
        save_json(HERE/'misc'/'large_sample_status.json',{'exit_code':1,'error':repr(error)});raise


if __name__=='__main__':main()
