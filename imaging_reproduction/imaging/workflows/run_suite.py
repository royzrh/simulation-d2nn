"""Sequential GPU queue for S6/S7 numerical reproduction; resumable per job.

All execution logs are miscellaneous material. Research configs and measurements
stay under results. Completion is per stage; an incomplete queue is never success.
"""

# Allow this tool to run directly while keeping project imports rooted correctly.
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _BootstrapPath
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import json
import os
import subprocess
import sys
import time
from train import HERE,save_json

SCRIPT_PATHS = {'optics.py': 'train.py', 'data_io.py': 'train.py', 'train.py': 'train.py', 'check_physics.py': 'imaging/validation/check_physics.py', 'acquire_imagenet.py': 'imaging/datasets/acquire_imagenet.py', 'prepare_large_sample.py': 'imaging/datasets/prepare_large_sample.py', 'evaluate.py': 'imaging/evaluation/evaluate.py', 's8_measurement.py': 'imaging/evaluation/s8_measurement.py', 'resolution_profiles.py': 'imaging/evaluation/resolution_profiles.py', 'reconstruct.py': 'imaging/evaluation/reconstruct.py', 'robustness.py': 'imaging/evaluation/robustness.py', 'analyze_s8_positions.py': 'imaging/evaluation/analyze_s8_positions.py', 'compare_large_sample.py': 'imaging/evaluation/compare_large_sample.py', 'error_convergence.py': 'imaging/evaluation/error_convergence.py', 'measure_sweep.py': 'imaging/evaluation/measure_sweep.py', 'figures.py': 'imaging/evaluation/figures.py', 'report.py': 'imaging/evaluation/report.py', 'select_main.py': 'imaging/workflows/select_main.py', 'run_large_sample.py': 'imaging/workflows/run_large_sample.py', 'run_suite.py': 'imaging/workflows/run_suite.py'}


def run(command,label):
    logs=HERE/'misc'/'logs';logs.mkdir(parents=True,exist_ok=True)
    print('RUN',label,flush=True)
    with (logs/(label+'.log')).open('a',encoding='utf-8') as log:
        proc=subprocess.Popen([sys.executable,'-B','-u',str(HERE/SCRIPT_PATHS[command[0]]),*command[1:]],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
        save_json(HERE/'misc'/'active_process.json',{'queue_pid':os.getpid(),'child_pid':proc.pid,'label':label,'command':command,'started_at_unix':time.time()})
        for line in proc.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
        status=proc.wait()
        if status:raise RuntimeError(f'{label} exit={status}; see misc/logs')


def train(name,extra):
    out=HERE/'results'/name
    if (out/'training_complete.json').exists():
        config=json.loads((out/'config.json').read_text(encoding='utf-8'))
        history=(out/'training_history.csv').read_text(encoding='utf-8').strip().splitlines()
        if config['epochs']==50 and len(history)==51 and not config.get('limit_train'):
            print('VERIFIED completed training',name,flush=True);return
        raise RuntimeError('Incomplete/limited run cannot satisfy full training: '+name)
    command=['train.py','--name',name,'--epochs','50',*extra]
    if (out/'last_checkpoint.pt').exists():command+=['--resume',str(out/'last_checkpoint.pt')]
    run(command,'train_'+name)


def adapted_jobs():
    jobs=[]
    for n in [1,2,3,4,5,10]:
        jobs.append((f'layers_n{n}_total30_1550nm_os16',
            ['--layers',str(n),'--gap-mm',str(19/(n-1) if n>1 else 0),'--output-mm',str(7 if n>1 else 26)]))
    for gap in [1,2,3,5,6,7,8,9,10]:
        jobs.append((f'gap_{gap}mm_1550nm_os16',['--layers','5','--gap-mm',str(gap),'--output-mm','7']))
    return jobs


def adapted_1550():
    """Complete S6/S7 scans after the visible main/large-data workflows finish."""
    destination=HERE/'results'/'suite1550_state.json'
    state={'pid':os.getpid(),'status':'waiting_for_prerequisites','wavelength_nm':1550,
           'completed':[],'planned_runs':[name for name,_ in adapted_jobs()],
           'started_at_unix':time.time(),'full_goal_complete':False}
    def update():save_json(destination,state)
    update();deadline=time.monotonic()+7*24*3600;last=0
    try:
        while True:
            statuses=[]
            for name in ['wavelength1550_status.json','large_sample_status.json']:
                path=HERE/'misc'/name
                item=json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}
                if item.get('exit_code') not in (None,0):raise RuntimeError('Prerequisite failed: '+name)
                statuses.append(item.get('exit_code')==0)
            if all(statuses):break
            if time.monotonic()>deadline:raise TimeoutError('Prerequisites still incomplete; no scan training was started')
            if time.monotonic()-last>50:
                print('1550 nm scans queued: waiting for main and 10000-image workflows; no scan optimizer is running.',flush=True);last=time.monotonic()
            time.sleep(10)
        root=HERE/'results'/'wavelength1550_paper_geometry_os16'
        config=json.loads((root/'config.json').read_text(encoding='utf-8'));g=config['geometry']
        if abs(299792458/g['frequency_hz']*1e9-1550)>1e-6 or g['input_mm']<=0:
            raise ValueError('1550 nm and a positive input distance are required')
        if abs(g['input_mm']-4)>1e-9 or abs(g['output_mm']-7)>1e-9:
            raise ValueError('The disclosed 30 mm allocation assumes input4/output7 mm')
        if not (root/'training_complete.json').exists():raise ValueError('Main control training is incomplete')
        if not (HERE/'results'/'imagenet_train10000_1550nm_os16'/'training_complete.json').exists():
            raise ValueError('Larger-data training is incomplete')
        common=['--wavelength-nm','1550','--pitch-um',str(g['pitch_m']*1e6),
                '--input-mm',str(g['input_mm']),'--oversample',str(g['oversample']),
                '--pad-factor',str(g['pad_factor']),'--batch-size',str(config['batch_size']),
                '--lr',str(config['lr']),'--seed',str(config['seed']),'--init-std',str(g['init_std']),
                '--input-encoding',config.get('input_encoding','amplitude_gray'),'--target',config['target']]
        if config.get('micro_batch_size'):common+=['--micro-batch-size',str(config['micro_batch_size'])]
        if config.get('activation_checkpointing'):common+=['--activation-checkpointing']
        if config.get('data_dir'):common+=['--data-dir',config['data_dir']]
        state['status']='running';update()
        run(['measure_sweep.py','--checkpoint',str(root/'best_model.pt')],'measure_1550_gap4_control')
        for name,extra in adapted_jobs():
            state['current_run']=name;update()
            train(name,common+extra)
            ck=HERE/'results'/name/'best_model.pt'
            run(['measure_sweep.py','--checkpoint',str(ck),'--sampling-audit'],'measure_'+name)
            run(['figures.py','--checkpoint',str(ck)],'figures_'+name)
            state['completed'].append(name);update()
            run(['report.py'],'report_'+name)
        state['status']='requested_queue_completed';state['finished_at_unix']=time.time();update()
        run(['report.py'],'report_1550_completed_queue')
    except Exception as error:
        state['status']='failed';state['error']=repr(error);update();raise


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['baseline','all','adapted1550'],default='all')
    a=p.parse_args()
    if a.stage=='adapted1550':return adapted_1550()
    state={'started_at_unix':time.time(),'pid':os.getpid(),'status':'running','completed':[]}
    def done(label):
        state['completed'].append(label);save_json(HERE/'results'/'suite_state.json',state)
    try:
        train('baseline_os2',[]);done('baseline_50epochs')
        ck=HERE/'results'/'baseline_os2'/'best_model.pt'
        run(['evaluate.py','--checkpoint',str(ck),'--convergence'],'baseline_evaluation');done('baseline_evaluation_and_numerics')
        run(['reconstruct.py','--checkpoint',str(ck)],'poisson_reconstruction');done('screened_poisson_surface_reconstruction')
        run(['robustness.py','--checkpoint',str(ck)],'combined_error_evaluation');done('combined_error_evaluation')
        if a.stage=='all':
            # S7A left: published 30mm total. Segment allocation is undocumented.
            # Preserve d0=4mm and dout=7mm for N>=2; the remaining 19mm spans masks.
            for n in [1,2,3,4,5,10]:
                gap=19/(n-1) if n>1 else 0
                output=7 if n>1 else 26
                name=f'layers_n{n}_total30_os2'
                train(name,['--layers',str(n),'--gap-mm',str(gap),'--output-mm',str(output)])
                run(['measure_sweep.py','--checkpoint',str(HERE/'results'/name/'best_model.pt')],'measure_'+name)
                done(name)
            # S7A right: 5 masks, d0=4mm/dout=7mm fixed. No undisclosed endpoint rescaling.
            for gap in [1,2,3,5,6,7,8,9,10]:
                name=f'gap_{gap}mm_os2'
                train(name,['--gap-mm',str(gap)])
                run(['measure_sweep.py','--checkpoint',str(HERE/'results'/name/'best_model.pt')],'measure_'+name)
                done(name)
            run(['measure_sweep.py','--checkpoint',str(ck)],'measure_baseline_gap4')
            run(['report.py'],'build_report');done('report')
        state['status']='requested_queue_completed';state['finished_at_unix']=time.time()
        save_json(HERE/'results'/'suite_state.json',state)
    except Exception as e:
        state['status']='failed';state['error']=repr(e);save_json(HERE/'results'/'suite_state.json',state);raise


if __name__=='__main__':main()
