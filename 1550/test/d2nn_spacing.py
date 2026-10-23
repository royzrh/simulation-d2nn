"""1550 nm five-layer D2NN with four independently specified propagation gaps.

Commands: train, search, refine, evaluate, reference, plot, report, all.
The four interlayer gaps sum to 400 mm; the input/output gaps stay 30/100 mm.
Default controlled experiment: 1,000 training and 1,000 validation images,
nine candidates for five epochs, then the baseline and validation-selected
nonuniform candidate continue to 15 total epochs before official testing.
Each candidate shares initialization, split and epoch-specific shuffle seeds.
All generated files, copied MNIST data and caches stay beside this file.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, random, shutil, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
for _key, _folder in [('MPLCONFIGDIR','mpl'),('TMP','tmp'),('TEMP','tmp')]:
    (ROOT/_folder).mkdir(exist_ok=True)
    os.environ[_key] = str(ROOT/_folder)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
EPS=1e-12
CANDIDATES=[(100,100,100,100),(50,75,125,150),(150,125,75,50),(75,125,125,75),(125,75,75,125),(50,150,50,150),(150,50,150,50),(75,75,125,125),(125,125,75,75)]

def write_json(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8'); tmp.replace(path)

def write_csv(path, rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def save_checkpoint(path,payload):
    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)

def candidate_name(z):return 'z_'+'_'.join(f'{v:g}' for v in z)

class OpticalD2NN(nn.Module):
    def __init__(self,z_mm,over=4,pad_factor=2):
        super().__init__()
        if len(z_mm)!=4 or any(not math.isfinite(v) or v<=0 for v in z_mm) or abs(sum(z_mm)-400)>1e-7:
            raise ValueError('Four positive interlayer distances must sum to 400 mm')
        self.z_mm=list(z_mm);self.over=over;self.active_n=200*over;self.fft_n=self.active_n*pad_factor
        self.dx=50e-6/over;self.left=(self.fft_n-self.active_n)//2;self.right=self.fft_n-self.active_n-self.left
        self.s=slice(self.left,self.left+self.active_n)
        self.raw_phases=nn.Parameter(.7*torch.randn(5,200,200))
        aperture=torch.zeros(self.fft_n,self.fft_n);aperture[self.s,self.s]=1
        self.register_buffer('aperture',aperture,persistent=False)
        f=torch.fft.fftfreq(self.fft_n,d=self.dx,dtype=torch.float64);fy,fx=torch.meshgrid(f,f,indexing='ij')
        k=2*math.pi/1550e-9;kz=k*torch.sqrt((1-(1550e-9*fx)**2-(1550e-9*fy)**2).to(torch.complex128))
        for i,z in enumerate([30]+list(z_mm)+[100]):
            self.register_buffer(f'H{i}',torch.exp(1j*(kz-k)*(z*1e-3)).to(torch.complex64),persistent=False)
        self.detector_boxes=[]
        side=round(.6e-3/self.dx)
        for y,xs in [(12,[-12,0,12]),(0,[-18,-6,6,18]),(-12,[-12,0,12])]:
            for x in xs:
                ix=round(x*1e-3*.125/self.dx)+self.fft_n//2;iy=round(y*1e-3*.125/self.dx)+self.fft_n//2
                self.detector_boxes.append((iy-side//2,iy-side//2+side,ix-side//2,ix-side//2+side))
    def encode(self,images,threshold=.5,scale=1):
        amp=(images[:,0]>=threshold).float().repeat_interleave(self.over*scale,-2).repeat_interleave(self.over*scale,-1)
        a=(self.active_n-amp.shape[-1])//2;b=self.active_n-amp.shape[-1]-a
        amp=F.pad(amp,(a+self.left,b+self.right,a+self.left,b+self.right))
        return (amp/amp.square().sum((-2,-1),keepdim=True).clamp_min(EPS).sqrt()).to(torch.complex64)
    def propagate(self,field,h):return torch.fft.ifft2(torch.fft.fft2(field)*h)
    def forward(self,field):
        field=self.propagate(field,self.H0)
        for i in range(5):
            phi=math.pi*torch.sigmoid(self.raw_phases[i]);phi=phi.repeat_interleave(self.over,-2).repeat_interleave(self.over,-1)
            phi=F.pad(phi,(self.left,self.right,self.left,self.right))
            field=field*self.aperture*torch.exp(1j*phi)
            field=self.propagate(field,getattr(self,f'H{i+1}'))
        intensity=field.abs().square()
        powers=torch.stack([intensity[:,y0:y1,x0:x1].sum((-2,-1)) for y0,y1,x0,x1 in self.detector_boxes],1)
        return intensity,powers

def compute_loss(power,intensity,labels,input_power,active_slice):
    correct=power.gather(1,labels[:,None]).squeeze(1);capture=power.sum(1);total=intensity.sum((-2,-1)).clamp_min(EPS)
    ce=F.cross_entropy(power.clamp_min(EPS).log(),labels);eff=correct/input_power.clamp_min(EPS)
    loss=ce+.5*(1-eff).mean()+.05*(1-capture/total).clamp(0,1).mean()
    return loss,{'loss':loss,'ce':ce,'correct_efficiency':eff.mean(),'detector_capture':(capture/input_power.clamp_min(EPS)).mean(),'fft_window_efficiency':(total/input_power.clamp_min(EPS)).mean(),'active_window_efficiency':(intensity[:,active_slice,active_slice].sum((-2,-1))/input_power.clamp_min(EPS)).mean()}

def prepare_data(a):
    target=ROOT/'data'/'MNIST'
    if not target.exists():
        sources=[ROOT.parent/'data'/'MNIST',ROOT.parent.parent/'data'/'MNIST']
        source=next((p for p in sources if p.is_dir()),None)
        if source is None:raise FileNotFoundError('Place existing MNIST files in test/data/MNIST; downloads are disabled')
        shutil.copytree(source,target)
    train=datasets.MNIST(str(ROOT/'data'),train=True,download=False,transform=transforms.ToTensor())
    permutation=torch.randperm(len(train),generator=torch.Generator().manual_seed(a.seed))
    split={'train':permutation[:a.train_count].tolist(),'validation':permutation[a.train_count:a.train_count+a.val_count].tolist()}
    if a.train_count+a.val_count>len(train):raise ValueError('Split exceeds dataset')
    split['sha256']=hashlib.sha256(json.dumps(split,sort_keys=True).encode()).hexdigest()
    split_path=a.root/'split.json'
    if split_path.exists() and json.loads(split_path.read_text())!=split:raise ValueError('Existing split differs')
    write_json(split_path,split)
    return train,split

def loader(ds,indices,a,epoch=None):
    return DataLoader(Subset(ds,indices),batch_size=a.batch_size,shuffle=epoch is not None,num_workers=0,pin_memory=a.device=='cuda',generator=torch.Generator().manual_seed(a.seed+1000+(epoch or 0)))

def run_epoch(model,dl,a,optimizer=None,collect=False):
    model.train(optimizer is not None);sums={};n=0;right=0;predictions=[];cm=torch.zeros(10,10,dtype=torch.int64)
    with torch.set_grad_enabled(optimizer is not None):
        for batch,(images,labels) in enumerate(dl,1):
            images=images.to(a.device);labels=labels.to(a.device);field=model.encode(images)
            if optimizer:optimizer.zero_grad(set_to_none=True)
            intensity,power=model(field);loss,stats=compute_loss(power,intensity,labels,field.abs().square().sum((-2,-1)),model.s)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            if optimizer:
                loss.backward()
                if not torch.isfinite(model.raw_phases.grad).all():raise FloatingPointError('Nonfinite gradients')
                optimizer.step()
            b=len(labels);n+=b;pred=power.argmax(1);right+=int((pred==labels).sum())
            for k,v in stats.items():sums[k]=sums.get(k,0)+float(v.detach())*b
            if collect:
                yp=labels.cpu().tolist();pp=pred.cpu().tolist();co=power.gather(1,labels[:,None]).squeeze(1).detach().cpu().tolist();ca=power.sum(1).detach().cpu().tolist()
                for y,p,e,c in zip(yp,pp,co,ca):
                    cm[y,p]+=1;predictions.append({'index':len(predictions),'label':y,'prediction':p,'correct_efficiency':e,'detector_capture':c})
            if batch%50==0:print(f'  batch {batch}/{len(dl)} {"train" if optimizer else "eval"}',flush=True)
    return {**{k:v/n for k,v in sums.items()},'accuracy':right/n,'n':n},predictions,cm.tolist()

def config(a,z,split):
    return {'z_mm':list(z),'input_gap_mm':30,'output_gap_mm':100,'interlayer_total_mm':400,'total_length_mm':530,'wavelength_nm':1550,'pitch_um':50,'aperture_mm':10,'layers':5,'neurons_per_side':200,'oversample':a.oversample,'pad_factor':a.pad_factor,'threshold':.5,'input_scale':1,'phase':'pi*sigmoid','loss':'baseline: CE(log(power)) + .5*(1-correct/input) + .05*(1-capture/output)','lr':a.lr,'batch_size':a.batch_size,'seed':a.seed,'train_count':a.train_count,'val_count':a.val_count,'split_sha256':split['sha256'],'torch_version':str(torch.__version__),'device':torch.cuda.get_device_name() if a.device=='cuda' else 'cpu'}

def train_candidate(a,z,epochs,ds,split):
    out=a.root/'runs'/candidate_name(z);out.mkdir(parents=True,exist_ok=True);cfg=config(a,z,split)
    torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    model=OpticalD2NN(z,a.oversample,a.pad_factor).to(a.device);optimizer=torch.optim.Adam(model.parameters(),lr=a.lr)
    hist=[];best_acc=-1.;best_loss=float('inf');best_epoch=0;start=1
    last=out/'last_checkpoint.pt';best=out/'best_model.pt'
    if last.exists():
        cp=torch.load(last,map_location=a.device,weights_only=False)
        if cp['config']!=cfg:raise ValueError('Resume config differs')
        model.load_state_dict(cp['model_state_dict']);optimizer.load_state_dict(cp['optimizer_state_dict']);hist=cp['history'];start=cp['epoch']+1
        best_acc=cp['best_validation_accuracy'];best_loss=cp['best_validation_loss'];best_epoch=cp['best_epoch']
    write_json(out/'config.json',cfg)
    for epoch in range(start,epochs+1):
        t=time.perf_counter();print(f'{candidate_name(z)} epoch {epoch}/{epochs}',flush=True)
        if a.device=='cuda':torch.cuda.reset_peak_memory_stats()
        tr,_,_=run_epoch(model,loader(ds,split['train'],a,epoch),a,optimizer)
        va,_,_=run_epoch(model,loader(ds,split['validation'],a),a)
        row={'epoch':epoch,**{'train_'+k:v for k,v in tr.items()},**{'val_'+k:v for k,v in va.items()},'seconds':time.perf_counter()-t,'peak_gpu_gb':torch.cuda.max_memory_allocated()/1024**3 if a.device=='cuda' else 0};hist.append(row)
        if va['accuracy']>best_acc or (va['accuracy']==best_acc and va['loss']<best_loss):
            best_acc=va['accuracy'];best_loss=va['loss'];best_epoch=epoch
            save_checkpoint(best,{'model_state_dict':model.state_dict(),'config':cfg,'epoch':epoch,'validation':va})
        save_checkpoint(last,{'model_state_dict':model.state_dict(),'optimizer_state_dict':optimizer.state_dict(),'config':cfg,'epoch':epoch,'history':hist,'best_validation_accuracy':best_acc,'best_validation_loss':best_loss,'best_epoch':best_epoch})
        write_csv(out/'history.csv',hist);write_json(out/'history.json',hist)
        print(f'  val={va["accuracy"]:.3%} correct={va["correct_efficiency"]:.3%} capture={va["detector_capture"]:.3%} seconds={row["seconds"]:.1f} peak={row["peak_gpu_gb"]:.2f} GB',flush=True)
    summary={'name':candidate_name(z),'z_mm':list(z),'epochs':len(hist),'best_epoch':best_epoch,'best_validation_accuracy':best_acc,'best_validation_loss':best_loss,'best_validation_correct_efficiency':hist[best_epoch-1]['val_correct_efficiency'],'best_validation_detector_capture':hist[best_epoch-1]['val_detector_capture']}
    write_json(out/'summary.json',summary);del optimizer,model
    if a.device=='cuda':torch.cuda.empty_cache()
    return summary

def search(a,ds,split):
    rows=[train_candidate(a,z,a.search_epochs,ds,split) for z in CANDIDATES]
    # Read exactly the first search_epochs: a later resume/refine cannot change the search decision.
    for row in rows:
        history=json.loads((a.root/'runs'/row['name']/'history.json').read_text())[:a.search_epochs]
        best=min(history,key=lambda r:(-r['val_accuracy'],r['val_loss']))
        row.update(epochs=a.search_epochs,best_epoch=best['epoch'],best_validation_accuracy=best['val_accuracy'],best_validation_loss=best['val_loss'],best_validation_correct_efficiency=best['val_correct_efficiency'],best_validation_detector_capture=best['val_detector_capture'])
    winner=min(rows[1:],key=lambda r:(-r['best_validation_accuracy'],r['best_validation_loss']))
    payload={'search_epochs':a.search_epochs,'candidates':rows,'selected_nonuniform':winner,'selection':'highest validation accuracy, tie broken by lower validation baseline loss; no test access'}
    write_json(a.root/'search_results.json',payload);write_csv(a.root/'search_results.csv',[{**r,'z_mm':','.join(map(str,r['z_mm']))} for r in rows]);return payload

def refine(a,ds,split):
    selection=json.loads((a.root/'search_results.json').read_text());z=selection['selected_nonuniform']['z_mm']
    summaries=[train_candidate(a,zz,a.epochs,ds,split) for zz in [CANDIDATES[0],z]]
    write_json(a.root/'refine_results.json',{'models':summaries,'selection':f'best validation checkpoint during all {a.epochs} epochs'})

def evaluate(a):
    selection=json.loads((a.root/'refine_results.json').read_text());test=datasets.MNIST(str(ROOT/'data'),train=False,download=False,transform=transforms.ToTensor());results=[]
    for row in selection['models']:
        out=a.root/'runs'/row['name'];cp=torch.load(out/'best_model.pt',map_location=a.device,weights_only=False)
        cfg=cp['config']
        if list(cfg['z_mm'])!=list(row['z_mm']):raise ValueError('Checkpoint and selection distances differ')
        model=OpticalD2NN(cfg['z_mm'],cfg['oversample'],cfg['pad_factor']).to(a.device);model.load_state_dict(cp['model_state_dict'])
        print(f'Official test: {row["name"]} checkpoint epoch {cp["epoch"]}',flush=True)
        metrics,pred,cm=run_epoch(model,loader(test,list(range(len(test))),a),a,collect=True)
        write_csv(out/'test_predictions.csv',pred);write_json(out/'test_metrics.json',{'metrics':metrics,'confusion':cm,'checkpoint_epoch':cp['epoch']})
        results.append({**row,'test':metrics,'confusion':cm});del model
        if a.device=='cuda':torch.cuda.empty_cache()
    paired=[]
    for r in results:
        with (a.root/'runs'/r['name']/'test_predictions.csv').open(encoding='utf-8-sig') as f:paired.append(list(csv.DictReader(f)))
    b=sum(x['label']==x['prediction'] and y['label']!=y['prediction'] for x,y in zip(*paired));c=sum(x['label']!=x['prediction'] and y['label']==y['prediction'] for x,y in zip(*paired))
    p=math.erfc(abs(b-c)/math.sqrt(2*(b+c))) if b+c else 1.
    payload={'models':results,'paired_comparison':{'baseline_only_correct':b,'nonuniform_only_correct':c,'mcnemar_chi2_uncorrected':(b-c)**2/(b+c) if b+c else 0,'mcnemar_asymptotic_p':p},'notes':['Single seed and fixed small train/validation subsets; no population-level superiority claim.','Test data used only after both models and checkpoints were fixed.','Efficiencies are integrated intensity relative to normalized input, not calibrated Poynting flux.']}
    write_json(a.root/'final_results.json',payload);write_csv(a.root/'final_results.csv',[{'name':r['name'],'z_mm':','.join(map(str,r['z_mm'])),'best_epoch':r['best_epoch'],**r['test']} for r in results]);plot_results(a.root)

def evaluate_reference(a):
    """Re-evaluate the user's copied prior best model; never use it for selection."""
    folder=ROOT/'reference'
    checkpoint=folder/'best_model.pt'
    metadata=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    cp=torch.load(checkpoint,map_location='cpu',weights_only=True)
    cfg=cp['config']
    expected={'wavelength_m':1550e-9,'pitch_m':50e-6,'interlayer_gap_m':.1,
              'output_gap_m':.1,'input_gap_mm':30,'detector_side_m':.6e-3,
              'detector_coordinate_scale':.125,'neurons_per_side':200,'layers':5,
              'threshold':.5,'input_pixel_scale':1}
    if cfg.get('loss')!='baseline' or any(not math.isclose(float(cfg[k]),v,rel_tol=1e-9,abs_tol=1e-15) for k,v in expected.items()):
        raise ValueError('The copied reference geometry/encoding differs from this experiment')
    model=OpticalD2NN([100]*4,cfg['oversample'],cfg['pad_factor']).to(a.device)
    model.load_state_dict(cp['model_state_dict'])
    ds=datasets.MNIST(str(ROOT/'data'),train=False,download=False,transform=transforms.ToTensor())
    print(f'Prior best reference: {cfg["train_count"]} training images, checkpoint epoch {cp["epoch"]}; official test only',flush=True)
    metrics,pred,cm=run_epoch(model,loader(ds,list(range(len(ds))),a),a,collect=True)
    out=a.root/'reference_evaluation';out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'test_predictions.csv',pred)
    write_json(out/'test_metrics.json',{'metrics':metrics,'confusion':cm,'checkpoint_epoch':cp['epoch'],
               'training_count':cfg['train_count'],'selection_role':'historical reference, excluded from search'})
    metadata.update(reevaluated_test=metrics,reevaluated_confusion=cm,
                    copied_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    reevaluation_accuracy_difference=metrics['accuracy']-metadata['test']['accuracy'])
    write_json(a.root/'historical_reference.json',metadata)
    print(f'Prior best reference official test accuracy={metrics["accuracy"]:.4%}',flush=True)
    del model
    if a.device=='cuda':torch.cuda.empty_cache()

def plot_results(root):
    root=Path(root)
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':9})
    rows=json.loads((root/'search_results.json').read_text())['candidates']
    names=['Equal','Nonuniform']; shades=['#222222','#999999']
    fig,axes=plt.subplots(3,1,figsize=(7.1,6.8))
    for ax,key,title in zip(axes,
            ['best_validation_accuracy','best_validation_correct_efficiency','best_validation_detector_capture'],
            ['Validation accuracy (%)','Correct-detector efficiency (%)','Total detector capture (%)']):
        values=[r[key]*100 for r in rows]
        bars=ax.bar(range(len(rows)),values,color=[shades[0]]+[shades[1]]*(len(rows)-1),width=.65)
        ax.bar_label(bars,labels=[f'{v:.3f}' for v in values],fontsize=8,padding=3)
        ax.set_xticks(range(len(rows)))
        ax.set(xlabel='Candidate ID (distance vectors in report / CSV)',ylabel='Percent',title=title,
               ylim=(0,max(values)*1.22))
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.tight_layout(h_pad=1.2);fig.savefig(root/'search_comparison.png',dpi=220);plt.close(fig)
    if not (root/'final_results.json').exists():
        return
    final=json.loads((root/'final_results.json').read_text());models=final['models']
    metric_keys=['accuracy','correct_efficiency','detector_capture']
    metric_titles=['Accuracy (%)','Correct-detector efficiency (%)','Total detector capture (%)']
    fig,axes=plt.subplots(2,2,figsize=(8,6.5))
    for ax,key,title in zip(axes.flat[:3],metric_keys,metric_titles):
        for i,r in enumerate(models):
            hist=json.loads((root/'runs'/r['name']/'history.json').read_text())
            ax.plot([h['epoch'] for h in hist],[h['val_'+key]*100 for h in hist],
                    label=names[i],linestyle=['-','--'][i],color=shades[i])
        ax.set(xlabel='Epoch',ylabel='Percent',title='Validation: '+title)
        ax.legend();ax.grid(alpha=.2)
    ax=axes.flat[3]
    for i,r in enumerate(models):
        ax.bar(np.arange(4)+i*.35,r['z_mm'],width=.35,label=names[i],color=shades[i])
    ax.set(xticks=np.arange(4)+.175,xticklabels=['z1','z2','z3','z4'],
           ylabel='Distance (mm)',title='Interlayer sum = 400 mm')
    ax.legend();fig.tight_layout();fig.savefig(root/'final_comparison.png',dpi=220);plt.close(fig)

    train_count=json.loads((root/'runs'/models[0]['name']/'config.json').read_text())['train_count']
    test_rows=[(f'{names[i]}\n{train_count:,} train',r['test']) for i,r in enumerate(models)]
    history_path=root/'historical_reference.json'
    if history_path.exists():
        history=json.loads(history_path.read_text(encoding='utf-8'))
        if 'reevaluated_test' in history:
            test_rows.append((f'Prior best\n{history["config"]["train_count"]:,} train',history['reevaluated_test']))
    fig,axes=plt.subplots(1,3,figsize=(10,3.8))
    for ax,key,title in zip(axes,metric_keys,metric_titles):
        values=[r[key]*100 for _,r in test_rows]
        bars=ax.bar([name for name,_ in test_rows],values,color=(shades+['#dddddd'])[:len(test_rows)],width=.55)
        ax.bar_label(bars,labels=[f'{v:.3f}' for v in values],padding=3)
        ax.set(title=title,ylim=(0,max(values)*1.2));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Official MNIST test: 10,000 images, validation-selected checkpoints',fontsize=10)
    fig.tight_layout();fig.savefig(root/'test_comparison.png',dpi=220);plt.close(fig)

    # A single aligned sheet directly relates each distance vector to all three search metrics.
    fig,axes=plt.subplots(1,4,figsize=(12,4.8),gridspec_kw={'width_ratios':[1.7,1,1,1]})
    z=np.asarray([r['z_mm'] for r in rows]);axes[0].imshow(z,cmap='Greys',aspect='auto',vmin=0,vmax=160)
    for j in range(len(rows)):
        for k in range(4):axes[0].text(k,j,f'{z[j,k]:g}',ha='center',va='center',color='white' if z[j,k]>=100 else 'black')
    axes[0].set(xticks=range(4),xticklabels=['z1','z2','z3','z4'],yticks=range(len(rows)),
                yticklabels=[str(j) for j in range(len(rows))],ylabel='Candidate ID',title='Distances (mm); sum = 400')
    for ax,key,title in zip(axes[1:],
            ['best_validation_accuracy','best_validation_correct_efficiency','best_validation_detector_capture'],
            ['Validation\naccuracy (%)','Correct detector\nefficiency (%)','Total detector\ncapture (%)']):
        values=[r[key]*100 for r in rows]
        ax.barh(range(len(rows)),values,color=[shades[0]]+[shades[1]]*(len(rows)-1))
        ax.set(yticks=range(len(rows)),yticklabels=[],ylim=(len(rows)-.5,-.5),xlabel='Percent',title=title)
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(root/'spacing_metrics.png',dpi=220);plt.close(fig)
    for i,r in enumerate(models):
        fig,ax=plt.subplots(figsize=(6,5));im=ax.imshow(r['confusion'],cmap='Greys');ax.set(xlabel='Prediction',ylabel='Label',title=['Equal spacing','Nonuniform spacing'][i],xticks=range(10),yticks=range(10));fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(root/f'confusion_{i}.png',dpi=160);plt.close(fig)

def write_manifest(a):
    """Record code and environment separately so report edits never invalidate resume."""
    source=Path(__file__).read_text(encoding='utf-8')
    training_source=source.split('# REPORT_SECTION_START')[0]+source.split('# REPORT_SECTION_END')[-1]
    path=a.root/'manifest.json'
    if not path.exists():
        write_json(path,{'training_source_sha256':hashlib.sha256(training_source.encode()).hexdigest(),'whole_source_sha256':hashlib.sha256(source.encode()).hexdigest(),'python':os.sys.version,'torch':str(torch.__version__),'cuda':torch.version.cuda,'detector_side_mm':.6,'detector_coordinate_scale':.125,'data_download':False,'initialization':'.7*randn(5,200,200), seed42','shuffle':'seed + 1000 + epoch','candidate_grid_mm':[list(z) for z in CANDIDATES],'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})

# REPORT_SECTION_START
def build_report(root):
    """Create a five-page Chinese report from completed, measured artifacts only."""
    import sys
    try:
        import reportlab
    except ImportError:
        sys.path.append(r'C:\Users\dell\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\site-packages')
    from reportlab.pdfgen import canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, Table, TableStyle
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.utils import ImageReader
    root=Path(root)
    search=json.loads((root/'search_results.json').read_text(encoding='utf-8'))
    final=json.loads((root/'final_results.json').read_text(encoding='utf-8'))
    history_path=root/'historical_reference.json'
    history=json.loads(history_path.read_text(encoding='utf-8')) if history_path.exists() else None
    rows=search['candidates']; models=final['models']; base,chosen=models
    cfg=json.loads((root/'runs'/base['name']/'config.json').read_text(encoding='utf-8'))
    pdfmetrics.registerFont(TTFont('CN',r'C:\Windows\Fonts\simhei.ttf'))
    width,height=595.28,841.89; margin=44; usable=width-2*margin
    path=root/'d2nn_spacing_report.pdf'
    c=canvas.Canvas(str(path),pagesize=(width,height));c.setTitle('1550 nm 五层 D2NN 固定总长层间距实验')
    ps=ParagraphStyle('body',fontName='CN',fontSize=10,leading=16,textColor=colors.black,wordWrap='CJK')
    small=ParagraphStyle('small',parent=ps,fontSize=8.5,leading=13)
    y=0
    def check_space(required):
        if y-required<45:
            raise ValueError('PDF content exceeds the page body; reduce layout height before delivery')
    def page(num,title):
        nonlocal y
        if num>1:c.showPage()
        c.setFont('CN',8);c.drawString(margin,height-29,'D2NN / 1550 nm / 固定总长距离搜索')
        c.line(margin,height-37,width-margin,height-37)
        c.setFont('CN',17);c.drawString(margin,height-68,title)
        c.setFont('CN',8);c.drawRightString(width-margin,25,f'{num} / 5')
        y=height-91
    def text(body,compact=False):
        nonlocal y
        p=Paragraph(body,small if compact else ps);_,h=p.wrap(usable,700)
        check_space(h)
        p.drawOn(c,margin,y-h);y-=h+10
    def heading(body):
        nonlocal y
        c.setFont('CN',11);c.drawString(margin,y-12,body);y-=25
    def table(data,widths=None):
        nonlocal y
        cells=[[Paragraph(str(v),small) for v in row] for row in data]
        t=Table(cells,colWidths=widths or [usable/len(data[0])]*len(data[0]),hAlign='LEFT')
        t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'MIDDLE'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#ededed')),('LINEABOVE',(0,0),(-1,0),.7,colors.black),('LINEBELOW',(0,0),(-1,0),.5,colors.black),('LINEBELOW',(0,-1),(-1,-1),.7,colors.black),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
        _,h=t.wrap(usable,700);check_space(h);t.drawOn(c,margin,y-h);y-=h+13
    def picture(name,max_h):
        nonlocal y
        img=ImageReader(str(root/name));iw,ih=img.getSize();scale=min(usable/iw,max_h/ih)
        w,h=iw*scale,ih*scale;check_space(h);c.drawImage(img,margin+(usable-w)/2,y-h,width=w,height=h);y-=h+10
    def pct(v):return f'{100*v:.3f}%'
    def gaps(r):return ' / '.join(f'{v:g}' for v in r['z_mm'])
    delta=100*(chosen['test']['accuracy']-base['test']['accuracy'])
    train_n=cfg.get('train_count',1000);val_n=cfg.get('val_count',1000)
    page(1,'五层 D2NN：固定总长的非等间距搜索')
    text('本报告比较等间距与预设非等间距的受控训练结果。仅改变四段层间距离；四段之和固定为 400 mm，输入和输出传播分别固定为 30 mm 与 100 mm，全光路为 530 mm。')
    heading('实际结果摘要')
    text(f'九组筛选选出的非等间距：{gaps(chosen)} mm。<br/>官方测试准确率：等距 {pct(base["test"]["accuracy"])}；非等距 {pct(chosen["test"]["accuracy"])}；差值 {delta:+.3f} 个百分点。<br/>两组均训练满 {base.get("epochs",15)} 轮。本结论仅适用于当前固定样本、单随机种子的训练预算。')
    text(f'测试集正确探测效率：等间距 {pct(base["test"]["correct_efficiency"])}，非等间距 {pct(chosen["test"]["correct_efficiency"])}；总捕获率分别为 {pct(base["test"]["detector_capture"])}、{pct(chosen["test"]["detector_capture"])}。候选排序以验证准确率为主，能量指标同时列出比较。',True)
    if history:
        hm=history.get('reevaluated_test',history['test'])
        mode='同一测试集复测' if 'reevaluated_test' in history else '历史记录'
        text(f'另列您原最佳模型作参考：{history["config"]["train_count"]} 张训练图像、15 轮，{mode}准确率 {pct(hm["accuracy"])}。它未参与本轮距离选择，训练数据量与本轮不同。',True)
    text(f'短程筛选时，等间距验证准确率为 {pct(rows[0]["best_validation_accuracy"])}，所选非等间距为 {pct(search["selected_nonuniform"]["best_validation_accuracy"])}；延长训练后分别为 {pct(base["best_validation_accuracy"])}、{pct(chosen["best_validation_accuracy"])}。短程排名不能直接视为最终测试优劣。',True)
    table([['项目','设置'],['相位层 / 波长','5 层；1550 nm'],['相位像素 / 间距 / 孔径','200 × 200；50 μm；10 mm × 10 mm'],['输入 / 输出传播','30 mm / 100 mm'],['四段层间总长','z1 + z2 + z3 + z4 = 400 mm'],['探测器','10 个；每个 0.6 mm × 0.6 mm；坐标比例 0.125'],['数值采样','OS = 4；Pad = 2；dx = 12.5 μm；FFT 1600 × 1600'],['输入设置','像素缩放 1；每个 MNIST 像素对应 1 个 50 μm 物理神经元；二值阈值 0.5'],['优化设置',f'Adam；学习率 {cfg.get("lr",0.005)}；batch {cfg.get("batch_size",8)}；seed {cfg.get("seed",42)}']],[147,usable-147])
    text('数据来源：本次 results 下的配置、搜索结果、训练历史及官方测试结果文件。历史 3000 张训练样本实验不作为本次距离比较的受控基线。',True)
    page(2,'实验预算与九组距离')
    text(f'MNIST 固定训练子集 {train_n} 张、验证子集 {val_n} 张。所有候选共享数据划分、初相位、每轮样本顺序及优化设置。每组先训练 {search.get("search_epochs",5)} 轮；只用验证准确率选择非等间距，平手时取验证损失更低者。')
    steps=math.ceil(train_n/cfg.get('batch_size',8))
    text(f'每候选短程为 {steps*search.get("search_epochs",5)} 次 Adam 更新；延长到 {base.get("epochs",15)} 轮后为 {steps*base.get("epochs",15)} 次。原最佳模型用 3000 张训练图像，即使同为 15 轮，其训练更新预算也更大。',True)
    data=[['编号','z1 / z2 / z3 / z4 (mm)','最佳轮','验证准确率']]
    for i,r in enumerate(rows):data.append([str(i),gaps(r),str(r['best_epoch']),pct(r['best_validation_accuracy'])])
    table(data,[34,255,60,usable-349])
    text('编号 0 为等间距；编号与下一页柱图对应。表中指标取各组短程训练期间的验证最佳检查点，不能直接当作最终测试性能。',True)
    heading('延长训练与独立测试')
    text(f'等间距和筛选获胜的非等间距均延长到 {base.get("epochs",15)} 轮；从短程训练的最后状态继续并保留优化器状态。分别从完整训练历史中按验证指标选择检查点。配置和检查点固定后，才评价官方测试集 {base["test"].get("n",10000)} 张图像。')
    text('相位约束保留原实现：φ = π · sigmoid(raw_phases)，范围为 0 至 π。距离变化之外，不改变相位参数化、层数、采样、探测器或损失函数。')
    text('Baseline 沿用原脚本：分类交叉熵 CE(log P, y) + 0.5 × (1 - Py / Pin) + 0.05 × (1 - Pdet / Pout)，对样本取平均。Py 为真实类别探测功率，Pdet 为十个探测区域功率之和，Pout 为输出 FFT 窗口内总功率。',True)
    page(3,'短程搜索：分类与能量指标')
    picture('search_comparison.png',430)
    text('图 1：九组短程训练的验证最佳检查点。横轴编号对应上一页的四段距离；柱顶标出实际百分比数值。同一编号的三项指标均取该候选按验证准确率选择的同一个检查点。',True)
    text('正确探测效率 = 真实类别探测区域积分强度 / 输入积分强度；总探测捕获率 = 全部 10 个探测区域积分强度之和 / 输入积分强度。每张输入已归一化，总输入积分强度为 1；报告为样本平均值。这些是标量模型的积分强度比例，未经绝对光功率标定。',True)
    page(4,'延长训练与官方测试')
    picture('final_comparison.png',380)
    text('图 2：两组完整训练的验证曲线。实线 / 深色为等间距，虚线 / 浅色为非等间距；右下为四段距离，二者总和均为 400 mm。官方测试仅评价所选检查点，结果单独列于下表。',True)
    comparison=[['模型','训练张数','最佳轮','测试准确率','正确效率','总捕获率'],
                ['本轮等距',str(train_n),str(base['best_epoch']),pct(base['test']['accuracy']),pct(base['test']['correct_efficiency']),pct(base['test']['detector_capture'])],
                ['本轮非等距',str(train_n),str(chosen['best_epoch']),pct(chosen['test']['accuracy']),pct(chosen['test']['correct_efficiency']),pct(chosen['test']['detector_capture'])]]
    if history:
        hm=history.get('reevaluated_test',history['test'])
        comparison.append(['原最佳（复测）' if 'reevaluated_test' in history else '原最佳（历史）',str(history['config']['train_count']),str(history['best_epoch']),pct(hm['accuracy']),pct(hm['correct_efficiency']),pct(hm['detector_capture'])])
    table(comparison,[102,51,48,104,101,usable-406])
    paired=final.get('paired_comparison',{})
    text(f'仅对本轮两组作配对比较：仅等间距正确 {paired.get("baseline_only_correct","未记录")} 张，仅非等间距正确 {paired.get("nonuniform_only_correct","未记录")} 张。McNemar 未作连续性校正的渐近 p = {paired.get("mcnemar_asymptotic_p",float("nan")):.4g}。该值不覆盖换种子重训的不确定性；原最佳参考模型不加入这项比较。',True)
    page(5,'解释边界与复现')
    heading('可以支持的判断')
    text('本次实验在固定总长、相同层数和相位参数化、相同训练预算下，比较所列九种预设距离。最终测试差异说明所选配置在当前受控设置中的表现；不证明已找到连续距离空间的全局最优，也不证明非等间距在其他训练样本量、种子或任务上普遍更好。')
    heading('主要局限')
    text(f'仅使用一个随机种子和 {train_n} 张训练图像，训练预算较小。验证集同时用于距离选择和检查点选择，验证结果存在选择偏好；官方测试用于最终比较。历史 3000 张训练数据、15 轮的结果与本次样本预算不同，差异不能直接归因于层间距。')
    text('传播采用角谱法（ASM），在有限 FFT 窗口上数值计算。<br/>零填充扩大窗口，但仍可能存在周期边界、频谱采样和离散化误差。<br/>各相位层使用 10 mm 孔径截断场。模型只包含理想相位调制。<br/>尚未计入加工误差、材料色散、吸收、对准偏差和探测器噪声。')
    text('0 至 π 的相位约束与原模型一致；没有把距离效果与放宽相位范围混合比较。功率比例采用积分强度口径，不能直接解释为实际器件的绝对能量转换效率。')
    heading('复现命令')
    text('先进入项目下的 1550/test 目录，使用配置好的 d2nn 环境执行。下列 all 命令的默认参数即本次完整实验设置；详细分步命令见 README.md。',True)
    text('完整流程：python -B d2nn_spacing.py all<br/>重建报告：python -B d2nn_spacing.py report<br/>复测原最佳模型：python -B d2nn_spacing.py reference',True)
    zargs=' '.join(f'--z{i+1} {v:g}' for i,v in enumerate(chosen['z_mm']))
    text(f'所选非等间距的单组训练命令（单位 mm）：<br/>python -B d2nn_spacing.py train {zargs} --epochs 15',True)
    text('结果索引：search_results.json / .csv；refine_results.json；final_results.json / .csv；runs/各候选/history.json、config.json 和检查点；图表与本 PDF 均保存在 1550/test/results 内。',True)
    c.save()
    print(f'Report saved: {path}',flush=True)
# REPORT_SECTION_END

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['train','search','refine','evaluate','reference','plot','report','all']);p.add_argument('--root',type=Path,default=ROOT/'results');p.add_argument('--z',type=float,nargs=4,default=CANDIDATES[0]);p.add_argument('--z1',type=float);p.add_argument('--z2',type=float);p.add_argument('--z3',type=float);p.add_argument('--z4',type=float)
    p.add_argument('--epochs',type=int,default=15);p.add_argument('--search-epochs',type=int,default=5);p.add_argument('--train-count',type=int,default=1000);p.add_argument('--val-count',type=int,default=1000);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--lr',type=float,default=.005);p.add_argument('--seed',type=int,default=42);p.add_argument('--oversample',type=int,default=4);p.add_argument('--pad-factor',type=int,default=2);p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--resume',action='store_true',help='All runs resume automatically; retained as explicit acknowledgement')
    a=p.parse_args();a.root=a.root.resolve()
    if not a.root.is_relative_to(ROOT):p.error('All outputs must stay under test')
    if min(a.epochs,a.search_epochs,a.train_count,a.val_count,a.batch_size,a.oversample)<1 or a.pad_factor<2 or not math.isfinite(a.lr) or a.lr<=0:p.error('Invalid experiment sizes')
    a.z=[getattr(a,f'z{i+1}') if getattr(a,f'z{i+1}') is not None else a.z[i] for i in range(4)]
    a.root.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4);write_manifest(a)
    if a.command in ['train','search','refine','all']:
        ds,split=prepare_data(a)
        if a.command=='train':train_candidate(a,a.z,a.epochs,ds,split)
        if a.command in ['search','all']:search(a,ds,split)
        if a.command in ['refine','all']:refine(a,ds,split)
    if a.command in ['evaluate','all']:evaluate(a)
    if a.command=='reference' or (a.command=='all' and (ROOT/'reference'/'best_model.pt').exists()):
        evaluate_reference(a)
        if (a.root/'final_results.json').exists():plot_results(a.root)
    if a.command=='plot':plot_results(a.root)
    if a.command in ['report','all']:build_report(a.root)

if __name__=='__main__':main()
