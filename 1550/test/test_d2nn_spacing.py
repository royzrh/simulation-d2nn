"""Numerical checks against the existing physics implementation; no import caches."""
import importlib.util, json, sys, time
sys.dont_write_bytecode=True
from pathlib import Path
from types import SimpleNamespace
import torch
import d2nn_spacing as new

def main():
    torch.set_num_threads(4)
    references=[new.ROOT.parent/'train_d2nn_1550nm.py',new.ROOT.parent/'1550'/'train_d2nn_1550nm.py']
    reference=next(p for p in references if p.is_file())
    spec=importlib.util.spec_from_file_location('reference',reference)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    old.WAVELENGTH=1550e-9;old.PITCH=50e-6;old.GAP=.1;old.DETECTOR_SIDE=.6e-3
    old.DETECTOR_CENTERS=[(x*1e-3*.125,y*1e-3*.125) for y,xs in [(12,[-12,0,12]),(0,[-18,-6,6,18]),(-12,[-12,0,12])] for x in xs]
    torch.manual_seed(42);a=old.OpticalD2NN(1,2,30)
    torch.manual_seed(42);b=new.OpticalD2NN([100]*4,1,2)
    image=torch.rand(2,1,28,28);label=torch.tensor([1,7]);ea=a.encode(image,.5,1);eb=b.encode(image)
    torch.testing.assert_close(ea,eb,rtol=0,atol=0)
    ia,pa=a(ea);ib,pb=b(eb)
    torch.testing.assert_close(ia,ib,rtol=0,atol=0);torch.testing.assert_close(pa,pb,rtol=2e-6,atol=1e-8)
    la,_=old.compute_loss('baseline',pa,ia,label,ea.abs().square().sum((-2,-1)),a.s)
    lb,_=new.compute_loss(pb,ib,label,eb.abs().square().sum((-2,-1)),b.s)
    la.backward();lb.backward();torch.testing.assert_close(la,lb,rtol=2e-6,atol=1e-7)
    torch.testing.assert_close(a.raw_phases.grad,b.raw_phases.grad,rtol=2e-4,atol=2e-8)
    for bad in [[99]*4,[0,100,100,200],[float('nan'),100,100,100],[float('inf'),100,100,100]]:
        try:new.OpticalD2NN(bad,1,2)
        except ValueError:pass
        else:raise AssertionError('Invalid distances accepted')
    result={'reference_output_max_abs':float((ia-ib).abs().max().detach()),'reference_power_max_abs':float((pa-pb).abs().max().detach()),'reference_loss_abs':float((la-lb).abs().detach()),'reference_gradient_max_abs':float((a.raw_phases.grad-b.raw_phases.grad).abs().max()),'energy_input':ea.abs().square().sum((-2,-1)).tolist(),'status':'passed'}
    del a,b,ea,eb,ia,ib,pa,pb,la,lb
    torch.manual_seed(42);m=new.OpticalD2NN([100]*4).cuda();x=m.encode(image.cuda());t=time.perf_counter();i,p=m(x);loss,_=new.compute_loss(p,i,label.cuda(),x.abs().square().sum((-2,-1)),m.s);loss.backward();torch.cuda.synchronize()
    assert torch.isfinite(m.raw_phases.grad).all()
    result.update(gpu_smoke_seconds=time.perf_counter()-t,gpu_smoke_peak_gb=torch.cuda.max_memory_allocated()/1024**3)
    del m,x,i,p,loss
    torch.cuda.empty_cache()
    # CPU-only physics and resume checks use synthetic data, separate from the experiment.
    torch.manual_seed(77)
    m=new.OpticalD2NN([50,75,125,150],1,2)
    field=torch.randn(1,m.fft_n,m.fft_n,dtype=torch.complex64)
    energy=field.abs().square().sum()
    propagated=m.propagate(field,m.H2)
    relative=float((propagated.abs().square().sum()/energy-1).abs())
    assert relative<2e-6,'Free propagation must conserve discrete energy'
    assert float((propagated*m.aperture).abs().square().sum())<=float(energy)*(1+2e-6)
    assert not torch.equal(m.H1,m.H2),'Independent distances must produce different kernels'
    assert all(sum(z)==400 and min(z)>0 for z in new.CANDIDATES)
    del m,field,propagated
    images=torch.rand(8,1,28,28);labels=torch.arange(8)%10
    dataset=torch.utils.data.TensorDataset(images,labels)
    split={'train':list(range(4)),'validation':list(range(4,8)),'sha256':'synthetic-resume-check'}
    parent=new.ROOT/'tmp'/'resume_verification'/f'check_{time.time_ns()}'
    args=SimpleNamespace(root=parent/'uninterrupted',seed=42,train_count=4,val_count=4,batch_size=2,
                         lr=.005,oversample=1,pad_factor=2,device='cpu')
    new.train_candidate(args,[50,75,125,150],2,dataset,split)
    args.root=parent/'resumed'
    new.train_candidate(args,[50,75,125,150],1,dataset,split)
    new.train_candidate(args,[50,75,125,150],2,dataset,split)
    paths=[parent/name/'runs'/'z_50_75_125_150'/'last_checkpoint.pt' for name in ['uninterrupted','resumed']]
    checkpoints=[torch.load(path,map_location='cpu',weights_only=False) for path in paths]
    for key,value in checkpoints[0]['model_state_dict'].items():
        torch.testing.assert_close(value,checkpoints[1]['model_state_dict'][key],rtol=0,atol=0)
    for left,right in zip(checkpoints[0]['history'],checkpoints[1]['history']):
        for key in left:
            if key not in ['seconds','peak_gpu_gb']:assert left[key]==right[key],key
    result.update(free_propagation_relative_energy_error=relative,resume_matches_uninterrupted_exactly=True,
                  reference_path=str(reference))
    new.write_json(new.ROOT/'verification.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
