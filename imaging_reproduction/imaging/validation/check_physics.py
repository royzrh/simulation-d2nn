"""CPU tests against analytic propagation and finite differences, no training data."""

# Allow this tool to run directly while keeping project imports rooted correctly.
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _BootstrapPath
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

from dataclasses import asdict
import math
import torch
from train import Geometry,ImagingD2NN,ssim
from train import HERE,save_json


def main():
    torch.set_num_threads(2);torch.manual_seed(42)
    results={}
    g=Geometry(neurons=24,layers=2,oversample=1,pad_factor=2,input_mm=0,interlayer_mm=0,output_mm=0)
    m=ImagingD2NN(g)
    with torch.no_grad():m.raw_phases.zero_()
    a=torch.rand(2,24,24)
    err=(m(a)-a.square()).abs().max().item();assert err<2e-6,err
    results['zero_distance_constant_phase_intensity_max_error']=err
    identity=ssim(a,a);assert torch.allclose(identity,torch.ones_like(identity),atol=2e-6)
    results['ssim_identity']=identity.tolist()
    g=Geometry(neurons=24,layers=2,oversample=2)
    m=ImagingD2NN(g);u=m.embed(a)
    # FFT plane wave is an exact eigenfunction of ASM.
    idx=torch.arange(m.fft_n);wave=torch.exp(2j*math.pi*idx/m.fft_n)[None,:].expand(m.fft_n,-1)[None]
    p=m.propagate(wave,index=0)
    expected=wave*getattr(m,'H_0')[0,1]
    err=(p-expected).abs().max().item();assert err<5e-6,err
    results['plane_wave_eigenmode_max_error']=err
    out,trace=m.field(a,trace=True)
    power0=u.abs().square().sum((-2,-1));power1=out.abs().square().sum((-2,-1))
    assert (power1<=power0*(1+1e-5)).all()
    results['passive_output_input_power_ratio']=(power1/power0).tolist()
    loss=m(a).square().mean();loss.backward();analytical=m.raw_phases.grad[0,12,12].item()
    eps=.01;v=m.raw_phases[0,12,12].item()
    with torch.no_grad():
        m.raw_phases[0,12,12]=v+eps;plus=m(a).square().mean().item()
        m.raw_phases[0,12,12]=v-eps;minus=m(a).square().mean().item()
        m.raw_phases[0,12,12]=v
    numerical=(plus-minus)/(2*eps)
    relative=abs(analytical-numerical)/max(abs(analytical),abs(numerical),1e-8)
    assert relative<.06,(analytical,numerical,relative)
    results['phase_gradient']={'autograd':analytical,'finite_difference':numerical,'relative_error':relative}
    # Changing numerical sampling preserves trainable physical parameters.
    m3=ImagingD2NN(Geometry(neurons=24,layers=2,oversample=3))
    assert m3.raw_phases.shape==m.raw_phases.shape
    results['sampling_preserves_mask_parameter_count']=True
    # Scaling ALL physical lengths together with wavelength must preserve ASM
    # intensity for identical phase masks. This is a numerical identity check,
    # not an assertion that a scaled device is manufacturable at 1550 nm.
    wavelength_m=1550e-9
    scale=wavelength_m/(299792458.0/g.frequency_hz)
    scaled=asdict(g)
    scaled.update(frequency_hz=299792458.0/wavelength_m,pitch_m=g.pitch_m*scale,
                  input_mm=g.input_mm*scale,interlayer_mm=g.interlayer_mm*scale,
                  output_mm=g.output_mm*scale)
    infrared=ImagingD2NN(Geometry(**scaled))
    with torch.no_grad():
        infrared.raw_phases.copy_(m.raw_phases)
        error=(infrared(a)-m(a)).abs().max().item()
    assert error<5e-6,error
    results['wavelength_1550nm_uniform_length_scaling_max_intensity_error']=error
    try:infrared.mask(0,absorption=True)
    except ValueError:results['non_thz_material_parameters_rejected']=True
    else:raise AssertionError('Unverified THz material parameters accepted at 1550 nm')
    # One optimizer update still sees four images, including with accumulation
    # and activation recomputation. Include a final incomplete loader batch.
    import copy
    from train import epoch
    full=copy.deepcopy(m);chunked=copy.deepcopy(m);chunked.activation_checkpointing=True
    examples=torch.rand(5,24,24)
    loader=torch.utils.data.DataLoader(examples,batch_size=4)
    full_opt=torch.optim.Adam(full.parameters(),lr=.005)
    chunk_opt=torch.optim.Adam(chunked.parameters(),lr=.005)
    full_metrics=epoch(full,loader,'cpu','amplitude_squared',full_opt)
    chunk_metrics=epoch(chunked,loader,'cpu','amplitude_squared',chunk_opt,micro_batch_size=1)
    deviation=(full.raw_phases-chunked.raw_phases).abs().max().item()
    assert deviation<2e-5,deviation
    assert abs(full_metrics['mse']-chunk_metrics['mse'])<1e-6
    results['effective_batch4_accumulation_and_checkpointing_max_parameter_difference']=deviation
    save_json(HERE/'results'/'checks'/'physics.json',results)
    print('ALL PHYSICS CHECKS PASSED',results)


if __name__=='__main__':main()
