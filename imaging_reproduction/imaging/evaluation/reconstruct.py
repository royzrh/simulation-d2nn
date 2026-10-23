"""Actual oriented-point Screened Poisson reconstruction, following S12.

Original sampling and MeshLab settings were not published. Settings, mesh files,
height residuals and ray coverage are saved instead of claiming identical errors.
"""

# Allow this tool to run directly while keeping project imports rooted correctly.
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _BootstrapPath
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
import math
import hashlib
from pathlib import Path
import numpy as np
import torch
from train import save_json


def reconstruct_layer(phase,pitch_m,wavelength_m,out,depth=9,stride=1):
    import pymeshlab
    import trimesh
    original_h=phase*wavelength_m/(2*math.pi*(1.7227-1))+.0005
    original_n=original_h.shape[0]
    original_axis=(np.arange(original_n)-(original_n-1)/2)*pitch_m
    ray_y,ray_x=np.meshgrid(original_axis,original_axis,indexing='ij')
    # Phase pixels occupy finite squares. The physical side wall is half a pixel
    # beyond outermost centers, not at those centers (which loses boundary rays).
    axis=np.r_[original_axis[0]-pitch_m/2,original_axis,original_axis[-1]+pitch_m/2]
    h=np.pad(original_h,1,mode='edge');n=h.shape[0]
    y,x=np.meshgrid(axis,axis,indexing='ij')
    gy,gx=np.gradient(h,axis,axis)
    top=np.stack([x,y,h],axis=-1)[::stride,::stride].reshape(-1,3)
    normals=np.stack([-gx,-gy,np.ones_like(h)],axis=-1)[::stride,::stride].reshape(-1,3)
    normals/=np.linalg.norm(normals,axis=1,keepdims=True)
    bottom=top.copy();bottom[:,2]=0
    pts=[top,bottom];ns=[normals,np.tile([0.,0.,-1.],(len(bottom),1))]
    # Closing side walls are part of S12's enclosed point cloud, not optional.
    for along,side,normal in [('x',0,[-1.,0.,0.]),('x',n-1,[1.,0.,0.]),('y',0,[0.,-1.,0.]),('y',n-1,[0.,1.,0.])]:
        ids=np.arange(0,n,stride)
        for f in np.linspace(.05,.95,8):
            if along=='x':p=np.stack([x[ids,side],y[ids,side],f*h[ids,side]],axis=1)
            else:p=np.stack([x[side,ids],y[side,ids],f*h[side,ids]],axis=1)
            pts.append(p);ns.append(np.tile(normal,(len(p),1)))
    vertices=np.concatenate(pts);normals=np.concatenate(ns)
    ms=pymeshlab.MeshSet();ms.add_mesh(pymeshlab.Mesh(vertex_matrix=vertices,v_normals_matrix=normals),'oriented_closed_surface')
    ms.save_current_mesh(str(out/'oriented_points.ply'),save_vertex_normal=True)
    ms.generate_surface_reconstruction_screened_poisson(depth=depth,fulldepth=5,samplespernode=1.5,pointweight=4.,threads=2)
    ms.save_current_mesh(str(out/'reconstructed.ply'))
    mesh=ms.current_mesh();surface=trimesh.Trimesh(mesh.vertex_matrix(),mesh.face_matrix(),process=False)
    origins=np.stack([ray_x.ravel(),ray_y.ravel(),np.full(original_n*original_n,float(h.max()+.005))],axis=1)
    directions=np.tile([0.,0.,-1.],(len(origins),1))
    recovered=np.full(original_n*original_n,np.nan)
    # Limit ray batches for memory stability.
    for start in range(0,len(origins),3000):
        locations,ids,_=surface.ray.intersects_location(origins[start:start+3000],directions[start:start+3000],multiple_hits=False)
        recovered[start+ids]=locations[:,2]
    missing=~np.isfinite(recovered)
    # Missing rays are reported and disqualify strict error evaluation; never filled silently.
    recovered=recovered.reshape(original_n,original_n)
    delta=recovered-original_h
    np.savez_compressed(out/'height_residual.npz',original_height_m=original_h,reconstructed_height_m=recovered,height_error_m=delta)
    metadata={'depth':depth,'point_stride':stride,'point_count':len(vertices),'vertex_count':surface.vertices.shape[0],
              'face_count':surface.faces.shape[0],'missing_ray_count':int(missing.sum()),
              'height_rmse_um':float(np.sqrt(np.nanmean(delta**2))*1e6),
              'height_max_error_um':float(np.nanmax(np.abs(delta))*1e6),
              'algorithm':'MeshLab Screened Poisson, oriented closed top/bottom/side point set',
              'matches_original_unpublished_reconstruction_settings':False}
    save_json(out/'reconstruction.json',metadata)
    return delta,metadata


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True,type=Path);p.add_argument('--depth',type=int,default=9)
    p.add_argument('--stride',type=int,default=1)
    a=p.parse_args();c=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    g=c['config']['geometry'];phases=(2*math.pi*torch.sigmoid(c['model']['raw_phases'])).numpy()
    out=a.checkpoint.parent/'surface_reconstruction';out.mkdir(exist_ok=True)
    residuals=[];stats=[]
    for i,phase in enumerate(phases):
        layer=out/f'layer_{i+1}';layer.mkdir(exist_ok=True)
        if (layer/'height_residual.npz').exists():
            with np.load(layer/'height_residual.npz') as cached:
                error=cached['height_error_m'].copy()
                expected=phase*(299792458/g['frequency_hz'])/(2*math.pi*(1.7227-1))+.0005
                if not np.allclose(cached['original_height_m'],expected,rtol=1e-6,atol=1e-10):
                    raise ValueError('Cached surface belongs to different phases; use a separate experiment directory')
                if not np.allclose(cached['reconstructed_height_m']-cached['original_height_m'],error,equal_nan=True):
                    raise ValueError('Cached height residual is inconsistent')
            import json
            meta=json.loads((layer/'reconstruction.json').read_text())
            if meta['depth']!=a.depth or meta['point_stride']!=a.stride:
                raise ValueError('Cached reconstruction settings differ; preserve it and use a separate directory')
        else:error,meta=reconstruct_layer(phase,g['pitch_m'],299792458/g['frequency_hz'],layer,a.depth,a.stride)
        residuals.append(error);stats.append(meta)
        print('RECONSTRUCTED layer',i+1,meta,flush=True)
    np.save(out/'height_errors_m.npy',np.stack(residuals))
    save_json(out/'summary.json',{'layers':stats,'ready_for_optical_evaluation':all(v['missing_ray_count']==0 for v in stats),
        'checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        'checkpoint':str(a.checkpoint.resolve()),'geometry':g,
        'height_errors_sha256':hashlib.sha256((out/'height_errors_m.npy').read_bytes()).hexdigest()})


if __name__=='__main__':main()
