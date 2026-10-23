"""Plot raw intensity line profiles from saved two-axis resolution evaluation."""

# Allow this tool to run directly while keeping project imports rooted correctly.
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _BootstrapPath
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import argparse
from pathlib import Path
import numpy as np
import torch
from torch.nn.functional import grid_sample
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_profiles(path):
    path=Path(path)
    with np.load(path) as data:
        arrays={name:torch.from_numpy(data[name].copy()).float().reshape(1,1,300,300)
                for name in ['input','ideal','free']}
    arrays['input']=arrays['input'].square()
    fig,axes=plt.subplots(6,2,figsize=(10,14),sharey=True)
    values={}
    for row,w in enumerate([4.2,3.6,3.,2.4,1.8,1.2]):
        cx=-23 if row%2==0 else 23;cy=-25+(row//2)*25
        cross=np.linspace(-3*w,3*w,241,dtype=np.float32)
        for col,orientation in enumerate(['vertical','horizontal']):
            pcx=cx-8.5 if col==0 else cx+10.5
            along=np.linspace(-6 if col==0 else -4,6 if col==0 else 4,41)
            coords=[[(pcx+t,cy+d) if col==0 else (pcx+d,cy+t) for t in cross] for d in along]
            loc=torch.tensor(coords,dtype=torch.float32)[None]/45
            ax=axes[row,col]
            for name,label in [('input','Target'),('ideal','D2NN'),('free','Free space')]:
                profile=grid_sample(arrays[name],loc,align_corners=False)[0,0].mean(0).numpy()
                values[f'{w:g}mm_{orientation}_{name}']=profile
                ax.plot(cross,profile,label=label,linewidth=1.3)
            for t in [-2*w,-w,0,w,2*w]:ax.axvline(t,color='gray',alpha=.18,linewidth=.7)
            ax.set_title(f'{w:g} mm, {orientation} bars')
            ax.set_xlabel('Distance across bars (mm)')
            if col==0:ax.set_ylabel('Raw intensity')
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Length-averaged profiles; fixed raw intensity scale; no fitted gain',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.98])
    fig.savefig(path.with_name('resolution_profiles_both_axes.png'),dpi=150)
    plt.close(fig)
    np.savez_compressed(path.with_name('resolution_profiles_both_axes.npz'),**values)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('saved_chart',type=Path)
    plot_profiles(parser.parse_args().saved_chart)
