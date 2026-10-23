"""Download an ImageNet-1K original-resolution public mirror shard, with provenance.

The paper does not publish its 2000 indices. We sample from a disclosed available
ImageNet pool, never substitute MNIST or a synthetic collection for ImageNet.
"""

# Allow this tool to run directly while keeping project imports rooted correctly.
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _BootstrapPath
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import hashlib
import json
from pathlib import Path
import time
import urllib.request
from train import HERE,save_json,prepare

REPO='benjamin-paine/imagenet-1k'
REV='ac6a45918fb830d6d1555b01090d7b991e8f8a4a'
FILE='data/validation-00000-of-00014.parquet'


def main():
    cache=HERE/'misc'/'downloads';cache.mkdir(parents=True,exist_ok=True)
    dest=cache/Path(FILE).name
    if not dest.exists():
        tmp=dest.with_suffix('.part')
        url=f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/{FILE}'
        print('Downloading ImageNet original-resolution shard',url,flush=True)
        with urllib.request.urlopen(url,timeout=90) as r,tmp.open('wb') as f:
            total=int(r.headers.get('Content-Length',0));done=0;last=0
            while True:
                chunk=r.read(1024*1024)
                if not chunk:break
                f.write(chunk);done+=len(chunk)
                if time.monotonic()-last>10:
                    print('downloaded_MB',round(done/1e6,1),'total_MB',round(total/1e6,1),flush=True);last=time.monotonic()
        if tmp.stat().st_size!=470121156:raise ValueError('Unexpected shard length')
        tmp.replace(dest)
    import pyarrow.parquet as pq
    table=pq.read_table(dest)
    root=HERE.parent/'data'/'ImageNet'/'imagenet1k_validation_shard0'
    root.mkdir(parents=True,exist_ok=True)
    for i,row in enumerate(table.to_pylist()):
        im=row['image'];raw=im['bytes'];p=root/f'imagenet_val_shard0_{i:05}.jpg'
        if not p.exists():p.write_bytes(raw)
    source={'repository':REPO,'revision':REV,'file':FILE,'shard_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
            'pool_count':len(table),'selection':'2000 random distinct files from this disclosed ImageNet validation shard',
            'note':'ImageNet validation split is used as unlabeled image pool; imaging split is separately disjoint 1500/200/300. Original paper index list unavailable.'}
    save_json(HERE/'data'/'source_provenance.json',source)
    manifest=prepare(root,json.dumps(source,sort_keys=True))
    print('PREPARED',manifest['manifest_sha256'],flush=True)


if __name__=='__main__':main()
