"""Separate 10,000-training-image sensitivity set (supplement p10).

The paper does not give that run's validation/test split. We retain the original
200/300 held-out images, retain its 1500 training images, and add 8500 unique
training images. This is an explicit reconstruction choice, not the author's
unpublished index list. Original images, manifest and cache remain read-only.
"""

# Allow this tool to run directly while keeping project imports rooted correctly.
if __package__ in (None, ""):
    import sys
    from pathlib import Path as _BootstrapPath
    sys.path.insert(0, str(_BootstrapPath(__file__).resolve().parents[2]))

import hashlib
import json
from pathlib import Path
import random
import time
import urllib.request
import numpy as np
from PIL import Image,ImageOps
from train import HERE,save_json
from imaging.datasets.acquire_imagenet import REPO,REV


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def download_shard(index):
    import pyarrow.parquet as pq
    name=f'validation-{index:05}-of-00014.parquet'
    dest=HERE/'misc'/'downloads'/name
    metadata_url=f'https://huggingface.co/api/datasets/{REPO}/tree/{REV}/data?limit=1000'
    with urllib.request.urlopen(metadata_url,timeout=30) as response:metadata=json.load(response)
    source=next(row for row in metadata if row.get('path')=='data/'+name)
    expected_size=source['size'];expected_sha=source['lfs']['oid']
    if not dest.exists():
        partial=dest.with_suffix('.part')
        if partial.exists() and partial.stat().st_size==expected_size:
            if digest(partial)!=expected_sha:raise ValueError('Preserve corrupt completed partial for investigation')
            partial.replace(dest)
        attempt=0;stalled_attempts=0;download_deadline=time.monotonic()+3*3600
        while not dest.exists():
            if time.monotonic()>download_deadline:raise TimeoutError('Shard download exceeded three hours; partial bytes preserved')
            attempt+=1
            if dest.exists():break
            offset=partial.stat().st_size if partial.exists() else 0
            try:
                host='huggingface.co' if attempt%2==1 else 'hf-mirror.com'
                url=f'https://{host}/datasets/{REPO}/resolve/{REV}/data/{name}?download=true&request={int(time.time())}'
                headers={'Range':f'bytes={offset}-'} if offset else {}
                with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=90) as response:
                    resume=offset>0 and response.status==206
                    if offset and not resume:raise ValueError('Server ignored resume range; existing partial preserved')
                    if resume and not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):
                        raise ValueError('Unexpected resumed response range')
                    start=offset if resume else 0
                    expected=response.headers.get('Content-Length')
                    expected=int(expected)+start if expected else None
                    done=start;last=0
                    with partial.open('ab' if resume else 'wb') as f:
                        while True:
                            chunk=response.read(1024*1024)
                            if not chunk:break
                            f.write(chunk);done+=len(chunk)
                            if time.monotonic()-last>10:
                                print(name,'downloaded_MB',round(done/1e6,1),flush=True);last=time.monotonic()
                    if expected is not None and done!=expected:raise ValueError('Incomplete download')
                pq.read_metadata(partial)  # Refuse HTML/error pages and truncated files.
                if partial.stat().st_size!=expected_size or digest(partial)!=expected_sha:
                    raise ValueError('Downloaded bytes differ from primary repository size/SHA256')
                partial.replace(dest);break
            except Exception as error:
                current_size=partial.stat().st_size if partial.exists() else 0
                stalled_attempts=0 if current_size>offset else stalled_attempts+1
                print('DOWNLOAD RETRY',name,attempt,'bytes preserved',current_size,
                      'consecutive attempts without progress',stalled_attempts,str(error),flush=True)
                if current_size>=expected_size or stalled_attempts>=6:raise
                time.sleep(min(30,2**stalled_attempts))
    if dest.stat().st_size!=expected_size or digest(dest)!=expected_sha:
        raise ValueError('Cached shard differs from primary repository metadata')
    meta=pq.read_metadata(dest)
    rawroot=HERE.parent/'data'/'ImageNet'/'imagenet1k_validation_expansion'/f'shard_{index:02}'
    rawroot.mkdir(parents=True,exist_ok=True)
    row_index=0
    for batch in pq.ParquetFile(dest).iter_batches(batch_size=64):
        for row in batch.to_pylist():
            i=row_index;row_index+=1
            image=row.get('image',row)
            raw=image['bytes'];target=rawroot/f'imagenet_val_shard{index}_{i:05}.jpg'
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest()!=hashlib.sha256(raw).hexdigest():
                    raise ValueError('Preserve differing existing source file: '+str(target))
            else:target.write_bytes(raw)
    print('SHARD READY',name,'rows',meta.num_rows,flush=True)
    return {'repository':REPO,'revision':REV,'file':'data/'+name,'rows':meta.num_rows,
            'sha256':digest(dest),'primary_repository_expected_sha256':expected_sha,
            'raw_root':str(rawroot),'transport_mirror_requires_identical_primary_hash':True}


def choose_extras(base,roots):
    candidates=sorted({p for root in roots for p in Path(root).glob('*.jpg')})
    random.Random(42).shuffle(candidates)
    used={r['sha256'] for r in base['images']};extras=[]
    for path in candidates:
        sha=digest(path)
        if sha in used:continue
        try:
            with Image.open(path) as image:image.verify()
        except Exception as error:
            print('SKIP undecodable image',path.name,str(error),flush=True);continue
        used.add(sha);extras.append({'path':str(path.resolve()),'sha256':sha,'split':'train'})
        if len(extras)==8500:break
    return extras


def main():
    base_dir=HERE/'data';out=HERE/'data_imagenet_train10000'
    if out.resolve()==base_dir.resolve():raise ValueError('Separate destination required')
    out.mkdir(exist_ok=True)
    ready=out/'data_ready.json'
    if ready.exists():
        saved=json.loads(ready.read_text(encoding='utf-8'))
        if digest(out/'amplitudes.npy')!=saved['cache_sha256']:raise ValueError('Existing large cache changed')
        print('VERIFIED existing separate large dataset',saved,flush=True);return
    base_file=base_dir/'manifest.json';base_bytes=digest(base_file);base_cache_sha=digest(base_dir/'amplitudes.npy')
    base=json.loads(base_file.read_text(encoding='utf-8'))
    rawroots=[base['source_root']];sources=[]
    extras=[]
    for index in range(1,14):
        sources.append(download_shard(index));rawroots.append(sources[-1]['raw_root'])
        if index<2:continue
        extras=choose_extras(base,rawroots)
        print('Distinct additional images available',len(extras),flush=True)
        if len(extras)==8500:break
    if len(extras)!=8500:raise ValueError('Insufficient additional unique images')
    manifest={'source':{'repository':REPO,'revision':REV,'additional_shards':sources,
                        'base_manifest_sha256':base['manifest_sha256']},
              'source_root':str(HERE.parent/'data'/'ImageNet'), 'seed':42,'size':300,
              'preprocessing':base['preprocessing'],'split_counts':{'train':10000,'validation':200,'test':300},
              'images':base['images']+extras,'original_authors_image_indices_available':False,
              'split_choice':'10,000 actual training images; original 200/300 held-out images retained. Paper larger-run split unpublished.'}
    manifest['manifest_sha256']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    existing=out/'manifest.json'
    if existing.exists() and json.loads(existing.read_text(encoding='utf-8'))!=manifest:
        raise ValueError('Preserve differing existing large-data manifest')
    partial=out/'amplitudes.npy.part'
    cache=np.lib.format.open_memmap(partial,mode='w+',dtype=np.float32,shape=(10500,300,300))
    original=np.load(base_dir/'amplitudes.npy',mmap_mode='r')
    for start in range(0,2000,100):cache[start:start+100]=original[start:start+100]
    for i,row in enumerate(extras,2000):
        with Image.open(row['path']) as image:
            cache[i]=np.asarray(ImageOps.exif_transpose(image).convert('L').resize((300,300),Image.Resampling.BILINEAR),dtype=np.float32)/255
        if i%500==0:print('PREPROCESSED',i,'/10500',flush=True)
    cache.flush()
    for start in range(0,2000,100):
        if not np.array_equal(cache[start:start+100],original[start:start+100]):raise ValueError('Original-image preprocessing differs')
    if len({r['sha256'] for r in manifest['images']})!=10500:raise ValueError('Image content overlaps')
    if not all(np.isfinite(cache[start:start+100]).all() for start in range(0,10500,100)):
        raise ValueError('Nonfinite preprocessed images')
    del cache,original
    if digest(base_file)!=base_bytes or digest(base_dir/'amplitudes.npy')!=base_cache_sha:
        raise ValueError('Original dataset changed during preparation')
    partial.replace(out/'amplitudes.npy');save_json(existing,manifest)
    result={'manifest_sha256':manifest['manifest_sha256'],'cache_sha256':digest(out/'amplitudes.npy'),
            'split_counts':manifest['split_counts'],'base_manifest_file_sha256':base_bytes,
            'base_cache_sha256':base_cache_sha,'original_dataset_unchanged':True,
            'all_original_2000_preprocessed_images_bitwise_identical':True,'unique_content_count':10500}
    save_json(ready,result)
    print('SEPARATE LARGE DATASET READY',result,flush=True)


if __name__=='__main__':main()
