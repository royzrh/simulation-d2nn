"""Read-only source-label coverage check; labels are never imaging targets."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq

root=Path(__file__).resolve().parents[2]
parquet=root/'misc'/'downloads'/'validation-00000-of-00014.parquet'
labels=pq.read_table(parquet,columns=['label']).column('label').to_pylist()
manifest_path=root/'data'/'manifest.json'
manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
canonical={key:value for key,value in manifest.items() if key!='manifest_sha256'}
assert hashlib.sha256(json.dumps(canonical,sort_keys=True).encode()).hexdigest()==manifest['manifest_sha256']
counts={split:Counter() for split in ['train','validation','test']}
selected=[]
for image in manifest['images']:
    index=int(Path(image['path']).stem.rsplit('_',1)[1])
    label=labels[index];counts[image['split']][label]+=1;selected.append(label)
selected_counts=Counter(selected)
result={'source_shard_rows':len(labels),'source_shard_class_count':len(set(labels)),
        'selected_image_count':len(selected),'selected_class_count':len(selected_counts),
        'selected_max_images_per_class':max(selected_counts.values()),
        'split_class_counts':{split:len(c) for split,c in counts.items()},
        'manifest_sha256':manifest['manifest_sha256'],
        'manifest_file_bytes_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'labels_used_for_training':False,'source_images_or_split_changed':False,
        'sampling_limitation':'Random subset of a disclosed shard; original author index list unavailable'}
out=root/'results'/'checks'/'sampling_class_coverage.json'
out.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
