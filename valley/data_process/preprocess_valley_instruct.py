import json as js
from tqdm import tqdm
file = js.load(open('/mnt/bn/luoruipu-disk/meta_data/finetune_data/Valley-Instruct/Valley_instruct_84k.json'))
for f in file:
    if 'source' not in f:
        f['source'] = 'webvid'

js.dump(file, open('/mnt/bn/luoruipu-disk/meta_data/finetune_data/Valley-Instruct/Valley_instruct_84k.json','w'))