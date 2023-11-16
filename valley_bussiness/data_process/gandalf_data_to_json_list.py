import json 
import os
from tqdm import tqdm
data_path = "/mnt/bn/luoruipu-disk/meta_data/sft_data/mfe_gandalf_test"
data_path_new = "/mnt/bn/luoruipu-disk/meta_data/sft_data/mfe_gandalf_test_str"

list_video_data_dict = []
video_data_path_list = os.listdir(data_path)
for file_name in tqdm(video_data_path_list):
    data_p = os.path.join(data_path, file_name)
    data = json.load(open(data_p, "r"))
    data_new = []
    for d in tqdm(data):
        if len(d['gandalf_vector']) != 1992:
            continue
        d['gandalf_vector'] = json.dumps(d['gandalf_vector'])
        data_new.append(d)
    json.dump(data_new, open(os.path.join(data_path_new,file_name), "w"), ensure_ascii=False)