import json as js 
from shutil import copyfile
import random
data = random.sample(js.load(open('valley/inference/sample_input/llava_bench_chat.json','r')),16)
save_path = 'valley/inference/sample_input/sample_image'
for d in data:
    file_name =d['image'].split('/')[-1]
    copyfile(d['image'], save_path+'/'+ file_name)
    d['image'] = save_path+'/'+ file_name

js.dump(data,open('valley/inference/sample_input/sample_input_image.json','w'),indent=2)