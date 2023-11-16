import argparse
import torch
from torch.utils.data import DataLoader
from transformers import LlamaTokenizer
from valley.model.language_model.valley_llama import ValleyLlamaForCausalLM
import torch
import os
from valley.utils import disable_torch_init
from transformers import  CLIPImageProcessor
import os
import random
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
import traceback
from torch.utils.data.distributed import DistributedSampler
from valley.util.config import DEFAULT_GANDALF_TOKEN
from valley.util.data_util import KeywordsStoppingCriteria
from peft import PeftConfig
from transformers import set_seed
from valley.data.dataset import LazySupervisedDataset, DataCollatorForSupervisedDataset
from valley.util.data_util import smart_tokenizer_and_embedding_resize
from valley import conversation as conversation_lib
os.environ['NCCL_DEBUG']=''
def setup(args,rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = args.DDP_port
    dist.init_process_group("nccl", rank=rank, world_size=world_size, )

def standardization(data):
        mu = torch.mean(data)
        sigma = torch.std(data)
        return (data - mu) / sigma

def inference(rank, world_size, args):
    set_seed(42)

    this_rank_gpu_index = rank

    if args.DDP:
        torch.cuda.set_device(this_rank_gpu_index)
        setup(args, rank, world_size)
        
    disable_torch_init()

    device = torch.device('cuda:'+str(this_rank_gpu_index)
                          if torch.cuda.is_available() else 'cpu')
    model_name = os.path.expanduser(args.model_name)

    # load model
    if 'lora' in model_name:
        config = PeftConfig.from_pretrained(model_name)
        print('load old model weight and lora weight')
        model_old = ValleyLlamaForCausalLM.from_pretrained(model_name)
        print('load no lora model')
        if os.path.exists(os.path.join(model_name,'non_lora_trainables.bin')):
            non_lora_state_dict = torch.load(os.path.join(model_name,'non_lora_trainables.bin'))
            new_state_dict = dict()
            for key in non_lora_state_dict.keys():
                key_new = '.'.join(key.split('.')[2:]) # base_model.model.model.xxxx
                new_state_dict[key_new] = non_lora_state_dict[key]
            model_old_state = model_old.state_dict()
            model_old_state.update(new_state_dict)
            model_old.load_state_dict(model_old_state)
        model = model_old
        tokenizer = LlamaTokenizer.from_pretrained(os.path.dirname(model_name), use_fast = False)
        if hasattr(model.model, 'gandalf_projector'):
            model_old.config.gandalf_token_index = tokenizer.convert_tokens_to_ids(DEFAULT_GANDALF_TOKEN)
        tokenizer.padding_side = 'left'
        print("load end")
    else:
        print('load model')
        model = ValleyLlamaForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16)
        tokenizer = LlamaTokenizer.from_pretrained(args.model_name, use_fast = False)
        tokenizer.padding_side = 'left'
        print('load end')
    
    image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower)
    model.eval()
    model = model.to(device)
    args.image_processor = image_processor
    args.is_multimodal = True
    args.mm_use_im_start_end = True
    args.only_mask_system = False
    
    if args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if args.prompt_version is not None:
        conversation_lib.default_conversation = conversation_lib.conv_templates[args.prompt_version]
    dataset = LazySupervisedDataset(args.data_path, tokenizer=tokenizer, data_args = args, inference= True)
    
    if args.DDP:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=DataCollatorForSupervisedDataset, pin_memory=True, sampler=sampler,)
        rf = open(args.out_path+".worker_"+str(rank), 'w')
    else:
        dataloader = DataLoader(dataset, num_workers=1, batch_size=args.batch_size, collate_fn=DataCollatorForSupervisedDataset, pin_memory=True)
        rf = open(args.out_path, 'w')

    prog_bar = tqdm(dataloader, total=len(dataloader),desc='worker_'+str(rank)) if rank == 0 else dataloader
    

    for test_batch in prog_bar:
        try:
            test_batch = test_batch.tokenizer[0]
            gt_label = [test_batch.pop('gt_label')]
            for key in test_batch:
                test_batch[key] = test_batch[key].to(device)
            stop_str = conversation_lib.default_conversation.sep if conversation_lib.default_conversation.sep_style != conversation_lib.SeparatorStyle.TWO else conversation_lib.default_conversation.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, test_batch['input_ids'].unsqueeze(0))
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids = test_batch['input_ids'].unsqueeze(0),
                    images=test_batch['image'].half().unsqueeze(0),
                    do_sample=args.do_sample,
                    temperature=args.temperature,
                    stopping_criteria=[stopping_criteria],
                    max_new_tokens = 1024,
                    return_dict_in_generate= True if args.ouput_logits else False, output_scores= True if args.ouput_logits else False
                )
            if not args.ouput_logits: 
                input_token_len = test_batch['input_ids'].unsqueeze(0).shape[1]
                n_diff_input_output = (test_batch['input_ids'].unsqueeze(0) != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
                outputs = tokenizer.batch_decode( output_ids[:, input_token_len:], skip_special_tokens=True)
                response = outputs
                print(response)
            if args.ouput_logits:
                outputs = tokenizer.batch_decode(output_ids.sequences[:, -3:], skip_special_tokens=True)
                scores = standardization(output_ids.scores[ 3])
                standardization_score = scores[:,[3869,1939]]
                standardization_logits = torch.softmax(standardization_score, dim=1).cpu().numpy().tolist()
                response = [format(yes_logits, '.8f') for yes_logits, no_logits in standardization_logits]

            for i in range(len(response)):
                rf.write('\t'.join(['None', str(gt_label[i]), response[i].replace('\n','')]) + '\n')
        except Exception as e:
            traceback.print_exc()
    rf.close()

def gather_result(args,world_size):
    num_worker = world_size
    with open(args.out_path, 'w') as f:
        for i in range(num_worker):
            with open(args.out_path+".worker_"+str(i), 'r') as tf:
                tmp_result = tf.readlines()
            f.writelines(tmp_result)
            os.remove(args.out_path+".worker_"+str(i))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default = "/mnt/bn/luoruipu-disk/checkpoints/stable-valley-13b-v1")
    parser.add_argument("--video_data_path", type=str, required = False, default = '/mnt/bn/luoruipu-disk/code_base/valley/valley/inference/sample_input/sample_input_video.json' )
    parser.add_argument("--data_path", type=str, required = False, default = '' )
    parser.add_argument("--video_folder", type=str, required = False, default = '')
    parser.add_argument("--image_folder", type=str, required = False, default = '')
    parser.add_argument("--out_path", type=str, required = False, default = 'inference_output/test2.txt' )
    parser.add_argument("--version", type=str, default="v0")
    parser.add_argument("--prompt_version", type=str, default="valley_v0")
    parser.add_argument("--image_aspect_ratio", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=False, default=1)
    parser.add_argument("--ouput_logits", action="store_true", default=False)
    parser.add_argument("--temperature", type = float, default=1)
    parser.add_argument("--do-sample", action="store_true", default=False)
    parser.add_argument("--DDP", action="store_true")
    parser.add_argument("--DDP_port", default = '12345')
    parser.add_argument("--world_size", type=int, default = 8)
    args = parser.parse_args()

    if args.DDP:
        mp.spawn( inference, args=(args.world_size, args), nprocs=args.world_size)
        gather_result(args, args.world_size)
    else: 
        inference(0, args.world_size, args)