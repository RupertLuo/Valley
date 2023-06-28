import torch
import logging
from dataclasses import dataclass, field
from typing import Optional
import torch.distributed as dist
from transformers.trainer_pt_utils import get_parameter_names
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from tqdm import tqdm
from accelerate import Accelerator
from torch.utils.data import  DataLoader
from valley.model.valley import ValleyLlamaForCausalLM
from transformers import set_seed
from util.data_util import smart_tokenizer_and_embedding_resize
import transformers
from transformers.trainer import Trainer
from valley.data.dataset import make_video_supervised_data_module
from transformers.optimization import get_scheduler
from deepspeed.accelerator import get_accelerator
from tqdm.auto import tqdm
import wandb
logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_FRAME_TOKEN = "<vi_frame>"
DEFAULT_VI_START_TOKEN = "<vi_start>"
DEFAULT_VI_END_TOKEN = "<vi_end>"



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_checkpoint: Optional[str] = field(default=None)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    tune_llm_layer: str=field(default='')


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    fashion_data_path: str = field(default = None,
                            metadata={"help": "Path to the new construct training data."})
    video_data_path:str = field(default = None,
                            metadata={"help": "Path to the video training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    sep_image_conv_front: bool = False
    image_token_len: int = 0
    image_folder: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    fashion_image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    num_image: int=field(default=4)
    multi_image: bool=field(default=True)
    multi_image_mode: str=field(default='concatenate')
    use_fashion: bool = field(default=False)
    fast_epoch: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    force_fsdp: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )

class SFTMetric:
    def __init__(self, device):
        self.n_step = 0
        self.right = torch.Tensor([0]).to(device=device)
        self.total = torch.Tensor([0]).to(device=device)
        self.total_loss = torch.Tensor([0]).to(device=device)
        self.world_size = dist.get_world_size()

    def __call__(self, logits, labels, loss):
        return self.update(logits, labels, loss)

    def update(self, logits, labels, loss):
        self.n_step += 1
        with torch.no_grad():
            shift_preds = logits[..., :-1, :].argmax(dim=-1)
            shift_labels = labels[..., 1:]
            self.right += (shift_preds == shift_labels).masked_fill(shift_labels.eq(-100), 0).sum().item()
            self.total += (shift_labels != -100).sum().item()
            self.total_loss += loss.item()

    def get_metric(self, reset=True):
        dist.all_reduce(self.right, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.total, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.total_loss, op=torch.distributed.ReduceOp.SUM)

        acc = (self.right / self.total).item()
        loss = self.total_loss.item() / (self.world_size * self.n_step)

        if reset:
            self.n_step = 0
            self.right.fill_(0)
            self.total.fill_(0)
            self.total_loss.fill_(0)
        return acc, loss
def train():
    
    
    # deepspeed needs to know your gradient accumulation steps before hand, so don't forget to pass it
    # Remember you still need to do gradient accumulation by yourself, just like you would have done without deepspeed
    # deepspeed_plugin = DeepSpeedPlugin(zero_stage=3, gradient_accumulation_steps=1)
    # deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = 2
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    accelerator = Accelerator(mixed_precision='fp16',log_with="wandb",project_dir=training_args.output_dir) 

    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = training_args.per_device_train_batch_size

    # load model
    model = ValleyLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    model.model.multi_image = data_args.multi_image
    model.model.multi_image_mode = data_args.multi_image_mode

    # wether freeze parameter or tune part of model
    
    if model_args.tune_llm_layer:
        tune_llm_layer = model_args.tune_llm_layer.split(',')
        print(tune_llm_layer)
        for k,v in model.model.named_parameters():
            if '.'.join(k.split('.')[0:2]) in tune_llm_layer:
                v.requires_grad=True
            else:
                v.requires_grad = False
    

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in model_args.model_name_or_path:
        tokenizer.add_special_tokens({
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        })

    if data_args.video_data_path:
        tokenizer.add_tokens([DEFAULT_VIDEO_FRAME_TOKEN,DEFAULT_VI_START_TOKEN,DEFAULT_VI_END_TOKEN], special_tokens=True)

    if model_args.vision_tower is not None:
        model_vision_dict = model.get_model().initialize_vision_modules(
            vision_tower=model_args.vision_tower,
            mm_vision_select_layer=model_args.mm_vision_select_layer,
            pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter
        )
        

        vision_config = model_vision_dict['vision_config']

        data_args.image_token_len = model_vision_dict['image_token_len']
        data_args.image_processor = model_vision_dict['image_processor']
        data_args.is_multimodal = True

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        vision_config.use_im_start_end = training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.sep_image_conv_front = data_args.sep_image_conv_front
        model.initialize_vision_tokenizer(mm_use_im_start_end=model_args.mm_use_im_start_end, tokenizer=tokenizer, device=training_args.device,
                                          tune_mm_mlp_adapter=model_args.tune_mm_mlp_adapter, pretrain_mm_mlp_adapter=model_args.pretrain_mm_mlp_adapter)

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() if (n in decay_parameters and p.requires_grad)
            ],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters() if (n not in decay_parameters and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    if accelerator.state.deepspeed_plugin.deepspeed_config['zero_optimization']['offload_optimizer']['device'] == 'cpu':
        training_args.optim = 'adam_deepspeed'
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
    optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)


    # dataset
    data_moudle  = make_video_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    train_dataset,  collate_fn = data_moudle['train_dataset'], data_moudle['data_collator']
    
    train_dataloader = DataLoader(train_dataset, 
                                  batch_size=training_args.per_device_train_batch_size,
                                  shuffle=True, 
                                  drop_last=True, 
                                  collate_fn=collate_fn)

    num_training_steps = (len(train_dataloader) * training_args.num_train_epochs) // accelerator.gradient_accumulation_steps

    lr_scheduler = get_scheduler(
                training_args.lr_scheduler_type,
                optimizer= optimizer,
                num_warmup_steps=training_args.get_warmup_steps(num_training_steps),
                num_training_steps=num_training_steps,
            )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(model, optimizer, train_dataloader, lr_scheduler)
    dtype = torch.float32
    if training_args.fp16:
        dtype = torch.float16
    if training_args.bf16:
        dtype = torch.bfloat16
    # print('current device',model.device)
    model.get_model().vision_tower.to(dtype=dtype)



    for k,v in model.model.named_parameters():
        if v.requires_grad==True:
            print(k)

    if model_args.model_checkpoint:
        accelerator.load_state(model_args.model_checkpoint)

    global_step = 0
    metric = SFTMetric(device=torch.cuda.current_device())
    model.train()
    if accelerator.is_main_process:
        wandb.init(project="video_llava")
    progress_bar = tqdm(range(len(train_dataloader)*int(training_args.num_train_epochs)), disable=not accelerator.is_local_main_process)
    
    for epoch in range(int(training_args.num_train_epochs)):
        for batch_cnt, inputs in enumerate(train_dataloader):
            try:
                if batch_cnt == 1 and epoch == 0:
                    torch.cuda.empty_cache()
                if accelerator.is_main_process:
                    starter, ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                optimizer.zero_grad()
                labels = inputs["labels"]
                output = model(**inputs)
                loss = output.loss

                metric(output.logits, labels, loss)
                acc, train_loss = metric.get_metric()

                accelerator.backward(loss)
                optimizer.step()
                if not accelerator.optimizer_step_was_skipped:
                    lr_scheduler.step()

                get_accelerator().empty_cache()
                global_step += 1

                if accelerator.is_main_process:
                    accelerator.print(f"epoch: {epoch}, cureent step: {batch_cnt}, total step: {len(train_dataloader)}, skip:{accelerator.optimizer_step_was_skipped}, loss:{round(train_loss, 3)}, acc:{round(acc, 3)}, length:{len(inputs.pop('input_ids')[0])}, lr:{lr_scheduler.get_last_lr()[0]}")

                    wandb.log({
                        "train/loss": train_loss,
                        "train/acc": acc,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }, step=global_step)
                    model.train()   

                progress_bar.update(1)
                if global_step % training_args.save_steps == 0:
                    accelerator.save_state(training_args.output_dir+'/checkpoint-'+str(global_step))
            except Exception as e:
                accelerator.print(e)
                continue
    if global_step % training_args.save_steps != 0:
        accelerator.save_state(training_args.output_dir+'/checkpoint-'+str(global_step))
    
    # unwrapped_model = accelerator.unwrap_model(model)
    # unwrapped_model.save_pretrained(
    #     training_args.output_dir,
    #     is_main_process=accelerator.is_main_process,
    #     save_function=accelerator.save,
    #     state_dict=accelerator.get_state_dict(model),
    # )
    tokenizer.save_pretrained(training_args.output_dir)

if __name__ == '__main__':
    set_seed(42)
    train()           