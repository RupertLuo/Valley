import torch
from transformers import StoppingCriteria
from typing import Dict, Sequence
from valley import conversation as conversation_lib
import transformers
from transformers import CLIPImageProcessor
from valley.util.config import *
import copy
from torchvision import transforms
from valley.data import video_transform
import decord
import os
import numpy as np
from pathlib import Path
from PIL import Image
from valley.constants import *
import re

def collate_wrapper(batch):
    try:
        image_list = [b[0] for b in batch]
        prompt_list = [b[2] for b in batch]
        # input_ids = pad_sequence(prompt_list, padding_value = 0, batch_first = True)
        conv_list = [b[3] for b in batch]
        save_id_list = [b[4] for b in batch]
        label_list = [b[1] for b in batch]
    except:
        prompt_list, image_list, conv_list, label_list, save_id_list = None, None, None, None, None
    return prompt_list, image_list, conv_list, label_list, save_id_list



def collate_process_image_text(batch, tokenizer, image_processor):
    batch_input_ids, batch_image, conv_list, label_list, save_id_list = batch
    input_ids = torch.stack(batch_input_ids,dim=0)
    videos = []
    for this_batch_images in batch_image:
        if '.mp4' not in save_id_list[0] and '.avi' not in save_id_list[0]: # if not a video file, do image list process func
            video = image_processor.preprocess(
                this_batch_images, return_tensors='pt')['pixel_values']
            videos.append(video)
        else:
            videos.append(this_batch_images)
    return input_ids, videos, conv_list, label_list, save_id_list


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(
                output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False

# for finetune


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""
    
    if trainer.args.should_save:
        if getattr(trainer.args, 'lora' ,None):
            trainer.model.save_pretrained(output_dir)
            if trainer.args.tune_mm_mlp_adapter:
                trainer.model.base_model.model.save_pretrained(output_dir)
        else:
            state_dict = trainer.model.state_dict()
            cpu_state_dict = {
                key: value.cpu()
                for key, value in state_dict.items()
            }
            del state_dict
            trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers, only_mask_system = False):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    if not only_mask_system:
        for tokenized_len, speaker in zip(tokenized_lens, speakers):
            if speaker == "human":
                target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
            cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"].strip()
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args,
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if data_args.model_class == 'valley-product':
                # for multi image
                segs = re.split(r'<image[\d]*>',sentence["value"])
                sentence["value"] = '<image>'.join(segs[:data_args.max_img_num+1]) + ' '.join(segs[data_args.max_img_num+1:])
            
            else:
                if DEFAULT_IMAGE_TOKEN in sentence['value'] or DEFAULT_VIDEO_TOKEN in sentence['value']:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                    sentence['value'] = sentence['value'].replace(DEFAULT_VIDEO_TOKEN, '').strip()
                    sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                    sentence['value'] = sentence['value'].strip()
                    if "mmtag" in conversation_lib.default_conversation.version:
                        sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
                replace_token = DEFAULT_IMAGE_TOKEN
                if data_args.mm_use_im_start_end:
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN + DEFAULT_VI_START_TOKEN + DEFAULT_VI_END_TOKEN
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
            
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN + DEFAULT_VI_START_TOKEN + DEFAULT_VI_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    inference: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        if inference:
            conv.append_message(conv.roles[1], None)
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_mistral(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    inference: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        if inference:
            conv.append_message(conv.roles[1], None)
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    
    # assert (input_ids == 1).sum() == 2 and input_ids.shape[0] ==1 
    # input_ids = input_ids[:,1:]
    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.MISTRAL

    # Mask targets
    sep = "[/INST]"
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1
                
            if i == 0:
                target[cur_len : cur_len + round_len] = IGNORE_INDEX
            else:
                target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    inference: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        if inference:
            conv.append_message(conv.roles[1], None)
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    inference: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        if inference:
            conv.append_message(conv.roles[1], None)
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(tokenizer_image_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    only_mask_system: bool = False,
    inference: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image, inference = inference)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image, inference = inference)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, inference)
    if conversation_lib.default_conversation.version == "mistral":
        return preprocess_mistral(sources, tokenizer, has_image=has_image, inference = inference)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers, only_mask_system = only_mask_system )

    return dict(input_ids=input_ids, labels=targets)


def load_video(
        path,
        image_processer = None,
        frame_mode='fixed',
        fixed_frame_number=8,
        fps_number=0.5,
        frame_process_method='centercrop',
):
    if os.path.isfile(path):
        video_reader = decord.VideoReader(
            path, num_threads=1, ctx=decord.cpu(0))
        decord.bridge.set_bridge('torch')
        video_len = len(video_reader)

        if frame_mode == 'fixed':
            video = video_reader.get_batch(np.linspace(
                0, video_len - 1, fixed_frame_number).astype(np.int_)).byte()  # 8, height,width,3
            video = video.permute(3, 0, 1, 2)  # 3 x 8 x height x width
        elif frame_mode == 'fps':
            fps_offset = int(round(video_reader.get_avg_fps())/fps_number)
            video = video_reader.get_batch(
                range(0, video_len, fps_offset)).byte()
            video = video.permute(3, 0, 1, 2)  # 3 x 8 x height x width
        input_mean = [0.48145466, 0.4578275, 0.40821073] # Consistent with clilp preprocessing
        input_std = [0.26862954, 0.26130258, 0.27577711] #Consistent with clilp preprocessing
        crop_size, scale_size = 224, 224
        trans = transforms.Compose([
            video_transform.TensorToNumpy(),
            video_transform.Resize(scale_size),
            video_transform.CenterCrop(crop_size),
            video_transform.ClipToTensor(channel_nb=3),
            video_transform.Normalize(mean=input_mean, std=input_std)
        ])
        video = trans(video)
        # print(video.shape)
    else:
        video_frames = list(Path(path).rglob('*'))
        # print(video_frames)
        if frame_mode == 'fixed':
            video_frames = [video_frames[i] for i in np.linspace(
                0, len(video_frames) - 1, fixed_frame_number).astype(np.int_)]
        elif frame_mode == 'fps':
            raise ValueError('Input folder is not support this frame mode')
        else:
            raise ValueError('Frame mode is only support "fps" or "fixed"')
        video_frames = [Image.open(str(path)) for path in video_frames]

        if frame_process_method == 'resize':
            min_length = min(video_frames[0].size)
            resize = transforms.Resize([min_length, min_length])
            video_frames = [resize(frame) for frame in video_frames]
            # test_frame = video_frames[0]
        if image_processer is None:
            processor_config = {"crop_size": 224,
                                "do_center_crop": True,
                                "do_normalize": True,
                                "do_resize": True,
                                "feature_extractor_type": "CLIPFeatureExtractor",
                                "image_mean": [0.48145466, 0.4578275, 0.40821073],
                                "image_std":  [0.26862954, 0.26130258,0.27577711],
                                "resample": 3,
                                "size": 224
                                }
            image_processer = CLIPImageProcessor(**processor_config)
        video = image_processer.preprocess(
            video_frames, return_tensors='pt')['pixel_values']
        video = video.permute(1, 0, 2, 3)
    return video


def process_output(output):
    return_list = []
    for i, out in enumerate(output):
        while True:
            cur_len = len(out)
            out = out.strip()
            for pattern in ['###', 'Assistant:', 'Response:','LLaVA:']:
                if out.startswith(pattern):
                    out = out[len(pattern):].strip()
            if len(out) == cur_len:
                break
        try:
            index = out.index(CONV_SEP)
        except ValueError:
            out += CONV_SEP
            index = out.index(CONV_SEP)

        out = out[:index].strip()
        string_out = out.replace('\n','') + '\n'
        return_list.append(string_out)
    return return_list