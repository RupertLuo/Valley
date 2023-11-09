"""
A model worker executes the model.
"""
import argparse
import asyncio
import dataclasses
import logging
import json
import time
from typing import List, Union
import threading
import uuid
from valley.utils import disable_torch_init
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import uvicorn
from functools import partial
import sys
sys.path.append('./valley')
from constants import WORKER_HEART_BEAT_INTERVAL
from util.config import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, DEFAULT_VIDEO_FRAME_TOKEN, DEFAULT_VI_START_TOKEN, DEFAULT_VI_END_TOKEN
from utils import (build_logger, server_error_msg,
    pretty_print_semaphore)
from valley.model.valley_model import ValleyLlamaForCausalLM
from transformers import CLIPImageProcessor, CLIPVisionModel
import decord
from torchvision import transforms
from data import video_transform
import numpy as np

GB = 1 << 30

worker_id = str(uuid.uuid4())[:6]
logger = build_logger("model_worker", f"model_worker_{worker_id}.log")
global_counter = 0

model_semaphore = None



def heart_beat_worker(controller):

    while True:
        time.sleep(WORKER_HEART_BEAT_INTERVAL)
        controller.send_heart_beat()


def load_model(model_path, model_name, num_gpus):
    if num_gpus == 1:
        kwargs = {}
    else:
        kwargs = {
            "device_map": "auto",
            "max_memory": {i: "13GiB" for i in range(num_gpus)},
        }
    disable_torch_init()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = ValleyLlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    # for multi image
    model.model.multi_image = True
    model.model.multi_image_mode = 'concatenate'

    # for image encoder
    image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=torch.float16)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

    vision_tower = model.get_model().vision_tower
    # if vision_tower.device.type == 'meta':
    #     vision_tower = CLIPVisionModel.from_pretrained(vision_tower.config._name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).cuda()
    #     model.get_model().vision_tower = vision_tower
    # else:
    vision_tower.to(device='cuda', dtype=torch.float16)
    vision_config = vision_tower.config
    vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
    vision_config.use_im_start_end = mm_use_im_start_end
    if mm_use_im_start_end:
        vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

    if num_gpus == 1:
        model.cuda()

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len


class ModelWorker:
    def __init__(self, controller_addr, worker_addr,
                 worker_id, no_register,
                 model_path, model_name,
                 keep_aspect_ratio,
                 num_gpus):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.worker_id = worker_id
        if model_path.endswith("/"):
            model_path = model_path[:-1]
        if model_name is None:
            model_paths = model_path.split("/")
            if model_paths[-1].startswith('checkpoint-'):
                self.model_name = model_paths[-2] + "_" + model_paths[-1]
            else:
                self.model_name = model_paths[-1]
        else:
            self.model_name = model_name

        logger.info(f"Loading the model {self.model_name} on worker {worker_id} ...")
        self.keep_aspect_ratio = keep_aspect_ratio
        self.tokenizer, self.model, self.image_processor, self.context_len = load_model(
            model_path, self.model_name, num_gpus)
        self.is_multimodal = 'valley' in model_path.lower()

        if not no_register:
            self.register_to_controller()
            self.heart_beat_thread = threading.Thread(
                target=heart_beat_worker, args=(self,))
            self.heart_beat_thread.start()

    def register_to_controller(self):
        logger.info("Register to controller")

        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        r = requests.post(url, json=data)
        assert r.status_code == 200

    def send_heart_beat(self):
        logger.info(f"Send heart beat. Models: {[self.model_name]}. "
                    f"Semaphore: {pretty_print_semaphore(model_semaphore)}. "
                    f"global_counter: {global_counter}")

        url = self.controller_addr + "/receive_heart_beat"

        while True:
            try:
                ret = requests.post(url, json={
                    "worker_name": self.worker_addr,
                    "queue_length": self.get_queue_length()}, timeout=5)
                exist = ret.json()["exist"]
                break
            except requests.exceptions.RequestException as e:
                logger.error(f"heart beat error: {e}")
            time.sleep(5)

        if not exist:
            self.register_to_controller()

    def get_queue_length(self):
        if model_semaphore is None:
            return 0
        else:
            return args.limit_model_concurrency - model_semaphore._value + (len(
                model_semaphore._waiters) if model_semaphore._waiters is not None else 0)

    def get_status(self):
        return {
            "model_names": [self.model_name],
            "speed": 1,
            "queue_length": self.get_queue_length(),
        }

    @torch.inference_mode()
    def generate_stream(self, params):
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor
        prompt = params["prompt"]
        ori_prompt = prompt
        images = params.get("images", None)
        if images is not None and len(images) > 0 and self.is_multimodal:
            from PIL import Image
            from io import BytesIO
            import base64
            assert type(images) is list
            if len(images) > 0:
                # assert len(images) == 1, "Only support one image for now"
                images = [Image.open(BytesIO(base64.b64decode(image))) for image in images]
                assert len(images) == prompt.count(DEFAULT_IMAGE_TOKEN), "Number of images does not match number of <image> tokens in prompt"

                if self.keep_aspect_ratio:
                    new_images = []
                    for image_idx, image in enumerate(images):
                        max_hw, min_hw = max(image.size), min(image.size)
                        aspect_ratio = max_hw / min_hw
                        max_len, min_len = 448, 224
                        shortest_edge = int(min(max_len / aspect_ratio, min_len))
                        image = image_processor.preprocess(image, return_tensors='pt', do_center_crop=False, size={"shortest_edge": shortest_edge})['pixel_values'][0]
                        new_images.append(image.to(self.model.device, dtype=torch.float16))
                        # replace the image token with the image patch token in the prompt (each occurrence)
                        cur_token_len = (image.shape[1]//14) * (image.shape[2]//14)
                        replace_token = DEFAULT_IMAGE_PATCH_TOKEN * cur_token_len
                        if getattr(self.model.config, 'mm_use_im_start_end', False):
                            replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                        prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token, 1)
                    images = new_images
                else:
                    images = image_processor(images, return_tensors='pt')['pixel_values']
                    images = images.to(self.model.device, dtype=torch.float16)
                    replace_token = DEFAULT_IMAGE_PATCH_TOKEN * 256    # HACK: 256 is the max image token length hacked
                    if getattr(self.model.config, 'mm_use_im_start_end', False):
                        replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN + DEFAULT_VI_START_TOKEN + DEFAULT_VIDEO_FRAME_TOKEN * 1 + DEFAULT_VI_END_TOKEN
                    prompt = prompt.replace(DEFAULT_IMAGE_TOKEN, replace_token)
            else:
                images = None
            image_args = {"images": images.unsqueeze(0)}
        else:
            images = None
            image_args = {}

        l_prompt = len(prompt)
        temperature = float(params.get("temperature", 1.0))
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", None)
        stop_idx = None
        if stop_str is not None:
            stop_idx = tokenizer(stop_str).input_ids
            if len(stop_idx) == 1:
                stop_idx = stop_idx[0]
            else:
                stop_idx = None

        input_ids = tokenizer(prompt).input_ids
        output_ids = list(input_ids)
        pred_ids = []

        max_src_len = self.context_len - max_new_tokens - 8
        input_ids = input_ids[-max_src_len:]
        
        print(input_ids)
        past_key_values = None
        for i in range(max_new_tokens):
            if i == 0:
                out = model(
                    torch.as_tensor([input_ids]).cuda(),
                    use_cache=True,
                    **image_args)
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                attention_mask = torch.ones(
                    1, past_key_values[0][0].shape[-2] + 1, device="cuda")
                out = model(input_ids=torch.as_tensor([[token]], device="cuda"),
                            use_cache=True,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values)
                logits = out.logits
                past_key_values = out.past_key_values

            last_token_logits = logits[0][-1]
            if temperature < 1e-4:
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits / temperature, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))

            output_ids.append(token)
            pred_ids.append(token)

            if stop_idx is not None and token == stop_idx:
                stopped = True
            elif token == tokenizer.eos_token_id:
                stopped = True
            else:
                stopped = False

            if i % args.stream_interval == 0 or i == max_new_tokens - 1 or stopped:
                cur_out = tokenizer.decode(pred_ids, skip_special_tokens=True)
                pos = cur_out.rfind(stop_str)
                if pos != -1:
                    cur_out = cur_out[:pos]
                    stopped = True
                output = ori_prompt + cur_out

                ret = {
                    "text": output,
                    "error_code": 0,
                }
                # logger.info(f"==== request ====\n{ret}")
                yield json.dumps(ret).encode() + b"\0"

            if stopped:
                logger.info(f"==== request ====\n{ret}")
                break

        if past_key_values is not None:
            del past_key_values
    def load_video(self, path):
        video_reader = decord.VideoReader(path, num_threads=1, ctx=decord.cpu(0))
        decord.bridge.set_bridge('torch')
        video_len = len(video_reader)
        video = video_reader.get_batch(np.linspace(0, video_len - 1, 8).astype(np.int_)).byte()
        video = video.permute(3, 0, 1, 2) # 3 x 8 x height x width


        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        crop_size, scale_size = 224, 256
        trans = transforms.Compose([
            video_transform.TensorToNumpy(),
            video_transform.Resize(scale_size),
            video_transform.CenterCrop(crop_size),
            video_transform.ClipToTensor(channel_nb=3),
            video_transform.Normalize(mean=input_mean, std=input_std)
        ])

        video = trans(video)
        return video
    @torch.inference_mode()
    def generate_video_stream(self, params):
        tokenizer, model, image_processor = self.tokenizer, self.model, self.image_processor
        prompt = params["prompt"]
        ori_prompt = prompt
        videos = params.get("videos", None)
        if videos is not None and len(videos) > 0 and self.is_multimodal:

            assert type(videos) is list
            if len(videos) > 0:
                assert len(videos) == 1, "Only support one image for now"
                logger.info('load video from '+str(videos))
                videos = [self.load_video(video) for video in videos]
                assert len(videos) == prompt.count(DEFAULT_VIDEO_TOKEN), "Number of video does not match number of <video> tokens in prompt"

                videos = videos[0]
                videos = videos.to(self.model.device, dtype=torch.float16)
                videos = videos.permute(1,0,2,3)
                replace_token = DEFAULT_IMAGE_PATCH_TOKEN * 256    # HACK: 256 is the max image token length hacked
                if getattr(self.model.config, 'mm_use_im_start_end', False):
                    replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN + DEFAULT_VI_START_TOKEN + DEFAULT_VIDEO_FRAME_TOKEN * videos.shape[0] + DEFAULT_VI_END_TOKEN
                prompt = prompt.replace(DEFAULT_VIDEO_TOKEN, replace_token)
                print(prompt)
            else:
                videos = None
            video_args = {"images": videos.unsqueeze(0)}
            print(videos.unsqueeze(0).shape)
        else:
            videos = None
            video_args = {}

        l_prompt = len(prompt)
        temperature = float(params.get("temperature", 1.0))
        max_new_tokens = min(int(params.get("max_new_tokens", 256)), 1024)
        stop_str = params.get("stop", None)
        stop_idx = None
        if stop_str is not None:
            stop_idx = tokenizer(stop_str).input_ids
            if len(stop_idx) == 1:
                stop_idx = stop_idx[0]
            else:
                stop_idx = None
        # print(prompt)
        input_ids = tokenizer(prompt).input_ids
        output_ids = list(input_ids)
        pred_ids = []

        max_src_len = self.context_len - max_new_tokens - 8
        input_ids = input_ids[-max_src_len:]
        
        past_key_values = None
        for i in range(max_new_tokens):
            if i == 0:
                out = model(
                    torch.as_tensor([input_ids]).cuda(),
                    use_cache=True,
                    **video_args)
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                attention_mask = torch.ones(
                    1, past_key_values[0][0].shape[-2] + 1, device="cuda")
                out = model(input_ids=torch.as_tensor([[token]], device="cuda"),
                            use_cache=True,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values)
                logits = out.logits
                past_key_values = out.past_key_values

            last_token_logits = logits[0][-1]
            if temperature < 1e-4:
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits / temperature, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))

            output_ids.append(token)
            pred_ids.append(token)

            if stop_idx is not None and token == stop_idx:
                stopped = True
            elif token == tokenizer.eos_token_id:
                stopped = True
            else:
                stopped = False

            if i % args.stream_interval == 0 or i == max_new_tokens - 1 or stopped:
                cur_out = tokenizer.decode(pred_ids, skip_special_tokens=True)
                pos = cur_out.rfind(stop_str)
                if pos != -1:
                    cur_out = cur_out[:pos]
                    stopped = True
                output = ori_prompt + cur_out

                ret = {
                    "text": output,
                    "error_code": 0,
                }
                # logger.info(f"==== request ====\n{ret}")
                yield json.dumps(ret).encode() + b"\0"

            if stopped:
                logger.info(f"==== request ====\n{ret}")
                break

        if past_key_values is not None:
            del past_key_values

    def generate_stream_gate(self, params):
        try:
            if 'videos' not in params:
                for x in self.generate_stream(params):
                    yield x
            else:
                for x in self.generate_video_stream(params):
                    yield x
        except ValueError as e:
            print("Caught ValueError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"
        except torch.cuda.CudaError as e:
            print("Caught torch.cuda.CudaError:", e)
            ret = {
                "text": server_error_msg,
                "error_code": 1,
            }
            yield json.dumps(ret).encode() + b"\0"


app = FastAPI()


def release_model_semaphore(fn=None):
    model_semaphore.release()
    if fn is not None:
        fn()


@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    global model_semaphore, global_counter
    global_counter += 1
    params = await request.json()

    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(args.limit_model_concurrency)
    await model_semaphore.acquire()
    worker.send_heart_beat()
    generator = worker.generate_stream_gate(params)
    background_tasks = BackgroundTasks()
    background_tasks.add_task(partial(release_model_semaphore, fn=worker.send_heart_beat))
    return StreamingResponse(generator, background=background_tasks)


@app.post("/worker_get_status")
async def get_status(request: Request):
    return worker.get_status()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=39999)
    parser.add_argument("--worker-address", type=str,
        default="http://localhost:39999")
    parser.add_argument("--controller-address", type=str,
        default="http://localhost:20000")
    parser.add_argument("--model-path", type=str, default="")
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--multi-modal", action="store_true", help="Multimodal mode is automatically detected with model name, please make sure `valley` is included in the model path.")
    parser.add_argument("--keep-aspect-ratio", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--limit-model-concurrency", type=int, default=5)
    parser.add_argument("--stream-interval", type=int, default=2)
    parser.add_argument("--no-register", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    if args.multi_modal:
        logger.warning("Multimodal mode is automatically detected with model name, please make sure `valley` is included in the model path.")

    worker = ModelWorker(args.controller_address,
                         args.worker_address,
                         worker_id,
                         args.no_register,
                         args.model_path,
                         args.model_name,
                         args.keep_aspect_ratio,
                         args.num_gpus)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
