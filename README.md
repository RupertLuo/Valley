# ⛰️Valley: Video Assistant with Large Language model Enhanced abilitY
Understanding Complex Videos Relying on Large Language and Vision Models


[[Project Page](https://valley-vl.github.io/)] [[Paper](https://arxiv.org/pdf/2306.07207.pdf)]~[[demo]()]~

The online demo is no longer available, because we released the code for offline demo deployment

**Video Assistant with Large Language model Enhanced abilitY** <br>
[Ruipu Luo*](https://github.com/RupertLuo), [Ziwang Zhao*](), [Min Yang*](https://github.com/feymanpriv) (*Equal Contribution)

<p align="center">
    <img src="valley/logo/lama_with_valley.jpeg" width="100%"><br>
    Generated by <a href="https://stablecog.com/">stablecog</a> via "A cute llama with valley"
</p>

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)
**Usage and License Notices**: The data, code and checkpoint is intended and licensed for research use only. They are also restricted to uses that follow the license agreement of LLaMA, Vicuna and GPT-4. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.

## Release
- [8/14] 🔥 We released the Chinese version of Valley! Now its 7B and 13b weights are available at [Chinese-Valley7B-V1](https://huggingface.co/Zhaoziwang/chinese_valley7b_v1) and [Chinese-Valley13B-V1](https://huggingface.co/Zhaoziwang/chinese_valley13b_v1).
- [8/10] 🔥 Realeased pretrain stage weight of 13b and 7b ,[Valley2-7b-pretrain](https://huggingface.co/luoruipu1/Valley2-7b-pretrain/), [valley-13b-pretrain](https://huggingface.co/luoruipu1/valley-13b-pretrain).
- [8/8] 🔥 We released the self-collected and expanded instruction fine-tuning dataset ([Valley-Instruct-73k](https://huggingface.co/datasets/luoruipu1/Valley-Instruct-73k)).
- [8/7]  🔥 We released [Valley2-7b](https://huggingface.co/luoruipu1/Valley2-7b), It replaces Vicuna with Llama 2.
- [7/23] 🫧 We modified the our training code to make it easier to train valley and also support the training of lora.
- [7/5]  🫧 Release training code for valley, and upload our pretraining data.
- [6/21] 🫧 upload offline demo code.
- [6/14] 🫧 build a share link ~[[demo]()]~.
- [6/13] 🫧 We uploaded model weight of [Valley-13b-v1-delta](https://huggingface.co/luoruipu1/valley-13b-v1-delta).
- [6/12] 🫧 We released Valley: Video Assistant with Large Language model Enhanced abilitY.  Checkout the [paper](https://arxiv.org/pdf/2306.07207.pdf).

## Todo
- ~~Release inference code~~
- ~~Upload weight of **Valley-v1** and build a share link demo~~
- ~~Upload offline demo code~~
- ~~Release 703k pretraining data and 73k instruction tuning data~~
- ~~Upload pretrain and tuning code~~
- ~~Upload weight of Valley2-7B~~ and Valley-v3

## Install
1. Clone this repository and navigate to Valley folder
```
git clone https://github.com/RupertLuo/Valley.git
cd Valley
```
2. Install Package
```
conda create -n valley python=3.10 -y
conda activate valley
pip install --upgrade pip
pip install -e .
```
## Data
In the pretrain stage, we use the data from [LLaVA-CC3M-Pretrain-595K](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) and the [Valley-webvid2M-Pretrain-703K](https://huggingface.co/datasets/luoruipu1/Valley-webvid2M-Pretrain-703K) collected and filtered by ourselves. The acquisition of picture and video data can refer to [LLAVA]( https://llava-vl.github.io/) and [Webvid](https://github.com/m-bain/webvid)

In the finetune stage, we use the data from [LLaVA-instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K), [VideoChat-instruct-11K](https://github.com/OpenGVLab/InternVideo/tree/main/Data/instruction_data) and our self-collected [Valley-Instruct-73k](https://huggingface.co/datasets/luoruipu1/Valley-Instruct-73k). For the images and videos of the first two parts, please refer to their official website. Here we describe how we obtain the data we collect ourselves ([Valley-Instruct-73k](https://huggingface.co/datasets/luoruipu1/Valley-Instruct-73k)).

1. Part of Valley-Instruct-73k is collected from the open source dataset [VATEX](https://eric-xw.github.io/vatex-website/explore.html), which contains about 20k downloadable videos. You can download the original annotation file ("ava_vatex_training_v1.0.json") from its official website. Its video comes from YouTube, and now there are many open source tools that can download YouTube videos by video id. We provide a tool to download its videos, the tool is located in the [Crawler](./Crawler/) folder, please read the tool's [Readme.md](./Crawler/README.md) to use it.
2. Another part of Valley-Instruct-73k is collected from a video site, named [JukinMedia](https://www.jukinmedia.com/). It contains a wide variety of videos.  We also provide a tool to download jukinmedia videos and its high quality descriptions, the tool is located in the [Crawler](./Crawler/) folder, please read the tool's [Readme.md](./Crawler/README.md) to use it.


## ValleyWeight
### Valley 13b v1
We release [Valley-13b-v1](https://huggingface.co/luoruipu1/valley-13b-v1-delta) delta weights weights to comply with the LLaMA model license. You can apply this delta weights to original LLaMA model weight through the instructions blew:

1. Get the original LLaMA weights in the huggingface format by following the instructions structions [here](https://huggingface.co/docs/transformers/main/model_doc/llama).
2. Use the following scripts to get Valley weights by applying our delta ([13b-v1](https://huggingface.co/luoruipu1/valley-13b-v1-delta)).
```bash
python3 valley/model/apply_delta.py \
    --base /path/to/llama-13b \
    --target /output/path/to/Valley-13B-v1 \
    --delta /path/to/valley-13b-v1-delta
```
### Valley2 7b
For the Valley2-7b model, we provide direct weights, the address is [here](https://huggingface.co/luoruipu1/Valley2-7b)

### Chinese Valley 13b
We now support **Chinese valley**. We use "BelleGroup/BELLE-LLaMA-EXT-13B" as LLM backbone, and  "OFA-Sys/chinese-clip-vit-large-patch14" for visual backbone, the address is [here](https://huggingface.co/Zhaoziwang/chinese_valley_v1).

### Pretrain Weight
We provide [13b](https://huggingface.co/luoruipu1/valley-13b-pretrain) and [7b](https://huggingface.co/luoruipu1/Valley2-7b-pretrain/) pre-trained weights so that people can fine-tune directly on our pre-trained weights with their own fine-tuning data.

## Web UI
<p align="center">
    <img src="valley/logo/demo.GIF" width="100%"><br>
</p>

The framework of this webUI comes from [LLaVA](https://github.com/haotian-liu/LLaVA) and [FastChat](https://github.com/lm-sys/FastChat), we modified a part of the code to make this demo support the input of video and images.
#### launch a controller
```bsah
python valley/serve/controller.py
```
#### launch a model worker
```bsah
python valley/serve/model_worker.py --model-path /path/to/valley-13b-v1
```
Ps: At present, only single card mode is supported to load the model, and at least 30G of video memory is required, so the graphics card needs at least one Tesla V100.
#### launch a gradio demo
```bash
python valley/serve/gradio_web_server_video.py --share
```


## Inference Valley in Command Line
We now update inference code which is more convient, and supports input in the form of openai api.

Inference CLI
```
python3 inference/run_valley.py --model-name [PATH TO VALLEY WEIGHT] --video_file [PATH TO VIDEO] --quary [YOUR QUERY ON THE VIDEO]
```

Inference Chinese Valley
```
python3 inference/run_valley.py --model-name [PATH TO CHINESE VALLEY WEIGHT] --video_file [PATH TO VIDEO] --query [YOUR QUERY ON THE VIDEO] --system-prompt "你是大型语言视觉助手 Chinese-Valley。你能够理解用户提供的视觉内容或视频，并使用自然语言协助用户完成各种任务。请仔细按照人类的指令进行回答，并详细解释你的答案。"
```

Inference in code

- You can utilize the code located at [valley/inference/run_valley_llamma_v2.py](valley/inference/run_valley_llamma_v2.py) to run inference on a video. All that's required is a video path

```bash
python valley/inference/run_valley_llamma_v2.py --video_file <path-to-video-file>
```

- luoruipu1/Valley2-7b is used in the provided code.

## Train Valley Step By Step

Inspired by LLAVA, we adopt a two-stage training method. The pre-training stage uses the [Valley-webvid2M-Pretrain-703K](https://huggingface.co/datasets/luoruipu1/Valley-webvid2M-Pretrain-703K) and [LLaVA-CC3M-Pretrain-595K](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain).  And fine-tune stage uses [LLaVA-instruct-150K](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K) ,  [VideoChat-instruct-11K](https://github.com/OpenGVLab/InternVideo/tree/main/Data/instruction_data)  and [Valley-Instruct-73k](https://huggingface.co/datasets/luoruipu1/Valley-Instruct-73k)

We modified our code for training valley and managed the model hyperparameters with yaml files. Run the following two scripts to perform valley training.

### Pretrain
The llm backbone that currently supports pre-training is Llama(7b,13b), vicuna(7b,13b), stable-vicuna(13b), Llama2(chat-7b, chat-13b). You need to download these open source language model weights yourself and convert them to the huggingface format.
```shell
bash valley/train/train.sh valley/configs/experiment/valley_stage1.yaml
```

#### Finetune

```shell
bash valley/train/train.sh valley/configs/experiment/valley_stage2.yaml
```



## Acknowledgement

- [LLaVA](https://github.com/haotian-liu/LLaVA) & [MOSS](https://github.com/OpenLMLab/MOSS): Thanks to these two repositories for providing high-quality code, our code is based on them.
## Citation
If the project is helpful to your research, please consider citing our paper as follows

```bibtex
@misc{luo2023valley,
      title={Valley: Video Assistant with Large Language model Enhanced abilitY},
      author={Ruipu Luo and Ziwang Zhao and Min Yang and Junwei Dong and Minghui Qiu and Pengcheng Lu and Tao Wang and Zhongyu Wei},
      year={2023},
      eprint={2306.07207},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
