# ⛰️Valley: Video assistant towards large language model makes everything easy
Understanding Complex Videos Relying on Large Language and Vision Models

[[Project Page]()] [[Paper]()]

**Video assistant towards large language model makes everything easy** <br>
[Ruipu Luo*](https://hliu.cc), [Ziwang Zhao*](), [Min Yang*]() (*Equal Contribution)

<p align="center">
    <img src="valley/logo/lama_with_valley.jpeg" width="100%"></a> 
</p>

[![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/LICENSE)
[![Data License](https://img.shields.io/badge/Data%20License-CC%20By%20NC%204.0-red.svg)](https://github.com/tatsu-lab/stanford_alpaca/blob/main/DATA_LICENSE)
**Usage and License Notices**: The data, code and checkpoint is intended and licensed for research use only. They are also restricted to uses that follow the license agreement of LLaMA, Vicuna and GPT-4. The dataset is CC BY NC 4.0 (allowing only non-commercial use) and models trained using the dataset should not be used outside of research purposes.

## Release
- [6/12] 🫧 We released Valley: Video assistant towards large language model makes everything easy**.  Checkout the [paper]().

## Todo
- ~~Release inference code~~
- Upload weight of **Valley-v1** and build a share link demo
- Release 47k instruction tuning data 
- Upload pretrain and tuning code

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

## Inference Valley in Command Line
**Valley weight will be upload soon!**

inference CLI
```
python3 inference/run_valley.py --model_name [PATH TO VALLEY WEIGHT] --video_file [PATH TO VIDEO] --quary [YOUR QUERY ON THE VIDEO]
```



## Acknowledgement
- [LLaVA](https://github.com/haotian-liu/LLaVA) & [MOSS](https://github.com/OpenLMLab/MOSS): Thanks to these two repositories for providing high-quality code, our code is based on them.
