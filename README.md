# Diffuse Everything: Multimodal Diffusion Models on Arbitrary State Spaces

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b)](https://arxiv.org/abs/2506.07903)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![X](https://img.shields.io/badge/X-000000?logo=x&logoColor=white&style=flat-square)](https://x.com/YuchenZhu_ZYC/status/1934700344707363180)


Welcome to the official repository for **Diffuse Everything** (accepted at ICML 2025), for  <span style="color:#A020F0">**text-image joint generation**</span>. For mixed-type tabular data synthesis, please see [Diffuse-Everything-Tabular](https://github.com/yuchen-zhu-zyc/Diffuse-Everything-Tabular).

![Demonstration](assets/promo_v2.png)

## Introduction
**Diffuse Everything** is a general framework for building multimodal diffusion models for data of mixed modality, with a minimum need for tokenizers/VAEs/extra encoders. 

**Diffuse Everything** is built on <span style="color:orange">denoising markov models</span>, a generalized notion of denoising diffusion that characterizes the process using notions of Markov processes and their generators. This enables us to derive training objectives with theoretical guarantees in a general and modular fashion.

![denoising](assets/denoising_markov_models.jpg)

Specifically, in **Diffuse Everything**, we allow the diffusion process on each state space to have a <span style="color:orange">decoupled, independent schedule</span>, allowing modalities to be noised at their own pace. This enables the following benefits:
- Enjoying any-to-any within a single model
- A new guidance mechanism for multimodal generation

![decoupled](assets/decoupled.jpg)


## Latest Update

[Jun 23, 2025] We have open sourced the code.

[Jun 05, 2025] Code will soon be public. Please stay tuned!

[May 01, 2025] Diffuse Everything was accepted to ICML 2025.

## Features

- **Dataset Processing**: Support for download and processing of the dataset 
- **Multimodal Generation**: Support for text-to-image, image-to-text, and unconditional generation
- **Flexible Training**: Supports different training strategies and configurations
- **Advanced Sampling**: Implements various sampling techniques for high-quality generation
- **FID and CLIP Score Evaluation**: Built-in evaluation metrics for generated content
- **Distributed Training**: Native support for multi-GPU training

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd Multimodal-Diffusion
```

2. Install the dependencies:
```bash
conda create -n diff-every python=3.12
conda activate diff-every
pip install -r requirements.txt
```

## Dataset Preparation
We provide scripts to download and preprocess the dataset. Because the SAM dataset is too heavy, we erase the original files after processing them into the latent space.  Please follow the steps as clearly outlined, you should only need to set `your_folder` or other variables like resolution, but do include the `images, tokens` as described in the scripts.
For the SAM dataset you need to download the file containing the links to the raw data from [SAM download](https://ai.meta.com/datasets/segment-anything-downloads/), please delete the first line of that file so that each line is of the form:
```
sa_000020.tar	link
```
If your file with links is named `cc12m_links.txt` you can use the following script to download the dataset using 8 gpus:
```bash
torchrun --nproc-per-node 8 download_sam.py main --download_links cc12m_links.txt --output_folder your_folder/images --target_size 256 --batch_size 256 
```
To preprocess the captions you can use the following command:
```bash
torchrun --nproc-per-node 8 download_sam.py process-captions --tokenizer clip --block_size 120 --output_folder your_folder/tokens
```

Finally we create an index file to provide to the dataloader. You can create this index using the following command:
```bash
torchrun --nproc-per-node 8 download_sam.py process-indexes --folder your_folder
```
Finally update the `configs/sam_llava.yaml` file with the paths to your files.

## Training
We use wandb to check our training. If wandb is enabled we will log images every `--log_rate` iterations. These will be generated either conditionally if only  one modality is being trained or unconditionally. In both cases the result will be without using guidance, so samples will look worse, and should be understood as rough guidelines. 

We also perform FID evaluation every `--eval_rate` iterations without guidance. To do so you will need the coco reference statistics, which can be downloaded [here](https://drive.google.com/drive/folders/11fT6UlL2h8PXPRjb8ogcLr9klkYA-UTN?usp=sharing).  For other options please consult the `training.py` file.


### Basic Training

To start training you can use one of the followin commands. We recommend using the training strategy that we recommended in our paper, as otherwise it can be hard for the model to learn all tasks appropriately. Special emphasis should be placed to the text to image task. Our commands are for 8 gpus.
 

**Stage 1 Training** In this stage we train the noisy text to image model. For this stage make sure to set `text-depth` to 0 in the `mmdit.yaml`. A sample command to train is as follows:

```bash
torchrun --nproc-per-node 8 training.py \
    --modality continuous \
    --use_all_times \
    --batch_size 256 \
    --dir output_directory \
    --dataset_config_path configs/sam_llava.yaml \
    --net_config_path configs/mmdit.yaml \
    --enable_wandb
```
**Stage 2 Training** In this stage set `text-depth` to your desired value in `mmdit.yaml`. When loading the checkpoint since the model has changed, you need to comment out the line where the optimizer is loaded. Additionally you should load from the EMA so that the weights are correct. You can also freeze the continuous component if you wish using `--freeze_image`. This can be helpful if you want to preserve the result from the previous stage.
```bash
torchrun --nproc-per-node 8 training.py \
    --modality multimodal \
    --use_all_times \
    --freeze_joint \
    --load_from_ema \
    --batch_size 256 \
    --dir output_directory \
    --dataset_config_path configs/sam_llava.yaml \
    --net_config_path configs/mmdit.yaml \
    --load_checkpoint ckpt_path \
    --enable_wandb
```
**Stage 3 Training** In the final stage we train the model for a couple extra iterations on all tasks. This stage can be tricky as it could destroy the previous learned features, so track your training carefully. A sample command is:
```bash
torchrun --nproc-per-node 8 training.py \
    --modality multimodal \
    --use_all_times \
    --batch_size 256 \
    --dir output_directory \
    --dataset_config_path configs/sam_llava.yaml \
    --net_config_path configs/mmdit.yaml \
    --load_checkpoint ckpt_path \
    --enable_wandb
```

## Sampling
By default the commands here will download the model from hugginface. If you wish to sample from a model you are training use the `--load_checkpoint` flag.
### Unconditional Sampling
You can generate samples using a trained model. For example to generate 16 images you can use the command:

```bash
torchrun  sampling.py sampling \
                --dir out \
                --num_steps 50 \
                --cfg_scale 4. \
                --num_samples 16 \
                --batch_size 16 \
                --seed 42 \
                --plot_with_text
```

### Conditional Sampling
If you have downloaded the dataset following the instructions above, you can sample conditionally by looping over the dataset using the following command:
```bash
torchrun sampling.py sample-dataset-conditional \
                    --modality continuous \
                    --dir out \
                    --num_steps 50 \
                    --cfg_scale 4. \
                    --guidance_left .3 \
                    --guidance_right .8 \
                    --seed 42 --batch_size 16 \
```
You can change the modality to `discrete` to generate samples from text to image.

You can also generate images from a `.json` file with prompts. You can use the following command. We provide the captions that we used to evaluate FID in the `coco-captions.json` file. 
```bash
torchrun --nproc_per_node=8 sampling.py sampling-conditional \
                    --dir out \
                    --num_steps 50 \
                    --cfg_scale 5. \
                    --guidance_left .3 \
                    --guidance_right .8 \
                    --seed 42 --batch_size 512 \
                    --limit_context_len 40 \
                    --prompt fid_computation/coco-captions.json 
```
### FID Evaluation
You can then evaluate FID by using the following command:

```bash
torchrun fid_score.py --path ${image_folder_path} --ref_path ${ref_path}
```
The reference path can be either a folder of images or a .npz containing a dictionary with the mean and std of the reference statistics. The script will print out the FID score and save the reference statistics for future reference.

## License
Unless otherwise stated in the header of the file, this repo is under the MIT license.

## Citation
If you find our work and repo help, we would appreciate your citations :smiling_face_with_three_hearts:

```
@inproceedings{
    rojas2025diffuse,
    title={Diffuse Everything: Multimodal Diffusion Models on Arbitrary State Spaces},
    author={Kevin Rojas and Yuchen Zhu and Sichen Zhu and Felix X-F. Ye and Molei Tao},
    booktitle={Forty-second International Conference on Machine Learning},
    year={2025},
    url={https://openreview.net/forum?id=AjbiIcRt6q}
}
```
```
@article{rojas2025diffuse,
    title={Diffuse Everything: Multimodal Diffusion Models on Arbitrary State Spaces},
    author={Rojas, Kevin and Zhu, Yuchen and Zhu, Sichen and Ye, Felix X-F and Tao, Molei},
    journal={arXiv preprint arXiv:2506.07903},
    year={2025}
}
```

## Acknowledgments
This project builds upon the SAM-LLaVA dataset introduced in Pixart-α. For the architecture we branched off of the great implementation from lucid rains. Other repos that have been useful in building this repo were score_sde, sedd, edm, uvit and DiT.