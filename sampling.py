import os
import json
import textwrap
import click
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import yaml
import PIL
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import utils.graph_lib
from utils.samplers import get_sampling_fn
from models.model_utils import (
    get_model,
    get_preconditioned_model,
)
from utils.datasets import get_dataset_and_encoders
from utils.misc import dotdict
from utils.sde_lib import get_sde
from torchmetrics.functional.multimodal import clip_score
from functools import partial

@click.group()
def sampling_group():
    pass

def initialize_model_and_components(opts):
    # Initialize distributed setup
    dist.init_process_group('nccl')
    world_size = dist.get_world_size()
    
    # Setup device and seed
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.manual_seed(opts.seed * world_size + rank)
    torch.cuda.set_device(device)

    # Load configs and initialize dataset components
    dataset_opts = dotdict(yaml.safe_load(open(opts.dataset_config_path)))
    net_opts = dotdict(yaml.safe_load(open(opts.net_config_path)))
    dataset, vae, tokenizer = get_dataset_and_encoders(opts.dataset,dataset_opts)
    tokenizer = tokenizer.to(device)
    if vae is not None:
        vae = vae.to(device)
    
    # Setup modality-specific components
    vocab_size = dataset.vocab_size
    img_res = dataset.res
    context_len = dataset.context_len
    
    # Initialize graph and SDE
    graph = utils.graph_lib.Absorbing(vocab_size)
    sde = get_sde(opts.sde)
    
    # Initialize and load model
    model = get_model(opts.model, img_res, 
                     vocab_size + 1,
                     context_len, dataset.num_channels, net_opts)
    
    if rank == 0:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)//1e6} M")
    
    if opts.load_checkpoint is not None:
        ckpt = torch.load(os.path.join(opts.load_checkpoint), weights_only=True, map_location=f'cuda:{device}')
        # ckpt = ckpt['ema' if opts.use_ema else 'model']
        model.load_state_dict(ckpt, strict=False)
        dist.barrier(device_ids=[device])
    
    model = get_preconditioned_model(model, sde, graph, tokenizer).to(device)
    model.eval()
    
    return {
        'model': model,
        'dataset': dataset,
        'vae': vae,
        'tokenizer': tokenizer,
        'world_size' : world_size,
        'rank': rank,
        'device': device,
        'context_len': context_len,
        'graph': graph,
        'sde': sde
    }

def plot_samples(path_samples, decoded_im, decoded_label, wrap_width=50):
    for j in range(decoded_im.shape[0]):
        # Create figure with two columns
        fig = plt.figure(figsize=(8, 4))
        
        # Create two subplots with specific width ratios
        gs = plt.GridSpec(1, 2, width_ratios=[1, 1], figure=fig)
        
        # Image subplot
        ax_img = fig.add_subplot(gs[0])
        ax_img.imshow(decoded_im[j].permute(1,2,0).clamp(0,1).cpu().numpy())
        ax_img.axis('off')
        
        # Text subplot
        ax_text = fig.add_subplot(gs[1])
        ax_text.text(0.05, 0.5, textwrap.fill(decoded_label[j], width=wrap_width),
                    wrap=True, verticalalignment='center', fontsize=11)
        ax_text.axis('off')
        
        try:
            plt.tight_layout()
            fig.savefig(os.path.join(path_samples, f'sample_{dist.get_rank()}_{j}.png'),
                       bbox_inches='tight',
                       pad_inches=0)
        except Exception as e:
            print(f'Error {e} happened')
        plt.close(fig)

@sampling_group.command()
@click.option('--dataset',type=click.Choice(['sam']), default='sam')
@click.option('--model',type=click.Choice(['mmdit']), default='mmdit')
@click.option('--dataset_config_path',type=str, default='configs/sam_llava.yaml')
@click.option('--net_config_path',type=str, default='configs/mmdit.yaml')
@click.option('--modality', type=click.Choice(['continuous','discrete','multimodal']), default='multimodal')
@click.option('--sde',type=click.Choice(['vp']), default='vp')
@click.option('--prompt',type=str, default=None)
@click.option('--num_steps', type=int, default=50)
@click.option('--cfg_scale', type=float, default=1.)
@click.option('--batch_size', type=int, default=64)
@click.option('--num_samples', type=int, default=64)
@click.option('--guidance_left', type=float, default=0.)
@click.option('--guidance_right', type=float, default=1.)
@click.option('--guidance_final_time', type=float, default=1.)
@click.option('--dir',type=str)
@click.option('--load_checkpoint',type=str, help='Directory where we can find the desired checkpoints')
@click.option('--plot_with_text', is_flag=True, default=False)
@click.option('--use_ema/--not_use_ema', is_flag=True, default=True)
@click.option('--seed', type=int, default=42)
def sampling(**opts):
    opts = dotdict(opts)
    components = initialize_model_and_components(opts)
    
    path_samples = os.path.join(opts.dir)
    os.makedirs(path_samples, exist_ok=True)
    # Rank and world size
    rank = components['rank']
    world_size = components['world_size']
    device = components['device']

    # Model and dataset
    model = components['model']
    dataset = components['dataset']
    context_len = components['context_len']
    graph = components['graph']
    sde = components['sde']

    
    # Batch size and number of samples
    assert opts.batch_size % world_size == 0, f"Batch size must be divisible by world size. Got {opts.batch_size} with {world_size}"
    batch_size = opts.batch_size//world_size
    num_samples = opts.num_samples//world_size
    
    sampling_fn = get_sampling_fn(model, sde, graph, device)
    model.eval()
    with torch.no_grad():
        num_batches = num_samples//batch_size
        for j in range(num_batches):
            encoded_im, encoded_text = sampling_fn(
                    cont_shape=(batch_size,dataset.num_channels,dataset.res,dataset.res),
                    disc_shape=(batch_size,context_len),
                    steps=opts.num_steps,
                    cfg_scale=opts.cfg_scale, 
                    mode=opts.modality,
                    guidance_left=opts.guidance_left,
                    guidance_right=opts.guidance_right,
                    guidance_final_time=opts.guidance_final_time,
                    return_traj=False)
        
            decoded_im, decoded_text = dataset.decode(encoded_im, encoded_text)

            folder = os.path.join(opts.dir, f'{rank}/{j}')
            os.makedirs(folder, exist_ok=True)
            images_np = (decoded_im * 255).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1)# .cpu().numpy()

            images_np = images_np.cpu().numpy()

            if opts.plot_with_text:
                plot_samples(folder, decoded_im, decoded_text)
            else:
                for i in range(images_np.shape[0]):
                    PIL.Image.fromarray(images_np[i], 'RGB').save(os.path.join(folder, f'{i}.png'))
        
    dist.barrier(device_ids=[device])
    dist.destroy_process_group()

def process_prompts(file):
    prompts = None
    with open(file, 'r') as file:
        data = json.load(file)  # Load the JSON data into a Python dictionary
        prompts = data['text']
        return prompts

@sampling_group.command()
@click.option('--dataset',type=click.Choice(['sam']), default='sam')
@click.option('--modality', type=click.Choice(['continuous','discrete']), default='continuous')
@click.option('--model',type=click.Choice(['mmdit']), default='mmdit')
@click.option('--dataset_config_path',type=str, default='configs/sam_llava.yaml')
@click.option('--net_config_path',type=str, default='configs/mmdit.yaml')
@click.option('--sde',type=click.Choice(['vp']), default='vp')
@click.option('--num_steps', type=int, default=50)
@click.option('--cfg_scale', type=float, default=1.)
@click.option('--guidance_left', type=float, default=0.)
@click.option('--guidance_right', type=float, default=1.)
@click.option('--guidance_final_time', type=float, default=1.)
@click.option('--seed', type=int, default=42)
@click.option('--batch_size', type=int, default=512)
@click.option('--num_samples', type=int, default=None)
@click.option('--from_snapshot/--from_ckpt', type=bool, is_flag=True, default=True)
@click.option('--dir',type=str)
@click.option('--load_checkpoint',type=str, help='Directory where we can find the desired checkpoints')
@click.option('--plot_with_text', type=bool, is_flag=True, default=False)
@click.option('--use_ema/--not_use_ema', is_flag=True, default=True)
@click.option('--repeat_text', type=bool, is_flag=True, default=True)
@click.option('--limit_context_len', type=int, default=None)
@click.option('--num_batches', type=int, default=1)
def sample_dataset_conditional(**opts):
    opts = dotdict(opts)
    components = initialize_model_and_components(opts)
    # Rank and world size
    rank = components['rank']
    world_size = components['world_size']
    device = components['device']

    # Model and dataset
    model = components['model']
    dataset = components['dataset']
    context_len = components['context_len']
    graph = components['graph']
    sde = components['sde']

    
    # Batch size and number of samples
    assert opts.batch_size % world_size == 0, f"Batch size must be divisible by world size. Got {opts.batch_size} with {world_size}"
    batch_size = opts.batch_size//world_size
    
    model.eval()
    rank = components['rank']
    device = components['device']
    context_len = components['context_len'] 
    if opts.limit_context_len is not None:
        context_len = min(context_len, opts.limit_context_len)
    graph = components['graph']
    sde = components['sde']
    
    model.eval()

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=opts.seed
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, sampler=sampler)

    sampling_fn = get_sampling_fn(model, sde, graph, device)
    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-large-patch14")
    avg_clip_score = 0
    k = 0
    for batch in tqdm(dataloader, leave=False):
        if k > opts.num_batches:
            break
        images, texts = batch[0], batch[1]
        images = images.to(device)
        texts = texts.to(device)[:,:context_len]
        n_samples = images.shape[0]
        dest_folder = os.path.join(opts.dir, f'images/{rank}_{k}')
        if rank == 0:
            os.makedirs(dest_folder,exist_ok=True)
        encoded_im, encoded_text = sampling_fn(
                cont_shape=(n_samples,dataset.num_channels,dataset.res,dataset.res), 
                disc_shape=(n_samples,context_len),
                steps=opts.num_steps,
                cond=images if opts.modality == 'discrete' else texts,
                cfg_scale=opts.cfg_scale, 
                mode=opts.modality,
                guidance_left=opts.guidance_left,
                guidance_right=opts.guidance_right,
                guidance_final_time=opts.guidance_final_time,
                return_traj=False)
        
        decoded_im, decoded_text = dataset.decode(encoded_im,encoded_text)
        
        images_np = (decoded_im * 255 ).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        folder = os.path.join(os.path.join(dest_folder,f'{rank}_{k}'))
        os.makedirs(folder, exist_ok=True)
        if opts.plot_with_text:
            plot_samples(folder, decoded_im, decoded_text)
        # else:
        for i in range(images_np.shape[0]):
            PIL.Image.fromarray(images_np[i], 'RGB').save(os.path.join(folder, f'{i}.png'))
            with open(os.path.join(folder, f'{i}.txt'), 'w') as f:
                f.write(decoded_text[i])
        k += 1

        with torch.no_grad():
            avg_clip_score += clip_score_fn(decoded_im, decoded_text)
            cur_avg_clip_score = avg_clip_score/k
            dist.all_reduce(cur_avg_clip_score, dist.reduce_op.AVG)
            if rank == 0:
                print(f'Iteration {k} Average Clip Score {cur_avg_clip_score}')
    dist.barrier(device_ids=[device])
    dist.destroy_process_group()


@sampling_group.command()
@click.option('--dataset',type=click.Choice(['sam']), default='sam')
@click.option('--model',type=click.Choice(['mmdit']), default='mmdit')
@click.option('--dataset_config_path',type=str, default='configs/sam_llava.yaml')
@click.option('--net_config_path',type=str, default='configs/mmdit.yaml')
@click.option('--sde',type=click.Choice(['vp']), default='vp')
@click.option('--prompt',type=str, default='data.json')
@click.option('--num_steps', type=int, default=50)
@click.option('--cfg_scale', type=float, default=1.)
@click.option('--guidance_final_time', type=float, default=1.)
@click.option('--guidance_left', type=float, default=0.)
@click.option('--guidance_right', type=float, default=1.)
@click.option('--seed', type=int, default=42)
@click.option('--batch_size', type=int, default=1024)
@click.option('--num_samples', type=int, default=None)
@click.option('--from_snapshot/--from_ckpt', type=bool, is_flag=True, default=True)
@click.option('--dir',type=str)
@click.option('--load_checkpoint',type=str, help='Directory where we can find the desired checkpoints')
@click.option('--plot_with_text', type=bool, is_flag=True, default=False)
@click.option('--repeat_text/--no_repeat_text', type=bool, is_flag=True, default=True)
@click.option('--limit_context_len', type=int, default=None)
@click.option('--use_ema', is_flag=True, default=True)
def sampling_conditional(**opts):
    opts = dotdict(opts)
    components = initialize_model_and_components(opts)
    # Rank and world size
    rank = components['rank']
    world_size = components['world_size']
    device = components['device']

    # Model and dataset
    model = components['model']
    dataset = components['dataset']
    context_len = components['context_len']
    graph = components['graph']
    sde = components['sde']

    
    # Batch size and number of samples
    assert opts.batch_size % world_size == 0, f"Batch size must be divisible by world size. Got {opts.batch_size} with {world_size}"
    batch_size = opts.batch_size//world_size
    
    model.eval()
    rank = components['rank']
    device = components['device']
    tokenizer = components['tokenizer']
    context_len = components['context_len'] 
    if opts.limit_context_len is not None:
        context_len = max(context_len, opts.limit_context_len)
    graph = components['graph']
    sde = components['sde']
    
    model.eval()

    prompt = process_prompts(opts.prompt)
    num_samples = opts.num_samples if opts.num_samples is not None else len(prompt)
    batches = num_samples//(batch_size * world_size) + (0 if num_samples % batch_size == 0 else 1)
    if prompt is not None:
        rank_prompts = prompt[rank :: world_size]
    
    sampling_fn = get_sampling_fn(model, sde, graph, device)

    for batch in tqdm(range(batches), leave=False):
        cur_prompt = rank_prompts[batch * batch_size : min((batch + 1) * batch_size, len(rank_prompts))]
        n_samples = len(cur_prompt)
        # Folders are not perfect but its fine
        dest_folder = os.path.join(opts.dir, f'images/{rank}/{batch}')
        if rank == 0:
            os.makedirs(dest_folder,exist_ok=True)
        tokens = tokenizer.tokenize(cur_prompt, max_length=context_len, repeat=opts.repeat_text).to(device)
        encoded_im, encoded_text = sampling_fn(
                cont_shape=(n_samples,dataset.num_channels,dataset.res,dataset.res), 
                disc_shape=(n_samples,context_len),
                steps=opts.num_steps,
                cond=tokens,
                cfg_scale=opts.cfg_scale, 
                mode='continuous',
                guidance_left=opts.guidance_left,
                guidance_right=opts.guidance_right,
                guidance_final_time=opts.guidance_final_time)
        
        decoded_im, decoded_text = dataset.decode(encoded_im,encoded_text)
        
        images_np = (decoded_im * 255 ).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        folder = os.path.join(os.path.join(dest_folder,f'{rank}_{batch}'))
        os.makedirs(folder, exist_ok=True)
        if opts.plot_with_text:
            plot_samples(folder, decoded_im, decoded_text)
        else:
            for i in range(images_np.shape[0]):
                PIL.Image.fromarray(images_np[i], 'RGB').save(os.path.join(folder, f'{i}.png'))

    dist.barrier(device_ids=[device])
    dist.destroy_process_group()

        

if __name__ == '__main__':
    sampling_group()