import os
import json
import textwrap

import PIL.Image
import click
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from copy import deepcopy
from collections import OrderedDict
import utils.graph_lib
from utils.samplers import get_sampling_fn
import wandb
from models.model_utils import get_model, get_preconditioned_model
from utils.datasets import get_dataset_and_encoders
from utils.losses import get_loss
from utils.misc import dotdict
from utils.optimizers import WarmUpScheduler
from utils.sde_lib import get_sde
from fid_score import calculate_fid_given_paths
from torchmetrics.functional.multimodal import clip_score
from functools import partial

# This makes training on A100s faster
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def init_wandb(opts, dataset_opts):
    wandb.init(
        # set the wandb project where this run will be logged
        project='multimodal-diffusion',
        name= f'{opts.model}-{opts.dataset}-{opts.modality}',
        tags= ['training',opts.dataset],
        # # track hyperparameters and run metadata
        config=opts
    )

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
        
def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


@click.command()
@click.option('--dataset',type=click.Choice(['sam']), default='sam')
@click.option('--model',type=click.Choice(['mmdit']), default='mmdit')
@click.option('--modality', type=click.Choice(['continuous','discrete','multimodal']), default='multimodal')
@click.option('--use_all_times/--not_use_all_times', is_flag=True, default=True)
@click.option('--freeze_joint', is_flag=True, default=False)
@click.option('--freeze_image', is_flag=True, default=False)
@click.option('--freeze_text', is_flag=True, default=False)
@click.option('--sde',type=click.Choice(['vp']), default='vp')
@click.option('--lr', type=float, default=2e-4)
@click.option('--ema_beta', type=float, default=.9999)
@click.option('--batch_size', type=int, default=256)
@click.option('--log_rate',type=int,default=10000)
@click.option('--eval_rate',type=int,default=100000)
@click.option('--eval_num',type=int,default=10000)
@click.option('--eval_batch_size',type=int,default=512)
@click.option('--num_iters',type=int,default=600000)
@click.option('--warmup_iters',type=int,default=2500)
@click.option('--num_workers',type=int,default=6)
@click.option('--seed',type=int,default=42)
@click.option('--dir',type=str)
@click.option('--net_config_path',type=str, default='configs/mmdit.yaml')
@click.option('--dataset_config_path',type=str, default='configs/sam_llava.yaml')
@click.option('--load_checkpoint',type=str, help='Directory where we can find the desired checkpoints')
@click.option('--enable_wandb', is_flag=True, default=False)
@click.option('--force_finite', is_flag=True, default=False)
@click.option('--load_from_ema', is_flag=True, default=False)
def training(**opts):
    opts = dotdict(opts)
    batch_size = opts.batch_size
    
    dist.init_process_group('nccl')
    world_size = dist.get_world_size()
    assert batch_size % world_size == 0, f'Batch size {batch_size} must be divisible by world size {world_size}.'
    batch_size = batch_size // world_size
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = opts.seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    
    net_opts = dotdict(yaml.safe_load(open(opts.net_config_path)))
    dataset_opts = dotdict(yaml.safe_load(open(opts.dataset_config_path)))
    if rank == 0:
        print(opts)
        print(net_opts)
        print(dataset_opts)
    wandb_enabled = opts.enable_wandb and rank == 0 # We only want to log once
    if wandb_enabled:
        init_wandb(opts, dataset_opts)
        wandb.config.update(net_opts)
    
    dataset, vae, tokenizer = get_dataset_and_encoders(opts.dataset, dataset_opts) 
    if tokenizer is not None:
        tokenizer = tokenizer.to(device)
    # vae = vae.to(device)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=opts.seed
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, persistent_workers=True,
                sampler=sampler, drop_last=True, num_workers=opts.num_workers, pin_memory=False)
       
    is_discrete = (opts.modality == 'discrete')
    is_multimodal = (opts.modality == 'multimodal')
    vocab_size = dataset.vocab_size 
    img_res = dataset.res 
    context_len = dataset.context_len 
    graph = utils.graph_lib.Absorbing(vocab_size)
    sde =  get_sde(opts.sde)

    model = get_model(opts.model,img_res,vocab_size + 1, context_len, dataset.num_channels,net_opts)
    ema = deepcopy(model)
    if opts.freeze_joint:
        model.freeze_joint()
    if opts.freeze_image:
        model.freeze_image()
    if opts.freeze_text:
        model.freeze_text()
    model = get_preconditioned_model(model, sde,graph, mode=opts.modality).to(device)
    ema = get_preconditioned_model(ema, sde, graph, mode=opts.modality).to(device)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),lr=opts.lr,betas=[0.9, 0.9],weight_decay=0.03)
    scheduler = WarmUpScheduler(opt, opts.warmup_iters)
    scaler = torch.amp.GradScaler(device)
    
    
    start_iter = 0
    dist.barrier(device_ids=[device])
    if opts.load_checkpoint is not None:
        print(f'Loading checkpoint from {opts.load_checkpoint} in rank {rank}')
        snapshot = torch.load(os.path.join(opts.load_checkpoint), weights_only=True, map_location=f'cuda:{device}')
        if opts.load_from_ema:
            model.net.load_state_dict(snapshot['ema'],strict=False)
        else:
            model.net.load_state_dict(snapshot['model'],strict=False)
        ema.net.load_state_dict(snapshot['ema'],strict=False)
        opt.load_state_dict(snapshot['optimizer'])
        scheduler.load_state_dict(snapshot['scheduler'])
        start_iter = scheduler.last_epoch
    if start_iter == 0:
        # EMA is initialized with model weights
        update_ema(ema, model, decay=0)  
    requires_grad(ema, False)

    dist.barrier(device_ids=[device])
    
    ema.eval()
    model.train()
    
    model = DDP(model)
    
    if rank == 0:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)//1e6} M")
    
    if rank == 0:
        os.makedirs(opts.dir, exist_ok=True)

    num_iters = opts.num_iters

    loss_fn = get_loss(opts.modality, sde, graph, use_all_times=opts.use_all_times)
    sampling_fn = get_sampling_fn(model, sde, graph, device)
    sampling_fn_ema = get_sampling_fn(ema, sde, graph, device)
    
    # torch.autograd.set_detect_anomaly(True)
    training_iter = start_iter
    iters_per_epoch = len(dataset)//(batch_size * world_size) + 1
    log_rate = opts.log_rate
    epochs = num_iters//iters_per_epoch + 1
    for epoch in range(epochs):
        pbar = tqdm(dataloader,total=iters_per_epoch,leave=False) if rank == 0 else dataloader
        for data_ in pbar:
            if training_iter > num_iters:
                break
            
            imgs, labels = data_[0], data_[1]
            imgs = imgs.to(device=device, dtype=torch.float)
            labels = labels.to(device=device)
            opt.zero_grad()
            
            with torch.amp.autocast('cuda', torch.bfloat16):
                loss_dir = loss_fn(imgs, labels, model)
                if is_multimodal:
                    loss = loss_dir['cont_loss'] + loss_dir['disc_loss'] 
                elif not is_discrete:
                    loss = loss_dir['cont_loss']
                else:
                    loss = loss_dir['disc_loss']

            scaler.scale(loss).backward()
            scaler.unscale_(opt)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            
            if opts.force_finite:
                for param in model.parameters():
                    if param.grad is not None:
                        torch.nan_to_num(param.grad, nan=0, posinf=0, neginf=0, out=param.grad)
            
            scaler.step(opt)
            scaler.update()
            # opt.step()
            update_ema(ema, model.module, decay=opts.ema_beta)
            scheduler.step()
            
            training_iter += 1
            
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            
            log_dir = {'iter': training_iter, 'loss': loss}
            for key, item in loss_dir.items():
                dist.all_reduce(item, op=dist.ReduceOp.AVG)
                log_dir[key] = item

            if rank == 0:
                if is_multimodal:
                    pbar.set_description(f'Epoch {epoch}/{epochs} --- Iter {training_iter} --- Loss : {loss :6.2f} - Cont : {loss_dir["cont_loss"] : 6.2f} - Disc : {loss_dir["disc_loss"] : 6.2f}')
                elif is_discrete:
                    pbar.set_description(f'Epoch {epoch}/{epochs} --- Iter {training_iter} --- Loss : {loss :6.2f}')
                else:
                    pbar.set_description(f'Epoch {epoch}/{epochs} --- Iter {training_iter} --- Loss : {loss :6.2f}')
                    
            if wandb_enabled:
                wandb.log(log_dir)
                    
            dist.barrier(device_ids=[device])
            # Evaluate sample accuracy
            if training_iter%log_rate == 0 or training_iter == num_iters:
                path = os.path.join(opts.dir, f'itr_{training_iter}/')
                path_samples = os.path.join(path,'samples/')
                path_samples_ema = os.path.join(path,'samples_ema/')
                os.makedirs(path_samples,exist_ok=True)
                os.makedirs(path_samples_ema,exist_ok=True)
                if rank == 0:
                    save_ckpt(model, ema, opt, scheduler, os.path.join(path, 'snapshot.pt'))
                model.eval()
                dist.barrier(device_ids=[device])
                
                generate_qualitative_samples(sampling_fn, opts.modality, wandb_enabled, dataset, imgs, labels, path_samples)
                generate_qualitative_samples(sampling_fn_ema, opts.modality, wandb_enabled, dataset, imgs, labels, path_samples_ema, ema=True)
                model.train()
                        
            
            with torch.no_grad():
                if training_iter > 0 and training_iter%opts.eval_rate == 0:
                    model.eval()
                    save_path = os.path.join(opts.dir, f'fid_{training_iter}/')
                    os.makedirs(save_path, exist_ok=True)
                    if rank == 0:
                        save_ckpt(model, ema, opt, scheduler, os.path.join(save_path, f'{training_iter}_snapshot.pt'))

                    if opts.dataset in ['sam','coco','cub-200']:
                        samp_batch_size = opts.eval_batch_size
                        n_samples = samp_batch_size//world_size
                        n_eval_iters = (opts.eval_num + samp_batch_size - 1) // samp_batch_size
                        
                        if not is_multimodal: 
                            file = 'data.json'
                            with open(file, 'r') as file:
                                prompt = json.load(file)['text']  # Load the JSON data into a Python dictionary

                            rank_prompts = prompt[dist.get_rank() :: dist.get_world_size()]
                        cont_shape  = (n_samples,dataset.num_channels,dataset.res,dataset.res)
                        bar = tqdm(range(n_eval_iters),leave=False) if rank == 0 else range(n_eval_iters)
                        clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-large-patch14")
                        avg_clip_score = 0
                        for batch in bar:
                            if is_multimodal:
                                encoded_im, encoded_text = sampling_fn_ema(cont_shape=(n_samples,dataset.num_channels,dataset.res,dataset.res), 
                                                                       disc_shape=(n_samples,context_len), steps=50,
                                                                       mode='multimodal', cond=None)
                                decoded_im, decoded_label = dataset.decode(encoded_im,encoded_text)
                            else: 
                                cur_prompt = rank_prompts[batch * n_samples : min((batch + 1) * n_samples, len(rank_prompts))]
                                n_samples = len(cur_prompt)
                                cont_shape  = (n_samples,dataset.num_channels,dataset.res,dataset.res)
                                tokens = tokenizer.tokenize(cur_prompt).to(device)
                                encoded_im, encoded_text = sampling_fn_ema(cont_shape=cont_shape, disc_shape=(n_samples,context_len), steps=50,
                                                         mode='continuous', cond=tokens, device=device)
                                decoded_im, decoded_label = dataset.decode(encoded_im,encoded_text)


                        
                            folder = os.path.join(save_path, f'{rank}/{batch}')
                            os.makedirs(folder, exist_ok=True)
                            images_np = (decoded_im * 255).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
                            avg_clip_score += clip_score_fn(images_np, decoded_label)
                            images_np = images_np.cpu().numpy()

                            for i in range(n_samples):
                                PIL.Image.fromarray(images_np[i], 'RGB').save(os.path.join(folder, f'{i}.png'))
                        
                        dist.barrier(device_ids=[device])
                        avg_clip_score /= n_eval_iters
                        dist.all_reduce(avg_clip_score, dist.reduce_op.AVG)
                        # THe *8 is because the VAE downsamples the image by 8x
                        fid_val = calculate_fid_given_paths(path=save_path, ref_path=dataset_opts.fid_ref_stats, res=img_res * 8, batch_size=samp_batch_size)
                        if rank == 0:
                            print('We got an FID of ', fid_val)
                            print('We got clip score of ', avg_clip_score)
                        if wandb_enabled:
                            wandb.log({'fid': fid_val, 
                                    'clip-score-big': avg_clip_score,
                                    'fid_training_iter' : training_iter
                                    })
                
            dist.barrier(device_ids=[device])                            
            model.train()
    if rank == 0:
        save_ckpt(model, ema, opt, scheduler, os.path.join(opts.dir, 'final_checkpoint.pt'))
    dist.barrier(device_ids=[device])
    if wandb_enabled:
        wandb.finish()
    dist.destroy_process_group()

@torch.no_grad()
def generate_qualitative_samples(sampling_fn, mode, wandb_enabled, dataset, images, labels, path_samples, ema=False):
    clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-large-patch14")
    context_len = labels.shape[-1]
    n_samples = 16 
    cond=None
    if mode == 'continuous':
        cond = labels[:n_samples]
    elif mode == 'discrete':
        cond = images[:n_samples]

    encoded_im, encoded_text = sampling_fn(cont_shape=(n_samples,dataset.num_channels,dataset.res,dataset.res), 
                                           disc_shape=(n_samples,context_len), steps=100,
                                           mode=mode, cond=cond)
                
    decoded_im, decoded_label = dataset.decode(encoded_im,encoded_text)
    
    images_np = (decoded_im * 255).clip(0, 255).to(torch.uint8).permute(0, 2, 3, 1)
    avg_clip_score = clip_score_fn(images_np, decoded_label)
                
    dist.all_reduce(avg_clip_score, dist.ReduceOp.AVG)
    if dist.get_rank() == 0:
        print(f'Avg clip score : {avg_clip_score}')
    if wandb_enabled:
        wandb.log({
            f'clip-score{'-ema' if ema else ''}' : avg_clip_score,
        })
    plot_samples(f'figures{'-ema' if ema else ''}', wandb_enabled, path_samples, decoded_im, decoded_label)

def save_ckpt(model, ema, opt, scheduler, path):
    snapshot = {
                    'model': model.module.net.state_dict(),
                    'ema': ema.net.state_dict(),
                    'optimizer': opt.state_dict(),
                    'scheduler': scheduler.state_dict()
                }
    torch.save(snapshot,path)

def plot_samples(log_variable, wandb_enabled, path_samples, decoded_im, decoded_label, wrap_width=50):
    for j in range(decoded_im.shape[0]):
        # Create figure with two columns
        fig = plt.figure(figsize=(10, 5))
        
        # Create two subplots with specific width ratios
        gs = plt.GridSpec(1, 2, width_ratios=[1, 1], figure=fig)
        
        # Image subplot
        ax_img = fig.add_subplot(gs[0])
        ax_img.imshow(decoded_im[j].permute(1,2,0).clamp(0,1).cpu().numpy())
        ax_img.axis('off')
        
        # Text subplot
        ax_text = fig.add_subplot(gs[1])
        ax_text.text(0.05, 0.5, textwrap.fill(decoded_label[j], width=wrap_width),
                    wrap=True, verticalalignment='center')
        ax_text.axis('off')
        
        try:
            plt.tight_layout()
            fig.savefig(os.path.join(path_samples, f'sample_{dist.get_rank()}_{j}.png'), 
                       bbox_inches='tight', 
                       pad_inches=0.5)
            if wandb_enabled:
                wandb.log({log_variable: fig})
        except Exception as e:
            print(f'Error {e} happened')
        plt.close(fig)

        
if __name__ == '__main__':
    training()