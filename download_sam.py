import os
import tarfile
import click
from datasets import load_dataset
from autoencoders.autoencoder import get_autoencoder
from autoencoders.tokenizers import get_tokenizer
from torch.utils.data import DistributedSampler
import torchvision.transforms as transforms
import torch.distributed as dist
import torch
from PIL import Image
import wget
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
import shutil
from multiprocessing import Pool
from utils.datasets import center_crop_arr
import warnings

# Suppress the specific PIL warning about transparency
warnings.filterwarnings('ignore', category=UserWarning, module='PIL.Image')

class ImageFolderDataset(Dataset):
    """Custom Dataset for loading images from a folder structure.
    
    Args:
        root_dir (str): Root directory path containing images
        transform (callable, optional): Optional transform to be applied to images
    """
    def __init__(self, root_dir, transform):
        self.transform = transform
        self.image_paths = self._load_image_paths(root_dir)
        print(f'Loaded {len(self.image_paths)} images')
        
        if len(self.image_paths) == 0:
            raise RuntimeError(f"Found 0 images in {root_dir}")
            
    def _load_image_paths(self, root_dir):
        image_paths = []
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.lower().endswith(valid_extensions):
                    image_paths.append(os.path.join(dirpath, filename))
        
        return image_paths

    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = img_path.split('/')[-1]
        img = Image.open(img_path)
        img = img.convert('RGB')
        img_tensor = self.transform(img)
        return img_tensor, img_name
            

def download_images(url, output_path):
    try:
        wget.download(url.strip(), output_path)
        return True
    except Exception as e:
        print(f"\nError downloading {url}: {str(e)}")
        return False

def save_token(args):
    key, token, output_folder = args
    num = int(key.split('_')[-1])
    num_str = f'{num:08d}'
    file_name = f'sa_{num}.npy'
    folder = os.path.join(output_folder, f'{num_str[:4]}')
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, file_name), token)

@click.group()
def cli():
    pass

@cli.command()
@click.option('--download_links', type=str, default='cc12m_links.txt')
@click.option('--output_folder', type=str, default='/network/rit/dgx/dgx_Yelab/kevin_rojas/datasets/latent_sam')
@click.option('--target_size', type=int, default=256)
@click.option('--batch_size', type=int, default=256)
@click.option('--num_workers', type=int, default=6)
def main(download_links, output_folder, target_size, batch_size, num_workers):
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    temp_folder = os.path.join(output_folder, f'temp_{rank}')
    os.makedirs(temp_folder, exist_ok=True)
    device = rank % torch.cuda.device_count()
    print(f'Starting rank {rank} on device {device} with world size {world_size}')
    autoencoder = get_autoencoder('stabilityai/sd-vae-ft-mse')
    autoencoder.to(device)
    with open(download_links, 'r') as f:
        lines = f.readlines()
    rank_lines = lines[rank::world_size]
    for line in rank_lines:
        file_name, url = line.strip().split()
        cur_folder = os.path.join(temp_folder, f'cur_folder_{rank}')
        path_to_file = os.path.join(cur_folder, file_name)
        os.makedirs(cur_folder, exist_ok=True)
        download_images(url, path_to_file)

        total_size = os.path.getsize(path_to_file)
        with tarfile.open(path_to_file, 'r') as tar:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Extracting") as pbar:
                for member in tar:
                    tar.extract(member, cur_folder)
                    pbar.update(member.size)


        transform = transforms.Compose([
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, target_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
        dataset = ImageFolderDataset(cur_folder, transform)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        for images, file_names in tqdm(dataloader):
            images = images.to(device)
            means, stds = autoencoder.encode_mean_std(images)
            latent_rep = torch.cat([means, stds], dim=1)
            for name, latent in zip(file_names, latent_rep):
                latent_np = latent.cpu().numpy()
                img_num = int(name.split('_')[-1].split('.')[0])
                img_str = f'{img_num:08d}'
                dest_folder  = os.path.join(output_folder, f'images/{img_str[:4]}/')
                os.makedirs(dest_folder, exist_ok=True)
                np.save(os.path.join(dest_folder, f'sa_{img_num}.npy'), latent_np)

        
        shutil.rmtree(cur_folder)
        print(f'Finished processing {line} in rank {rank}')


@cli.command()
@click.option('--tokenizer', type=click.Choice(['clip']), default='clip')
@click.option('--block_size', type=int, default=120)
@click.option('--output_folder', type=str, default='/network/rit/lab/Yelab/kevin-back/kevin_rojas/datasets/tokens')
def process_captions(tokenizer, block_size, output_folder):
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = rank % torch.cuda.device_count()
    print(f'Starting rank {rank} on device {device} with world size {world_size}')
    tokenizer = get_tokenizer(tokenizer, block_size=block_size)
    ds = load_dataset("PixArt-alpha/SAM-LLaVA-Captions10M")
    ds = ds['train']
    ds.set_format(type="torch", columns=['__key__', 'txt'])

    sampler = DistributedSampler(
        ds,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    dataloader = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False, num_workers=6, sampler=sampler)
    for elem in tqdm(dataloader):
        tokens = tokenizer.tokenize(elem['txt'])
        # Create args list for parallel processing
        save_args = [(key, token.numpy(), output_folder) 
                    for key, token in zip(elem['__key__'], tokens)]
        
        with Pool(processes=8) as pool:
            pool.map(save_token, save_args)


def is_path(path):
    return os.path.exists(path) and os.path.isfile(path)

def process_folder_chunk(inputs):
    chunk, folder, prefix = inputs
    chunk_indexes = []
    for i in tqdm(chunk):
        i_str = f'{i:04d}'
        im_path = os.path.join(folder, 'images', i_str)
        txt_path = os.path.join(folder, 'tokens', i_str)
        if os.path.exists(im_path):
            indexes = [int(f.split('_')[-1].split('.')[0]) for f in os.listdir(im_path) if is_path(os.path.join(im_path, f))]
            indexes = [index for index in indexes if is_path(os.path.join(txt_path, f'{prefix}{index}.npy'))]
            indexes = sorted(indexes)
            chunk_indexes.extend(indexes)
    return chunk_indexes

@cli.command()
@click.option('--prefix', type=str, default='')
@click.option('--folder', type=str, default='/network/rit/lab/Yelab/kevin-back/kevin_rojas/datasets/')
@click.option('--num_workers', type=int, default=8)
def process_indexes(folder, num_workers, prefix):
    total_folders = 1118
    range_ = list(range(total_folders))
    chunks = [range_[i::num_workers] for i in range(num_workers)]
    
    # Create input tuples with both chunk and folder
    chunk_inputs = [(chunk, folder, prefix) for chunk in chunks]
    
    # Process chunks in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_folder_chunk, chunk_inputs)
    
    # Combine results
    all_indexes = []
    for chunk_result in results:
        all_indexes.extend(chunk_result)
    all_indexes = sorted(all_indexes)
    
    print(f"Total indexes: {len(all_indexes)}")
    np.save(os.path.join(folder, 'all_indexes.npy'), all_indexes)
    
    
if __name__ == "__main__":
    cli()