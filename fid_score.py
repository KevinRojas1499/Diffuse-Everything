import os

import click
import numpy as np
import scipy
import torch
import torch.distributed as dist
import torchvision.transforms as transforms
from fid_computation.inception import InceptionV3
from PIL import Image
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from torchvision.datasets import CIFAR100

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}




def ensure_process_group_initialized(backend='nccl'):
    try:
        if not dist.is_initialized():
            # Try to initialize only if environment variables are set
            if all(var in os.environ for var in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK']):
                dist.init_process_group(backend=backend)
                print(f"Process group initialized with backend {backend}.")
                return True
            else:
                # Running in non-distributed mode
                return False
        return False
    except Exception as e:
        print(f"Warning: Could not initialize process group: {e}")
        return False

class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, res=256):
        self.root_dir = root_dir
        self.image_paths = self._load_image_paths(root_dir)
        self.res = res

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
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.image_paths[idx]
        if img_path[-4:] == '.npy':
            image = torch.tensor(np.load(img_path))
            return image
        else:
            image = Image.open(img_path).convert('RGB').resize((self.res, self.res))

            return transforms.functional.to_tensor(image)



def get_dataset(path, res):
    if path == 'cifar100':
        return CIFAR100('./saved_datasets/',True,transform=transforms.ToTensor(), download=True)
    else:
        return ImageFolderDataset(path, res=res)
    
@torch.no_grad()
def compute_statistics(dataset, model, batch_size=50, dims=2048, device='cuda', num_workers=8, save_path=''):
    """
    Compute statistics for both distributed and non-distributed scenarios
    """
    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    world_size = dist.get_world_size() if is_distributed else 1
    
    model.eval()
    if rank == 0:
        print(f'Computing FID with {len(dataset)} images')
    
    if is_distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        sampler = None
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size//world_size if is_distributed else batch_size,
        shuffle=False,
        sampler=sampler,
        drop_last=False,
        num_workers=num_workers
    )

    N = torch.tensor([0], device=device)
    mu = 0.
    sigma = 0.
    for batch in tqdm(dataloader):
        if not isinstance(batch, torch.Tensor):
            # For instance if the dataset is (im, labels)
            batch = batch[0]
        batch = batch.to(device)
        features = model(batch)[0].to(torch.float64).reshape(batch.shape[0], dims)
        mu += features.sum(0)
        sigma += features.T @ features
        N+= batch.shape[0]


    if is_distributed:
        dist.all_reduce(mu, op=dist.ReduceOp.SUM)
        dist.all_reduce(sigma, op=dist.ReduceOp.SUM)
        dist.all_reduce(N, op=dist.ReduceOp.SUM)
        dist.barrier()
    mu /= N
    sigma -= mu.ger(mu) * N
    sigma /= (N - 1)

    mu = mu.cpu().numpy()
    sigma = sigma.cpu().numpy()
    if rank == 0:
        np.save(os.path.join(save_path,'fid_stats'), {'mu' : mu, 'sigma' : sigma})
    return  mu, sigma



def calculate_fid_from_inception_stats(mu, sigma, mu_ref, sigma_ref):
    m = np.square(mu - mu_ref).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma, sigma_ref), disp=False)
    fid = m + np.trace(sigma + sigma_ref - s * 2)
    return float(np.real(fid))



@click.command()
@click.option('--path', type=str)
@click.option('--ref_path', type=str)
@click.option('--batch_size', type=int, default=50)
@click.option('--res', type=int, default=256)
@click.option('--num_workers', type=int, default=4)
def cli(path, ref_path, batch_size, num_workers, res):
    calculate_fid_given_paths(path, ref_path, res, batch_size, num_workers)



def calculate_fid_given_paths(path, ref_path, res, batch_size=50, num_workers=4):
    had_to_initialize = ensure_process_group_initialized()
    is_distributed = dist.is_initialized()
    
    if is_distributed:
        device = dist.get_rank() % torch.cuda.device_count()
    else:
        device = 0
    torch.cuda.set_device(device)

    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    if os.path.isdir(os.path.join(path, 'ref')):
        os.makedirs(os.path.join(path, 'ref'),exist_ok=True)
    if path[-4:] == '.npy':
        fid_stats = np.load(path, allow_pickle=True).item()
        m1 = fid_stats['mu']
        s1 = fid_stats['sigma']
    else:
        dataset = get_dataset(path, res)
        m1, s1 = compute_statistics(dataset, model, batch_size,
                                        dims, device, num_workers, path)
    if ref_path[-4:] == '.npy':
        fid_stats = np.load(ref_path, allow_pickle=True).item()
        m2 = fid_stats['mu']
        s2 = fid_stats['sigma']
    else:
        ref_dataset = get_dataset(ref_path, res)
        m2, s2 = compute_statistics(ref_dataset, model, batch_size,
                                        dims, device, num_workers, os.path.join(path,'ref'))
    
    fid_val_tensor = torch.tensor([-1.], device=device)
    if not is_distributed or is_distributed and dist.get_rank() == 0:
        fid_value = calculate_fid_from_inception_stats(m1, s1, m2, s2)
        print(f'We got an FID of {fid_value}')
        fid_val_tensor[0] = fid_value
    
    if is_distributed:
        dist.barrier(device_ids=[device])
        dist.broadcast(fid_val_tensor, src=0)
        dist.barrier()
    
    if had_to_initialize:
        dist.destroy_process_group()
    return fid_val_tensor


if __name__ == '__main__':
    cli()