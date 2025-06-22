import torch
import torch.utils
import torch.utils.data
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import os
import random
from autoencoders.autoencoder import get_autoencoder
from autoencoders.tokenizers import get_tokenizer

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

class LatentMultimodalDataset(torch.utils.data.Dataset):
    """Some Information about MyDataset"""
    def __init__(self, image_folder_path, text_folder_path, tokenizer, vae, prefix='img_', scale_factor=0.18215, 
                res=32, num_channels=4, block_size=77, dataset_size=None, **kwargs):
        super(LatentMultimodalDataset, self).__init__()
        self.tokenizer = tokenizer
        self.vae = vae
        self.scale_factor = scale_factor
        self.decoder_batch_size = 16 
      
        self.vocab_size = self.tokenizer.vocab_size
        self.res = res
        
        self.context_len = block_size
        self.num_channels = num_channels

        self.needs_encoding = False
        self.prefix = prefix
        self.image_folder_path = image_folder_path
        self.text_folder_path = text_folder_path

        self.has_index = False
        if 'index_path' in kwargs:
            self.has_index = True
            self.index = np.load(kwargs['index_path'])
            dataset_size = len(self.index)
        # Count total number of .npy files recursively
        if dataset_size is None:
            self.tot = sum(len([f for f in files if f.endswith('.npy')])
                      for _, _, files in os.walk(image_folder_path))
        else:
            self.tot = dataset_size
        
        
    def __getitem__(self, idx):
        if self.has_index:
            idx = self.index[idx]
        i_str = f'{idx:08d}'
        subdir = i_str[:4]
        file_suffix = f'{self.prefix}{idx}.npy'
        
        image_fname = f'{self.image_folder_path}/{subdir}/{file_suffix}'
        tokens_fname = f'{self.text_folder_path}/{subdir}/{file_suffix}'
        
        image = np.load(image_fname, mmap_mode='r').copy()
        mean, std = np.split(image, 2, axis=0)
        mean = torch.from_numpy(mean)
        std = torch.from_numpy(std)
        image = mean + std * torch.randn_like(mean)
        image = image * self.scale_factor

        tokens_arr = np.load(tokens_fname, mmap_mode='r')
        if tokens_arr.ndim == 1:
            token = torch.from_numpy(tokens_arr.copy())
        else:
            token = torch.from_numpy(tokens_arr[random.randint(0, len(tokens_arr)-1)].copy())
        return image, token

    def __len__(self):
        return self.tot

    @torch.no_grad()
    def decode_labels(self, encoded_labels):
        return self.tokenizer.decode(encoded_labels)
    
    @torch.no_grad() 
    def decode_images(self, encoded_images):
        encoded_images = encoded_images / self.scale_factor
        img = self.vae.to(encoded_images.device).decode(encoded_images)
        img = (img + 1)/2
        img = img.clamp(0,1)
        return img 
    
    @torch.no_grad()
    def decode(self, encoded_images, encoded_labels):
        return self.decode_images(encoded_images), self.decode_labels(encoded_labels)

def get_default_transform(target_size):
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    return transform


def get_dataset_and_encoders(name, dataset_opts):
    # Returns dataset, vae, tokenizer
    if name == 'sam':
        vae = get_autoencoder('stabilityai/sd-vae-ft-mse')
        tokenizer = get_tokenizer(dataset_opts.tokenizer_version, block_size=dataset_opts.block_size)
        return LatentMultimodalDataset(**dataset_opts, vae=vae, tokenizer=tokenizer), vae, tokenizer