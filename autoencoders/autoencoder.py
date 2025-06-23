import torch
import torch.nn as nn
from diffusers import AutoencoderKL


class Autoencoder(nn.Module):
    def __init__(self, name, max_batch_size=64) -> None:
        super().__init__()
        self.vae = AutoencoderKL.from_pretrained(name).requires_grad_(False)
        self.batch_size = max_batch_size

    @torch.no_grad() 
    def _encode_batch_mean_std(self, image):
        dist = self.vae.encode(image)['latent_dist']
        return dist.mean, dist.std
    
    @torch.no_grad()
    def encode_mean_std(self, encoded_images):
        dists = [self._encode_batch_mean_std(im) for im in encoded_images.split(self.batch_size)]
        means = torch.cat([dist[0] for dist in dists])
        stds = torch.cat([dist[1] for dist in dists])
        return means, stds

    @torch.no_grad() 
    def _encode_batch(self, image):
        return self.vae.encode(image)['latent_dist'].sample()
    
    @torch.no_grad()
    def encode(self, encoded_images):
        return torch.cat([self._encode_batch(im) for im in encoded_images.split(self.batch_size)])
    
    
    @torch.no_grad() 
    def _decode_batch(self, encoded_images):
        return self.vae.decode(encoded_images)['sample']
    
    @torch.no_grad()
    def decode(self, encoded_images):
        return torch.cat([self._decode_batch(im) for im in encoded_images.split(self.batch_size)])
    
def get_autoencoder(name='stabilityai/sd-vae-ft-mse'):
    return Autoencoder(name)
    