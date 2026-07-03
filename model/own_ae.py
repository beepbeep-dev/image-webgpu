"""
Our OWN from-scratch autoencoder — replaces the pretrained TAESD so every
learned component in the Diffusion HQ pipeline is trained by us. Same
generic conv topology class as TAESD (3x stride-2 downsampling to a 4-channel
32x32 latent, mirror decoder with nearest-upsampling) — topology is just
standard conv-net structure; what made TAESD "pretrained" was its weights,
and these are trained from zero on our own scraped photos.
"""
import torch
import torch.nn as nn


def conv(n_in, n_out, **kwargs):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Block(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.conv = nn.Sequential(conv(n, n), nn.ReLU(), conv(n, n), nn.ReLU(), conv(n, n))
        self.fuse = nn.ReLU()

    def forward(self, x):
        return self.fuse(self.conv(x) + x)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


def OwnEncoder(latent_channels=4):
    return nn.Sequential(
        conv(3, 64), Block(64),
        conv(64, 64, stride=2, bias=False), Block(64), Block(64), Block(64),
        conv(64, 64, stride=2, bias=False), Block(64), Block(64), Block(64),
        conv(64, 64, stride=2, bias=False), Block(64), Block(64), Block(64),
        conv(64, latent_channels),
    )


def OwnDecoder(latent_channels=4):
    return nn.Sequential(
        Clamp(), conv(latent_channels, 64), nn.ReLU(),
        Block(64), Block(64), Block(64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64), Block(64), Block(64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64), Block(64), Block(64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64), conv(64, 3),
    )
