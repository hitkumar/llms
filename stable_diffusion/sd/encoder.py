import torch
from torch import nn
import torch.nn.functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (B, 3, H, W) -> (B, 128, H, W)
            nn.Conv2d(3, 128, kernel_size=3, padding=1),

            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),

            # (B, 128, H, W) -> (B, 128, H/2, W/2)
            nn.Conv2d(128, 128, kernel_size=3, padding=0, stride=2),
            # (B, 128, H/2, W/2) -> (B, 256, H/2, W/2)
            VAE_ResidualBlock(128, 256),
            VAE_ResidualBlock(256, 256),

            # (B, 256, H/2, W/2) -> (B, 256, H/4, W/4)
            nn.Conv2d(256, 256, kernel_size=3, padding=0, stride=2),
            # (B, 256, H/4, W/4) -> (B, 512, H/4, W/4)
            VAE_ResidualBlock(256, 512),
            VAE_ResidualBlock(512, 512),
            
            # (B, 512, H/4, W/4) -> (B, 512, H/8, W/8)
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),

            nn.GroupNorm(32, 512),
            nn.SiLU(),
            
            # (B, 8, H/8, W/8)
            nn.Conv2d(512, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 8, kernel_size=1, padding=0),
        )
    
    def forward(self, x, noise):
        # x: (B, 3, H, w), noise: (B, 4, H/8, W/8)
        
        for module in self:
            if getattr(module, 'stride', None) == (2, 2):
                # Assymetric padding during downsampling
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
        
        # x: (B, 8, H/8, H/8)
        # Splits x into two tensors of dim (B, 4, H/8, W/8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        log_variance = torch.clamp(log_variance, -30, 20)
        variance = log_variance.exp()
        stddev = variance.sqrt()

        # Transform noise into N(mean, stddev)
        x = mean + stddev * noise
        x += 0.18215
        return x
