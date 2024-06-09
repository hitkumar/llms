import torch
from torch import nn
import torch.nn.functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 32 is the number of groups and channels is the number of channels in the input image
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x: torch.tensor):
        # x: (B, Features, height, width)
        residue = x
        # (B, Features, height, width) -> (B, Features, height, width)
        x = self.groupnorm(x)
        n, c, h, w = x.shape
        # (B, h*w, channels)
        x = x.view((n, c, h*w)).transpose(1, 2)

        # (B, h*w, channels)
        x = self.attention(x)

        # (B, h*w, channels) -> (B, channels, h*w)
        x = x.transpose(-1, -2).view((n, c, h, w))
        x += residue

        return x


class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x):
        # x: (B, in_channels, H, W)
        residue = x
        # (B, in_channels, H, W)
        x = self.groupnorm_1(x)
        x = F.silu(x)
        # (B, out_channels, H, W)
        x = self.conv_1(x)

        # (B, out_channels, H, W)
        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)

class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (B, 4, H/8, W/8) -> latent which is the output of encoder
            nn.Conv2d(4, 4, kernel_size=1, padding=0),
            # (B, 512, H/8, W/8)
            nn.Conv2d(4, 512, kernel_size=3, padding=1),

            VAE_ResidualBlock(512, 512),
            VAE_AttentionBlock(512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # (B, 512, H/4, W/4)
            nn.Upsample(scale_factor=2),
            # (B, 512, H/4, W/4)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),
            VAE_ResidualBlock(512, 512),

            # (B, 512, H/2, W/2)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # (B, 256, H/2, W/2)
            VAE_ResidualBlock(512, 256),
            VAE_ResidualBlock(256, 256),
            VAE_ResidualBlock(256, 256),

            # (B, 256, H, W)
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),

            # (B, 128, H, W)
            VAE_ResidualBlock(256, 128),
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            nn.GroupNorm(32, 128),

            nn.SiLU(),
            # (B, 3, H, W) -> reconstruct the original image
            nn.Conv2d(128, 3, kernel_size=3, padding=1),          
        )
    
    def forward(self, input: torch.tensor):
        # input: (B, 4, H/8, W/8)

        # encoder adds this scaling, reverse this
        input /= 0.18215

        for module in self:
            input = module(input)
        
        # (B, 3, H, W)
        return input
    