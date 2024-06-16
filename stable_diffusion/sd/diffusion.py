import torch
from torch import nn
import torch.nn.functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.linear_1 = nn.Linear(n_embed, 4 * n_embed)
        self.linear_2 = nn.Linear(4 * n_embed, 4 * n_embed)
    
    def forward(self, x):
        # x : (1, 320)
        x = F.silu(self.linear_1(x))
        # (1, 1280)
        return self.linear_2(x)


class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(32, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, feature, time):
        # feature: (B, C, H, W)
        # time: (1, 1280)
        residue = feature

        # (B, in_channels, H, W)
        feature = self.groupnorm_feature(feature)
        feature = F.silu(feature)
        # (B, out_channels, H, W)
        feature = self.conv_feature(feature)

        # (1, 1280)
        time = F.silu(time)
        # (1, 1280) -> (1. out_channels)
        time = self.linear_time(time)

        # Add height and width dimension to time
        # (B, out_channels, H, W)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residual_layer(residue)


class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, n_embed: int, d_context: int = 768):
        super().__init__()
        channels = n_head * n_embed

        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1 = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
    def forward(self, image, context):
        # image: (B, F, H, W)
        # context: (B, seq_len, Dim)
        residue = image

        # (B, F, H, W)
        image = self.groupnorm(image)
        image = self.conv_input(image)

        n, c, h, w = image.shape

        # (B, H*W, F)
        image = image.view((n, c, h * w)).transpose(1, 2)

        # Self attention
        # (B, H*W, F)
        image = image + self.attention_1(self.layernorm_1(image))

        # Cross Attention
        # (B, H*W, F)
        image = image + self.attention_2(self.layernorm_2(image), context)

        residue_tmp = image
        # (B, H*W, F)
        image = self.layernorm_3(image)

        # (B, H*W, F) -> two tensors of dim (B, H*W, F * 4)
        image, gate = self.linear_geglu_1(image).chunk(2, dim=-1)

        # (B, H*W, F * 4)
        image = image * F.gelu(gate)
        # (B, H*W, F)
        image = self.linear_geglu_2(image)

        image += residue_tmp
        # (B, F, H, W)
        image = image.transpose(-1, -2).contiguous().view((n, c, h, w))
        # print(f"image shape is {image.shape}")

        # residual connection -> (B, F, H, W)
        return residue + self.conv_output(image)
    
class UpSample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # X : (B, F, H, W)

        # (B, F, H * 2, W * 2)
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class SwitchSequential(nn.Sequential):
    def forward(self, image, context, time):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                image = layer(image, context)
            elif isinstance(layer, UNET_ResidualBlock):
                image = layer(image, time)
            else:
                image = layer(image)
    
        return image

class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        # decrease image dimension and increase the number of channels, input to this is the output of VAEEncoder (latent)
        self.encoders = nn.ModuleList([
            # (B, 4, H / 8, W / 8) -> (B, 320, H / 8, W / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            # (B, 320, H / 8, W / 8)
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            # (B, 320, H / 16, W / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, padding=1, stride=2)),
            # (B, 640, H / 16, W / 16)
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),

            # (B, 640, H / 32, W / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, padding=1, stride=2)),
            # (B, 1280, H / 32, W / 32)
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

            # (B, 1280, H / 64, W / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, padding=1, stride=2)),
            # (B, 1280, H / 64, W / 64)
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(
            # (B, 1280, H / 64, W / 64)
            UNET_ResidualBlock(1280, 1280),
            UNET_AttentionBlock(8, 160),
            UNET_ResidualBlock(1280, 1280),
        )

        self.decoders = nn.ModuleList([
            # (B, 1280, H / 64, W / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)), # 1280 concat
            # (B, 1280, H / 64, W / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),  # 1280 concat

            # (B, 1280, H / 32, W / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UpSample(1280)), # 1280 concat

            # (B, 1280, H / 32, W / 32)
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)), # 1280 concat
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)), # 1280 concat

            # (B, 1280, H / 16, W / 16)
            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), UpSample(1280)), # 640 concat
            # (B, 640, H / 16, W / 16)
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)), # 640 concat
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)), # 640 concat

            # (B, 640, H / 8, W / 8)
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), UpSample(640)), # 320 concat
            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)), # 320 concat
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)), # 320 concat
            # (B, 320, H / 8, W / 8)
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)), # 320 concat
        ])
    
    def forward(self, image, context, time):
        '''
        image: (B, 4, H / 8, W / 8)
        context: (B, s_len, Dim)
        time: (1, 1280)
        '''

        skip_connections = []
        for layers in self.encoders:
            image = layers(image, context, time)
            skip_connections.append(image)
        
        image = self.bottleneck(image, context, time)
        
        for layer in self.decoders:
            # this concat increases the number of images sent to decoder layers
            # 
            image = torch.cat((image, skip_connections.pop()), dim=1)
            image = layer(image, context, time)

            '''
            encoder last layer output is (B, 1280, H / 64, W / 64)
            image from bottleneck layer output is (B, 1280, H / 64, W / 64)
            These are concat in first decoder layer so first dim is 2560
            '''
        # (B, 320, H / 8, W / 8)
        return image

class UNET_OutputLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, image):
        # image: (B, 320, H / 8, W / 8)
        image = self.groupnorm(image)
        image = F.silu(image)
        # (B, 4, H / 8, W / 8)
        image = self.conv(image)
        return image

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)
    
    def forward(self, latent, context, time):
        '''
        latent is (B, 4, H / 8, W / 8)
        context: (B, seq_len, dim)
        time: (1, 320)
        '''
        time = self.time_embedding(time)
        # (B, 4, H / 8, W / 8)
        output = self.final(self.unet(latent, context, time))
        return output