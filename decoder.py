import torch
from torch import nn
import torch.nn.functional as F
from attention import SelfAttention


class VAE_AttentionBlock(nn.Module):   
    def __init__(self, channels: int):
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels)
        self.self_attention = SelfAttention(1, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Features, Height, Width)
        residue = x
        n, c, h, w = x.size()
        
        # (Batch, Features, Height, Width) -> (Batch, Features, Height * Width)
        x = x.view(n, c, h * w)
        # (Batch, Features, Height * Width) -> (Batch, Height * Width, Features)
        x = x.permute(-1, -2)
        
        # (Batch, Height * Width, Features) -> (Batch, Height * Width, Features)
        x = self.self_attention(x)
        
        # (Batch, Height * Width, Features) -> (Batch, Features, Height * Width)
        x = x.transpose(-1, -2)
        
        # (Batch, Features, Height * Width) -> (Batch, Features, Height, Width)
        x = x.view(n, c, h, w)
        
        x += residue
        
        return x
    

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.group_norm1 = nn.GroupNorm(32, in_channel)
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        
        self.group_norm2 = nn.GroupNorm(32, out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        
        if in_channel == out_channel:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channel, out_channel, 1, padding=0)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, in_channel, Height, Width)
        
        residue = x
        x = F.silu(self.group_norm1(x))
        x = self.conv1(x)
        
        x = F.silu(self.group_norm2(x))
        x = self.conv2(x)
        
        return x + self.residual_layer(residue)
    
class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(in_channels=4, out_channels=4, kernel_size=1, padding=0),
            nn.Conv2d(in_channels=4, out_channels=512, kernel_size=3, padding=1),
            
            VAE_ResidualBlock(in_channel=512, out_channel=512),
            VAE_AttentionBlock(channels=512),
            VAE_ResidualBlock(in_channel=512, out_channel=512),
            VAE_ResidualBlock(in_channel=512, out_channel=512),
            VAE_ResidualBlock(in_channel=512, out_channel=512),
            
            # (Batch, 512, Height/8, Width/8) -> (Batch, 512, Height/8, Width/8)
            VAE_ResidualBlock(in_channel=512, out_channel=512),
            
            # (Batch, 512, Height/8, Width/8) -> (Batch, 512, Height/4, Width/4)
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            
            VAE_ResidualBlock(in_channel=512, out_channel=256),
            VAE_ResidualBlock(in_channel=256, out_channel=256),
            VAE_ResidualBlock(in_channel=256, out_channel=256),
            
            # (Batch, 256, Height/2, Width/2) -> (Batch, 256, Height, Width)
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            
            VAE_ResidualBlock(in_channel=256, out_channel=128),
            VAE_ResidualBlock(in_channel=128, out_channel=128),
            VAE_ResidualBlock(in_channel=128, out_channel=128),
            
            nn.GroupNorm(32, 128),
            nn.SiLU(),
            
            # (Batch, 128, Height, Width) -> (Batch, 3, Height, Width)
            nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, padding=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, 4, Height/8, Width/8)
        x /= 0.18215
        
        for module in self:
            x = module(x)
            
        # (Batch, 3, Height, Width)
        return x