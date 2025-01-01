import torch
from torch import nn
from torch.nn import functional as F
import torch.nn as nn
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super().__init__(
            # (Batch, Channel, Height, Width) -> (Batch, 128, Height, Width)
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1),
            
            # (Batch, 128, Height, Width) -> (Batch, 128, Height, Width)
            VAE_ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            
            # (Batch, 128, Height, Width) -> (Batch, 128, Height, Width)
            VAE_ResidualBlock(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            
            # (Batch, 128, Height, Width) -> (Batch, 256, Height/2, Width/2)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=0),
            
            # (Batch, 256, Height/2, Width/2) -> (Batch, 256, Height/2, Width/2)
            VAE_ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            
            # (Batch, 256, Height/2, Width/2) -> (Batch, 256, Height/4, Width/4)
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=0),
            
            # (Batch, 256, Height/4, Width/4) -> (Batch, 512, Height/4, Width/4)
            VAE_ResidualBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            
            # (Batch, 512, Height/4, Width/4) -> (Batch, 512, Height/4, Width/4)
            VAE_ResidualBlock(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            
            # (Batch, 512, Height/4, Width/4) -> (Batch, 512, Height/8, Width/8)
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=0),
            
            
            
            
            # (Batch, 512, Height/8, Width/8) -> (Batch, 512, Height/8, Width/8)
            VAE_ResidualBlock(in_channels=512, out_channels=512),
            
            # (Batch, 512, Height/8, Width/8) -> (Batch, 512, Height/8, Width/8)
            VAE_ResidualBlock(in_channels=512, out_channels=512),
            
            
            
            
            # (Batch, 512, Height/8, Width/8) -> (Batch, 512, Height/8, Width/8)
            VAE_AttentionBlock(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            
            # (Batch, 512, Height/8, Width/8) -> (Batch, 512, Height/8, Width/8)
            VAE_ResidualBlock(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            
            # (Batch, 512, Height/8, Width/8) -> (Batch, 512, Height/8, Width/8)
            VAE_ResidualBlock(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            
            # (Batch, 512, Height/8, Width/8) -> (Batch, 512, Height/8, Width/8)
            nn.GroupNorm(32, 512),
            
            # (Batch, 512, Height/8, Width/8) -> (Batch, 512, Height/8, Width/8)
            nn.SiLU(),
            
            #  (Batch, 512, Height/8, Width/8) -> (Batch, 8, Height/8, Width/8)
            nn.Conv2d(in_channels=512, out_channels=8, kernel_size=3, stride=1, padding=1),
            
            # (Batch, 8, Height/8, Width/8) -> (Batch, 8, Height/8, Width/8)
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        )
    
    def forward(self, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Channel, Height, Width)
        # noise: (Batch, Channel, Height, Width)
        
        for module in self: 
            if getattr(module, "in_channels", None) == (2,2):
                # Padding_left, padding_right, padding_top, padding_bottom
                x = F.pad(x, (0, 1, 0, 1))
            x = module(x)
            
        # (Batch, 8, Height/8, Width/8) -> two tensor of shape (Batch, 4, Height/8, Width/8)
        mean, log_variance = torch.chunk(x, chunks=2, dim=1)
        
        # (Batch, 4, Height/8, Width/8) -> (Batch, 4, Height, Width)
        log_variance = torch.clamp(log_variance, min=-30., max=20.)
        
        # (Batch, 4, Height/8, Width/8) -> (Batch, 4, Height, Width)
        variance = log_variance.exp()
        
        # (Batch, 4, Height, Width) -> (Batch, 4, Height, Width)
        stdev = variance.sqrt()
        
        # Z = N(0, 1) -> N(mean, stdev) ?
        # X = mean + stdev * Z
        x = mean + stdev * noise
        
        # Scale the output by a constant factor
        x = x * 0.18215
    
        return x