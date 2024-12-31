import torch
from torch import nn
import torch.nn.functional as F
from attention import SelfAttention


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