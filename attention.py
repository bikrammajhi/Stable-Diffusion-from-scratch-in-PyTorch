import torch
from torch import nn
from torch.nn import functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_embed: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.n_heads = n_heads
        self.d_embed = d_embed
        self.d_head = d_embed // n_heads
        
        self.in_proj = nn.Linear(d_embed, 3 * d_embed, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_embed, d_embed, bias=out_proj_bias)
    
    def forward(self, x: torch.Tensor, causal_mask= False):
        # x: (Batch_Size, Seq_Len, Dim)
        
        input_shape = x.shape
        batch_size, seq_len, d_embed = input_shape
        
        intermim_shape = (batch_size, seq_len, self.n_heads, self.d_head)
        
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, 3 * Dim) -> 3 tensor of shape (Batch_Size, Seq_Len, Dim)
        q, k, v = self.in_proj(x).chunk(3, dim=-1)
        
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, H, Dim/H) -> (Batch_Size, H, Seq_Len, Dim/H)
        q = q.view(*intermim_shape).transpose(1, 2)
        k = k.view(*intermim_shape).transpose(1, 2)
        v = v.view(*intermim_shape).transpose(1, 2)
        
        # (Batch_Size, H, Seq_Len, Dim/H) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        
        if causal_mask:
            # Mask where the lower triangular part of the matrix is filled with -inf and above it with 1
            mask = torch.ones_like(weights, dtype=torch.bool).triu(1)
            weights.masked_fill_(mask, float('-inf'))
            
        weights /= math.sqrt(self.d_head)
        weights = F.softmax(weights, dim=-1)
        
        # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_size, H, Seq_Len, Dim/H) -> (Batch_Size, H, Seq_Len, Dim/H)
        output = weights @ v
        
        # (Batch_Size, H, Seq_Len, Dim/H) -> (Batch_Size, Seq_Len, H, Dim/H) 
        output = output.transpose(1, 2)
        
        output = output.reshape(input_shape)
        output = self.out_proj(output)
        
        # (Batch_Size, Seq_Len, Dim) 
        return output
        
        
        