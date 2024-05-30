import torch
from einops import rearrange


# B, C, H, W
a = torch.arange(9 * 2 * 2).view(1, 9, 2, 2)
print(a)

b = rearrange(a, 'b c h w -> b c (h w)')
print(b.shape)
