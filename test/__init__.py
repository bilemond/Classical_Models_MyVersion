import torch
from einops import rearrange
import os
import time

# # B, C, H, W
# a = torch.arange(9 * 2 * 2).view(1, 9, 2, 2)
# print(a)
#
# b = rearrange(a, 'b c h w -> b c (h w)')
# print(b.shape)

save_path = f"results/"
if os.path.exists(save_path) is False:
    os.makedirs(save_path)

filename = save_path + f"models_{str(time.time())}.pth"
print(filename)
