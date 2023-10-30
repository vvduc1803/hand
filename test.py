import torch


mask = torch.zeros(
    (2, 32, 32), device='cuda', dtype=torch.bool
)
not_mask = ~mask

y_embed = not_mask.cumsum(1, dtype=torch.float32)
x_embed = not_mask.cumsum(2, dtype=torch.float32)
x_embed = x_embed[:, :, :, None]


x_embed= x_embed[:, :, :, 0::2].sin()
x_embed=x_embed[:, :, :, 1::2].cos()