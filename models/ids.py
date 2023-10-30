from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange

from models.utils import (AttentionLayer, PositionEmbeddingSine, _get_activation_cls, get_norm)

class ISDHead(nn.Module):
    def __init__(
        self,
        depth: int = 2,
        pixel_dim: int = 256,
        query_dim: int = 256,
        num_heads: int = 4,
        output_dim: int = 1,
        expansion: int = 2,
        activation: str = "silu",
        norm: str = "LN",
        eps: float = 1e-6,
    ):
        super().__init__()
        self.depth = depth
        self.eps = eps
        self.pixel_pe = PositionEmbeddingSine(pixel_dim, normalize=True)
        for i in range(self.depth):
            setattr(
                self,
                f"cross_attn_{i+1}",
                AttentionLayer(
                    sink_dim=pixel_dim,
                    hidden_dim=pixel_dim,
                    source_dim=query_dim,
                    output_dim=pixel_dim,
                    num_heads=num_heads,
                    dropout=0.0,
                    pre_norm=True,
                    sink_competition=False,
                ),
            )
            setattr(
                self,
                f"mlp_{i+1}",
                nn.Sequential(
                    get_norm(norm, pixel_dim),
                    nn.Linear(pixel_dim, expansion * pixel_dim),
                    _get_activation_cls(activation),
                    nn.Linear(expansion * pixel_dim, pixel_dim),
                ),
            )
        setattr(
            self,
            "proj_output",
            nn.Sequential(
                get_norm(norm, pixel_dim),
                nn.Linear(pixel_dim, pixel_dim),
                get_norm(norm, pixel_dim),
                nn.Linear(pixel_dim, output_dim),
            ),
        )

    def forward(self, feature_map: torch.Tensor, idrs: torch.Tensor):
        b, c, h, w = feature_map.shape
        a = feature_map + self.pixel_pe(feature_map)
        feature_map = rearrange(
            a, "b c h w -> b (h w) c"
        )

        for i in range(self.depth):
            update = getattr(self, f"cross_attn_{i+1}")(feature_map.clone(), idrs)
            feature_map = feature_map + update
            feature_map = feature_map + getattr(self, f"mlp_{i+1}")(feature_map.clone())
        out = getattr(self, "proj_output")(feature_map)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w)

        return out


class ISD(nn.Module):
    def __init__(
        self,
        num_resolutions: int = 1,
        depth: int = 2,
        pixel_dim=256,
        query_dim=256,
        num_heads: int = 4,
        output_dim: int = 1,
        expansion: int = 2,
        activation: str = "silu",
        norm: str = "torchLN",
    ):
        super().__init__()
        self.num_resolutions = num_resolutions
        self.head = ISDHead(
                    depth=depth,
                    pixel_dim=pixel_dim,
                    query_dim=query_dim,
                    num_heads=num_heads,
                    output_dim=output_dim,
                    expansion=expansion,
                    activation=activation,
                    norm=norm,
                )

    def forward(
        self, xs: torch.Tensor, idrs: torch.Tensor
    ) -> torch.Tensor:
        # outs, attns = [], []
        outs = self.head(xs, idrs)
        return outs

if __name__ == '__main__':
    model = ISD().to('cuda')
    a = torch.rand((2, 256, 32, 32)).to('cuda')
    b = torch.rand((2, 128, 256)).to('cuda')
    model(a, b)