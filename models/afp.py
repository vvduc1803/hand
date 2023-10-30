"""
Author: Luigi Piccinelli
Licensed under the CC-BY NC 4.0 license (http://creativecommons.org/licenses/by-nc/4.0/)
"""

from typing import Tuple

import torch
import torch.nn as nn
from einops import rearrange

from models.utils import (AttentionLayer, PositionEmbeddingSine,
                         _get_activation_cls, get_norm)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = get_norm('torchLN', hidden_features or in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self._init_weights()

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

class AFP(nn.Module):
    def __init__(
        self,
        num_resolutions: int = 1,
        pixel_dim: int = 256,
        latent_dim: int = 256,
        num_latents: int = 128,
        num_heads: int = 4,
        activation: str = "silu",
        norm: str = "torchLN",
        expansion: int = 2,
        eps: float = 1e-6,
        drop: float = 0.0
    ):
        super().__init__()
        self.num_resolutions = num_resolutions
        self.num_slots = num_latents
        self.latent_dim = latent_dim
        self.pixel_dim = pixel_dim
        self.eps = eps

        bottlenck_dim = expansion * latent_dim
        self.pe = PositionEmbeddingSine(pixel_dim, normalize=True)
        self.sink = nn.Parameter(torch.randn(1, self.num_slots, latent_dim))

        # Set up attention iterations
        self.attn = AttentionLayer(
                    sink_dim=latent_dim,
                    hidden_dim=latent_dim,
                    source_dim=pixel_dim,
                    output_dim=latent_dim,
                    num_heads=num_heads,
                    dropout=0.0,
                    pre_norm=True,
                    sink_competition=True,
                )

        self.mlp = Mlp(in_features=pixel_dim, hidden_features=latent_dim, act_layer=nn.SiLU)

    def forward(
        self, feature_map: torch.Tensor, idrs=None
    ) -> torch.Tensor:
        b, *_ = feature_map.shape

        # feature maps embedding pre-process
        feature_map, (h, w) = feature_map, feature_map.shape[-2:]
        feature_maps_flat = rearrange(
                feature_map + self.pe(feature_map),
                "b d h w-> b (h w) d",
            )

        if idrs is None:
            # IDRs generation
            idrs = self.sink.expand(b, -1, -1)

        # layers
        # Cross attention ops
        idrs = idrs + self.attn(
            idrs.clone(), feature_maps_flat
        )
        idrs = idrs + self.mlp(
            idrs.clone()
        )
        return idrs

    @classmethod
    def build(cls, config):
        output_num_resolutions = (
            len(config["model"]["pixel_encoder"]["embed_dims"])
            - config["model"]["afp"]["context_low_resolutions_skip"]
        )
        obj = cls(
            num_resolutions=output_num_resolutions,
            pixel_dim=config["model"]["pixel_decoder"]["hidden_dim"],
            num_latents=config["model"]["afp"]["num_latents"],
            latent_dim=config["model"]["afp"]["latent_dim"],
            num_heads=config["model"]["num_heads"],
            expansion=config["model"]["expansion"],
            activation=config["model"]["activation"],
        )
        return obj

if __name__ == '__main__':
    model = AFP().to('cuda')
    a = torch.rand((2, 256, 32, 32)).to('cuda')
    model(a)
    from torchsummary import summary
