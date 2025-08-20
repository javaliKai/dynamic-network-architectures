# BELOW IS THE PARTITION OF VSS3D CLASS
import math
import torch
import torch.nn as nn
from typing import Optional, Union, Type, List, Tuple, Callable, Dict
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from dynamic_network_architectures.building_blocks.new.ss3d import SS3D_v6

class FeedForward(nn.Module):
    def __init__(self, dim, dropout_rate, hidden_dim = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim=dim
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)

class VSSBlock3D_v6(nn.Module): #no multiplicative path, added MLP. more like transformer block used in TABSurfer now
  def __init__(
      self,
      hidden_dim: int = 0,
      drop_path: float = 0,
      norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
      attn_drop_rate: float = 0,
      d_state: int = 16,
      expansion_factor = 1, # can only be 1 for v3, no linear projection to increase channels
      mlp_drop_rate=0.,
      orientation = 0,
      scan_type = 'scan',
      size = 12,
      **kwargs,
      ):
    super().__init__()
    print(orientation, end='')
    self.ln_1 = norm_layer(hidden_dim)
    self.self_attention = SS3D_v6(d_model=hidden_dim,
                                  dropout=attn_drop_rate,
                                  d_state=d_state,
                                  expand=expansion_factor,
                                  orientation=orientation,
                                  mamba_layers=3,  # Control the number of consequent SSM (mamba) block here
                                  **kwargs)

    self.ln_2 = norm_layer(hidden_dim)
    self.mlp = FeedForward(dim = hidden_dim, hidden_dim=expansion_factor*hidden_dim, dropout_rate = mlp_drop_rate)

    self.drop_path = DropPath(drop_path)

  def forward(self, input: torch.Tensor):
    x = input + self.drop_path(self.self_attention(self.ln_1(input)))
    x = x + self.drop_path(self.mlp(self.ln_2(x)))
    return x