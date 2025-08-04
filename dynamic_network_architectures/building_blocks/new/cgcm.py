# cgcm.py
"""
Channel-Graph Convolution Module (CGCM) – refactored 2025-08-04
---------------------------------------------------------------
Implements the channel–wise graph reasoning block described in
M2GCNet.  Fixes vs. first draft:
    1. Correct D-H-W axis order and contiguous reshapes
    2. √C scaling (optional softmax) for VC
    3. BN + LeakyReLU inside residual graph conv
    4. Residual shortcut to Xi
    5. Dynamic spatial shape, no manual args
    6. Optional 1×1×1 compression back to in_channels
"""

from __future__ import annotations
import math
from typing import Literal, Optional

import torch
import torch.nn as nn


# ------------------------------------------------------------------ #
#                       Helper blocks                                #
# ------------------------------------------------------------------ #
class EtaThetaProjectionBlock(nn.Module):
    """η(Xi) and θ(Xj) projection – 1×1×1 convs"""

    def __init__(self, in_channels: int, c1_ratio: float = 0.5, c2_ratio: float = 0.25):
        super().__init__()
        self.c1 = int(in_channels * c1_ratio)  # η channels
        self.c2 = int(in_channels * c2_ratio)  # θ channels

        self.eta_conv = nn.Conv3d(in_channels, self.c1, kernel_size=1, bias=False)
        self.theta_conv = nn.Conv3d(in_channels, self.c2, kernel_size=1, bias=False)

    def forward(self, Xi: torch.Tensor, Xj: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Xi/Xj shape: [B, C, D, H, W]
        B, _, D, H, W = Xi.shape
        N = D * H * W

        # 1×1×1 projections
        eta   = self.eta_conv(Xi)     # [B, C1, D, H, W]
        theta = self.theta_conv(Xj)   # [B, C2, D, H, W]

        # → [B, N, C*]  (flatten spatial)
        eta   = eta.reshape(B, self.c1, N).transpose(1, 2).contiguous()
        theta = theta.reshape(B, self.c2, N).transpose(1, 2).contiguous()
        return eta, theta


class VCBlock(nn.Module):
    """Compute channel-wise adjacency VC = θᵀ · η"""

    def __init__(self, scale: bool = True, softmax: bool = False):
        super().__init__()
        self.scale = scale
        self.softmax = nn.Softmax(dim=-1) if softmax else None

    def forward(self, eta: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        # eta  : [B, N, C1]
        # theta: [B, N, C2]
        C1 = eta.shape[-1]
        theta_T = theta.transpose(1, 2)          # [B, C2, N]
        VC = torch.bmm(theta_T, eta)             # [B, C2, C1]
        if self.scale:
            VC = VC / math.sqrt(C1)
        if self.softmax is not None:
            VC = self.softmax(VC)                # row-wise softmax on C1
        return VC


class ResidualGraphConv1D(nn.Module):
    """Two 1-D convs with BN+LeakyReLU and residual"""

    def __init__(self, channels: int, groups: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, VC: torch.Tensor) -> torch.Tensor:
        return self.block(VC) + VC          # residual over VC


class FusionAndReshapeBlock(nn.Module):
    """Multiply Zc with θ(Xj), reshape back to 3-D, then 1×1×1 projection"""

    def __init__(self, c1: int, out_channels: int, norm_type: Literal["bn", "in"] = "bn"):
        super().__init__()
        Norm = nn.BatchNorm3d if norm_type == "bn" else nn.InstanceNorm3d
        self.post_conv = nn.Sequential(
            nn.Conv3d(c1, out_channels, kernel_size=1, bias=False),
            Norm(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(
        self,
        Zc: torch.Tensor,        # [B, C2, C1]
        theta: torch.Tensor,     # [B, N, C2]
        spatial_shape: tuple[int, int, int],  # (D,H,W)
    ) -> torch.Tensor:
        # θ · Zc → [B, N, C1]
        M = torch.bmm(theta, Zc)                                 # [B, N, C1]
        B, N, C1 = M.shape
        D, H, W = spatial_shape
        M = (
            M.transpose(1, 2)
            .contiguous()
            .view(B, C1, D, H, W)                                # [B, C1, D, H, W]
        )
        return self.post_conv(M)                                 # [B, out_C, D,H,W]


# ------------------------------------------------------------------ #
#                        Main CGCM                                   #
# ------------------------------------------------------------------ #
class ChannelGraphConvModule(nn.Module):
    """
    Channel-wise Graph Convolution Module.
    Out-of-the-box settings:
      * residual shortcut to Xi (+B)
      * 1×1×1 compression back to in_channels (+C)
    """

    def __init__(
        self,
        in_channels: int,
        *,
        c1_ratio: float = 0.5,
        c2_ratio: float = 0.25,
        groups: int = 1,
        norm_type: Literal["bn", "in"] = "bn",
        use_residual: bool = True,
        compress: bool = True,
        vc_softmax: bool = False,
    ):
        super().__init__()
        self.use_residual = use_residual
        self.compress_flag = compress

        # building blocks
        self.proj   = EtaThetaProjectionBlock(in_channels, c1_ratio, c2_ratio)
        self.vc     = VCBlock(scale=True, softmax=vc_softmax)
        self.reason = ResidualGraphConv1D(channels=int(in_channels * c2_ratio),
                                          groups=groups)
        out_channels = in_channels                     # keep width identical
        self.fusion = FusionAndReshapeBlock(
            c1=int(in_channels * c1_ratio),
            out_channels=out_channels,
            norm_type=norm_type,
        )

        if compress:
            Norm = nn.BatchNorm3d if norm_type == "bn" else nn.InstanceNorm3d
            self.compress = nn.Sequential(
                nn.Conv3d(out_channels, in_channels, kernel_size=1, bias=False),
                Norm(in_channels),
                nn.LeakyReLU(inplace=True),
            )
        else:
            self.compress = nn.Identity()

    # ----------------------------- forward -------------------------- #
    def forward(self, Xi: torch.Tensor, Xj: torch.Tensor) -> torch.Tensor:
        # Xi/Xj shape: [B, C, D, H, W]
        D, H, W = Xi.shape[2:]

        # Step 1–2: projections
        eta, theta = self.proj(Xi, Xj)                 # [B,N,C1], [B,N,C2]

        # Step 3: channel adjacency
        VC = self.vc(eta, theta)                       # [B,C2,C1]

        # Step 4: graph reasoning
        Zc = self.reason(VC)                           # [B,C2,C1]

        # Step 5: fusion & reshape
        out = self.fusion(Zc, theta, (D, H, W))        # [B,C,D,H,W]

        # Step 6: residual to Xi
        if self.use_residual:
            out = out + Xi

        # Step 7: optional 1×1×1 compression
        return self.compress(out)
