# cgcm_fusion.py
"""
CGCMFusion – plug-and-play nnU-Net bottleneck enhancer
-----------------------------------------------------
Mirrors SGCMFusion’s API so you can toggle blocks with one line.
"""

from __future__ import annotations
from typing import Literal

import torch
import torch.nn as nn

from dynamic_network_architectures.building_blocks.new.cgcm import (
    ChannelGraphConvModule,
)


class CGCMFusion(nn.Module):
    """
    Parameters
    ----------
    in_channels : int
        Bottleneck channels from encoder.
    two_modality_mode : bool
        Split into 2 chunks (student) or 4 (teacher).
    norm_type : {'bn', 'in'}
        Normalisation style for internal 3-D convs.
    compress : bool
        If True, final tensor is restored to `in_channels`.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        two_modality_mode: bool = False,
        norm_type: Literal["bn", "in"] = "bn",
        compress: bool = True,
    ) -> None:
        super().__init__()

        if two_modality_mode:
            assert in_channels % 2 == 0
            pair_ch = in_channels // 2
        else:
            assert in_channels % 4 == 0
            pair_ch = in_channels // 4

        # two CGCM blocks if teacher-mode, else one
        self.cgcm_pair1 = ChannelGraphConvModule(
            in_channels=pair_ch,
            norm_type=norm_type,
            compress=False,          # compress after concat
        )
        if two_modality_mode:
            self.cgcm_pair2 = None
            out_after_pairs = pair_ch
        else:
            self.cgcm_pair2 = ChannelGraphConvModule(
                in_channels=pair_ch,
                norm_type=norm_type,
                compress=False,
            )
            out_after_pairs = pair_ch * 2  # concat

        # final compression back to bottleneck width (if requested)
        if compress:
            Norm = nn.BatchNorm3d if norm_type == "bn" else nn.InstanceNorm3d
            self.compress = nn.Sequential(
                nn.Conv3d(out_after_pairs, in_channels, kernel_size=1, bias=False),
                Norm(in_channels),
                nn.LeakyReLU(inplace=True),
            )
        else:
            self.compress = nn.Identity()

        self.two_modality_mode = two_modality_mode

    # ---------------------------- forward --------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, D, H, W]"""
        if self.two_modality_mode:
            Xi, Xj = torch.chunk(x, 2, dim=1)
            cgcm_out = self.cgcm_pair1(Xi, Xj)
        else:
            Xi, Xj, Xm, Xn = torch.chunk(x, 4, dim=1)
            out1 = self.cgcm_pair1(Xi, Xj)
            out2 = self.cgcm_pair2(Xm, Xn)
            cgcm_out = torch.cat([out1, out2], dim=1)

        return self.compress(cgcm_out)
