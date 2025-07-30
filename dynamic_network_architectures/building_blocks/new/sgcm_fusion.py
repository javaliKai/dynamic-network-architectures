import math
from typing import Literal, Optional

import torch
import torch.nn as nn

# Make sure the import path matches where you placed the file
from dynamic_network_architectures.building_blocks.new.sgcm import SpatialGraphConvModule


class SGCMFusion(nn.Module):
    """
    Parameters
    ----------
    in_channels : int
        Channel count of the bottleneck feature map coming from nnU-Net.
    two_modality_mode : bool, optional
        If True, run one SGCM pair (Xi, Xj) and skip (Xm, Xn).  Useful
        for the student model that only processes 2 modalities.  Default
        is False (teacher mode with 4 modalities → 2 pairs).
    norm_type : {'bn', 'in'}, optional
        Normalization layer to use after the final 1×1×1 compression conv.
        Default is 'bn' (match nnU-Net style).
    compress : bool, optional
        If True (default) a 1×1×1 conv brings channels back to
        `in_channels`.  Set False only if the downstream decoder expects
        `in_channels // 2` instead.
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
            assert in_channels % 2 == 0, (
                "In two-modality mode `in_channels` must be divisible by 2 "
                f"(got {in_channels})."
            )
            pair_channels = in_channels // 2
        else:
            assert in_channels % 4 == 0, (
                "Teacher mode expects `in_channels` divisible by 4 "
                f"(got {in_channels})."
            )
            pair_channels = in_channels // 4

        # --- SGCM blocks ---------------------------------------------------
        # Each block processes one modality-pair and returns   [B, C_pair, D,H,W]
        self.sgcm_pair1 = SpatialGraphConvModule(
            in_channels=pair_channels,
            proj_channels=pair_channels,
            out_channels=pair_channels,
        )

        if two_modality_mode:
            self.sgcm_pair2 = None  # kept for clarity
            out_channels_after_pairs = pair_channels
        else:
            # Second pair for (Xm, Xn)
            self.sgcm_pair2 = SpatialGraphConvModule(
                in_channels=pair_channels,
                proj_channels=pair_channels,
                out_channels=pair_channels,
            )
            out_channels_after_pairs = pair_channels * 2  # concatenation

        # --- Optional compression back to bottleneck width -----------------
        if compress:
            norm_layer = nn.BatchNorm3d if norm_type == "bn" else nn.InstanceNorm3d
            self.compress = nn.Sequential(
                nn.Conv3d(out_channels_after_pairs, in_channels, kernel_size=1, bias=False),
                norm_layer(in_channels),
                nn.LeakyReLU(inplace=True),
            )
        else:
            self.compress = nn.Identity()

        self.two_modality_mode = two_modality_mode

    # --------------------------------------------------------------------- #
    #                                forward                                #
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Bottleneck feature map, shape ``[B, C, D, H, W]``.

        Returns
        -------
        torch.Tensor
            Feature map of identical shape (channels = `C`) ready for the
            decoder.  If `compress=False`, channels will be `C // 2`.
        """
        B, C, D, H, W = x.shape  # only for asserts / readability

        if self.two_modality_mode:
            # Split into two equal channel chunks
            Xi, Xj = torch.chunk(x, 2, dim=1)
            S_ij = self.sgcm_pair1(Xi, Xj)           # [B, C/2, D,H,W]
            x_out = S_ij
        else:
            # Quarter-split: (Xi,Xj) and (Xm,Xn)
            Xi, Xj, Xm, Xn = torch.chunk(x, 4, dim=1)
            S_ij = self.sgcm_pair1(Xi, Xj)           # [B, C/4, ...]
            S_mn = self.sgcm_pair2(Xm, Xn)           # [B, C/4, ...]
            x_out = torch.cat([S_ij, S_mn], dim=1)   # [B, C/2, ... ]

        # Optional 1×1×1 compression back to `C`
        x_out = self.compress(x_out)                 # [B, C, ...] or [B, C/2, ...]
        return x_out
