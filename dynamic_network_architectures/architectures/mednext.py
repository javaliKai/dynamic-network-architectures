# mednext.py
# MedNeXt U-Net for nnU-Net v2 (BraTS-ready) with Deep Supervision
# Place this file under: dynamic_network_architectures/architectures/mednext.py

from typing import List, Union, Type
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd

# Optional: use nnU-Net's default He init
from dynamic_network_architectures.initialization.weight_init import InitWeights_He

# Import MedNeXt atomic blocks (ported from original blocks.py)
from dynamic_network_architectures.building_blocks.new.mednext_blocks import (
    MedNeXtBlock,
    MedNeXtDownBlock,
    MedNeXtUpBlock,
    OutBlock,
)


class MedNeXtUNet(nn.Module):
    """
    Monolithic MedNeXt-style U-Net with Deep Supervision (DS).
    Mirrors the original MedNeXt topology:
      stem → [enc0 → enc1 → enc2 → enc3] → bottleneck → [dec3 → dec2 → dec1 → dec0] → out

    Args:
        input_channels: number of modalities (BraTS=4)
        n_channels: base feature width (e.g., 32)
        num_classes: number of segmentation classes (include background, e.g., 4)
        conv_op: nn.Conv3d (for 3D BraTS) or nn.Conv2d
        deep_supervision: if True, returns list of [out_main, out_ds1..out_ds4]
        kernel_size: depthwise conv kernel (original paper uses 7)
        exp_r: expansion ratio(s), either int or list of length 9
        block_counts: how many MedNeXtBlocks per stage (length 9: enc0..dec0)
        do_res: residual connection inside MedNeXtBlock
        do_res_up_down: residual connection for Up/Down blocks
        norm_type: "group" (default) or "layer"
        grn: enable Global Response Normalization (optional, default False)
    """

    def __init__(
        self,
        input_channels: int,
        n_channels: int,
        num_classes: int,
        conv_op: Type[_ConvNd] = nn.Conv3d,
        deep_supervision: bool = True,
        kernel_size: int = 7,
        exp_r: Union[int, List[int]] = 4,
        block_counts: List[int] = None,
        do_res: bool = True,
        do_res_up_down: bool = True,
        norm_type: str = "group",
        grn: bool = False,
    ) -> None:
        super().__init__()

        # Sanity check for conv type
        assert conv_op in (nn.Conv3d, nn.Conv2d), "conv_op must be nn.Conv3d or nn.Conv2d"
        self.dim = "3d" if conv_op is nn.Conv3d else "2d"
        self.do_ds = deep_supervision

        # Default block counts if none provided
        # 9 stages total = enc0, enc1, enc2, enc3, bottleneck, dec3, dec2, dec1, dec0
        if block_counts is None:
            block_counts = [2, 2, 2, 2, 2, 2, 2, 2, 2]

        assert len(block_counts) == 9, "block_counts must have length 9"

        # Handle expansion ratios
        if isinstance(exp_r, int):
            exp_r = [exp_r for _ in range(len(block_counts))]
        assert len(exp_r) == 9, "exp_r must have length 9"

        # === Stem ===
        self.stem = conv_op(input_channels, n_channels, kernel_size=1, stride=1, padding=0, bias=True)

        # === Encoder ===
        self.enc_block_0 = nn.Sequential(*[
            MedNeXtBlock(n_channels, n_channels, exp_r[0], kernel_size,
                         do_res=do_res, norm_type=norm_type, dim=self.dim, grn=grn)
            for _ in range(block_counts[0])
        ])
        self.down_0 = MedNeXtDownBlock(n_channels, 2 * n_channels, exp_r[1], kernel_size,
                                       do_res=do_res_up_down, norm_type=norm_type, dim=self.dim, grn=grn)

        self.enc_block_1 = nn.Sequential(*[
            MedNeXtBlock(2 * n_channels, 2 * n_channels, exp_r[1], kernel_size,
                         do_res=do_res, norm_type=norm_type, dim=self.dim, grn=grn)
            for _ in range(block_counts[1])
        ])
        self.down_1 = MedNeXtDownBlock(2 * n_channels, 4 * n_channels, exp_r[2], kernel_size,
                                       do_res=do_res_up_down, norm_type=norm_type, dim=self.dim, grn=grn)

        self.enc_block_2 = nn.Sequential(*[
            MedNeXtBlock(4 * n_channels, 4 * n_channels, exp_r[2], kernel_size,
                         do_res=do_res, norm_type=norm_type, dim=self.dim, grn=grn)
            for _ in range(block_counts[2])
        ])
        self.down_2 = MedNeXtDownBlock(4 * n_channels, 8 * n_channels, exp_r[3], kernel_size,
                                       do_res=do_res_up_down, norm_type=norm_type, dim=self.dim, grn=grn)

        self.enc_block_3 = nn.Sequential(*[
            MedNeXtBlock(8 * n_channels, 8 * n_channels, exp_r[3], kernel_size,
                         do_res=do_res, norm_type=norm_type, dim=self.dim, grn=grn)
            for _ in range(block_counts[3])
        ])
        self.down_3 = MedNeXtDownBlock(8 * n_channels, 16 * n_channels, exp_r[4], kernel_size,
                                       do_res=do_res_up_down, norm_type=norm_type, dim=self.dim, grn=grn)

        # === Bottleneck ===
        self.bottleneck = nn.Sequential(*[
            MedNeXtBlock(16 * n_channels, 16 * n_channels, exp_r[4], kernel_size,
                         do_res=do_res, norm_type=norm_type, dim=self.dim, grn=grn)
            for _ in range(block_counts[4])
        ])

        # === Decoder ===
        self.up_3 = MedNeXtUpBlock(16 * n_channels, 8 * n_channels, exp_r[5], kernel_size,
                                   do_res=do_res_up_down, norm_type=norm_type, dim=self.dim, grn=grn)
        self.dec_block_3 = nn.Sequential(*[
            MedNeXtBlock(8 * n_channels, 8 * n_channels, exp_r[5], kernel_size,
                         do_res=do_res, norm_type=norm_type, dim=self.dim, grn=grn)
            for _ in range(block_counts[5])
        ])

        self.up_2 = MedNeXtUpBlock(8 * n_channels, 4 * n_channels, exp_r[6], kernel_size,
                                   do_res=do_res_up_down, norm_type=norm_type, dim=self.dim, grn=grn)
        self.dec_block_2 = nn.Sequential(*[
            MedNeXtBlock(4 * n_channels, 4 * n_channels, exp_r[6], kernel_size,
                         do_res=do_res, norm_type=norm_type, dim=self.dim, grn=grn)
            for _ in range(block_counts[6])
        ])

        self.up_1 = MedNeXtUpBlock(4 * n_channels, 2 * n_channels, exp_r[7], kernel_size,
                                   do_res=do_res_up_down, norm_type=norm_type, dim=self.dim, grn=grn)
        self.dec_block_1 = nn.Sequential(*[
            MedNeXtBlock(2 * n_channels, 2 * n_channels, exp_r[7], kernel_size,
                         do_res=do_res, norm_type=norm_type, dim=self.dim, grn=grn)
            for _ in range(block_counts[7])
        ])

        self.up_0 = MedNeXtUpBlock(2 * n_channels, n_channels, exp_r[8], kernel_size,
                                   do_res=do_res_up_down, norm_type=norm_type, dim=self.dim, grn=grn)
        self.dec_block_0 = nn.Sequential(*[
            MedNeXtBlock(n_channels, n_channels, exp_r[8], kernel_size,
                         do_res=do_res, norm_type=norm_type, dim=self.dim, grn=grn)
            for _ in range(block_counts[8])
        ])

        # === Output heads ===
        self.out_0 = OutBlock(n_channels, num_classes, self.dim)   # final output
        if self.do_ds:
            self.out_1 = OutBlock(2 * n_channels, num_classes, self.dim)
            self.out_2 = OutBlock(4 * n_channels, num_classes, self.dim)
            self.out_3 = OutBlock(8 * n_channels, num_classes, self.dim)
            self.out_4 = OutBlock(16 * n_channels, num_classes, self.dim)

        # Initialize weights
        InitWeights_He(1e-2)(self)

    def forward(self, x):
        # Stem
        x = self.stem(x)

        # Encoder
        x_res_0 = self.enc_block_0(x)
        x = self.down_0(x_res_0)

        x_res_1 = self.enc_block_1(x)
        x = self.down_1(x_res_1)

        x_res_2 = self.enc_block_2(x)
        x = self.down_2(x_res_2)

        x_res_3 = self.enc_block_3(x)
        x = self.down_3(x_res_3)

        # Bottleneck
        x = self.bottleneck(x)
        if self.do_ds:
            out_ds4 = self.out_4(x)

        # Decoder with skip connections
        x = self.up_3(x)
        x = self.dec_block_3(x + x_res_3)
        if self.do_ds:
            out_ds3 = self.out_3(x)

        x = self.up_2(x)
        x = self.dec_block_2(x + x_res_2)
        if self.do_ds:
            out_ds2 = self.out_2(x)

        x = self.up_1(x)
        x = self.dec_block_1(x + x_res_1)
        if self.do_ds:
            out_ds1 = self.out_1(x)

        x = self.up_0(x)
        x = self.dec_block_0(x + x_res_0)
        out_main = self.out_0(x)

        # Return multiple outputs if DS enabled
        if self.do_ds:
            return [out_main, out_ds1, out_ds2, out_ds3, out_ds4]
        else:
            return out_main
