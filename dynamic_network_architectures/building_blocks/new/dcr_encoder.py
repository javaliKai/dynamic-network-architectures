# dynamic_network_architectures/building_blocks/new/dcr_encoder.py

from typing import Union, Type, List, Tuple
import inspect
import numpy as np
import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.helper import (
    maybe_convert_scalar_to_list,
    get_matching_pool_op,
)
from dynamic_network_architectures.building_blocks.simple_conv_blocks import (
    StackedConvBlocks,
)
from dynamic_network_architectures.building_blocks.new.dcr_blocks import DCRBlock


def _supports_kwarg(cls_or_fn, name: str) -> bool:
    try:
        sig = inspect.signature(cls_or_fn)
        return name in sig.parameters
    except Exception:
        return False


class DCREncoder(nn.Module):
    """
    Plans-driven encoder that mirrors ResidualEncoder/MedNeXtEncoder APIs,
    but uses DCR blocks per stage with the following policy:

      • Two blocks per stage by default (controlled via n_blocks_per_stage from plans).
      • Block-1: stride = 1, FORCE XY-dilation (e.g., (1,2,2) for Conv3d) to boost RF.
      • Block-2: stride = strides[s] from plans (downsample happens here), no dilation.
      • Extra blocks (if any): stride = 1, no dilation.

    All ops (conv_op, norm_op, nonlin, dropout, etc.) and shapes (features_per_stage,
    kernel_sizes, strides, n_blocks_per_stage) come from the trainer/plans.

    NOTE:
      - We assume conv-style downsampling (pool_type='conv'), just like your current setup.
      - If you want plain 1x1x1 stride on the skip instead of ResNet-D (avg-pool + 1x1),
        set resnet_d_skip=False when constructing blocks (see below).
    """

    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[List[int], Tuple[int, ...]],
        conv_op: Type[_ConvNd],
        kernel_sizes: Union[
            List[Union[int, Tuple[int, ...]]], Tuple[Union[int, Tuple[int, ...]], ...]
        ],
        strides: Union[
            List[Union[int, Tuple[int, ...]]], Tuple[Union[int, Tuple[int, ...]], ...]
        ],
        n_blocks_per_stage: Union[List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[nn.Module]] = None,
        nonlin_kwargs: dict = None,
        return_skips: bool = False,
        disable_default_stem: bool = False,
        stem_channels: int = None,
        pool_type: str = "conv",
        stochastic_depth_p: float = 0.0,
        squeeze_excitation: bool = False,
        squeeze_excitation_reduction_ratio: float = 1.0 / 16,
        # Optional knobs for fine control (kept internal defaults to match spec):
        dilation_xy: int = 2,
        resnet_d_skip: bool = True,
    ):
        super().__init__()

        # Normalize to per-stage lists
        features_per_stage = maybe_convert_scalar_to_list(
            conv_op, features_per_stage, length=n_stages
        )
        kernel_sizes = maybe_convert_scalar_to_list(
            conv_op, kernel_sizes, length=n_stages
        )
        strides = maybe_convert_scalar_to_list(conv_op, strides, length=n_stages)
        if isinstance(n_blocks_per_stage, (list, tuple)):
            n_blocks_per_stage = list(n_blocks_per_stage)
        else:
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages

        assert len(kernel_sizes) == n_stages
        assert len(strides) == n_stages
        assert len(features_per_stage) == n_stages
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must match n_stages"

        self.strides = strides
        self.output_channels = features_per_stage
        self.return_skips = return_skips

        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {}

        # Optional stem (no downsample)
        if not disable_default_stem:
            if stem_channels is None:
                stem_channels = features_per_stage[0]
            self.stem = StackedConvBlocks(
                1,
                conv_op,
                input_channels,
                stem_channels,
                kernel_sizes[0],
                1,  # no downsample in stem
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
            )
            in_ch = stem_channels
        else:
            self.stem = None
            in_ch = input_channels

        if pool_type != "conv":
            # nnU-Net commonly uses conv downsampling in its encoders.
            # If you truly need pooling, we can extend this — but it's not used in your setup.
            _ = get_matching_pool_op(conv_op, pool_type=pool_type)
            raise NotImplementedError(
                "DCREncoder currently assumes conv-style downsampling (pool_type='conv')."
            )

        # Prepare common kwargs fed into DCRBlock
        common_block_kwargs = dict(
            conv_op=conv_op,
            conv_bias=conv_bias,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op,
            dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            stochastic_depth_p=stochastic_depth_p,
            squeeze_excitation=squeeze_excitation,
            squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio,
        )
        # Some installs may have older DCRBlock without the new flags; guard them:
        dcr_extra_flags = {}
        if _supports_kwarg(DCRBlock.__init__, "force_xy_dilation"):
            dcr_extra_flags["force_xy_dilation"] = True  # for Block-1
        if _supports_kwarg(DCRBlock.__init__, "dilation_xy"):
            dcr_extra_flags["dilation_xy"] = dilation_xy
        if _supports_kwarg(DCRBlock.__init__, "resnet_d_skip"):
            dcr_extra_flags["resnet_d_skip"] = resnet_d_skip

        stages = []
        for s in range(n_stages):
            blocks = []

            # ---- Block 1: stride=1, FORCE XY dilation to get the RF bump
            kwargs_b1 = dict(common_block_kwargs)
            kwargs_b1.update(dcr_extra_flags)
            if "force_xy_dilation" in kwargs_b1:
                kwargs_b1["force_xy_dilation"] = True  # ensure forced dilation on block 1

            blocks.append(
                DCRBlock(
                    input_channels=in_ch,
                    output_channels=features_per_stage[s],
                    kernel_size=kernel_sizes[s],
                    stride=1,  # no downsample on the first block
                    **kwargs_b1,
                )
            )
            in_ch = features_per_stage[s]

            # ---- Block 2: if present, downsample here with plans' stride; NO extra dilation
            if n_blocks_per_stage[s] >= 2:
                kwargs_b2 = dict(common_block_kwargs)
                # If flags exist, pass them but turn forced dilation off for Block-2
                if "force_xy_dilation" in dcr_extra_flags:
                    kwargs_b2.update(dcr_extra_flags)
                    kwargs_b2["force_xy_dilation"] = False

                blocks.append(
                    DCRBlock(
                        input_channels=in_ch,
                        output_channels=in_ch,  # keep width
                        kernel_size=kernel_sizes[s],
                        stride=strides[s],  # downsample here
                        **kwargs_b2,
                    )
                )

            # ---- Any additional blocks at this stage: stride=1, no dilation
            for _ in range(2, n_blocks_per_stage[s]):
                kwargs_bx = dict(common_block_kwargs)
                if "force_xy_dilation" in dcr_extra_flags:
                    kwargs_bx.update(dcr_extra_flags)
                    kwargs_bx["force_xy_dilation"] = False
                blocks.append(
                    DCRBlock(
                        input_channels=in_ch,
                        output_channels=in_ch,
                        kernel_size=kernel_sizes[s],
                        stride=1,
                        **kwargs_bx,
                    )
                )

            stages.append(nn.Sequential(*blocks))

        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor):
        skips: List[torch.Tensor] = []
        if self.stem is not None:
            x = self.stem(x)
        for stg in self.stages:
            x = stg(x)
            if self.return_skips:
                skips.append(x)
        return (skips, x) if self.return_skips else x

    def compute_conv_feature_map_size(self, input_size: Tuple[int, ...]) -> np.int64:
        # Basic accounting similar to other encoders: count stem, then apply per-stage strides.
        out = np.int64(0)
        if self.stem is not None and hasattr(self.stem, "compute_conv_feature_map_size"):
            out += self.stem.compute_conv_feature_map_size(input_size)

        # Apply stage-wise shrinking for rough estimate
        for s in range(len(self.stages)):
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return out
