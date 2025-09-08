# dynamic_network_architectures/building_blocks/mednext_encoder.py
from typing import Union, Type, List, Tuple
import numpy as np
import torch
from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list

# Adjust this import path if your mednext_blocks.py lives elsewhere
from dynamic_network_architectures.building_blocks.new.mednext_blocks import MedNeXtBlock, MedNeXtDownBlock


def _to_int_kernel(ks) -> int:
    """
    nnU-Net often supplies per-dim kernels like [3,3,3]. MedNeXtBlock takes an int.
    We'll just take the first entry (assuming cubic kernels, which is nnU-Net default).
    """
    if isinstance(ks, int):
        return ks
    if isinstance(ks, (list, tuple)):
        return int(ks[0])
    return int(ks)


def _conv_op_to_dim(conv_op: Type[_ConvNd]) -> str:
    if conv_op is torch.nn.Conv3d:
        return "3d"
    if conv_op is torch.nn.Conv2d:
        return "2d"
    raise NotImplementedError(f"MedNeXtEncoder only supports Conv2d/Conv3d, got {conv_op}")

def _norm_type_from(norm_op, conv_op) -> str:
    # Handle common cases and nnU-Net helpers
    if norm_op is None:
        return 'group'  # fallback to previous default
    try:
        if issubclass(norm_op, nn.GroupNorm):
            return 'group'
        if issubclass(norm_op, nn.LayerNorm):
            return 'layer'
        if issubclass(norm_op, (nn.InstanceNorm2d, nn.InstanceNorm3d)):
            return 'instance'
    except TypeError:
        pass
    # name-based fallback (handles get_matching_instancenorm, etc.)
    name = getattr(norm_op, '__name__', str(norm_op))
    if 'InstanceNorm' in name:
        return 'instance'
    if 'GroupNorm' in name:
        return 'group'
    if 'LayerNorm' in name:
        return 'layer'
    return 'group'


class MedNeXtEncoder(nn.Module):
    """
    Mirrors the public API of ResidualEncoder so the decoder & trainer can treat it identically.
    Stages are built from MedNeXtDownBlock/MedNeXtBlock according to strides and features_per_stage.
    """
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 return_skips: bool = False,
                 disable_default_stem: bool = False,
                 stem_channels: int = None,
                 pool_type: str = 'conv',
                 # MedNeXt-specific (kept explicit so we don't fight plans.json kwargs)
                 exp_r: int = 4,
                 use_grn: bool = False):
        super().__init__()

        # Normalize scalar inputs to per-stage lists (like ResidualEncoder)
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages

        assert len(kernel_sizes) == n_stages, "kernel_sizes must have n_stages entries"
        assert len(features_per_stage) == n_stages, "features_per_stage must have n_stages entries"
        assert len(n_blocks_per_stage) == n_stages, "n_blocks_per_stage must have n_stages entries"
        assert len(strides) == n_stages, "strides must have n_stages entries"
        # We only support in-block downsampling (pool_type='conv' style). avg/max pooling not implemented here.
        assert pool_type == 'conv', "MedNeXtEncoder currently supports only pool_type='conv' (strided conv in-block)"

        dim = _conv_op_to_dim(conv_op)

        # Decide which norm to use
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        norm_type = _norm_type_from(norm_op, conv_op)
        


        # --- Stem (identical contract to ResidualEncoder) ---
        if not disable_default_stem:
            if stem_channels is None:
                stem_channels = features_per_stage[0]
            self.stem = StackedConvBlocks(
                1,                      # one conv in the stem (matches ResidualEncoder behavior)
                conv_op,
                input_channels,
                stem_channels,
                kernel_sizes[0],
                1,                      # no downsample in stem
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs
            )
            in_ch = stem_channels
        else:
            self.stem = None
            in_ch = input_channels

        # --- Build stages ---
        stages = []
        for s in range(n_stages):
            out_ch = features_per_stage[s]
            stride_s = strides[s]
            k_s = _to_int_kernel(kernel_sizes[s])

            # Convert stride to scalar (1 or 2 expected in nnU-Net plans)
            if isinstance(stride_s, (tuple, list)):
                # Take first entry; nnU-Net uses equal per-dim strides by design
                stride_s_scalar = int(stride_s[0])
            else:
                stride_s_scalar = int(stride_s)

            n_blocks = int(n_blocks_per_stage[s])
            assert n_blocks >= 1, f"Each stage needs at least 1 block, got {n_blocks} at s={s}"

            blocks = []
            if stride_s_scalar > 1:
                # First block performs downsampling and in->out projection
                blocks.append(
                    MedNeXtDownBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        exp_r=exp_r,
                        kernel_size=k_s,
                        do_res=True,          # residual downsample path
                        norm_type=norm_type,    # internal to MedNeXt; stem follows plans.json
                        dim=dim,
                        grn=use_grn,
                        norm_kwargs=self.norm_op_kwargs
                    )
                )
            else:
                # No downsample; regular block does in->out on first block
                blocks.append(
                    MedNeXtBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        exp_r=exp_r,
                        kernel_size=k_s,
                        do_res=True,
                        norm_type=norm_type,
                        dim=dim,
                        grn=use_grn,
                        norm_kwargs=self.norm_op_kwargs
                    )
                )

            # Remaining blocks keep channels = out_ch
            for _ in range(n_blocks - 1):
                blocks.append(
                    MedNeXtBlock(
                        in_channels=out_ch,
                        out_channels=out_ch,
                        exp_r=exp_r,
                        kernel_size=k_s,
                        do_res=True,
                        norm_type=norm_type,
                        dim=dim,
                        grn=use_grn,
                        norm_kwargs=self.norm_op_kwargs
                    )
                )

            stage = nn.Sequential(*blocks)
            stages.append(stage)
            in_ch = out_ch  # next stage input

        self.stages = nn.Sequential(*stages)

        # --- Attributes the decoder expects (mirror ResidualEncoder) ---
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # Store meta for decoder/factories
        self.conv_op = conv_op
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        return ret if self.return_skips else ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        """
        Conservative estimate; nnU-Net uses this for planning/VRAM heuristics.
        We follow ResidualEncoder’s contract but don’t try to introspect MedNeXt internals.
        """
        if self.stem is not None:
            output = self.stem.compute_conv_feature_map_size(input_size)
        else:
            output = np.int64(0)

        # Account for spatial shrinking; we don't add inner-block conv counts here
        for s in range(len(self.stages)):
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]

        return output
