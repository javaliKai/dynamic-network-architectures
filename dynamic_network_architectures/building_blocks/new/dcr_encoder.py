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
        return name in inspect.signature(cls_or_fn).parameters
    except Exception:
        return False


def _to_list_per_stage(val, n_stages: int):
    if isinstance(val, (list, tuple)):
        assert len(val) == n_stages, f"Expected length {n_stages}, got {len(val)}"
        return list(val)
    return [val] * n_stages


class DCREncoder(nn.Module):
    """
    DCR-based encoder with the exact external contract of ResidualEncoder:
      - self.stages: nn.Sequential([... per-stage modules ...])
      - self.output_channels, self.strides (per-dim via maybe_convert_scalar_to_list)
      - forward(x) -> list of per-stage feature maps when return_skips=True; else last feature

    Stage policy (per your spec):
      Block-1: stride = 1, forced XY dilation (e.g., (1,2,2) in 3D)  -> RF boost
      Block-2: stride = strides[s] from plans (downsample here), no forced dilation
      Block-3+: stride = 1, no forced dilation
    """
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[Union[int, Tuple[int, ...]]], Tuple[Union[int, Tuple[int, ...]], ...]],
                 strides: Union[int, List[Union[int, Tuple[int, ...]]], Tuple[Union[int, Tuple[int, ...]], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 # extras that match ResidualEncoder API
                 block=None,                       # ignored; present for signature parity
                 bottleneck_channels=None,         # ignored
                 return_skips: bool = False,
                 disable_default_stem: bool = False,
                 stem_channels: int = None,
                 pool_type: str = 'conv',
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16,
                 # DCR tuning knobs (kept dynamic-safe)
                 dilation_xy: int = 2,
                 resnet_d_skip: bool = True):
        super().__init__()

        # normalize per-stage lists (no 'length=' arg -> compatible with older helpers)
        features_per_stage = _to_list_per_stage(features_per_stage, n_stages)
        kernel_sizes       = _to_list_per_stage(kernel_sizes,       n_stages)
        strides            = _to_list_per_stage(strides,            n_stages)
        n_blocks_per_stage = _to_list_per_stage(n_blocks_per_stage, n_stages)

        if norm_op_kwargs is None: norm_op_kwargs = {}
        if nonlin_kwargs is None:  nonlin_kwargs = {}
        if dropout_op_kwargs is None: dropout_op_kwargs = {}

        # ----- stem (no downsample), mirrors ResidualEncoder behavior -----
        if not disable_default_stem:
            if stem_channels is None:
                stem_channels = features_per_stage[0]
            self.stem = StackedConvBlocks(
                1, conv_op, input_channels, stem_channels, kernel_sizes[0], 1, conv_bias,
                norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs
            )
            in_ch = stem_channels
        else:
            self.stem = None
            in_ch = input_channels

        # pooling option parity (we still default to conv-style downsampling)
        pool_op = None
        if pool_type != 'conv':
            pool_op = get_matching_pool_op(conv_op=conv_op, pool_type=pool_type)

        # common kwargs for DCR blocks
        common = dict(
            conv_op=conv_op,
            conv_bias=conv_bias,
            norm_op=norm_op, norm_op_kwargs=norm_op_kwargs,
            dropout_op=dropout_op, dropout_op_kwargs=dropout_op_kwargs,
            nonlin=nonlin, nonlin_kwargs=nonlin_kwargs,
            stochastic_depth_p=broadcast_stochdepth(stochastic_depth_p, n_stages),
            squeeze_excitation=squeeze_excitation,
            squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio,
        )
        # pass optional flags only if DCRBlock supports them
        dcr_flags = {}
        if _supports_kwarg(DCRBlock.__init__, "dilation_xy"):
            dcr_flags["dilation_xy"] = dilation_xy
        if _supports_kwarg(DCRBlock.__init__, "resnet_d_skip"):
            dcr_flags["resnet_d_skip"] = resnet_d_skip

        stages = []
        for s in range(n_stages):
            blocks = []

            # If pooling is requested, do it BEFORE the stage stack and keep stride=1 within the stage
            stage_stride_for_block2 = strides[s] if pool_op is None else maybe_convert_scalar_to_list(conv_op, 1)

            # --- Block-1: stride=1, forced XY dilation (RF boost)
            kwargs_b1 = dict(common); kwargs_b1.update(dcr_flags)
            if _supports_kwarg(DCRBlock.__init__, "force_xy_dilation"):
                kwargs_b1["force_xy_dilation"] = True
            blocks.append(DCRBlock(
                input_channels=in_ch, output_channels=features_per_stage[s],
                kernel_size=kernel_sizes[s], stride=maybe_convert_scalar_to_list(conv_op, 1),
                **kwargs_b1
            ))
            in_ch = features_per_stage[s]

            # --- Block-2: apply plans' stride here (downsample), no forced dilation
            if n_blocks_per_stage[s] >= 2:
                kwargs_b2 = dict(common); kwargs_b2.update(dcr_flags)
                if "force_xy_dilation" in kwargs_b2:
                    kwargs_b2["force_xy_dilation"] = False
                blocks.append(DCRBlock(
                    input_channels=in_ch, output_channels=in_ch,
                    kernel_size=kernel_sizes[s], stride=stage_stride_for_block2,
                    **kwargs_b2
                ))

            # --- Extra blocks: stride=1, no forced dilation
            for _ in range(2, n_blocks_per_stage[s]):
                kwargs_bx = dict(common); kwargs_bx.update(dcr_flags)
                if "force_xy_dilation" in kwargs_bx:
                    kwargs_bx["force_xy_dilation"] = False
                blocks.append(DCRBlock(
                    input_channels=in_ch, output_channels=in_ch,
                    kernel_size=kernel_sizes[s], stride=maybe_convert_scalar_to_list(conv_op, 1),
                    **kwargs_bx
                ))

            stage_stack = nn.Sequential(*blocks)
            if pool_op is not None:
                # Mirror ResidualEncoder: pool first, then stage stack when pool_type != 'conv'
                stage_stack = nn.Sequential(pool_op(strides[s]), stage_stack)

            stages.append(stage_stack)

        # ----- PUBLIC CONTRACT (mirror ResidualEncoder) -----
        self.stages = nn.Sequential(*stages)                                        # nn.Sequential of per-stage modules  【residual_encoders.py†turn6file0†L26-L33】
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]  # per-dim strides                    【residual_encoders.py†turn6file0†L27-L29】
        self.return_skips = return_skips

        # attributes the decoder reads
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes                                           # decoder/size estimators use this    【residual_encoders.py†turn6file0†L31-L41】

    def forward(self, x):
        # exact behavior: run stem (if any), then accumulate per-stage outputs and return list if return_skips
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in self.stages:
            x = s(x)
            ret.append(x)
        return ret if self.return_skips else ret[-1]                               # matches ResidualEncoder forward     【residual_encoders.py†turn6file0†L42-L52】

    def compute_conv_feature_map_size(self, input_size):
        # mirrors ResidualEncoder’s accounting (conservative)
        if self.stem is not None and hasattr(self.stem, "compute_conv_feature_map_size"):
            output = self.stem.compute_conv_feature_map_size(input_size)
        else:
            output = np.int64(0)

        for s in range(len(self.stages)):
            # rely on each stage stack / DCRBlock to implement compute_conv_feature_map_size
            output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]
        return output                                                               # same reduction step as residual     【residual_encoders.py†turn6file0†L54-L64】


def broadcast_stochdepth(p: float, n_stages: int):
    """
    Simple helper: allow passing a single float for stochastic depth rate; we keep it constant per stage.
    If you want a schedule (e.g., linearly increasing with depth), adjust here.
    """
    return p
