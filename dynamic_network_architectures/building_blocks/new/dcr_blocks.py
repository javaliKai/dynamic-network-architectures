
from typing import Tuple, List, Union, Type
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from dynamic_network_architectures.building_blocks.regularization import DropPath, SqueezeExcite


def _is_conv3d(conv_op: Type[_ConvNd]) -> bool:
    return getattr(conv_op, "__name__", "").lower().endswith("conv3d")


def _tuple_for_dim(conv_op: Type[_ConvNd], v: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
    """Ensure v matches dimensionality of conv_op."""
    if isinstance(v, tuple):
        return v
    if _is_conv3d(conv_op):
        return (v, v, v)
    return (v, v)


def _pad_for(kernel_size: Union[int, Tuple[int, ...]], dilation: Union[int, Tuple[int, ...]]) -> Tuple[int, ...]:
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size, kernel_size) if len(_tuple_for_dim(type(nn.Conv3d), 1)) == 3 else (kernel_size, kernel_size)
    if isinstance(dilation, int):
        dilation = (dilation,) * len(kernel_size)
    return tuple(((k - 1) * d) // 2 for k, d in zip(kernel_size, dilation))


class DCRBlock(nn.Module):
    """
    Dilated-Conv Residual Block (nnU-Net friendly, plan-injected ops).
    Layout:
        conv(k, stride=s, dilation=(1,d,d)) -> norm -> act ->
        conv(k, stride=1, dilation=1)       -> norm -> (+ skip proj) -> act

    - XY-only dilation is used automatically for Conv3d; Conv2d uses (d,d).
    - Dilation is enabled when stride != 1 (i.e., the first block of a stage in conv-downsample encoders).
      Subsequent blocks (stride=1) run without dilation to avoid gridding.
    - Matches the signature of BasicBlockD so it can be stacked similarly.
    """
    def __init__(self,
                 conv_op: Type[_ConvNd],
                 input_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, List[int], Tuple[int, ...]],
                 stride: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16):
        super().__init__()
        if norm_op_kwargs is None: norm_op_kwargs = {}
        if nonlin_kwargs is None: nonlin_kwargs = {}
        if dropout_op_kwargs is None: dropout_op_kwargs = {}

        # normalize args to lists/tuples matching dimensionality
        kernel_size = maybe_convert_scalar_to_list(conv_op, kernel_size)
        stride      = maybe_convert_scalar_to_list(conv_op, stride)

        # policy: enable dilation on "first block of the stage" i.e., when stride!=1
        has_stride = (isinstance(stride, int) and stride != 1) or any(i != 1 for i in stride)
        if _is_conv3d(conv_op):
            dil_xy = (1, 2, 2) if has_stride else (1, 1, 1)
        else:
            dil_xy = (2, 2) if has_stride else (1, 1)

        # ---- conv1 ----
        padding1 = _pad_for(kernel_size, dil_xy)
        self.conv1 = conv_op(input_channels, output_channels, kernel_size,
                             stride=stride, dilation=dil_xy, padding=padding1, bias=conv_bias)
        self.do1   = None if dropout_op is None else dropout_op(**dropout_op_kwargs)
        self.n1    = nn.Identity() if norm_op is None else norm_op(output_channels, **norm_op_kwargs)
        self.a1    = nn.Identity() if nonlin   is None else nonlin(**nonlin_kwargs)

        # ---- conv2 ----
        padding2 = _pad_for(kernel_size, 1)
        self.conv2 = conv_op(output_channels, output_channels, kernel_size,
                             stride=_tuple_for_dim(conv_op, 1), dilation=_tuple_for_dim(conv_op, 1),
                             padding=padding2, bias=conv_bias)
        self.n2    = nn.Identity() if norm_op is None else norm_op(output_channels, **norm_op_kwargs)

        # skip path (avg pool if striding, then 1x1 conv to match channels), ResNet-D style
        requires_projection = (has_stride or (input_channels != output_channels))
        if requires_projection:
            ops = []
            if has_stride:
                ops.append(get_matching_pool_op(conv_op=conv_op, adaptive=False, pool_type='avg')(stride, stride))
            ops.append(conv_op(input_channels, output_channels, _tuple_for_dim(conv_op, 1),
                               stride=_tuple_for_dim(conv_op, 1), bias=False))
            if norm_op is not None:
                ops.append(norm_op(output_channels, **norm_op_kwargs))
            self.proj = nn.Sequential(*ops)
        else:
            self.proj = None

        # optional SE and stochastic depth
        self.apply_se = squeeze_excitation
        if squeeze_excitation:
            self.se = SqueezeExcite(output_channels, rd_ratio=squeeze_excitation_reduction_ratio)
        self.apply_stochdepth = (stochastic_depth_p is not None and stochastic_depth_p > 0.0)
        if self.apply_stochdepth:
            self.drop_path = DropPath(stochastic_depth_p)

        # final nonlinearity
        self.a2 = nn.Identity() if nonlin is None else nonlin(**nonlin_kwargs)

        # book-keeping for compute_conv_feature_map_size
        self.output_channels = output_channels
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        y = self.conv1(x)
        if self.do1 is not None:
            y = self.do1(y)
        y = self.a1(self.n1(y))
        y = self.n2(self.conv2(y))
        if self.apply_se:
            y = self.se(y)
        if self.proj is not None:
            identity = self.proj(identity)
        if self.apply_stochdepth:
            y = self.drop_path(y)
        y = y + identity
        return self.a2(y)

    def compute_conv_feature_map_size(self, input_size: Tuple[int, ...]) -> np.int64:
        # mirrors Residual BasicBlockD estimation style
        assert len(input_size) == len(self.stride)
        size_after_stride = [i // j for i, j in zip(input_size, self.stride)]
        # conv1
        out1 = np.prod([self.output_channels, *size_after_stride], dtype=np.int64)
        # conv2
        out2 = np.prod([self.output_channels, *size_after_stride], dtype=np.int64)
        return out1 + out2
