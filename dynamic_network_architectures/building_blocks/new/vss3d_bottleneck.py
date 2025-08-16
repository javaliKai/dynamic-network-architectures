import math
import torch
import torch.nn as nn
from typing import Optional, Sequence
from dynamic_network_architectures.building_blocks.new.vss3d import VSSBlock3D_v6

class VSSLayer3D(nn.Module):
    """ A basic layer for one stage.
    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        depth,
        attn_drop=0.,
        mlp_drop=0.,
        drop_path=0.,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        d_state=64,
        version = 'v5', #None, v5, v6
        expansion_factor = 1,
        scan_type = 'scan',
        orientation_order = None,
        size = 12,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint
        self.blocks = nn.ModuleList([
                    VSSBlock3D_v6(
                        hidden_dim=dim,
                        drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                        norm_layer=norm_layer,
                        attn_drop_rate=attn_drop,
                        d_state=d_state,
                        expansion_factor=expansion_factor,
                        mlp_drop_rate=mlp_drop,
                        scan_type=scan_type,
                        size = size,
                        orientation=i%8, # 0 1 2 3 4 5 6 7 8 > 0 1 2 3 4 5 0 1 2
                    )
                    for i in range(depth)])

        if True:
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_() # fake init, just to keep the seed ....
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))
            self.apply(_init_weights)

        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None


    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x



class VSS3DBottleneck(nn.Module):
    """
    VSS3DBottleneck
    ----------------
    A thin, shape-safe wrapper that inserts a VSS3D_v6 layer into nnU-Net’s
    bottleneck while keeping the rest of the architecture unchanged.

    Inputs/Outputs:
        x: torch.Tensor of shape [B, C, D, H, W]  (NCDHW)
        y: torch.Tensor of shape [B, C, D, H, W]  (same as input)

    What it does:
        1) Permutes to channel-last volume tokens: [B, H, W, D, C] (BHWD C)
        2) Runs VSSLayer3D (version='v6') with LayerNorm inside, as intended
        3) Optional post LayerNorm in channel-last
        4) Permutes back to NCDHW

    Notes:
        - Keeps channels fixed (no expand/compress); set `embed_channels`
          or wrap with 1x1x1 convs yourself if you want a widened bottleneck.
        - Use small depth first (e.g., 2–4) to check stability/memory, then scale.
    """

    def __init__(
        self,
        channels: int,
        depth: int = 4,
        d_state: int = 16,
        drop_path_rate: float = 0.1,
        attn_drop: float = 0.0,
        mlp_drop: float = 0.0,
        expansion_factor: int = 1,
        use_checkpoint: bool = False,
        scan_type: str = "scan",
        orientation_order: Optional[Sequence[int]] = None,
        add_post_layernorm: bool = True,
    ) -> None:
        super().__init__()

        # Per-block stochastic depth schedule (like timm)
        if depth > 1 and drop_path_rate > 0:
            import torch as _torch
            drop_path = _torch.linspace(0, drop_path_rate, steps=depth).tolist()
        else:
            drop_path = drop_path_rate

        self.vss = VSSLayer3D(
            dim=channels,
            depth=depth,
            drop_path=drop_path,
            attn_drop=attn_drop,
            mlp_drop=mlp_drop,
            d_state=d_state,
            version="v6",             # <- SS3D_v6 (non-cubic friendly)
            expansion_factor=expansion_factor,
            use_checkpoint=use_checkpoint,
            scan_type=scan_type,
            orientation_order=orientation_order,
        )
        # Optional extra LN after the VSS stack (MedSegMamba does this before compress) :contentReference[oaicite:2]{index=2}
        self.post_ln = nn.LayerNorm(channels) if add_post_layernorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [B, C, D, H, W]

        Returns:
            [B, C, D, H, W]  (same shape)
        """
        # --- shape hints ---
        # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape

        # NCDHW -> BHWD C (channel-last, as expected by VSS/SS3D) :contentReference[oaicite:3]{index=3}
        x_bhwdc = x.permute(0, 3, 4, 2, 1).contiguous()    # [B, H, W, D, C]

        # VSS stack (keeps last-dim = C)
        y_bhwdc = self.vss(x_bhwdc)                        # [B, H, W, D, C]
        y_bhwdc = self.post_ln(y_bhwdc)                    # [B, H, W, D, C]

        # Back to NCDHW
        y = y_bhwdc.permute(0, 4, 3, 1, 2).contiguous()    # [B, C, D, H, W]
        return y
