# --- SS3D.py ---------------------------------------------------------------
import torch
import torch.nn as nn

# mambapy import (note the path!)
from mambapy.mamba import Mamba, MambaConfig


# class SS3D_v6_mambapy(nn.Module):
class SS3D_v6(nn.Module):
    """
    SS3D_v6_mambapy
    ----------------
    A 3D Selective-Scan (SSM) block implemented with **mambapy**.
    It avoids calling low-level selective_scan kernels and instead flattens
    3D volumes to sequences for a Mamba stack configured via MambaConfig.

    Input  (channel-last):  x  [B, H, W, D, C]     (BHWD C)
    Output (channel-last):  y  [B, H, W, D, C]

    Pipeline:
      1) Linear proj: C -> C'                             (channel-last)
      2) Depthwise 3D conv for local mixing              (channel-first)
      3) Multi-orientation, bi-directional Mamba passes:
         - base, transpose(H↔D), transpose(W↔D)
         - forward + reversed-sequence for each
         - sum the six outputs, undo rotations/transposes
      4) LayerNorm (channel-last) + Linear proj: C' -> C

    Notes:
      - Keeps spatial size; preserves channels after out_proj.
      - Use small `mamba_layers` (e.g., 1–2) to start. VSS depth is controlled
        by your VSSLayer3D stack; this SS3D block itself should stay light.
    """

    def __init__(
        self,
        d_model: int,            # C (input/output channels)
        d_state: int = 16,       # kept for signature parity; not all mambapy configs use it
        d_conv: int = 3,
        expand: int = 1,         # C' = expand * C
        dropout: float = 0.0,
        conv_bias: bool = True,
        bias: bool = False,
        orientation: int = 0,    # 0..7 rotation code (same semantics as your SS3D)
        num_direction: int = 6,  # 3 axes × 2 directions (fixed in this impl)
        mamba_layers: int = 6,   # <-- mambapy's MambaConfig.n_layers, default is 6 following the paper
        **factory_kwargs
    ) -> None:
        super().__init__()
        assert num_direction == 6, "This implementation assumes 3 axes × 2 directions."

        self.C = d_model
        self.Ci = int(expand * d_model)   # C'
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

        # (1) token projection (channel-last)
        self.in_proj = nn.Linear(self.C, self.Ci, bias=bias, **factory_kwargs)

        # (2) depthwise 3D conv (channel-first)
        self.conv3d = nn.Conv3d(
            in_channels=self.Ci,
            out_channels=self.Ci,
            groups=self.Ci,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        # (3) Mamba stack via MambaConfig (mambapy)
        # Minimal, robust config: d_model + n_layers. Add fields if your wheel supports them.
        self.mcfg = MambaConfig(d_model=self.Ci, n_layers=mamba_layers)
        self.mamba = Mamba(self.mcfg)

        # (4) post-norm + proj (channel-last)
        self.out_norm = nn.LayerNorm(self.Ci)
        self.out_proj = nn.Linear(self.Ci, self.C, bias=bias, **factory_kwargs)

        # Orientation helpers (same mapping style you used originally)
        if (orientation % 8) == 0:
            # Mode 0 is the original -- no rotation or translation applied
            self.rot   = lambda x: x
            self.unrot = lambda x: x
        elif (orientation % 8) == 1:
            self.rot   = lambda x: torch.rot90(x,  1, (2, 3))
            self.unrot = lambda x: torch.rot90(x, -1, (2, 3))
        elif (orientation % 8) == 2:
            self.rot   = lambda x: torch.rot90(x,  1, (3, 4))
            self.unrot = lambda x: torch.rot90(x, -1, (3, 4))
        elif (orientation % 8) == 3:
            self.rot   = lambda x: torch.rot90(x, -1, (2, 4))
            self.unrot = lambda x: torch.rot90(x,  1, (2, 4))
        elif (orientation % 8) == 4:
            self.rot   = lambda x: torch.transpose(torch.transpose(torch.rot90(torch.rot90(x, 2, (2, 3)), 1, (2, 4)), 2, 4), 2, 3)
            self.unrot = lambda x: torch.rot90(torch.rot90(torch.transpose(torch.transpose(x, 3, 4), 2, 3), -1, (2, 4)), 2, (2, 3))
        elif (orientation % 8) == 5:
            self.rot   = lambda x: torch.rot90(x, 2, (2, 4))
            self.unrot = lambda x: torch.rot90(x, 2, (2, 4))
        elif (orientation % 8) == 6:
            self.rot   = lambda x: torch.transpose(torch.transpose(torch.rot90(x, 2, (2, 3)), 3, 4), 2, 3)
            self.unrot = lambda x: torch.rot90(torch.transpose(torch.transpose(x, 2, 3), 3, 4), 2, (2, 3))
        else:
            self.rot   = lambda x: torch.rot90(x, -1, (2, 4))
            self.unrot = lambda x: torch.rot90(x,  1, (2, 4))

    @staticmethod
    def _seq_forward(mamba: Mamba, x_3d: torch.Tensor) -> torch.Tensor:
        """
        Run Mamba over flattened spatial tokens.

        Args:
          x_3d: [B, C', H, W, D]   # channel-first (C' first), D last
        Returns:
          y_3d: [B, C', H, W, D]
        """
        B, C, H, W, D = x_3d.shape
        L = H * W * D
        # Flatten to [B, L, C'] for Mamba
        x_seq = x_3d.reshape(B, C, L).transpose(1, 2).contiguous()     # [B, L, C']
        y_seq = mamba(x_seq)                                           # [B, L, C']
        # Restore 3D
        y_3d = y_seq.transpose(1, 2).contiguous().reshape(B, C, H, W, D)
        return y_3d

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
          x: [B, H, W, D, C]     # channel-last (BHWD C)
        Returns:
          y: [B, H, W, D, C]
        """
        # (1) project to inner width (channel-last)
        x = self.in_proj(x)                                            # [B, H, W, D, C']

        # (2) depthwise conv (switch to channel-first)
        x_cf = x.permute(0, 4, 1, 2, 3).contiguous()                   # [B, C', H, W, D]
        x_cf = self.act(self.conv3d(x_cf))                             # [B, C', H, W, D]

        # (3) orientation variants
        x0 = self.rot(x_cf)                                            # [B, C', H, W, D]
        x1 = torch.transpose(x0, 2, 4).contiguous()                    # [B, C', D, W, H]
        x2 = torch.transpose(x0, 3, 4).contiguous()                    # [B, C', H, D, W]

        outs = []
        # axis-0 (H,W,D) forward & reverse
        y0_f = self._seq_forward(self.mamba, x0)                       # [B, C', H, W, D]
        y0_b = torch.flip(self._seq_forward(self.mamba,
                         torch.flip(x0, dims=[2, 3, 4])), dims=[2, 3, 4])
        outs += [y0_f, y0_b]

        # axis-1 (D,W,H)
        y1_f = self._seq_forward(self.mamba, x1)                       # [B, C', D, W, H]
        y1_b = torch.flip(self._seq_forward(self.mamba,
                         torch.flip(x1, dims=[2, 3, 4])), dims=[2, 3, 4])
        y1_f = torch.transpose(y1_f, 2, 4).contiguous()                # -> [B, C', H, W, D]
        y1_b = torch.transpose(y1_b, 2, 4).contiguous()                # -> [B, C', H, W, D]
        outs += [y1_f, y1_b]

        # axis-2 (H,D,W)
        y2_f = self._seq_forward(self.mamba, x2)                       # [B, C', H, D, W]
        y2_b = torch.flip(self._seq_forward(self.mamba,
                         torch.flip(x2, dims=[2, 3, 4])), dims=[2, 3, 4])
        y2_f = torch.transpose(y2_f, 3, 4).contiguous()                # -> [B, C', H, W, D]
        y2_b = torch.transpose(y2_b, 3, 4).contiguous()                # -> [B, C', H, W, D]
        outs += [y2_f, y2_b]

        # merge and undo rotation
        y_cf = self.unrot(sum(outs))                                   # [B, C', H, W, D]

        # (4) back to channel-last, norm & proj
        y_cl = y_cf.permute(0, 2, 3, 4, 1).contiguous()                # [B, H, W, D, C']
        y_cl = self.out_norm(y_cl)
        y = self.out_proj(y_cl)                                        # [B, H, W, D, C]

        if self.dropout is not None:
            y = self.dropout(y)
        return y
