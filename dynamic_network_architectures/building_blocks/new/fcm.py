import torch
import torch.nn as nn
import torch.nn.functional as F

# Helpers
def _ratio(n: int, r: float, min_ch: int = 1) -> int:
    """Compute at-least-1 channel reduction."""
    return max(min_ch, int(round(n * r)))


class FCM3D(nn.Module):
    """
    Feature Complement Module (3D), faithful to Pocket Convolution Mamba:
      - Upsample H to L's spatial size (trilinear by default)
      - Compress L -> alpha*C_L via 1x1x1
      - Compress H -> beta*C_H  via 1x1x1
      - Concat [Lc, Hc], fuse via 1x1x1 to C_L
    """
    def __init__(self, c_low, c_high, alpha=0.75, beta=0.50, up_mode="trilinear"):
        super().__init__()
        self.c_low, self.c_high = c_low, c_high
        self.cL_comp = _ratio(c_low,  alpha)
        self.cH_comp = _ratio(c_high, beta)
        self.up_mode = up_mode

        self.reduce_low  = nn.Conv3d(c_low,  self.cL_comp, 1, bias=False)
        self.reduce_high = nn.Conv3d(c_high, self.cH_comp, 1, bias=False)
        self.fuse        = nn.Conv3d(self.cL_comp + self.cH_comp, c_low, 1, bias=False)

    def forward(self, L, H):
        # Upsample H to match L
        H_up = F.interpolate(H, size=L.shape[2:], mode="trilinear", align_corners=False) \
               if self.up_mode == "trilinear" else H
        Lc = self.reduce_low(L)
        Hc = self.reduce_high(H_up)
        Fcat = torch.cat([Lc, Hc], dim=1)
        out = self.fuse(Fcat)
        return out

# -----------------------------
# 2) Modality-Aware FCM (MA-FCM)
# -----------------------------
class MAFCM3D(nn.Module):
    """
    Modality-Aware FCM (3D) with soft channel–modality assignment:
      - Same compress/upsample as FCM
      - Gates g \in [0,1]^M predicted from high-level context (Hc)
      - Learnable A \in R^{cL_comp x M} maps gates to per-channel scales
      - L_mod = Lc * (A @ g), then concat with Hc and fuse
    No regularization included (pure behavior).
    """
    def __init__(self, c_low, c_high, num_modalities=4,
                 alpha=0.75, beta=0.50, up_mode="trilinear"):
        super().__init__()
        self.c_low, self.c_high = c_low, c_high
        self.M = num_modalities
        self.cL_comp = _ratio(c_low,  alpha)
        self.cH_comp = _ratio(c_high, beta)
        self.up_mode = up_mode

        # Compression
        self.reduce_low  = nn.Conv3d(c_low,  self.cL_comp, 1, bias=False)
        self.reduce_high = nn.Conv3d(c_high, self.cH_comp, 1, bias=False)

        # Gates from high-level context (Hc)
        hidden = max(8, self.cH_comp // 4)
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.cH_comp, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.M),
        )

        # Learnable channel–modality assignment matrix A
        self.A = nn.Parameter(torch.randn(self.cL_comp, self.M) * 0.02)

        # Final fuse after concat
        # instead of a fixed out_channels,
        # set it dynamically based on the input channels of the skip + upsampled bottleneck
        fused_channels = self.cL_comp + self.cH_comp
        self.fuse = nn.Conv3d(fused_channels, 2*c_low, kernel_size=1, bias=False)


    def _upsample(self, H, target_spatial):
        if self.up_mode == "trilinear":
            return F.interpolate(H, size=target_spatial, mode="trilinear", align_corners=False)
        elif self.up_mode == "deconv":
            # For arbitrary sizes, deconv needs fixed stride/kernels; trilinear is safer.
            raise NotImplementedError("Use trilinear for arbitrary shape matching.")
        return H

    def forward(self, L, H, modality_mask: torch.Tensor = None):
        """
        L : (B, C_L, D, H, W)   - low-level skip
        H : (B, C_H, d, h, w)   - high-level/bottleneck or current decoder state
        modality_mask (optional): (B, M) with 1 for present modality, else 0
        """
        B = L.shape[0]

        # 1) Upsample & compress
        H_up = self._upsample(H, L.shape[2:])
        Lc   = self.reduce_low(L)     # (B, cL_comp, D,H,W)
        Hc   = self.reduce_high(H_up) # (B, cH_comp, D,H,W)

        # 2) Gates from Hc context
        ctx   = Hc.mean(dim=(2,3,4))          # (B, cH_comp)
        g_raw = self.gate_mlp(ctx)            # (B, M)
        gates = torch.sigmoid(g_raw)          # (B, M)

        if modality_mask is not None:
            # Zero-out missing modalities and renormalize to sum ~ 1
            gates = gates * modality_mask
            denom = gates.sum(dim=1, keepdim=True).clamp_min(1e-6)
            gates = gates / denom

        # 3) Soft assignment: per-channel scaling s = A @ g
        weights = torch.softmax(self.A, dim=1)    # (cL_comp, M), row-wise distribution over M
        # (B, cL_comp) = (B, M) @ (M, cL_comp)^T
        scale = torch.matmul(gates, weights.t())  # (B, cL_comp)
        scale = scale.view(B, self.cL_comp, 1, 1, 1)

        L_mod = Lc * scale

        # 4) Fuse like FCM
        Fcat = torch.cat([L_mod, Hc], dim=1)
        out  = self.fuse(Fcat)  # -> (B, C_L, D, H, W)
        return out