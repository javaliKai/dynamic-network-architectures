import math
import torch
import torch.nn as nn

class QKProjectionBlock(nn.Module):
    """
    Projects input features into Query and Key using 1x1x1 convolutions.
    """
    def __init__(self, in_channels, proj_channels=None):
        super(QKProjectionBlock, self).__init__()

        if proj_channels is None:
            proj_channels = in_channels  # keep same dim if not specified

        self.q_conv = nn.Conv3d(in_channels, proj_channels, kernel_size=1, bias=False)
        self.k_conv = nn.Conv3d(in_channels, proj_channels, kernel_size=1, bias=False)

    def forward(self, x):
        q = self.q_conv(x)  # shape: [B, C, D, H, W]
        k = self.k_conv(x)
        return q, k

class AdjacencyComputer(nn.Module):
    """
    Computes attention-based adjacency matrix A = softmax(Q^T @ K)
    """
    def __init__(self):
        super(AdjacencyComputer, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K):
        B, C, D, H, W = Q.shape
        N = H * W * D

        # Flatten spatial dimensions
        Q_flat = Q.reshape(B, C, N)         # [B, C, N]
        K_flat = K.reshape(B, C, N)         # [B, C, N]

        # Qᵀ @ K → [B, N, N] -- with additional channel scaling following attention mech in transformer
        A = torch.bmm(Q_flat.transpose(1, 2), K_flat) / math.sqrt(C)  # [B, N, N]

        # Normalize with softmax over last dim (each row = attention weights)
        A = self.softmax(A)  # [B, N, N]

        return A  # adjacency matrix


class ValueProjectionBlock(nn.Module):
    """
    Projects input features into Value using a 1x1x1 convolution.
    Used for message passing in SGCM.
    """
    def __init__(self, in_channels, proj_channels=None):
        super(ValueProjectionBlock, self).__init__()
        if proj_channels is None:
            proj_channels = in_channels  # identity projection if unspecified

        self.v_conv = nn.Conv3d(in_channels, proj_channels, kernel_size=1, bias=False)

    def forward(self, x):
        v = self.v_conv(x)  # shape: [B, C, D, H, W]
        return v


class GraphMessagePassing(nn.Module):
    """
    Performs message passing: S = V @ A
    Where:
    - V is [B, C, N] (value features)
    - A is [B, N, N] (adjacency matrix)
    """
    def __init__(self):
        super(GraphMessagePassing, self).__init__()

    def forward(self, V, A):
        B, C, D, H, W = V.shape
        N = H * W * D

        # Flatten spatial dimensions
        V_flat = V.reshape(B, C, N)  # [B, C, N]

        # Message passing: S = V @ A
        S = torch.bmm(V_flat, A)  # [B, C, N]

        # Reshape back to [B, C, D, H, W]
        S = S.reshape(B, C, D, H, W)

        return S


class SpatialGraphConvModule(nn.Module):
    def __init__(self, in_channels, proj_channels=None, out_channels=None):
        super(SpatialGraphConvModule, self).__init__()

        if proj_channels is None:
            proj_channels = in_channels
        if out_channels is None:
            out_channels = in_channels

        # Q & K projection
        self.qk_proj = QKProjectionBlock(in_channels, proj_channels)

        # V projection
        self.v_proj = ValueProjectionBlock(in_channels, proj_channels)

        # Graph attention (Adjacency computer)
        self.attn = AdjacencyComputer()

        # Message passing
        self.msg_passing = GraphMessagePassing()

        # Post-conv 1x1x1 for Si and Sj
        # using sequence of BN and LeakyReLU for enhanced activations
        # TODO: change to IN later on
        self.post_conv_i = nn.Sequential(
            nn.Conv3d(proj_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )
        self.post_conv_j = nn.Sequential(
            nn.Conv3d(proj_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, Xi, Xj):
        # Step 1: Q & K projections
        Qi, Ki = self.qk_proj(Xi)
        Qj, Kj = self.qk_proj(Xj)

        # Step 2: Adjacency matrices
        Aii = self.attn(Qi, Ki)  # self
        Aij = self.attn(Qi, Kj)  # cross
        Ai = Aii + Aij           # combined graph for Xi

        Ajj = self.attn(Qj, Kj)
        Aji = self.attn(Qj, Ki)
        Aj = Ajj + Aji           # combined graph for Xj

        # Step 3: Value projection
        Vi = self.v_proj(Xi)
        Vj = self.v_proj(Xj)

        # Step 4: Message passing
        Si = self.msg_passing(Vi, Ai)
        Sj = self.msg_passing(Vj, Aj)

        # add residual connection
        Si += Xi
        Sj += Xj

        # Step 5: 1x1x1 convs before fusion
        Si = self.post_conv_i(Si)
        Sj = self.post_conv_j(Sj)

        # Step 6: Element-wise summation
        S_ij = Si + Sj

        return S_ij