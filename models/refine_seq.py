# models/refine_seq.py

import sys, os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
import numpy as np

from .temporal_attention import TemporalBiasMHA
from build_ieee33_with_pp import build_ieee33
from .gnn_blocks import GraphRefiner


class BusSmooth(nn.Module):
    def __init__(self, nbus, alpha_theta=0.05, alpha_vm=0.05):
        super().__init__()
        self.nbus = nbus
        self.alpha_th = nn.Parameter(torch.tensor(alpha_theta))
        self.alpha_vm = nn.Parameter(torch.tensor(alpha_vm))
        # 固定邻接（由 Ybus 构出 |Y|，行归一）
        _, ybus, *_ = build_ieee33()
        W = np.abs(ybus).astype(np.float32)
        np.fill_diagonal(W, 0.0)
        D = W.sum(axis=1, keepdims=True) + 1e-6
        self.register_buffer("Wrow", torch.from_numpy(W / D))  # 行归一

    def forward(self, x):  # x: (..., 2N)
        N = self.nbus
        th, vm = x[..., :N], x[..., N:]
        th_s = th + self.alpha_th * (th @ self.Wrow.T - th)
        vm_s = vm + self.alpha_vm * (vm @ self.Wrow.T - vm)
        return torch.cat([th_s, vm_s], dim=-1)


class RefineSeqTAModel(nn.Module):
    """
    结构:  前置 MLP -> TemporalBiasMHA -> GRU -> 残差MLP -> (BusSmooth) -> (GNN)
    输入:
      r      : (B, W, M)         # 建议用白化残差
      x_wls  : (B, W, 2N)
      feat   : (B, W, F)
      A_time : (B, W, W)
      E_time : (B, W, W)
    输出:
      x_hat  : (B, W, 2N)
    """
    def __init__(self, meas_dim: int, state_dim: int,
                 hidden: int = 256, num_layers: int = 2, dropout: float = 0.1,
                 nhead: int = 4, bias_scale: float = 4.0,
                 use_mask: bool = True, use_post_attn: bool = False,
                 feat_dim: int = 4,
                 use_bus_smooth: bool = True,
                 use_gnn: bool = False,                 # 新增：开关 GNN
                 gnn_type: str = "tag",                 # "tag" | "gcn2" | "gat"
                 gnn_hidden: int = 128,
                 gnn_layers: int = 2,
                 gnn_dropout: float = 0.0):
        super().__init__()
        self.state_dim = state_dim
        self.nbus = state_dim // 2
        self.use_post_attn = use_post_attn
        self.use_bus_smooth = use_bus_smooth
        self.use_gnn = use_gnn

        # (可选) BusSmooth
        if self.use_bus_smooth:
            self.bus_smooth = BusSmooth(nbus=self.nbus)

        # (可选) GraphRefiner (GNN)
        if self.use_gnn:
            _, ybus, *_ = build_ieee33()
            ybus = ybus.astype(np.complex128)
            self.graph_refiner = GraphRefiner(
                ybus=ybus, state_dim=state_dim,
                gnn_type=gnn_type, hidden=gnn_hidden,
                layers=gnn_layers, K=3, dropout=gnn_dropout
            )

        # 前置编码
        d_in = meas_dim + state_dim + feat_dim
        self.pre = nn.Sequential(
            nn.Linear(d_in, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
        )

        self.tattn_pre = TemporalBiasMHA(
            d_model=hidden, nhead=nhead, dropout=dropout,
            bias_scale=bias_scale, use_mask=use_mask
        )

        self.encoder = nn.GRU(
            input_size=hidden, hidden_size=hidden,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )

        if self.use_post_attn:
            self.tattn_post = TemporalBiasMHA(
                d_model=hidden, nhead=nhead, dropout=dropout,
                bias_scale=bias_scale, use_mask=use_mask
            )

        # 预测 Δx
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, state_dim)
        )

    def forward(self, r, x_wls, feat, A_time=None, E_time=None):
        x_in = torch.cat([r, x_wls, feat], dim=-1)   # (B, W, d_in)
        h0 = self.pre(x_in)                          # (B, W, H)
        h1 = self.tattn_pre(h0, E_time, A_time) if E_time is not None else h0
        h, _ = self.encoder(h1)
        if self.use_post_attn and (E_time is not None):
            h = self.tattn_post(h, E_time, A_time)

        dx = self.head(h)                            # (B, W, 2N)
        x_hat = x_wls + dx                           # 残差回到 WLS

        if self.use_bus_smooth:
            x_hat = self.bus_smooth(x_hat)           # 线性图平滑

        if self.use_gnn:
            x_hat = self.graph_refiner(x_hat)        # GNN 空间细化（自适配 2D/3D）

        return x_hat
