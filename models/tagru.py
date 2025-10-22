# models/tagru.py
#
# 版本 4：结合 借鉴2 (白化残差输入 r) 和 借鉴3 (残差精炼 dx)
# -----------------------------------------------------------
# 变化:
# 1. forward 接收 r_seq (白化残差) 而不是 z_seq
# 2. temporal_input_dim 只依赖 r_seq 和 x_wls_seq

import sys, os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.nn import TAGConv

from .temporal_attention import TemporalBiasMHA
from build_ieee33_with_pp import build_ieee33


# --- GNN 编码器 (无变化) ---
class TAGConvStack(nn.Module):
    def __init__(self, in_channels, hidden_channels, layers=2, K=3, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(TAGConv(in_channels, hidden_channels, K=K))
        for _ in range(layers - 2):
            self.layers.append(TAGConv(hidden_channels, hidden_channels, K=K))
        if layers > 1:
            self.layers.append(TAGConv(hidden_channels, hidden_channels, K=K))

        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x, edge_index, edge_weight=None):
        B_W, N, F = x.shape
        x_flat = x.view(B_W * N, F)

        edge_index_list = [edge_index + i * N for i in range(B_W)]
        batch_edge_index = torch.cat(edge_index_list, dim=1)

        batch_edge_weight = None
        if edge_weight is not None:
            edge_weight_list = [edge_weight for _ in range(B_W)]
            batch_edge_weight = torch.cat(edge_weight_list, dim=0)

        h = x_flat
        for i, layer in enumerate(self.layers):
            h = layer(h, batch_edge_index, batch_edge_weight)
            if i < len(self.layers) - 1:
                h = self.act(h)
                h = self.dropout(h)

        h_nodes = h.view(B_W, N, -1)
        return h_nodes

# --- 门控融合 (无变化) ---
class GatedFusion(nn.Module):
    def __init__(self, spatial_dim: int, temporal_dim: int, hidden_dim: int):
        super().__init__()
        self.align_g = nn.Linear(spatial_dim, hidden_dim)
        self.align_s = nn.Linear(temporal_dim, hidden_dim)
        self.gate_alpha = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, g_t, s_t):
        g_tilde = self.align_g(g_t)
        s_tilde = self.align_s(s_t)
        alpha_t = self.gate_alpha(torch.cat([g_tilde, s_tilde], dim=-1))
        u_t = alpha_t * g_tilde + (1 - alpha_t) * s_tilde
        return u_t

# --- 主模型 (重大修改) ---
class TopoAlignGRU(nn.Module):
    """
    TopoAlign-GRU (TA-GRU) - 版本 4 (残差精炼 + 白化残差输入)

    输入:
      r_seq    : (B, W, M)  # !! 白化残差序列 !!
      feat_seq : (B, W, F)
      x_wls_seq: (B, W, 2N)
      ...
    输出:
      x_hat    : (B, W, 2N) # 最终估计 (x_wls + dx)
    """
    def __init__(self, meas_dim: int, feat_dim: int, state_dim: int,
                 hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1,
                 nhead: int = 4, bias_scale: float = 4.0, use_mask: bool = True,
                 gnn_hidden: int = 128, gnn_layers: int = 2, gnn_dropout: float = 0.0):
        super().__init__()
        self.state_dim = state_dim
        self.nbus = state_dim // 2

        # --- 1. 获取图拓扑结构 (无变化) ---
        _, ybus, *_ = build_ieee33()
        adj = np.abs(ybus).astype(np.float32)
        np.fill_diagonal(adj, 0.0)
        from scipy.sparse import coo_matrix
        adj_coo = coo_matrix(adj)
        edge_index = torch.from_numpy(np.vstack((adj_coo.row, adj_coo.col))).long()
        edge_weight = torch.from_numpy(adj_coo.data).float()

        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)

        # --- 2a. 空间编码器 (GNN) ---
        # GNN 输入仍然是 feat + (va_wls, vm_wls)
        self.gnn_feat_dim = feat_dim + 2

        self.spatial_encoder_pre = nn.Linear(self.gnn_feat_dim, gnn_hidden)
        self.spatial_encoder = TAGConvStack(
            in_channels=gnn_hidden,
            hidden_channels=gnn_hidden,
            layers=gnn_layers,
            K=3,
            dropout=gnn_dropout
        )
        self.spatial_encoder_post = nn.Linear(gnn_hidden, hidden_dim)

        # --- 2b. 时间编码器 (GRU) ---
        # 变化：GRU 输入现在是 白化残差 r + 全局 WLS 状态 x_wls
        # meas_dim 现在代表白化残差 r 的维度 M
        self.temporal_input_dim = meas_dim + state_dim

        self.temporal_encoder_pre = nn.Sequential(
            nn.Linear(self.temporal_input_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
        )

        self.tattn_pre = TemporalBiasMHA(
            d_model=hidden_dim, nhead=nhead, dropout=dropout,
            bias_scale=bias_scale, use_mask=use_mask
        )

        self.temporal_encoder = nn.GRU(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )

        # --- 3. 门控融合 (无变化) ---
        self.fusion_module = GatedFusion(
            spatial_dim=hidden_dim,
            temporal_dim=hidden_dim,
            hidden_dim=hidden_dim
        )

        # --- 4. 解码器 (Decoder) ---
        # 输出 dx (d_va, d_vm)
        self.decoder_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2)
        )
        nn.init.zeros_(self.decoder_head[-1].bias)

    def forward(self, r_seq, feat_seq, x_wls_seq, A_time=None, E_time=None):
        # r_seq: (B, W, M) - !! 新输入: 白化残差 !!
        # feat_seq: (B, W, F)
        # x_wls_seq: (B, W, 2N)
        # z_seq 不再是输入

        B, W, M = r_seq.shape
        N = self.nbus

        # --- 2a. 空间编码 (GNN) ---
        # (这部分无变化, GNN 输入仍然是 feat + x_wls 节点状态)
        va_wls = x_wls_seq[..., :N].unsqueeze(-1)
        vm_wls = x_wls_seq[..., N:].unsqueeze(-1)
        x_wls_nodes = torch.cat([va_wls, vm_wls], dim=-1)

        if feat_seq.dim() == 3:
            feat_seq_nodes = feat_seq.unsqueeze(2).expand(B, W, N, -1)
        else:
            feat_seq_nodes = feat_seq

        gnn_input = torch.cat([feat_seq_nodes, x_wls_nodes], dim=-1)

        g_in = self.spatial_encoder_pre(gnn_input)
        g_in_flat = g_in.view(B*W, N, -1)
        g_out_flat = self.spatial_encoder(g_in_flat, self.edge_index, self.edge_weight)
        g_t_nodes = g_out_flat.view(B, W, N, -1)
        g_t_aligned_nodes = self.spatial_encoder_post(g_t_nodes)

        # --- 2b. 时间编码 (GRU) ---
        # 变化：GRU 输入现在是 白化残差 r + 全局 WLS 状态 x_wls
        gru_input = torch.cat([r_seq, x_wls_seq], dim=-1) # (B, W, M+2N)

        s_in = self.temporal_encoder_pre(gru_input)
        s_att = self.tattn_pre(s_in, E_time, A_time) if E_time is not None else s_in
        s_t, _ = self.temporal_encoder(s_att)

        # --- 3. 门控融合 (在节点级) ---
        s_t_expanded = s_t.unsqueeze(2).expand(B, W, N, -1)
        u_t_nodes = self.fusion_module(g_t_aligned_nodes, s_t_expanded)

        # --- 4. 解码器 (输出 dx) ---
        dx_nodes = self.decoder_head(u_t_nodes) # (B, W, N, 2) -> (d_va, d_vm)

        d_va = dx_nodes[..., 0]
        d_vm = dx_nodes[..., 1]
        dx = torch.cat([d_va, d_vm], dim=-1) # (B, W, 2N) - 格式 [d_va, d_vm]

        # --- 5. 最终步骤：残差连接 ---
        x_hat = x_wls_seq + dx

        return x_hat