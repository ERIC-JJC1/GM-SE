# 位置: models/tagru.py
#
# 版本 5：真正的弱监督模型 (PGR 基础估计器)
# -----------------------------------------------------------
# 变化:
# 1. 匹配 train_tagru_weakly.py 脚本。
# 2. forward 接收 z_seq (原始量测) 和 feat_seq。
# 3. *不* 接收 x_wls_seq。
# 4. GNN 编码器只使用 feat_seq (或不使用)。
# 5. GRU 编码器使用 z_seq 和 feat_seq。
# 6. 解码器输出绝对状态 x_hat (va, vm)，而不是残差 dx。

import sys, os
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import from_scipy_sparse_matrix
from torch_geometric.nn import TAGConv

import sys, os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# [已删除] 移除了 sys.path 注入
# from .temporal_attention import TemporalBiasMHA
# 备注：为了简化，我们暂时移除 TemporalBiasMHA，如果需要，后续再加回来。
# 您 models/ 目录下的 temporal_attention.py 似乎未上传，我们先用标准 GRU。
from build_ieee33_with_pp import build_ieee33


# --- GNN 编码器 ---
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
        # x 形状: (B*W, N, F)
        B_W, N, F = x.shape
        x_flat = x.view(B_W * N, F)

        # 为批处理中的每个图复制边索引
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

        h_nodes = h.view(B_W, N, -1) # (B*W, N, hidden)
        return h_nodes

# --- 门控融合 ---
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

# --- 主模型 (弱监督版本) ---
class TopoAlignGRU(nn.Module):
    """
    TopoAlign-GRU (TA-GRU) - 版本 5 (弱监督)

    输入:
      z_seq    : (B, W, M)  # !! 原始量测 !!
      feat_seq : (B, W, F)
      ...
    输出:
      x_hat    : (B, W, 2N) # 最终估计 (va, vm)
    """
    def __init__(self, meas_dim: int, feat_dim: int, state_dim: int,
                 hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1,
                 nhead: int = 4, bias_scale: float = 4.0, use_mask: bool = True, # nhead 等参数暂时保留，为 TAttn 预留
                 gnn_hidden: int = 128, gnn_layers: int = 2, gnn_dropout: float = 0.0):
        super().__init__()
        self.state_dim = state_dim
        self.nbus = state_dim // 2

        # --- 1. 获取图拓扑结构 ---
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
        # GNN 输入只使用 feat_seq (节点级特征)
        self.gnn_feat_dim = feat_dim
        
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
        # GRU 输入是 z_seq (全局量测) + feat_seq (全局特征)
        self.temporal_input_dim = meas_dim + feat_dim

        self.temporal_encoder_pre = nn.Sequential(
            nn.Linear(self.temporal_input_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
        )
        
        # 暂时使用标准 GRU
        self.temporal_encoder = nn.GRU(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )

        # --- 3. 门控融合 ---
        self.fusion_module = GatedFusion(
            spatial_dim=hidden_dim,
            temporal_dim=hidden_dim,
            hidden_dim=hidden_dim
        )

        # --- 4. 解码器 (Decoder) ---
        # 输出绝对状态 (va, vm)
        self.decoder_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2) # 输出 (va, vm)
        )
        # nn.init.zeros_(self.decoder_head[-1].bias) # 预测绝对值时，不一定初始化为0

    def forward(self, z_seq, feat_seq, A_time=None, E_time=None):
        # z_seq: (B, W, M) - !! 原始量测 !!
        # feat_seq: (B, W, F)
        # x_wls_seq 不再是输入

        B, W, M = z_seq.shape
        N = self.nbus

        # --- 2a. 空间编码 (GNN) ---
        # GNN 输入只使用 feat_seq
        
        # 确保 feat_seq 是节点级别的 (B, W, N, F)
        if feat_seq.dim() == 3:
            # 如果 feat_seq 是 (B, W, F)，假设 F 是全局特征，广播到所有节点
            feat_seq_nodes = feat_seq.unsqueeze(2).expand(B, W, N, -1)
        else:
            # 否则，假设它已经是 (B, W, N, F)
            feat_seq_nodes = feat_seq

        gnn_input = feat_seq_nodes # (B, W, N, F)

        g_in = self.spatial_encoder_pre(gnn_input)
        g_in_flat = g_in.view(B*W, N, -1) # (B*W, N, gnn_hidden)
        g_out_flat = self.spatial_encoder(g_in_flat, self.edge_index, self.edge_weight)
        g_t_nodes = g_out_flat.view(B, W, N, -1)
        g_t_aligned_nodes = self.spatial_encoder_post(g_t_nodes) # (B, W, N, hidden_dim)

        # --- 2b. 时间编码 (GRU) ---
        # GRU 输入是 z_seq + feat_seq (假设 feat_seq 是全局特征)
        if feat_seq.dim() > 3:
            # 如果 feat_seq 是节点级 (B, W, N, F)，取均值作为全局特征
            feat_seq_global = feat_seq.mean(dim=2) # (B, W, F)
        else:
            feat_seq_global = feat_seq # (B, W, F)

        gru_input = torch.cat([z_seq, feat_seq_global], dim=-1) # (B, W, M+F)

        s_in = self.temporal_encoder_pre(gru_input)
        
        # s_att = self.tattn_pre(s_in, E_time, A_time) if E_time is not None else s_in
        # 暂时使用 s_in
        s_att = s_in
        
        s_t, _ = self.temporal_encoder(s_att) # (B, W, hidden_dim)

        # --- 3. 门控融合 (在节点级) ---
        s_t_expanded = s_t.unsqueeze(2).expand(B, W, N, -1)
        u_t_nodes = self.fusion_module(g_t_aligned_nodes, s_t_expanded) # (B, W, N, hidden_dim)

        # --- 4. 解码器 (输出 x_hat) ---
        x_hat_nodes = self.decoder_head(u_t_nodes) # (B, W, N, 2) -> (va, vm)

        va_hat = x_hat_nodes[..., 0]
        vm_hat = x_hat_nodes[..., 1]
        
        # [重要] 确保输出状态顺序匹配 'vm_va' (如 tools.metrics 中所定义)
        # 我们的弱监督损失 (PhysicsInformedLoss) 可能对顺序敏感
        # 我们在这里统一为 [vm, va]
        x_hat = torch.cat([vm_hat, va_hat], dim=-1) # (B, W, 2N) - 格式 [vm, va]

        return x_hat