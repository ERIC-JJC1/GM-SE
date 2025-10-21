# models/tagru.py
# 这是根据您的论文描述 "TopoAlign-GRU" 构建的新模型文件。
# 版本 2：修复了 'TAGConvStack' 未定义的 Import Error

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
# 导入我们需要的 GNN 基础层
from torch_geometric.nn import TAGConv 

# 从现有代码中重用关键组件
from .temporal_attention import TemporalBiasMHA
# 变化：我们不再从 gnn_blocks 导入，而是在下面本地定义 TAGConvStack
# from .gnn_blocks import TAGConvStack 
from build_ieee33_with_pp import build_ieee33


# ================== 修复：定义缺失的 TAGConvStack ==================
# 这个模块是我们需要的 GNN 空间编码器
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
        # x 形状: (B*W, N, F_in)
        
        # PyG 的 GNN 层期望 (num_nodes, features)
        # 我们需要将其展平并通过循环或批处理来处理
        
        B_W, N, F = x.shape
        x_flat = x.view(B_W * N, F) # (B*W*N, F_in)
        
        # 创建一个适用于 (B*W, N) 图的 edge_index
        # 这是通过偏移 edge_index 来实现的
        edge_index_list = [edge_index + i * N for i in range(B_W)]
        batch_edge_index = torch.cat(edge_index_list, dim=1)
        
        # (如果存在 edge_weight，也需要同样处理)
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
        
        h_nodes = h.view(B_W, N, -1) # (B*W, N, F_out)
        return h_nodes
# =================================================================


class GatedFusion(nn.Module):
    """
    实现论文中的 Eq. (6) Gated Fusion 机制
    u_t = alpha_t * g_t + (1 - alpha_t) * s_t
    """
    def __init__(self, spatial_dim: int, temporal_dim: int, hidden_dim: int):
        super().__init__()
        self.align_g = nn.Linear(spatial_dim, hidden_dim)
        self.align_s = nn.Linear(temporal_dim, hidden_dim)
        self.gate_alpha = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
    def forward(self, g_t, s_t):
        # 1. 对齐维度
        g_tilde = self.align_g(g_t)  # (..., H)
        s_tilde = self.align_s(s_t)  # (..., H)
        
        # 2. 计算门控 alpha_t
        alpha_t = self.gate_alpha(torch.cat([g_tilde, s_tilde], dim=-1)) # (..., H)
        
        # 3. 融合
        u_t = alpha_t * g_tilde + (1 - alpha_t) * s_tilde
        return u_t


class TopoAlignGRU(nn.Module):
    """
    论文中描述的 TopoAlign-GRU (TA-GRU) 模型
    架构:
    1.  (未来添加: 异步模块)
    2.  并行的时空编码器:
        a.  GNN 空间编码器 (处理节点特征 feat_seq)
        b.  GRU 时间编码器 (处理量测 z_seq)
    3.  门控融合 (Gated Fusion)
    4.  轻量级解码器 (Decoder)
    
    输入:
      z_seq    : (B, W, M)  # 量测序列 (z)
      feat_seq : (B, W, F)  # 节点特征序列 (X_t)
      A_time   : (B, W, W)
      E_time   : (B, W, W)
    输出:
      x_hat    : (B, W, 2N) # ！！直接估计的状态！！
    """
    def __init__(self, meas_dim: int, feat_dim: int, state_dim: int,
                 hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1,
                 nhead: int = 4, bias_scale: float = 4.0, use_mask: bool = True,
                 gnn_hidden: int = 128, gnn_layers: int = 2, gnn_dropout: float = 0.0):
        super().__init__()
        self.state_dim = state_dim
        self.nbus = state_dim // 2
        
        # --- 1. 获取图拓扑结构 (来自您的 build_ieee33.py) ---
        _, ybus, *_ = build_ieee33()
        adj = np.abs(ybus).astype(np.float32)
        np.fill_diagonal(adj, 0.0)
        # 变化：我们从稀疏矩阵创建 edge_index 和 edge_weight
        from scipy.sparse import coo_matrix
        adj_coo = coo_matrix(adj)
        edge_index = torch.from_numpy(np.vstack((adj_coo.row, adj_coo.col))).long()
        edge_weight = torch.from_numpy(adj_coo.data).float()
        
        self.register_buffer("edge_index", edge_index)
        self.register_buffer("edge_weight", edge_weight)

        # --- 2a. 空间编码器 (GNN) ---
        # 变化：现在使用我们上面定义的 TAGConvStack
        self.spatial_encoder_pre = nn.Linear(feat_dim, gnn_hidden)
        self.spatial_encoder = TAGConvStack(
            in_channels=gnn_hidden,
            hidden_channels=gnn_hidden,
            layers=gnn_layers,
            K=3,
            dropout=gnn_dropout
        )
        self.spatial_encoder_post = nn.Linear(gnn_hidden, hidden_dim)

        # --- 2b. 时间编码器 (GRU) ---
        self.temporal_encoder_pre = nn.Sequential(
            nn.Linear(meas_dim, hidden_dim), nn.ReLU(inplace=True),
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

        # --- 3. 门控融合 (Gated Fusion) ---
        # 变化：这是一个新模块，用于融合时空特征
        # GNN 输出 (B,W,N,H), GRU 输出 (B,W,H)
        # 我们将在节点级进行融合
        self.fusion_module = GatedFusion(
            spatial_dim=hidden_dim,  # GNN 节点输出
            temporal_dim=hidden_dim, # GRU 广播后的输出
            hidden_dim=hidden_dim
        )
        
        # --- 4. 解码器 (Decoder) ---
        # 变化：解码器在 *每个节点* 上运行
        # (B, W, N, H) -> (B, W, N, 2)  (2 = Vm, Va)
        self.decoder_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2) # ！！直接预测 (Vm, Va) 
        )

    def forward(self, z_seq, feat_seq, A_time=None, E_time=None):
        # z_seq: (B, W, M) - 论文中的 z_tilde (暂时假设 M = M)
        # feat_seq: (B, W, F) - 论文中的 X_t (暂时假设 F = F)
        
        B, W, M = z_seq.shape
        _B, _W, F = feat_seq.shape
        N = self.nbus

        # --- 2a. 空间编码 (GNN) ---
        # GNN 需要 (B*W, N, F_in)
        
        # 假设 feat_seq 包含的是 *全局* 特征, 形状 (B, W, F)
        # 我们需要将其广播到 (B, W, N, F)
        if feat_seq.dim() == 3:
            feat_seq_nodes = feat_seq.unsqueeze(2).expand(B, W, N, F)
        elif feat_seq.dim() == 4:
             feat_seq_nodes = feat_seq # 假设 (B, W, N, F)
        else:
            raise ValueError(f"feat_seq 维度不正确: {feat_seq.shape}")
            
        g_in = self.spatial_encoder_pre(feat_seq_nodes) # (B, W, N, G_H)
        
        g_in_flat = g_in.view(B*W, N, -1) # (B*W, N, G_H)
        g_out_flat = self.spatial_encoder(g_in_flat, self.edge_index, self.edge_weight) # (B*W, N, G_H)
        g_t_nodes = g_out_flat.view(B, W, N, -1) # (B, W, N, G_H)
        
        # GNN 特征后处理 -> (B, W, N, H)
        g_t_aligned_nodes = self.spatial_encoder_post(g_t_nodes) # (B, W, N, H)

        # --- 2b. 时间编码 (GRU) ---
        s_in = self.temporal_encoder_pre(z_seq)       # (B, W, H)
        s_att = self.tattn_pre(s_in, E_time, A_time) if E_time is not None else s_in
        s_t, _ = self.temporal_encoder(s_att)         # (B, W, H)
        
        # --- 3. 门控融合 (在节点级) ---
        # 策略 2：在节点级融合 (更符合论文思想)
        # 将 GRU 的 s_t (B,W,H) 广播回 (B,W,N,H)
        s_t_expanded = s_t.unsqueeze(2).expand(B, W, N, -1) # (B, W, N, H)
        
        # 融合
        u_t_nodes = self.fusion_module(g_t_aligned_nodes, s_t_expanded) # (B, W, N, H)
        
        # --- 4. 解码器 ---
        # 解码器在每个节点上运行
        # (B, W, N, H) -> (B, W, N, 2)
        x_hat_nodes = self.decoder_head(u_t_nodes) # (B, W, N, 2)
        
        # 将 (V_mag, V_ang) 重新组合为 state_dim (2N)
        # [|V1|...|Vn|, Ang1...AngN]
        vm_hat = x_hat_nodes[..., 0] # (B, W, N)
        va_hat = x_hat_nodes[..., 1] # (B, W, N)
        
        # 变化：统一为 [theta, vm] 格式，与 x_true/x_wls 保持一致
        x_hat = torch.cat([va_hat, vm_hat], dim=-1) # (B, W, 2N)
        
        return x_hat