# 位置: models/pgr.py
#
# 我们的核心创新模型：Physically-Guided Refiner (PGR)
# 结合了 tagru (弱监督) 和 refine-wls (残差学习) 的优点。

import torch
import torch.nn as nn
import numpy as np

# --- 关键导入 ---
# 1. 导入 BaseEstimator
from models.tagru import TopoAlignGRU

# 2. 导入 BaseEstimator 和 RefinerNet 所需的 *构件*
#    (我们假设 GNN 堆栈和门控融合在 tagru.py 中定义，如 Step 1.5 所示)
try:
    from models.tagru import TAGConvStack, GatedFusion
except ImportError:
    print("错误：PGR 模型依赖于 models.tagru.py 中的 TAGConvStack 和 GatedFusion。")
    print("请确保它们在 models/tagru.py 中已定义 (如 Step 1.5 所示)。")
    raise

# 3. 导入 RefinerNet 所需的拓扑
from build_ieee33_with_pp import build_ieee33


# --- RefinerNet (PGR 的阶段 2) ---
class RefinerNet(nn.Module):
    """
    PGR 的第二阶段：残差精炼网络。
    结构与 TopoAlignGRU 类似，但输入不同。
    
    输入:
      state_base: (B, W, 2N) - 来自 BaseEstimator 的粗略估计
      z_seq:      (B, W, M)  - 原始量测
      feat_seq:   (B, W, F)  - 特征
    输出:
      state_residual: (B, W, 2N) - 状态残差
    """
    def __init__(self, meas_dim: int, feat_dim: int, state_dim: int,
                 hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1,
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
        # GNN 输入是 state_base 的节点级分解 + 节点特征
        self.gnn_feat_dim = feat_dim + 2 # (feat, vm_base, va_base)
        
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
        # GRU 输入是 z_seq (全局量测) + state_base (全局状态) + feat_seq (全局特征)
        self.temporal_input_dim = meas_dim + state_dim + feat_dim

        self.temporal_encoder_pre = nn.Sequential(
            nn.Linear(self.temporal_input_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
        )
        
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
        # 输出残差 (d_vm, d_va)
        self.decoder_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2) # 输出 (d_vm, d_va)
        )
        # 残差解码器应初始化为 0，以确保初始 state_final = state_base
        nn.init.zeros_(self.decoder_head[-1].weight)
        nn.init.zeros_(self.decoder_head[-1].bias)

    def forward(self, state_base, z_seq, feat_seq, A_time=None, E_time=None):
        # state_base: (B, W, 2N) - !! 新输入 !!
        # z_seq: (B, W, M)
        # feat_seq: (B, W, F)

        B, W, _ = z_seq.shape
        N = self.nbus

        # --- 2a. 空间编码 (GNN) ---
        # GNN 输入: (feat, vm_base, va_base)
        vm_base = state_base[..., :N].unsqueeze(-1) # (B, W, N, 1)
        va_base = state_base[..., N:].unsqueeze(-1) # (B, W, N, 1)

        if feat_seq.dim() == 3:
            feat_seq_nodes = feat_seq.unsqueeze(2).expand(B, W, N, -1)
        else:
            feat_seq_nodes = feat_seq

        gnn_input = torch.cat([feat_seq_nodes, vm_base, va_base], dim=-1) # (B, W, N, F+2)

        g_in = self.spatial_encoder_pre(gnn_input)
        g_in_flat = g_in.view(B*W, N, -1)
        g_out_flat = self.spatial_encoder(g_in_flat, self.edge_index, self.edge_weight)
        g_t_nodes = g_out_flat.view(B, W, N, -1)
        g_t_aligned_nodes = self.spatial_encoder_post(g_t_nodes) # (B, W, N, hidden_dim)

        # --- 2b. 时间编码 (GRU) ---
        # GRU 输入: (z_seq, state_base, feat_global)
        if feat_seq.dim() > 3:
            feat_seq_global = feat_seq.mean(dim=2)
        else:
            feat_seq_global = feat_seq

        gru_input = torch.cat([z_seq, state_base, feat_seq_global], dim=-1) # (B, W, M+2N+F)

        s_in = self.temporal_encoder_pre(gru_input)
        # 暂时不使用 TAttn
        s_att = s_in
        s_t, _ = self.temporal_encoder(s_att) # (B, W, hidden_dim)

        # --- 3. 门控融合 ---
        s_t_expanded = s_t.unsqueeze(2).expand(B, W, N, -1)
        u_t_nodes = self.fusion_module(g_t_aligned_nodes, s_t_expanded) # (B, W, N, hidden_dim)

        # --- 4. 解码器 (输出残差) ---
        residual_nodes = self.decoder_head(u_t_nodes) # (B, W, N, 2) -> (d_vm, d_va)

        d_vm = residual_nodes[..., 0]
        d_va = residual_nodes[..., 1]
        
        # 保持 [vm, va] 顺序
        state_residual = torch.cat([d_vm, d_va], dim=-1) # (B, W, 2N)

        return state_residual


# --- PGR 主模型 (PGR) ---
class PGR(nn.Module):
    """
    Physically-Guided Refiner (PGR)
    
    两阶段模型，端到端训练：
    1. BaseEstimator (TopoAlignGRU): z -> state_base
    2. RefinerNet (RefinerNet): (state_base, z) -> state_residual
    
    输出:
      state_final = state_base + state_residual
    """
    def __init__(self, cfg_base, cfg_refiner):
        """
        cfg_base: BaseEstimator (TopoAlignGRU) 的配置
        cfg_refiner: RefinerNet 的配置
        
        配置示例:
        cfg = {
            'meas_dim': M, 
            'feat_dim': F, 
            'state_dim': S,
            'hidden_dim': 256, 
            'num_layers': 2, 
            'gnn_hidden': 128, 
            'gnn_layers': 2
        }
        """
        super(PGR, self).__init__()
        
        # 阶段一：基础估计器 (弱监督)
        self.base_estimator = TopoAlignGRU(
            meas_dim=cfg_base['meas_dim'],
            feat_dim=cfg_base['feat_dim'],
            state_dim=cfg_base['state_dim'],
            hidden_dim=cfg_base.get('hidden_dim', 256),
            num_layers=cfg_base.get('num_layers', 2),
            gnn_hidden=cfg_base.get('gnn_hidden', 128),
            gnn_layers=cfg_base.get('gnn_layers', 2)
        )
        
        # 阶段二：残差精炼器 (全监督)
        self.refiner_net = RefinerNet(
            meas_dim=cfg_refiner['meas_dim'],
            feat_dim=cfg_refiner['feat_dim'],
            state_dim=cfg_refiner['state_dim'],
            hidden_dim=cfg_refiner.get('hidden_dim', 256),
            num_layers=cfg_refiner.get('num_layers', 2),
            gnn_hidden=cfg_refiner.get('gnn_hidden', 128),
            gnn_layers=cfg_refiner.get('gnn_layers', 2)
        )
        
    def forward(self, z_seq, feat_seq, A_time=None, E_time=None):
        """
        z_seq: (B, W, M) 量测
        feat_seq: (B, W, F) 或 (B, W, N, F) 特征
        (A_time, E_time 暂时未使用)
        """
        
        # --- 阶段一：基础估计 ---
        # 冻结梯度，防止监督损失污染基础估计器 (可选，但推荐)
        # state_base = self.base_estimator(z_seq, feat_seq, A_time, E_time).detach()
        
        # (不冻结梯度，允许端到端训练，但依赖损失权重)
        state_base = self.base_estimator(z_seq, feat_seq, A_time, E_time)
        
        
        # --- 阶段二：残差精炼 ---
        state_residual = self.refiner_net(state_base, z_seq, feat_seq, A_time, E_time)
        
        # --- 最终估计 ---
        state_final = state_base + state_residual
        
        # 返回两个阶段的输出，用于计算混合损失
        # state_base -> 物理损失
        # state_final -> 监督损失
        return state_base, state_final