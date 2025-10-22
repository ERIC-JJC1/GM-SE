# models/gru_baseline.py
#
# 一个纯数据驱动的 GRU Baseline 模型。
# - 遵循残差精炼范式 (输入 r, x_wls, feat; 输出 dx)
# - 不使用 GNN 或任何图拓扑信息
# - 使用全监督 (StateLoss vs x_true) 进行训练

import torch
import torch.nn as nn

class GRUBaseline(nn.Module):
    """
    纯数据驱动的 GRU Baseline 模型
    
    输入:
      r_seq    : (B, W, M)  # 白化残差序列
      feat_seq : (B, W, F)
      x_wls_seq: (B, W, 2N)
    输出:
      x_hat    : (B, W, 2N) # 最终估计 (x_wls + dx)
    """
    def __init__(self, meas_dim: int, feat_dim: int, state_dim: int,
                 hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.state_dim = state_dim
        self.nbus = state_dim // 2

        # 输入是 r, x_wls, 和 feat 的拼接
        self.input_dim = meas_dim + state_dim + feat_dim

        # MLP 预处理器
        self.encoder_pre = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
        )

        # GRU 核心
        self.gru = nn.GRU(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        
        # MLP 解码器，输出修正量 dx
        self.decoder_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, state_dim) # 输出 dx
        )
        
        # 初始化最后一层 bias 为 0，使 dx 初始为 0
        nn.init.zeros_(self.decoder_head[-1].bias)

    def forward(self, r_seq, feat_seq, x_wls_seq, **kwargs):
        # **kwargs 用于接收 A_time, E_time 等额外参数，但本模型不使用它们
        
        B, W, _ = r_seq.shape

        # 1. 拼接所有输入特征
        gru_input = torch.cat([r_seq, feat_seq, x_wls_seq], dim=-1) # (B, W, M+F+2N)
        
        # 2. 通过 MLP 和 GRU
        h_in = self.encoder_pre(gru_input) # (B, W, H)
        h_out, _ = self.gru(h_in)          # (B, W, H)
        
        # 3. 解码器输出 dx
        dx = self.decoder_head(h_out)     # (B, W, 2N) - 格式 [d_va, d_vm]
        
        # 4. 最终步骤：残差连接
        x_hat = x_wls_seq + dx
        
        return x_hat