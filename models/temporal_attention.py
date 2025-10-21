# models/temporal_attention.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalBiasMHA(nn.Module):
    """
    多头时序自注意力，支持外部 bias（来自 E_time）和可选的硬掩码（A_time）。
    输入:  x      (B, W, M)
           E_time (B, W, W)  连续相似度[-1,1]或[0,1] -> 转为logits bias
           A_time (B, W, W)  二值邻接(0/1)，可选
    输出:  y      (B, W, M)
    """
    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.1,
                 bias_scale: float = 4.0, use_mask: bool = True):
        super().__init__()
        assert d_model % nhead == 0, "d_model必须能被nhead整除"
        self.d_model = d_model
        self.nhead = nhead
        self.dk = d_model // nhead
        self.bias_scale = bias_scale
        self.use_mask = use_mask

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x, E_time, A_time=None):
        B, W, M = x.shape
        residual = x

        q = self.q_proj(x).view(B, W, self.nhead, self.dk).transpose(1, 2)  # (B, H, W, dk)
        k = self.k_proj(x).view(B, W, self.nhead, self.dk).transpose(1, 2)  # (B, H, W, dk)
        v = self.v_proj(x).view(B, W, self.nhead, self.dk).transpose(1, 2)  # (B, H, W, dk)

        # 注意力 logits
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dk)  # (B, H, W, W)

        # 外部 bias：把 E_time 线性缩放后加到 logits
        # 预期 E_time ∈ [-1,1] 或 [0,1]；做个归一：先clip到[-1,1]，再乘 bias_scale
        bias = torch.clamp(E_time, -1.0, 1.0) * self.bias_scale
        bias = bias.unsqueeze(1).expand(-1, self.nhead, -1, -1)              # (B, H, W, W)
        attn_logits = attn_logits + bias

        # 可选掩码：A_time==0 的位置赋 -inf，强制不关注
        if self.use_mask and (A_time is not None):
            mask = (A_time <= 0)  # True -> 屏蔽
            mask = mask.unsqueeze(1).expand(-1, self.nhead, -1, -1)          # (B,H,W,W)
            attn_logits = attn_logits.masked_fill(mask, float('-inf'))

        attn = F.softmax(attn_logits, dim=-1)  # (B, H, W, W)
        attn = self.dropout(attn)
        y = torch.matmul(attn, v)              # (B, H, W, dk)
        y = y.transpose(1, 2).contiguous().view(B, W, M)
        y = self.o_proj(y)
        y = self.ln(y + residual)              # Residual + LN
        return y
