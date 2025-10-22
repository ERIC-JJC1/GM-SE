# physics/losses.py
#
# 版本 2: 加入 Phi_pf (功率平衡惩罚) 和 L_smooth (空间平滑惩罚)

import torch
import torch.nn as nn
import numpy as np

from .torch_ac_model import get_torch_net_params, h_measure_torch, unpack_state
from build_ieee33_with_pp import build_ieee33

def calculate_admittance_laplacian(ybus, device):
    """计算导纳加权的图拉普拉斯 L = D - W"""
    adj = np.abs(ybus)
    np.fill_diagonal(adj, 0) # W: 权重矩阵 (邻接矩阵)
    D = np.diag(np.sum(adj, axis=1)) # D: 度矩阵
    L = D - adj # L: 拉普拉斯矩阵
    return torch.from_numpy(L).float().to(device)

class PhysicsInformedLoss(nn.Module):
    """
    实现论文 (Eq. 7) 中的复合损失函数 (完整版)
    L = L_residual + lambda_pf * L_pf + lambda_op * L_op + lambda_smooth * L_smooth
    """
    def __init__(self, nbus: int, slack_pos: int,
                 lambda_pf: float = 0.1, # 新增：功率平衡权重
                 lambda_op: float = 1.0,
                 lambda_smooth: float = 0.01, # 新增：空间平滑权重
                 vm_min: float = 0.95,
                 vm_max: float = 1.05):
        super().__init__()
        self.nbus = nbus
        self.slack_pos = slack_pos
        self.lambda_pf = lambda_pf # 新增
        self.lambda_op = lambda_op
        self.lambda_smooth = lambda_smooth # 新增
        self.vm_min = vm_min
        self.vm_max = vm_max

        self.huber_loss = nn.HuberLoss(reduction='mean', delta=1.0)
        self.mse_loss = nn.MSELoss(reduction='mean') # 用于 L_pf

        self.G_torch, self.B_torch, self.baseMVA = None, None, None
        self.L_torch = None # 图拉普拉斯

    def _init_params(self, device):
        """在第一次 forward call 时初始化 Ybus 和 Laplacian"""
        if self.G_torch is None:
            self.G_torch, self.B_torch, self.baseMVA = get_torch_net_params(device)
            # 计算拉普拉斯矩阵
            _, ybus_np, _, _, _, _ = build_ieee33()
            self.L_torch = calculate_admittance_laplacian(ybus_np, device)

    def forward(self, x_hat, batch):
        """
        计算总的物理损失 (完整版)
        """
        self._init_params(x_hat.device)

        z_meas = batch["z"]
        R_cov = batch["R"]
        ztype = batch["ztype"]

        # 1. ================= L_residual (鲁棒测量残差) =================
        # 调用更新后的 h_measure_torch，获取 P_inj, Q_inj
        h_hat, P_inj_hat, Q_inj_hat = h_measure_torch(
            x_hat, ztype,
            self.baseMVA, self.slack_pos,
            self.G_torch, self.B_torch
        )
        r = z_meas - h_hat
        w = 1.0 / torch.sqrt(R_cov + 1e-9)
        r_w = r * w
        loss_residual = self.huber_loss(r_w, torch.zeros_like(r_w))

        # 2. ================= L_pf (功率平衡惩罚) =======================
        # 惩罚所有节点的注入功率接近于 0 (假设没有发电机，负荷已通过量测z体现)
        # 注意：这是一个简化假设，实际应用中可能需要区分 PQ/PV/Slack 节点
        loss_pf_p = self.mse_loss(P_inj_hat, torch.zeros_like(P_inj_hat))
        loss_pf_q = self.mse_loss(Q_inj_hat, torch.zeros_like(Q_inj_hat))
        loss_pf = loss_pf_p + loss_pf_q

        # 3. ================= L_op (运行约束惩罚) =======================
        vm, va = unpack_state(x_hat, self.nbus)
        loss_op_vm_low = torch.relu(self.vm_min - vm).mean()
        loss_op_vm_high = torch.relu(vm - self.vm_max).mean()
        loss_op = loss_op_vm_low + loss_op_vm_high

        # 4. ================= L_smooth (空间平滑惩罚) ===================
        # 计算 va^T L va
        # va: (B, W, N) -> (B*W, N) for batch matmul
        B, W, N = va.shape
        va_flat = va.view(B*W, N)
        # 需要计算 quadratic form: x^T A x = sum(x * (A @ x)) element-wise
        # 或者使用爱因斯坦求和: torch.einsum('bi,ij,bj->b', ...)
        # 为了简单起见，我们计算 L2 norm of L @ va
        # loss_smooth = torch.mean(va_flat @ self.L_torch @ va_flat.T) # 这样计算维度不对
        
        # 正确计算 x^T L x (逐批次)
        # L @ va^T -> (N, N) @ (N, B*W) -> (N, B*W)
        L_vaT = self.L_torch @ va_flat.T
        # va @ (L @ va^T) -> (B*W, N) @ (N, B*W) -> (B*W, B*W) ? 不对
        
        # 使用 einsum 计算每个样本的 quadratic form
        # va_b: (N,), L: (N, N) -> va_b @ L @ va_b
        # L_torch 已经是 (N, N)
        # va 是 (B, W, N)
        # 我们需要在 B*W 维度上迭代或使用 einsum
        
        # 尝试 einsum: 'bwn, nm, bwm -> bw' ?
        # 'bwn' (va), 'nm' (L), 'bwm' (va)
        # 结果应该是 (B, W)
        # loss_smooth_bw = torch.einsum('bwn, nm, bwn -> bw', va, self.L_torch, va) # einsum 可能较慢
        
        # 或者直接计算： sum_{i,j} L_ij * (va_i - va_j)^2
        # 我们用一个简化的形式：惩罚 L@va 的 L2 范数
        L_va = va @ self.L_torch.T # (B, W, N) @ (N, N) -> (B, W, N)
        loss_smooth = self.mse_loss(L_va, torch.zeros_like(L_va))

        # 5. ================= 总损失 ================================
        total_loss = (loss_residual
                      + self.lambda_pf * loss_pf
                      + self.lambda_op * loss_op
                      + self.lambda_smooth * loss_smooth)

        loss_dict = {
            "total_loss": total_loss,
            "loss_residual": loss_residual.detach(),
            "loss_pf": loss_pf.detach(),
            "loss_op": loss_op.detach(),
            "loss_smooth": loss_smooth.detach(),
        }

        return total_loss, loss_dict