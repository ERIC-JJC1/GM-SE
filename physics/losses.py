# physics/losses.py
#
# 实现了论文中的 (Eq. 7) 物理信息弱监督损失函数。
# 这个损失函数 *不* 依赖 x_true，只依赖 x_hat 和 z。

import torch
import torch.nn as nn
import numpy as np

# 从我们刚创建的文件中导入可微分的 h(x) 函数
from .torch_ac_model import get_torch_net_params, h_measure_torch, unpack_state

class PhysicsInformedLoss(nn.Module):
    """
    实现论文 (Eq. 7) 中的复合损失函数
    L = L_residual + lambda_op * L_op + ...
    """
    def __init__(self, nbus: int, slack_pos: int, 
                 lambda_op: float = 1.0, 
                 vm_min: float = 0.95, 
                 vm_max: float = 1.05):
        """
        初始化物理损失模块
        
        参数:
          nbus: int, 总母线数 (e.g., 33)
          slack_pos: int, Slack 母线的 0-based 索引
          lambda_op: float, 运行约束的权重 (论文中的 lambda_op)
          vm_min: float, P.U. 电压下限
          vm_max: float, P.U. 电压上限
        """
        super().__init__()
        self.nbus = nbus
        self.slack_pos = slack_pos
        self.lambda_op = lambda_op
        self.vm_min = vm_min
        self.vm_max = vm_max
        
        # 1. 定义鲁棒核 (Robust Kernel)
        # 论文中提到使用 rho()，例如 Huber loss
        self.huber_loss = nn.HuberLoss(reduction='mean', delta=1.0) # delta 是 Huber loss 的阈值
        
        # 2. 缓存网络参数 (Ybus, baseMVA)
        # 我们假设模型在单个设备上运行
        self.G_torch, self.B_torch, self.baseMVA = None, None, None
        
    def _init_params(self, device):
        """在第一次 forward call 时初始化 Ybus"""
        if self.G_torch is None:
            self.G_torch, self.B_torch, self.baseMVA = get_torch_net_params(device)

    def forward(self, x_hat, batch):
        """
        计算总的物理损失
        
        输入:
          x_hat : (B, W, 2N) - 模型的估计状态 (来自 TopoAlignGRU)
          batch : dict - 包含 z, R, ztype 的数据批次
          
        输出:
          total_loss: torch.Tensor - 可反向传播的总损失
          loss_dict : dict - 用于日志记录的子损失
        """
        
        # 0. 确保网络参数已加载到正确设备
        self._init_params(x_hat.device)
        
        # 1. =======================================================
        #    论文 Eq. 7 - 项 1: 鲁棒测量残差
        #    L_residual = rho( (z - h(x_hat))^T * R^-1 * (z - h(x_hat)) )
        # ==========================================================
        
        z_meas = batch["z"]     # (B, W, M) - 真实量测
        R_cov = batch["R"]      # (B, W, M) - 量测方差 (R 是对角线)
        ztype = batch["ztype"]  # (B, W, 4, M) - Numpy array
        
        # 1a. 计算 h(x_hat)
        h_hat = h_measure_torch(
            x_hat, ztype, 
            self.baseMVA, self.slack_pos, 
            self.G_torch, self.B_torch
        )
        
        # 1b. 计算带权残差 (Whitened Residual)
        # r = z - h(x_hat)
        r = z_meas - h_hat
        
        # R^-1 的对角线是 1/sigma^2。R^-0.5 的对角线是 1/sigma。
        # R_cov 存储的是方差 (sigma^2)
        w = 1.0 / torch.sqrt(R_cov + 1e-9) # w = R_t^(-1/2)
        
        r_w = r * w  # (B, W, M) - 带权残差 r_w = R_t^(-1/2) * r
        
        # 1c. 应用鲁棒核
        # 我们计算 rho(r_w) - 0
        loss_residual = self.huber_loss(r_w, torch.zeros_like(r_w))
        
        
        # 2. =======================================================
        #    论文 Eq. 7 - 项 3: 运行约束惩罚 (Phi_op)
        #    L_op = hinge_loss(Vm) + ...
        # ==========================================================
        
        vm, va = unpack_state(x_hat, self.nbus) # (B, W, N)
        
        # 2a. 电压幅值约束 (Hinge Loss)
        # torch.relu(x) 等价于 max(0, x)，是 Hinge Loss 的一种形式
        loss_op_vm_low = torch.relu(self.vm_min - vm).mean()
        loss_op_vm_high = torch.relu(vm - self.vm_max).mean()
        
        loss_op = loss_op_vm_low + loss_op_vm_high
        
        # (未来可以添加: 论文中的相角差约束 Phi_op_angle)
        
        
        # 3. =======================================================
        #    论文 Eq. 7 - 项 2 (Phi_pf) & 4 (Smooth)
        # ==========================================================
        # 备注：
        # L_pf (Phi_pf): 论文中的"功率平衡"惩罚。对于具有P/Q量测的节点，
        # 这已经包含在 L_residual 中了。对于没有量测的节点，
        # 它可以是 P_inj 和 Q_inj 的 L2 惩罚。
        #
        # L_smooth: 拓扑平滑项。
        #
        # 为保持第一版简洁，我们暂时只实现 L_residual 和 L_op
        
        loss_pf = torch.tensor(0.0, device=x_hat.device)
        loss_smooth = torch.tensor(0.0, device=x_hat.device)
        
        
        # 4. =======================================================
        #    总损失
        # ==========================================================
        
        total_loss = loss_residual + self.lambda_op * loss_op
        # (未来添加: + self.lambda_pf * loss_pf + self.lambda_smooth * loss_smooth)
        
        
        loss_dict = {
            "total_loss": total_loss,
            "loss_residual": loss_residual.detach(),
            "loss_op": loss_op.detach()
        }
        
        return total_loss, loss_dict