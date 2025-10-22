# 位置: tools/metrics.py
import torch
import numpy as np
import math

def theta_wrap(pred_rad, gt_rad):
    """
    计算角度的最小弧度差 (pred - gt), 范围 [-pi, pi].
    支持 torch.Tensor 和 np.ndarray
    """
    if isinstance(pred_rad, torch.Tensor):
        return torch.remainder(pred_rad - gt_rad + math.pi, 2 * math.pi) - math.pi
    else:
        # 假设为 numpy
        return (pred_rad - gt_rad + math.pi) % (2 * math.pi) - math.pi

def rmse_metrics(x_hat, x_true, state_order='vm_va'):
    """
    使用 Numpy 计算 RMSE (仅用于评估，不可微分)
    x_hat, x_true: (..., 2*N)
    
    参数:
        state_order (str): 状态向量中 Vm 和 Va 的顺序。
            'vm_va': 电压在前 N, 角度在后 N (用于 tagru)
            'va_vm': 角度在前 N, 电压在后 N (用于 refine-wls)
    """
    # 确保在 CPU 和 Numpy 上
    x_hat_np = x_hat.detach().cpu().numpy() if isinstance(x_hat, torch.Tensor) else np.array(x_hat)
    x_true_np = x_true.detach().cpu().numpy() if isinstance(x_true, torch.Tensor) else np.array(x_true)
    
    N = x_true_np.shape[-1] // 2
    
    if state_order == 'va_vm':
        va_hat, vm_hat = x_hat_np[..., :N], x_hat_np[..., N:]
        va_true, vm_true = x_true_np[..., :N], x_true_np[..., N:]
    elif state_order == 'vm_va':
        vm_hat, va_hat = x_hat_np[..., :N], x_hat_np[..., N:]
        vm_true, va_true = x_true_np[..., :N], x_true_np[..., N:]
    else:
        raise ValueError(f"Unknown state_order: {state_order}")

    dth_rad = theta_wrap(va_hat, va_true)
    dv      = vm_hat - vm_true
    
    th_rmse_deg = np.sqrt(np.mean(dth_rad**2)) * 180.0 / math.pi
    vm_rmse_pu  = np.sqrt(np.mean(dv**2))
    
    return float(th_rmse_deg), float(vm_rmse_pu)

class StateLoss(torch.nn.Module):
    """
    来自 train_refine_baseline.py 的监督损失 (MSE)
    """
    def __init__(self, bus_count: int, w_theta: float = 2.0, w_vm: float = 1.0, state_order='va_vm'):
        super().__init__()
        self.N = bus_count
        self.wt, self.wv = w_theta, w_vm
        self.state_order = state_order
        self.mse_loss = torch.nn.MSELoss(reduction='mean')

    def forward(self, x_hat, x_true):
        N = self.N
        
        if self.state_order == 'va_vm':
            va_hat, vm_hat = x_hat[..., :N], x_hat[..., N:]
            va_true, vm_true = x_true[..., :N], x_true[..., N:]
        elif self.state_order == 'vm_va':
            vm_hat, va_hat = x_hat[..., :N], x_hat[..., N:]
            vm_true, va_true = x_true[..., :N], x_true[..., N:]
        else:
            raise ValueError(f"Unknown state_order: {self.state_order}")

        dth = theta_wrap(va_hat, va_true)
        dv  = vm_hat - vm_true
        
        # 使用 MSELoss 计算均方误差
        l_th = self.mse_loss(dth, torch.zeros_like(dth))
        l_v  = self.mse_loss(dv, torch.zeros_like(dv))
        
        loss = self.wt * l_th + self.wv * l_v
        return loss, (l_th.detach(), l_v.detach())