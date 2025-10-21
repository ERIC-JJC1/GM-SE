# physics/torch_ac_model.py
#
# 这是一个 *完全可微分* 的 h(x) 量测函数实现，使用纯 PyTorch。
# 它是阶段 2（物理弱监督）的核心。
# 它将取代 physics/ac_model.py 在损失计算中的作用。

import torch
import numpy as np
from build_ieee33_with_pp import build_ieee33

# --- 缓存网络参数 (Ybus, baseMVA) ---
_ybus_torch, _baseMVA_torch = None, None
_G_torch, _B_torch = None, None # Ybus 的实部和虚部

def get_torch_net_params(device):
    """
    加载一次 Ybus 并将其转换为 PyTorch 张量，缓存在 GPU/CPU 上。
    """
    global _ybus_torch, _baseMVA_torch, _G_torch, _B_torch
    if _ybus_torch is None:
        _, ybus, baseMVA, *_ = build_ieee33()
        ybus_complex = ybus.astype(np.complex128)
        
        # 转换为 PyTorch 复数张量
        _ybus_torch = torch.from_numpy(ybus_complex).to(device)
        # Ybus = G + jB
        _G_torch = _ybus_torch.real # Ybus 的实部 (G)
        _B_torch = _ybus_torch.imag # Ybus 的虚部 (B)
        _baseMVA_torch = float(baseMVA)
    
    return _G_torch, _B_torch, _baseMVA_torch

# --- PyTorch 状态分解与角度包裹 ---

def wrap_angle(a):
    """(a + pi) % (2*pi) - pi"""
    return (a + torch.pi) % (2 * torch.pi) - torch.pi

def unpack_state(x, nbus):
    """ x: (..., 2N) -> (..., N), (..., N) """
    # 保持与 models/tagru.py 中输出一致: [vm, va]
    vm = x[..., :nbus]
    va = x[..., nbus:]
    return vm, va

# --- 核心：可微分的 h(x) 函数 ---

def h_measure_torch(x_hat, ztype, baseMVA, slack_pos, G, B):
    """
    PyTorch 可微分版本的 h_measure
    
    输入:
      x_hat     : (B, W, 2N) - 模型的估计状态 (Vm, Va)
      ztype     : (B, W, 4, M) - 量测类型 (CPU, numpy)
      baseMVA   : float
      slack_pos : int
      G, B      : (N, N) - Ybus 的 G 和 B 矩阵 (PyTorch 张量)
    
    输出:
      h         : (B, W, M) - 计算得到的量测值 (PyTorch 张量, 附带梯度)
    """
    
    B_size, W_size, _ = x_hat.shape
    M = ztype.shape[-1]
    N = G.shape[0]

    # 1. 分解状态 x_hat -> Vm, Va
    vm, va = unpack_state(x_hat, N) # (B, W, N)
    
    # 2. 计算节点注入功率 P_inj, Q_inj (可微分)
    #    这部分是 AC Power Flow 的核心方程
    #    P_i = sum_j V_i V_j [ G_ij * cos(th_i - th_j) + B_ij * sin(th_i - th_j) ]
    #    Q_i = sum_j V_i V_j [ G_ij * sin(th_i - th_j) - B_ij * cos(th_i - th_j) ]

    # 扩展维度以进行矩阵运算
    vm_i = vm.unsqueeze(-1) # (B, W, N, 1)
    vm_j = vm.unsqueeze(-2) # (B, W, 1, N)
    va_i = va.unsqueeze(-1) # (B, W, N, 1)
    va_j = va.unsqueeze(-2) # (B, W, 1, N)

    vm_i_vm_j = vm_i * vm_j                 # (B, W, N, N)
    va_diff = va_i - va_j                   # (B, W, N, N)
    
    cos_va_diff = torch.cos(va_diff)
    sin_va_diff = torch.sin(va_diff)
    
    # (B,W,N,N) * (N,N) -> (B,W,N,N)
    P_matrix = vm_i_vm_j * (G * cos_va_diff + B * sin_va_diff)
    Q_matrix = vm_i_vm_j * (G * sin_va_diff - B * cos_va_diff)
    
    # 沿 j 轴求和
    P_inj = P_matrix.sum(dim=-1) * baseMVA  # (B, W, N)
    Q_inj = Q_matrix.sum(dim=-1) * baseMVA  # (B, W, N)

    # 3. 根据 ztype 构建 h(x) 向量
    #    这部分是纯索引操作，无法高效并行，但仍然可微分
    h_list = []
    
    # 提取 slack 角用于参考
    va_slack = va[..., slack_pos].unsqueeze(-1) # (B, W, 1)
    
    for k in range(M):
        # ztype 是 numpy，在 CPU 上
        t = ztype[0, 0, 0, k] # 假设 ztype 在窗口 W 内不变, 且在 Batch B 内不变
        b = ztype[0, 0, 1, k] # (如果它们变化，这里的索引需要是 [b, t, ...])
        
        bi = int(b - 1) # 转换为 0-based 索引

        if t == 5:   # Vm
            h_val = vm[..., bi] # (B, W)
        elif t == 6: # Va (相对 slack)
            if bi == slack_pos:
                h_val = torch.zeros_like(vm[..., 0]) # (B, W)
            else:
                h_val = wrap_angle(va[..., bi] - va_slack.squeeze(-1)) # (B, W)
        elif t == 2: # Pi
            h_val = P_inj[..., bi] # (B, W)
        elif t == 4: # Qi
            h_val = Q_inj[..., bi] # (B, W)
        else:
            # 为简单起见，暂不支持其他类型
            h_val = torch.zeros_like(vm[..., 0]) # (B, W)
            
        h_list.append(h_val.unsqueeze(-1)) # (B, W, 1)

    h = torch.cat(h_list, dim=-1) # (B, W, M)
    return h