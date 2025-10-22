# physics/torch_ac_model.py
#
# 版本 2: 确保 h_measure_torch 返回 h, P_inj, Q_inj

import torch
import numpy as np
from build_ieee33_with_pp import build_ieee33

# --- 缓存网络参数 ---
_ybus_torch, _baseMVA_torch = None, None
_G_torch, _B_torch = None, None

def get_torch_net_params(device):
    global _ybus_torch, _baseMVA_torch, _G_torch, _B_torch
    if _ybus_torch is None:
        _, ybus, baseMVA, *_ = build_ieee33()
        ybus_complex = ybus.astype(np.complex128)
        _ybus_torch = torch.from_numpy(ybus_complex).to(device)
        _G_torch = _ybus_torch.real
        _B_torch = _ybus_torch.imag
        _baseMVA_torch = float(baseMVA)
    # 确保返回在正确设备上的张量
    return _G_torch.to(device), _B_torch.to(device), _baseMVA_torch

# --- 状态分解与角度包裹 ---
def wrap_angle(a): return (a + torch.pi) % (2 * torch.pi) - torch.pi
def unpack_state(x, nbus): # 格式 [theta, vm]
    va = x[..., :nbus]
    vm = x[..., nbus:]
    return vm, va

# --- 核心：可微分的 h(x) 函数 ---
def h_measure_torch(x_hat, ztype, baseMVA, slack_pos, G, B):
    B_size, W_size, _ = x_hat.shape
    M = ztype.shape[-1]
    N = G.shape[0]

    vm, va = unpack_state(x_hat, N)

    vm_i = vm.unsqueeze(-1); vm_j = vm.unsqueeze(-2)
    va_i = va.unsqueeze(-1); va_j = va.unsqueeze(-2)
    vm_i_vm_j = vm_i * vm_j; va_diff = va_i - va_j
    cos_va_diff = torch.cos(va_diff); sin_va_diff = torch.sin(va_diff)

    P_matrix = vm_i_vm_j * (G * cos_va_diff + B * sin_va_diff)
    Q_matrix = vm_i_vm_j * (G * sin_va_diff - B * cos_va_diff)

    P_inj = P_matrix.sum(dim=-1) * baseMVA
    Q_inj = Q_matrix.sum(dim=-1) * baseMVA

    h_list = []
    va_slack = va[..., slack_pos].unsqueeze(-1)

# 提前处理 ztype，确保它是一个 Tensor
    # 注意：如果 ztype 在 batch 或 window 内变化，这里的索引需要是 [b, t, ...]
    if isinstance(ztype, np.ndarray):
        # 假设 ztype 在窗口 W 内不变, 且在 Batch B 内不变
        ztype_t = torch.from_numpy(ztype[0, 0]).long() # (4, M)
    elif isinstance(ztype, torch.Tensor):
         # 假设 ztype 在窗口 W 内不变, 且在 Batch B 内不变
        ztype_t = ztype[0, 0].long() # (4, M)
    else:
        raise TypeError(f"Unsupported ztype type: {type(ztype)}")

    # 确保 ztype_t 在正确的设备上 (如果 ztype 来自 batch，可能已经在 GPU 上)
    ztype_t = ztype_t.to(x_hat.device) # <-- 添加这一行确保设备一致

    for k in range(M):
        t = ztype_t[0, k].item() # 获取类型
        b = ztype_t[1, k].item() # 获取母线号 (1-based)
        bi = int(b - 1) # 转换为 0-based 索引

        if t == 5:   # Vm
            h_val = vm[..., bi]
        elif t == 6: # Va (相对 slack)
            h_val = torch.zeros_like(vm[..., 0]) if bi == slack_pos else wrap_angle(va[..., bi] - va_slack.squeeze(-1))
        elif t == 2: # Pi
            h_val = P_inj[..., bi]
        elif t == 4: # Qi
            h_val = Q_inj[..., bi]
        else:
            h_val = torch.zeros_like(vm[..., 0])

        h_list.append(h_val.unsqueeze(-1))

    h = torch.cat(h_list, dim=-1)

    # ================== 关键修复：确保返回三个值 ==================
    return h, P_inj, Q_inj
    # ==========================================================