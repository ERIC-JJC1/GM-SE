# physics/feasibility.py
import numpy as np
from .ac_model import h_measure, wrap_angle, unpack_state, pack_state

def gn_projection(x_init, z, R_diag, ztype, ybus, baseMVA, slack_pos, max_steps=1, lm=1e-2):
    """
    简化版 GN/LM：只用数值雅可比近似（小扰动差分），步数=1 即可显著降物理残差
    x_init: [2N]
    R_diag: [M] 方差
    """
    x = x_init.copy()
    nbus = x.shape[-1] // 2
    M = z.shape[0]
    eps = 1e-5
    W = 1.0 / np.sqrt(np.maximum(R_diag, 1e-9))

    for _ in range(max_steps):
        hx = h_measure(ybus, x, ztype, baseMVA, slack_pos)
        r = (z - hx) * W
        # 数值雅可比 J ≈ dh/dx
        J = np.zeros((M, 2*nbus), dtype=float)
        for i in range(2*nbus):
            x_pert = x.copy()
            x_pert[i] += eps if i < nbus else eps
            h2 = h_measure(ybus, x_pert, ztype, baseMVA, slack_pos)
            J[:, i] = (h2 - hx) / eps
        JW = (J.T * W).T  # 每行乘权
        H = JW.T @ JW + lm * np.eye(2*nbus)
        g = JW.T @ r
        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        x[:nbus] = wrap_angle(x[:nbus] + delta[:nbus])
        x[nbus:] = np.clip(x[nbus:] + delta[nbus:], 0.8, 1.2)  # 电压硬限略保守
    return x.astype(np.float32)
