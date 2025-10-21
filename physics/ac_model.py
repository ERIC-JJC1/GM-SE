# physics/ac_model.py
import numpy as np

def wrap_angle(a): return (a + np.pi) % (2*np.pi) - np.pi
def unpack_state(x, nbus):
    theta = x[..., :nbus]
    vm    = x[..., nbus:]
    return theta, vm

def h_measure(ybus, x, ztype, baseMVA, slack_pos):
    x = np.asarray(x, dtype=np.float64)          # ★ 强制双精度
    nbus = x.shape[-1] // 2
    theta, vm = unpack_state(x, nbus)
    v = vm * (np.cos(theta) + 1j*np.sin(theta))
    I = ybus @ v
    S = v * np.conj(I)
    P = S.real * baseMVA
    Q = S.imag * baseMVA

    M = ztype.shape[1]
    h = np.zeros(M, dtype=np.float64)            # ★ 双精度
    for k in range(M):
        t, b, _, _ = ztype[:, k]
        bi = int(b - 1)
        if t == 5:   h[k] = vm[bi]
        elif t == 6: h[k] = 0.0 if bi == slack_pos else wrap_angle(theta[bi] - theta[slack_pos])
        elif t == 2: h[k] = P[bi]
        elif t == 4: h[k] = Q[bi]
    return h                                      # ★ 返回 float64
