# eval/eval_refine.py
# 评估两种“精炼器”：
# 1) identity: 直接使用 x_wls
# 2) projection: 在 x_wls 上做 GN/LM 投影（白化 + 轻阻尼 + 数值雅可比）
# 输出：θ-RMSE（deg）、|V|-RMSE（p.u.）、带权物理残差均值
import os, numpy as np
import argparse
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from build_ieee33_with_pp import build_ieee33
# 使用我们之前写的 physics/ac_model.py
try:
    from physics.ac_model import h_measure
except ImportError:
    raise SystemExit("未找到 physics/ac_model.py。请先按上一步创建该文件并实现 h_measure(...)。")

# -------------------------
# 数学小工具
# -------------------------
def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def angle_rmse_deg(est, true):
    """
    est,true: [..., N] 角度（rad）
    返回：整体 RMSE in degree
    """
    d = wrap_angle(est - true)
    rmse_rad = np.sqrt(np.mean(d**2))
    return float(np.degrees(rmse_rad))

def voltage_rmse(est_vm, true_vm):
    return float(np.sqrt(np.mean((est_vm - true_vm)**2)))

def phys_residual_weighted(z, R, x, ztype, ybus, baseMVA, slack_pos):
    """
    计算单个时刻的带权物理残差均值：mean( ((z - h(x))/sqrt(R))^2 )
    """
    hx = h_measure(ybus, x, ztype, baseMVA, slack_pos)
    wres = (z - hx) / np.sqrt(np.maximum(R, 1e-12))
    return float(np.mean(wres**2))


# -------------------------
# GN/LM 一步或多步投影（数值雅可比）
# -------------------------
def gn_lm_project(x_init, z, R, ztype, ybus, baseMVA, slack_pos,
                  steps=1, eps=1e-5, lm=1e-2, vm_clip=(0.8, 1.2)):
    """
    改进点：
    - 冻结 slack 角：不对 theta_slack 求导、不更新
    - 角/幅维度分开差分步长：eps_theta, eps_vm
    - 回溯线搜索：保证每步物理残差下降
    """
    x = x_init.astype(np.float64).copy()
    z = z.astype(np.float64)
    R = R.astype(np.float64)
    n = x.size // 2
    lo, hi = vm_clip
    eps_theta = min(eps, 1e-6)
    eps_vm    = max(eps, 1e-4)

    W = 1.0 / np.sqrt(np.maximum(R, 1e-18))

    for _ in range(max(1, steps)):
        hx = h_measure(ybus, x, ztype, baseMVA, slack_pos)
        r  = z - hx
        r_w = r * W
        base_res = float((r_w**2).mean())

        M = z.size
        J = np.zeros((M, x.size), dtype=float)

        # 数值雅可比（冻结 slack 角）
        for i in range(x.size):
            if i < n and i == slack_pos:
                # 冻结 slack 角：该列保持 0
                continue
            x2 = x.copy()
            if i < n:
                x2[i] = (x2[i] + eps_theta)
            else:
                x2[i] = (x2[i] + eps_vm)
            J[:, i] = (h_measure(ybus, x2, ztype, baseMVA, slack_pos) - hx) / (eps_theta if i < n else eps_vm)

        J_w = (J.T * W).T
        H = J_w.T @ J_w + lm * np.eye(x.size)
        g = J_w.T @ r_w
        
        
        jn = np.linalg.norm(J, ord='fro')
        gnorm = np.linalg.norm(g)
        # 打个很少出现的警告，不会刷屏
        if (jn == 0 or not np.isfinite(jn)) or (gnorm == 0 or not np.isfinite(gnorm)):
            print(f"[warn] J||={jn:.3e}, g||={gnorm:.3e}")     
            
               
        try:
            dx = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            H = J_w.T @ J_w + (10.0 * lm) * np.eye(x.size)
            dx = np.linalg.solve(H, g)

        # 冻结 slack 角的增量
        dx[:n][slack_pos] = 0.0

        # 回溯线搜索（确保残差下降）
        alpha = 1.0
        while alpha > 1e-3:
            x_try = x.copy()
            x_try[:n] = wrap_angle(x[:n] + alpha * dx[:n])
            x_try[n:] = np.clip(x[n:] + alpha * dx[n:], lo, hi)
            hx_try = h_measure(ybus, x_try, ztype, baseMVA, slack_pos)
            r_try_w = (z - hx_try) * W
            new_res = float((r_try_w**2).mean())
            if new_res <= base_res* (1 - 1e-9):  # 接受
                x = x_try
                base_res = new_res
                break
            alpha *= 0.5
        # 若回溯失败（极罕见），就保持原 x 进入下一步

    return x.astype(np.float32)



# -------------------------
# 评估主流程
# -------------------------
def evaluate_file(path, mode="identity", steps=1, lm=1e-2, eps=1e-5, vm_clip=(0.8,1.2), slack_pos=0):
    D = np.load(path)
    Z = D["z"]          # [Nwin, W, M]
    R = D["R"]          # [Nwin, W, M]
    Xw= D["x_wls"]      # [Nwin, W, 2N]
    Xt= D["x"]          # [Nwin, W, 2N]

    # ztype 可能有两种保存方式： [Nwin, W, 4, M] 或 [Nwin, 4, M]
    if "ztype" not in D:
        raise ValueError("该 npz 不包含 ztype，请在生成阶段写入 ztype。")
    ZT = D["ztype"]
    if ZT.ndim == 3:
        # [Nwin, 4, M] -> 扩展成 [Nwin, W, 4, M]（相同的 ztype 复用于整个窗口）
        ZT = np.repeat(ZT[:, None, :, :], Z.shape[1], axis=1)
    elif ZT.ndim != 4:
        raise ValueError(f"不支持的 ztype 维度: {ZT.shape}")

    # 网络参数（确保与数据生成一致）
    _, ybus, baseMVA, *_ = build_ieee33()

    Nwin, W, M = Z.shape
    nbus = Xw.shape[-1] // 2

    # 累计指标
    th_err_deg, vm_err, phys_vals = [], [], []

    for i in range(Nwin):
        for t in range(W):
            z    = Z[i, t]
            Rvec = R[i, t]
            xw   = Xw[i, t]
            xt   = Xt[i, t]
            ztyp = ZT[i, t]

            if mode == "identity":
                xhat = xw
            elif mode == "projection":
                xhat = gn_lm_project(xw, z, Rvec, ztyp, ybus, baseMVA, slack_pos,
                                     steps=steps, eps=eps, lm=lm, vm_clip=vm_clip)
            else:
                raise ValueError(f"未知 mode: {mode}")

            # 拆分角度与电压
            th_hat, vm_hat = xhat[:nbus], xhat[nbus:]
            th_true, vm_true = xt[:nbus], xt[nbus:]

            th_err_deg.append(angle_rmse_deg(th_hat, th_true))
            vm_err.append(voltage_rmse(vm_hat, vm_true))
            phys_vals.append(phys_residual_weighted(z, Rvec, xhat, ztyp, ybus, baseMVA, slack_pos))

    # 汇总
    th_mean = float(np.mean(th_err_deg))
    vm_mean = float(np.mean(vm_err))
    phys_mean = float(np.mean(phys_vals))

    print(f"[{os.path.basename(path)} | {mode}] "
          f"θ-RMSE={th_mean:.3f} deg | |V|-RMSE={vm_mean:.4f} p.u. | 物理残差(带权)均值={phys_mean:.3f}")
    return th_mean, vm_mean, phys_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="npz 文件路径，如 data/windows_ieee33/W24_test.npz")
    ap.add_argument("--mode", type=str, default="identity", choices=["identity","projection"])
    ap.add_argument("--steps", type=int, default=1, help="GN/LM 投影步数（projection 模式有效）")
    ap.add_argument("--lm", type=float, default=1e-2, help="LM 阻尼（projection 模式有效）")
    ap.add_argument("--eps", type=float, default=1e-5, help="数值雅可比扰动幅度（projection 模式有效）")
    ap.add_argument("--vm_lo", type=float, default=0.8, help="电压幅值下限(p.u.)")
    ap.add_argument("--vm_hi", type=float, default=1.2, help="电压幅值上限(p.u.)")
    ap.add_argument("--slack_pos", type=int, default=0, help="slack 的 0-based 位置（IEEE33 默认 0）")
    args = ap.parse_args()

    vm_clip = (args.vm_lo, args.vm_hi)
    evaluate_file(args.data, mode=args.mode, steps=args.steps, lm=args.lm,
                  eps=args.eps, vm_clip=vm_clip, slack_pos=args.slack_pos)


if __name__ == "__main__":
    main()
