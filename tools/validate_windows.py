import os, sys, numpy as np

def _check_file(path):
    if not os.path.exists(path):
        print(f"[MISS] {path} 不存在")
        return False
    try:
        D = np.load(path)
        print(f"\n[LOAD] {path}")
        z, m, R, x = D["z"], D["mask"], D["R"], D["x"]
        A, E, F = D["A_time"], D["E_time"], D["feat"]
        print(f" shapes: z{z.shape}, mask{m.shape}, R{R.shape}, x{x.shape}, A{A.shape}, E{E.shape}, feat{F.shape}")

        # 1) 基本尺寸一致性
        assert z.shape == m.shape == R.shape, "z/m/R 维度必须一致 [N, W, M]"
        assert x.shape[0] == z.shape[0] and x.shape[1] == z.shape[1], "x 的前两维需等于 [N, W]"
        assert A.shape[:2] == z.shape[:2] and A.shape[2] == z.shape[1], "A_time 需是 [N,W,W]"
        assert E.shape == A.shape, "E_time 需与 A_time 同形状"
        assert F.shape[:2] == z.shape[:2], "feat 需是 [N,W,F]"

        # 2) 数值健康度
        def ok(arr, name):
            bad = np.isnan(arr).sum() + np.isinf(arr).sum()
            print(f"  - {name}: min={arr.min():.4g} max={arr.max():.4g} NaN/Inf={bad}")
            return bad == 0
        ok(z, "z"); ok(m, "mask"); ok(R, "R"); ok(x, "x"); ok(A, "A_time"); ok(E, "E_time"); ok(F, "feat")

        # 3) 掩码与方差一致性（mask==0 的地方不做硬约束，但建议 R>0；统计下观测率）
        obs_rate = m.mean()
        zero_var_on_obs = (R[(m>0)] <= 0).sum()
        print(f"  - 观测率 overall = {obs_rate:.3f}")
        print(f"  - (mask>0) 但 R<=0 的条目数 = {int(zero_var_on_obs)} (应为 0)")

        # 4) A/E 对称性和范围
        sym_A = np.abs(A - np.transpose(A, (0,2,1))).max()
        sym_E = np.abs(E - np.transpose(E, (0,2,1))).max()
        print(f"  - A_time 对称性 max|A-A^T|={sym_A:.3e} (应接近0)")
        print(f"  - E_time 对称性 max|E-E^T|={sym_E:.3e} (应接近0)")
        print(f"  - A_time 值域 [{A.min():.3f}, {A.max():.3f}] (应在[0,1])")
        print(f"  - E_time 值域 [{E.min():.3f}, {E.max():.3f}] (应在[-1,1])")

        # 5) feat 简单检查（[obs_rate, miss_rate, mean_t, std_t]）
        Fnames = ["obs_rate","miss_rate","mean_t","std_t"]
        for j,nm in enumerate(Fnames[:F.shape[-1]]):
            col = F[..., j]
            print(f"  - feat[{nm}] mean={col.mean():.4f} min={col.min():.4f} max={col.max():.4f}")

        # 6) 随机抽一窗看看一步观测计数
        rng = np.random.default_rng(123)
        ridx = rng.integers(0, z.shape[0])
        t_idx = rng.integers(0, z.shape[1])
        obs_cnt = int(m[ridx, t_idx].sum())
        print(f"  - 随机样本[{ridx}] 时间步[{t_idx}] 观测数 = {obs_cnt}/{z.shape[2]}")

        print("[OK] 该文件初步通过体检。")
        return True
    except Exception as e:
        print(f"[ERR] 读取或检查失败: {e}")
        return False

def main():
    base = "data/windows_ieee33"
    tags = ["W24", "W96"]
    parts = ["train","val","test"]
    any_ok = False
    for tag in tags:
        for p in parts:
            path = os.path.join(base, f"{tag}_{p}.npz")
            if _check_file(path):
                any_ok = True
    if not any_ok:
        print("\n没有任何数据文件通过检查。请先在生成脚本里调用 build_dataset_from_sequences(...) 产出 npz。")

if __name__ == "__main__":
    main()