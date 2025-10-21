# run_wls_ieee33_from_simbench.py
import numpy as np
import pandapower as pp

from build_ieee33_with_pp import build_ieee33
from profiles_from_simbench import get_scalers_from_simbench
from measurements_and_wls import state_estimation  # 只用 WLS 核心求解
from data_pipeline.make_windows import build_dataset_from_sequences

# ========== 配置 ==========
SEED = 42
N_DAYS = 28
STEPS_PER_DAY = 96
SLACK_BUS = 1                # IEEE33 默认 1 号为平衡母线

# 量测布置（1-based 母线号）
PMU_BUSES   = [2, 6, 9, 29]
SCADA_BUSES = [4, 12, 14, 27]

# 噪声设定
PMU_VM_STD      = 0.005                 # PMU |V| 噪声 [p.u.]
PMU_TH_STD      = np.deg2rad(0.2)       # PMU θ 噪声 [rad]
SCADA_P_STD     = 0.01                  # SCADA P 注入噪声 [MW]
SCADA_Q_STD     = 0.01                  # SCADA Q 注入噪声 [MVAr]
PSEUDO_P_STD    = 0.10                 # 未装 SCADA 的注入“伪量测”
PSEUDO_Q_STD    = 0.10

ITER_MAX  = 30
THRESHOLD = 1e-5

# 滑窗数据输出
OUT_DIR = "data/windows_ieee33"
TAG_W24 = "W24";  W24 = 24; STRIDE24 = 12
TAG_W96 = "W96";  W96 = 96; STRIDE96 = 24


def main():
    rng = np.random.default_rng(SEED)

    # 1) IEEE33 + Ybus + 基准负荷
    net, ybus, baseMVA, load_nodes, base_loadPQ, bus_ids = build_ieee33()
    nbus = len(bus_ids)
    slack_pos = int(SLACK_BUS - 1)  # 0-based
    print(f"[NET] IEEE33: |bus|={nbus}, |load_nodes|={len(load_nodes)}, baseMVA={baseMVA}")
    print(f"[MEAS] PMU = {PMU_BUSES} | SCADA = {SCADA_BUSES} | SLACK = {SLACK_BUS}")

    # 2) SimBench 缩放系数（按 15min）
    P_scale, Q_scale = get_scalers_from_simbench(n_days=N_DAYS, steps_per_day=STEPS_PER_DAY)
    T = P_scale.size

    th_rmse_list = []; vm_rmse_list = []

    # === 为“滑窗数据集”准备的时序容器 ===
    Z_seq, M_seq, R_seq, X_seq = [], [], [], []

    # 3) 主循环
    for t in range(T):
        # 3.1 缩放负荷 (MW/MVAr)
        P_t = base_loadPQ[0] * P_scale[t]
        Q_t = base_loadPQ[1] * Q_scale[t]

        # 写入 pandapower 负荷（按 bus_ids）
        net.load["p_mw"] = 0.0; net.load["q_mvar"] = 0.0
        for idx_l, bus1b in enumerate(load_nodes):  # load_nodes 是 1-based（与 base_loadPQ 对齐）
            pp_bus = bus_ids[bus1b-1]
            sel = (net.load.bus == pp_bus)
            cnt = int(sel.sum()) if int(sel.sum())>0 else 1
            net.load.loc[sel, "p_mw"]   = P_t[idx_l] / cnt
            net.load.loc[sel, "q_mvar"] = Q_t[idx_l] / cnt

        # 3.2 潮流 → 真值
        pp.runpp(net, calculate_voltage_angles=True, numba=False)
        vm = net.res_bus.vm_pu.to_numpy()
        va = np.deg2rad(net.res_bus.va_degree.to_numpy())
        v_true = vm * (np.cos(va) + 1j*np.sin(va))

        # 3.3 生成“噪声量测”：PMU(|V|,θ) + SCADA(P,Q) + 未装SCADA母线的伪量测(P,Q)
        vm_true = np.abs(v_true)
        th_true = np.angle(v_true)

        # 为复现实验，每步固定子种子
        rng_t = np.random.default_rng(SEED + t)

        # --- PMU 测量：|V| 与 θ（θ以SLACK为参考；slack的θ测量可不加入等式）---
        pmu = sorted(set(PMU_BUSES))
        z_pmu_vm, z_pmu_th = [], []
        rows_pmu_vm, rows_pmu_th = [], []

        for b in pmu:
            z_pmu_vm.append(vm_true[b-1] + rng_t.normal(0, PMU_VM_STD))
            rows_pmu_vm.append(("vm", b))
            if b != SLACK_BUS:
                z_pmu_th.append((th_true[b-1] - th_true[SLACK_BUS-1]) + rng_t.normal(0, PMU_TH_STD))
                rows_pmu_th.append(("th", b))

        # --- SCADA 测量：P、Q 注入（只在 SCADA_BUSES，其他负荷母线用伪量测）---
        scada = sorted(set(SCADA_BUSES))
        pseudo_nodes = [int(n) for n in load_nodes if n not in scada]  # 1-based

        # 真值注入（直接用 pp 的 res_bus）
        P_inj_true = net.res_bus.p_mw.to_numpy()
        Q_inj_true = net.res_bus.q_mvar.to_numpy()

        z_scada_P, z_scada_Q = [], []
        rows_scada_P, rows_scada_Q = [], []

        for b in scada:
            z_scada_P.append(P_inj_true[b-1] + rng_t.normal(0, SCADA_P_STD))
            z_scada_Q.append(Q_inj_true[b-1] + rng_t.normal(0, SCADA_Q_STD))
            rows_scada_P.append(("P", b))
            rows_scada_Q.append(("Q", b))

        # --- 伪量测：在未装SCADA的负荷母线添加较松的 P/Q ---
        z_pseudo_P, z_pseudo_Q = [], []
        rows_pseudo_P, rows_pseudo_Q = [], []

        for b in pseudo_nodes:
            z_pseudo_P.append(P_inj_true[b-1] + rng_t.normal(0, PSEUDO_P_STD))
            z_pseudo_Q.append(Q_inj_true[b-1] + rng_t.normal(0, PSEUDO_Q_STD))
            rows_pseudo_P.append(("P", b))
            rows_pseudo_Q.append(("Q", b))

        # --- 拼 measurement 向量 z 与类型行标 rows（方便构造误差协方差）---
        z_list = (
            [(val, row) for val, row in zip(z_pmu_vm, rows_pmu_vm)] +
            [(val, row) for val, row in zip(z_pmu_th, rows_pmu_th)] +
            [(val, row) for val, row in zip(z_scada_P, rows_scada_P)] +
            [(val, row) for val, row in zip(z_scada_Q, rows_scada_Q)] +
            [(val, row) for val, row in zip(z_pseudo_P, rows_pseudo_P)] +
            [(val, row) for val, row in zip(z_pseudo_Q, rows_pseudo_Q)]
        )
        z = np.array([v for v, _ in z_list], dtype=float)

        # 3.4 构造误差协方差（对角）与类型编码 ztype
        # ztype 第一维：1:Pij 2:Pi 3:Qij 4:Qi 5:|Vi| 6:theta Vi（与你的 WLS 实现一致）
        # 这里只用 2(Pi),4(Qi),5(|V|),6(θ)
        ztype = np.zeros((4, len(z)), dtype=int)
        sig = np.zeros(len(z), dtype=float)

        for k, (_, row) in enumerate(z_list):
            kind, b = row
            if kind == "vm":
                ztype[:, k] = [5, b, 0, 0]
                sig[k] = PMU_VM_STD
            elif kind == "th":
                ztype[:, k] = [6, b, 0, 0]
                sig[k] = PMU_TH_STD
            elif kind == "P":
                ztype[:, k] = [2, b, 0, 0]
                sig[k] = SCADA_P_STD if b in scada else PSEUDO_P_STD
            elif kind == "Q":
                ztype[:, k] = [4, b, 0, 0]
                sig[k] = SCADA_Q_STD if b in scada else PSEUDO_Q_STD
            else:
                raise ValueError("unknown measurement kind")

        err_cov = np.diag(sig**2)          # WLS 用到的协方差矩阵
        R_vec   = (sig.astype(np.float32))**2  # 数据集里每个观测对应的方差

        # 3.5 运行 WLS（用真值热启动，让迭代更稳）
        v_hat, iters = state_estimation(
            ybus, z, ztype, err_cov, ITER_MAX, THRESHOLD, vtrue=v_true
        )

        # 3.6 评估
        dth = (np.angle(v_hat) - np.angle(v_true))
        dth = (dth + np.pi) % (2*np.pi) - np.pi
        dth[SLACK_BUS-1] = 0.0
        th_rmse = float(np.sqrt(np.mean(dth**2)))
        vm_rmse = float(np.sqrt(np.mean((np.abs(v_hat) - np.abs(v_true))**2)))
        th_rmse_list.append(th_rmse); vm_rmse_list.append(vm_rmse)

        if (t % 10) == 0:
            print(f"[t={t:03d}] it={iters:02d}  θ-RMSE={np.rad2deg(th_rmse):.3f} deg, |V|-RMSE={vm_rmse:.4f} p.u.")

        # === 3.7 追加到时序：z/m/R/x_true/x_wls/ztype（供滑窗数据集） ===
        m_t = np.ones(len(z), dtype=np.float32)

        # 真值相对角
        va_rel = va - va[slack_pos]
        va_rel = (va_rel + np.pi) % (2*np.pi) - np.pi
        x_true_t = np.concatenate([va_rel, vm]).astype(np.float32)

        # WLS 相对角 & 幅值
        va_wls = np.angle(v_hat); vm_wls = np.abs(v_hat)
        va_wls_rel = va_wls - va_wls[slack_pos]
        va_wls_rel = (va_wls_rel + np.pi) % (2*np.pi) - np.pi
        x_wls_t = np.concatenate([va_wls_rel.astype(np.float32), vm_wls.astype(np.float32)]).astype(np.float32)

        Z_seq.append(z.astype(np.float32))
        M_seq.append(m_t)
        R_seq.append(R_vec)                    # 每个观测的方差
        X_seq.append(x_true_t)
        # 👇 新增：WLS 与 ztype（保存 1-based 的 ztype 原样）
        try:
            ZTYPE_seq.append(ztype.copy().astype(np.int16))
        except NameError:
            ZTYPE_seq = [ztype.copy().astype(np.int16)]
        try:
            XWLS_seq.append(x_wls_t)
        except NameError:
            XWLS_seq = [x_wls_t]

    print("\n==== Summary over T steps ====")
    print(f"θ-RMSE (deg) mean = {np.rad2deg(np.mean(th_rmse_list)):.3f}")
    print(f"|V|-RMSE (p.u.) mean = {np.mean(vm_rmse_list):.4f}")

    # 4) 打包成滑窗数据集（W24 与 W96）
    Z = np.stack(Z_seq)        # [T, M]
    M = np.stack(M_seq)        # [T, M]
    R = np.stack(R_seq)        # [T, M]
    X_true = np.stack(X_seq)   # [T, 2N]
    X_wls  = np.stack(XWLS_seq)   # [T, 2N]
    ZTYPE  = np.stack(ZTYPE_seq)  # [T, 4, M]
    
    meta = {
        "bus_count": nbus,
        "slack_pos": slack_pos,
        "baseMVA": float(baseMVA),
    }

    print(f"\n[BUILD] sequences => Z{Z.shape} M{M.shape} R{R.shape} X{X_true.shape}")
    build_dataset_from_sequences(
        Z, M, R, X_true, meta,
        out_dir=OUT_DIR,
        tag=TAG_W24, W=W24, stride=STRIDE24, state_layout="thetaV",
        extra={"x_wls": X_wls, "ztype": ZTYPE}
    )
    build_dataset_from_sequences(
        Z, M, R, X_true, meta,
        out_dir=OUT_DIR,
        tag=TAG_W96, W=W96, stride=STRIDE96, state_layout="thetaV",
        extra={"x_wls": X_wls, "ztype": ZTYPE}
    )
    print(f"[DONE] npz 写入 {OUT_DIR}，请运行 tools/validate_windows.py 体检。")


if __name__ == "__main__":
    main()
