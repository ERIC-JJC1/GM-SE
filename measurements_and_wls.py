# measurements_and_wls.py
import numpy as np
import math

# =========================
# 1) 单相量测构造（推荐用这个）
# =========================
def get_measurements_sp(vnoisy, load_power, load_nodes, pmu_loc, slack_bus=1):
    """
    单相网络的量测构造：
    - 2: Pi   注入有功（负负荷为正注入，z = -P_load）
    - 4: Qi   注入无功（z = -Q_load）
    - 5: |Vi| 电压幅值
    - 6: θi   电压相角（以 slack 为参考，θ_i - θ_slack）
    返回: z (M,), z_type (4×M int)
    """
    z = -load_power[0, :]  # 注入有功
    z_type = np.array([2*np.ones(len(load_nodes)),
                       load_nodes,
                       np.zeros(len(load_nodes)),
                       np.zeros(len(load_nodes))]).astype(int)

    z = np.concatenate((z, -load_power[1, :]))  # 注入无功
    z_type_q = np.array([4*np.ones(len(load_nodes)),
                         load_nodes,
                         np.zeros(len(load_nodes)),
                         np.zeros(len(load_nodes))]).astype(int)
    z_type = np.concatenate((z_type, z_type_q), axis=1).astype(int)

    # PMU 电压幅值
    z = np.concatenate((z, np.abs(vnoisy[pmu_loc - 1])))
    z_type_vm = np.array([5*np.ones(len(pmu_loc)),
                          pmu_loc,
                          np.zeros(len(pmu_loc)),
                          np.zeros(len(pmu_loc))]).astype(int)
    z_type = np.concatenate((z_type, z_type_vm), axis=1)

    # PMU 电压相角（相对 slack）
    thetas = np.angle(vnoisy)
    th_ref = thetas[slack_bus - 1]
    pmu_loc_wo_ref = np.array([i for i in pmu_loc if i != slack_bus], dtype=int)
    if pmu_loc_wo_ref.size:
        z = np.concatenate((z, (thetas[pmu_loc_wo_ref - 1] - th_ref)))
        z_type_th = np.array([6*np.ones(len(pmu_loc_wo_ref)),
                              pmu_loc_wo_ref,
                              np.zeros(len(pmu_loc_wo_ref)),
                              np.zeros(len(pmu_loc_wo_ref))]).astype(int)
        z_type = np.concatenate((z_type, z_type_th), axis=1)

    return z, z_type.astype(int)


# =========================
# 2) 你给的原函数（保留），兼容补丁
# =========================
def get_measurements(vnoisy, inoisy, line_name, load_power, load_nodes, pmu_loc, include_current):
    """
    原作者：Moosa Moghimi
    注：这个函数按“3相×节点数”的索引构造量测。如果你跑单相，建议用上面的 get_measurements_sp。
    """
    # types: 1:Pij 2:Pi 3:Qij 4:Qi 5:|Vi| 6:theta Vi 7:|Ireal| 8:|Iimag|
    z = -load_power[0, :]
    z_type = np.array([2 * np.ones(len(load_nodes)), load_nodes,
                       np.zeros(len(load_nodes)), np.zeros(len(load_nodes))]).astype(int)

    z = np.concatenate((z, -load_power[1, :]))
    z_type_temp = np.array([4 * np.ones(len(load_nodes)), load_nodes,
                            np.zeros(len(load_nodes)), np.zeros(len(load_nodes))])
    z_type = np.concatenate((z_type, z_type_temp), axis=1).astype(int)

    # 注意：下面两段默认 3 相展开；若你是单相，这段可能不适用
    pmu_nodes = np.sort(np.concatenate((3 * pmu_loc - 2, 3 * pmu_loc - 1, 3 * pmu_loc)))
    z = np.concatenate((z, np.absolute(vnoisy[pmu_nodes - 1])))
    z_type_temp = np.array([5 * np.ones(len(pmu_nodes)), pmu_nodes,
                            np.zeros(len(pmu_nodes)), np.zeros(len(pmu_nodes))])
    z_type = np.concatenate((z_type, z_type_temp), axis=1).astype(int)

    z = np.concatenate((z, np.angle(vnoisy[pmu_nodes[1:] - 1]) - np.angle(vnoisy[0])))
    z_type_temp = np.array([6 * np.ones(len(pmu_nodes) - 1), pmu_nodes[1:],
                            np.zeros(len(pmu_nodes) - 1), np.zeros(len(pmu_nodes) - 1)])
    z_type = np.concatenate((z_type, z_type_temp), axis=1).astype(int)

    if include_current:
        for pm in pmu_loc:
            z = np.concatenate((z, np.real(inoisy[pm - 1])))
            z_type_temp = np.array([7 * np.ones(3), line_name[pm - 1, 0] * np.ones(3),
                                    line_name[pm - 1, 1] * np.ones(3), [1, 2, 3]])
            z_type = np.concatenate((z_type, z_type_temp), axis=1).astype(int)

        for pm in pmu_loc:
            z = np.concatenate((z, np.imag(inoisy[pm - 1])))
            z_type_temp = np.array([8 * np.ones(3), line_name[pm - 1, 0] * np.ones(3),
                                    line_name[pm - 1, 1] * np.ones(3), [1, 2, 3]])
            z_type = np.concatenate((z_type, z_type_temp), axis=1).astype(int)

    return z, z_type


# =========================
# 3) WLS（原函数 + 小补丁）
# =========================
def state_estimation(ybus, z, ztype, err_cov, iter_max, threshold, vtrue=None, baseMVA=1.0, slack_bus=1):
    """
    - 若给 vtrue（复杂电压），用它热启动（稳定很多）
    - 单相 WLS：x = [theta_2..theta_N, |V_1|..|V_N|]，theta_1=0（以 1 号为参考）
    - 注意：Pi/Qi 与 run_wls 脚本中的 z 是 MW/MVAr，故 h(x) 与雅可比相应乘 baseMVA
    """
    n = len(ybus)
    g = np.real(ybus); b = np.imag(ybus)

    # （可选）防误用：当前实现默认 slack_bus=1，后续要支持其它 slack 再扩展
    if slack_bus != 1:
        raise NotImplementedError("当前 WLS 仅支持 slack_bus=1，后续扩展再放开。")

    # 初始化
    if vtrue is not None:
        x = np.concatenate((np.angle(vtrue[1:]), np.abs(vtrue)))
    else:
        if (n % 3) == 0 and n >= 6:
            x = np.concatenate(
                ([-2*np.pi/3, -4*np.pi/3],
                 np.tile([0, -2*np.pi/3, -4*np.pi/3], int(np.floor(n/3))-1),
                 np.ones(n)*(1 + .000001*np.random.randn(n)))
            )
        else:
            x = np.concatenate((np.zeros(n-1), np.ones(n)))

    k = 0
    cont = True
    while k < iter_max and cont:
        v = x[n-1:]                                     # |V|
        th = np.concatenate(([0], x[0:n-1]))            # θ，以 1 号为参考

        # 量测函数 h(x)
        h = np.zeros(len(z))
        for m in range(len(z)):
            t = ztype[0, m]
            if t == 2:  # Pi
                i = ztype[1, m] - 1
                for jj in range(n):
                    h[m] += v[i]*v[jj]*(g[i,jj]*math.cos(th[i]-th[jj]) + b[i,jj]*math.sin(th[i]-th[jj]))
                h[m] *= baseMVA  # <<< 关键：MW 对齐
            elif t == 4:  # Qi
                i = ztype[1, m] - 1
                for jj in range(n):
                    h[m] += v[i]*v[jj]*(g[i,jj]*math.sin(th[i]-th[jj]) - b[i,jj]*math.cos(th[i]-th[jj]))
                h[m] *= baseMVA  # <<< 关键：MVAr 对齐
            elif t == 5:  # |Vi|
                i = ztype[1, m] - 1
                h[m] = v[i]
            elif t == 6:  # θi
                i = ztype[1, m] - 1
                h[m] = th[i]
            elif t in (7, 8):
                raise NotImplementedError("电流量测仅在三相实现里支持；单相请勿使用类型 7/8。")
            else:
                raise ValueError("Measurement type not defined!")

        # 雅可比
        h_jacob = np.zeros([len(z), len(x)])
        for m in range(len(z)):
            t = ztype[0, m]
            scale = baseMVA if t in (2, 4) else 1.0  # <<< 关键：Pi/Qi 整行同尺度
            if t == 2:  # Pi
                i = ztype[1, m] - 1
                for jj in range(n):
                    if jj != i:
                        if jj > 0:
                            h_jacob[m, jj-1] = v[i]*v[jj]*(g[i,jj]*math.sin(th[i]-th[jj]) - b[i,jj]*math.cos(th[i]-th[jj]))
                        h_jacob[m, jj+n-1] = v[i]*(g[i,jj]*math.cos(th[i]-th[jj]) + b[i,jj]*math.sin(th[i]-th[jj]))
                if i > 0:
                    h_jacob[m, i-1] = -v[i]**2 * b[i,i]
                    for jj in range(n):
                        h_jacob[m, i-1] += v[i]*v[jj]*(-g[i,jj]*math.sin(th[i]-th[jj]) + b[i,jj]*math.cos(th[i]-th[jj]))
                h_jacob[m, i+n-1] = v[i]*g[i,i]
                for jj in range(n):
                    h_jacob[m, i+n-1] += v[jj]*(g[i,jj]*math.cos(th[i]-th[jj]) + b[i,jj]*math.sin(th[i]-th[jj]))

            elif t == 4:  # Qi
                i = ztype[1, m] - 1
                for jj in range(n):
                    if jj != i:
                        if jj > 0:
                            h_jacob[m, jj-1] = v[i]*v[jj]*(-g[i,jj]*math.cos(th[i]-th[jj]) - b[i,jj]*math.sin(th[i]-th[jj]))
                        h_jacob[m, jj+n-1] = v[i]*(g[i,jj]*math.sin(th[i]-th[jj]) - b[i,jj]*math.cos(th[i]-th[jj]))
                if i > 0:
                    h_jacob[m, i-1] = -v[i]**2 * g[i,i]
                    for jj in range(n):
                        h_jacob[m, i-1] += v[i]*v[jj]*(g[i,jj]*math.cos(th[i]-th[jj]) + b[i,jj]*math.sin(th[i]-th[jj]))
                h_jacob[m, i+n-1] = -v[i]*b[i,i]
                for jj in range(n):
                    h_jacob[m, i+n-1] += v[jj]*(g[i,jj]*math.sin(th[i]-th[jj]) - b[i,jj]*math.cos(th[i]-th[jj]))

            elif t == 5:  # |Vi|
                i = ztype[1, m] - 1
                h_jacob[m, i+n-1] = 1.0

            elif t == 6:  # θi
                i = ztype[1, m] - 1
                if i > 0:
                    h_jacob[m, i-1] = 1.0

            # 统一尺度
            if scale != 1.0:
                h_jacob[m, :] *= scale

        # WLS 步（先保持你原实现，后面再做白化替换）
        sigma = np.sqrt(np.clip(np.diag(err_cov), 1e-12, None))
        w = 1.0 / sigma
        r_w = (z - h) * w
        J_w = (h_jacob.T * w).T
        gain = J_w.T @ J_w + 1e-6 * np.eye(J_w.shape[1])  # 轻微阻尼
        rhs  = J_w.T @ r_w
        try:
            delta_x = np.linalg.solve(gain, rhs)
        except np.linalg.LinAlgError:
            gain = gain + 1e-6*np.eye(gain.shape[0])
            delta_x = np.linalg.solve(gain, rhs)

        x += delta_x
        if np.max(np.abs(delta_x)) < threshold:
            cont = False
        k += 1

    v = x[n-1:]
    th = np.concatenate(([0], x[0:n-1]))
    v_phasor = v * (np.cos(th) + np.sin(th)*1j)
    return v_phasor, k