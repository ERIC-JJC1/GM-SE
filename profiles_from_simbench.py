# profiles_from_simbench.py
import numpy as np
import simbench as sb

def get_scalers_from_simbench(code="1-MV-urban--0-sw", n_days=1, steps_per_day=96, seed=42):
    """
    返回: (P_scale, Q_scale), shape = (T,), T = n_days*steps_per_day
    用 SimBench 的总负荷曲线(同网型下的 load总和)做比例系数，均值归一化。
    """
    rng = np.random.default_rng(seed)
    net = sb.get_simbench_net(code)
    abs_vals = sb.get_absolute_values(net, profiles_instead_of_study_cases=True)
    load_P = abs_vals[('load','p_mw')]      # [T_total, n_load]
    load_Q = abs_vals[('load','q_mvar')]    # [T_total, n_load]

    T_total = load_P.shape[0]
    T = n_days*steps_per_day
    assert T <= T_total, "T 超出 SimBench 全年的步数"

    # 取前 T 步（你也可以随机起点切片）
    P_tot = load_P.iloc[:T].to_numpy().sum(axis=1) + 1e-6
    Q_tot = load_Q.iloc[:T].to_numpy().sum(axis=1) + 1e-6
    P_scale = P_tot / P_tot.mean()
    Q_scale = Q_tot / Q_tot.mean()
    return P_scale.astype(np.float64), Q_scale.astype(np.float64)
