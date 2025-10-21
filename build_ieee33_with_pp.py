# build_ieee33_with_pp.py
import numpy as np
import pandapower as pp
import pandapower.networks as pn
from pandapower.pypower.makeYbus import makeYbus

def build_ieee33():
    """
    返回:
      net        : pandapower 网络
      ybus       : (N,N) 复数导纳矩阵
      baseMVA    : 标么容量
      load_nodes : 有负荷的母线号(1-based)
      base_loadPQ: (2, |load_nodes|) IEEE33 的静态基准负荷 [MW, MVAr]
      bus_lookup : pandas Index -> 顺序(0..N-1) 的映射（实际上就是排好序的 bus 索引）
    """
    net = pn.case33bw()  # IEEE 33-bus
    # 统一母线顺序
    bus_ids = np.array(sorted(net.bus.index.to_numpy()), dtype=int)

    # 先做一次潮流，保证结果表齐全
    pp.runpp(net, calculate_voltage_angles=True, numba=False)

    # 取 ppc/Ybus
    ppc = net._ppc
    baseMVA = float(ppc["baseMVA"])
    Ybus, Yf, Yt = makeYbus(baseMVA, ppc["bus"], ppc["branch"])

    # pandapower 的 pp->ppc 映射：pp_bus_id -> ppc_bus_id
    lookup = net["_pd2ppc_lookups"]["bus"]
    ppc_idx = np.array([int(lookup[b]) for b in bus_ids], dtype=int)

    # 投影/重排 Ybus
    Ybus_proj = Ybus.tocsr()[ppc_idx, :][:, ppc_idx].toarray().astype(np.complex128)

    # 基准负荷（按 bus 聚合）
    # 注意：pandapower 的 net.load 以 bus 索引计
    bus_to_pos = {b:i for i,b in enumerate(bus_ids)}
    P = np.zeros(len(bus_ids), dtype=float)
    Q = np.zeros(len(bus_ids), dtype=float)
    for _, row in net.load.iterrows():
        b = int(row["bus"]); p = float(row["p_mw"]); q = float(row["q_mvar"])
        P[bus_to_pos[b]] += p; Q[bus_to_pos[b]] += q

    load_nodes = (np.where(P > 1e-9)[0] + 1).astype(int)  # 1-based
    base_loadPQ = np.vstack([P[P>1e-9], Q[Q>1e-9]])  # (2, n_load)

    return net, Ybus_proj, baseMVA, load_nodes, base_loadPQ, bus_ids
