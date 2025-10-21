# tools/check_radial.py（或直接粘到你的 run 脚本里）
import networkx as nx
import numpy as np

def check_radiality(net, verbose=True):
    # 只用 in_service=True 的线路
    use = net.line[net.line.in_service].copy()
    G = nx.Graph()
    for _, row in use.iterrows():
        u = int(row.from_bus); v = int(row.to_bus)
        G.add_edge(u, v)
    n = net.bus.shape[0]
    m = G.number_of_edges()
    comps = list(nx.connected_components(G))
    has_cycle = len(list(nx.cycle_basis(G))) > 0
    if verbose:
        print(f"[CHECK] nodes={n}, edges={m}, components={len(comps)}, cycles={has_cycle}")
        if not has_cycle and len(comps)==1 and m==n-1:
            print("[OK] 拓扑为单连通树（径向）。")
        else:
            print("[WARN] 非标准径向；请检查是否误把联络线画/合上了。")
    return (not has_cycle) and len(comps)==1 and (m==n-1)