# tools/plot_ieee33.py
import os, argparse, math, sys
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# 让脚本可从项目根目录找到模块
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from build_ieee33_with_pp import build_ieee33

# 可选：径向性自检
try:
    from tools.check_radial import check_radiality
except Exception:
    def check_radiality(*args, **kwargs):
        return True


def build_graph_from_net(net, bus_ids):
    """
    用 pandapower 的 line.from_bus/to_bus 构建无向图，节点用 1-based 母线号（按 bus_ids 顺序）。
    只把 net.line 中出现过的母线加入图（更贴近线路连通关系）。
    """
    id2pos = {int(b): i for i, b in enumerate(bus_ids)}  # 原始 pp bus 索引 -> 0..N-1
    G = nx.Graph()

    # 先把所有出现在线路中的母线作为节点加入
    used_buses = set()
    for _, row in net.line.iterrows():
        u_pp = int(row["from_bus"]); v_pp = int(row["to_bus"])
        if u_pp in id2pos and v_pp in id2pos:
            used_buses.add(id2pos[u_pp] + 1)
            used_buses.add(id2pos[v_pp] + 1)
    for n in sorted(used_buses):
        G.add_node(n)

    # 边：附带 in_service 属性（便于样式区分）
    for _, row in net.line.iterrows():
        u_pp = int(row["from_bus"]); v_pp = int(row["to_bus"])
        if u_pp in id2pos and v_pp in id2pos:
            u = id2pos[u_pp] + 1
            v = id2pos[v_pp] + 1
            G.add_edge(u, v, in_service=bool(row.get("in_service", True)))
    return G


def tree_layout(G, root=1, xgap=1.8, ygap=1.2):
    """
    径向树布局（BFS 分层 + 子树宽度中位对齐）。
    对单连通树效果最好；若有少量虚线联络线，也能较稳地可视化。
    """
    if root not in G.nodes:
        # 若指定根不在图里，选一个度数最大的点作为根
        root = max(G.degree, key=lambda kv: kv[1])[0] if len(G) else 1

    # 分层（BFS）
    levels = {n: None for n in G.nodes}
    parents = {}
    levels[root] = 0
    q = [root]
    while q:
        u = q.pop(0)
        for v in G.neighbors(u):
            if levels[v] is None:
                levels[v] = levels[u] + 1
                parents[v] = u
                q.append(v)

    # 孩子表
    children = {n: [] for n in G.nodes}
    for n, p in parents.items():
        children[p].append(n)

    # 子树大小（由叶到根累积）
    subtree_size = {n: 1 for n in G.nodes}
    for n in sorted(G.nodes, key=lambda x: (levels[x] if levels[x] is not None else 0), reverse=True):
        if children.get(n):
            subtree_size[n] = sum(subtree_size[c] for c in children[n])

    # 递归中序安放 x
    x_pos = {}
    cur_x = 0.0

    def assign_x(n):
        nonlocal cur_x
        if not children.get(n):  # 叶子
            x_pos[n] = cur_x
            cur_x += xgap
        else:
            for c in children[n]:
                assign_x(c)
            xs = [x_pos[c] for c in children[n]]
            x_pos[n] = sum(xs) / len(xs)

    assign_x(root)

    # y 由层决定
    pos = {n: (x_pos.get(n, 0.0), - (levels[n] if levels[n] is not None else 0) * ygap) for n in G.nodes}

    # 归一/居中
    xs = np.array([p[0] for p in pos.values()], dtype=float)
    ys = np.array([p[1] for p in pos.values()], dtype=float)
    if xs.std() > 1e-9:
        xs = (xs - xs.mean()) / xs.std()
    if ys.std() > 1e-9:
        ys = (ys - ys.mean()) / ys.std()
    pos = {n: (xs[i], ys[i]) for i, n in enumerate(pos.keys())}
    return pos


def draw_ieee33(net, bus_ids, pmu=None, scada=None, out_path="ieee33.png", root=1,
                show_off_edges=True, figsize=(10, 10)):
    G = build_graph_from_net(net, bus_ids)
    pos = tree_layout(G, root=root)

    pmu = set(pmu or [])
    scada = set(scada or [])

    # 边分组
    in_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d.get("in_service", True)]
    off_edges = [(u, v) for (u, v, d) in G.edges(data=True) if not d.get("in_service", True)]

    plt.figure(figsize=figsize)
    # 先画在运行业务的边（实线）
    if in_edges:
        nx.draw_networkx_edges(G, pos, edgelist=in_edges, width=1.8, alpha=0.9, edge_color="#4d4d4d")

    # 再画停运/联络边（虚线）
    if show_off_edges and off_edges:
        nx.draw_networkx_edges(G, pos, edgelist=off_edges, width=1.6, alpha=0.4,
                               style="dashed", edge_color="#a6a6a6")

    # 分三类节点涂色：PMU、SCADA、其它（1-based 号）
    all_nodes = list(G.nodes)
    others = [n for n in all_nodes if n not in pmu and n not in scada]
    if others:
        nx.draw_networkx_nodes(G, pos, nodelist=others, node_size=360,
                               node_color="#9ecae1", edgecolors="#084594",
                               linewidths=1.0, label="Other")
    if scada:
        nx.draw_networkx_nodes(G, pos, nodelist=sorted(scada), node_size=440,
                               node_color="#74c476", edgecolors="#006d2c",
                               linewidths=1.2, label="SCADA")
    if pmu:
        nx.draw_networkx_nodes(G, pos, nodelist=sorted(pmu), node_size=500,
                               node_color="#fd8d3c", edgecolors="#7f2704",
                               linewidths=1.2, label="PMU")

    # 节点标签（母线号 1..N）
    nx.draw_networkx_labels(G, pos, labels={n: str(n) for n in G.nodes}, font_size=9)

    plt.axis("off")
    # 只有存在 PMU/SCADA 才显示图例，避免空图例
    handles = []
    if pmu or scada:
        plt.legend(frameon=False, loc="upper left")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()
    print(f"[PLOT] saved → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="ieee33.png", help="output png path")
    ap.add_argument("--pmu", type=int, nargs="*", default=[], help="PMU 母线号（1-based）")
    ap.add_argument("--scada", type=int, nargs="*", default=[], help="SCADA 母线号（1-based）")
    ap.add_argument("--slack", type=int, default=1, help="根母线号（1-based），用于树布局")
    ap.add_argument("--hide_off_edges", action="store_true", help="不显示停运/联络（in_service=False）的虚线")
    args = ap.parse_args()

    net, ybus, baseMVA, load_nodes, base_loadPQ, bus_ids = build_ieee33()

    # 自检：应为单连通树
    ok = check_radiality(net, verbose=True)
    if not ok:
        print("[WARN] 网络不是标准径向树，图形布局仍可显示，但请留意联络线/环。")

    draw_ieee33(
        net, bus_ids,
        pmu=args.pmu, scada=args.scada,
        out_path=args.out, root=args.slack,
        show_off_edges=(not args.hide_off_edges),
        figsize=(10, 10)
    )


if __name__ == "__main__":
    main()
