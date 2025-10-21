# temporal_graph/build_time_graph.py
import numpy as np

def _corr(x, y, eps=1e-8):
    x = (x - x.mean()) / (x.std() + eps)
    y = (y - y.mean()) / (y.std() + eps)
    return float(np.clip((x * y).mean(), -1.0, 1.0))

def build_time_graph(window_stats, alpha=0.7, mode="pearson", topk=None):
    """
    window_stats: list[dict], 每个 t 的统计特征（如: {'pmu_rate':..., 'scada_rate':..., 'pz_rate':..., 'load_mean':..., 'resid_wls':...}）
    返回:
      idx_i, idx_j: 边两端索引（np.int64）
      s_ij: 边权（np.float32, 0~1）
    """
    W = len(window_stats)
    # 取几个稳定维度作为相似度载体（可以先用 load_mean/resid_wls 两三个）
    keys = [k for k in window_stats[0].keys()]
    X = np.array([[ws[k] for k in keys] for ws in window_stats], dtype=float)  # [W, D]

    S = np.zeros((W, W), dtype=float)
    for i in range(W):
        for j in range(W):
            if i == j: continue
            if mode == "pearson":
                S[i, j] = _corr(X[i], X[j])  # [-1,1]
            else:
                S[i, j] = _corr(X[i], X[j])

    # 映射到[0,1]
    S = (S + 1.0) / 2.0
    # 稀疏化：阈值或 top-k
    mask = S >= alpha
    if topk is not None:
        for i in range(W):
            row = S[i].copy()
            row[i] = -1
            if (row >= 0).sum() > topk:
                th = np.sort(row)[-topk]
                mask[i] = S[i] >= th
    ii, jj = np.where(mask)
    s_ij = S[ii, jj].astype(np.float32)
    return ii.astype(np.int64), jj.astype(np.int64), s_ij
