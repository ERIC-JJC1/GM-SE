# wls_33/data_pipeline/make_windows.py
#把逐时刻的测量序列 z, m, R, x_true 打成 W=24/96 的滑窗样本，并计算皮尔逊相关的时序边权与若干窗口统计特征；保存为 npz

import os
import math
import numpy as np

def _ensure_slack_pos(meta: dict):
    if "slack_pos" in meta:
        return int(meta["slack_pos"])
    if ("slack_id" in meta) and ("bus_ids" in meta):
        bus_ids = np.asarray(meta["bus_ids"]).astype(int)
        slack_id = int(meta["slack_id"])
        pos = np.where(bus_ids == slack_id)[0]
        if pos.size:
            return int(pos[0])
    # 最后回退：当作已是位置
    return int(meta.get("slack_id", 0))

def _pearson_edges(window_Z):  # window_Z: [W, M]
    # 皮尔逊时序相关：对时间维做，得到 W×W 的相关
    Zc = window_Z - window_Z.mean(axis=0, keepdims=True)
    # 避免全零列导致 NaN
    std = np.std(Zc, axis=0, keepdims=True) + 1e-8
    Zc = Zc / std
    C = (Zc @ Zc.T) / Zc.shape[1]   # [W, W]
    # 将对角线设为1，裁剪到[-1,1]
    C = np.clip(C, -1.0, 1.0)
    # 非负边（也可保留全值；此处先取非负部分做加权）
    A = np.maximum(C, 0.0)
    return A.astype(np.float32), C.astype(np.float32)

def _window_stats(window_Z, window_M):
    # 简单而通用的统计量（可扩展）：
    # 1) 观测率 per step（时间维度上）
    obs_rate_t = (window_M > 0).mean(axis=1, keepdims=True)  # [W,1]
    # 2) 各量测分量的均值/方差（时间维度上） -> 聚合成每步的全局统计（避免超长特征）
    mean_t = window_Z.mean(axis=1, keepdims=True)            # [W,1]
    std_t  = window_Z.std(axis=1, keepdims=True)             # [W,1]
    # 3) 缺测率 per step
    miss_rate_t = 1.0 - obs_rate_t
    # 最后拼成 [W, F]
    F = np.concatenate([obs_rate_t, miss_rate_t, mean_t, std_t], axis=1)
    return F.astype(np.float32)  # [W, 4]

def windowize_sequences(Z, M, R, X_true, meta: dict,
                        W=24, stride=None, state_layout="thetaV",
                        extra=None):
    """
    Z: [T, M], M: [T, M], R: [T, M], X_true: [T, S]
    extra: 可选字典，形如：
       {"x_wls": [T, 2N], "ztype": [T, 4, M], ...}
    返回的每个 sample 会额外包含这些键的对应切片：[W, ...]
    """
    T, Mdim = Z.shape
    if stride is None:
        stride = max(1, W // 2)

    samples = []
    for s in range(0, T - W + 1, stride):
        e = s + W
        z_win = Z[s:e]
        m_win = M[s:e]
        r_win = R[s:e]
        x_win = X_true[s:e]

        # 时序边权（皮尔逊）
        A_time, E_time = _pearson_edges(z_win)

        # 窗口统计特征
        feat = _window_stats(z_win, m_win)

        sample = dict(
            Z=z_win.astype(np.float32),
            M=m_win.astype(np.float32),
            R=r_win.astype(np.float32),
            X=x_win.astype(np.float32),
            A_time=A_time,    # 非负邻接（可直接做GCN）
            E_time=E_time,    # 原始相关（可做注意力权重）
            feat=feat,        # [W,4]
        )

        # ====== 这里切 extra 并塞进 sample ======
        if extra is not None:
            for k, arr in extra.items():
                # arr 形状预期是 [T, ...]；窗口切 [s:s+W]
                sample[k] = arr[s:s+W].astype(arr.dtype if hasattr(arr, "dtype") else np.float32)
        # =====================================

        samples.append(sample)
    return samples


def split_and_save(samples, out_dir, tag,
                   train_ratio=0.7, val_ratio=0.15,
                   meta=None):
    """
    将 windowized 的样本列表切分并保存为 npz。
    - 基础键：z, mask, R, x, A_time, E_time, feat
    - 额外键：自动发现（如 x_wls, ztype 等），逐批 stack 并写入
    - meta  ：若给定，则一并写入
    """
    os.makedirs(out_dir, exist_ok=True)
    N = len(samples)
    n_tr = int(N * train_ratio)
    n_va = int(N * val_ratio)
    parts = {
        "train": samples[:n_tr],
        "val":   samples[n_tr:n_tr+n_va],
        "test":  samples[n_tr+n_va:],
    }

    base_keys = {"Z", "M", "R", "X", "A_time", "E_time", "feat"}

    for name, ss in parts.items():
        if not ss:
            continue

        # 基础字段堆叠
        Z = np.stack([s["Z"]      for s in ss])
        M = np.stack([s["M"]      for s in ss])
        R = np.stack([s["R"]      for s in ss])
        X = np.stack([s["X"]      for s in ss])
        A = np.stack([s["A_time"] for s in ss])
        E = np.stack([s["E_time"] for s in ss])
        F = np.stack([s["feat"]   for s in ss])

        out = dict(z=Z, mask=M, R=R, x=X, A_time=A, E_time=E, feat=F)

        # 自动发现并保存“额外键”
        extra_keys = set(ss[0].keys()) - base_keys
        for k in sorted(extra_keys):
            try:
                out[k] = np.stack([s[k] for s in ss])
            except Exception:
                # 允许单窗就缺失的可选字段（比如某些数据集没有 ztype）
                # 这里跳过或可以打印 warning
                continue

        # meta 写入（作为 object）
        if meta is not None:
            out["meta"] = meta

        path = os.path.join(out_dir, f"{tag}_{name}.npz")
        np.savez_compressed(path, **out)
        print(f"[SAVE] {name}: {path}  (N={len(ss)}, W={Z.shape[1]}, M={Z.shape[2]})")


def build_dataset_from_sequences(Z, M, R, X_true, meta,
                                 out_dir="data/windows_ieee33",
                                 tag="W24",
                                 W=24, stride=None, state_layout="thetaV",
                                 extra=None):                      # <<< 新增
    samples = windowize_sequences(Z, M, R, X_true, meta, W=W, stride=stride,
                                  state_layout=state_layout, extra=extra)
    split_and_save(samples, out_dir, tag, meta=meta)
    return True
