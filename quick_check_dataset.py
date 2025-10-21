# quick_check_dataset.py (修正版)
import numpy as np, torch
from train.dataset import WindowDataset
from physics.ac_model import h_measure
from build_ieee33_with_pp import build_ieee33

npz_path = "data/windows_ieee33/W24_train.npz"
ds = WindowDataset(npz_path, input_mode="whiten")
b = ds[0]

print("keys:", list(b.keys()))
for k,v in b.items():
    if isinstance(v, torch.Tensor):
        print(f"{k}: {tuple(v.shape)}  dtype={v.dtype}")

# 1) 白化统计（仅作大致参考，不强行等于0/1）
z = b["z"]  # (W, M)
print("z(mean)=", z.mean().item(), "  z(std)=", z.std().item())

# 2) 逐步重算 r 并对比
D = np.load(npz_path, allow_pickle=True)
Z_raw = torch.from_numpy(D["z"]).float()[0]          # 原始z，同一窗口
R     = torch.from_numpy(D["R"]).float()[0]
ZT    = torch.from_numpy(D["ztype"])
if ZT.ndim == 3:
    ZT = torch.from_numpy(D["ztype"]).long()[0].repeat(Z_raw.shape[0],1,1)  # [W,4,M]
elif ZT.ndim == 4:
    ZT = torch.from_numpy(D["ztype"]).long()[0]
else:
    raise RuntimeError(f"ztype shape {ZT.shape} unsupported")

# 从 batch 里拿 meta（npz 里可能没有）
slack_pos = int(b["meta"]["slack_pos"])
baseMVA   = float(b["meta"]["baseMVA"])
_, ybus, _, *_ = build_ieee33()  # ybus来自同一构网函数
mx = 0.0
for t in range(Z_raw.shape[0]):
    h = torch.from_numpy(
        h_measure(ybus, b["x_wls"][t].numpy(), ZT[t].numpy(), baseMVA, slack_pos)
    ).float()
    r = (Z_raw[t] - h) / torch.sqrt(R[t].clamp_min(1e-12))
    mx = max(mx, (r - z[t]).abs().max().item())
print("max |recomputed_r - batch['z']| =", mx)
