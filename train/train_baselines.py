# train/train_baselines.py
#
# 一个新的、通用的全监督训练脚本，可以适配多种 Baseline 模型。
# - 通过命令行参数选择模型类型。
# - 使用 StateLoss(x_hat, x_true) [cite: eric-jjc1/gm-se/GM-SE-177dda6e4bf15c41857643dd8953cd26082c8c3f/train/train_tagru_residual_r.py] 进行训练。
# - 可以通过命令行参数设置场景（数据缺失、异常值）。

import os, math, argparse, sys
import numpy as np
import torch
import torch.nn as nn

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from torch.utils.data import DataLoader
# ================== 关键改动 1: 导入新版 Dataset ==================
from train.dataset import WindowDataset
# ===============================================================

# ================== 关键改动 2: 动态导入 Baseline 模型 ===========
from models.baselines import GRUBaseline
# from models.baselines import TransformerBaseline # 未来可以取消注释
# ===============================================================

from physics.ac_model import h_measure
from build_ieee33_with_pp import build_ieee33

# --- 辅助函数 (与 train_tagru_residual_r.py [cite: eric-jjc1/gm-se/GM-SE-177dda6e4bf15c41857643dd8953cd26082c8c3f/train/train_tagru_residual_r.py] 相同) ---
def theta_wrap(pred, gt): return (pred - gt + math.pi) % (2*math.pi) - math.pi
class StateLoss(nn.Module):
    def __init__(self, bus_count: int, w_theta: float = 2.0, w_vm: float = 1.0):
        super().__init__(); self.N = bus_count; self.wt, self.wv = w_theta, w_vm
    def forward(self, x_hat, x_target):
        N = self.N; dth = theta_wrap(x_hat[..., :N], x_target[..., :N]); dv  = x_hat[..., N:] - x_target[..., N:]
        l_th = (dth**2).mean(); l_v  = (dv**2).mean(); return self.wt*l_th + self.wv*l_v, (l_th.detach(), l_v.detach())
def rmse_metrics(x_hat, x_true):
    x_hat_np = x_hat.detach().cpu().numpy(); x_true_np = x_true.detach().cpu().numpy()
    N = x_true_np.shape[-1] // 2; dth = theta_wrap(x_hat_np[..., :N], x_true_np[..., :N]); dv  = x_hat_np[..., N:] - x_true_np[..., N:]
    th_rmse = np.sqrt(np.mean(dth**2)) * 180.0 / math.pi; vm_rmse = np.sqrt(np.mean(dv**2)); return float(th_rmse), float(vm_rmse)

# --- 数据加载器 (现在传递场景参数) ---
def make_loader(args, npz_path, shuffle):
    ds = WindowDataset(
        npz_path,
        input_mode="raw", # 我们总是在脚本中计算 r
        missing_rate=args.missing_rate,
        outlier_rate=args.outlier_rate,
        outlier_magnitude=args.outlier_magnitude
    )
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=0, drop_last=False)
    D = np.load(npz_path, allow_pickle=True)
    return dl, D

_ybus_np, _baseMVA_np, _slack_pos_np = None, None, None
def init_phys_meta_np(slack_pos):
    global _ybus_np, _baseMVA_np, _slack_pos_np
    if _ybus_np is None: _, ybus, baseMVA, *_ = build_ieee33(); _ybus_np = ybus.astype(np.complex128); _baseMVA_np = float(baseMVA)
    _slack_pos_np = int(slack_pos)
def compute_whitened_residual(z_seq_np, x_wls_seq_np, R_seq_np, ztype_np):
    B, W, M = z_seq_np.shape; r_seq_list = []
    for b in range(B):
        r_w_list = []
        for t in range(W):
            z_t = z_seq_np[b, t]
            # 处理 NaN (缺失值)
            valid_meas_mask = ~np.isnan(z_t)
            
            x_wls_t = x_wls_seq_np[b, t]; h_wls_t = h_measure(_ybus_np, x_wls_t, ztype_np[b, t], _baseMVA_np, _slack_pos_np)
            
            res_t = z_t - h_wls_t
            sigma_t = np.sqrt(np.maximum(R_seq_np[b, t], 1e-9))
            r_w_t = res_t / sigma_t
            
            # 将缺失处的残差设为 0
            r_w_t[~valid_meas_mask] = 0.0
            
            r_w_list.append(r_w_t[np.newaxis, :])
        r_seq_list.append(np.concatenate(r_w_list, axis=0)[np.newaxis, :, :])
    return np.concatenate(r_seq_list, axis=0)
# --------------------------------

def main():
    ap = argparse.ArgumentParser(description="通用 Baseline 训练脚本")
    # 模型选择
    ap.add_argument("--model_type", type=str, default="gru", choices=["gru", "transformer"], help="要训练的 Baseline 模型类型")
    # 场景设置参数
    ap.add_argument("--missing_rate", type=float, default=0.0, help="量测通道缺失率 (0.0 to 1.0)")
    ap.add_argument("--outlier_rate", type=float, default=0.0, help="异常值比例 (0.0 to 1.0)")
    ap.add_argument("--outlier_magnitude", type=float, default=5.0, help="异常值乘数")
    # 训练/数据参数
    ap.add_ar
