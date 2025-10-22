# train/dataset.py
#
# 版本 2: 增强 Dataset 以支持场景生成 (量测缺失和异常值)

import numpy as np
import torch
from torch.utils.data import Dataset
import sys, os
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
from physics.ac_model import h_measure

class WindowDataset(Dataset):
    """
    一个可以动态生成带有量测缺失和异常值场景的数据集。
    """
    def __init__(self, npz_path, window_size=None, input_mode="raw",
                 # ================== 关键改动 1: 添加场景参数 ==================
                 missing_rate: float = 0.0,
                 outlier_rate: float = 0.0,
                 outlier_magnitude: float = 5.0,
                 # ==========================================================
                 eps=1e-9):
        super().__init__()
        
        # 加载数据
        D = np.load(npz_path, allow_pickle=True)
        self.x      = torch.from_numpy(D["x"].astype(np.float32))
        self.x_wls  = torch.from_numpy(D["x_wls"].astype(np.float32))
        self.z      = torch.from_numpy(D["z"].astype(np.float32))
        self.R      = torch.from_numpy(D["R"].astype(np.float32))
        self.feat   = torch.from_numpy(D["feat"].astype(np.float32))
        self.A_time = torch.from_numpy(D["A_time"].astype(np.float32))
        self.E_time = torch.from_numpy(D["E_time"].astype(np.float32))
        if "ztype" in D.files:
            self.ztype = D["ztype"]
        else:
            self.ztype = None
            
        self.input_mode = input_mode
        self.eps = eps
        
        # ================== 关键改动 2: 保存场景参数 ==================
        assert 0.0 <= missing_rate < 1.0, "missing_rate must be in [0, 1)"
        assert 0.0 <= outlier_rate < 1.0, "outlier_rate must be in [0, 1)"
        self.missing_rate = missing_rate
        self.outlier_rate = outlier_rate
        self.outlier_magnitude = outlier_magnitude
        # ==========================================================

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        # 初始数据
        z_sample = self.z[idx].clone()      # (W, M)
        R_sample = self.R[idx].clone()      # (W, M)
        
        W, M = z_sample.shape

        # ================== 关键改动 3: 实现场景生成逻辑 ==================

        # --- 场景 1: 量测缺失 ---
        if self.missing_rate > 0:
            # 在 M 个通道中随机选择要缺失的通道
            num_missing_channels = int(M * self.missing_rate)
            missing_channels_idx = np.random.choice(M, num_missing_channels, replace=False)
            
            # 将这些通道的所有时间步数据设为无效
            if num_missing_channels > 0:
                # 使用 NaN 标记缺失的 z 值，方便后续处理
                z_sample[:, missing_channels_idx] = torch.nan 
                # 将对应的方差 R 设为极大值，让损失函数忽略它们
                R_sample[:, missing_channels_idx] = 1e9 

        # --- 场景 2: 异常值注入 ---
        if self.outlier_rate > 0:
            # 在所有 W*M 个点中随机选择要污染的点
            num_outliers = int(W * M * self.outlier_rate)
            if num_outliers > 0:
                # 随机选择 W*M 个点中的 `num_outliers` 个点的索引
                outlier_indices = np.random.choice(W * M, num_outliers, replace=False)
                
                # 将 1D 索引转换为 2D 索引 (time, channel)
                time_idx, channel_idx = np.unravel_index(outlier_indices, (W, M))
                
                # 注入异常值
                z_sample[time_idx, channel_idx] *= self.outlier_magnitude
                # 注意：我们 *不* 修改 R，因为模型不知道这些是异常值
        
        # ===================================================================

        # 准备返回的字典
        sample = {
            "x": self.x[idx],
            "x_wls": self.x_wls[idx],
            "z": z_sample, # !! 可能已被污染 !!
            "R": R_sample, # !! 可能已被修改 !!
            "feat": self.feat[idx],
            "A_time": self.A_time[idx],
            "E_time": self.E_time[idx],
        }

        # 保持 ztype 为 numpy
        if self.ztype is not None:
            sample["ztype"] = self.ztype[idx]
        
        return sample
