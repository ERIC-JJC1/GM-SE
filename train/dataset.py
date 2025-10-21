# wls_33/train/dataset.py
import os, sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
    
import numpy as np
import torch
from torch.utils.data import Dataset

# 需要用来算 h(x_wls)
from build_ieee33_with_pp import build_ieee33
from physics.ac_model import h_measure

class WindowDataset(Dataset):
    """
    读取窗口化 npz，并按需构造训练输入：
      - 默认输入: r = (z - h(x_wls)) / sqrt(R)   （白化残差，强烈推荐）
      - 备选输入: 标准化 z （z_standardized）
      - 兜底输入: 原始 z
    额外返回：x_wls, feat, A_time, E_time, ztype（后续物理损失/投影可用）
    """
    def __init__(self,
                 npz_path: str,
                 device: str = "cpu",
                 input_mode: str = "whiten",   # "whiten" | "z_std" | "z_raw"
                 eps_var: float = 1e-9):
        """
        input_mode:
          - "whiten": 用 r=(z-h(x_wls))/sqrt(R) 作为模型输入
          - "z_std" : 用 (z-mean)/std （按训练集统计；若npz未包含统计量则退化为逐样本均值方差）
          - "z_raw" : 直接用原始 z
        """
        D = np.load(npz_path, allow_pickle=True)

        # 基础张量（float32 即可；内部计算 h(x) 时会升到 float64）
        self.Z = torch.from_numpy(D["z"]).float()         # [N, W, M]
        self.M = torch.from_numpy(D["mask"]).float()      # [N, W, M]
        self.R = torch.from_numpy(D["R"]).float()         # [N, W, M]
        self.X = torch.from_numpy(D["x"]).float()         # [N, W, S]
        self.A = torch.from_numpy(D["A_time"]).float()    # [N, W, W]
        self.E = torch.from_numpy(D["E_time"]).float()    # [N, W, W]
        self.F = torch.from_numpy(D["feat"]).float()      # [N, W, F]
        self.device = device

        # 可选：x_wls / ztype 供 r 计算与后续评估
        if "x_wls" in D.files:
            self.XWLS = torch.from_numpy(D["x_wls"]).float()  # [N, W, 2N]
        else:
            self.XWLS = None

        if "ztype" in D.files:
            self.ZTYPE = torch.from_numpy(D["ztype"]).int()   # [N, W, 4, M] 或 [N, 4, M]
            # 若是 [N,4,M]，扩成 [N,W,4,M]
            if self.ZTYPE.ndim == 3:
                self.ZTYPE = self.ZTYPE.unsqueeze(1).repeat(1, self.Z.shape[1], 1, 1)
        else:
            self.ZTYPE = None

        # 记录 meta（slack_pos、baseMVA 等）
        self.meta = D["meta"].item() if "meta" in D.files else {}
        self.slack_pos = int(self.meta.get("slack_pos", 0))

        # 网络参数（用于 h(x_wls)）
        # 注意：只构造一次；ybus 用 complex128，避免精度问题
        _, ybus, baseMVA, *_ = build_ieee33()
        self.ybus = ybus.astype(np.complex128)
        self.baseMVA = float(baseMVA)

        # 选择输入构造方式
        self.input_mode = input_mode
        self.eps_var = float(eps_var)

        # 预计算输入张量 self.INPUT
        self.INPUT = self._build_input_tensor(D)

    def _build_input_tensor(self, D: np.lib.npyio.NpzFile) -> torch.Tensor:
        """
        根据 input_mode 构造输入：
          - whiten: r = (z - h(x_wls)) / sqrt(R)
          - z_std : z 标准化（优先使用 npz 内保存的统计量；否则用本文件内统计）
          - z_raw : 直接用 z
        """
        N, W, M = self.Z.shape

        if self.input_mode == "whiten":
            if (self.XWLS is None) or (self.ZTYPE is None):
                # 缺少必要字段就回退到 z_raw
                print("[WindowDataset] warn: npz 缺少 x_wls 或 ztype，whiten 模式回退为 z_raw。")
                return self.Z.clone()

            # 逐 (n,t) 计算 h(x_wls) 并白化
            inp = torch.empty_like(self.Z)  # [N,W,M]
            # 为了速度，用 numpy 计算再回填
            Z_np    = self.Z.numpy()
            R_np    = self.R.numpy()
            XW_np   = self.XWLS.numpy()
            ZT_np   = self.ZTYPE.numpy()
            for n in range(N):
                for t in range(W):
                    hx = h_measure(self.ybus, XW_np[n, t], ZT_np[n, t], self.baseMVA, self.slack_pos)  # float64
                    # 白化
                    denom = np.sqrt(np.maximum(R_np[n, t], self.eps_var))
                    rinp  = (Z_np[n, t] - hx) / denom
                    inp[n, t] = torch.from_numpy(rinp.astype(np.float32))
            return inp

        elif self.input_mode == "z_std":
            # 优先使用 npz 内保存的 z_mean, z_std（若你在生成阶段保存了的话）
            if ("z_mean" in D.files) and ("z_std" in D.files):
                mu = torch.from_numpy(D["z_mean"]).float()   # [M] 或 [1,M]
                sd = torch.from_numpy(D["z_std"]).float()    # [M] 或 [1,M]
                while mu.dim() < 3: mu = mu.unsqueeze(0)
                while sd.dim() < 3: sd = sd.unsqueeze(0)
                return (self.Z - mu) / (sd.clamp_min(1e-6))
            else:
                # 用当前文件内统计（非跨文件），也能稳住训练
                mu = self.Z.mean(dim=(0, 1), keepdim=True)   # [1,1,M]
                sd = self.Z.std(dim=(0, 1), keepdim=True)    # [1,1,M]
                return (self.Z - mu) / (sd.clamp_min(1e-6))

        else:  # "z_raw"
            return self.Z.clone()

    def __len__(self):
        return self.Z.shape[0]

    def __getitem__(self, idx):
        sample = dict(
            # 模型主输入（默认是白化残差 r）
            z=self.INPUT[idx],                 # [W, M]
            # 监督目标
            x=self.X[idx],                     # [W, S]
            # 辅助信息 / 先验
            R=self.R[idx],                     # [W, M]
            mask=self.M[idx],                  # [W, M]
            A_time=self.A[idx],                # [W, W]
            E_time=self.E[idx],                # [W, W]
            feat=self.F[idx],                  # [W, F]
        )
        # 同时返回 x_wls / ztype 以便可选的投影层或物理损失
        if self.XWLS is not None:  sample["x_wls"]  = self.XWLS[idx]   # [W, 2N]
        if self.ZTYPE is not None: sample["ztype"]  = self.ZTYPE[idx]  # [W, 4, M]
        # meta 可选返回
        sample["meta"] = {"slack_pos": self.slack_pos, "baseMVA": self.baseMVA}
        return sample
