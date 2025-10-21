# eval/eval_physics_baseline.py
import os, numpy as np
import sys
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from build_ieee33_with_pp import build_ieee33
from physics.ac_model import h_measure

def phys_res_for_file(path, slack_pos=0):
    D = np.load(path)
    Z    = D["z"]        # [Nwin, W, M]
    R    = D["R"]        # [Nwin, W, M]
    Xwls = D["x_wls"]    # [Nwin, W, 2N]
    ZT   = D["ztype"]    # [Nwin, W, 4, M]  # 你存的是按窗口的；若是 [Nwin, 4, M] 就相应改索引

    # 网络参数（与生成时一致）
    _, ybus, baseMVA, *_ = build_ieee33()

    vals = []
    for i in range(Z.shape[0]):        # 遍历窗口
        for t in range(Z.shape[1]):    # 遍历时间步
            z    = Z[i, t]
            Rvec = R[i, t]
            x    = Xwls[i, t]
            ztyp = ZT[i, t]            # [4, M]
            h    = h_measure(ybus, x, ztyp, baseMVA, slack_pos)
            wres = (z - h) / np.sqrt(np.maximum(Rvec, 1e-12))
            vals.append( float((wres**2).mean()) )
    return np.mean(vals)

def main():
    base = "data/windows_ieee33"
    for tag in ["W24","W96"]:
        for split in ["train","val","test"]:
            p = os.path.join(base, f"{tag}_{split}.npz")
            if os.path.exists(p):
                v = phys_res_for_file(p, slack_pos=0)
                print(f"[{os.path.basename(p)}] 物理残差(带权)均值 = {v:.3f}")

if __name__ == "__main__":
    main()
