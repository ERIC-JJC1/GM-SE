# train/train_refine_baseline.py (已修改)

import os, math, argparse,sys
import numpy as np
import torch
import torch.nn as nn
# [已删除] 移除了 sys.path 注入代码
from torch.utils.data import DataLoader
from train.dataset import WindowDataset
from models.refine_seq import RefineSeqTAModel   # 改这里

from tools.train_sweep_and_compare import temporal_smooth
from build_ieee33_with_pp import build_ieee33
from physics.ac_model import h_measure

# [已修改] 从 tools.metrics 导入
from tools.metrics import StateLoss, rmse_metrics 

_ybus, _baseMVA, _slack_pos = None, None, None
def init_phys_meta(slack_pos):
    global _ybus, _baseMVA, _slack_pos
    if _ybus is None:
        _, ybus, baseMVA, *_ = build_ieee33()
        _ybus = ybus.astype(np.complex128)
        _baseMVA = float(baseMVA)
    _slack_pos = int(slack_pos)

def physics_loss(x_hat, z, R, ztype, eps=1e-9):
    """
    x_hat: (B,W,2N)  模型输出
    z,R  : (B,W,M)
    ztype: (B,W,4,M)
    """
    B, W, M = z.shape
    loss = 0.0
    for b in range(B):
        for t in range(W):
            # 用 numpy 双精度算 h(x)
            hx = h_measure(_ybus, x_hat[b,t].detach().cpu().numpy(),
                           ztype[b,t].cpu().numpy(), _baseMVA, _slack_pos)
            # 带权平方残差
            wres = (z[b,t].cpu().numpy() - hx) / np.sqrt(np.maximum(R[b,t].cpu().numpy(), eps))
            loss += float((wres**2).mean())
    return x_hat.new_tensor(loss / (B*W))

# [已删除] 移除了本地定义的 theta_wrap, StateLoss, rmse_metrics

def make_loader(npz_path, batch_size, shuffle, input_mode="whiten"):
    ds = WindowDataset(npz_path, input_mode=input_mode)   # 传路径+选择白化输入
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)
    # 为了拿维度/元数据，读一次 npz
    D = np.load(npz_path, allow_pickle=True)
    return dl, D

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/windows_ieee33")
    ap.add_argument("--tag", type=str, default="W24")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    # 注意力超参
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--bias_scale", type=float, default=3.0)
    ap.add_argument("--use_mask", action="store_true", default=False)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # [新增] 物理损失和时间损失的权重 (在原脚本中缺失，补充默认值)
    ap.add_argument("--lambda_phys", type=float, default=0.0)
    ap.add_argument("--w_temp_th", type=float, default=0.0)
    ap.add_argument("--w_temp_vm", type=float, default=0.0)
    
    args = ap.parse_args()

    p_train = os.path.join(args.data_dir, f"{args.tag}_train.npz")
    p_val   = os.path.join(args.data_dir, f"{args.tag}_val.npz")
    p_test  = os.path.join(args.data_dir, f"{args.tag}_test.npz")

    dl_tr, Dtr = make_loader(p_train, args.batch_size, True,  input_mode="whiten")
    dl_va, Dva = make_loader(p_val,   args.batch_size, False, input_mode="whiten")
    dl_te, Dte = make_loader(p_test,  args.batch_size, False, input_mode="whiten")

    S  = Dtr["x"].shape[2]
    F  = Dtr["feat"].shape[2]
    M  = Dtr["z"].shape[2]          # 此时 batch["z"] 就是 r（白化残差）的维度 M
    meta = Dtr["meta"].item() if "meta" in Dtr.files else {"bus_count": S//2}
    Nbus = int(meta.get("bus_count", S//2))

    # === model ===
    model = RefineSeqTAModel(
        meas_dim=M, state_dim=S, feat_dim=F,
        hidden=args.hidden, num_layers=args.layers,
        nhead=args.nhead, bias_scale=args.bias_scale,
        use_mask=args.use_mask
    ).to(args.device)

    # [已修改] 使用从 metrics 导入的 StateLoss
    # 指定 state_order='va_vm' (refine-wls 的输出顺序：角度在前)
    loss_fn = StateLoss(bus_count=Nbus, w_theta=2.0, w_vm=1.0, state_order='va_vm').to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5, factor=0.5)

    best_va = float("inf"); best = None

    for epoch in range(1, args.epochs+1):
        # ---- train ----
        model.train()
        tot, tot_lth, tot_lv = 0.0, 0.0, 0.0
        for batch in dl_tr:
            r     = batch["z"].to(args.device)         # 已是白化残差
            x_wls = batch["x_wls"].to(args.device)     # 需要你在 npz 里有 x_wls
            feat  = batch["feat"].to(args.device)
            A     = batch["A_time"].to(args.device)
            E     = batch["E_time"].to(args.device)
            x_gt  = batch["x"].to(args.device)

            x_hat = model(r, x_wls, feat, A_time=A, E_time=E)
            sup_loss, (lth, lv) = loss_fn(x_hat, x_gt)

            # 物理损失
            if "ztype" in batch:  # 你的 dataset 会给
                if _ybus is None:
                    # [已修改] 假设 meta 在 batch 中
                    meta_batch = batch["meta"]  # slack_pos 等
                    # 假设 'slack_pos' 是一个 tensor 或 list，取第一个元素
                    init_phys_meta(meta_batch["slack_pos"][0])
                L_phys = physics_loss(x_hat, batch["z"], batch["R"], batch["ztype"])
            else:
                L_phys = x_hat.new_zeros(())            
                
            L_temp = temporal_smooth(x_hat, args.w_temp_th, args.w_temp_vm)           
            
            total = sup_loss + args.lambda_phys * L_phys + L_temp
            opt.zero_grad(); total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = r.size(0)
            tot    += sup_loss.item()*bs # [已修改] 使用 sup_loss.item()
            tot_lth+= lth.item()*bs
            tot_lv += lv.item()*bs

        tr_loss = tot/len(dl_tr.dataset); tr_lth=tot_lth/len(dl_tr.dataset); tr_lv=tot_lv/len(dl_tr.dataset)

        # ---- val ----
        model.eval()
        with torch.no_grad():
            va_loss, ths, vms = 0.0, [], []
            for batch in dl_va:
                r     = batch["z"].to(args.device)
                x_wls = batch["x_wls"].to(args.device)
                feat  = batch["feat"].to(args.device)
                A     = batch["A_time"].to(args.device)
                E     = batch["E_time"].to(args.device)
                x_gt  = batch["x"].to(args.device)

                x_hat = model(r, x_wls, feat, A_time=A, E_time=E)
                loss, _ = loss_fn(x_hat, x_gt)
                va_loss += loss.item()*r.size(0)

                # [已修改] 指定 state_order='va_vm'
                th_rmse, vm_rmse = rmse_metrics(x_hat, x_gt, state_order='va_vm')
                ths.append(th_rmse); vms.append(vm_rmse)

            va_loss /= len(dl_va.dataset)
            th_m = float(np.mean(ths)); vm_m = float(np.mean(vms))

        sched.step(va_loss)
        print(f"[E{epoch:03d}] train={tr_loss:.4e} (θ={tr_lth:.4e},V={tr_lv:.4e}) | "
              f"val={va_loss:.4e}  θ-RMSE={th_m:.3f}°, |V|-RMSE={vm_m:.4f}")

        if va_loss < best_va:
            best_va = va_loss
            best = {"state_dict": model.state_dict(), "meta": {"Nbus": Nbus, "S": S}}

    # ---- test ----
    if best: model.load_state_dict(best["state_dict"])
    model.eval()
    with torch.no_grad():
        ths, vms = [], []
        for batch in dl_te:
            r     = batch["z"].to(args.device)
            x_wls = batch["x_wls"].to(args.device)
            feat  = batch["feat"].to(args.device)
            A     = batch["A_time"].to(args.device)
            E     = batch["E_time"].to(args.device)
            x_gt  = batch["x"].to(args.device)
            x_hat = model(r, x_wls, feat, A_time=A, E_time=E)
            # [已修改] 指定 state_order='va_vm'
            th_rmse, vm_rmse = rmse_metrics(x_hat, x_gt, state_order='va_vm')
            ths.append(th_rmse); vms.append(vm_rmse)
        print(f"[TEST] θ-RMSE={np.mean(ths):.3f}°, |V|-RMSE={np.mean(vms):.4f}")

if __name__ == "__main__":
    main()