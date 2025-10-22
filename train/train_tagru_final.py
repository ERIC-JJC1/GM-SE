# train/train_tagru_final.py
#
# 这是最终的“残差精炼”训练脚本。
# 1. 模型预测 dx
# 2. x_hat = x_wls + dx
# 3. 唯一的损失是 PhysicsInformedLoss(x_hat, batch)

import os, math, argparse,sys
import numpy as np
import torch
import torch.nn as nn
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from torch.utils.data import DataLoader
from train.dataset import WindowDataset
from models.tagru import TopoAlignGRU
from tools.train_sweep_and_compare import temporal_smooth
from physics.losses import PhysicsInformedLoss

# --- 评估函数 (无变化) ---
def theta_wrap(pred, gt):
    return (pred - gt + math.pi) % (2*math.pi) - math.pi

def rmse_metrics(x_hat, x_true):
    x_hat_np = x_hat.detach().cpu().numpy()
    x_true_np = x_true.detach().cpu().numpy()
    N = x_true_np.shape[-1] // 2
    # 格式: [theta, vm]
    dth = theta_wrap(x_hat_np[..., :N], x_true_np[..., :N])
    dv  = x_hat_np[..., N:] - x_true_np[..., N:]
    th_rmse = np.sqrt(np.mean(dth**2)) * 180.0 / math.pi
    vm_rmse = np.sqrt(np.mean(dv**2))
    return float(th_rmse), float(vm_rmse)
# --------------------------------

def make_loader(npz_path, batch_size, shuffle, input_mode="raw"):
    ds = WindowDataset(npz_path, input_mode=input_mode)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)
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
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--bias_scale", type=float, default=3.0)
    ap.add_argument("--use_mask", action="store_true", default=False)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # ================== 关键改动 1: 恢复损失权重 ==================
    ap.add_argument("--lambda_op", type=float, default=1.0, help="权重: 运行约束 (Phi_op)")
    ap.add_argument("--w_temp_th", type=float, default=0.01)
    ap.add_argument("--w_temp_vm", type=float, default=0.01)
    # 移除了 --lambda_wls_reg
    # ==========================================================
    
    args = ap.parse_args()

    p_train = os.path.join(args.data_dir, f"{args.tag}_train.npz")
    p_val   = os.path.join(args.data_dir, f"{args.tag}_val.npz")
    p_test  = os.path.join(args.data_dir, f"{args.tag}_test.npz")

    dl_tr, Dtr = make_loader(p_train, args.batch_size, True,  input_mode="raw")
    dl_va, Dva = make_loader(p_val,   args.batch_size, False, input_mode="raw")
    dl_te, Dte = make_loader(p_test,  args.batch_size, False, input_mode="raw")

    S  = Dtr["x"].shape[2]
    F  = Dtr["feat"].shape[2]
    M  = Dtr["z"].shape[2]
    
    meta = Dtr["meta"].item() if "meta" in Dtr.files else {"bus_count": S//2, "slack_pos": 0}
    Nbus = int(meta.get("bus_count", S//2))
    slack_pos_val = int(meta.get("slack_pos", 0))

    # ================== 关键改动 2: 实例化模型和损失 ==================
    model = TopoAlignGRU(
        meas_dim=M, feat_dim=F, state_dim=S, # S = 2N, state_dim 现在是输入的一部分
        hidden_dim=args.hidden, num_layers=args.layers,
        nhead=args.nhead, bias_scale=args.bias_scale,
        use_mask=args.use_mask
    ).to(args.device)

    # 唯一的损失函数
    phys_loss_fn = PhysicsInformedLoss(
        nbus=Nbus, 
        slack_pos=slack_pos_val, 
        lambda_op=args.lambda_op
    ).to(args.device)
    
    # 移除了 wls_anchor_loss_fn
    # ===================================================================

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5, factor=0.5)

    best_va = float("inf"); best = None

    for epoch in range(1, args.epochs+1):
        model.train()
        tot_res, tot_op, tot_tmp = 0.0, 0.0, 0.0
        
        for batch in dl_tr:
            z_seq = batch["z"].to(args.device)
            feat_seq = batch["feat"].to(args.device)
            x_wls = batch["x_wls"].to(args.device) # !! 加载 x_wls 作为输入 !!
            A     = batch["A_time"].to(args.device)
            E     = batch["E_time"].to(args.device)
            
            batch_gpu = {
                "z": z_seq,
                "R": batch["R"].to(args.device),
                "ztype": batch["ztype"]
            }

            # ================== 关键改动 3: 新的模型调用 ==================
            # 模型现在预测 x_hat = x_wls + dx
            x_hat = model(z_seq, feat_seq, x_wls, A_time=A, E_time=E)
            # =============================================================

            # ================== 关键改动 4: 唯一的损失 ==================
            # 移除了 wls_reg_loss
            
            # 损失 1: 物理损失 (唯一目标)
            phys_loss, loss_dict = phys_loss_fn(x_hat, batch_gpu)
            
            # (可选) 损失 2: 时间平滑损失
            L_temp = temporal_smooth(x_hat, args.w_temp_th, args.w_temp_vm)
            
            # 最终总损失
            total_loss = phys_loss + L_temp
            # =============================================================
            
            opt.zero_grad(); total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = z_seq.size(0)
            tot_res += loss_dict["loss_residual"].item()*bs
            tot_op  += loss_dict["loss_op"].item()*bs
            tot_tmp += L_temp.item()*bs

        tr_res = tot_res/len(dl_tr.dataset)
        tr_op  = tot_op/len(dl_tr.dataset)
        tr_tmp = tot_tmp/len(dl_tr.dataset)
        tr_tot = tr_res + tr_op + tr_tmp

        # ---- val ----
        model.eval()
        with torch.no_grad():
            va_loss_phys = 0.0
            ths, vms = [], []
            
            for batch in dl_va:
                z_seq = batch["z"].to(args.device)
                feat_seq = batch["feat"].to(args.device)
                x_wls = batch["x_wls"].to(args.device) # !! 作为输入传递 !!
                A     = batch["A_time"].to(args.device)
                E     = batch["E_time"].to(args.device)
                x_gt  = batch["x"].to(args.device)
                
                batch_gpu = { "z": z_seq, "R": batch["R"].to(args.device), "ztype": batch["ztype"] }

                x_hat = model(z_seq, feat_seq, x_wls, A_time=A, E_time=E)
                
                val_loss_batch, _ = phys_loss_fn(x_hat, batch_gpu)
                va_loss_phys += val_loss_batch.item()*z_seq.size(0)

                th_rmse, vm_rmse = rmse_metrics(x_hat, x_gt)
                ths.append(th_rmse); vms.append(vm_rmse)

            va_loss = va_loss_phys / len(dl_va.dataset)
            th_m = float(np.mean(ths)); vm_m = float(np.mean(vms))

        sched.step(va_loss)
        # ================== 关键改动 5: 更新日志 ==================
        print(f"[E{epoch:03d}] train_loss={tr_tot:.4e} (Res={tr_res:.4e},Op={tr_op:.4e}) | "
              f"val_loss(phys)={va_loss:.4e}  | "
              f"EVAL: θ-RMSE={th_m:.3f}°, |V|-RMSE={vm_m:.4f}")
        # ========================================================

        if va_loss < best_va:
            best_va = va_loss
            best = {"state_dict": model.state_dict(), "meta": {
                "Nbus": Nbus, "S": S, "M": M, "F": F,
                "hidden": args.hidden, "layers": args.layers, "nhead": args.nhead
            }}

    # ---- test ----
    if best: model.load_state_dict(best["state_dict"])
    model.eval()
    with torch.no_grad():
        ths, vms = [], []
        for batch in dl_te:
            z_seq = batch["z"].to(args.device)
            feat_seq = batch["feat"].to(args.device)
            x_wls = batch["x_wls"].to(args.device) # !! 作为输入传递 !!
            A     = batch["A_time"].to(args.device)
            E     = batch["E_time"].to(args.device)
            x_gt  = batch["x"].to(args.device)
            
            x_hat = model(z_seq, feat_seq, x_wls, A_time=A, E_time=E)
            
            th_rmse, vm_rmse = rmse_metrics(x_hat, x_gt)
            ths.append(th_rmse); vms.append(vm_rmse)
            
        print(f"[TEST] θ-RMSE={np.mean(ths):.3f}°, |V|-RMSE={np.mean(vms):.4f}")

if __name__ == "__main__":
    main()