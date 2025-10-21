# train/train_tagru_supervised.py
# 版本 2：修复了 'ValueError'
# 关键改动：将 init_phys_meta 移出训练循环

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
from build_ieee33_with_pp import build_ieee33
from physics.ac_model import h_measure

# --- 物理损失函数 (保持不变) ---
_ybus, _baseMVA, _slack_pos = None, None, None
def init_phys_meta(slack_pos):
    global _ybus, _baseMVA, _slack_pos
    if _ybus is None:
        _, ybus, baseMVA, *_ = build_ieee33()
        _ybus = ybus.astype(np.complex128)
        _baseMVA = float(baseMVA)
    # ================== 关键改动 1: 修复 ValueError ==================
    # 确保 slack_pos 是一个标量整数
    _slack_pos = int(slack_pos) 
    # ==============================================================

def physics_loss(x_hat, z, R, ztype, eps=1e-9):
    # (此函数内容保持不变)
    B, W, M = z.shape
    loss = 0.0
    for b in range(B):
        for t in range(W):
            hx = h_measure(_ybus, x_hat[b,t].detach().cpu().numpy(),
                           ztype[b,t].cpu().numpy(), _baseMVA, _slack_pos)
            wres = (z[b,t].cpu().numpy() - hx) / np.sqrt(np.maximum(R[b,t].cpu().numpy(), eps))
            loss += float((wres**2).mean())
    return x_hat.new_tensor(loss / (B*W))

# --- 监督损失函数 (保持不变) ---
def theta_wrap(pred, gt):
    return (pred - gt + math.pi) % (2*math.pi) - math.pi

class StateLoss(nn.Module):
    # (此函数内容保持不变)
    def __init__(self, bus_count: int, w_theta: float = 2.0, w_vm: float = 1.0):
        super().__init__()
        self.N = bus_count
        self.wt, self.wv = w_theta, w_vm
    def forward(self, x_hat, x_true):
        N = self.N
        dth = theta_wrap(x_hat[..., :N], x_true[..., :N])
        dv  = x_hat[..., N:] - x_true[..., N:]
        l_th = (dth**2).mean()
        l_v  = (dv**2).mean()
        return self.wt*l_th + self.wv*l_v, (l_th.detach(), l_v.detach())

# --- RMSE 指标 (保持不变) ---
def rmse_metrics(x_hat, x_true):
    # (此函数内容保持不变)
    N = x_true.shape[-1] // 2
    dth = theta_wrap(x_hat[..., :N], x_true[..., :N])
    dv  = x_hat[..., N:] - x_true[..., N:]
    th_rmse = torch.sqrt((dth**2).mean()).item() * 180.0 / math.pi
    vm_rmse = torch.sqrt((dv**2).mean()).item()
    return th_rmse, vm_rmse

# --- 数据加载器 (保持不变) ---
def make_loader(npz_path, batch_size, shuffle, input_mode="raw"):
    ds = WindowDataset(npz_path, input_mode=input_mode)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)
    D = np.load(npz_path, allow_pickle=True)
    return dl, D

def main():
    ap = argparse.ArgumentParser()
    # (参数部分保持不变)
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
    ap.add_argument("--lambda_phys", type=float, default=0.1)
    ap.add_argument("--w_temp_th", type=float, default=0.01)
    ap.add_argument("--w_temp_vm", type=float, default=0.01)
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
    
    # ================== 关键改动 2: 在循环外初始化 ==================
    meta = Dtr["meta"].item() if "meta" in Dtr.files else {"bus_count": S//2, "slack_pos": 0}
    Nbus = int(meta.get("bus_count", S//2))
    # 从数据集元数据中获取 *单个* slack_pos
    slack_pos_val = int(meta.get("slack_pos", 0)) # 假设 IEEE33 slack 是 0
    # 在循环开始前 *一次性* 初始化物理引擎
    init_phys_meta(slack_pos_val)
    # ==============================================================

    model = TopoAlignGRU(
        meas_dim=M, feat_dim=F, state_dim=S,
        hidden_dim=args.hidden, num_layers=args.layers,
        nhead=args.nhead, bias_scale=args.bias_scale,
        use_mask=args.use_mask
    ).to(args.device)

    loss_fn = StateLoss(bus_count=Nbus, w_theta=2.0, w_vm=1.0)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5, factor=0.5)

    best_va = float("inf"); best = None

    for epoch in range(1, args.epochs+1):
        model.train()
        tot_sup, tot_lth, tot_lv, tot_phy, tot_tmp = 0.0, 0.0, 0.0, 0.0, 0.0
        
        for batch in dl_tr:
            z_seq = batch["z"].to(args.device)
            feat_seq = batch["feat"].to(args.device)
            A     = batch["A_time"].to(args.device)
            E     = batch["E_time"].to(args.device)
            x_gt  = batch["x"].to(args.device)

            x_hat = model(z_seq, feat_seq, A_time=A, E_time=E)
            sup_loss, (lth, lv) = loss_fn(x_hat, x_gt)

            # 物理损失
            # ================== 关键改动 3: 移除内部初始化 ==================
            if "ztype" in batch:
                # if _ybus is None: # 这一整块被移除了，因为已在外部初始化
                #     ...
                L_phys = physics_loss(x_hat, z_seq, batch["R"].to(args.device), batch["ztype"].to(args.device))
            else:
                L_phys = x_hat.new_zeros(())
            # ==============================================================
            
            L_temp = temporal_smooth(x_hat, args.w_temp_th, args.w_temp_vm)
            
            total = sup_loss + args.lambda_phys * L_phys + L_temp
            opt.zero_grad(); total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = z_seq.size(0)
            tot_sup += sup_loss.item()*bs
            tot_lth += lth.item()*bs
            tot_lv  += lv.item()*bs
            tot_phy += L_phys.item()*bs
            tot_tmp += L_temp.item()*bs

        # (训练/验证/测试的剩余部分保持不变)
        tr_sup = tot_sup/len(dl_tr.dataset); tr_lth=tot_lth/len(dl_tr.dataset); tr_lv=tot_lv/len(dl_tr.dataset)
        tr_phy = tot_phy/len(dl_tr.dataset); tr_tmp = tot_tmp/len(dl_tr.dataset)

        model.eval()
        with torch.no_grad():
            va_loss, ths, vms = 0.0, [], []
            for batch in dl_va:
                z_seq = batch["z"].to(args.device)
                feat_seq = batch["feat"].to(args.device)
                A     = batch["A_time"].to(args.device)
                E     = batch["E_time"].to(args.device)
                x_gt  = batch["x"].to(args.device)

                x_hat = model(z_seq, feat_seq, A_time=A, E_time=E)
                loss, _ = loss_fn(x_hat, x_gt)
                va_loss += loss.item()*z_seq.size(0)

                th_rmse, vm_rmse = rmse_metrics(x_hat, x_gt)
                ths.append(th_rmse); vms.append(vm_rmse)

            va_loss /= len(dl_va.dataset)
            th_m = float(np.mean(ths)); vm_m = float(np.mean(vms))

        sched.step(va_loss)
        print(f"[E{epoch:03d}] train={tr_sup:.4e} (θ={tr_lth:.4e},V={tr_lv:.4e}) Phys={tr_phy:.3f} Tmp={tr_tmp:.3f} | "
              f"val={va_loss:.4e}  θ-RMSE={th_m:.3f}°, |V|-RMSE={vm_m:.4f}")

        if va_loss < best_va:
            best_va = va_loss
            best = {"state_dict": model.state_dict(), "meta": {
                "Nbus": Nbus, "S": S, "M": M, "F": F,
                "hidden": args.hidden, "layers": args.layers, "nhead": args.nhead
            }}

    if best: model.load_state_dict(best["state_dict"])
    model.eval()
    with torch.no_grad():
        ths, vms = [], []
        for batch in dl_te:
            z_seq = batch["z"].to(args.device)
            feat_seq = batch["feat"].to(args.device)
            A     = batch["A_time"].to(args.device)
            E     = batch["E_time"].to(args.device)
            x_gt  = batch["x"].to(args.device)
            x_hat = model(z_seq, feat_seq, A_time=A, E_time=E)
            th_rmse, vm_rmse = rmse_metrics(x_hat, x_gt)
            ths.append(th_rmse); vms.append(vm_rmse)
        print(f"[TEST] θ-RMSE={np.mean(ths):.3f}°, |V|-RMSE={np.mean(vms):.4f}")

if __name__ == "__main__":
    main()