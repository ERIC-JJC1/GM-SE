# tools/train_sweep_and_compare.py
import os, sys, math, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# --- import project modules ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from train.dataset import WindowDataset
from models.refine_seq import RefineSeqTAModel

# ---------- utils ----------
def theta_wrap(pred, gt):
    return (pred - gt + math.pi) % (2*math.pi) - math.pi

def rmse_metrics(x_hat, x_true):
    """返回 θ-RMSE(deg), |V|-RMSE(p.u.)"""
    N = x_true.shape[-1] // 2
    dth = theta_wrap(x_hat[..., :N], x_true[..., :N])
    dv  = x_hat[..., N:] - x_true[..., N:]
    th_rmse = torch.sqrt((dth**2).mean()).item() * 180.0 / math.pi
    vm_rmse = torch.sqrt((dv**2).mean()).item()
    return th_rmse, vm_rmse

class StateLoss(nn.Module):
    def __init__(self, bus_count: int, w_theta: float = 2.0, w_vm: float = 1.0, trust_lambda: float = 0.0):
        super().__init__()
        self.N = bus_count
        self.wt, self.wv = w_theta, w_vm
        self.trust_lambda = trust_lambda  # 信任域正则系数
    def forward(self, x_hat, x_true, x_wls=None):
        N = self.N
        dth = theta_wrap(x_hat[..., :N], x_true[..., :N])
        dv  = x_hat[..., N:] - x_true[..., N:]
        l_th = (dth**2).mean()
        l_v  = (dv**2).mean()
        loss = self.wt*l_th + self.wv*l_v
        if (x_wls is not None) and (self.trust_lambda > 0):
            dx = (x_hat - x_wls)
            loss = loss + self.trust_lambda * (dx**2).mean()
        return loss, (l_th.detach(), l_v.detach())

def make_loader(npz_path, batch_size, shuffle, input_mode="whiten"):
    ds = WindowDataset(npz_path, input_mode=input_mode)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)
    D = np.load(npz_path, allow_pickle=True)
    return dl, D

@torch.no_grad()
def compute_wls_baseline(npz_path):
    """在指定文件上计算 WLS baseline（用 x_wls 和 x 的差）"""
    D = np.load(npz_path, allow_pickle=True)
    if ("x_wls" not in D.files) or ("x" not in D.files):
        return None
    Xw = torch.from_numpy(D["x"]).float()      # [N,W,2N]
    Xb = torch.from_numpy(D["x_wls"]).float()  # [N,W,2N]
    th, vm = rmse_metrics(Xb, Xw)
    return {"theta_deg": th, "|V|": vm}

def train_one_config(
    data_dir="data/windows_ieee33",
    tag="W96",
    epochs=60,
    batch_size=16,
    lr=1e-3,
    hidden=256,
    layers=2,
    nhead=4,
    bias_scale=3.0,
    use_mask=False,
    trust_lambda=0.0,
    w_temp_th=0.0,
    w_temp_vm=0.0,
    use_bus_smooth=True, 
    use_gnn=False, gnn_type="tag", gnn_hidden=128, gnn_layers=2, gnn_dropout=0.0,
    device=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    p_train = os.path.join(data_dir, f"{tag}_train.npz")
    p_val   = os.path.join(data_dir, f"{tag}_val.npz")
    p_test  = os.path.join(data_dir, f"{tag}_test.npz")

    # loaders (whitened residual input)
    dl_tr, Dtr = make_loader(p_train, batch_size, True,  input_mode="whiten")
    dl_va, Dva = make_loader(p_val,   batch_size, False, input_mode="whiten")
    dl_te, Dte = make_loader(p_test,  batch_size, False, input_mode="whiten")

    S  = Dtr["x"].shape[2]   # state_dim
    F  = Dtr["feat"].shape[2]
    M  = Dtr["z"].shape[2]   # here 'z' is whitened residual dimension
    meta = Dtr["meta"].item() if "meta" in Dtr.files else {"bus_count": S//2}
    Nbus = int(meta.get("bus_count", S//2))


    # model    
    model = RefineSeqTAModel(
        meas_dim=M, state_dim=S, feat_dim=F,
        hidden=hidden, num_layers=layers,
        nhead=nhead, bias_scale=bias_scale,
        use_mask=use_mask,
        use_bus_smooth=use_bus_smooth,
        use_gnn=use_gnn, gnn_type=gnn_type,
        gnn_hidden=gnn_hidden, gnn_layers=gnn_layers, gnn_dropout=gnn_dropout
    ).to(device)


    loss_fn = StateLoss(bus_count=Nbus, w_theta=2.0, w_vm=1.0, trust_lambda=trust_lambda)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5, factor=0.5)

    best_va = float("inf"); best_sd = None

    for epoch in range(1, epochs+1):
        # ---- train ----
        model.train()
        tot, tot_lth, tot_lv = 0.0, 0.0, 0.0
        for batch in dl_tr:
            r     = batch["z"].to(device)          # whitened residual
            x_wls = batch.get("x_wls", None)
            if x_wls is not None: x_wls = x_wls.to(device)
            feat  = batch["feat"].to(device)
            A     = batch["A_time"].to(device)
            E     = batch["E_time"].to(device)
            x_gt  = batch["x"].to(device)

            x_hat = model(r, x_wls if x_wls is not None else torch.zeros_like(x_gt), feat, A_time=A, E_time=E)
            loss, (lth, lv) = loss_fn(x_hat, x_gt, x_wls)
            reg_temp = temporal_smooth(x_hat, w_th=w_temp_th, w_vm=w_temp_vm)
            loss = loss + reg_temp
            
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = r.size(0)
            tot    += loss.item()*bs
            tot_lth+= lth.item()*bs
            tot_lv += lv.item()*bs

        tr_loss = tot/len(dl_tr.dataset); tr_lth=tot_lth/len(dl_tr.dataset); tr_lv=tot_lv/len(dl_tr.dataset)

        # ---- val ----
        model.eval()
        with torch.no_grad():
            va_loss, ths, vms = 0.0, [], []
            for batch in dl_va:
                r     = batch["z"].to(device)
                x_wls = batch.get("x_wls", None)
                if x_wls is not None: x_wls = x_wls.to(device)
                feat  = batch["feat"].to(device)
                A     = batch["A_time"].to(device)
                E     = batch["E_time"].to(device)
                x_gt  = batch["x"].to(device)
                
                x_hat = model(r, x_wls if x_wls is not None else torch.zeros_like(x_gt), feat, A_time=A, E_time=E)
                loss, _ = loss_fn(x_hat, x_gt, x_wls)
                reg_temp = temporal_smooth(x_hat, w_th=w_temp_th, w_vm=w_temp_vm)
                loss = loss + reg_temp
                va_loss += loss.item()*r.size(0)

                th_rmse, vm_rmse = rmse_metrics(x_hat, x_gt)
                ths.append(th_rmse); vms.append(vm_rmse)

            va_loss /= len(dl_va.dataset)
            th_m = float(np.mean(ths)); vm_m = float(np.mean(vms))

        sched.step(va_loss)
        # 可选：打印每N轮
        if (epoch % 5 == 0) or (epoch == 1) or (epoch == epochs):
            print(f"[E{epoch:03d}] train={tr_loss:.4e} (θ={tr_lth:.4e},V={tr_lv:.4e}) | "
                  f"val={va_loss:.4e}  θ-RMSE={th_m:.3f}°, |V|-RMSE={vm_m:.4f}")

        if va_loss < best_va:
            best_va = va_loss
            best_sd = {k: v.cpu() for k, v in model.state_dict().items()}

    # ---- test ----
    if best_sd is not None:
        model.load_state_dict(best_sd)

    model.eval()
    with torch.no_grad():
        ths, vms = [], []
        for batch in dl_te:
            r     = batch["z"].to(device)
            x_wls = batch.get("x_wls", None)
            if x_wls is not None: x_wls = x_wls.to(device)
            feat  = batch["feat"].to(device)
            A     = batch["A_time"].to(device)
            E     = batch["E_time"].to(device)
            x_gt  = batch["x"].to(device)
            x_hat = model(r, x_wls if x_wls is not None else torch.zeros_like(x_gt), feat, A_time=A, E_time=E)
            th_rmse, vm_rmse = rmse_metrics(x_hat, x_gt)
            ths.append(th_rmse); vms.append(vm_rmse)
        test_th = float(np.mean(ths))
        test_vm = float(np.mean(vms))

    return {
        "val_loss": float(best_va),
        "test_theta_deg": test_th,
        "test_|V|": test_vm,
        "S": S, "M": M, "Nbus": Nbus
    }

def temporal_smooth(x_hat, w_th: float, w_vm: float):
    """
    x_hat: (B, W, 2N), 预测的 [theta_rel, |V|]
    返回: 标量正则项
    """
    if (w_th == 0.0) and (w_vm == 0.0):
        # 省开销：直接返回 0 标量
        return x_hat.new_zeros(())
    dt = x_hat[:, 1:, :] - x_hat[:, :-1, :]        # (B, W-1, 2N)
    N  = dt.shape[-1] // 2
    # 角度做 wrap，避免 2π 跳变被当作大差分
    dth = (dt[..., :N] + math.pi) % (2*math.pi) - math.pi
    dv  = dt[..., N:]
    reg = w_th * (dth**2).mean() + w_vm * (dv**2).mean()
    return reg

def main():
    import argparse, itertools, csv
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/windows_ieee33")
    ap.add_argument("--tag", type=str, default="W24")
    ap.add_argument("--epochs", type=int, nargs="+", default=[60])          # 支持列表
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, nargs="+", default=[1e-3, 3e-4])
    ap.add_argument("--hidden", type=int, nargs="+", default=[256])
    ap.add_argument("--layers", type=int, nargs="+", default=[2])
    ap.add_argument("--nhead", type=int, nargs="+", default=[4])
    ap.add_argument("--bias_scale", type=float, nargs="+", default=[0.0, 2.0, 3.0, 4.0])
    ap.add_argument("--use_mask", action="store_true", default=False)
    ap.add_argument("--trust_lambda", type=float, nargs="+", default=[0.0, 1e-3])
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--lambda_phys", type=float, default=1e-3,help="物理一致性损失权重")    
    ap.add_argument("--no_bus_smooth", action="store_true", default=False)
    ap.add_argument("--w_temp_th", type=float, default=0.0,help="θ 的时间平滑权重")
    ap.add_argument("--w_temp_vm", type=float, default=0.0,help="θ 的时间平滑权重")    
    ap.add_argument("--use_gnn", action="store_true", default=False)
    ap.add_argument("--gnn_type", type=str, default="tag", choices=["tag","gcn2","gat"])
    ap.add_argument("--gnn_hidden", type=int, nargs="+", default=[128])
    ap.add_argument("--gnn_layers", type=int, nargs="+", default=[2])
    ap.add_argument("--gnn_dropout", type=float, nargs="+", default=[0.0])
    ap.add_argument("--sweep_use_gnn", type=int, nargs="+", default=[0, 1], help="0=off, 1=on")
    ap.add_argument("--sweep_use_mask", type=int, nargs="+", default=[1], help="0=off, 1=on")
    ap.add_argument("--sweep_bus_smooth", type=int, nargs="+", default=[1], help="0=off, 1=on")
    ap.add_argument("--gnn_type_list", type=str, nargs="+", default=["tag"], choices=["tag","gcn2","gat"])

    
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.tag

    # 1) 先算 WLS baseline（test）
    wls_path = os.path.join(args.data_dir, f"{tag}_test.npz")
    wls_baseline = compute_wls_baseline(wls_path)
    if wls_baseline is None:
        print("[WARN] 该数据集不含 x_wls，无法计算 WLS baseline。")
    else:
        print(f"[WLS baseline | {tag}_test] θ={wls_baseline['theta_deg']:.3f}°, |V|={wls_baseline['|V|']:.4f}")

    # 2) sweep
    grid = list(itertools.product(
        args.epochs, args.lr, args.hidden, args.layers, args.nhead,
        args.bias_scale, args.trust_lambda,
        args.gnn_hidden, args.gnn_layers, args.gnn_dropout,
        args.sweep_use_gnn, args.sweep_use_mask, args.sweep_bus_smooth, args.gnn_type_list
    ))


    results = []
    t0 = time.time()

    for (E, LR, H, L, Hn, BS, TR, GH, GL, GD, UG, UM, UBS, GT) in grid:
        use_gnn = bool(UG)
        use_mask = bool(UM)
        use_bus_smooth = bool(UBS)

        cfg = dict(
            epochs=E, lr=LR, hidden=H, layers=L, nhead=Hn,
            bias_scale=BS, trust_lambda=TR,
            use_mask=use_mask,
            use_bus_smooth=use_bus_smooth,
            use_gnn=use_gnn, gnn_type=GT,
            gnn_hidden=GH, gnn_layers=GL, gnn_dropout=GD
        )

        out = train_one_config(
            data_dir=args.data_dir, tag=tag,
            epochs=E, batch_size=args.batch_size, lr=LR,
            hidden=H, layers=L, nhead=Hn, bias_scale=BS,
            use_mask=use_mask, trust_lambda=TR,
            use_bus_smooth=use_bus_smooth,
            use_gnn=use_gnn, gnn_type=GT,
            gnn_hidden=GH, gnn_layers=GL, gnn_dropout=GD,
            device=args.device
        )

        row = {
            "tag": tag,
            **cfg,
            **out,
            "w_temp_th": args.w_temp_th,
            "w_temp_vm": args.w_temp_vm,
        }
        if wls_baseline is not None:
            row["wls_theta_deg"] = wls_baseline["theta_deg"]
            row["wls_|V|"] = wls_baseline["|V|"]

        results.append(row)


    # 3) 保存结果
    csv_path = os.path.join(args.out_dir, f"ta_gru_sweep_{tag}.csv")
    json_path = os.path.join(args.out_dir, f"ta_gru_sweep_{tag}.json")

    # CSV
    keys = sorted({k for r in results for k in r.keys()})
    with open(csv_path, "w", newline="") as f:
        import csv
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(results)

    # JSON
    with open(json_path, "w") as f:
        json.dump({"wls_baseline": wls_baseline, "results": results}, f, indent=2)

    dt = time.time() - t0
    print(f"\n[DONE] saved:\n  - {csv_path}\n  - {json_path}\nElapsed: {dt/60:.1f} min")

if __name__ == "__main__":
    main()
