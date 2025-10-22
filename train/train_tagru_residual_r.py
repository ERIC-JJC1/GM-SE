# train/train_tagru_residual_r.py
#
# 版本 3: 封装核心逻辑到 train_evaluate 函数，方便被 sweep 脚本调用

import os, math, argparse, sys
import numpy as np
import torch
import torch.nn as nn
import wandb # 导入 wandb 用于日志记录

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from torch.utils.data import DataLoader
from train.dataset import WindowDataset
from models.tagru import TopoAlignGRU
from tools.train_sweep_and_compare import temporal_smooth
from physics.losses import PhysicsInformedLoss
from physics.ac_model import h_measure
from build_ieee33_with_pp import build_ieee33

# --- 辅助函数 (无变化) ---
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
def make_loader(npz_path, batch_size, shuffle, input_mode="raw"):
    ds = WindowDataset(npz_path, input_mode=input_mode); dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)
    D = np.load(npz_path, allow_pickle=True); return dl, D
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
            x_wls_t = x_wls_seq_np[b, t]; h_wls_t = h_measure(_ybus_np, x_wls_t, ztype_np[b, t], _baseMVA_np, _slack_pos_np)
            res_t = z_seq_np[b, t] - h_wls_t; sigma_t = np.sqrt(np.maximum(R_seq_np[b, t], 1e-9)); r_w_t = res_t / sigma_t
            r_w_list.append(r_w_t[np.newaxis, :])
        r_seq_list.append(np.concatenate(r_w_list, axis=0)[np.newaxis, :, :])
    return np.concatenate(r_seq_list, axis=0)
# --------------------------------

# ================== 关键改动 1: 封装核心逻辑 ==================
def train_evaluate(args, use_wandb=False):
    """
    执行一次完整的训练和评估流程。

    Args:
        args (Namespace): 包含所有超参数和配置的对象。
        use_wandb (bool): 是否使用 wandb 记录日志。

    Returns:
        dict: 包含最终测试指标的字典 (e.g., {'test_th_rmse': ..., 'test_vm_rmse': ...})
    """
    if use_wandb:
        wandb.init(config=args)
        # 如果使用 wandb sweep，wandb 会自动更新 args
        args = wandb.config

    p_train = os.path.join(args.data_dir, f"{args.tag}_train.npz")
    p_val   = os.path.join(args.data_dir, f"{args.tag}_val.npz")
    p_test  = os.path.join(args.data_dir, f"{args.tag}_test.npz")

    dl_tr, Dtr = make_loader(p_train, args.batch_size, True,  input_mode="raw")
    dl_va, Dva = make_loader(p_val,   args.batch_size, False, input_mode="raw")
    dl_te, Dte = make_loader(p_test,  args.batch_size, False, input_mode="raw")

    S  = Dtr["x"].shape[2]; F  = Dtr["feat"].shape[2]; M  = Dtr["z"].shape[2]
    meta = Dtr["meta"].item() if "meta" in Dtr.files else {"bus_count": S//2, "slack_pos": 0}
    Nbus = int(meta.get("bus_count", S//2)); slack_pos_val = int(meta.get("slack_pos", 0))
    init_phys_meta_np(slack_pos_val)

    model = TopoAlignGRU( # 模型是版本 4
        meas_dim=M, feat_dim=F, state_dim=S,
        hidden_dim=args.hidden, num_layers=args.layers,
        nhead=args.nhead, bias_scale=args.bias_scale, use_mask=args.use_mask
    ).to(args.device)

    phys_loss_fn = PhysicsInformedLoss(
        nbus=Nbus, slack_pos=slack_pos_val,
        lambda_pf=args.lambda_pf, lambda_op=args.lambda_op, lambda_smooth=args.lambda_smooth
    ).to(args.device)
    wls_anchor_loss_fn = StateLoss(
        bus_count=Nbus, w_theta=2.0, w_vm=1.0
    ).to(args.device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5, factor=0.5)

    best_va = float("inf"); best_epoch = -1

    for epoch in range(1, args.epochs+1):
        model.train()
        log_train = {}
        # ... (训练循环内部逻辑不变, 但累加到 log_train 字典) ...
        tot_res, tot_pf, tot_op, tot_smooth, tot_wls, tot_tmp = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for batch in dl_tr:
            z_seq = batch["z"].to(args.device); feat_seq = batch["feat"].to(args.device)
            x_wls = batch["x_wls"].to(args.device); A = batch["A_time"].to(args.device)
            E = batch["E_time"].to(args.device); R_seq = batch["R"].to(args.device)
            ztype = batch["ztype"]
            r_seq_np = compute_whitened_residual(z_seq.cpu().numpy(), x_wls.cpu().numpy(), R_seq.cpu().numpy(), ztype)
            r_seq = torch.from_numpy(r_seq_np).float().to(args.device)
            batch_gpu = {"z": z_seq, "R": R_seq, "ztype": ztype}
            x_hat = model(r_seq, feat_seq, x_wls, A_time=A, E_time=E)
            phys_loss, loss_dict = phys_loss_fn(x_hat, batch_gpu)
            wls_reg_loss, _ = wls_anchor_loss_fn(x_hat, x_wls)
            L_temp = temporal_smooth(x_hat, args.w_temp_th, args.w_temp_vm)
            total_loss = phys_loss + args.lambda_wls_reg * wls_reg_loss + L_temp
            opt.zero_grad(); total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bs = r_seq.size(0)
            tot_res += loss_dict["loss_residual"].item()*bs; tot_pf += loss_dict["loss_pf"].item()*bs
            tot_op += loss_dict["loss_op"].item()*bs; tot_smooth += loss_dict["loss_smooth"].item()*bs
            tot_wls += wls_reg_loss.item()*bs; tot_tmp += L_temp.item()*bs

        log_train["tr_res"] = tot_res/len(dl_tr.dataset); log_train["tr_pf"] = tot_pf/len(dl_tr.dataset)
        log_train["tr_op"] = tot_op/len(dl_tr.dataset); log_train["tr_smooth"] = tot_smooth/len(dl_tr.dataset)
        log_train["tr_wls"] = tot_wls/len(dl_tr.dataset); log_train["tr_tmp"] = tot_tmp/len(dl_tr.dataset)
        log_train["tr_total_approx"] = sum(log_train.values())

        model.eval()
        log_val = {}
        with torch.no_grad():
            va_loss_phys = 0.0; ths, vms = [], []
            for batch in dl_va:
                # ... (验证循环内部逻辑不变) ...
                z_seq = batch["z"].to(args.device); feat_seq = batch["feat"].to(args.device)
                x_wls = batch["x_wls"].to(args.device); A = batch["A_time"].to(args.device)
                E = batch["E_time"].to(args.device); R_seq = batch["R"].to(args.device)
                ztype = batch["ztype"]; x_gt  = batch["x"].to(args.device)
                r_seq_np = compute_whitened_residual(z_seq.cpu().numpy(), x_wls.cpu().numpy(), R_seq.cpu().numpy(), ztype)
                r_seq = torch.from_numpy(r_seq_np).float().to(args.device)
                batch_gpu = {"z": z_seq, "R": R_seq, "ztype": ztype}
                x_hat = model(r_seq, feat_seq, x_wls, A_time=A, E_time=E)
                val_loss_batch, _ = phys_loss_fn(x_hat, batch_gpu)
                va_loss_phys += val_loss_batch.item()*r_seq.size(0)
                th_rmse, vm_rmse = rmse_metrics(x_hat, x_gt); ths.append(th_rmse); vms.append(vm_rmse)

            log_val["val_loss"] = va_loss_phys / len(dl_va.dataset) # 使用物理损失作为验证指标
            log_val["val_th_rmse"] = float(np.mean(ths))
            log_val["val_vm_rmse"] = float(np.mean(vms))

        sched.step(log_val["val_loss"])

        # 打印日志
        print(f"[E{epoch:03d}] tr_loss={log_train['tr_total_approx']:.3f} (Res={log_train['tr_res']:.3f},Pf={log_train['tr_pf']:.3f},Op={log_train['tr_op']:.3f},Sm={log_train['tr_smooth']:.3f},WLS={log_train['tr_wls']:.3f}) | "
              f"val_loss(phys)={log_val['val_loss']:.4e}  | EVAL: θ={log_val['val_th_rmse']:.3f}°,|V|={log_val['val_vm_rmse']:.4f}")

        # Wandb 日志记录
        if use_wandb:
            wandb.log({**log_train, **log_val}, step=epoch)

        # 保存最佳模型 (基于验证损失)
        if log_val["val_loss"] < best_va:
            best_va = log_val["val_loss"]
            best_epoch = epoch
            # 注意：保存模型状态字典，而不是整个 args 对象
            best_model_state = model.state_dict()

    print(f"Best validation loss {best_va:.4e} at epoch {best_epoch}")

    # ---- test ----
    if best_epoch != -1:
        model.load_state_dict(best_model_state) # 加载最佳模型
    model.eval()
    log_test = {}
    with torch.no_grad():
        ths, vms = [], []
        for batch in dl_te:
            # ... (测试循环内部逻辑不变) ...
            z_seq = batch["z"].to(args.device); feat_seq = batch["feat"].to(args.device)
            x_wls = batch["x_wls"].to(args.device); A = batch["A_time"].to(args.device)
            E = batch["E_time"].to(args.device); R_seq = batch["R"].to(args.device)
            ztype = batch["ztype"]; x_gt  = batch["x"].to(args.device)
            r_seq_np = compute_whitened_residual(z_seq.cpu().numpy(), x_wls.cpu().numpy(), R_seq.cpu().numpy(), ztype)
            r_seq = torch.from_numpy(r_seq_np).float().to(args.device)
            x_hat = model(r_seq, feat_seq, x_wls, A_time=A, E_time=E)
            th_rmse, vm_rmse = rmse_metrics(x_hat, x_gt); ths.append(th_rmse); vms.append(vm_rmse)

        log_test["test_th_rmse"] = float(np.mean(ths))
        log_test["test_vm_rmse"] = float(np.mean(vms))
        print(f"[TEST] θ-RMSE={log_test['test_th_rmse']:.3f}°, |V|-RMSE={log_test['test_vm_rmse']:.4f}")

    # Wandb 记录最终测试结果
    if use_wandb:
        wandb.log(log_test)
        wandb.finish() # 结束本次 wandb run

    return log_test # 返回测试结果
# ==============================================================

# ================== 关键改动 2: 修改 main 函数 ==================
def main():
    ap = argparse.ArgumentParser()
    # 添加所有需要的参数 (与 train_evaluate 中使用的保持一致)
    ap.add_argument("--data_dir", type=str, default="data/windows_ieee33")
    ap.add_argument("--tag", type=str, default="W24")
    ap.add_argument("--epochs", type=int, default=100) # 增加默认 epoch
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--bias_scale", type=float, default=3.0)
    ap.add_argument("--use_mask", action="store_true", default=False)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lambda_pf", type=float, default=0.1)
    ap.add_argument("--lambda_op", type=float, default=1.0)
    ap.add_argument("--lambda_smooth", type=float, default=0.01)
    ap.add_argument("--lambda_wls_reg", type=float, default=10.0)
    ap.add_argument("--w_temp_th", type=float, default=0.01)
    ap.add_argument("--w_temp_vm", type=float, default=0.01)
    # 新增: 控制是否启用 wandb 的参数 (可选)
    ap.add_argument("--use_wandb", action="store_true", help="Enable Wandb logging")

    args = ap.parse_args()

    # 直接调用封装好的函数
    train_evaluate(args, use_wandb=args.use_wandb)

if __name__ == "__main__":
    main()
# ==============================================================