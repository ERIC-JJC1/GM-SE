# 位置: train/train_refine_baseline.py
#
# Refine-WLS 模型全监督训练脚本
# v3:
# - 统一参数命名 (hidden_dim, num_layers, gnn_hidden, gnn_layers)
# - 完整对齐 wandb sweep 超参: batch_size, data_dir, epochs, hidden_dim, lr, num_layers, tag
# - 布尔参数与 wandb 兼容 (str2bool)
# - 新增/补齐: lambda_phys, w_temp_th, w_temp_vm
# - 更稳健的 physics_loss（支持 Tensor/ndarray 的 ztype/raw_z/R）
# - 可选 WandB、模型保存、随机种子、tqdm
# - 更稳健的列表/标量解析（gnn_* 支持标量/列表）

import os, sys, math, argparse, pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 可选 WandB ---
try:
    import wandb
except Exception:
    wandb = None

# --- 项目导入 ---
from train.dataset import WindowDataset
from models.refine_seq import RefineSeqTAModel   # 确认 forward(r, x_wls, feat, A_time=None, E_time=None)
from tools.metrics import StateLoss, rmse_metrics
from tools.train_sweep_and_compare import temporal_smooth
from build_ieee33_with_pp import build_ieee33
from physics.ac_model import h_measure

# ----------------- 工具函数 -----------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).lower()
    if v in ('yes', 'true', 't', 'y', '1'):
        return True
    if v in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_int_list(v):
    # 支持 list / "128,256" / "128 256" / "128"
    if isinstance(v, (list, tuple, np.ndarray)):
        return [int(x) for x in v]
    s = str(v).replace(',', ' ')
    return [int(x) for x in s.split()]

def parse_float_list(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return [float(x) for x in v]
    s = str(v).replace(',', ' ')
    return [float(x) for x in s.split()]

# ----------------- 物理损失依赖 -----------------
_YBUS = None
_BASE_MVA = None
_SLACK_POS = None

def init_phys_meta(slack_pos):
    global _YBUS, _BASE_MVA, _SLACK_POS
    if _YBUS is None:
        _, ybus, baseMVA, *_ = build_ieee33()
        _YBUS = ybus.astype(np.complex128)
        _BASE_MVA = float(baseMVA)
    _SLACK_POS = int(slack_pos)

def _to_numpy(x):
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    # 尽量转成 ndarray
    return np.asarray(x)

def physics_loss(x_hat, z, R, ztype, eps=1e-9):
    """
    x_hat: (B, W, S) or (W, S) or (S,)
    z, R : (B, W, M) 同形
    ztype: (B, W, 4, M) 或 (W, 4, M) 或 (4, M)
    返回: torch.tensor 标量（在 x_hat 同设备）
    """
    if _YBUS is None or _BASE_MVA is None or _SLACK_POS is None:
        # 未初始化时，不施加物理损失
        return x_hat.new_tensor(0.0)

    x_hat_np = _to_numpy(x_hat)
    z_np     = _to_numpy(z)
    R_np     = _to_numpy(R)
    zt_np    = _to_numpy(ztype)

    # 统一批次和时间维度
    # 允许输入是 (S,), (W,S), (B,W,S)
    def ensure_bws(a, is_ztype=False):
        if a is None:
            return None
        if a.ndim == 1:   # (S,) or (M,) or (4,) —— 基本不适用，但做容错
            return a[np.newaxis, np.newaxis, ...]
        if a.ndim == 2:   # (W, S) or (W, M) or (4, M)
            if is_ztype and a.shape[0] == 4:
                return a[np.newaxis, ...]            # -> (1, 4, M)
            return a[np.newaxis, ...]                # -> (1, W, *)
        if a.ndim == 3:
            return a                                  # (B, W, *)
        if a.ndim == 4:   # ztype 常见 (B, W, 4, M)
            return a
        # 超出预期，尽量拉到 (1,1,*)
        return a[np.newaxis, np.newaxis, ...]

    x_hat_np = ensure_bws(x_hat_np)
    z_np     = ensure_bws(z_np)
    R_np     = ensure_bws(R_np)
    # ztype 特殊：可能是 (B,W,4,M) 或 (W,4,M) 或 (4,M)
    zt_np    = ensure_bws(zt_np, is_ztype=True)

    # 标准化 ztype 形状为 (B, W, 4, M)
    # 当前 zt_np 可能是 (B, W, 4, M) / (W,4,M) / (1,4,M)
    if zt_np.ndim == 3 and zt_np.shape[0] == 4:      # (4, M) -> (1,1,4,M)
        zt_np = zt_np[np.newaxis, np.newaxis, ...]
    elif zt_np.ndim == 3 and zt_np.shape[1] == 4:    # (1,4,M) -> (1,1,4,M)
        zt_np = zt_np[np.newaxis, ...]
    elif zt_np.ndim == 4 and zt_np.shape[2] != 4:
        # 若用户维度次序不符，放弃本次物理损失
        return x_hat.new_tensor(0.0)

    B = max(x_hat_np.shape[0], z_np.shape[0], R_np.shape[0], zt_np.shape[0])
    W = max(x_hat_np.shape[1], z_np.shape[1], R_np.shape[1], zt_np.shape[1])

    # 广播/截断到 (B,W,*) —— 这里简单取 min 以避免越界
    B = min(B, x_hat_np.shape[0], z_np.shape[0], R_np.shape[0], zt_np.shape[0])
    W = min(W, x_hat_np.shape[1], z_np.shape[1], R_np.shape[1], zt_np.shape[1])

    loss, cnt = 0.0, 0
    for b in range(B):
        for t in range(W):
            try:
                cur_zt = zt_np[b, t]      # (4, M)
                if cur_zt.ndim != 2 or cur_zt.shape[0] != 4:
                    continue
                hx = h_measure(_YBUS, x_hat_np[b, t], cur_zt, _BASE_MVA, _SLACK_POS)  # -> (M,)
                sigma_sq = np.maximum(R_np[b, t], eps)                                 # (M,)
                if len(hx) == len(z_np[b, t]):
                    wres = (z_np[b, t] - hx) / np.sqrt(sigma_sq)
                    loss += float(np.mean(wres ** 2))
                    cnt += 1
            except Exception:
                # 静默跳过坏样本，避免训练中断
                pass

    return x_hat.new_tensor(loss / cnt if cnt > 0 else 0.0)

# ----------------- 数据加载 -----------------
def make_loader(npz_path, batch_size, input_mode="whiten", shuffle=False, num_workers=0):
    ds = WindowDataset(npz_path, input_mode=input_mode)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                    num_workers=num_workers, drop_last=False)
    # 推断维度与 meta
    try:
        with np.load(npz_path, allow_pickle=True) as D:
            S = int(D["x"].shape[2])
            F = int(D["feat"].shape[2])
            M_input = int(D["z"].shape[2])

            if "raw_z" in D:
                M_z = int(D["raw_z"].shape[2])
            elif "z" in D:
                M_z = int(D["z"].shape[2])
            else:
                print(f"警告: {npz_path} 未找到 raw_z/z，M_z 设为 0")
                M_z = 0

            meta = D["meta"].item() if "meta" in D and D["meta"] else {}
    except FileNotFoundError:
        print(f"错误: 数据文件未找到 {npz_path}")
        sys.exit(1)
    except Exception as e:
        print(f"从 {npz_path} 读取维度失败: {e}")
        sys.exit(1)

    Nbus = int(meta.get("bus_count", S // 2))
    slack_pos_val = int(meta.get("slack_pos", 0))
    dims = {'S': S, 'F': F, 'M_input': M_input, 'M_z': M_z, 'Nbus': Nbus, 'slack_pos': slack_pos_val}
    return dl, dims

# ----------------- 主程序 -----------------
def build_argparser():
    ap = argparse.ArgumentParser(description="Train Refine-WLS Baseline (RefineSeqTAModel)")

    # —— 与 sweep 对齐的关键参数 ——
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--data_dir", type=str, default="data/windows_ieee33")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden_dim", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--tag", type=str, default="W96")

    # —— 注意力与通用超参 ——
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--bias_scale", type=float, default=3.0)
    ap.add_argument("--use_mask", type=str2bool, nargs='?', const=True, default=False,
                    help="Enable temporal mask in attention (default: False)")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # —— 物理/时序正则权重 ——
    ap.add_argument("--lambda_phys", type=float, default=0.0, help="Physics loss weight")
    ap.add_argument("--w_temp_th", type=float, default=0.0, help="Temporal smooth weight for theta")
    ap.add_argument("--w_temp_vm", type=float, default=0.0, help="Temporal smooth weight for |V|")

    # —— BusSmooth / GNN 开关（与 wandb 兼容） ——
    ap.add_argument("--use_bus_smooth", type=str2bool, nargs='?', const=True, default=True,
                    help="Enable BusSmooth layer (default: True)")
    ap.add_argument("--use_gnn", type=str2bool, nargs='?', const=True, default=False,
                    help="Enable GNN blocks (default: False)")

    # —— GNN 具体参数（脚本内可暂不使用，但保持可解析） ——
    ap.add_argument("--gnn_type", type=str, choices=["tag", "gcn2", "gat"], default="tag")
    ap.add_argument("--gnn_hidden", type=parse_int_list, default=[128])
    ap.add_argument("--gnn_layers", type=parse_int_list, default=[3])
    ap.add_argument("--gnn_dropout", type=parse_float_list, default=[0.0])

    # —— 训练/记录 ——
    ap.add_argument("--wandb_project", type=str, default=None, help="WandB 项目名称（可选）")
    ap.add_argument("--save_dir", type=str, default="checkpoints")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=0)
    #----wandb参数--
    ap.add_argument("--wandb_group", type=str, default=None, help="WandB group name (e.g., for grouping seeds)")
    ap.add_argument("--wandb_name", type=str, default=None, help="WandB run name (overrides default naming)")
    return ap

def main():
    ap = build_argparser()
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # —— 可选 WandB 初始化 ——
    wandb_run = None
    if wandb is not None and args.wandb_project:
        try:
            wandb_run = wandb.init(
                project=args.wandb_project,
                config=args,
                group=args.wandb_group, # Pass group
                name=args.wandb_name or f"tagru_sup_h{args.hidden_dim}_l{args.num_layers}_lr{args.lr}_{args.tag}_seed{args.seed}" # Pass name or default
            )
        except Exception as e:
            print(f"WandB 初始化失败（忽略继续）: {e}")
            wandb_run = None

    # === 数据 ===
    input_mode = "whiten"  # Refine-WLS 以白化残差 r 作为输入
    p_train = os.path.join(args.data_dir, f"{args.tag}_train.npz")
    p_val   = os.path.join(args.data_dir, f"{args.tag}_val.npz")
    p_test  = os.path.join(args.data_dir, f"{args.tag}_test.npz")

    dl_tr, dims_tr = make_loader(p_train, args.batch_size, input_mode=input_mode,
                                 shuffle=True, num_workers=args.num_workers)
    dl_va, dims_va = make_loader(p_val,   args.batch_size, input_mode=input_mode,
                                 shuffle=False, num_workers=args.num_workers)
    dl_te, dims_te = make_loader(p_test,  args.batch_size, input_mode=input_mode,
                                 shuffle=False, num_workers=args.num_workers)

    S = dims_tr['S']; F = dims_tr['F']
    M_input = dims_tr['M_input']
    Nbus = dims_tr['Nbus']
    slack_pos_val = dims_tr['slack_pos']

    if args.lambda_phys > 0:
        init_phys_meta(slack_pos_val)

    # === 模型 ===
    model = RefineSeqTAModel(
        meas_dim=M_input,
        state_dim=S,
        feat_dim=F,
        hidden=args.hidden_dim,
        num_layers=args.num_layers,
        nhead=args.nhead,
        bias_scale=args.bias_scale,
        use_mask=args.use_mask
    ).to(device)
    print(f"模型实例化完成: {model.__class__.__name__} | hidden={args.hidden_dim}, layers={args.num_layers}, nhead={args.nhead}")

    # === 优化/损失 ===
    loss_fn = StateLoss(bus_count=Nbus, w_theta=2.0, w_vm=1.0, state_order='va_vm').to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=10, factor=0.5)

    best_va_loss = float('inf')

    # === 训练循环 ===
    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        tot_mse = tot_lth = tot_lv = tot_phys = tot_temp = 0.0

        pbar_train = tqdm(dl_tr, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for batch in pbar_train:
            r     = batch["z"].to(device)         # 白化残差 r
            x_wls = batch["x_wls"].to(device)
            feat  = batch["feat"].to(device)
            A     = batch.get("A_time")
            E     = batch.get("E_time")
            A     = A.to(device) if A is not None else None
            E     = E.to(device) if E is not None else None
            x_gt  = batch["x"].to(device)

            try:
                x_hat = model(r, x_wls, feat, A_time=A, E_time=E)
            except TypeError as e:
                print(f"\n错误: model.forward 参数不匹配: {e}\n"
                      f"请检查 models/refine_seq.RefineSeqTAModel.forward 接口。")
                if wandb_run: wandb_run.finish(exit_code=1)
                return

            sup_loss, (lth, lv) = loss_fn(x_hat, x_gt)

            L_phys = torch.tensor(0.0, device=device)
            if args.lambda_phys > 0:
                raw_z = batch.get("raw_z")
                R_val = batch.get("R")
                ztype = batch.get("ztype")
                if raw_z is not None and R_val is not None and ztype is not None:
                    L_phys = physics_loss(x_hat, raw_z, R_val, ztype)
                # else: 首次可打印提示，避免刷屏，这里从简不打印

            L_temp = temporal_smooth(x_hat, args.w_temp_th, args.w_temp_vm)

            total = sup_loss + args.lambda_phys * L_phys + L_temp
            opt.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = r.size(0)
            tot_mse  += sup_loss.item() * bs
            tot_lth  += lth.item() * bs
            tot_lv   += lv.item() * bs
            tot_phys += L_phys.item() * bs
            tot_temp += L_temp.item() * bs

            pbar_train.set_postfix({
                "Loss": f"{sup_loss.item():.4e}",
                "Phys": f"{L_phys.item():.2e}",
                "Temp": f"{L_temp.item():.2e}",
            })

        total_samples_train = len(dl_tr.dataset)
        avg_tr_loss = tot_mse  / total_samples_train if total_samples_train > 0 else 0.0
        avg_tr_lth  = tot_lth  / total_samples_train if total_samples_train > 0 else 0.0
        avg_tr_lv   = tot_lv   / total_samples_train if total_samples_train > 0 else 0.0
        avg_tr_phys = tot_phys / total_samples_train if total_samples_train > 0 else 0.0
        avg_tr_temp = tot_temp / total_samples_train if total_samples_train > 0 else 0.0

        # ---- val ----
        model.eval()
        va_loss = 0.0
        ths, vms = [], []
        pbar_val = tqdm(dl_va, desc=f"Epoch {epoch}/{args.epochs} [Val]")
        with torch.no_grad():
            for batch in pbar_val:
                r     = batch["z"].to(device)
                x_wls = batch["x_wls"].to(device)
                feat  = batch["feat"].to(device)
                A     = batch.get("A_time")
                E     = batch.get("E_time")
                A     = A.to(device) if A is not None else None
                E     = E.to(device) if E is not None else None
                x_gt  = batch["x"].to(device)

                try:
                    x_hat = model(r, x_wls, feat, A_time=A, E_time=E)
                except TypeError as e:
                    print(f"\n错误: model.forward(验证) 参数不匹配: {e}")
                    continue

                loss, _ = loss_fn(x_hat, x_gt)
                va_loss += loss.item() * r.size(0)

                th_rmse, vm_rmse = rmse_metrics(x_hat, x_gt, state_order='va_vm')
                ths.append(th_rmse); vms.append(vm_rmse)

        total_samples_val = len(dl_va.dataset)
        avg_va_loss = va_loss / total_samples_val if total_samples_val > 0 else 0.0
        avg_th_rmse = float(np.nanmean(ths)) if ths else float('nan')
        avg_vm_rmse = float(np.nanmean(vms)) if vms else float('nan')

        sched.step(avg_va_loss)
        current_lr = opt.param_groups[0]['lr']

        print(f"[E{epoch:03d}] Train MSE={avg_tr_loss:.4e} (θ={avg_tr_lth:.4e}, |V|={avg_tr_lv:.4e}) "
              f"Phys={avg_tr_phys:.2e} Tmp={avg_tr_temp:.2e} | "
              f"Val MSE={avg_va_loss:.4e} | θ-RMSE={avg_th_rmse:.3f}°, |V|-RMSE={avg_vm_rmse:.4f} | LR={current_lr:.2e}")

        if wandb_run is not None:
            log_data = {
                "epoch": epoch,
                "train_loss_mse": avg_tr_loss,
                "train_loss_theta": avg_tr_lth,
                "train_loss_vm": avg_tr_lv,
                "train_loss_phys": avg_tr_phys,
                "train_loss_temp": avg_tr_temp,
                "val_loss_mse": avg_va_loss,
                "val_rmse_theta_deg": avg_th_rmse,
                "val_rmse_vm_pu": avg_vm_rmse,
                "learning_rate": current_lr
            }
            # 过滤 nan
            log_data = {k: v for k, v in log_data.items()
                        if not (isinstance(v, float) and (np.isnan(v) or np.isinf(v)))}
            wandb_run.log(log_data)

        # —— 保存最优 —— 
        is_best = avg_va_loss < best_va_loss
        if is_best:
            best_va_loss = avg_va_loss
            save_path = os.path.join(args.save_dir, f"refine_wls_best_{args.tag}.pt")
            print(f"   => Val 改善至 {best_va_loss:.4e}，保存模型到 {save_path}")
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'val_loss_mse': best_va_loss,
                    'args': vars(args)
                }, save_path)
                if wandb_run is not None:
                    wandb_run.summary["best_val_loss_mse"] = best_va_loss
                    wandb_run.summary["best_epoch"] = epoch
            except Exception as e:
                print(f"保存模型失败: {e}")

    # === 测试 ===
    print("\n--- Training Finished ---")
    best_model_path = os.path.join(args.save_dir, f"refine_wls_best_{args.tag}.pt")
    print(f"Best checkpoint: {best_model_path}")

    if os.path.exists(best_model_path):
        try:
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            model.eval()

            test_ths, test_vms = [], []
            pbar_test = tqdm(dl_te, desc="[Test]")
            with torch.no_grad():
                for batch in pbar_test:
                    r     = batch["z"].to(device)
                    x_wls = batch["x_wls"].to(device)
                    feat  = batch["feat"].to(device)
                    A     = batch.get("A_time")
                    E     = batch.get("E_time")
                    A     = A.to(device) if A is not None else None
                    E     = E.to(device) if E is not None else None
                    x_gt  = batch["x"].to(device)

                    try:
                        x_hat = model(r, x_wls, feat, A_time=A, E_time=E)
                    except TypeError as e:
                        print(f"\n错误: model.forward(测试) 参数不匹配: {e}")
                        continue

                    th_rmse, vm_rmse = rmse_metrics(x_hat, x_gt, state_order='va_vm')
                    test_ths.append(th_rmse); test_vms.append(vm_rmse)

            avg_test_th_rmse = float(np.nanmean(test_ths)) if test_ths else float('nan')
            avg_test_vm_rmse = float(np.nanmean(test_vms)) if test_vms else float('nan')
            print(f"[TEST] θ-RMSE={avg_test_th_rmse:.3f}°, |V|-RMSE={avg_test_vm_rmse:.4f}")

            if wandb_run is not None:
                wandb_run.summary["test_rmse_theta_deg"] = avg_test_th_rmse
                wandb_run.summary["test_rmse_vm_pu"] = avg_test_vm_rmse
        except (pickle.UnpicklingError, KeyError, RuntimeError, Exception) as e:
            print(f"加载或评测最佳模型失败: {e}")
    else:
        print(f"警告: 未找到最佳检查点 {best_model_path}，跳过测试。")

    if wandb_run is not None:
        wandb_run.finish()

if __name__ == "__main__":
    main()
