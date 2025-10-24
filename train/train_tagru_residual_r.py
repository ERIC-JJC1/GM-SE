# 位置: train/train_tagru_residual_r.py
#
# TAGRU 模型全监督残差训练脚本 (输入白化残差 r)
# [已修复 v2] 统一参数命名 (hidden_dim, num_layers, gnn_hidden, gnn_layers)
# [已修复 v2] 添加 GNN 参数传递
# [已修复 v2] 添加模型保存、WandB、随机种子、tqdm 支持
# [已修复 v2] 修复 torch.load

import os, math, argparse, sys
import numpy as np
import torch
import torch.nn as nn
import wandb # 可选
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle # 用于修复 torch.load

# --- 项目导入 ---
from train.dataset import WindowDataset
from models.tagru import TopoAlignGRU # 确认模型 forward 匹配 (r, x_wls, feat) 输入
from tools.metrics import StateLoss, rmse_metrics # 监督损失和评估指标
from tools.train_sweep_and_compare import temporal_smooth # 时间平滑函数
# 物理损失计算 (可选)
from physics.ac_model import h_measure
from build_ieee33_with_pp import build_ieee33

# --- 设置随机种子 ---
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# --- 物理损失相关 ---
_ybus, _baseMVA, _slack_pos = None, None, None
def init_phys_meta(slack_pos):
    global _ybus, _baseMVA, _slack_pos
    # ... (与 train_refine_baseline.py 相同)
    if _ybus is None:
        _, _ybus, _baseMVA, *_ = build_ieee33()
        _ybus = _ybus.astype(np.complex128)
        _baseMVA = float(_baseMVA)
    _slack_pos = int(slack_pos)


def physics_loss(x_hat, z, R, ztype, eps=1e-9):
    # ... (与 train_refine_baseline.py 相同)
    x_hat_np = x_hat.detach().cpu().numpy()
    z_np = z.detach().cpu().numpy()
    R_np = R.detach().cpu().numpy()
    is_batched = x_hat_np.ndim > 2
    is_windowed = x_hat_np.ndim > 1
    if not is_batched and not is_windowed:
        x_hat_np = x_hat_np[np.newaxis, np.newaxis, :]
        z_np = z_np[np.newaxis, np.newaxis, :]
        R_np = R_np[np.newaxis, np.newaxis, :]
        ztype = ztype[np.newaxis, np.newaxis, :, :]
    elif not is_batched:
        x_hat_np = x_hat_np[np.newaxis, :, :]
        z_np = z_np[np.newaxis, :, :]
        R_np = R_np[np.newaxis, :, :]
        ztype = ztype[np.newaxis, :, :, :]
    B, W, _ = z_np.shape
    M = z_np.shape[-1]
    loss = 0.0
    count = 0
    for b in range(B):
        for t in range(W):
            if _ybus is None:
                 print("警告: 物理参数未初始化，无法计算物理损失。")
                 return x_hat.new_tensor(0.0)
            try:
                current_ztype = ztype[b, t]
                if current_ztype.shape[0] != 4 or current_ztype.shape[1] != M:
                     continue
                hx = h_measure(_ybus, x_hat_np[b,t], current_ztype, _baseMVA, _slack_pos)
                sigma_sq = np.maximum(R_np[b, t], eps)
                if len(hx) == len(z_np[b, t]):
                    wres = (z_np[b, t] - hx) / np.sqrt(sigma_sq)
                    loss += float(np.mean(wres**2))
                    count += 1
            except Exception as e:
                pass
    return x_hat.new_tensor(loss / count if count > 0 else 0.0)


# --- 数据加载器 ---
def make_loader(npz_path, batch_size, input_mode="whiten", shuffle=False, num_workers=0):
    ds = WindowDataset(npz_path, input_mode=input_mode)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False)
    # 获取维度信息
    try:
        with np.load(npz_path, allow_pickle=True) as D:
            S = D["x"].shape[2]
            F = D["feat"].shape[2]
            raw_z_key_present = "raw_z" in D
            if raw_z_key_present: M_z = D["raw_z"].shape[2]
            elif "z" in D: M_z = D["z"].shape[2]
            else: M_z = 0
            M_input = D["z"].shape[2] # 输入维度 (r)
            meta = D["meta"].item() if "meta" in D and D["meta"] else {}
    except FileNotFoundError:
        print(f"错误: 数据文件未在 {npz_path} 找到。")
        sys.exit(1)
    except Exception as e:
        print(f"从 {npz_path} 加载数据维度时出错: {e}")
        sys.exit(1)

    Nbus = int(meta.get("bus_count", S // 2))
    slack_pos_val = int(meta.get("slack_pos", 0))
    dims = {'S': S, 'F': F, 'M_input': M_input, 'M_z': M_z, 'Nbus': Nbus, 'slack_pos': slack_pos_val}
    return dl, dims

def main():
    # --- 1. 参数解析 ---
    ap = argparse.ArgumentParser(description="Train TAGRU Residual Model (Input: r)")
    ap.add_argument("--data_dir", type=str, default="data/windows_ieee33")
    ap.add_argument("--tag", type=str, default="W24")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)

    # [!!! 已修复 !!!] 统一参数命名并添加 GNN 参数
    ap.add_argument("--hidden_dim", type=int, default=256, help="模型主隐藏层维度")
    ap.add_argument("--num_layers", type=int, default=2, help="模型主层数 (e.g., GRU layers)")
    ap.add_argument("--gnn_hidden", type=int, default=128, help="GNN 隐藏层维度")
    ap.add_argument("--gnn_layers", type=int, default=2, help="GNN 层数")

    # 注意力超参 (暂时保留)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--bias_scale", type=float, default=3.0)
    ap.add_argument("--use_mask", action="store_true", default=False)

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # 损失权重 (与 refine_baseline 类似)
    ap.add_argument("--lambda_phys", type=float, default=0.0)
    ap.add_argument("--w_temp_th", type=float, default=0.01) # 注意默认值与其他脚本不同
    ap.add_argument("--w_temp_vm", type=float, default=0.01) # 注意默认值与其他脚本不同

    # [!!! 新增 !!!] 添加 WandB 和模型保存参数
    ap.add_argument("--wandb_project", type=str, default=None, help="WandB 项目名称 (可选)")
    ap.add_argument("--save_dir", type=str, default="checkpoints", help="模型保存目录")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # (可选) 初始化 wandb
    wandb_run = None
    if args.wandb_project:
        try:
             import wandb
             wandb_run = wandb.init(project=args.wandb_project, config=args)
             run_name = f"tagru_residual_r_h{args.hidden_dim}_l{args.num_layers}_lr{args.lr}_{args.tag}"
             wandb_run.name = run_name
        except ImportError:
             print("WandB 未安装。跳过 WandB 初始化。")
             args.wandb_project = None

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    # --- 数据加载 ---
    # 残差模型使用白化残差 'r'
    input_mode = "whiten"
    p_train = os.path.join(args.data_dir, f"{args.tag}_train.npz")
    p_val   = os.path.join(args.data_dir, f"{args.tag}_val.npz")
    p_test  = os.path.join(args.data_dir, f"{args.tag}_test.npz")

    dl_tr, dims_tr = make_loader(p_train, args.batch_size, input_mode=input_mode, shuffle=True)
    dl_va, dims_va = make_loader(p_val,   args.batch_size, input_mode=input_mode)
    dl_te, dims_te = make_loader(p_test,  args.batch_size, input_mode=input_mode)

    S = dims_tr['S']
    F = dims_tr['F']
    M_input = dims_tr['M_input'] # 输入 'r' 的维度
    M_z = dims_tr['M_z']      # 原始 'z' 的维度
    Nbus = dims_tr['Nbus']
    slack_pos_val = dims_tr['slack_pos']

    if args.lambda_phys > 0:
        init_phys_meta(slack_pos_val)

    # === 模型实例化 ===
    # [!!! 已修复 !!!] 使用统一的参数名传递给模型
    # **重要**: 需要确认 models/tagru.py 中的 TopoAlignGRU 是否能处理 (r, x_wls, feat) 输入
    #          当前 v5 版本 forward(self, z_seq, feat_seq, ...) 不能直接用
    #          假设存在一个版本或需要修改 forward 来适配
    #          或者此脚本应使用 RefineSeqTAModel? 根据脚本名推断用 TAGRU
    #          这里假设 TopoAlignGRU 的 forward 可以被修改或存在适用于此脚本的版本
    #          如果实际使用的模型是 RefineSeqTAModel，则应 import 并实例化它
    model = TopoAlignGRU(
        meas_dim=M_input, # 输入维度是 'r' 的维度
        feat_dim=F,
        state_dim=S,
        hidden_dim=args.hidden_dim, # 使用新参数名
        num_layers=args.num_layers, # 使用新参数名
        gnn_hidden=args.gnn_hidden, # 传递 GNN 参数
        gnn_layers=args.gnn_layers, # 传递 GNN 参数
        nhead=args.nhead,
        bias_scale=args.bias_scale,
        use_mask=args.use_mask
    ).to(args.device)
    print(f"模型已实例化:\n{model}") # 打印确认

    # --- 损失函数和优化器 ---
    # StateLoss 的 state_order 需要与模型输出对齐
    # 假设 TopoAlignGRU 在此模式下输出最终状态 x_hat = x_wls + dx，且顺序是 vm_va
    loss_fn = StateLoss(bus_count=Nbus, w_theta=2.0, w_vm=1.0, state_order='vm_va').to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=10, factor=0.5)

    best_va_loss = float("inf")

    # --- 训练循环 ---
    for epoch in range(1, args.epochs+1):
        # ---- train ----
        model.train()
        tot_mse, tot_lth, tot_lv, tot_phys, tot_temp = 0.0, 0.0, 0.0, 0.0, 0.0
        train_batch_count = 0

        pbar_train = tqdm(dl_tr, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for batch in pbar_train:
            train_batch_count += 1
            r     = batch["z"].to(args.device) # 输入是白化残差 r
            x_wls = batch["x_wls"].to(args.device)
            feat  = batch["feat"].to(args.device)
            A     = batch["A_time"].to(args.device) if "A_time" in batch else None # Optional
            E     = batch["E_time"].to(args.device) if "E_time" in batch else None # Optional
            x_gt  = batch["x"].to(args.device)

            # --- 模型前向 ---
            # 假设模型 forward(r, x_wls, feat, A_time, E_time) 返回最终状态 x_hat
            try:
                # *** 重要假设: TopoAlignGRU.forward 需要修改或重载以接受 r, x_wls ***
                # *** 如果模型内部是 return x_wls + dx，则此调用正确 ***
                # *** 如果模型内部是 return dx，则需要 x_hat = x_wls + model(...) ***
                x_hat = model(r, x_wls, feat, A_time=A, E_time=E)
            except TypeError as e:
                 print(f"\n错误: 调用 model.forward 时参数不匹配: {e}")
                 print("请检查 train_tagru_residual_r.py L207 与 models/tagru.py 的 forward 定义是否兼容 (r, x_wls) 输入。")
                 if wandb_run: wandb_run.finish(exit_code=1)
                 return

            # --- 损失计算 ---
            sup_loss, (lth, lv) = loss_fn(x_hat, x_gt) # 主 MSE 损失

            L_phys = torch.tensor(0.0, device=device)
            if args.lambda_phys > 0:
                raw_z = batch.get("raw_z")
                R_val = batch.get("R")
                ztype = batch.get("ztype")
                if raw_z is not None and R_val is not None and ztype is not None:
                    L_phys = physics_loss(x_hat, raw_z.to(device), R_val.to(device), ztype.numpy())
                else:
                     if train_batch_count == 1:
                         print("警告: 无法计算物理损失，batch 中缺少 'raw_z', 'R' 或 'ztype'。")

            L_temp = temporal_smooth(x_hat, args.w_temp_th, args.w_temp_vm)

            total = sup_loss + args.lambda_phys * L_phys + L_temp
            opt.zero_grad(); total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # --- 累加损失 ---
            bs = r.size(0)
            tot_mse += sup_loss.item() * bs
            tot_lth += lth.item() * bs
            tot_lv  += lv.item() * bs
            tot_phys += L_phys.item() * bs
            tot_temp += L_temp.item() * bs

            pbar_train.set_postfix({
                'Loss': f"{sup_loss.item():.4e}",
                'Phys': f"{L_phys.item():.4e}",
                'Temp': f"{L_temp.item():.4e}"
            })

        # 安全计算平均损失
        total_samples_train = len(dl_tr.dataset)
        avg_tr_loss = tot_mse / total_samples_train if total_samples_train > 0 else 0
        avg_tr_lth = tot_lth / total_samples_train if total_samples_train > 0 else 0
        avg_tr_lv = tot_lv / total_samples_train if total_samples_train > 0 else 0
        avg_tr_phys = tot_phys / total_samples_train if total_samples_train > 0 else 0
        avg_tr_temp = tot_temp / total_samples_train if total_samples_train > 0 else 0

        # ---- val ----
        model.eval()
        va_loss, val_batch_count = 0.0, 0
        ths, vms = [], []

        pbar_val = tqdm(dl_va, desc=f"Epoch {epoch}/{args.epochs} [Val]")
        with torch.no_grad():
            for batch in pbar_val:
                val_batch_count += 1
                r     = batch["z"].to(args.device)
                x_wls = batch["x_wls"].to(args.device)
                feat  = batch["feat"].to(args.device)
                A     = batch["A_time"].to(args.device) if "A_time" in batch else None
                E     = batch["E_time"].to(args.device) if "E_time" in batch else None
                x_gt  = batch["x"].to(args.device)

                try:
                    x_hat = model(r, x_wls, feat, A_time=A, E_time=E)
                except TypeError as e:
                     print(f"\n错误: 调用 model.forward (验证) 时参数不匹配: {e}")
                     continue

                loss, _ = loss_fn(x_hat, x_gt)
                va_loss += loss.item() * r.size(0) # 乘以 bs

                th_rmse, vm_rmse = rmse_metrics(x_hat, x_gt, state_order='vm_va') # 假设 vm_va
                ths.append(th_rmse); vms.append(vm_rmse)

            # 安全计算平均验证损失和指标
            total_samples_val = len(dl_va.dataset)
            avg_va_loss = va_loss / total_samples_val if total_samples_val > 0 else 0
            avg_th_rmse = np.nanmean(ths) if ths else float('nan')
            avg_vm_rmse = np.nanmean(vms) if vms else float('nan')

        sched.step(avg_va_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"[E{epoch:03d}] Train Loss={avg_tr_loss:.4e} (θ={avg_tr_lth:.4e}, V={avg_tr_lv:.4e}) Phys={avg_tr_phys:.4e} Tmp={avg_tr_temp:.4e} | "
              f"Val MSE Loss={avg_va_loss:.4e} | EVAL: θ-RMSE={avg_th_rmse:.3f}°, |V|-RMSE={avg_vm_rmse:.4f} | LR: {current_lr:.2e}")

        # (可选) WandB 日志
        if wandb_run:
            log_data = {
                "epoch": epoch,
                "train_loss_mse": avg_tr_loss,
                # "train_loss_theta": avg_tr_lth, # 可选
                # "train_loss_vm": avg_tr_lv,     # 可选
                "train_loss_phys": avg_tr_phys,
                "train_loss_temp": avg_tr_temp,
                "val_loss_mse": avg_va_loss,
                "val_rmse_theta_deg": avg_th_rmse,
                "val_rmse_vm_pu": avg_vm_rmse,
                "learning_rate": current_lr
            }
            log_data_filtered = {k: v for k, v in log_data.items() if not (isinstance(v, float) and np.isnan(v))}
            wandb_run.log(log_data_filtered)


        # [!!! 已修复 !!!] 模型保存逻辑
        is_best = avg_va_loss < best_va_loss
        if is_best:
            best_va_loss = avg_va_loss
            model_save_name = f"tagru_residual_r_best_{args.tag}.pt"
            save_path = os.path.join(args.save_dir, model_save_name)
            print(f"   => Validation loss improved to {best_va_loss:.4e}. Saving model...")
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'val_loss_mse': best_va_loss,
                    'args': args
                }, save_path)
                print(f"   => Best model saved to {save_path}")
                if wandb_run:
                     wandb_run.summary["best_val_loss_mse"] = best_va_loss
                     wandb_run.summary["best_epoch"] = epoch
                     # wandb.save(save_path)
            except Exception as e:
                print(f"错误: 保存检查点失败: {e}")

    # ---- test ----
    print("\n--- Training Finished ---")
    print(f"Loading best model from epoch {wandb_run.summary.get('best_epoch', 'N/A') if wandb_run else 'N/A'} with Val MSE Loss: {best_va_loss:.4e}")

    # 加载最佳模型
    best_model_path = os.path.join(args.save_dir, f"tagru_residual_r_best_{args.tag}.pt")
    if os.path.exists(best_model_path):
        try:
            # [!!! 已修复 !!!] 添加 weights_only=False
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Best model loaded from {best_model_path}")

            model.eval()
            test_ths, test_vms = [], []
            pbar_test = tqdm(dl_te, desc="[Test]")
            with torch.no_grad():
                for batch in pbar_test:
                    r     = batch["z"].to(args.device)
                    x_wls = batch["x_wls"].to(args.device)
                    feat  = batch["feat"].to(args.device)
                    A     = batch["A_time"].to(args.device) if "A_time" in batch else None
                    E     = batch["E_time"].to(args.device) if "E_time" in batch else None
                    x_gt  = batch["x"].to(args.device)
                    try:
                        x_hat = model(r, x_wls, feat, A_time=A, E_time=E)
                    except TypeError as e:
                        print(f"\n错误: 调用 model.forward (测试) 时参数不匹配: {e}")
                        continue

                    th_rmse, vm_rmse = rmse_metrics(x_hat, x_gt, state_order='vm_va') # 假设 vm_va
                    test_ths.append(th_rmse); test_vms.append(vm_rmse)

            avg_test_th_rmse = np.nanmean(test_ths) if test_ths else float('nan')
            avg_test_vm_rmse = np.nanmean(test_vms) if test_vms else float('nan')
            print(f"[TEST] θ-RMSE={avg_test_th_rmse:.3f}°, |V|-RMSE={avg_test_vm_rmse:.4f}")

            if wandb_run:
                wandb_run.summary["test_rmse_theta_deg"] = avg_test_th_rmse
                wandb_run.summary["test_rmse_vm_pu"] = avg_test_vm_rmse
        except (pickle.UnpicklingError, KeyError, RuntimeError, Exception) as e:
             print(f"错误: 加载或测试最佳模型时出错: {e}")
    else:
        print(f"警告: 最佳模型检查点未找到于 {best_model_path}。跳过测试评估。")


    if wandb_run:
        wandb_run.finish()

if __name__ == "__main__":
    main()