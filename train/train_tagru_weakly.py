# 位置: train/train_tagru_weakly.py
#
# TAGRU 模型弱监督训练脚本
# [已修复 v2] 统一参数命名 (hidden_dim, num_layers, gnn_hidden, gnn_layers)
# [已修复 v2] 添加 GNN 参数传递
# [已修复 v2] 添加模型保存、WandB、随机种子支持

import os, math, argparse, sys
import numpy as np
import torch
import torch.nn as nn
import wandb # 可选
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- 项目导入 ---
from train.dataset import WindowDataset
from models.tagru import TopoAlignGRU # 确保这是 v5 (弱监督) 版本
from tools.metrics import rmse_metrics
from physics.losses import PhysicsInformedLoss
from tools.train_sweep_and_compare import temporal_smooth # 时间平滑函数
import pickle

# --- 设置随机种子 ---
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    # CuDNN settings (optional but recommended for reproducibility)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

# --- 数据加载器 (保持不变) ---
def make_loader(npz_path, batch_size, input_mode="raw", shuffle=False, num_workers=0):
    ds = WindowDataset(npz_path, input_mode=input_mode)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False)
    # 获取维度信息
    try:
        with np.load(npz_path, allow_pickle=True) as D:
            S = D["x"].shape[2]
            F = D["feat"].shape[2]
            M = D["z"].shape[2]
            meta = D["meta"].item() if "meta" in D and D["meta"] else {}
    except FileNotFoundError:
        print(f"错误: 数据文件未在 {npz_path} 找到。")
        sys.exit(1)
    except Exception as e:
        print(f"从 {npz_path} 加载数据维度时出错: {e}")
        sys.exit(1)

    Nbus = int(meta.get("bus_count", S // 2))
    slack_pos_val = int(meta.get("slack_pos", 0))
    dims = {'S': S, 'F': F, 'M': M, 'Nbus': Nbus, 'slack_pos': slack_pos_val}
    return dl, dims

def main():
    # --- 1. 参数解析 ---
    ap = argparse.ArgumentParser(description="Train TAGRU Weakly Supervised Model")
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

    # 注意力超参 (暂时保留，后续 P1 处理)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--bias_scale", type=float, default=3.0)
    ap.add_argument("--use_mask", action="store_true", default=False)

    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    # PhysicsInformedLoss 权重
    ap.add_argument("--lambda_op", type=float, default=1.0, help="权重: 运行约束 (Phi_op)")

    # 时间平滑损失权重
    ap.add_argument("--w_temp_th", type=float, default=0.01)
    ap.add_argument("--w_temp_vm", type=float, default=0.01)

    # [!!! 新增 !!!] 添加 WandB 和模型保存参数
    ap.add_argument("--wandb_project", type=str, default=None, help="WandB 项目名称 (可选)")
    ap.add_argument("--save_dir", type=str, default="checkpoints", help="模型保存目录")
    ap.add_argument("--seed", type=int, default=42) # 添加随机种子

    args = ap.parse_args()

    # 设置随机种子
    set_seed(args.seed)

    # (可选) 初始化 wandb
    wandb_run = None
    if args.wandb_project:
        try:
             import wandb
             wandb_run = wandb.init(project=args.wandb_project, config=args)
             # 设置一个有意义的 run 名称
             run_name = f"tagru_weakly_h{args.hidden_dim}_l{args.num_layers}_gh{args.gnn_hidden}_gl{args.gnn_layers}_lr{args.lr}_{args.tag}"
             wandb_run.name = run_name
        except ImportError:
             print("WandB 未安装。跳过 WandB 初始化。")
             args.wandb_project = None

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    # --- 数据加载 ---
    p_train = os.path.join(args.data_dir, f"{args.tag}_train.npz")
    p_val   = os.path.join(args.data_dir, f"{args.tag}_val.npz")
    p_test  = os.path.join(args.data_dir, f"{args.tag}_test.npz")

    dl_tr, dims_tr = make_loader(p_train, args.batch_size, input_mode="raw", shuffle=True)
    dl_va, dims_va = make_loader(p_val,   args.batch_size, input_mode="raw")
    dl_te, dims_te = make_loader(p_test,  args.batch_size, input_mode="raw")

    S = dims_tr['S']
    F = dims_tr['F']
    M = dims_tr['M']
    Nbus = dims_tr['Nbus']
    slack_pos_val = dims_tr['slack_pos']

    # ================== 模型和损失实例化 ==================
    # [!!! 已修复 !!!] 使用统一的参数名传递给模型
    model = TopoAlignGRU(
        meas_dim=M, feat_dim=F, state_dim=S,
        hidden_dim=args.hidden_dim, num_layers=args.num_layers,
        gnn_hidden=args.gnn_hidden, gnn_layers=args.gnn_layers,
        nhead=args.nhead, bias_scale=args.bias_scale, # 暂时保留
        use_mask=args.use_mask # 暂时保留
    ).to(args.device)
    print(f"模型已实例化:\n{model}") # 打印模型结构以确认参数

    # 实例化物理损失
    phys_loss_fn = PhysicsInformedLoss(
        nbus=Nbus,
        slack_pos=slack_pos_val,
        lambda_op=args.lambda_op
    ).to(args.device)
    # =======================================================

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # [!!! 已修复 !!!] 再次移除不支持的 'verbose' 参数
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=10, factor=0.5)

    best_va_loss = float("inf") # 使用验证损失作为标准

    for epoch in range(1, args.epochs+1):
        # ---- train ----
        model.train()
        tot_res, tot_op, tot_tmp, tot_combined = 0.0, 0.0, 0.0, 0.0
        train_batch_count = 0

        pbar_train = tqdm(dl_tr, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for batch in pbar_train:
            train_batch_count += 1
            z_seq = batch["z"].to(args.device)
            feat_seq = batch["feat"].to(args.device)
            # A, E 似乎未在当前 v5 模型中使用
            # A     = batch["A_time"].to(args.device)
            # E     = batch["E_time"].to(args.device)

            # 物理损失所需
            R_seq = batch["R"].to(args.device)
            if "ztype" not in batch:
                print("错误: batch 中未找到 'ztype'。")
                if wandb_run: wandb_run.finish(exit_code=1)
                return
            ztype_np = batch["ztype"].numpy()

            x_hat = model(z_seq, feat_seq) # A_time, E_time 移除

            # ================== 损失计算 ==================
            batch_gpu = { "z": z_seq, "R": R_seq, "ztype": ztype_np }
            phys_loss, loss_dict = phys_loss_fn(x_hat, batch_gpu)

            L_temp = temporal_smooth(x_hat, args.w_temp_th, args.w_temp_vm)

            total_loss = phys_loss + L_temp # 总损失是物理损失+时间平滑
            # ===============================================

            opt.zero_grad(); total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = z_seq.size(0)
            tot_combined += total_loss.item()*bs
            tot_res += loss_dict["loss_residual"].item()*bs
            tot_op  += loss_dict["loss_op"].item()*bs
            tot_tmp += L_temp.item()*bs

            # 更新 tqdm 后缀
            pbar_train.set_postfix({
                'Loss': f"{total_loss.item():.4e}",
                'Res': f"{loss_dict['loss_residual'].item():.4e}",
                'Op': f"{loss_dict['loss_op'].item():.4e}",
                'Temp': f"{L_temp.item():.4e}"
            })

        # 安全计算平均损失
        avg_tr_combined = tot_combined / train_batch_count if train_batch_count > 0 else 0
        avg_tr_res = tot_res / train_batch_count if train_batch_count > 0 else 0
        avg_tr_op  = tot_op / train_batch_count if train_batch_count > 0 else 0
        avg_tr_tmp = tot_tmp / train_batch_count if train_batch_count > 0 else 0

        # ---- val ----
        model.eval()
        va_loss_phys, val_batch_count = 0.0, 0
        ths, vms = [], [] # 用于 *评估* 的 RMSE 指标

        pbar_val = tqdm(dl_va, desc=f"Epoch {epoch}/{args.epochs} [Val]")
        with torch.no_grad():
            for batch in pbar_val:
                val_batch_count += 1
                z_seq = batch["z"].to(args.device)
                feat_seq = batch["feat"].to(device)
                x_gt  = batch["x"].to(args.device) # !! 仅用于评估 !!
                R_seq = batch["R"].to(args.device)
                if "ztype" not in batch:
                    print("错误: 验证 batch 中未找到 'ztype'。")
                    continue
                ztype_np = batch["ztype"].numpy()

                x_hat = model(z_seq, feat_seq)

                # 使用 *物理损失* 作为验证指标
                batch_gpu = { "z": z_seq, "R": R_seq, "ztype": ztype_np }
                val_loss_batch, _ = phys_loss_fn(x_hat, batch_gpu)
                va_loss_phys += val_loss_batch.item()*z_seq.size(0)

                # 仅用于 *报告*：计算 RMSE
                th_rmse, vm_rmse = rmse_metrics(x_hat, x_gt, state_order='vm_va')
                ths.append(th_rmse); vms.append(vm_rmse)

            # 安全计算平均验证损失和指标
            avg_va_loss = va_loss_phys / val_batch_count if val_batch_count > 0 else 0
            avg_th_rmse = np.nanmean(ths) if ths else float('nan')
            avg_vm_rmse = np.nanmean(vms) if vms else float('nan')

        sched.step(avg_va_loss) # 基于物理损失调整学习率
        current_lr = opt.param_groups[0]['lr']

        print(f"[E{epoch:03d}] Train Loss={avg_tr_combined:.4e} (Res={avg_tr_res:.4e}, Op={avg_tr_op:.4e}, Tmp={avg_tr_tmp:.4e}) | "
              f"Val Loss(phys)={avg_va_loss:.4e} | EVAL: θ-RMSE={avg_th_rmse:.3f}°, |V|-RMSE={avg_vm_rmse:.4f} | LR: {current_lr:.2e}")

        # (可选) WandB 日志
        if wandb_run:
            log_data = {
                "epoch": epoch,
                "train_loss_combined": avg_tr_combined,
                "train_loss_res": avg_tr_res,
                "train_loss_op": avg_tr_op,
                "train_loss_temp": avg_tr_tmp,
                "val_loss_phys": avg_va_loss,
                "val_rmse_theta_deg": avg_th_rmse,
                "val_rmse_vm_pu": avg_vm_rmse,
                "learning_rate": current_lr
            }
            log_data_filtered = {k: v for k, v in log_data.items() if not (isinstance(v, float) and np.isnan(v))}
            wandb_run.log(log_data_filtered)

        # [!!! 新增 !!!] 模型保存逻辑
        if avg_va_loss < best_va_loss:
            best_va_loss = avg_va_loss
            # 使用特定名称保存
            model_save_name = f"tagru_weakly_best_{args.tag}.pt"
            save_path = os.path.join(args.save_dir, model_save_name)
            print(f"   => Validation loss improved to {best_va_loss:.4e}. Saving model...")
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'val_loss_phys': best_va_loss, # 保存基于物理损失的最佳值
                    'args': args
                }, save_path)
                print(f"   => Best model saved to {save_path}")
                if wandb_run:
                     wandb_run.summary["best_val_loss_phys"] = best_va_loss
                     wandb_run.summary["best_epoch"] = epoch
                     # wandb.save(save_path) # 保存为 artifact
            except Exception as e:
                print(f"错误: 保存检查点失败: {e}")

# ---- test ---- (在训练结束后可选地运行测试)
    print("\n--- Training Finished ---")
    print(f"Loading best model from epoch {wandb_run.summary.get('best_epoch', 'N/A') if wandb_run else 'N/A'} with Val Loss(phys): {best_va_loss:.4e}")

    # 加载最佳模型
    best_model_path = os.path.join(args.save_dir, f"tagru_weakly_best_{args.tag}.pt")
    if os.path.exists(best_model_path):
        try:
            # [!!! 已修复 !!!] 添加 weights_only=False
            checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Best model loaded from {best_model_path}")

            model.eval()
            test_ths, test_vms = [], []
            test_batch_count = 0
            pbar_test = tqdm(dl_te, desc="[Test]")
            with torch.no_grad():
                for batch in pbar_test:
                    test_batch_count += 1
                    z_seq = batch["z"].to(args.device)
                    feat_seq = batch["feat"].to(args.device)
                    x_gt  = batch["x"].to(args.device) # !! 仅用于评估 !!

                    x_hat = model(z_seq, feat_seq)

                    # 计算 RMSE
                    th_rmse, vm_rmse = rmse_metrics(x_hat, x_gt, state_order='vm_va')
                    test_ths.append(th_rmse); test_vms.append(vm_rmse)

            avg_test_th_rmse = np.nanmean(test_ths) if test_ths else float('nan')
            avg_test_vm_rmse = np.nanmean(test_vms) if test_vms else float('nan')
            print(f"[TEST] θ-RMSE={avg_test_th_rmse:.3f}°, |V|-RMSE={avg_test_vm_rmse:.4f}")

            if wandb_run:
                wandb_run.summary["test_rmse_theta_deg"] = avg_test_th_rmse
                wandb_run.summary["test_rmse_vm_pu"] = avg_test_vm_rmse
        except pickle.UnpicklingError as e:
             print(f"错误: 加载检查点失败，即使设置了 weights_only=False。文件可能已损坏或包含无法识别的类。错误: {e}")
        except KeyError as e:
             print(f"错误: 检查点文件中缺少预期的键 (例如 'model_state_dict')。错误: {e}")
        except Exception as e: # 捕获其他可能的加载错误
             print(f"加载或测试模型时发生未知错误: {e}")
             
    else:
        print(f"警告: 最佳模型检查点未找到于 {best_model_path}。跳过测试评估。")


    if wandb_run:
        wandb_run.finish()

if __name__ == "__main__":
    main()