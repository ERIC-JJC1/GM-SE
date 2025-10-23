# 位置: train/train_pgr_hybrid.py
#
# PGR 模型混合训练脚本
# [已修复 v3] 使用正确的 PhysicsInformedLoss
# [已修复 v4] 添加 tqdm 导入

import os, math, argparse, sys
import numpy as np
import torch
import torch.nn as nn
import wandb # 可选，用于实验跟踪
import pandapower as pp
from torch.utils.data import DataLoader
from tqdm import tqdm # [!!! 已修复 !!!] 添加 tqdm 导入

# --- 项目导入 ---
from models.pgr import PGR
from physics.losses import PhysicsInformedLoss
# [!!! 已移除 !!!] 不再需要 ACModel
# from physics.torch_ac_model import ACModel
from train.dataset import WindowDataset
from tools.metrics import StateLoss, rmse_metrics

# --- (可选) 辅助函数 ---
# def setup_wandb(args): ...
# def get_data_path(args): ...
# def save_checkpoint(model, optimizer, epoch, path): ...

def main():
    # --- 1. 参数解析 ---
    ap = argparse.ArgumentParser(description="Train PGR Hybrid Model")
    ap.add_argument("--data_dir", type=str, default="data/windows_ieee33")
    ap.add_argument("--tag", type=str, default="W24")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4)

    # 模型超参数
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--gnn_hidden", type=int, default=128)
    ap.add_argument("--gnn_layers", type=int, default=2)

    # 混合损失权重
    ap.add_argument("--alpha", type=float, default=1.0, help="物理损失 (PhysicsInformedLoss) 的权重")
    ap.add_argument("--beta", type=float, default=10.0, help="监督损失 (StateLoss/MSE) 的权重")

    # PhysicsInformedLoss 内部权重
    ap.add_argument("--lambda_op", type=float, default=1.0, help="权重: 运行约束 (Phi_op) in PhysicsInformedLoss")

    # 其他
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb_project", type=str, default="PGR_Hybrid_Training", help="WandB 项目名称 (可选)")
    ap.add_argument("--save_dir", type=str, default="checkpoints", help="模型保存目录")

    args = ap.parse_args()

    # 设置随机种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed(args.seed)

    # (可选) 初始化 wandb
    if args.wandb_project:
        try:
             # Check if wandb is installed before initializing
             import wandb
             wandb.init(project=args.wandb_project, config=args)
             wandb.run.name = f"pgr_alpha{args.alpha}_beta{args.beta}_lr{args.lr}_{args.tag}"
        except ImportError:
             print("WandB not installed. Skipping WandB initialization.")
             args.wandb_project = None # Disable wandb logging if not installed


    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    # --- 2. 加载数据 ---
    data_path = args.data_dir

    p_train = os.path.join(data_path, f"{args.tag}_train.npz")
    p_val   = os.path.join(data_path, f"{args.tag}_val.npz")
    p_test  = os.path.join(data_path, f"{args.tag}_test.npz")

    train_dataset = WindowDataset(p_train, input_mode="raw")
    val_dataset = WindowDataset(p_val, input_mode="raw")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 获取维度信息
    # Use try-except for robustness when loading .npz
    try:
        with np.load(p_train, allow_pickle=True) as D_sample:
            S = D_sample["x"].shape[2]
            F = D_sample["feat"].shape[2]
            M = D_sample["z"].shape[2]
            meta = D_sample["meta"].item() if "meta" in D_sample else {}
    except FileNotFoundError:
        print(f"Error: Training data file not found at {p_train}. Please ensure data exists.")
        return
    except Exception as e:
        print(f"Error loading data dimensions from {p_train}: {e}")
        return

    Nbus = int(meta.get("bus_count", S // 2))
    slack_pos_val = int(meta.get("slack_pos", 0))

    # --- 3. 初始化模型 ---
    cfg = {
        'meas_dim': M,
        'feat_dim': F,
        'state_dim': S,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'gnn_hidden': args.gnn_hidden,
        'gnn_layers': args.gnn_layers,
    }

    model = PGR(cfg_base=cfg, cfg_refiner=cfg).to(device)
    if args.wandb_project:
        wandb.watch(model, log="all")

    # --- 4. 初始化损失函数 ---
    loss_physics_fn = PhysicsInformedLoss(
        nbus=Nbus,
        slack_pos=slack_pos_val,
        lambda_op=args.lambda_op
    ).to(device)

    loss_mse_fn = StateLoss(bus_count=Nbus, state_order='vm_va').to(device)

    # --- 5. 初始化优化器和调度器 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # --- 6. 训练循环 ---
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # --- 训练 ---
        model.train()
        total_loss, total_physics_loss, total_mse_loss = 0, 0, 0

        # [!!! 已修复 !!!] 导入 tqdm
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for batch in pbar_train:
            # 数据移动到设备
            z_seq = batch["z"].to(device)
            feat_seq = batch["feat"].to(device)
            x_gt_seq = batch["x"].to(device) # 真值标签
            R_seq = batch["R"].to(device)
            # Ensure 'ztype' exists in batch, handle potential KeyError
            if "ztype" not in batch:
                print("Error: 'ztype' not found in batch. Check dataset preparation.")
                # Option 1: Skip batch (may lead to biased training)
                # continue
                # Option 2: Stop training
                return
            ztype_np = batch["ztype"].numpy() # ztype 保持 numpy


            optimizer.zero_grad()

            # 模型前向传播
            state_base, state_final = model(z_seq, feat_seq)

            # --- 计算混合损失 ---
            batch_gpu = {
                "z": z_seq,
                "R": R_seq,
                "ztype": ztype_np
            }
            loss_physics, loss_dict_phys = loss_physics_fn(state_base, batch_gpu)

            loss_mse, (l_th_train, l_v_train) = loss_mse_fn(state_final, x_gt_seq)

            loss_total = (args.alpha * loss_physics) + (args.beta * loss_mse)

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss_total.item()
            total_physics_loss += loss_physics.item()
            total_mse_loss += loss_mse.item()

            pbar_train.set_postfix({
                'Loss': f"{loss_total.item():.4e}",
                'Phys': f"{loss_physics.item():.4e}",
                'MSE': f"{loss_mse.item():.4e}"
            })

        avg_train_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_physics_loss = total_physics_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_mse_loss = total_mse_loss / len(train_loader) if len(train_loader) > 0 else 0

        # --- 验证 ---
        model.eval()
        total_val_loss_mse = 0
        all_val_th_rmse, all_val_vm_rmse = [], []

        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]")
        with torch.no_grad():
            for batch in pbar_val:
                z_seq = batch["z"].to(device)
                feat_seq = batch["feat"].to(device)
                x_gt_seq = batch["x"].to(device)

                _, state_final = model(z_seq, feat_seq)

                val_loss, (l_th_val, l_v_val) = loss_mse_fn(state_final, x_gt_seq)
                total_val_loss_mse += val_loss.item()

                th_rmse, vm_rmse = rmse_metrics(state_final, x_gt_seq, state_order='vm_va')
                all_val_th_rmse.append(th_rmse)
                all_val_vm_rmse.append(vm_rmse)

        avg_val_loss = total_val_loss_mse / len(val_loader) if len(val_loader) > 0 else 0
        # Use np.nanmean to handle potential NaNs from rmse_metrics if WLS failed
        avg_val_th_rmse = np.nanmean(all_val_th_rmse) if all_val_th_rmse else float('nan')
        avg_val_vm_rmse = np.nanmean(all_val_vm_rmse) if all_val_vm_rmse else float('nan')


        scheduler.step()

        print(f"[Epoch {epoch:03d}] Train Loss: {avg_train_loss:.4e} (Phys: {avg_physics_loss:.4e}, MSE: {avg_mse_loss:.4e}) | "
              f"Val MSE Loss: {avg_val_loss:.4e} | Val RMSE: θ={avg_val_th_rmse:.3f}°, V={avg_val_vm_rmse:.5f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # (可选) WandB 日志
        if args.wandb_project:
            wandb.log({
                "epoch": epoch,
                "train_loss_total": avg_train_loss,
                "train_loss_physics": avg_physics_loss,
                "train_loss_mse": avg_mse_loss,
                "val_loss_mse": avg_val_loss,
                "val_rmse_theta_deg": avg_val_th_rmse,
                "val_rmse_vm_pu": avg_val_vm_rmse,
                "learning_rate": optimizer.param_groups[0]['lr']
            })

        # --- 保存最佳模型 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(args.save_dir, f"pgr_hybrid_best_{args.tag}.pt")
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'args': args
                }, save_path)
                print(f"   => Best model saved to {save_path} (Val Loss: {best_val_loss:.4e})")
                if args.wandb_project:
                     wandb.summary["best_val_loss_mse"] = best_val_loss
                     wandb.summary["best_epoch"] = epoch
                     # wandb.save(save_path)
            except Exception as e:
                print(f"Error saving checkpoint: {e}")


    # --- 训练结束 ---
    print(f"\nTraining completed. Best Validation MSE Loss: {best_val_loss:.4e}")

    # (可选) 加载最佳模型并进行最终测试集评估
    # ...

    if args.wandb_project:
        wandb.finish()

if __name__ == "__main__":
    main()