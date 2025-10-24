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
    # 增加 try-except 块以防 wandb 未安装
    wandb_run = None
    if args.wandb_project:
        try:
             # Check if wandb is installed before initializing
             import wandb
             wandb_run = wandb.init(project=args.wandb_project, config=args)
             wandb_run.name = f"pgr_alpha{args.alpha}_beta{args.beta}_lr{args.lr}_{args.tag}"
        except ImportError:
             print("WandB 未安装。跳过 WandB 初始化。")
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
    # 使用 try-except 增加鲁棒性
    try:
        # 使用 with 确保文件关闭
        with np.load(p_train, allow_pickle=True) as D_sample:
            S = D_sample["x"].shape[2]
            F = D_sample["feat"].shape[2]
            M = D_sample["z"].shape[2]
            meta = D_sample["meta"].item() if "meta" in D_sample and D_sample["meta"] else {} # 添加检查确保 meta 存在
    except FileNotFoundError:
        print(f"错误: 训练数据文件未在 {p_train} 找到。请确保数据存在。")
        return
    except Exception as e:
        print(f"从 {p_train} 加载数据维度时出错: {e}")
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
    if wandb_run: # 仅在 wandb 初始化成功时 watch
        wandb_run.watch(model, log="all")

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
        batch_count = 0 # 用于安全计算平均值

        # [!!! 已修复 !!!] 导入 tqdm 后可以使用
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for batch in pbar_train:
            batch_count += 1
            # 数据移动到设备
            z_seq = batch["z"].to(device)
            feat_seq = batch["feat"].to(device)
            x_gt_seq = batch["x"].to(device) # 真值标签
            R_seq = batch["R"].to(device)
            # 确保 'ztype' 存在于 batch 中, 处理可能的 KeyError
            if "ztype" not in batch:
                print("错误: batch 中未找到 'ztype'。请检查数据集准备过程。")
                # 选项 1: 跳过此 batch (可能导致训练偏差)
                # continue
                # 选项 2: 停止训练
                if wandb_run: wandb_run.finish(exit_code=1) # 结束 wandb run
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
                'MSE': f"{loss_mse.item():.4e}",
                'Res': f"{loss_dict_phys.get('loss_residual', torch.tensor(0.0)).item():.4e}",
                'Op': f"{loss_dict_phys.get('loss_op', torch.tensor(0.0)).item():.4e}"
            })

        # 安全计算平均损失
        avg_train_loss = total_loss / batch_count if batch_count > 0 else 0
        avg_physics_loss = total_physics_loss / batch_count if batch_count > 0 else 0
        avg_mse_loss = total_mse_loss / batch_count if batch_count > 0 else 0

        # --- 验证 ---
        model.eval()
        total_val_loss_mse = 0
        all_val_th_rmse, all_val_vm_rmse = [], []
        val_batch_count = 0

        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]")
        with torch.no_grad():
            for batch in pbar_val:
                val_batch_count += 1
                z_seq = batch["z"].to(device)
                feat_seq = batch["feat"].to(device)
                x_gt_seq = batch["x"].to(device)

                _, state_final = model(z_seq, feat_seq)

                val_loss, (l_th_val, l_v_val) = loss_mse_fn(state_final, x_gt_seq)
                total_val_loss_mse += val_loss.item()

                th_rmse, vm_rmse = rmse_metrics(state_final, x_gt_seq, state_order='vm_va')
                all_val_th_rmse.append(th_rmse)
                all_val_vm_rmse.append(vm_rmse)

        # 安全计算平均验证损失和指标
        avg_val_loss = total_val_loss_mse / val_batch_count if val_batch_count > 0 else 0
        # 使用 np.nanmean 处理 rmse_metrics 可能因 WLS 失败返回的 NaNs
        avg_val_th_rmse = np.nanmean(all_val_th_rmse) if all_val_th_rmse else float('nan')
        avg_val_vm_rmse = np.nanmean(all_val_vm_rmse) if all_val_vm_rmse else float('nan')


        scheduler.step()

        print(f"[Epoch {epoch:03d}] Train Loss: {avg_train_loss:.4e} (Phys: {avg_physics_loss:.4e}, MSE: {avg_mse_loss:.4e}) | "
              f"Val MSE Loss: {avg_val_loss:.4e} | Val RMSE: θ={avg_val_th_rmse:.3f}°, V={avg_val_vm_rmse:.5f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        # (可选) WandB 日志
        if wandb_run:
            log_data = {
                "epoch": epoch,
                "train_loss_total": avg_train_loss,
                "train_loss_physics": avg_physics_loss,
                "train_loss_mse": avg_mse_loss,
                # 从 loss_dict_phys 中获取 Res 和 Op (需要修改 train_epoch 返回更多值，或在此处计算平均值)
                # "train_loss_res": avg_res_loss,
                # "train_loss_op": avg_op_loss,
                "val_loss_mse": avg_val_loss,
                "val_rmse_theta_deg": avg_val_th_rmse,
                "val_rmse_vm_pu": avg_val_vm_rmse,
                "learning_rate": optimizer.param_groups[0]['lr']
            }
            # 过滤掉 NaN 值
            log_data_filtered = {k: v for k, v in log_data.items() if not (isinstance(v, float) and np.isnan(v))}
            wandb_run.log(log_data_filtered)


        # --- 保存最佳模型 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # 使用更明确的文件名
            model_save_name = f"{args.wandb_project or 'pgr_hybrid'}_best_{args.tag}.pt"
            save_path = os.path.join(args.save_dir, model_save_name)
            print(f"   => Validation loss improved to {best_val_loss:.4e}. Saving model...")
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'args': args
                }, save_path)
                print(f"   => Best model saved to {save_path}")
                if wandb_run:
                     wandb_run.summary["best_val_loss_mse"] = best_val_loss
                     wandb_run.summary["best_epoch"] = epoch
                     # wandb.save(save_path) # 保存为 artifact
            except Exception as e:
                print(f"错误: 保存检查点失败: {e}")


    # --- 训练结束 ---
    print(f"\n训练完成。最佳验证 MSE 损失: {best_val_loss:.4e}")

    # (可选) 加载最佳模型并进行最终测试集评估
    # ...

    if wandb_run:
        wandb_run.finish()

if __name__ == "__main__":
    main()