# 位置: train/train_pgr_hybrid.py
#
# PGR 模型混合训练脚本
# [已修复 v3] 使用正确的 PhysicsInformedLoss
# [已修复 v4] 添加 tqdm 导入
# [已更新 v5] 添加训练结束后的测试集评估和 WandB 日志记录
# [已更新 v6] 修复 torch.load 因 weights_only=True (PyTorch >= 2.6 默认) 导致加载失败的问题

import os, math, argparse, sys
import numpy as np
import torch
import torch.nn as nn
import wandb # 可选，用于实验跟踪
import pandapower as pp
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle # 用于修复 torch.load 可能遇到的问题

# --- 项目导入 ---
# <<< NEW: 确保根目录在 sys.path 中 >>>
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
# <<< END NEW >>>

from models.pgr import PGR
from physics.losses import PhysicsInformedLoss
from train.dataset import WindowDataset
from tools.metrics import StateLoss, rmse_metrics

# <<< NEW: 添加 str2bool 函数以兼容 wandb sweep >>>
def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
# <<< END NEW >>>

# --- (可选) 辅助函数 ---
# <<< NEW: 设置随机种子函数 >>>
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # Ensure all GPUs are seeded if using multiple
# <<< END NEW >>>


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
    ap.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for GRU/MLP layers")
    ap.add_argument("--gnn_dropout", type=float, default=0.0, help="Dropout rate for GNN layers")
    ap.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for AdamW optimizer")

    # 混合损失权重
    ap.add_argument("--alpha", type=float, default=1.0, help="物理损失 (PhysicsInformedLoss) 的权重")
    ap.add_argument("--beta", type=float, default=10.0, help="监督损失 (StateLoss/MSE) 的权重")

    # PhysicsInformedLoss 内部权重
    ap.add_argument("--lambda_op", type=float, default=1.0, help="权重: 运行约束 (Phi_op) in PhysicsInformedLoss")
    ap.add_argument("--lambda_pf", type=float, default=0.1, help="Weight for power balance loss")
    ap.add_argument("--lambda_smooth", type=float, default=0.01, help="Weight for spatial smoothing loss")

    # 其他
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb_project", type=str, default="PGR_Hybrid_Training", help="WandB 项目名称 (可选)")
    ap.add_argument("--wandb_group", type=str, default=None, help="WandB group name (e.g., for grouping seeds)")
    ap.add_argument("--wandb_name", type=str, default=None, help="WandB run name (overrides default naming)")
    ap.add_argument("--save_dir", type=str, default="checkpoints", help="模型保存目录")
    #----wandb参数--
    ap.add_argument("--wandb_group", type=str, default=None, help="WandB group name (e.g., for grouping seeds)")
    ap.add_argument("--wandb_name", type=str, default=None, help="WandB run name (overrides default naming)")

    args = ap.parse_args()

    set_seed(args.seed)

    # (可选) 初始化 wandb
    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                config=args,
                group=args.wandb_group,
                name=args.wandb_name or f"pgr_a{args.alpha}_b{args.beta}_lr{args.lr}_{args.tag}_seed{args.seed}"
            )
        except ImportError:
            print("WandB 未安装。跳过 WandB 初始化。")
            args.wandb_project = None


    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    # --- 2. 加载数据 ---
    data_path = args.data_dir
    p_train = os.path.join(data_path, f"{args.tag}_train.npz")
    p_val   = os.path.join(data_path, f"{args.tag}_val.npz")
    p_test  = os.path.join(data_path, f"{args.tag}_test.npz")

    try:
        train_dataset = WindowDataset(p_train, input_mode="raw")
        val_dataset = WindowDataset(p_val, input_mode="raw")
        test_dataset = WindowDataset(p_test, input_mode="raw")
    except FileNotFoundError as e:
        print(f"错误: 数据文件未找到: {e}. 请先运行 run_wls_ieee33_from_simbench.py 生成数据。")
        if wandb_run: wandb_run.finish(exit_code=1)
        return

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)


    # 获取维度信息
    try:
        with np.load(p_train, allow_pickle=True) as D_sample:
            S = D_sample["x"].shape[2]
            F = D_sample["feat"].shape[2]
            M = D_sample["z"].shape[2]
            meta = D_sample["meta"].item() if "meta" in D_sample and D_sample["meta"] else {}
    except Exception as e:
        print(f"从 {p_train} 加载数据维度时出错: {e}")
        if wandb_run: wandb_run.finish(exit_code=1)
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
        'dropout': args.dropout,
        'gnn_hidden': args.gnn_hidden,
        'gnn_layers': args.gnn_layers,
        'gnn_dropout': args.gnn_dropout
    }

    model = PGR(cfg_base=cfg, cfg_refiner=cfg).to(device)
    if wandb_run:
        wandb_run.watch(model, log="all", log_freq=max(100, len(train_loader)//2))

    # --- 4. 初始化损失函数 ---
    loss_physics_fn = PhysicsInformedLoss(
        nbus=Nbus,
        slack_pos=slack_pos_val,
        lambda_pf=args.lambda_pf,
        lambda_op=args.lambda_op,
        lambda_smooth=args.lambda_smooth
    ).to(device)

    loss_mse_fn = StateLoss(bus_count=Nbus, state_order='vm_va').to(device)

    # --- 5. 初始化优化器和调度器 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)


    # --- 6. 训练循环 ---
    best_val_loss = float('inf')
    best_epoch = -1
    model_base_name = args.wandb_name if args.wandb_name else f"pgr_hybrid_best_{args.tag}_seed{args.seed}"
    save_path = os.path.join(args.save_dir, f"{model_base_name}.pt")

    for epoch in range(1, args.epochs + 1):
        # --- 训练 ---
        model.train()
        total_loss, total_physics_loss, total_mse_loss = 0, 0, 0
        total_phys_res, total_phys_op, total_phys_pf, total_phys_smooth = 0, 0, 0, 0
        batch_count = 0

        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", leave=False)
        for batch in pbar_train:
            batch_count += 1
            z_seq = batch["z"].to(device)
            feat_seq = batch["feat"].to(device)
            x_gt_seq = batch["x"].to(device)
            R_seq = batch["R"].to(device)
            if "ztype" not in batch:
                print("错误: batch 中未找到 'ztype'。")
                if wandb_run: wandb_run.finish(exit_code=1)
                return
            ztype_np_batch = batch["ztype"].numpy()
            if ztype_np_batch.ndim == 3 and ztype_np_batch.shape[1] == 4:
                 ztype_np_batch = np.repeat(ztype_np_batch[:, np.newaxis, :, :], z_seq.shape[1], axis=1)
            elif ztype_np_batch.ndim != 4:
                 print(f"错误: ztype 维度不匹配 {ztype_np_batch.shape}, 跳过 batch。")
                 continue

            optimizer.zero_grad()
            state_base, state_final = model(z_seq, feat_seq)

            batch_gpu = {"z": z_seq, "R": R_seq, "ztype": ztype_np_batch}
            loss_physics, loss_dict_phys = loss_physics_fn(state_base, batch_gpu)
            loss_mse, (l_th_train, l_v_train) = loss_mse_fn(state_final, x_gt_seq)
            loss_total = (args.alpha * loss_physics) + (args.beta * loss_mse)

            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss_total.item()
            total_physics_loss += loss_physics.item()
            total_mse_loss += loss_mse.item()
            total_phys_res += loss_dict_phys.get('loss_residual', torch.tensor(0.0)).item()
            total_phys_op += loss_dict_phys.get('loss_op', torch.tensor(0.0)).item()
            total_phys_pf += loss_dict_phys.get('loss_pf', torch.tensor(0.0)).item()
            total_phys_smooth += loss_dict_phys.get('loss_smooth', torch.tensor(0.0)).item()

            pbar_train.set_postfix({
                'Loss': f"{loss_total.item():.4e}",
                'Phys': f"{loss_physics.item():.4e}",
                'MSE': f"{loss_mse.item():.4e}",
            })

        avg_train_loss = total_loss / batch_count if batch_count > 0 else 0
        avg_physics_loss = total_physics_loss / batch_count if batch_count > 0 else 0
        avg_mse_loss = total_mse_loss / batch_count if batch_count > 0 else 0
        avg_phys_res = total_phys_res / batch_count if batch_count > 0 else 0
        avg_phys_op = total_phys_op / batch_count if batch_count > 0 else 0
        avg_phys_pf = total_phys_pf / batch_count if batch_count > 0 else 0
        avg_phys_smooth = total_phys_smooth / batch_count if batch_count > 0 else 0

        # --- 验证 ---
        model.eval()
        total_val_loss_mse = 0
        all_val_th_rmse, all_val_vm_rmse = [], []
        val_batch_count = 0

        pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]", leave=False)
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

        avg_va_loss = total_val_loss_mse / val_batch_count if val_batch_count > 0 else 0
        avg_val_th_rmse = np.nanmean(all_val_th_rmse) if all_val_th_rmse else float('nan')
        avg_val_vm_rmse = np.nanmean(all_val_vm_rmse) if all_val_vm_rmse else float('nan')

        # --- Scheduler Step & Logging ---
        scheduler.step(avg_va_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"[Epoch {epoch:03d}] Train Loss: {avg_train_loss:.4e} (Phys: {avg_physics_loss:.4e}, MSE: {avg_mse_loss:.4e}) | "
              f"Val MSE Loss: {avg_va_loss:.4e} | Val RMSE: θ={avg_val_th_rmse:.3f}°, V={avg_val_vm_rmse:.5f} | LR: {current_lr:.2e}")

        if wandb_run:
            log_data = {
                "epoch": epoch,
                "train_loss_total": avg_train_loss,
                "train_loss_physics": avg_physics_loss,
                "train_loss_mse": avg_mse_loss,
                "train_loss_phys_residual": avg_phys_res,
                "train_loss_phys_op": avg_phys_op,
                "train_loss_phys_pf": avg_phys_pf,
                "train_loss_phys_smooth": avg_phys_smooth,
                "val_loss_mse": avg_va_loss,
                "val_rmse_theta_deg": avg_val_th_rmse,
                "val_rmse_vm_pu": avg_val_vm_rmse,
                "learning_rate": current_lr
            }
            log_data_filtered = {k: v for k, v in log_data.items() if not (isinstance(v, float) and np.isnan(v))}
            wandb_run.log(log_data_filtered)

        # --- 保存最佳模型 ---
        if avg_va_loss < best_val_loss:
            best_val_loss = avg_va_loss
            best_epoch = epoch
            print(f"   => Validation loss improved to {best_val_loss:.4e} at epoch {epoch}. Saving model...")
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss_mse': best_val_loss,
                    'args': args
                }, save_path)
                print(f"   => Best model saved to {save_path}")
                if wandb_run:
                    wandb_run.summary["best_val_loss_mse"] = best_val_loss
                    wandb_run.summary["best_epoch"] = epoch
                    wandb_run.summary["best_val_rmse_theta_deg"] = avg_val_th_rmse
                    wandb_run.summary["best_val_rmse_vm_pu"] = avg_val_vm_rmse
            except Exception as e:
                print(f"错误: 保存检查点失败: {e}")

    # --- 训练结束 ---
    print(f"\n训练完成。最佳验证 MSE 损失: {best_val_loss:.4e} 在 epoch {best_epoch}")

    # <<< NEW: 测试集评估 >>>
    print("\n--- 开始测试集评估 ---")
    if os.path.exists(save_path):
        try:
            # <<< MODIFIED: Set weights_only=False >>>
            # 尝试加载 checkpoint，明确允许加载 Python 对象
            checkpoint = torch.load(save_path, map_location=device, weights_only=False)
            # <<< END MODIFIED >>>

            loaded_args = checkpoint.get('args', args)
            test_cfg = {
                'meas_dim': M, 'feat_dim': F, 'state_dim': S,
                'hidden_dim': loaded_args.hidden_dim,
                'num_layers': loaded_args.num_layers,
                'dropout': getattr(loaded_args, 'dropout', 0.1),
                'gnn_hidden': loaded_args.gnn_hidden,
                'gnn_layers': loaded_args.gnn_layers,
                'gnn_dropout': getattr(loaded_args, 'gnn_dropout', 0.0),
            }
            model_test = PGR(cfg_base=test_cfg, cfg_refiner=test_cfg).to(device)

            state_dict = checkpoint['model_state_dict']
            # --- vvvvv 使用上一轮的 State Dict Key Remapping vvvvv ---
            new_state_dict = {}
            for k, v in state_dict.items():
                new_k = k
                if new_k.startswith('module.'):
                    new_k = new_k[len('module.'):]
                # Base Estimator remapping
                if new_k.startswith('base_estimator.fusion_align_g.'):
                    new_k = new_k.replace('base_estimator.fusion_align_g.', 'base_estimator.fusion_module.align_g.')
                elif new_k.startswith('base_estimator.fusion_align_s.'):
                    new_k = new_k.replace('base_estimator.fusion_align_s.', 'base_estimator.fusion_module.align_s.')
                elif new_k.startswith('base_estimator.fusion_gate_alpha.'):
                    new_k = new_k.replace('base_estimator.fusion_gate_alpha.', 'base_estimator.fusion_module.gate_alpha.')
                # Refiner Net remapping
                elif new_k.startswith('refiner_net.fusion_align_g.'):
                    new_k = new_k.replace('refiner_net.fusion_align_g.', 'refiner_net.fusion_module.align_g.')
                elif new_k.startswith('refiner_net.fusion_align_s.'):
                    new_k = new_k.replace('refiner_net.fusion_align_s.', 'refiner_net.fusion_module.align_s.')
                elif new_k.startswith('refiner_net.fusion_gate_alpha.'):
                    new_k = new_k.replace('refiner_net.fusion_gate_alpha.', 'refiner_net.fusion_module.gate_alpha.')
                new_state_dict[new_k] = v
            # --- ^^^^^ 使用上一轮的 State Dict Key Remapping ^^^^^ ---

            # 加载 remapped state dict, 可以尝试 strict=True
            load_result = model_test.load_state_dict(new_state_dict, strict=True) # <<< MODIFIED: Try strict=True >>>
            print("State dict loaded successfully!")
            # --- END MODIFIED ---

            model_test.eval()
            print(f"成功加载并映射最佳模型: {save_path}")

            all_test_th_rmse, all_test_vm_rmse = [], []
            pbar_test = tqdm(test_loader, desc="[Test]")
            with torch.no_grad():
                for batch in pbar_test:
                    z_seq = batch["z"].to(device)
                    feat_seq = batch["feat"].to(device)
                    x_gt_seq = batch["x"].to(device)

                    _, state_final = model_test(z_seq, feat_seq)

                    th_rmse, vm_rmse = rmse_metrics(state_final, x_gt_seq, state_order='vm_va')
                    all_test_th_rmse.append(th_rmse)
                    all_test_vm_rmse.append(vm_rmse)

            avg_test_th_rmse = np.nanmean(all_test_th_rmse) if all_test_th_rmse else float('nan')
            avg_test_vm_rmse = np.nanmean(all_test_vm_rmse) if all_test_vm_rmse else float('nan')

            print(f"[TEST] 最佳模型评估结果: θ-RMSE={avg_test_th_rmse:.3f}°, |V|-RMSE={avg_test_vm_rmse:.5f}")

            if wandb_run:
                wandb_run.summary["test_rmse_theta_deg"] = avg_test_th_rmse
                wandb_run.summary["test_rmse_vm_pu"] = avg_test_vm_rmse
                print("测试结果已记录到 WandB summary。")

        except Exception as e: # 保留通用错误捕获
            print(f"错误: 加载最佳模型或进行测试时出错: {e}")
            if wandb_run:
                 wandb_run.summary["test_rmse_theta_deg"] = float('nan')
                 wandb_run.summary["test_rmse_vm_pu"] = float('nan')

    else:
        print(f"警告: 最佳模型检查点未找到于 {save_path}。跳过测试评估。")
        if wandb_run:
             wandb_run.summary["test_rmse_theta_deg"] = float('nan')
             wandb_run.summary["test_rmse_vm_pu"] = float('nan')
    # <<< END NEW >>>

    if wandb_run:
        wandb_run.finish()

if __name__ == "__main__":
    main()

