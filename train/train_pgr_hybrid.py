# 位置: train/train_pgr_hybrid.py
#
# PGR 模型混合训练脚本
# 结合物理损失 (弱监督) 和状态 MSE 损失 (全监督)

import os, math, argparse, sys
import numpy as np
import torch
import torch.nn as nn
import wandb # 可选，用于实验跟踪
import pandapower as pp
from torch.utils.data import DataLoader

# --- 项目导入 ---
# (假设 pip install -e . 已执行)
from models.pgr import PGR
from physics.losses import InfeasibilityLoss # 假设物理损失在这里
from physics.torch_ac_model import ACModel  # 可微分物理模型
from train.dataset import WindowDataset     # 您的数据集类
from tools.metrics import StateLoss, rmse_metrics # 监督损失和评估指标

# --- (可选) 辅助函数 (可从其他训练脚本导入或复制) ---
# def setup_wandb(args): ...
# def get_data_path(args): ...
# def save_checkpoint(model, optimizer, epoch, path): ...

def main():
    # --- 1. 参数解析 ---
    ap = argparse.ArgumentParser(description="Train PGR Hybrid Model")
    ap.add_argument("--data_dir", type=str, default="data/windows_ieee33")
    ap.add_argument("--tag", type=str, default="W24")
    ap.add_argument("--epochs", type=int, default=100) # 混合训练可能需要更多 epochs
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-4) # 可能需要调整学习率
    
    # 模型超参数 (应与 tagru/refine 保持一致)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--gnn_hidden", type=int, default=128)
    ap.add_argument("--gnn_layers", type=int, default=2)
    
    # 混合损失权重
    ap.add_argument("--alpha", type=float, default=1.0, help="物理损失 (InfeasibilityLoss) 的权重")
    ap.add_argument("--beta", type=float, default=10.0, help="监督损失 (StateLoss/MSE) 的权重")
    
    # 物理损失内部权重 (如果 InfeasibilityLoss 支持)
    # ap.add_argument("--w_v", type=float, default=1.0)
    # ap.add_argument("--w_p", type=float, default=1.0)
    # ap.add_argument("--w_q", type=float, default=1.0)

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
        wandb.init(project=args.wandb_project, config=args)
        wandb.run.name = f"pgr_alpha{args.alpha}_beta{args.beta}_lr{args.lr}_{args.tag}"

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    # --- 2. 加载数据 ---
    # (假设 get_data_path 返回数据根目录)
    # data_path = get_data_path(args) 
    data_path = args.data_dir # 简化
    
    p_train = os.path.join(data_path, f"{args.tag}_train.npz")
    p_val   = os.path.join(data_path, f"{args.tag}_val.npz")
    p_test  = os.path.join(data_path, f"{args.tag}_test.npz") # 测试集用于最终评估

    # **重要**: PGR 输入原始量测 z，所以 input_mode="raw"
    train_dataset = WindowDataset(p_train, input_mode="raw")
    val_dataset = WindowDataset(p_val, input_mode="raw")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 获取维度信息
    D_sample = np.load(p_train, allow_pickle=True)
    S = D_sample["x"].shape[2]  # state_dim (2N)
    F = D_sample["feat"].shape[2] # feat_dim
    M = D_sample["z"].shape[2]  # meas_dim
    meta = D_sample["meta"].item() if "meta" in D_sample else {}
    Nbus = int(meta.get("bus_count", S // 2))

    # --- 3. 初始化模型 ---
    # PGR 的两个阶段使用相同的配置
    cfg = {
        'meas_dim': M, 
        'feat_dim': F, 
        'state_dim': S,
        'hidden_dim': args.hidden_dim, 
        'num_layers': args.num_layers,
        'gnn_hidden': args.gnn_hidden,
        'gnn_layers': args.gnn_layers,
        # TODO: 如果您的模型需要 nbus，请添加 'nbus': Nbus
    }
    
    model = PGR(cfg_base=cfg, cfg_refiner=cfg).to(device)
    if args.wandb_project:
        wandb.watch(model, log="all")

    # --- 4. 初始化物理模型和损失函数 ---
    # 加载电网拓扑 (用于 ACModel)
    try:
        grid_path = os.path.join(data_path, "grid.json")
        net = pp.from_json(grid_path)
    except FileNotFoundError:
        print(f"警告: grid.json 未在 {data_path} 找到。物理损失可能无法正确工作。")
        # TODO: 您可能需要一个 fallback 或确保 grid.json 存在
        net = None # 或者从 build_ieee33 创建一个默认的
        # _, _, _, net, *_ = build_ieee33(create_test_case=False) # 假设 build_ieee33 返回 net

    if net is None:
        print("错误：无法加载电网拓扑。")
        return

    ac_model = ACModel(net=net, device=device)
    
    # 物理损失 (作用于 state_base)
    # TODO: 确认 InfeasibilityLoss 的参数
    # loss_physics_fn = InfeasibilityLoss(ac_model, weight_v=args.w_v, weight_p=args.w_p, weight_q=args.w_q)
    loss_physics_fn = InfeasibilityLoss(ac_model) # 假设默认权重

    # 监督损失 (作用于 state_final)
    # 使用 tools.metrics 中的 StateLoss，注意 state_order
    loss_mse_fn = StateLoss(bus_count=Nbus, state_order='vm_va').to(device) # PGR 输出是 vm_va

    # --- 5. 初始化优化器和调度器 ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5) # AdamW 可能更好
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5) # 或者 StepLR

    # --- 6. 训练循环 ---
    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        # --- 训练 ---
        model.train()
        total_loss, total_physics_loss, total_mse_loss = 0, 0, 0
        
        pbar_train = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for batch in pbar_train:
            # 数据移动到设备
            z_seq = batch["z"].to(device)
            feat_seq = batch["feat"].to(device)
            x_gt_seq = batch["x"].to(device) # 真值标签
            
            # (物理损失所需，如果您的 InfeasibilityLoss 需要它们)
            # R_seq = batch["R"].to(device)
            # meas_mask = (R_seq < 1e9).bool() # 从 R 创建 mask
            # ybus_batch = ... # 可能需要从 ACModel 或 batch 获取
            # v_slack_batch = ...
            # pq_idx_batch = ...

            optimizer.zero_grad()
            
            # 模型前向传播
            state_base, state_final = model(z_seq, feat_seq) # A_time, E_time 暂时省略
            
            # --- 计算混合损失 ---
            # TODO: 确认 InfeasibilityLoss 的调用参数
            # loss_physics = loss_physics_fn(state_base, z_seq, meas_mask, ybus_batch, ...)
            
            # 简化版调用 (假设 InfeasibilityLoss 只需要 state 和 z)
            # 您需要根据 physics/losses.py 调整这里的调用
            try:
                # 尝试一个简化的调用，您需要根据实际情况修改
                # 假设 InfeasibilityLoss 内部处理了 ac_model, z_seq 等
                loss_physics = loss_physics_fn(state_base, z_seq) 
            except TypeError as e:
                print("\n错误：调用 InfeasibilityLoss 时参数不匹配。请检查 train_pgr_hybrid.py 和 physics/losses.py。")
                print(f"错误信息: {e}")
                print("训练中断。")
                return # 停止训练
            
            loss_mse, (l_th_train, l_v_train) = loss_mse_fn(state_final, x_gt_seq)
            
            loss_total = (args.alpha * loss_physics) + (args.beta * loss_mse)
            
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度裁剪
            optimizer.step()
            
            total_loss += loss_total.item()
            total_physics_loss += loss_physics.item()
            total_mse_loss += loss_mse.item()
            
            pbar_train.set_postfix({
                'Loss': f"{loss_total.item():.4e}", 
                'Phys': f"{loss_physics.item():.4e}", 
                'MSE': f"{loss_mse.item():.4e}"
            })

        avg_train_loss = total_loss / len(train_loader)
        avg_physics_loss = total_physics_loss / len(train_loader)
        avg_mse_loss = total_mse_loss / len(train_loader)
        
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
                
                # 主要验证指标：最终输出的 MSE 损失
                val_loss, (l_th_val, l_v_val) = loss_mse_fn(state_final, x_gt_seq)
                total_val_loss_mse += val_loss.item()
                
                # 计算 RMSE 用于报告
                th_rmse, vm_rmse = rmse_metrics(state_final, x_gt_seq, state_order='vm_va')
                all_val_th_rmse.append(th_rmse)
                all_val_vm_rmse.append(vm_rmse)

        avg_val_loss = total_val_loss_mse / len(val_loader)
        avg_val_th_rmse = np.mean(all_val_th_rmse)
        avg_val_vm_rmse = np.mean(all_val_vm_rmse)
        
        scheduler.step() # 更新学习率 (如果是 StepLR)
        # scheduler.step(avg_val_loss) # 如果是 ReduceLROnPlateau

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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'args': args # 保存超参数
            }, save_path)
            print(f"   => Best model saved to {save_path} (Val Loss: {best_val_loss:.4e})")
            if args.wandb_project:
                 wandb.summary["best_val_loss_mse"] = best_val_loss
                 wandb.summary["best_epoch"] = epoch
                 # (可选) 保存模型到 wandb artifacts
                 # wandb.save(save_path)

    # --- 训练结束 ---
    print(f"\nTraining completed. Best Validation MSE Loss: {best_val_loss:.4e}")
    
    # (可选) 加载最佳模型并进行最终测试集评估
    # ...

    if args.wandb_project:
        wandb.finish()

if __name__ == "__main__":
    main()