# tools/sweep_tagru.py
#
# 使用 Wandb 对 train_tagru_residual_r.py 进行超参数扫描

import wandb
import argparse
import yaml
import sys, os
import torch
# 将项目根目录添加到 sys.path，以便导入 train_tagru_residual_r
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# 从修改后的训练脚本导入核心函数
from train.train_tagru_residual_r import train_evaluate, rmse_metrics # 我们需要 rmse_metrics 来定义目标

# --- 1. 定义扫描配置 ---
def get_sweep_config():
    """定义 Wandb Sweep 的配置字典"""
    sweep_config = {
        'method': 'bayes',  # 使用贝叶斯优化 (或者 'grid', 'random')
        'metric': {
            'name': 'val_th_rmse', # 优化目标：验证集上的相角 RMSE
            'goal': 'minimize'
        },
        'parameters': {
            'lr': {
                'values': [1e-3, 5e-4, 1e-4] # 学习率选项
            },
            'lambda_wls_reg': {
                'values': [1.0, 2.0, 5.0, 10.0, 15.0] # WLS 锚点强度
            },
            'lambda_pf': {
                'values': [0.01, 0.1, 0.5, 1.0] # 功率平衡强度
            },
            'lambda_smooth': {
                'values': [0.001, 0.01, 0.1] # 空间平滑强度
            },
            # --- 固定参数 (也可以加入扫描范围) ---
            'epochs': {'value': 100}, # 增加训练轮数
            'batch_size': {'value': 16},
            'hidden': {'value': 256},
            'layers': {'value': 2},
            'nhead': {'value': 4},
            'lambda_op': {'value': 1.0},
            'w_temp_th': {'value': 0.01},
            'w_temp_vm': {'value': 0.01},
            'data_dir': {'value': 'data/windows_ieee33'},
            'tag': {'value': 'W24'}, # 使用 W24 进行快速扫描
            'device': {'value': 'cuda' if torch.cuda.is_available() else 'cpu'},
            'use_mask': {'value': False},
            'bias_scale': {'value': 3.0}
            # ------------------------------------
        }
    }
    return sweep_config

# --- 2. 定义 Wandb Agent 调用的函数 ---
def run_sweep_iteration():
    """
    执行一次扫描迭代：初始化 wandb，调用训练函数。
    wandb agent 会自动管理超参数。
    """
    # 初始化 wandb run (必须，即使 train_evaluate 内部也 init)
    # project 和 entity 需要替换为您自己的 wandb 项目信息
    run = wandb.init(project="gm-se-sweep", entity="a530029885-southeast-university") # <--- 修改这里

    # wandb.config 包含了当前迭代的超参数组合
    # train_evaluate 函数内部会再次调用 wandb.init 并读取 config
    # 我们直接将 wandb.config (它类似一个字典) 转换为 Namespace
    args = argparse.Namespace(**wandb.config)

    # 调用核心训练评估函数，并启用 wandb 日志
    try:
        results = train_evaluate(args, use_wandb=True)
        # (可选) 可以在这里额外记录一些最终结果到 wandb summary
        if results:
             wandb.summary['final_test_th_rmse'] = results.get('test_th_rmse', float('nan'))
             wandb.summary['final_test_vm_rmse'] = results.get('test_vm_rmse', float('nan'))
    except Exception as e:
        print(f"!!! Training failed for config: {wandb.config} !!!")
        print(e)
        # (可选) 可以在 wandb 中记录失败状态
        # wandb.log({"error": str(e)})
    finally:
        # 确保 wandb run 结束 (即使 train_evaluate 内部也 finish)
        if run:
             run.finish()


# --- 3. 主程序入口 ---
if __name__ == "__main__":
    # 使用 argparse 来控制是初始化 sweep 还是运行 agent
    parser = argparse.ArgumentParser()
    parser.add_argument('--sweep_id', type=str, default=None, help='Existing sweep ID to join')
    parser.add_argument('--project', type=str, default="gm-se-sweep", help='Wandb project name')
    parser.add_argument('--entity', type=str, default="a530029885-southeast-university", help='Wandb entity (username or team)') # <--- 修改这里
    parser.add_argument('--count', type=int, default=20, help='Number of runs for the agent to execute')
    cli_args = parser.parse_args()

    if cli_args.sweep_id:
        sweep_id = cli_args.sweep_id
        print(f"Joining existing sweep: {sweep_id}")
    else:
        # 初始化新的 Sweep
        print("Initializing a new sweep...")
        sweep_config = get_sweep_config()
        sweep_id = wandb.sweep(sweep_config, project=cli_args.project, entity=cli_args.entity)
        print(f"Sweep initialized. ID: {sweep_id}")
        print("Run the following command in potentially multiple terminals to start agents:")
        print(f"wandb agent {cli_args.entity}/{cli_args.project}/{sweep_id}")

    # 启动 Agent (如果提供了 sweep_id 或刚刚初始化了新的)
    if sweep_id:
        print(f"Starting wandb agent for sweep {sweep_id}...")
        try:
            wandb.agent(sweep_id, function=run_sweep_iteration, count=cli_args.count, project=cli_args.project, entity=cli_args.entity)
        except KeyboardInterrupt:
            print("Agent stopped manually.")
        except Exception as e:
             print(f"An error occurred during agent execution: {e}")