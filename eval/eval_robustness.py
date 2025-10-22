# 位置: eval/eval_robustness.py
#
# 核心评估脚本：鲁棒性测试 (量测缺失 & 坏数据)
# [已修复 v3] 移除不必要的 'v_slack' 访问

import os, math, argparse, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

# --- 项目导入 ---
from train.dataset import WindowDataset
from tools.metrics import rmse_metrics

# --- 模型导入 ---
from models.refine_seq import RefineSeqTAModel
from models.tagru import TopoAlignGRU
from models.pgr import PGR

# --- 物理/WLS 导入 ---
from build_ieee33_with_pp import build_ieee33
from measurements_and_wls import state_estimation 
import pandapower as pp

# ===================================================================
# 辅助函数 
# ===================================================================

_ybus, _baseMVA, _slack_pos = None, None, None

def init_phys_meta(data_dir):
    """ 从 grid.json 初始化 WLS 所需的物理参数 """
    global _ybus, _baseMVA, _slack_pos
    if _ybus is not None:
        return
        
    try:
        net = pp.from_json(os.path.join(data_dir, "grid.json"))
        _ybus = net["_ppc"]["internal"]["Ybus"].toarray()
        _baseMVA = net["sn_mva"]
        _slack_pos = int(net.ext_grid["bus"].iloc[0])
        if _slack_pos != 0:
            _slack_pos = 0 
            
    except Exception as e:
        print(f"加载 grid.json 时出错: {e}。 Fallback 到 build_ieee33...")
        _, _ybus, _baseMVA, *_ = build_ieee33()
        _ybus = _ybus.astype(np.complex128)
        _slack_pos = 0 # 对应 slack_bus=1

# [已修改] get_wls_estimate 不再接收 v_slack
def get_wls_estimate(z_k, R_k, ztype_k):
    """
    (已修改 v2)
    在 *单* 个时间步上运行 WLS 估计。
    调用 'state_estimation' 函数。
    假定 R_k (M,) 是协方差矩阵 (sigma^2) 的 *对角线*。
    """
    global _ybus, _baseMVA, _slack_pos
    
    err_cov = np.diag(np.maximum(R_k, 1e-12)) # 加 epsilon 防奇异
    
    try:
        v_phasor, k = state_estimation(
            ybus=_ybus,
            z=z_k,
            ztype=ztype_k,
            err_cov=err_cov,
            iter_max=10,
            threshold=1e-5,
            vtrue=None,
            baseMVA=_baseMVA,
            slack_bus=1 
        )
        
        success = not np.isnan(v_phasor).any()
        v_mag_wls = np.abs(v_phasor)
        v_ang_wls = np.angle(v_phasor)
        
    except (np.linalg.LinAlgError, NotImplementedError, ValueError) as e:
        success = False
        n_bus = _ybus.shape[0]
        v_mag_wls = np.full(n_bus, np.nan)
        v_ang_wls = np.full(n_bus, np.nan)
    
    if not success:
        n_bus = _ybus.shape[0]
        v_mag_wls.fill(np.nan)
        v_ang_wls.fill(np.nan)
        
    x_wls = np.concatenate([v_mag_wls, v_ang_wls], axis=0)
    return x_wls

def make_loader(npz_path, batch_size, shuffle=False):
    """ (Copied from train_refine_baseline.py) """
    ds = WindowDataset(npz_path, input_mode="raw")
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)
    D = np.load(npz_path, allow_pickle=True)
    
    meta = {}
    if "meta" in D.files:
        meta = D["meta"].item()
        
    S = D["x"].shape[2]
    F = D["feat"].shape[2]
    M = D["z"].shape[2]
    Nbus = int(meta.get("bus_count", S // 2))
    
    dims = {'S': S, 'F': F, 'M': M, 'Nbus': Nbus}
    return dl, dims, meta


# ===================================================================
# 攻击函数 (Attack Functions) - [无变更]
# ===================================================================

def add_missing_measurements(z, R, meas_mask, missing_rate=0.2):
    """
    在已有掩码的基础上，随机移除更多量测。
    输入: torch.Tensors (B, W, M)
    """
    B, W, M = z.shape
    device = z.device
    
    new_mask = meas_mask.clone().float()
    
    for b in range(B):
        for t in range(W):
            sample_mask = new_mask[b, t]
            valid_indices = torch.where(sample_mask > 0.5)[0]
            
            n_valid = len(valid_indices)
            if n_valid == 0:
                continue
            
            n_to_remove = int(np.floor(n_valid * missing_rate))
            if n_to_remove == 0:
                continue
                
            remove_indices_local = np.random.choice(n_valid, n_to_remove, replace=False)
            remove_indices_global = valid_indices[torch.from_numpy(remove_indices_local)]
            
            new_mask[b, t, remove_indices_global] = 0.0
            
    R_attacked = R.clone()
    R_attacked[~new_mask.bool()] = 1e9 # 设置为极大的协方差 (极小的权重)
    
    return z, R_attacked, new_mask.bool()


def add_bad_data(z, R, meas_mask, noise_level=10.0, bad_data_rate=0.05):
    """
    向少量 *有效* 量测注入大噪声。
    输入: torch.Tensors (B, W, M)
    """
    z_corrupted = z.clone()
    R_attacked = R.clone() # R 矩阵不变
    
    B, W, M = z.shape
    for b in range(B):
        for t in range(W):
            sample_z = z_corrupted[b, t]
            sample_R = R_attacked[b, t]
            sample_mask = meas_mask[b, t]
            
            valid_indices = torch.where(sample_mask)[0]
            if len(valid_indices) == 0:
                continue
                
            n_to_corrupt = int(np.ceil(len(valid_indices) * bad_data_rate))
            if n_to_corrupt == 0:
                continue
            
            corrupt_indices_in_valid = np.random.choice(len(valid_indices), n_to_corrupt, replace=False)
            corrupt_indices_global = valid_indices[torch.from_numpy(corrupt_indices_in_valid)]
            
            sigma = torch.sqrt(sample_R[corrupt_indices_global] + 1e-9)
            
            noise = (torch.randn(n_to_corrupt, device=z.device) * sigma * noise_level)
            
            z_corrupted[b, t, corrupt_indices_global] += noise

    return z_corrupted, R_attacked, meas_mask


# ===================================================================
# 主评估函数 - [已修复 v3]
# ===================================================================

def evaluate_models(loader, device, models, attack_fn, attack_level):
    """
    在给定的数据集和攻击下评估所有模型。
    """
    results = {name: {'th_rmse': [], 'vm_rmse': []} for name in models.keys()}
    
    for model in models.values():
        if isinstance(model, nn.Module):
            model.eval()

    pbar = tqdm(loader, desc=f"Attack Lvl {attack_level}")
    for batch in pbar:
        # 1. --- 数据准备 ---
        z_seq      = batch["z"].to(device)
        R_seq      = batch["R"].to(device) # (B, W, M) 协方差对角线
        feat_seq   = batch["feat"].to(device)
        x_gt_seq   = batch["x"].to(device)
        
        ztype_np   = batch["ztype"].numpy()
        # [!!! 已删除 !!!] v_slack_np 不再需要
        # v_slack_np = batch["v_slack"].numpy() 
        
        B, W, Nbus = z_seq.shape[0], z_seq.shape[1], (x_gt_seq.shape[-1] // 2)

        # 从 R 矩阵创建初始 meas_mask
        meas_mask  = (R_seq < 1e9).bool()

        # 2. --- 应用攻击 ---
        z_atk, R_atk, mask_atk = attack_fn(z_seq, R_seq, meas_mask, attack_level)
        
        # 3. --- 模型评估 ---
        
        # --- WLS 基线 (在污染数据上重算) ---
        if "WLS" in models:
            x_wls_atk_batch = []
            z_atk_np = z_atk.cpu().numpy()
            R_atk_np = R_atk.cpu().numpy() # (B, W, M)
            
            for b in range(B):
                x_wls_atk_window = []
                for t in range(W):
                    z_k = z_atk_np[b, t]
                    R_k = R_atk_np[b, t] # 协方差对角线 (M,)
                    
                    # [已修改] 不再传递 v_slack_np
                    x_wls_k = get_wls_estimate(
                        z_k, 
                        R_k, # (M,) 协方差对角线
                        ztype_np[b, t] # (4, M)
                    )
                    x_wls_atk_window.append(x_wls_k)
                x_wls_atk_batch.append(np.stack(x_wls_atk_window, axis=0))
            
            x_wls_atk_seq = torch.from_numpy(np.stack(x_wls_atk_batch, axis=0)).float().to(device)
            th, vm = rmse_metrics(x_wls_atk_seq, x_gt_seq, state_order='vm_va')
            results['WLS']['th_rmse'].append(th)
            results['WLS']['vm_rmse'].append(vm)

        # --- tagru (弱监督模型) ---
        if "tagru" in models:
            with torch.no_grad():
                x_hat_tagru = models['tagru'](z_atk, feat_seq)
            th, vm = rmse_metrics(x_hat_tagru, x_gt_seq, state_order='vm_va')
            results['tagru']['th_rmse'].append(th)
            results['tagru']['vm_rmse'].append(vm)

        # (refine-wls 暂时跳过)
        if "refine-wls" in models:
            pass 

        # --- PGR (混合模型) ---
        if "pgr" in models:
            with torch.no_grad():
                _, x_hat_pgr = models['pgr'](z_atk, feat_seq)
            th, vm = rmse_metrics(x_hat_pgr, x_gt_seq, state_order='vm_va')
            results['pgr']['th_rmse'].append(th)
            results['pgr']['vm_rmse'].append(vm)

    # 计算均值
    final_results = {}
    for name, res in results.items():
        if res['th_rmse']:
            final_results[name] = {
                'th_rmse': np.nanmean(res['th_rmse']),
                'vm_rmse': np.nanmean(res['vm_rmse'])
            }
    return final_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/windows_ieee33", help="数据目录 (包含 grid.json 和 W24_test.npz)")
    ap.add_argument("--tag", type=str, default="W24", help="数据标签 (W24, W96)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # --- 模型检查点路径 ---
    ap.add_argument("--ckpt_tagru", type=str, default="", help="已训练的 'tagru' (弱监督) 模型 checkpoint 路径")
    ap.add_argument("--ckpt_refine", type=str, default="", help="(可选) 已训练的 'refine-wls' 模型 checkpoint 路径")
    ap.add_argument("--ckpt_pgr", type=str, default="", help="已训练的 'PGR' (混合) 模型 checkpoint 路径")
    
    # --- 攻击级别 ---
    ap.add_argument("--missing_levels", type=float, nargs='+', default=[0.1, 0.3, 0.5], help="量测缺失率")
    ap.add_argument("--baddata_levels", type=float, nargs='+', default=[0.05, 0.1, 0.15], help="坏数据率 (噪声固定为 10σ)")
    
    args = ap.parse_args()

    device = torch.device(args.device)
    
    # --- 1. 初始化物理参数和数据加载器 ---
    init_phys_meta(args.data_dir)
    p_test = os.path.join(args.data_dir, f"{args.tag}_test.npz")
    
    test_loader, dims, meta = make_loader(p_test, args.batch_size, shuffle=False)
    
    Nbus = dims['Nbus']
    
    # --- 2. 加载模型 ---
    models = {}
    models['WLS'] = "algorithm"

    cfg = {
        'meas_dim': dims['M'], 
        'feat_dim': dims['F'], 
        'state_dim': dims['S'],
        'hidden_dim': 256, 
        'num_layers': 2,
        'gnn_hidden': 128,
        'gnn_layers': 2
    }

    if args.ckpt_tagru:
        try:
            model_tagru = TopoAlignGRU(**cfg).to(device)
            model_tagru.load_state_dict(torch.load(args.ckpt_tagru, map_location=device)['state_dict'])
            models['tagru'] = model_tagru
            print(f"成功加载 'tagru' (BaseEstimator) 模型: {args.ckpt_tagru}")
        except Exception as e:
            print(f"加载 'tagru' 失败: {e}")

    # (refine-wls 加载暂时跳过)

    if args.ckpt_pgr:
        try:
            model_pgr = PGR(cfg_base=cfg, cfg_refiner=cfg).to(device)
            model_pgr.load_state_dict(torch.load(args.ckpt_pgr, map_location=device)['state_dict'])
            models['pgr'] = model_pgr
            print(f"成功加载 'PGR' (Hybrid) 模型: {args.ckpt_pgr}")
        except Exception as e:
            print(f"加载 'PGR' 失败: {e}")

    # --- 3. 定义评估场景 ---
    scenarios = {}
    scenarios["Clean"] = (lambda z, R, m, l: (z, R, m), 0.0)
    
    for level in args.missing_levels:
        scenarios[f"Missing {int(level*100)}%"] = (add_missing_measurements, level)
        
    for level in args.baddata_levels:
        noise_sigma = 10.0
        scenarios[f"Bad Data {int(level*100)}% ({noise_sigma:.0f}σ)"] = (
            lambda z, R, m, l_tuple: add_bad_data(z, R, m, bad_data_rate=l_tuple[0], noise_level=l_tuple[1]),
            (level, noise_sigma)
        )

    all_results = []

    # --- 4. 运行评估 ---
    for name, (attack_fn, level) in scenarios.items():
        print(f"\n--- 运行场景: {name} ---")
        
        results = evaluate_models(test_loader, device, models, attack_fn, level)
        
        for model_name, metrics in results.items():
            all_results.append({
                'Scenario': name,
                'Attack Level': f"{level}",
                'Model': model_name,
                'VM_RMSE (pu)': metrics['vm_rmse'],
                'VA_RMSE (deg)': metrics['th_rmse'],
            })

    # --- 5. 打印最终结果 ---
    print("\n\n--- 最终鲁棒性评估结果 ---")
    df = pd.DataFrame(all_results)
    
    df = df.sort_values(by=['Scenario', 'Model'])
    
    try:
        df_pivot = df.pivot_table(
            index=['Scenario', 'Attack Level'], 
            columns='Model', 
            values='VM_RMSE (pu)'
        )
        print("--- VM_RMSE (pu) ---")
        print(df_pivot.to_markdown(floatfmt=".5f"))
        
        df_pivot_va = df.pivot_table(
            index=['Scenario', 'Attack Level'], 
            columns='Model', 
            values='VA_RMSE (deg)'
        )
        print("\n--- VA_RMSE (deg) ---")
        print(df_pivot_va.to_markdown(floatfmt=".3f"))
    except Exception as e:
        print(f"生成 pivot 表格失败 (可能是因为只有 WLS 结果): {e}")
        print("原始数据:")
        print(df.to_markdown(index=False, floatfmt=".5f"))

    output_csv = os.path.join(os.path.dirname(p_test), "robustness_results.csv")
    df.to_csv(output_csv, index=False)
    print(f"\n结果已保存到: {output_csv}")


if __name__ == "__main__":
    main()