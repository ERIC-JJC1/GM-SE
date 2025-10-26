# 位置: eval/eval_robustness.py
#
# 核心评估脚本：鲁棒性测试 (量测缺失 & 坏数据)
# [已修复 v3] 移除不必要的 'v_slack' 访问
# [已更新 v4] 添加 --output_csv 参数
# [已更新 v5] 修复 torch.load 因 weights_only=True 报错
# [已更新 v6] 修复数据泄漏问题：根据攻击后的数据重新计算特征 feat_seq

import os, math, argparse, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import pickle # 用于修复 torch.load 可能遇到的问题

# --- 项目导入 ---
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(THIS_DIR)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from train.dataset import WindowDataset
from tools.metrics import rmse_metrics

# --- 模型导入 ---
from models.refine_seq import RefineSeqTAModel
from models.tagru import TopoAlignGRU
from models.pgr import PGR
from models.baselines import GRUBaseline

# --- 物理/WLS 导入 ---
from build_ieee33_with_pp import build_ieee33
from measurements_and_wls import state_estimation
from physics.ac_model import h_measure
import pandapower as pp
from models.pgr import PGR
from models.tagru import TopoAlignGRU
from models.refine_seq import RefineSeqTAModel
from models.baselines import GRUBaseline

# ===================================================================
# 辅助函数
# ===================================================================

_ybus, _baseMVA, _slack_pos = None, None, None
Nbus = None # Global Nbus needed for recalculate_features_torch

def init_phys_meta(data_dir):
    """ 从 grid.json 初始化 WLS 所需的物理参数 (Fallback 到 build_ieee33) """
    global _ybus, _baseMVA, _slack_pos
    if _ybus is not None: return
    grid_json_path = os.path.join(data_dir, "grid.json")
    try:
        # ...(省略了详细的 grid.json 加载逻辑，与上一版相同)...
        if os.path.exists(grid_json_path):
             net = pp.from_json(grid_json_path)
             if "_ppc" not in net or not isinstance(net._ppc, dict) or "internal" not in net._ppc or "Ybus" not in net._ppc["internal"]:
                 print(f"警告: {grid_json_path} 文件结构不完整，缺少 Ybus。 Fallback 到 build_ieee33...")
                 raise FileNotFoundError
             _ybus = net["_ppc"]["internal"]["Ybus"].toarray().astype(np.complex128)
             _baseMVA = net["sn_mva"]
             ext_grid_bus_id = int(net.ext_grid["bus"].iloc[0])
             bus_ids_ordered = net.bus.index.to_numpy()
             slack_pos_candidates = np.where(bus_ids_ordered == ext_grid_bus_id)[0]
             if len(slack_pos_candidates) > 0:
                 _slack_pos = int(slack_pos_candidates[0])
                 print(f"从 grid.json 加载参数: slack bus ID {ext_grid_bus_id} 位于索引 {_slack_pos}")
             else:
                 print(f"警告: 在 grid.json 的 bus 列表中未找到 slack bus ID {ext_grid_bus_id}。 Fallback 到索引 0。")
                 _slack_pos = 0
             if not np.iscomplexobj(_ybus): _ybus = _ybus.astype(np.complex128)
        else:
             print(f"警告: {grid_json_path} 不存在。 Fallback 到 build_ieee33...")
             raise FileNotFoundError
    except Exception as e:
        print(f"加载 grid.json 时出错: {e}。 Fallback 到 build_ieee33...")
        _, ybus_np, baseMVA_np, _, _, bus_ids_np = build_ieee33()
        _ybus = ybus_np.astype(np.complex128)
        _baseMVA = float(baseMVA_np)
        _slack_pos = 0
        print(f"使用 build_ieee33 参数: slack bus ID 1 位于索引 {_slack_pos}")

def get_wls_estimate(z_k, R_k_diag, ztype_k):
    """ (同上一版) 在 *单* 个时间步上运行 WLS 估计 """
    global _ybus, _baseMVA, _slack_pos
    if _ybus is None: raise RuntimeError("网络参数未初始化。")
    if np.all(np.isnan(z_k)):
        n_bus = _ybus.shape[0]
        return np.full(2 * n_bus, np.nan, dtype=np.float32)
    err_cov = np.diag(np.maximum(R_k_diag, 1e-12))
    try:
        v_phasor, k = state_estimation(
            ybus=_ybus, z=z_k, ztype=ztype_k, err_cov=err_cov,
            iter_max=10, threshold=1e-5, vtrue=None, baseMVA=_baseMVA,
            slack_bus=_slack_pos + 1
        )
        success = not np.isnan(v_phasor).any()
    except Exception as e:
        # print(f"警告: WLS state_estimation 失败: {e}") # 减少打印
        success = False
        n_bus = _ybus.shape[0]
        v_phasor = np.full(n_bus, np.nan + 1j*np.nan)
    if success:
        v_mag_wls = np.abs(v_phasor)
        v_ang_wls = np.angle(v_phasor)
        v_ang_wls = v_ang_wls - v_ang_wls[_slack_pos]
        v_ang_wls = (v_ang_wls + np.pi) % (2 * np.pi) - np.pi
    else:
        n_bus = _ybus.shape[0]
        v_mag_wls = np.full(n_bus, np.nan)
        v_ang_wls = np.full(n_bus, np.nan)
    # 输出: [vm, va_rel]
    return np.concatenate([v_mag_wls, v_ang_wls], axis=0).astype(np.float32)

def compute_whitened_residual_torch(z_seq, x_wls_seq, R_seq, ztype_np, baseMVA, slack_pos, ybus_np, device):
    """ (同上一版) Computes whitened residual on torch tensors """
    global Nbus
    if Nbus is None: raise ValueError("Nbus not initialized")
    B, W, M = z_seq.shape
    r_seq_list = []
    z_seq_np = z_seq.cpu().numpy()
    x_wls_seq_np = x_wls_seq.cpu().numpy()
    R_seq_np = R_seq.cpu().numpy()
    for b in range(B):
        r_w_list = []
        for t in range(W):
            z_t = z_seq_np[b, t]
            valid_meas_mask = ~np.isnan(z_t)
            x_wls_t = x_wls_seq_np[b, t]
            if np.isnan(x_wls_t).any():
                r_w_t = np.zeros(M, dtype=np.float32)
            else:
                try:
                    vm_wls_t = x_wls_t[:Nbus]
                    va_rel_wls_t = x_wls_t[Nbus:]
                    va_abs_wls_t = va_rel_wls_t + 0.0 # Assuming slack angle 0
                    x_for_h = np.concatenate([va_abs_wls_t, vm_wls_t])
                    h_wls_t = h_measure(ybus_np, x_for_h, ztype_np[b, t], baseMVA, slack_pos)
                    res_t = z_t - h_wls_t
                    sigma_t = np.sqrt(np.maximum(R_seq_np[b, t], 1e-9))
                    r_w_t = res_t / sigma_t
                    r_w_t[~valid_meas_mask] = 0.0
                    r_w_t[~np.isfinite(r_w_t)] = 0.0
                except Exception as e:
                     # print(f"警告: h_measure 失败 (b={b}, t={t}). Error: {e}") # 减少打印
                     r_w_t = np.zeros(M, dtype=np.float32)
            r_w_list.append(r_w_t[np.newaxis, :])
        r_seq_list.append(np.concatenate(r_w_list, axis=0)[np.newaxis, :, :])
    r_seq_np_final = np.concatenate(r_seq_list, axis=0)
    r_seq_np_final[~np.isfinite(r_seq_np_final)] = 0.0
    return torch.from_numpy(r_seq_np_final).float().to(device)


def make_loader(npz_path, batch_size, shuffle=False, num_workers=0):
    """ (同上一版) Loads data """
    # ...(省略了详细的 loader 逻辑，与上一版相同)...
    try:
        ds = WindowDataset(npz_path, input_mode="raw")
        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False)
        with np.load(npz_path, allow_pickle=True) as D:
            meta = D["meta"].item() if "meta" in D and D["meta"] else {}
            S = D["x"].shape[2]
            F = D["feat"].shape[2]
            M = D["z"].shape[2]
    except FileNotFoundError: print(f"错误: 数据文件未在 {npz_path} 找到。"); sys.exit(1)
    except Exception as e: print(f"加载数据或维度时出错 {npz_path}: {e}"); sys.exit(1)
    Nbus_local = int(meta.get("bus_count", S // 2))
    dims = {'S': S, 'F': F, 'M': M, 'Nbus': Nbus_local}
    return dl, dims, meta


# ===================================================================
# 攻击函数 (Attack Functions) - [无变更]
# ===================================================================
def add_missing_measurements(z, R, meas_mask, missing_rate=0.2):
    """ (同上一版) 随机移除更多量测 """
    # ...(省略了详细的 missing data 逻辑，与上一版相同)...
    B, W, M = z.shape
    new_mask = meas_mask.clone().bool()
    num_to_remove_total = int(B * W * M * missing_rate)
    valid_indices_flat = torch.where(new_mask.view(-1))[0]
    num_valid = len(valid_indices_flat)
    if num_valid == 0 or missing_rate <= 0:
        R_attacked = R.clone(); R_attacked[~new_mask] = 1e9
        return z, R_attacked, new_mask
    num_actually_remove = min(num_to_remove_total, num_valid)
    remove_indices_flat = valid_indices_flat[torch.randperm(num_valid, device=z.device)[:num_actually_remove]]
    new_mask.view(-1)[remove_indices_flat] = False
    R_attacked = R.clone(); R_attacked[~new_mask] = 1e9
    z_attacked = z.clone(); z_attacked[~new_mask] = torch.nan # Mark missing z as NaN
    return z_attacked, R_attacked, new_mask

def add_bad_data(z, R, meas_mask, noise_level=10.0, bad_data_rate=0.05):
    """ (同上一版) 向少量 *有效* 量测注入大噪声 """
    # ...(省略了详细的 bad data 逻辑，与上一版相同)...
    z_corrupted = z.clone()
    R_attacked = R.clone()
    new_mask = meas_mask.clone().bool()
    B, W, M = z.shape
    num_to_corrupt_total = int(B * W * M * bad_data_rate)
    valid_indices_flat = torch.where(new_mask.view(-1))[0]
    num_valid = len(valid_indices_flat)
    if num_valid == 0 or bad_data_rate <= 0:
        return z_corrupted, R_attacked, new_mask
    num_actually_corrupt = min(num_to_corrupt_total, num_valid)
    corrupt_indices_flat = valid_indices_flat[torch.randperm(num_valid, device=z.device)[:num_actually_corrupt]]
    sigma = torch.sqrt(R.view(-1)[corrupt_indices_flat] + 1e-9)
    noise = torch.randn(num_actually_corrupt, device=z.device) * sigma * noise_level
    z_corrupted.view(-1)[corrupt_indices_flat] += noise
    return z_corrupted, R_attacked, new_mask


# <<< NEW: Function to recalculate features based on attacked data >>>
def recalculate_features_torch(z_atk, mask_atk):
    """
    根据攻击后的 z_atk 和 mask_atk (BxWxM) 重新计算特征 (BxWx4)。
    特征: [obs_rate, miss_rate, mean_valid, std_valid]
    """
    B, W, M = z_atk.shape
    feat_dim = 4
    feat_seq_atk = torch.zeros((B, W, feat_dim), device=z_atk.device, dtype=torch.float32)

    for b in range(B):
        for t in range(W):
            z_step = z_atk[b, t]       # (M,)
            mask_step = mask_atk[b, t] # (M,) bool

            # 1 & 2: Observation rate and Missing rate per step
            num_valid = mask_step.sum().float()
            obs_rate = num_valid / M if M > 0 else 0.0
            miss_rate = 1.0 - obs_rate
            feat_seq_atk[b, t, 0] = obs_rate
            feat_seq_atk[b, t, 1] = miss_rate

            # 3 & 4: Mean and Std of *valid* measurements per step
            if num_valid > 0:
                valid_z = z_step[mask_step]
                # Ensure we handle potential NaNs introduced by attacks (e.g., missing data attack sets NaN)
                valid_z_finite = valid_z[torch.isfinite(valid_z)]
                if valid_z_finite.numel() > 0:
                     mean_valid = valid_z_finite.mean()
                     # Use std deviation corrected for bias (ddof=1) if num_valid > 1
                     std_valid = torch.std(valid_z_finite, unbiased=(valid_z_finite.numel() > 1))
                     feat_seq_atk[b, t, 2] = mean_valid
                     feat_seq_atk[b, t, 3] = std_valid
                else: # All valid measurements were NaN?
                     feat_seq_atk[b, t, 2] = 0.0 # Or NaN? Let's use 0 for simplicity
                     feat_seq_atk[b, t, 3] = 0.0
            else: # No valid measurements in this step
                feat_seq_atk[b, t, 2] = 0.0
                feat_seq_atk[b, t, 3] = 0.0

    # Handle potential NaNs/Infs in calculated features (e.g., std=0 leading to Inf later?)
    feat_seq_atk = torch.nan_to_num(feat_seq_atk, nan=0.0, posinf=0.0, neginf=0.0)

    return feat_seq_atk
# <<< END NEW >>>


# ===================================================================
# 主评估函数
# ===================================================================
def evaluate_models(loader, device, models, attack_fn, attack_level):
    # ... (函数开头和 WLS/残差计算逻辑不变) ...
    global Nbus, _ybus, _baseMVA, _slack_pos
    ybus_np = _ybus; baseMVA = _baseMVA; slack_pos_idx = _slack_pos
    results = {name: {'th_rmse': [], 'vm_rmse': []} for name in models.keys()}
    for name in models.keys(): results[name] = {'th_rmse': [], 'vm_rmse': []} # Ensure init
    for model in models.values():
        if isinstance(model, nn.Module): model.eval()

    pbar = tqdm(loader, desc=f"Attack Lvl {attack_level:.2f}")
    for batch_idx, batch in enumerate(pbar):
        # 1. --- 数据准备 ---
        try:
            z_seq      = batch["z"].to(device)
            R_seq      = batch["R"].to(device)
            feat_seq   = batch["feat"].to(device) # Original features
            x_gt_seq   = batch["x"].to(device)
            ztype_np   = batch["ztype"].numpy()
            B, W, _S = x_gt_seq.shape
            if Nbus is None: Nbus = _S // 2
            if ztype_np.ndim == 3 and ztype_np.shape[1] == 4:
                 ztype_np = np.repeat(ztype_np[:, np.newaxis, :, :], W, axis=1)
            elif ztype_np.ndim != 4: continue
            meas_mask  = (R_seq < 1e8).bool()
        except Exception as e: print(f"错误: Batch {batch_idx} 数据准备失败: {e}，跳过。"); continue

        # 2. --- 应用攻击 ---
        try:
            z_atk, R_atk, mask_atk = attack_fn(z_seq, R_seq, meas_mask, attack_level)
        except Exception as e: print(f"错误: Batch {batch_idx} 应用攻击失败: {e}，跳过。"); continue

        # 3. --- 重新计算特征 ---
        try:
            feat_seq_atk = recalculate_features_torch(z_atk, mask_atk)
        except Exception as e: print(f"错误: Batch {batch_idx} 重新计算特征失败: {e}，跳过。"); continue

        # 4. --- 模型评估 ---
        x_wls_atk_seq = None
        r_atk_seq = None

        # --- WLS 基线 & 输入准备 ---
        if "WLS" in models or "refine-wls" in models or "gru_baseline" in models:
            # ...(WLS 计算逻辑不变)...
            x_wls_atk_batch = []
            z_atk_np = z_atk.cpu().numpy(); R_atk_np = R_atk.cpu().numpy()
            for b in range(B):
                x_wls_atk_window = []
                for t in range(W):
                     z_k = z_atk_np[b, t]; R_k_diag = R_atk_np[b, t]
                     if ztype_np.shape[2:] == (4, R_k_diag.shape[0]):
                         ztype_k = ztype_np[b, t]
                         x_wls_k = get_wls_estimate(z_k, R_k_diag, ztype_k)
                     else: x_wls_k = np.full(2 * Nbus, np.nan, dtype=np.float32)
                     x_wls_atk_window.append(x_wls_k)
                x_wls_atk_batch.append(np.stack(x_wls_atk_window, axis=0))
            x_wls_atk_seq_np = np.stack(x_wls_atk_batch, axis=0)
            wls_nan_mask_np = np.isnan(x_wls_atk_seq_np)
            x_wls_atk_seq_np[wls_nan_mask_np] = 0.0
            x_wls_atk_seq = torch.from_numpy(x_wls_atk_seq_np).float().to(device)
            r_atk_seq = compute_whitened_residual_torch(z_atk, x_wls_atk_seq, R_atk, ztype_np, baseMVA, slack_pos_idx, ybus_np, device)
            if "WLS" in models:
                gt_state_order = 'va_vm'; wls_state_order = 'vm_va'
                th, vm = rmse_metrics(x_wls_atk_seq, x_gt_seq, state_order=wls_state_order)
                if not np.all(wls_nan_mask_np):
                    results['WLS']['th_rmse'].append(th); results['WLS']['vm_rmse'].append(vm)

        # --- Evaluate DL Models ---
        with torch.no_grad():
            # --- tagru (输入 z_atk, feat_seq_atk) ---
            if "tagru" in models and isinstance(models['tagru'], nn.Module):
                try:
                    x_hat_tagru = models['tagru'](z_atk, feat_seq_atk)
                    th, vm = rmse_metrics(x_hat_tagru, x_gt_seq, state_order='vm_va') # TAGRU outputs vm_va
                    results['tagru']['th_rmse'].append(th); results['tagru']['vm_rmse'].append(vm)
                except Exception as e: print(f"错误: 评估 tagru 失败 (Batch {batch_idx}): {e}")

            # --- refine-wls (输入 r_atk, x_wls_atk, feat_seq_atk) ---
            if "refine-wls" in models and isinstance(models['refine-wls'], nn.Module):
                if r_atk_seq is not None and x_wls_atk_seq is not None:
                    try:
                        A_time = batch.get("A_time", None); E_time = batch.get("E_time", None)
                        if A_time is not None: A_time = A_time.to(device)
                        if E_time is not None: E_time = E_time.to(device)
                        x_hat_refine = models['refine-wls'](r_atk_seq, x_wls_atk_seq, feat_seq_atk, A_time=A_time, E_time=E_time)
                        th, vm = rmse_metrics(x_hat_refine, x_gt_seq, state_order='va_vm') # Refine outputs va_vm
                        results['refine-wls']['th_rmse'].append(th); results['refine-wls']['vm_rmse'].append(vm)
                    except Exception as e: print(f"错误: 评估 refine-wls 失败 (Batch {batch_idx}): {e}")

            # --- gru_baseline (输入 r_atk, x_wls_atk, feat_seq_atk) ---
            if "gru_baseline" in models and isinstance(models['gru_baseline'], nn.Module):
                 if r_atk_seq is not None and x_wls_atk_seq is not None:
                     try:
                         x_hat_gru = models['gru_baseline'](r_atk_seq, feat_seq_atk, x_wls_atk_seq)
                         th, vm = rmse_metrics(x_hat_gru, x_gt_seq, state_order='va_vm') # Assume va_vm
                         results['gru_baseline']['th_rmse'].append(th); results['gru_baseline']['vm_rmse'].append(vm)
                     except Exception as e: print(f"错误: 评估 gru_baseline 失败 (Batch {batch_idx}): {e}")

            # --- PGR (输入 z_atk, feat_seq_atk) ---
            if "pgr" in models and isinstance(models['pgr'], nn.Module):
                try:
                    _, state_final = models['pgr'](z_atk, feat_seq_atk)
                    th, vm = rmse_metrics(state_final, x_gt_seq, state_order='vm_va') # PGR outputs vm_va
                    results['pgr']['th_rmse'].append(th); results['pgr']['vm_rmse'].append(vm)
                except Exception as e: print(f"错误: 评估 pgr 失败 (Batch {batch_idx}): {e}")

    # --- Final Results Aggregation ---
    final_results = {}
    for name in models.keys():
        final_results[name] = {'th_rmse': np.nan, 'vm_rmse': np.nan}
        if name in results and results[name]['th_rmse']:
            final_results[name]['th_rmse'] = np.nanmean(results[name]['th_rmse'])
            final_results[name]['vm_rmse'] = np.nanmean(results[name]['vm_rmse'])
    return final_results


# Function to load checkpoint (incorporating weights_only=False and PGR remapping)
def load_model_from_ckpt(ckpt_path, model_class, config_keys, base_cfg, device, is_pgr=False):
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"警告: Checkpoint 未提供或未找到: {ckpt_path}")
        return None
    try:
        # <<< MODIFIED: Ensure weights_only=False >>>
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        # <<< END MODIFIED >>>

        loaded_args = checkpoint.get('args', argparse.Namespace())
        model_cfg = {}
        for key in config_keys:
             default_val = base_cfg.get(key)
             # Handle potential naming differences more robustly
             possible_arg_names = [key, key.replace('_dim', ''), key.replace('_layers', ''), key.replace('_hidden', '')]
             found_arg = False
             for arg_name in possible_arg_names:
                 if hasattr(loaded_args, arg_name):
                     model_cfg[key] = getattr(loaded_args, arg_name)
                     found_arg = True
                     break
             if not found_arg and default_val is not None:
                 model_cfg[key] = default_val
                 # print(f"Info: Using default for '{key}' ({default_val})")
             elif not found_arg:
                 print(f"警告: 无法确定参数 '{key}' 的值。")

        # Ensure essential dims are present
        if 'meas_dim' not in model_cfg: model_cfg['meas_dim'] = base_cfg['M']
        if 'feat_dim' not in model_cfg: model_cfg['feat_dim'] = base_cfg['F']
        if 'state_dim' not in model_cfg: model_cfg['state_dim'] = base_cfg['S']


        if is_pgr:
            model = model_class(cfg_base=model_cfg, cfg_refiner=model_cfg).to(device)
        else:
            # Filter config keys to only those accepted by the model's __init__
            import inspect
            sig = inspect.signature(model_class.__init__)
            accepted_keys = {p for p in sig.parameters if p != 'self'}
            filtered_cfg = {k: v for k, v in model_cfg.items() if k in accepted_keys}
            model = model_class(**filtered_cfg).to(device)


        state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
        new_state_dict = {}
        key_remapped = False
        for k, v in state_dict.items():
            new_k = k.replace('module.', '')
            original_new_k = new_k
            if is_pgr: # Apply PGR specific remapping
                # ...(省略了 PGR remapping 逻辑，与上一版相同)...
                 if new_k.startswith('base_estimator.fusion_align_g.'): new_k = new_k.replace('base_estimator.fusion_align_g.', 'base_estimator.fusion_module.align_g.')
                 elif new_k.startswith('base_estimator.fusion_align_s.'): new_k = new_k.replace('base_estimator.fusion_align_s.', 'base_estimator.fusion_module.align_s.')
                 elif new_k.startswith('base_estimator.fusion_gate_alpha.'): new_k = new_k.replace('base_estimator.fusion_gate_alpha.', 'base_estimator.fusion_module.gate_alpha.')
                 elif new_k.startswith('refiner_net.fusion_align_g.'): new_k = new_k.replace('refiner_net.fusion_align_g.', 'refiner_net.fusion_module.align_g.')
                 elif new_k.startswith('refiner_net.fusion_align_s.'): new_k = new_k.replace('refiner_net.fusion_align_s.', 'refiner_net.fusion_module.align_s.')
                 elif new_k.startswith('refiner_net.fusion_gate_alpha.'): new_k = new_k.replace('refiner_net.fusion_gate_alpha.', 'refiner_net.fusion_module.gate_alpha.')
                 if new_k != original_new_k: key_remapped = True
            new_state_dict[new_k] = v

        if key_remapped: print("应用了 PGR state dict key remapping。")

        try:
            model.load_state_dict(new_state_dict, strict=True)
            print(f"成功加载模型 (strict): {ckpt_path}")
        except RuntimeError as load_error:
             print(f"警告: Strict loading 失败 for {ckpt_path}. Trying non-strict. Error: {load_error}")
             model.load_state_dict(new_state_dict, strict=False)
             print(f"成功加载模型 (non-strict): {ckpt_path}")

        model.eval()
        return model

    except FileNotFoundError:
        print(f"错误: Checkpoint 文件未找到: {ckpt_path}")
        return None
    except Exception as e:
        print(f"加载模型失败 {ckpt_path}: {e}")
        return None


def main():
    ap = argparse.ArgumentParser(description="鲁棒性评估脚本 (量测缺失 & 坏数据)")
    # ...(省略了 argparse 定义，与上一版相同)...
    ap.add_argument("--data_dir", type=str, default="data/windows_ieee33", help="数据目录")
    ap.add_argument("--tag", type=str, default="W24", help="数据标签")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ckpt_tagru", type=str, default="", help="TAGRU checkpoint")
    ap.add_argument("--ckpt_refine", type=str, default="", help="Refine-WLS checkpoint")
    ap.add_argument("--ckpt_gru", type=str, default="", help="GRU Baseline checkpoint")
    ap.add_argument("--ckpt_pgr", type=str, default="", help="PGR checkpoint")
    ap.add_argument("--missing_levels", type=float, nargs='+', default=[0.1, 0.3, 0.5], help="缺失率")
    ap.add_argument("--baddata_levels", type=float, nargs='+', default=[0.05, 0.1, 0.15], help="坏数据率")
    ap.add_argument("--output_csv", type=str, default=None, help="输出 CSV 文件路径")

    args = ap.parse_args()
    device = torch.device(args.device)

    # --- 1. 初始化物理参数和数据加载器 ---
    init_phys_meta(args.data_dir)
    p_test = os.path.join(args.data_dir, f"{args.tag}_test.npz")
    test_loader, dims, meta = make_loader(p_test, args.batch_size, shuffle=False)
    global Nbus
    Nbus = dims['Nbus']

    # --- 2. 加载模型 ---
    models = {}
    models['WLS'] = "algorithm"

    # Base config for loading, primarily for dims
    base_cfg = {
        'M': dims['M'], 'F': dims['F'], 'S': dims['S'],
        'meas_dim': dims['M'], 'feat_dim': dims['F'], 'state_dim': dims['S'],
        # Add other common defaults that might be missing in older checkpoints' args
        'hidden_dim': 256, 'num_layers': 2, 'dropout': 0.1,
        'gnn_hidden': 128, 'gnn_layers': 2, 'gnn_dropout': 0.0,
        'nhead': 4, 'bias_scale': 3.0, 'use_mask': False,
        'use_bus_smooth': True, 'use_gnn': False, 'gnn_type': 'tag'
    }

    # Define keys needed for each model's config reconstruction
    tagru_config_keys = ['meas_dim', 'feat_dim', 'state_dim', 'hidden_dim', 'num_layers', 'dropout', 'gnn_hidden', 'gnn_layers', 'gnn_dropout', 'nhead', 'bias_scale', 'use_mask']
    pgr_config_keys = tagru_config_keys # Assume PGR uses same keys
    # RefineSeq/GRU needs meas_dim based on residual 'r', assume same as 'M' for eval robustness where we compute 'r'
    refine_config_keys = ['meas_dim', 'state_dim', 'feat_dim', 'hidden_dim', 'num_layers', 'dropout', 'nhead', 'bias_scale', 'use_mask', 'use_bus_smooth', 'use_gnn', 'gnn_type', 'gnn_hidden', 'gnn_layers', 'gnn_dropout']
    gru_config_keys = ['meas_dim', 'state_dim', 'feat_dim', 'hidden_dim', 'num_layers', 'dropout']


    model_tagru = load_model_from_ckpt(args.ckpt_tagru, TopoAlignGRU, tagru_config_keys, base_cfg, device)
    if model_tagru: models['tagru'] = model_tagru

    model_pgr = load_model_from_ckpt(args.ckpt_pgr, PGR, pgr_config_keys, base_cfg, device, is_pgr=True)
    if model_pgr: models['pgr'] = model_pgr

    model_refine = load_model_from_ckpt(args.ckpt_refine, RefineSeqTAModel, refine_config_keys, base_cfg, device)
    if model_refine: models['refine-wls'] = model_refine

    model_gru = load_model_from_ckpt(args.ckpt_gru, GRUBaseline, gru_config_keys, base_cfg, device)
    if model_gru: models['gru_baseline'] = model_gru

    # --- 3. 定义评估场景 ---
    # ...(省略了场景定义逻辑，与上一版相同)...
    scenarios = {}
    scenarios["Clean"] = (lambda z, R, m, lvl=0.0: (z, R, m), 0.0)
    for level in args.missing_levels:
        if level >= 0: # Include 0 level if present
            scenarios[f"Missing {int(level*100)}%"] = (lambda z, R, m, lvl=level: add_missing_measurements(z, R, m, missing_rate=lvl), level)
    for level in args.baddata_levels:
        if level >= 0: # Include 0 level if present
            noise_sigma = 10.0
            scenarios[f"Bad Data {int(level*100)}% ({noise_sigma:.0f}σ)"] = (
                lambda z, R, m, current_level=level, sigma=noise_sigma: add_bad_data(z, R, m, bad_data_rate=current_level, noise_level=sigma),
                level
            )

    all_results = []

    # --- 4. 运行评估 ---
    # ...(省略了评估循环逻辑，与上一版相同, 确保调用了修复后的 evaluate_models)...
    for name, (attack_fn, level) in scenarios.items():
        print(f"\n--- 运行场景: {name} (Level: {level}) ---")
        active_models = {k: v for k, v in models.items() if v is not None}
        if not active_models or (len(active_models) == 1 and 'WLS' in active_models):
             print("没有成功加载任何 DL 模型，跳过场景评估。")
             continue
        results = evaluate_models(test_loader, device, active_models, attack_fn, level)
        for model_name, metrics in results.items():
            all_results.append({
                'Scenario': name,
                'Attack Level': f"{level:.2f}",
                'Model': model_name,
                'VM_RMSE (pu)': metrics['vm_rmse'],
                'VA_RMSE (deg)': metrics['th_rmse'],
            })


    # --- 5. 打印最终结果 ---
    # ...(省略了打印逻辑，与上一版相同)...
    print("\n\n--- 最终鲁棒性评估结果 ---")
    if not all_results: print("没有可显示的结果。"); return
    df = pd.DataFrame(all_results)
    df = df.sort_values(by=['Scenario', 'Attack Level', 'Model']) # Sort Attack Level numerically might be better
    df['Attack Level'] = pd.to_numeric(df['Attack Level'])
    df = df.sort_values(by=['Attack Level', 'Scenario', 'Model'])

    try:
        df_pivot_vm = df.pivot_table(index=['Scenario', 'Attack Level'], columns='Model', values='VM_RMSE (pu)')
        print("--- VM_RMSE (pu) ---")
        print(df_pivot_vm.to_markdown(floatfmt=".5f"))
        df_pivot_va = df.pivot_table(index=['Scenario', 'Attack Level'], columns='Model', values='VA_RMSE (deg)')
        print("\n--- VA_RMSE (deg) ---")
        print(df_pivot_va.to_markdown(floatfmt=".3f"))
    except Exception as e:
        print(f"生成 pivot 表格失败: {e}"); print("原始数据:"); print(df.to_markdown(index=False, floatfmt=".5f"))


    # --- 6. 保存结果到 CSV ---
    # ...(省略了保存逻辑，与上一版相同)...
    if args.output_csv:
        try:
            output_dir = os.path.dirname(args.output_csv)
            if output_dir: os.makedirs(output_dir, exist_ok=True)
            df.to_csv(args.output_csv, index=False, float_format='%.6f')
            print(f"\n结果已保存到: {args.output_csv}")
        except Exception as e: print(f"错误: 保存 CSV 文件失败: {e}")


if __name__ == "__main__":
    main()