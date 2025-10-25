# 实验计划：GridSE 系列模型的参数搜索与敏感性分析

## 一、研究目标

本实验旨在系统评估 **PGR 模型** 及其对照基线模型（TopoAlignGRU、RefineSeq、GRUBaseline）在 **IEEE-33 母线系统** 上的性能表现与超参数敏感性。  
通过对 **W24** 和 **W96** 两类时序数据集的实验，研究模型在以下方面的综合性能：

- 超参数变化对精度和收敛速度的影响；
- 模型在 **量测缺失（Missing Measurements）** 与 **坏数据（Bad Data）** 情境下的鲁棒性（Robustness）。

---

## 二、模型与训练脚本

| 模型名称 | 模型说明 | 主模型文件 | 训练脚本文件 |
|-----------|-----------|-------------|----------------|
| **PGR（Hybrid）** | 混合式模型，融合物理约束与数据驱动估计 | `models/pgr.py` | `train/train_pgr_hybrid.py` |
| **TopoAlignGRU（Supervised）** | 全监督版本（以状态误差为主损失） | `models/tagru.py` | `train/train_tagru_supervised.py` |
| **TopoAlignGRU（Weakly Supervised）** | 弱监督版本（物理一致性约束） | `models/tagru.py` | `train/train_tagru_final.py` |
| **RefineSeqTAModel** | 序列细化模型，融合注意力机制 | `models/refine_seq.py` | `train/train_refine_baseline.py` |
| **GRUBaseline** | 标准 GRU 基线模型 | `models/baselines.py` | `train/train_baselines.py` |
| **Evaluation** | 鲁棒性评估脚本 | —— | `eval/eval_robustness.py` |

---

## 三、参数搜索方案

### 3.1 策略说明

采用 **WandB 的贝叶斯优化（Bayesian Optimization）** 进行自动化超参数搜索。  
若 WandB 不可用，则使用 **随机搜索（Random Search）**，搜索规模建议为 50–100 组参数。

### 3.2 实现与工具

- 搜索配置文件：`sweeps/*.yaml`
- 执行脚本：`scripts/run_wandb_sweep.sh`
- 配置管理：使用 YAML 合并方式（`defaults.yaml` + `model_sweep.yaml`）

### 3.3 搜索参数空间

**通用超参数**：  
`lr`, `hidden_dim`, `num_layers`, `gnn_hidden`, `gnn_layers`,  
`dropout`, `weight_decay`, `batch_size`, `tag`（首选 `W24`）

**PGR / 弱监督模型专属参数**：  
`alpha`, `beta`, `lambda_op`

---

### 3.4 评估指标与优化目标

- **主优化目标：**
  - 全监督 / 混合模型：`val_loss_mse`
  - 弱监督模型：`val_loss_phys`
- **同时跟踪的辅助指标：**  
  `train_loss_*`, `val_loss_*`, `val_rmse_theta_deg`, `val_rmse_vm_pu`, `learning_rate`

---

### 3.5 实验预算与流程

- 每个模型在 **W24 数据集** 上进行约 **30–50 次实验**。
- 每次实验配置由 WandB 自动生成或 YAML 指定。

#### 实施步骤：
1. 为每个模型准备对应的 Sweep 配置文件，例如：
   - `sweeps/pgr_sweep.yaml`
   - `sweeps/tagru_supervised_sweep.yaml`
   - `sweeps/refineseq_sweep.yaml`
2. 启动 Sweep：
   ```bash
   bash scripts/run_wandb_sweep.sh --config sweeps/pgr_sweep.yaml --entity <你的WandB账号> --count 40
   ```
3. 在 WandB 平台上实时监控训练与验证曲线。
4. 以验证集主指标为准，选取性能最优的实验并下载其 `.pt` 检查点模型。

---

## 四、最终评估与重复实验

### 4.1 随机种子设定

为检验模型稳定性与随机性敏感度，设定三组独立随机种子：
```
{42, 2025, 3407}
```

### 4.2 复现实验流程

以各模型在 Step 3 中得到的最优超参数为基准，重新训练 3 次：

```bash
# 示例：PGR 模型
python train/train_pgr_hybrid.py --tag W24 --lr <best_lr> --hidden_dim <best_hd> ... --seed 42 --save_dir checkpoints/seed42
python train/train_pgr_hybrid.py --tag W24 --lr <best_lr> --hidden_dim <best_hd> ... --seed 2025 --save_dir checkpoints/seed2025
python train/train_pgr_hybrid.py --tag W24 --lr <best_lr> --hidden_dim <best_hd> ... --seed 3407 --save_dir checkpoints/seed3407
```

### 4.3 测试与结果统计

1. 使用相同模型在测试集（`W24_test.npz` 或 `W96_test.npz`）上进行评估；
2. 对三个种子的结果取平均与标准差，指标包括：  
   - `test_rmse_theta_deg`（相角 RMSE）  
   - `test_rmse_vm_pu`（电压幅值 RMSE）  
3. 若结果记录在 CSV 或 WandB，可使用脚本：  
   ```
   scripts/aggregate_results.py
   ```
   自动汇总平均值与方差。

---

## 五、敏感性分析（Sensitivity Analysis）

### 5.1 方法简介

采用 **SHAP（Shapley Additive Explanations）** 方法进行参数敏感性分析，利用随机森林回归器作为代理模型（Surrogate Model），以解释各超参数对模型性能的相对重要性。

### 5.2 实现工具

- 分析脚本：`analysis/sensitivity_shap.py`
- 数据输入：sweep 结果导出的 CSV 文件，例如：  
  ```
  results/pgr_sweep_W24.csv
  ```

### 5.3 实验流程

1. 确保 sweep 结果已导出为 CSV 或从 WandB 下载；
2. 运行 SHAP 分析脚本：  
   ```bash
   python analysis/sensitivity_shap.py --csv_path results/pgr_sweep_W24.csv --target_metric test_rmse_vm_pu --output_dir figs
   python analysis/sensitivity_shap.py --csv_path results/pgr_sweep_W24.csv --target_metric test_rmse_theta_deg --output_dir figs
   ```
3. 生成各参数的重要性可视化图，分析最敏感的超参数。

---

## 六、总结

该实验计划定义了 **GridSE 系列模型** 的标准化研究流程，覆盖：

1. **超参数搜索** → 自动优化性能；
2. **多随机种子重复实验** → 验证稳定性；
3. **敏感性分析** → 解释模型鲁棒性与泛化能力。

通过该流程，可系统比较各类模型在不同数据集、不同约束条件下的表现，为后续的模型改进与论文撰写提供可靠的实验依据。
