#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
wandb_fetch_best.py
读取指定 entity / project 下的 runs，从给定指标里选“最好”的 run，
导出该 run 的超参数与指标，并打印链接与简表。

依赖：
  pip install wandb pandas

环境变量：
  需要先配置 WANDB_API_KEY（或已经 wandb login）
    export WANDB_API_KEY=xxxx
"""

import argparse
import os
from typing import List, Dict, Any
import pandas as pd
import wandb

def summary_to_dict(run):
    """Robustly convert wandb run.summary to a plain dict."""
    s = run.summary
    # Try common APIs in order
    for getter in (
        lambda x: x.to_dict(),                     # new API
        lambda x: dict(x._json_dict),              # old API internal
        lambda x: dict(x),                         # mapping-like
        lambda x: {k: x[k] for k in x.keys()},     # fallback
    ):
        try:
            return getter(s)
        except Exception:
            pass
    return {}  # last resort


def config_to_dict(run):
    """Robustly convert wandb run.config to a plain dict."""
    c = run.config
    # wandb.Config usually has as_dict()
    for getter in (
        lambda x: x.as_dict(),                     # common API
        lambda x: dict(x),                         # mapping-like
        lambda x: {k: x[k] for k in x.keys()},     # fallback
    ):
        try:
            return getter(c)
        except Exception:
            pass
    # last fallback
    try:
        return dict(c) if isinstance(c, dict) else {}
    except Exception:
        return {}

def pick_metric(summary: dict, metric_priority: List[str]):
    """按优先级在 summary 里挑第一个存在的指标，返回 (metric_name, value) 或 (None, None)"""
    for m in metric_priority:
        if m in summary and summary[m] is not None:
            return m, summary[m]
    return None, None

def flatten_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """拉平 wandb config，只保留一层，并过滤掉 _ 开头的内部字段"""
    flat = {}
    for k, v in cfg.items():
        if isinstance(k, str) and not k.startswith('_'):
            # 保持简单：只取一层（大部分超参都在一层）
            if isinstance(v, (str, int, float, bool)) or v is None:
                flat[k] = v
            else:
                # 如果是 list/dict，尽量转成字符串
                flat[k] = str(v)
    return flat

def fetch_best_runs(entity: str,
                    projects: List[str],
                    metric_priority: List[str],
                    goal: str = "minimize",
                    per_sweep: bool = False) -> pd.DataFrame:
    """
    从多个 project 拉取 runs，按 metric_priority 挑指标，然后按 goal 选最优 run。
    如果 per_sweep=True，则每个 sweep 选一个最佳；否则整个 project 选一个最佳。
    返回汇总 DataFrame。
    """
    api = wandb.Api()
    rows = []

    for project in projects:
        print(f"\n>>> Scanning project: {entity}/{project}")
        runs = api.runs(path=f"{entity}/{project}")
        # 只取已完成，且至少有一个目标指标的 run
        valid = []
        for r in runs:
            if r.state != "finished":
                continue
            summary = summary_to_dict(r)
            metric_name, metric_val = pick_metric(summary, metric_priority)
            if metric_name is None:
                continue
            valid.append((r, metric_name, metric_val))

        if not valid:
            print(f"  (No finished runs with any of metrics: {metric_priority})")
            continue

        # 分组逻辑：按 sweep or 整个 project
        if per_sweep:
            # 以 r.sweep.id 分组；没有 sweep 的 run 归为 "nosweep"
            by_sw = {}
            for r, mname, mval in valid:
                sid = getattr(r, "sweep", None).id if getattr(r, "sweep", None) else "nosweep"
                by_sw.setdefault(sid, []).append((r, mname, mval))

            for sid, items in by_sw.items():
                if goal == "minimize":
                    best = min(items, key=lambda x: x[2])
                else:
                    best = max(items, key=lambda x: x[2])
                r, mname, mval = best
                cfg = flatten_config(config_to_dict(r))
                row = {
                    "entity": entity,
                    "project": project,
                    "sweep_id": sid,
                    "run_id": r.id,
                    "run_name": r.name,
                    "run_url": f"https://wandb.ai/{entity}/{project}/runs/{r.id}",
                    "best_metric": mname,
                    "best_value": mval
                }
                row.update(cfg)
                rows.append(row)
        else:
            # 整个 project 只取一个最佳
            if goal == "minimize":
                best = min(valid, key=lambda x: x[2])
            else:
                best = max(valid, key=lambda x: x[2])
            r, mname, mval = best
            cfg = flatten_config(config_to_dict(r))
            sid = getattr(r, "sweep", None).id if getattr(r, "sweep", None) else "nosweep"
            row = {
                "entity": entity,
                    "project": project,
                    "sweep_id": sid,
                    "run_id": r.id,
                    "run_name": r.name,
                    "run_url": f"https://wandb.ai/{entity}/{project}/runs/{r.id}",
                    "best_metric": mname,
                    "best_value": mval
            }
            row.update(cfg)
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    # 列顺序：把关键列放前面
    df = pd.DataFrame(rows)
    key_cols = ["entity", "project", "sweep_id", "run_id", "run_name", "run_url", "best_metric", "best_value"]
    front = [c for c in key_cols if c in df.columns]
    others = [c for c in df.columns if c not in front]
    df = df[front + others]
    return df

def main():
    parser = argparse.ArgumentParser(description="Fetch best hyperparameters from W&B projects/sweeps.")
    parser.add_argument("--entity", required=True, help="W&B entity, e.g., a530029885-southeast-university")
    parser.add_argument("--projects", nargs="+", required=True,
                        help="One or more W&B projects, e.g., GridSE_Pgr GridSE_Tagru_sup GridSE_Refine_seq")
    parser.add_argument("--metric_priority", default="val_loss_mse,val_loss_phys,val_rmse_vm_pu,val_rmse_theta_deg",
                        help="Comma-separated metric names, pick the first existing one in run.summary")
    parser.add_argument("--goal", choices=["minimize", "maximize"], default="minimize",
                        help="Whether to minimize or maximize the chosen metric")
    parser.add_argument("--per_sweep", action="store_true",
                        help="If set, pick best run per sweep inside each project; otherwise pick only one per project")
    parser.add_argument("--out_csv", default="results/wandb_best_runs.csv", help="Output CSV path")
    parser.add_argument("--out_md", default="results/wandb_best_runs.md", help="Output Markdown path")

    args = parser.parse_args()
    metric_priority = [m.strip() for m in args.metric_priority.split(",") if m.strip()]

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    df = fetch_best_runs(
        entity=args.entity,
        projects=args.projects,
        metric_priority=metric_priority,
        goal=args.goal,
        per_sweep=args.per_sweep
    )

    if df.empty:
        print("\nNo results found. Check your entity/project names and metric names.")
        return

    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved CSV to: {args.out_csv}")

    try:
        md = df.to_markdown(index=False)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Saved Markdown to: {args.out_md}")
    except Exception as e:
        print(f"Markdown export failed: {e}")

    # 友好打印 Top-N
    print("\n=== Best runs (top 10 rows) ===")
    print(df.head(10).to_string(index=False))

    # 打印每个 project 的最佳 run 链接
    print("\n=== Best run URLs by project ===")
    for proj in df["project"].unique():
        sub = df[df["project"] == proj].sort_values("best_value", ascending=(args.goal=="minimize"))
        best_row = sub.iloc[0]
        print(f"- {proj}: {best_row['run_name']}  ->  {best_row['run_url']}  [{best_row['best_metric']}={best_row['best_value']}]")

if __name__ == "__main__":
    main()
