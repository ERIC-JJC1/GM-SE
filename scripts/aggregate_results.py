import argparse
import pandas as pd
import numpy as np
import os

def aggregate_runs(csv_path, group_by_cols, metric_cols, seeds=[42, 2025, 3407]):
    """
    Aggregates results from multiple runs (different seeds) in a CSV file.

    Args:
        csv_path (str): Path to the CSV file containing results from grid search or multiple runs.
        group_by_cols (list): List of column names (hyperparameters) to group by.
        metric_cols (list): List of column names (metrics) to aggregate (mean ± std).
        seeds (list): List of seed values used in the runs.

    Returns:
        pandas.DataFrame: DataFrame with aggregated results (mean and std for metrics).
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

    # Ensure 'seed' column exists if provided, otherwise assume rows correspond to seeds
    if 'seed' not in df.columns:
         print("Warning: 'seed' column not found. Assuming rows for the same config are grouped and correspond to seeds.")
         # Add a pseudo seed column based on grouping if possible (requires careful thought on structure)
         # Simplified: Filter or group assuming consecutive rows are seeds - This is brittle!
         # A better approach: The grid search script should *always* record the seed.
         # For now, let's try grouping and hoping for the best.
         pass # Proceed with caution


    # Define aggregation functions
    agg_funcs = {}
    new_col_names_map = {}
    for col in metric_cols:
        mean_col = f"{col}_mean"
        std_col = f"{col}_std"
        agg_funcs[col] = ['mean', 'std']
        new_col_names_map[(col, 'mean')] = mean_col
        new_col_names_map[(col, 'std')] = std_col

    try:
        # Group by hyperparameters and aggregate metrics
        aggregated_df = df.groupby(group_by_cols).agg(agg_funcs)

        # Flatten MultiIndex columns and rename
        aggregated_df.columns = aggregated_df.columns.to_flat_index()
        aggregated_df = aggregated_df.rename(columns=new_col_names_map)

        # Keep only the mean and std columns, plus the group_by columns (which are in the index)
        cols_to_keep = [f"{col}_mean" for col in metric_cols] + [f"{col}_std" for col in metric_cols]
        aggregated_df = aggregated_df[cols_to_keep]

        # Reset index to make group_by_cols regular columns
        aggregated_df = aggregated_df.reset_index()

        # Format metrics nicely (e.g., "mean ± std") in new columns
        for col in metric_cols:
             mean_col = f"{col}_mean"
             std_col = f"{col}_std"
             # Handle potential NaN std deviations if only one run per group
             aggregated_df[f"{col}_agg"] = aggregated_df.apply(
                 lambda row: f"{row[mean_col]:.4f} ± {row[std_col]:.4f}" if pd.notna(row[std_col]) and row[std_col] != 0 else f"{row[mean_col]:.4f}",
                 axis=1
             )


        return aggregated_df

    except KeyError as e:
         print(f"Error during aggregation: Column {e} not found. Check group_by_cols and metric_cols.")
         print("Available columns:", df.columns.tolist())
         return None
    except Exception as e:
        print(f"An unexpected error occurred during aggregation: {e}")
        return None


def save_tables(df, output_dir, filename_base, main_metric):
    """Saves the aggregated DataFrame in Markdown and LaTeX formats."""
    if df is None:
        return

    os.makedirs(output_dir, exist_ok=True)
    md_path = os.path.join(output_dir, f"{filename_base}.md")
    tex_path = os.path.join(output_dir, f"{filename_base}.tex")

    # Select and reorder columns for presentation
    # Keep grouping columns + aggregated metric columns
    display_cols = df.columns.tolist() # Adjust this list as needed for the final table

    # Sort by the main metric for better readability
    try:
         df_sorted = df.sort_values(by=f"{main_metric}_mean", ascending=True)
    except KeyError:
         print(f"Warning: Main metric '{main_metric}_mean' not found for sorting.")
         df_sorted = df


    # Save Markdown
    try:
        md_table = df_sorted[display_cols].to_markdown(index=False, floatfmt=".4f")
        with open(md_path, 'w') as f:
            f.write(md_table)
        print(f"Markdown table saved to: {md_path}")
    except Exception as e:
        print(f"Error saving Markdown table: {e}")

    # Save LaTeX
    try:
        # Use a subset of columns for LaTeX if too wide
        latex_cols = display_cols # Select relevant columns
        tex_table = df_sorted[latex_cols].to_latex(index=False, float_format="%.4f", escape=False)
        # Add booktabs formatting
        tex_table = tex_table.replace('\\toprule', '\\toprule\n')
        tex_table = tex_table.replace('\\midrule', '\n\\midrule\n')
        tex_table = tex_table.replace('\\bottomrule', '\n\\bottomrule')
        with open(tex_path, 'w') as f:
            f.write("\\begin{table}[htbp]\n\\centering\n")
            f.write("\\caption{Aggregated Experiment Results (Mean ± Std)}\n")
            f.write("\\label{tab:aggregated_results}\n")
            f.write(tex_table)
            f.write("\\end{table}\n")
        print(f"LaTeX table saved to: {tex_path}")
    except Exception as e:
        print(f"Error saving LaTeX table: {e}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate results from multiple runs (seeds).")
    parser.add_argument("--csv_path", required=True, help="Path to the input CSV file with raw run results.")
    parser.add_argument("--group_by", nargs='+', required=True, help="List of hyperparameter column names to group by.")
    parser.add_argument("--metrics", nargs='+', default=['test_rmse_theta_deg', 'test_rmse_vm_pu', 'best_val_loss_mse'], help="List of metric column names to aggregate.")
    parser.add_argument("--main_metric", default='test_rmse_vm_pu', help="Metric column to sort the final table by (mean value).")
    parser.add_argument("--output_dir", default="tables", help="Directory to save the output Markdown and LaTeX tables.")
    parser.add_argument("--filename_base", default="aggregated_results", help="Base name for the output table files (without extension).")

    args = parser.parse_args()

    aggregated_df = aggregate_runs(args.csv_path, args.group_by, args.metrics)

    if aggregated_df is not None:
        print("\nAggregated Results (Top 5):")
        print(aggregated_df.head().to_markdown(index=False))

        save_tables(aggregated_df, args.output_dir, args.filename_base, args.main_metric)

if __name__ == "__main__":
    main()