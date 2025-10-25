import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os
import wandb

def load_data_from_csv(csv_path):
    """Loads sweep results from a CSV file."""
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded data from {csv_path}, shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

def load_data_from_wandb(wandb_path, target_metric):
    """Loads sweep results from WandB API."""
    try:
        api = wandb.Api()
        # wandb_path should be "entity/project/sweep_id" or "entity/project"
        runs = api.runs(path=wandb_path)
        print(f"Found {len(runs)} runs in WandB path: {wandb_path}")

        data = []
        param_keys = set()
        for run in runs:
            # Only include completed runs with the target metric
            if run.state == "finished" and target_metric in run.summary:
                run_data = {k: v for k, v in run.config.items() if not k.startswith('_')}
                run_data[target_metric] = run.summary[target_metric]
                run_data['run_id'] = run.id
                run_data['run_name'] = run.name
                data.append(run_data)
                param_keys.update(run.config.keys())

        if not data:
            print("No finished runs with the target metric found.")
            return None

        df = pd.DataFrame(data)
        print(f"Loaded data for {len(df)} runs from WandB.")
        return df

    except Exception as e:
        print(f"Error fetching data from WandB: {e}")
        return None


def preprocess_data(df, target_metric):
    """Prepares data for SHAP analysis (handling NaNs, encoding)."""
    # Drop rows where target is NaN
    df = df.dropna(subset=[target_metric])
    if df.empty:
        print("Error: No valid data remaining after dropping NaNs in target metric.")
        return None, None

    # Identify potential feature columns (exclude IDs, names, target)
    exclude_cols = [target_metric, 'run_id', 'run_name', 'name', 'sweep_id', 'state',
                    # Add other non-hyperparameter columns if they exist
                    'train_duration_s', 'best_val_loss_mse', 'val_loss_mse', # Exclude other metrics
                    'test_rmse_theta_deg', 'test_rmse_vm_pu',
                    'wls_theta_deg', 'wls_|V|'
                   ]
    feature_cols = [col for col in df.columns if col not in exclude_cols and not col.startswith('_')]

    X = df[feature_cols].copy()
    y = df[target_metric]

    print(f"Identified Features: {feature_cols}")
    print(f"Target: {target_metric}")

    # Handle missing values in features (e.g., fill with median or mean)
    for col in X.select_dtypes(include=np.number).columns:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            print(f"Filled NaNs in numeric column '{col}' with median ({median_val}).")

    # Encode categorical features (if any)
    label_encoders = {}
    for col in X.select_dtypes(include='object').columns:
        le = LabelEncoder()
        X[col] = X[col].astype(str).fillna('missing') # Fill NaNs before encoding
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        print(f"Label encoded categorical column '{col}'.")
    # Handle boolean features -> convert to int
    for col in X.select_dtypes(include='bool').columns:
         X[col] = X[col].astype(int)
         print(f"Converted boolean column '{col}' to integer.")


    # Ensure all columns are numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    # Check for NaNs again after potential coercion errors
    if X.isnull().any().any():
         print("Warning: NaNs detected after converting to numeric. Filling with 0.")
         print(X.isnull().sum())
         X = X.fillna(0) # Or use a more sophisticated imputer


    return X, y


def run_shap_analysis(X, y, output_dir, metric_name):
    """Trains a RandomForest and performs SHAP analysis."""
    if X is None or y is None or X.empty or y.empty:
        print("Cannot run SHAP analysis due to invalid input data.")
        return

    # Split data for model training (optional but good practice)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForest model
    print("Training RandomForestRegressor...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print(f"Model trained. R^2 score on test set: {model.score(X_test, y_test):.3f}")

    # Calculate SHAP values
    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    # Use X_test for explaining predictions on unseen data, or X for overall importance
    shap_values = explainer.shap_values(X)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --- Generate Plots ---
    plt.figure()
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title(f'Feature Importance for {metric_name}')
    plt.tight_layout()
    bar_path = os.path.join(output_dir, f"shap_importance_{metric_name}.png")
    plt.savefig(bar_path)
    plt.close()
    print(f"SHAP bar plot saved to: {bar_path}")

    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.title(f'SHAP Summary Plot for {metric_name}')
    plt.tight_layout()
    beeswarm_path = os.path.join(output_dir, f"shap_beeswarm_{metric_name}.png")
    plt.savefig(beeswarm_path)
    plt.close()
    print(f"SHAP beeswarm plot saved to: {beeswarm_path}")


def main():
    parser = argparse.ArgumentParser(description="Perform SHAP sensitivity analysis on sweep results.")
    parser.add_argument("--csv_path", help="Path to the sweep results CSV file.")
    parser.add_argument("--wandb_path", help="WandB path (entity/project[/sweep_id]). Use instead of --csv_path.")
    parser.add_argument("--target_metric", required=True, help="Name of the target metric column (e.g., test_rmse_vm_pu).")
    parser.add_argument("--output_dir", default="figs", help="Directory to save SHAP plots.")

    args = parser.parse_args()

    if args.csv_path and args.wandb_path:
        print("Error: Please provide either --csv_path or --wandb_path, not both.")
        return
    if not args.csv_path and not args.wandb_path:
        print("Error: Please provide either --csv_path or --wandb_path.")
        return

    if args.csv_path:
        df = load_data_from_csv(args.csv_path)
    else:
        df = load_data_from_wandb(args.wandb_path, args.target_metric)

    if df is not None:
        X, y = preprocess_data(df, args.target_metric)
        run_shap_analysis(X, y, args.output_dir, args.target_metric)
    else:
        print("Failed to load data. Exiting.")

if __name__ == "__main__":
    main()