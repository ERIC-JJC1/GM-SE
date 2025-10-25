import argparse
import itertools
import subprocess
import json
import os
import csv
import time

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def generate_param_combinations(grid):
    keys = grid.keys()
    values = grid.values()
    for instance in itertools.product(*values):
        yield dict(zip(keys, instance))

def run_single_trial(script_path, params, default_args=None):
    command = ['python', script_path]
    current_params = default_args.copy() if default_args else {}
    current_params.update(params)

    for key, value in current_params.items():
        # Handle boolean flags
        if isinstance(value, bool):
            if value:
                command.append(f"--{key}")
            # else: # Assumes absence of flag means False
            #     command.append(f"--no_{key}") # Or adjust based on script's argparse
        elif isinstance(value, list):
             # Handle list arguments if necessary (e.g., --layers 2 3)
             command.append(f"--{key}")
             command.extend(map(str, value))
        else:
            command.append(f"--{key}")
            command.append(str(value))

    print(f"Running command: {' '.join(command)}")
    start_time = time.time()
    result = subprocess.run(command, capture_output=True, text=True)
    end_time = time.time()
    duration = end_time - start_time

    print(f"Finished in {duration:.2f} seconds.")
    if result.returncode != 0:
        print("Error occurred:")
        print(result.stderr)
        return None, duration

    # --- Parse metrics from stdout (NEEDS CUSTOMIZATION) ---
    # This part is highly dependent on the print format of your training script
    # Example: Assume script prints "Val RMSE: θ=X°, V=Y" at the end
    metrics = {'train_duration_s': duration}
    try:
        lines = result.stdout.strip().split('\n')
        last_line = lines[-1] # Or find the line containing metrics
        # Example parsing logic (adjust regex or splitting based on actual output)
        if "EVAL:" in last_line:
             import re
             match_th = re.search(r'θ-RMSE=([\d.]+)', last_line)
             match_vm = re.search(r'\|V\|-RMSE=([\d.]+)', last_line)
             if match_th:
                 metrics['test_rmse_theta_deg'] = float(match_th.group(1))
             if match_vm:
                 metrics['test_rmse_vm_pu'] = float(match_vm.group(1))
        # Add parsing for validation loss if printed
        val_loss_line = next((line for line in reversed(lines) if "Val MSE Loss=" in line), None)
        if val_loss_line:
             match_loss = re.search(r'Val MSE Loss=([\d.e+-]+)', val_loss_line)
             if match_loss:
                 metrics['best_val_loss_mse'] = float(match_loss.group(1))

    except Exception as e:
        print(f"Warning: Could not parse metrics from stdout. {e}")
        print("STDOUT:")
        print(result.stdout[-500:]) # Print last 500 chars for debugging

    return metrics, duration

def main():
    parser = argparse.ArgumentParser(description="Run grid search for a Python script.")
    parser.add_argument("--script", required=True, help="Path to the training script (e.g., train/train_pgr_hybrid.py)")
    parser.add_argument("--param_grid", required=True, help="Path to the JSON/YAML file defining the parameter grid.")
    parser.add_argument("--default_args", help="Path to JSON/YAML file with default arguments.")
    parser.add_argument("--output_dir", default="results", help="Directory to save the results CSV.")
    parser.add_argument("--output_file", help="Specific output CSV file name (overrides default).")

    args = parser.parse_args()

    # Load parameter grid
    try:
        import yaml # Use YAML for better readability
        with open(args.param_grid, 'r') as f:
            param_grid = yaml.safe_load(f)
    except ImportError:
         print("PyYAML not installed, trying JSON.")
         with open(args.param_grid, 'r') as f:
            param_grid = json.load(f)
    except Exception as e:
        print(f"Error loading parameter grid from {args.param_grid}: {e}")
        return

    # Load default arguments
    default_args = {}
    if args.default_args:
        try:
            with open(args.default_args, 'r') as f:
                 if args.default_args.endswith('.yaml') or args.default_args.endswith('.yml'):
                      default_args = yaml.safe_load(f)
                 else:
                      default_args = json.load(f)
        except Exception as e:
             print(f"Warning: Could not load default args from {args.default_args}: {e}")


    os.makedirs(args.output_dir, exist_ok=True)
    if args.output_file:
        output_csv_path = os.path.join(args.output_dir, args.output_file)
    else:
        script_name = os.path.splitext(os.path.basename(args.script))[0]
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_csv_path = os.path.join(args.output_dir, f"{script_name}_grid_search_{timestamp}.csv")

    print(f"Saving results to: {output_csv_path}")

    param_combinations = list(generate_param_combinations(param_grid))
    print(f"Total combinations to run: {len(param_combinations)}")

    results_list = []
    fieldnames = set(param_grid.keys()) | set(default_args.keys())

    try:
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = None # Initialize later when we know all fieldnames

            for i, params in enumerate(param_combinations):
                print(f"\n--- Running trial {i+1}/{len(param_combinations)} ---")
                print(f"Params: {params}")

                metrics, duration = run_single_trial(args.script, params, default_args)

                trial_result = default_args.copy()
                trial_result.update(params)
                if metrics:
                    trial_result.update(metrics)
                    # Dynamically add new metric keys to fieldnames
                    new_fields = set(metrics.keys()) - fieldnames
                    if new_fields:
                        fieldnames.update(new_fields)
                        # Re-initialize writer if header wasn't written or needs update
                        # This simple version assumes header is written first time or file is overwritten
                        if writer is None: # Write header on first successful run
                             sorted_fieldnames = sorted(list(fieldnames))
                             writer = csv.DictWriter(csvfile, fieldnames=sorted_fieldnames)
                             writer.writeheader()


                results_list.append(trial_result)

                # Write incrementally
                if writer:
                    try:
                        # Ensure all fields are present, fill missing with None or ''
                        row_to_write = {f: trial_result.get(f, '') for f in writer.fieldnames}
                        writer.writerow(row_to_write)
                        csvfile.flush() # Ensure data is written to disk
                    except Exception as e:
                         print(f"Error writing row to CSV: {e}")
                elif i == 0 and metrics: # If first run was successful, write header now
                     sorted_fieldnames = sorted(list(fieldnames))
                     writer = csv.DictWriter(csvfile, fieldnames=sorted_fieldnames)
                     writer.writeheader()
                     row_to_write = {f: trial_result.get(f, '') for f in writer.fieldnames}
                     writer.writerow(row_to_write)
                     csvfile.flush()


    except KeyboardInterrupt:
        print("\nGrid search interrupted by user.")
    finally:
        print(f"\nGrid search finished. Results saved to {output_csv_path}")

if __name__ == "__main__":
    main()