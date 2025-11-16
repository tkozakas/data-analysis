import re
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

SCRIPT_DIR = Path(__file__).parent.resolve()
WEKA_JAR_PATH = SCRIPT_DIR / "weka.jar"
CSV_DATA_DIR = SCRIPT_DIR / "data"
ARFF_DATA_DIR = SCRIPT_DIR / "discretization"
RESULTS_DIR = SCRIPT_DIR / "results"
VISUALIZATION_DIR = SCRIPT_DIR / "visualizations"
WEKA_MEMORY_MB = 2048
DATASET_PREFIXES = ["communities_and_crime"]

ATTRIBUTES_TO_SELECT = "15,23,33,40,52,62,71,79,91,100,112,121,last"

DEFAULT_ALGORITHM_CONFIG = {
    "SimpleLinearRegression": {"class": "weka.classifiers.functions.SimpleLinearRegression", "params": ""},
    "LinearRegression": {"class": "weka.classifiers.functions.LinearRegression",
                         "params": "-S 0 -R 1.0E-8 -num-decimal-places 4"}
}


def build_experiment_grid() -> List[Dict[str, Any]]:
    experiment_grid = []

    slr_class = DEFAULT_ALGORITHM_CONFIG["SimpleLinearRegression"]["class"]
    slr_params = DEFAULT_ALGORITHM_CONFIG["SimpleLinearRegression"]["params"]
    experiment_grid.append({"name": "SimpleLinearRegression (Default)", "class": slr_class, "params": slr_params})

    lr_class = DEFAULT_ALGORITHM_CONFIG["LinearRegression"]["class"]

    lr_default_params = DEFAULT_ALGORITHM_CONFIG["LinearRegression"]["params"]
    experiment_grid.append({"name": "LinearRegression (Default)", "class": lr_class, "params": lr_default_params})

    lr_params_r = [1.0E-10, 1.0E-8, 1.0E-6, 1.0E-4, 1.0E-2, 0.1, 1.0]
    for r in lr_params_r:
        params_str = f"-S 0 -R {r} -num-decimal-places 4"
        name = f"LinearRegression_R={r}_Dec=4"
        if params_str == lr_default_params:
            name += " (Default)"
        experiment_grid.append({"name": name, "class": lr_class, "params": params_str})

    return experiment_grid


def run_weka_command(command_list: list) -> Optional[str]:
    cmd_list_str = [str(item) for item in command_list]
    try:
        return subprocess.run(cmd_list_str, capture_output=True, text=True, encoding='utf-8', check=True).stdout
    except FileNotFoundError:
        print("\nERROR: 'java' command not found.")
        return None
    except subprocess.CalledProcessError as e:
        print(f"--- Weka Command Failed ---")
        print(f"Command: {' '.join(cmd_list_str)}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        return None


def save_dataframe_to_csv(df: pd.DataFrame, csv_path: Path) -> bool:
    try:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"--- Successfully saved data to '{csv_path.name}' ---")
        return True
    except IOError:
        print(f"Error writing CSV file to {csv_path}")
        return False


def convert_csv_to_arff(csv_path: Path, arff_path: Path, java_cmd_base: list) -> bool:
    print(f"--- Converting '{csv_path.name}' to ARFF...")
    cmd_convert = java_cmd_base + ["weka.core.converters.CSVLoader", str(csv_path)]
    arff_content = run_weka_command(cmd_convert)
    if arff_content:
        try:
            arff_path.parent.mkdir(parents=True, exist_ok=True)
            with open(arff_path, "w", encoding='utf-8') as f:
                f.write(arff_content)
            return True
        except IOError:
            print(f"Error writing ARFF file to {arff_path}")
            return False
    print(f"Failed to get ARFF content for {csv_path}")
    return False


def apply_attribute_filter(input_arff: Path, output_arff: Path, java_cmd_base: list, attributes_to_select: str) -> bool:
    print(f"--- Applying Remove filter to select attributes: {attributes_to_select} ---")
    cmd_filter = java_cmd_base + ["weka.filters.unsupervised.attribute.Remove", "-R", attributes_to_select, "-V", "-i",
                                  str(input_arff), "-o", str(output_arff)]
    output = run_weka_command(cmd_filter)
    if output is None: return False
    return output_arff.is_file() and output_arff.stat().st_size > 0


def parse_regression_metrics(weka_output: Optional[str]) -> Dict[str, Optional[float]]:
    metrics = {
        "Correlation coefficient": None,  # Added this
        "Mean absolute error": None,
        "Root mean squared error": None,
        "Relative absolute error": None,
        "Root relative squared error": None
    }
    if not weka_output:
        return metrics

    for metric_name in metrics:
        match = re.search(rf"^\s*{re.escape(metric_name)}\s+([0-9.-]+)", weka_output,
                          re.MULTILINE)  # Adjusted regex for potential negative correlation
        if match:
            try:
                metrics[metric_name] = float(match.group(1))
            except ValueError:
                pass
    return metrics


def main():
    if not WEKA_JAR_PATH.is_file():
        print(f"FATAL: weka.jar not found at '{WEKA_JAR_PATH}'. Exiting.");
        return

    RESULTS_DIR.mkdir(exist_ok=True);
    VISUALIZATION_DIR.mkdir(exist_ok=True);
    ARFF_DATA_DIR.mkdir(exist_ok=True)
    CSV_DATA_DIR.mkdir(exist_ok=True)

    MTJ_JAR_PATH = SCRIPT_DIR / "mtj-1.0.4.jar"

    java_cmd_base = [
        "java", f"-Xmx{WEKA_MEMORY_MB}m",
        "--add-opens=java.base/java.lang=ALL-UNNAMED",
        "-cp", f"{WEKA_JAR_PATH}:{MTJ_JAR_PATH}"
    ]
    full_experiment_grid = build_experiment_grid()

    print("\n--- Fetching Communities and Crime dataset from UCI ML Repo ---")
    try:
        communities_and_crime = fetch_ucirepo(id=183)
        X = communities_and_crime.data.features
        y = communities_and_crime.data.targets

        full_df = pd.concat([X, y], axis=1)

        for col in full_df.columns:
            if full_df[col].dtype == 'object':
                full_df[col] = pd.to_numeric(full_df[col], errors='coerce')

        full_df = full_df.fillna(full_df.mean(numeric_only=True))

        if 'ViolentCrimesPerPop' in full_df.columns:
            full_df.dropna(subset=['ViolentCrimesPerPop'], inplace=True)
        else:
            print("FATAL: 'ViolentCrimesPerPop' not found in the combined DataFrame. Check dataset structure.")
            return

        dataset_prefix = DATASET_PREFIXES[0]
        csv_file_path = CSV_DATA_DIR / f"{dataset_prefix}.csv"

        if not save_dataframe_to_csv(full_df, csv_file_path):
            print(f"FATAL: Could not save fetched data to CSV. Exiting.")
            return

    except Exception as e:
        print(f"FATAL: Error fetching or processing Communities and Crime dataset: {e}. Exiting.")
        return

    for prefix in DATASET_PREFIXES:
        print(f"\n{'=' * 20} RUNNING EXPERIMENTS FOR DATASET: {prefix} {'=' * 20}")
        csv_file = CSV_DATA_DIR / f"{prefix}.csv"
        initial_arff_file = ARFF_DATA_DIR / f"{prefix}_initial.arff"
        final_arff_file = ARFF_DATA_DIR / f"{prefix}.arff"

        if initial_arff_file.exists(): initial_arff_file.unlink()
        if final_arff_file.exists(): final_arff_file.unlink()

        if not csv_file.is_file():
            print(f"FATAL: Source CSV file '{csv_file}' not found. Ensure it was downloaded/processed correctly.");
            continue

        if not convert_csv_to_arff(csv_file, initial_arff_file, java_cmd_base):
            print(f"FATAL: Could not create initial ARFF file. Skipping.");
            continue

        if not apply_attribute_filter(initial_arff_file, final_arff_file, java_cmd_base, ATTRIBUTES_TO_SELECT):
            print(f"FATAL: Could not apply attribute filter. Skipping.");
            continue

        if initial_arff_file.exists(): initial_arff_file.unlink()

        print("--- Data preparation complete. Starting experiments. ---")

        all_results = []
        best_models_by_family = {
            "SimpleLinearRegression": {"config": None, "rmse": float('inf')},
            "LinearRegression": {"config": None, "rmse": float('inf')}
        }

        for config in full_experiment_grid:
            print(f"\n--- Running: {config['name']} ---")
            results = {"Algorithm": config['name']}
            classifier_class = config['class']
            classifier_params = config['params'].split() if config['params'] else []

            cmd_cv = java_cmd_base + [classifier_class, "-t", str(final_arff_file), "-c", "last", "-x",
                                      "10"] + classifier_params
            output_cv = run_weka_command(cmd_cv)
            cv_metrics = parse_regression_metrics(output_cv)

            results["Cross validation (RMSE)"] = cv_metrics.get("Root mean squared error")
            results["Cross validation (Corr Coeff)"] = cv_metrics.get("Correlation coefficient")  # Added

            cmd_split = java_cmd_base + [classifier_class, "-t", str(final_arff_file), "-c", "last",
                                         "-split-percentage", "66"] + classifier_params
            output_split = run_weka_command(cmd_split)
            split_metrics = parse_regression_metrics(output_split)

            results["Percentage split (RMSE)"] = split_metrics.get("Root mean squared error")
            results["Percentage split (Corr Coeff)"] = split_metrics.get("Correlation coefficient")  # Added

            results["Parameters"] = config['params'] if config['params'] else " "

            rmses = [rmse for rmse in [results["Cross validation (RMSE)"], results["Percentage split (RMSE)"]] if
                     rmse is not None]
            results["Vidurkis (RMSE)"] = sum(rmses) / len(rmses) if rmses else np.nan

            corrs = [corr for corr in
                     [results["Cross validation (Corr Coeff)"], results["Percentage split (Corr Coeff)"]] if
                     corr is not None]
            results["Vidurkis (Corr Coeff)"] = sum(corrs) / len(corrs) if corrs else np.nan  # Added

            if not rmses:
                print(f"FATAL: Evaluation failed for '{config['name']}'. Skipping.");
                continue

            all_results.append(results)

            family_name = config['name'].split(' ')[0].split('_')[0]
            if family_name in best_models_by_family:
                current_rmse = results["Vidurkis (RMSE)"]
                best_so_far = best_models_by_family[family_name]

                if current_rmse < best_so_far["rmse"]:
                    print(f"*** New best {family_name} model found: {config['name']} (Avg RMSE: {current_rmse}) ***")
                    best_models_by_family[family_name] = {"config": config, "rmse": current_rmse}

        if not all_results:
            print(f"\nNo results generated for dataset '{prefix}'.");
            continue

        results_df = pd.DataFrame(all_results)
        # Sort by RMSE ascending (lower is better) and then by Correlation Coefficient descending (higher is better)
        results_df = results_df.sort_values(by=["Vidurkis (RMSE)", "Vidurkis (Corr Coeff)"], ascending=[True, False])

        print(f"\n\n--- Experiment Results Summary for '{prefix}' ---\n{results_df.set_index('Algorithm')}")
        results_df.to_csv(RESULTS_DIR / f"{prefix}_regression_results.csv", index=False)
        print(f"Results saved to '{RESULTS_DIR / f'{prefix}_regression_results.csv'}'")

        print("\n--- Generating final outputs (model summaries)... ---")

        for algo_name, default_config_params in DEFAULT_ALGORITHM_CONFIG.items():
            default_config = {"name": f"{algo_name} (Default)", **default_config_params}

            generate_summary(algo_name, default_config, final_arff_file, java_cmd_base, prefix)

            best_model_entry = best_models_by_family.get(algo_name)
            if best_model_entry and best_model_entry.get("config"):
                best_config = best_model_entry["config"]
                if best_config["params"] != default_config["params"]:
                    generate_summary(algo_name, best_config, final_arff_file, java_cmd_base, prefix)
                else:
                    print(
                        f"--- Best {algo_name} model was the default. Skipping duplicate optimised model summary. ---")


def generate_summary(algo_name: str, default_config: dict[str, str], final_arff_file: Path, java_cmd_base: list[str],
                     prefix: str):
    output_filename_base_default = f"{prefix}_{algo_name}_default_model_summary.txt"
    cmd_model_summary_default = java_cmd_base + [default_config['class'], "-t", str(final_arff_file), "-c",
                                                 "last"] + (
                                    default_config['params'].split() if default_config['params'] else [])
    model_summary_output_default = run_weka_command(cmd_model_summary_default)
    if model_summary_output_default:
        with open(RESULTS_DIR / output_filename_base_default, 'w', encoding='utf-8') as f:
            f.write(model_summary_output_default)
        print(f"Generated default model summary: {RESULTS_DIR / output_filename_base_default}")
    else:
        print(f"Failed to generate default model summary for {algo_name}.")


if __name__ == "__main__":
    main()
