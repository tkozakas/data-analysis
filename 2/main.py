import subprocess
import re
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import itertools
import random

SCRIPT_DIR = Path(__file__).parent.resolve()

WEKA_JAR_PATH = SCRIPT_DIR / "weka.jar"
DATA_DIR = SCRIPT_DIR / "discretization"
RESULTS_DIR = SCRIPT_DIR / "results"
VISUALIZATION_DIR = SCRIPT_DIR / "visualizations"
WEKA_MEMORY_MB = 2048
PATIENCE = 20

DATASET_PREFIXES = [
    "original_sup",
    "removed_missing_sup",
    "replace_sup"
]

DEFAULT_ALGORITHM_CONFIG = {
    "J48": {
        "class": "weka.classifiers.trees.J48",
        "params": "-C 0.25 -M 2"
    },
    "RandomTree": {
        "class": "weka.classifiers.trees.RandomTree",
        "params": "-K 0 -M 1.0 -V 0.001 -S 1"
    },
    "REPTree": {
        "class": "weka.classifiers.trees.REPTree",
        "params": "-M 2 -V 0.001 -N 3 -S 1 -L -1 -I 0.0"
    }
}


def build_experiment_grid() -> List[Dict[str, Any]]:
    experiment_grid = []

    j48_class = "weka.classifiers.trees.J48"
    j48_params_c = [0.1, 0.25, 0.4, 0.6, 0.7]
    j48_params_m = [2, 5, 10, 20]

    for c, m in itertools.product(j48_params_c, j48_params_m):
        params_str = f"-C {c} -M {m}"
        name = f"J48_C={c}_M={m}"
        if params_str == DEFAULT_ALGORITHM_CONFIG["J48"]["params"]:
            name += " (Original Default)"

        experiment_grid.append({
            "name": name,
            "class": j48_class,
            "params": params_str
        })

    reptree_class = "weka.classifiers.trees.REPTree"
    reptree_params_l = [3, 5, -1, -2]
    reptree_params_m = [2, 5, 10, 20]

    for l, m in itertools.product(reptree_params_l, reptree_params_m):
        params_str = f"-M {m} -V 0.001 -N 3 -S 1 -L {l} -I 0.0"
        name = f"REPTree_L={l}_M={m}"
        if params_str == DEFAULT_ALGORITHM_CONFIG["REPTree"]["params"]:
            name += " (Original Default)"

        experiment_grid.append({
            "name": name,
            "class": reptree_class,
            "params": params_str
        })

    randomtree_class = "weka.classifiers.trees.RandomTree"
    randomtree_params_k = [0, 5, 10]
    randomtree_params_m = [1.0, 5, 10]

    for k, m in itertools.product(randomtree_params_k, randomtree_params_m):
        params_str = f"-K {k} -M {m} -V 0.001 -S 1"
        name = f"RandomTree_K={k}_M={m}"
        if params_str == DEFAULT_ALGORITHM_CONFIG["RandomTree"]["params"]:
            name += " (Original Default)"

        experiment_grid.append({
            "name": name,
            "class": randomtree_class,
            "params": params_str
        })

    return experiment_grid


def run_weka_command(command_list: list) -> Optional[str]:
    cmd_list_str = [str(item) for item in command_list]
    print(f"Executing: {' '.join(cmd_list_str)}")
    try:
        result = subprocess.run(
            cmd_list_str,
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=True
        )
        return result.stdout
    except FileNotFoundError:
        print("\nERROR: 'java' command not found. Is Java installed and in your PATH?")
        return None
    except subprocess.CalledProcessError as e:
        print(f"--- WEKA ERROR DETECTED ---")
        print(f"--- STDERR from Java: ---")
        print(e.stderr)
        print(f"--------------------------")
        return None


def run_dot_command(dot_source: str, output_png_path: Path) -> bool:
    dot_cmd = ["dot", "-Tpng", "-o", str(output_png_path)]
    print(f"Executing: {' '.join(dot_cmd)} (piping .dot data)")

    try:
        subprocess.run(
            dot_cmd,
            input=dot_source,
            text=True,
            encoding='utf-8',
            check=True,
            capture_output=True
        )
        print(f"Successfully generated visualization: {output_png_path}")
        return True
    except FileNotFoundError:
        print("\nERROR: 'dot' command not found. Is Graphviz installed in your PATH?")
        return False
    except subprocess.CalledProcessError as e:
        print(f"--- DOT/Graphviz ERROR DETECTED ---")
        print(f"--- STDERR from dot: ---")
        print(e.stderr)
        print(f"---------------------------------")
        return False
    except Exception as e:
        print(f"An unexpected error occurred running dot: {e}")
        return False


def parse_correctly_classified_percentage(weka_output: Optional[str]) -> Optional[float]:
    if not weka_output:
        return None
    matches = re.findall(r"^\s*Correctly Classified Instances\s+.*?([\d.]+)\s*%", weka_output, re.MULTILINE)
    if matches:
        try:
            return float(matches[-1])
        except (ValueError, IndexError):
            return None
    return None


def parse_tree_size(weka_output: Optional[str]) -> Optional[float]:
    if not weka_output:
        return None
    match = re.search(r"^\s*Size of the tree\s*:\s*(\d+)", weka_output, re.MULTILINE)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, IndexError):
            return None
    return np.nan


def generate_visualization_for_config(config: Dict[str, Any], main_file: Path, java_cmd_base: List[str],
                                      output_filename_base: str):
    classifier_class = config['class']
    classifier_params = config['params'].split() if config['params'] else []

    safe_filename = output_filename_base.replace(" ", "_").replace("(", "").replace(")", "")
    output_png_file = VISUALIZATION_DIR / safe_filename

    cmd_graph = java_cmd_base + [classifier_class, "-t", str(main_file)] + classifier_params + ["-g"]
    output_dot_source = run_weka_command(cmd_graph)

    if output_dot_source:
        run_dot_command(output_dot_source, output_png_file)
    else:
        print(f"--- Failed to get .dot source for {config['name']}, skipping visualization. ---")


def main():
    if not WEKA_JAR_PATH.is_file():
        print(f"FATAL: weka.jar not found at '{WEKA_JAR_PATH}'. Exiting.")
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    VISUALIZATION_DIR.mkdir(exist_ok=True)

    java_cmd_base = ["java", f"-Xmx{WEKA_MEMORY_MB}m", "-cp", str(WEKA_JAR_PATH)]
    print(f"--- Using Java memory setting: -Xmx{WEKA_MEMORY_MB}m ---")

    full_experiment_grid = build_experiment_grid()
    print(f"--- Generated {len(full_experiment_grid)} total experiment configurations to run per dataset. ---")

    for prefix in DATASET_PREFIXES:
        print(f"\n\n{'=' * 20} RUNNING EXPERIMENTS FOR DATASET: {prefix} {'=' * 20}")

        main_file = DATA_DIR / f"{prefix}.arff"
        train_file = DATA_DIR / f"{prefix}_train.arff"
        test_file = DATA_DIR / f"{prefix}_test.arff"

        if not all(f.is_file() for f in [main_file, train_file, test_file]):
            print(f"FATAL: One or more data files missing for prefix '{prefix}'. Skipping.")
            continue

        all_results = []
        best_models_by_family = {
            "J48": {"config": None, "stats": {"Cross validation": -1.0, "Tree size": float('inf')}},
            "REPTree": {"config": None, "stats": {"Cross validation": -1.0, "Tree size": float('inf')}},
            "RandomTree": {"config": None, "stats": {"Cross validation": -1.0, "Tree size": float('inf')}}
        }
        experiments_since_improvement = 0

        shuffled_grid = full_experiment_grid.copy()
        random.shuffle(shuffled_grid)

        print(f"--- Starting randomized search with patience={PATIENCE} ---")

        for config in shuffled_grid:
            algo_name = config['name']
            classifier_class = config['class']
            classifier_params = config['params'].split() if config['params'] else []

            print(f"\n--- Running: {algo_name} on {prefix} ---")
            results: Dict[str, Any] = {"Algorithm": algo_name}

            cmd_cv = java_cmd_base + [classifier_class, "-t", main_file, "-x", "10"] + classifier_params
            output_cv = run_weka_command(cmd_cv)
            results["Cross validation"] = parse_correctly_classified_percentage(output_cv)

            cmd_test = java_cmd_base + [classifier_class, "-t", train_file, "-T",
                                        test_file] + classifier_params
            output_test = run_weka_command(cmd_test)
            results["Test / training"] = parse_correctly_classified_percentage(output_test)

            cmd_train = java_cmd_base + [classifier_class, "-t", main_file, "-T",
                                         main_file] + classifier_params
            output_train = run_weka_command(cmd_train)
            results["Training"] = parse_correctly_classified_percentage(output_train)

            cmd_split = java_cmd_base + [classifier_class, "-t", main_file, "-split-percentage",
                                         "66"] + classifier_params
            output_split = run_weka_command(cmd_split)
            results["Percentage split"] = parse_correctly_classified_percentage(output_split)

            cmd_model_summary = java_cmd_base + [classifier_class, "-t", main_file, "-no-cv"] + classifier_params
            output_model_summary = run_weka_command(cmd_model_summary)
            results["Tree size"] = parse_tree_size(output_model_summary)

            results["Parameters"] = config['params'] if config['params'] else " "

            if any(v is None for v in results.values() if v != " "):
                print(f"\nFATAL: One or more Weka commands failed for algorithm '{algo_name}'. Skipping this config.")
                continue

            accuracies = [results["Cross validation"], results["Test / training"], results["Training"],
                          results["Percentage split"]]
            valid_accuracies = [acc for acc in accuracies if acc is not None]
            results["Vidurkis"] = sum(valid_accuracies) / len(valid_accuracies) if valid_accuracies else np.nan
            all_results.append(results)

            current_cv_val = results["Cross validation"] if results["Cross validation"] is not None else -1.0
            current_size_val = results["Tree size"] if not pd.isna(results["Tree size"]) else float('inf')

            current_family = None
            if "J48" in algo_name:
                current_family = "J48"
            elif "REPTree" in algo_name:
                current_family = "REPTree"
            elif "RandomTree" in algo_name:
                current_family = "RandomTree"

            is_better = False
            if current_family:
                best_so_far_stats = best_models_by_family[current_family]["stats"]

                if current_cv_val > best_so_far_stats["Cross validation"]:
                    is_better = True
                elif current_cv_val == best_so_far_stats["Cross validation"] and current_size_val < best_so_far_stats[
                    "Tree size"]:
                    is_better = True

            if is_better:
                print(
                    f"*** New best {current_family} model found: {algo_name} (CV: {current_cv_val}%, Size: {current_size_val}) ***")
                best_models_by_family[current_family]["config"] = config
                best_models_by_family[current_family]["stats"] = results
                experiments_since_improvement = 0
            else:
                experiments_since_improvement += 1
                print(f"--- No improvement. Patience: {experiments_since_improvement}/{PATIENCE} ---")

            if "(Original Default)" in algo_name and not pd.isna(results["Tree size"]):
                if current_family:
                    algo_family_lower = current_family.lower()
                    print(f"--- Visualizing default model: {algo_name} ---")
                    generate_visualization_for_config(config, main_file, java_cmd_base,
                                                      f"{prefix}_{algo_family_lower}_default.png")

            if experiments_since_improvement >= PATIENCE:
                print(f"\n--- Patience of {PATIENCE} exceeded. Stopping search for dataset '{prefix}'. ---")
                break

        if not all_results:
            print(f"\nNo results were generated for dataset '{prefix}' due to errors.")
            continue

        for family, best_model in best_models_by_family.items():
            if best_model["config"]:
                algo_family_lower = family.lower()
                if "(Original Default)" not in best_model["config"]["name"]:
                    print(f"\n--- Generating final optimized visualization for: {family} ---")
                    generate_visualization_for_config(
                        best_model["config"],
                        main_file,
                        java_cmd_base,
                        f"{prefix}_{algo_family_lower}_optimised.png"
                    )
                else:
                    print(
                        f"\n--- Best model for {family} was the default, which is already visualized. Skipping duplicate optimized PNG. ---")
            else:
                print(f"\n--- No successful optimized model found for {family}, skipping visualization. ---")

        results_df = pd.DataFrame(all_results).set_index("Algorithm")
        results_df = results_df.sort_values(by=["Cross validation", "Tree size"], ascending=[False, True])

        print(f"\n\n--- Experiment Results Summary for '{prefix}' (Sorted by Best CV Accuracy) ---")
        print(results_df)

        sorted_csv_filename = RESULTS_DIR / f"{prefix}_results_sorted_list.csv"
        results_df.to_csv(sorted_csv_filename)
        print(f"Sorted list of models (from tests run) saved to '{sorted_csv_filename}'")

        default_results_df = results_df[results_df.index.str.contains(r"\(Original Default\)", regex=True)]

        if not default_results_df.empty:
            default_results_transposed = default_results_df.T

            desired_order = ["Cross validation", "Test / training", "Training", "Percentage split", "Vidurkis",
                             "Tree size",
                             "Parameters"]
            default_results_transposed = default_results_transposed.reindex(desired_order)

            numeric_rows = [row for row in desired_order if row != "Parameters"]
            default_results_transposed.loc[numeric_rows] = (default_results_transposed.loc[numeric_rows]
                                                            .apply(pd.to_numeric, errors='coerce')
                                                            .round(4))

            print(f"\n\n--- Default Models Summary for '{prefix}' ---")
            print(default_results_transposed)

            csv_filename = RESULTS_DIR / f"{prefix}_results.csv"
            default_results_transposed.to_csv(csv_filename)
            print(f"Default-only results (like image) saved to '{csv_filename}'")
        else:
            print("\n--- No default models were successfully run, skipping default results CSV. ---")


if __name__ == "__main__":
    main()
