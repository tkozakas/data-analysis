import subprocess
import re
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas.api.types  # Import pandas.api.types for pd.isna check

# --- CONFIGURATION ---
WEKA_JAR_PATH = Path("weka.jar")
DATA_DIR = Path("discretization")
RESULTS_DIR = Path("results")
VISUALIZATION_DIR = Path("visualizations")  # <-- NEW: Directory for PNGs
WEKA_MEMORY_MB = 2048

DATASET_PREFIXES = [
    "original_sup",
    "removed_missing_sup",
    "replace_sup"
]

ALGORITHM_CONFIG = {
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
        "params": "-M 2 -V 0.001 -N 3 -S 1"
    }
}


def run_weka_command(command_list: list) -> Optional[str]:
    """Executes a Weka command and returns its stdout, or None on error."""
    print(f"Executing: {' '.join(command_list)}")
    try:
        result = subprocess.run(
            command_list,
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


# --- NEW FUNCTION ---
def run_dot_command(dot_source: str, output_png_path: Path) -> bool:
    """Executes the 'dot' command to generate a PNG from .dot source string."""
    dot_cmd = ["dot", "-Tpng", "-o", str(output_png_path)]
    print(f"Executing: {' '.join(dot_cmd)} (piping .dot data)")

    try:
        subprocess.run(
            dot_cmd,
            input=dot_source,
            text=True,
            encoding='utf-8',
            check=True,
            capture_output=True  # Capture stdout/stderr for dot
        )
        print(f"Successfully generated visualization: {output_png_path}")
        return True
    except FileNotFoundError:
        print("\nERROR: 'dot' command not found. Is Graphviz installed and in your PATH?")
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


# --- END NEW FUNCTION ---

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
    """Parses Weka's output and returns the size of the tree, if available."""
    if not weka_output:
        return None

    match = re.search(r"^\s*Size of the tree\s*:\s*(\d+)", weka_output, re.MULTILINE)
    if match:
        try:
            return float(match.group(1))
        except (ValueError, IndexError):
            return None

    return np.nan


def main():
    if not WEKA_JAR_PATH.is_file():
        print(f"FATAL: weka.jar not found at '{WEKA_JAR_PATH}'. Exiting.")
        return

    # --- MODIFIED: Create output directories ---
    RESULTS_DIR.mkdir(exist_ok=True)
    VISUALIZATION_DIR.mkdir(exist_ok=True)
    # --- END MODIFICATION ---

    java_cmd_base = ["java", f"-Xmx{WEKA_MEMORY_MB}m", "-cp", str(WEKA_JAR_PATH)]
    print(f"--- Using Java memory setting: -Xmx{WEKA_MEMORY_MB}m ---")

    for prefix in DATASET_PREFIXES:
        print(f"\n\n{'=' * 20} RUNNING EXPERIMENTS FOR DATASET: {prefix} {'=' * 20}")

        main_file = DATA_DIR / f"{prefix}.arff"
        train_file = DATA_DIR / f"{prefix}_train.arff"
        test_file = DATA_DIR / f"{prefix}_test.arff"

        if not all(f.is_file() for f in [main_file, train_file, test_file]):
            print(f"FATAL: One or more data files missing for prefix '{prefix}'. Skipping.")
            continue

        all_results = []
        for algo_name, config in ALGORITHM_CONFIG.items():
            print(f"\n--- Running experiments for {algo_name} on {prefix} ---")
            results: Dict[str, Any] = {"Algorithm": algo_name}
            classifier_class = config['class']
            classifier_params = config['params'].split() if config['params'] else []

            cmd_cv = java_cmd_base + [classifier_class, "-t", str(main_file), "-x", "10"] + classifier_params
            output_cv = run_weka_command(cmd_cv)
            results["Cross validation"] = parse_correctly_classified_percentage(output_cv)

            cmd_test = java_cmd_base + [classifier_class, "-t", str(train_file), "-T",
                                        str(test_file)] + classifier_params
            output_test = run_weka_command(cmd_test)
            results["Test / training"] = parse_correctly_classified_percentage(output_test)

            cmd_train = java_cmd_base + [classifier_class, "-t", str(main_file), "-T",
                                         str(main_file)] + classifier_params
            output_train = run_weka_command(cmd_train)
            results["Training"] = parse_correctly_classified_percentage(output_train)

            cmd_split = java_cmd_base + [classifier_class, "-t", str(main_file), "-split-percentage",
                                         "66"] + classifier_params
            output_split = run_weka_command(cmd_split)
            results["Percentage split"] = parse_correctly_classified_percentage(output_split)

            cmd_model_summary = java_cmd_base + [classifier_class, "-t", str(main_file), "-no-cv"] + classifier_params
            output_model_summary = run_weka_command(cmd_model_summary)
            results["Tree size"] = parse_tree_size(output_model_summary)

            results["Parameters"] = config['params'] if config['params'] else " "

            # --- NEW: Visualization logic ---
            # Check if parse_tree_size found a valid number (not np.nan)
            if not pd.api.types.is_scalar(results["Tree size"]) or not np.isnan(results["Tree size"]):
                print(
                    f"\n--- Generating visualization for {algo_name} on {prefix} (Tree size: {results['Tree size']}) ---")

                # 1. Define Weka command to get .dot source (using -g)
                cmd_graph = java_cmd_base + [classifier_class, "-t", str(main_file)] + classifier_params + ["-g"]
                output_dot_source = run_weka_command(cmd_graph)

                if output_dot_source:
                    # 2. Define output PNG file path
                    output_png_file = VISUALIZATION_DIR / f"{prefix}_{algo_name}_tree.png"

                    # 3. Run the dot command helper
                    run_dot_command(output_dot_source, output_png_file)
            else:
                print(f"\n--- Skipping visualization for {algo_name} (not a tree or tree size is 0) ---")
            # --- END NEW: Visualization logic ---

            if any(v is None for v in results.values() if v != " "):
                print(
                    f"\nFATAL: One or more Weka commands failed for algorithm '{algo_name}' on dataset '{prefix}'. Aborting this dataset.")
                break

            accuracies = [results["Cross validation"], results["Test / training"], results["Training"],
                          results["Percentage split"]]
            valid_accuracies = [acc for acc in accuracies if acc is not None]
            if valid_accuracies:
                results["Vidurkis"] = sum(valid_accuracies) / len(valid_accuracies)
            else:
                results["Vidurkis"] = np.nan
            all_results.append(results)

        if not all_results:
            print(f"\nNo results were generated for dataset '{prefix}' due to errors.")
            continue

        results_df = pd.DataFrame(all_results).set_index("Algorithm").T
        desired_order = ["Cross validation", "Test / training", "Training", "Percentage split", "Vidurkis", "Tree size",
                         "Parameters"]
        results_df = results_df.reindex(desired_order)
        numeric_rows = [row for row in desired_order if row != "Parameters"]
        results_df.loc[numeric_rows] = results_df.loc[numeric_rows].apply(pd.to_numeric, errors='coerce').round(4)

        print(f"\n\n--- Experiment Results Summary for '{prefix}' ---")
        print(results_df)

        csv_filename = RESULTS_DIR / f"{prefix}_results.csv"  # <-- MODIFIED: Use RESULTS_DIR
        results_df.to_csv(csv_filename)
        print(f"Results for '{prefix}' also saved to '{csv_filename}'")


if __name__ == "__main__":
    main()