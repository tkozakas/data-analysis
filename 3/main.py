# main.py (with robust NumericToNominal filter)
import subprocess
import re
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, List
import itertools
import random

# --- Constants ---
SCRIPT_DIR = Path(__file__).parent.resolve()
WEKA_JAR_PATH = SCRIPT_DIR / "weka.jar"
CSV_DATA_DIR = SCRIPT_DIR / "data"
ARFF_DATA_DIR = SCRIPT_DIR / "discretization"
RESULTS_DIR = SCRIPT_DIR / "results"
VISUALIZATION_DIR = SCRIPT_DIR / "visualizations"
WEKA_MEMORY_MB = 2048
PATIENCE = 50
DATASET_PREFIXES = ["email_features"]

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
    randomtree_params_k = [0, 2, 5, 8]
    randomtree_params_m = [1.0, 10, 20, 30]

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
        return subprocess.run(cmd_list_str, capture_output=True, text=True, encoding='utf-8', check=True).stdout
    except FileNotFoundError:
        print("\nERROR: 'java' command not found."); return None
    except subprocess.CalledProcessError as e:
        print(f"--- WEKA ERROR ---\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}\n----------")
        return None

def convert_csv_to_arff(csv_path: Path, arff_path: Path, java_cmd_base: list) -> bool:
    print(f"--- Converting '{csv_path.name}' to ARFF...")
    cmd_convert = java_cmd_base + ["weka.core.converters.CSVLoader", str(csv_path)]
    arff_content = run_weka_command(cmd_convert)
    if arff_content:
        try:
            arff_path.parent.mkdir(parents=True, exist_ok=True)
            with open(arff_path, "w", encoding='utf-8') as f: f.write(arff_content)
            return True
        except IOError: return False
    return False

def apply_numeric_to_nominal_filter(input_arff: Path, output_arff: Path, java_cmd_base: list) -> bool:
    print(f"--- Applying NumericToNominal filter on the class attribute...")
    cmd_filter = java_cmd_base + [
        "weka.filters.unsupervised.attribute.NumericToNominal",
        "-R", "last", "-i", str(input_arff), "-o", str(output_arff)
    ]
    output = run_weka_command(cmd_filter)
    if output is None: return False
    return output_arff.is_file() and output_arff.stat().st_size > 0

def parse_correctly_classified_percentage(weka_output: Optional[str]) -> Optional[float]:
    if not weka_output: return None
    matches = re.findall(r"^\s*Correctly Classified Instances\s+.*?([\d.]+)\s*%", weka_output, re.MULTILINE)
    return float(matches[-1]) if matches else None

def main():
    if not WEKA_JAR_PATH.is_file():
        print(f"FATAL: weka.jar not found at '{WEKA_JAR_PATH}'. Exiting."); return
    RESULTS_DIR.mkdir(exist_ok=True); VISUALIZATION_DIR.mkdir(exist_ok=True); ARFF_DATA_DIR.mkdir(exist_ok=True)
    java_cmd_base = [
        "java", f"-Xmx{WEKA_MEMORY_MB}m",
        "--add-opens=java.base/java.lang=ALL-UNNAMED",
        "-cp", str(WEKA_JAR_PATH)
    ]
    full_experiment_grid = build_experiment_grid()

    for prefix in DATASET_PREFIXES:
        print(f"\n{'=' * 20} RUNNING EXPERIMENTS FOR DATASET: {prefix} {'=' * 20}")
        csv_file = CSV_DATA_DIR / f"{prefix}.csv"
        temp_arff_file = ARFF_DATA_DIR / f"{prefix}_temp.arff"
        final_arff_file = ARFF_DATA_DIR / f"{prefix}.arff"

        if final_arff_file.exists(): final_arff_file.unlink()
        if not csv_file.is_file():
            print(f"FATAL: Source CSV file '{csv_file}' not found. Run build_dataset.py first."); continue
        
        if not convert_csv_to_arff(csv_file, temp_arff_file, java_cmd_base):
            print(f"FATAL: Could not create initial ARFF file. Skipping."); continue
        
        if not apply_numeric_to_nominal_filter(temp_arff_file, final_arff_file, java_cmd_base):
            print(f"FATAL: Could not create final filtered ARFF file. Skipping."); continue
        
        if temp_arff_file.exists(): temp_arff_file.unlink()
        print("--- Data preparation complete. Starting experiments. ---")

        all_results = []
        for config in full_experiment_grid:
            algo_name = config['name']
            classifier_class = config['class']
            classifier_params = config['params'].split() if config['params'] else []
            print(f"\n--- Running: {algo_name} on {prefix} ---")
            results = {"Algorithm": algo_name}
            
            cmd_cv = java_cmd_base + [classifier_class, "-t", final_arff_file, "-x", "10"] + classifier_params
            output_cv = run_weka_command(cmd_cv)
            results["Cross validation"] = parse_correctly_classified_percentage(output_cv)
            
            if results["Cross validation"] is None:
                print(f"FATAL: Cross-validation failed for '{algo_name}'. Skipping."); continue
            
            all_results.append(results)
            
        if not all_results:
            print(f"\nNo results generated for dataset '{prefix}'."); continue
        
        results_df = pd.DataFrame(all_results).set_index("Algorithm")
        results_df = results_df.sort_values(by="Cross validation", ascending=False)
        print(f"\n\n--- Experiment Results Summary for '{prefix}' ---\n{results_df}")
        results_df.to_csv(RESULTS_DIR / f"{prefix}_results.csv")
        print(f"Results saved to '{RESULTS_DIR / f'{prefix}_results.csv'}'")

if __name__ == "__main__":
    main()
