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
CSV_DATA_DIR = SCRIPT_DIR / "data"
ARFF_DATA_DIR = SCRIPT_DIR / "discretization"
RESULTS_DIR = SCRIPT_DIR / "results"
VISUALIZATION_DIR = SCRIPT_DIR / "visualizations"
WEKA_MEMORY_MB = 2048
DATASET_PREFIXES = ["email_features"]

DEFAULT_ALGORITHM_CONFIG = {
    "J48": { "class": "weka.classifiers.trees.J48", "params": "-C 0.25 -M 2" },
    "RandomTree": { "class": "weka.classifiers.trees.RandomTree", "params": "-K 0 -M 1.0 -V 0.001 -S 1" },
    "REPTree": { "class": "weka.classifiers.trees.REPTree", "params": "-M 2 -V 0.001 -N 3 -S 1 -L -1 -I 0.0" },
    "ZeroR": { "class": "weka.classifiers.rules.ZeroR", "params": "" },
    "OneR": { "class": "weka.classifiers.rules.OneR", "params": "-B 6" },
    "JRip": { "class": "weka.classifiers.rules.JRip", "params": "-F 3 -N 2.0 -O 2 -S 1" }
}

def build_experiment_grid() -> List[Dict[str, Any]]:
    experiment_grid = []
    
    # --- Tree-based algorithms ---
    j48_class = "weka.classifiers.trees.J48"
    j48_params_c = [0.1, 0.25, 0.4]; j48_params_m = [2, 5, 10]
    for c, m in itertools.product(j48_params_c, j48_params_m):
        params_str = f"-C {c} -M {m}"
        name = f"J48_C={c}_M={m}"
        if params_str == DEFAULT_ALGORITHM_CONFIG["J48"]["params"]: name += " (Default)"
        experiment_grid.append({"name": name, "class": j48_class, "params": params_str})

    reptree_class = "weka.classifiers.trees.REPTree"
    reptree_params_l = [3, 5, -1]; reptree_params_m = [2, 5, 10]
    for l, m in itertools.product(reptree_params_l, reptree_params_m):
        params_str = f"-M {m} -V 0.001 -N 3 -S 1 -L {l} -I 0.0"
        name = f"REPTree_L={l}_M={m}"
        if params_str == DEFAULT_ALGORITHM_CONFIG["REPTree"]["params"]: name += " (Default)"
        experiment_grid.append({"name": name, "class": reptree_class, "params": params_str})

    randomtree_class = "weka.classifiers.trees.RandomTree"
    randomtree_params_k = [0, 2, 5]; randomtree_params_m = [1.0, 10, 20]
    for k, m in itertools.product(randomtree_params_k, randomtree_params_m):
        params_str = f"-K {k} -M {m} -V 0.001 -S 1"
        name = f"RandomTree_K={k}_M={m}"
        if params_str == DEFAULT_ALGORITHM_CONFIG["RandomTree"]["params"]: name += " (Default)"
        experiment_grid.append({"name": name, "class": randomtree_class, "params": params_str})

    # --- Rule-based algorithms ---
    oner_class = "weka.classifiers.rules.OneR"
    oner_params_b = [2, 6, 10, 20]
    for b in oner_params_b:
        params_str = f"-B {b}"
        name = f"OneR_B={b}"
        # --- THE FIX IS HERE ---
        if params_str == DEFAULT_ALGORITHM_CONFIG["OneR"]["params"]: name += " (Default)"
        experiment_grid.append({"name": name, "class": oner_class, "params": params_str})
        
    jrip_class = "weka.classifiers.rules.JRip"
    jrip_params_n = [1.0, 2.0, 4.0]
    jrip_params_o = [1, 2, 4]
    for n, o in itertools.product(jrip_params_n, jrip_params_o):
        params_str = f"-F 3 -N {n} -O {o} -S 1"
        name = f"JRip_N={n}_O={o}"
        # --- AND THE FIX IS HERE ---
        if params_str == DEFAULT_ALGORITHM_CONFIG["JRip"]["params"]: name += " (Default)"
        experiment_grid.append({"name": name, "class": jrip_class, "params": params_str})

    # --- Baseline (no tuning) ---
    experiment_grid.append({"name": "ZeroR", "class": DEFAULT_ALGORITHM_CONFIG["ZeroR"]["class"], "params": ""})
    return experiment_grid

def run_weka_command(command_list: list) -> Optional[str]:
    cmd_list_str = [str(item) for item in command_list]
    try:
        return subprocess.run(cmd_list_str, capture_output=True, text=True, encoding='utf-8', check=True).stdout
    except FileNotFoundError:
        print("\nERROR: 'java' command not found."); return None
    except subprocess.CalledProcessError:
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
    cmd_filter = java_cmd_base + ["weka.filters.unsupervised.attribute.NumericToNominal", "-R", "last", "-i", str(input_arff), "-o", str(output_arff)]
    output = run_weka_command(cmd_filter)
    if output is None: return False
    return output_arff.is_file() and output_arff.stat().st_size > 0

def run_dot_command(dot_source: str, output_png_path: Path) -> bool:
    dot_cmd = ["dot", "-Tpng", "-o", str(output_png_path)]
    try:
        subprocess.run(dot_cmd, input=dot_source, text=True, encoding='utf-8', check=True, capture_output=True)
        print(f"Successfully generated visualization: {output_png_path}")
        return True
    except FileNotFoundError:
        print("\nERROR: 'dot' command not found. Is Graphviz installed in your PATH?")
        return False
    except subprocess.CalledProcessError:
        print(f"--- DOT/Graphviz ERROR ---"); return False
    return False

def parse_correctly_classified_percentage(weka_output: Optional[str]) -> Optional[float]:
    if not weka_output: return None
    matches = re.findall(r"^\s*Correctly Classified Instances\s+.*?([\d.]+)\s*%", weka_output, re.MULTILINE)
    return float(matches[-1]) if matches else None

def parse_tree_size(weka_output: Optional[str]) -> Optional[float]:
    if not weka_output: return None
    match = re.search(r"^\s*Size of the tree\s*:\s*(\d+)", weka_output, re.MULTILINE)
    return float(match.group(1)) if match else np.nan

def generate_visualization_for_config(config: Dict[str, Any], main_file: Path, java_cmd_base: List[str], output_filename_base: str):
    print(f"--- Generating visualization for {output_filename_base}... ---")
    classifier_class = config['class']
    classifier_params = config['params'].split() if config['params'] else []
    output_png_file = VISUALIZATION_DIR / f"{output_filename_base}.png"
    
    cmd_graph = java_cmd_base + [classifier_class, "-t", str(main_file), "-c", "last"] + classifier_params + ["-g"]
    output_dot_source = run_weka_command(cmd_graph)
    if output_dot_source:
        run_dot_command(output_dot_source, output_png_file)
    else:
        print(f"--- Failed to get .dot source for {config['name']}, skipping visualization (this is expected for REPTree). ---")

def generate_rules_for_config(config: Dict[str, Any], main_file: Path, java_cmd_base: List[str], output_filename_base: str):
    print(f"--- Generating rules file for {output_filename_base}... ---")
    classifier_class = config['class']
    classifier_params = config['params'].split() if config['params'] else []
    output_txt_file = RESULTS_DIR / f"{output_filename_base}.txt"

    cmd_rules = java_cmd_base + [classifier_class, "-t", str(main_file), "-c", "last"] + classifier_params
    rules_output = run_weka_command(cmd_rules)
    if rules_output:
        with open(output_txt_file, 'w', encoding='utf-8') as f:
            f.write(rules_output)
        print(f"Successfully generated rules file: {output_txt_file}")

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
        best_models_by_family = {
            "J48": {"config": None, "cv_score": -1.0, "tree_size": float('inf')},
            "REPTree": {"config": None, "cv_score": -1.0, "tree_size": float('inf')},
            "RandomTree": {"config": None, "cv_score": -1.0, "tree_size": float('inf')},
            "OneR": {"config": None, "cv_score": -1.0},
            "JRip": {"config": None, "cv_score": -1.0}
        }

        for config in full_experiment_grid:
            print(f"\n--- Running: {config['name']} ---")
            results = {"Algorithm": config['name']}
            classifier_class = config['class']
            classifier_params = config['params'].split() if config['params'] else []
            
            cmd_cv = java_cmd_base + [classifier_class, "-t", final_arff_file, "-c", "last", "-x", "10"] + classifier_params
            output_cv = run_weka_command(cmd_cv)
            results["Cross validation"] = parse_correctly_classified_percentage(output_cv)
            
            cmd_split = java_cmd_base + [classifier_class, "-t", final_arff_file, "-c", "last", "-split-percentage", "66"] + classifier_params
            output_split = run_weka_command(cmd_split)
            results["Percentage split"] = parse_correctly_classified_percentage(output_split)

            cmd_model_summary = java_cmd_base + [classifier_class, "-t", final_arff_file, "-c", "last", "-no-cv"] + classifier_params
            output_model_summary = run_weka_command(cmd_model_summary)
            results["Tree size"] = parse_tree_size(output_model_summary)

            results["Parameters"] = config['params'] if config['params'] else " "

            if results["Cross validation"] is None:
                print(f"FATAL: Evaluation failed for '{config['name']}'. Skipping."); continue

            accuracies = [acc for acc in [results["Cross validation"], results["Percentage split"]] if acc is not None]
            results["Vidurkis"] = sum(accuracies) / len(accuracies) if accuracies else np.nan
            all_results.append(results)

            family_name = config['name'].split('_')[0]
            if family_name in best_models_by_family:
                current_cv = results["Cross validation"]
                current_size = results["Tree size"] if not pd.isna(results["Tree size"]) else float('inf')
                best_so_far = best_models_by_family[family_name]

                if current_cv > best_so_far["cv_score"] or \
                   ('tree_size' in best_so_far and current_cv == best_so_far["cv_score"] and current_size < best_so_far["tree_size"]):
                    print(f"*** New best {family_name} model found: {config['name']} (CV: {current_cv}%) ***")
                    best_models_by_family[family_name] = {"config": config, "cv_score": current_cv, "tree_size": current_size}

        if not all_results:
            print(f"\nNo results generated for dataset '{prefix}'."); continue
        
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values(by=["Cross validation", "Tree size"], ascending=[False, True])
        
        print(f"\n\n--- Experiment Results Summary for '{prefix}' ---\n{results_df.set_index('Algorithm')}")
        results_df.to_csv(RESULTS_DIR / f"{prefix}_results.csv", index=False)
        print(f"Results saved to '{RESULTS_DIR / f'{prefix}_results.csv'}'")

        print("\n--- Generating final outputs (visualizations and rule files)... ---")
        
        for algo_name, default_config_params in DEFAULT_ALGORITHM_CONFIG.items():
            default_config = {"name": f"{algo_name} (Default)", **default_config_params}
            
            if "trees" in default_config['class']:
                generate_visualization_for_config(default_config, final_arff_file, java_cmd_base, f"{prefix}_{algo_name}_default")
                
                best_model = best_models_by_family.get(algo_name)
                if best_model and best_model.get("config"):
                    best_config = best_model["config"]
                    if best_config["params"] != default_config["params"]:
                        generate_visualization_for_config(best_config, final_arff_file, java_cmd_base, f"{prefix}_{algo_name}_optimised")
                    else:
                        print(f"--- Best {algo_name} model was the default. Skipping duplicate optimised visualization. ---")

            elif "rules" in default_config['class']:
                generate_rules_for_config(default_config, final_arff_file, java_cmd_base, f"{prefix}_{algo_name}_default_rules")
                
                best_model = best_models_by_family.get(algo_name)
                if best_model and best_model.get("config"):
                    best_config = best_model["config"]
                    if best_config["params"] != default_config["params"]:
                        generate_rules_for_config(best_config, final_arff_file, java_cmd_base, f"{prefix}_{algo_name}_optimised_rules")
                    else:
                        print(f"--- Best {algo_name} model was the default. Skipping duplicate optimised rules file. ---")
                
if __name__ == "__main__":
    main()
