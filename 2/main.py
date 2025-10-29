import subprocess
import re
import pandas as pd
from pathlib import Path
from typing import Dict, Any

WEKA_JAR_PATH = Path("weka.jar")
DATA_DIR = Path("discretization")

ALGORITHM_CONFIG = {
    "J48": {
        "class": "weka.classifiers.trees.J48",
        "params": ""
    },
    "RandomTree": {
        "class": "weka.classifiers.trees.RandomTree",
        "params": ""
    },
    "REPTree": {
        "class": "weka.classifiers.trees.REPTree",
        "params": ""
    }
}


def run_weka_command(command_list):
    print(f"Executing: {' '.join(command_list)}")
    try:
        result = subprocess.run(
            command_list,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        if result.returncode != 0:
            print(f"--- WEKA ERROR DETECTED ---")
            print(result.stderr)
            print(f"--------------------------")
            return None
        return result.stdout
    except FileNotFoundError:
        print("\nERROR: 'java' command not found. Is Java installed and in your PATH?")
        return None


def parse_correctly_classified_line(weka_output):
    if not weka_output:
        return "WEKA process failed"

    match = re.search(r"^\s*Correctly Classified Instances.*$", weka_output, re.MULTILINE)
    if match:
        return match.group(0).replace("Correctly Classified Instances", "").strip()

    return "Evaluation summary not found"


def main():
    if not WEKA_JAR_PATH.is_file():
        print(f"FATAL: weka.jar not found at '{WEKA_JAR_PATH}'. Exiting.")
        return

    main_file = DATA_DIR / "original_sup.arff"
    train_file = DATA_DIR / "original_sup_train.arff"
    test_file = DATA_DIR / "original_sup_test.arff"

    for f in [main_file, train_file, test_file]:
        if not f.is_file():
            print(f"FATAL: Required data file not found at '{f}'. Exiting.")
            return

    all_results = []

    for algo_name, config in ALGORITHM_CONFIG.items():
        print(f"\n--- Running experiments for {algo_name} ---")
        results: Dict[str, Any] = {"Algorithm": algo_name}
        classifier_class = config['class']
        classifier_params = config['params'].split()

        cmd_cv = ["java", "-cp", str(WEKA_JAR_PATH), classifier_class, "-t", str(main_file), "-x",
                  "10"] + classifier_params
        output_cv = run_weka_command(cmd_cv)
        results["Cross-validation"] = parse_correctly_classified_line(output_cv)

        cmd_test = ["java", "-cp", str(WEKA_JAR_PATH), classifier_class, "-t", str(train_file), "-T",
                    str(test_file)] + classifier_params
        output_test = run_weka_command(cmd_test)
        result_line_test = parse_correctly_classified_line(output_test)
        results["Test / Training"] = result_line_test

        if result_line_test == "Evaluation summary not found" and output_test is not None:
            print(f"--- DEBUG: RAW WEKA OUTPUT FOR FAILED 'Test / Training' ---")
            print(output_test)
            print(f"----------------------------------------------------------")

        cmd_train = ["java", "-cp", str(WEKA_JAR_PATH), classifier_class, "-t", str(main_file)] + classifier_params
        output_train = run_weka_command(cmd_train)
        results["Training"] = parse_correctly_classified_line(output_train)

        cmd_split = ["java", "-cp", str(WEKA_JAR_PATH), classifier_class, "-t", str(main_file), "-split-percentage",
                     "66"] + classifier_params
        output_split = run_weka_command(cmd_split)
        results["Percentage split"] = parse_correctly_classified_line(output_split)

        all_results.append(results)

    if not all_results:
        print("\nNo results were generated.")
        return

    results_df = pd.DataFrame(all_results).set_index("Algorithm").T

    pd.set_option('display.max_colwidth', None)

    print("\n\n--- Experiment Results Summary ---")
    print(results_df)


if __name__ == "__main__":
    main()