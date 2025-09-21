import re
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

WEKA_JAR_PATH = Path("./weka.jar")
INPUT_DATA_FILE = Path("./parkinsons.data")
OUTPUT_DIR = Path("./weka_pipeline_output")

ATTRIBUTES_TO_ANALYZE = {"MDVP:Fo(Hz)": 2, "MDVP:Shimmer": 10}
CLASS_ATTRIBUTE_INDEX = 18
UNSUPERVISED_BINS = [2, 3, 5]


def create_arff_from_data(data_file_path, output_arff_path):
    print(f"Converting '{data_file_path}' to ARFF format at '{output_arff_path}'...")

    arff_header = """
@relation parkinsons

@attribute name string
@attribute 'MDVP:Fo(Hz)' numeric
@attribute 'MDVP:Fhi(Hz)' numeric
@attribute 'MDVP:Flo(Hz)' numeric
@attribute 'MDVP:Jitter(%)' numeric
@attribute 'MDVP:Jitter(Abs)' numeric
@attribute 'MDVP:RAP' numeric
@attribute 'MDVP:PPQ' numeric
@attribute 'Jitter:DDP' numeric
@attribute 'MDVP:Shimmer' numeric
@attribute 'MDVP:Shimmer(dB)' numeric
@attribute 'Shimmer:APQ3' numeric
@attribute 'Shimmer:APQ5' numeric
@attribute 'MDVP:APQ' numeric
@attribute 'Shimmer:DDA' numeric
@attribute NHR numeric
@attribute HNR numeric
@attribute status {0,1}
@attribute RPDE numeric
@attribute DFA numeric
@attribute spread1 numeric
@attribute spread2 numeric
@attribute D2 numeric
@attribute PPE numeric

@data
"""
    try:
        with open(data_file_path, 'r') as f_in:
            lines = f_in.readlines()

        data_content = "".join(lines[1:])

        with open(output_arff_path, 'w') as f_out:
            f_out.write(arff_header.strip() + "\n")
            f_out.write(data_content)

        print("Conversion successful.")
        return True
    except FileNotFoundError:
        print(f"Error: Input data file not found at '{data_file_path}'")
        return False


def run_weka_command(command_list):
    print(f"Executing: {' '.join(command_list)}")
    try:
        result = subprocess.run(
            command_list, check=False, capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"\nERROR executing Weka command. Return code: {result.returncode}")
            print(f"Stderr:\n{result.stderr}")
            return None

        return result
    except FileNotFoundError:
        print("\nERROR: 'java' command not found. Is Java installed and in your PATH?")
        return None


def parse_infogain_output(result_object, attributes_to_find):
    if not result_object:
        return {}
    full_output = result_object.stdout + "\n" + result_object.stderr

    scores = {}
    pattern = re.compile(r"^\s*([\d.]+)\s+\d+\s+(.*)$")
    target_attr_names = list(attributes_to_find.keys())

    for line in full_output.splitlines():
        match = pattern.match(line.strip())
        if match:
            score = float(match.group(1))
            found_attr_name = match.group(2).strip()

            if found_attr_name in target_attr_names:
                scores[found_attr_name] = score
    return scores


def main():
    if not WEKA_JAR_PATH.is_file():
        print(f"FATAL: weka.jar not found at '{WEKA_JAR_PATH}'. Exiting.")
        return
    if not INPUT_DATA_FILE.is_file():
        print(f"FATAL: {INPUT_DATA_FILE.name} not found at '{INPUT_DATA_FILE}'. Exiting.")
        return

    OUTPUT_DIR.mkdir(exist_ok=True)

    master_arff_file = OUTPUT_DIR / "parkinsons_master.arff"
    if not create_arff_from_data(INPUT_DATA_FILE, master_arff_file):
        return

    subset_arff = OUTPUT_DIR / "parkinsons_subset.arff"
    indices_to_keep = ",".join(map(str, list(ATTRIBUTES_TO_ANALYZE.values()) + [CLASS_ATTRIBUTE_INDEX]))
    cmd_subset = [
        "java", "-cp", str(WEKA_JAR_PATH),
        "weka.filters.unsupervised.attribute.Remove",
        "-V", "-R", indices_to_keep,
        "-i", str(master_arff_file), "-o", str(subset_arff)
    ]
    if not run_weka_command(cmd_subset): return

    all_results = []

    # Step 1: Run analysis for unsupervised methods
    for n_bins in UNSUPERVISED_BINS:
        discretized_file = OUTPUT_DIR / f"subset_unsupervised_{n_bins}bins.arff"
        cmd_discretize = [
            "java", "-cp", str(WEKA_JAR_PATH),
            "weka.filters.unsupervised.attribute.Discretize",
            "-B", str(n_bins), "-i", str(subset_arff), "-o", str(discretized_file)
        ]
        if not run_weka_command(cmd_discretize): continue

        cmd_evaluate = [
            "java", "-cp", str(WEKA_JAR_PATH),
            "weka.attributeSelection.InfoGainAttributeEval",
            "-s", "weka.attributeSelection.Ranker", "-i", str(discretized_file)
        ]
        result = run_weka_command(cmd_evaluate)
        if result:
            scores = parse_infogain_output(result, ATTRIBUTES_TO_ANALYZE)
            for attr_name, score in scores.items():
                all_results.append({
                    "Attribute": attr_name,
                    "Method": f"Unsupervised ({n_bins} bins)",
                    "Mutual_Info": score
                })

    # Step 2: Run analysis for supervised method
    discretized_file_sup = OUTPUT_DIR / "subset_supervised.arff"
    new_class_index = len(ATTRIBUTES_TO_ANALYZE) + 1
    cmd_discretize_sup = [
        "java", "-cp", str(WEKA_JAR_PATH),
        "weka.filters.supervised.attribute.Discretize",
        "-c", str(new_class_index), "-i", str(subset_arff), "-o", str(discretized_file_sup)
    ]
    if not run_weka_command(cmd_discretize_sup): return

    cmd_evaluate_sup = [
        "java", "-cp", str(WEKA_JAR_PATH),
        "weka.attributeSelection.InfoGainAttributeEval",
        "-s", "weka.attributeSelection.Ranker", "-i", str(discretized_file_sup)
    ]
    result_sup = run_weka_command(cmd_evaluate_sup)
    if result_sup:
        scores = parse_infogain_output(result_sup, ATTRIBUTES_TO_ANALYZE)
        for attr_name, score in scores.items():
            all_results.append({
                "Attribute": attr_name,
                "Method": "Supervised",
                "Mutual_Info": score
            })

    # Step 3: Display results
    if not all_results:
        print("\nAnalysis failed to produce results.")
        return

    results_df = pd.DataFrame(all_results)
    print("\n--- Final Results ---")
    print(results_df.round(4))

    # Step 4: Generate plot
    print("\nGenerating and saving the plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(
        data=results_df, x='Attribute', y='Mutual_Info', hue='Method', palette='viridis'
    )
    ax.set_title('Attribute Mutual Information using Weka', fontsize=16)
    ax.set_xlabel('Attribute', fontsize=12)
    ax.set_ylabel('Mutual Information (InfoGain Score)', fontsize=12)
    plt.tight_layout()
    plt.savefig("weka_analysis_results.png", dpi=300, bbox_inches='tight')
    print("Plot saved as 'weka_analysis_results.png'")


if __name__ == "__main__":
    main()
