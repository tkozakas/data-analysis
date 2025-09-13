import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mutual_info_score
from ucimlrepo import fetch_ucirepo

def print_available_attributes(repo):
    print("Available Attributes and their Indices:")
    for i, col_name in enumerate(repo.data.features.columns):
        print(f"{i}: {col_name}")

def load_parkinsons_data(repo_id, col_names_to_select):
    parkinsons_repo = fetch_ucirepo(id=repo_id)
    X = parkinsons_repo.data.features
    y = parkinsons_repo.data.targets
    data = X[col_names_to_select].copy()
    data['status'] = y

    return data

def calculate_unsupervised_mi(X, y, attribute, n_bins, random_state):
    discretizer = KBinsDiscretizer(
        n_bins=n_bins, encode='ordinal', strategy='uniform', subsample=None, random_state=random_state
    )
    discretized_attribute = discretizer.fit_transform(X[[attribute]])
    return mutual_info_score(discretized_attribute.ravel(), y)


def calculate_supervised_mi(X, y, attribute, max_leaf_nodes, random_state, threshold_ignore_val):
    tree_discretizer = DecisionTreeClassifier(
        max_leaf_nodes=max_leaf_nodes, random_state=random_state
    )
    tree_discretizer.fit(X[[attribute]], y)

    thresholds = sorted([t for t in tree_discretizer.tree_.threshold if t != threshold_ignore_val])
    bins = [-np.inf] + thresholds + [np.inf]

    discretized_attribute = pd.cut(X[attribute], bins=bins, labels=False, include_lowest=True)
    num_bins_created = len(bins) - 1
    mi_score = mutual_info_score(discretized_attribute, y)

    return mi_score, num_bins_created


def generate_results_plot(results_df, plot_config):
    plt.style.use(plot_config['style'])
    fig, ax = plt.subplots(figsize=plot_config['figsize'])

    sns.barplot(
        data=results_df,
        x=plot_config['x_col'],
        y=plot_config['y_col'],
        hue=plot_config['hue_col'],
        ax=ax,
        palette=plot_config['palette']
    )

    ax.set_title(plot_config['title'], fontsize=16)
    ax.set_xlabel(plot_config['x_label'], fontsize=12)
    ax.set_ylabel(plot_config['y_label'], fontsize=12)
    ax.legend(title=plot_config['legend_title'], fontsize=10)
    plt.tight_layout()
    plt.show()


def main():
    UCI_REPO_ID = 174

    parkinsons_repo = fetch_ucirepo(id=UCI_REPO_ID)
    print_available_attributes(parkinsons_repo)

    COL_NAME_MDVP_FO = 'MDVP:Fo'
    COL_NAME_HNR = 'HNR'
    COL_NAME_STATUS = 'status'

    ATTRIBUTES_TO_TEST = [COL_NAME_MDVP_FO, COL_NAME_HNR]
    TARGET_COL = COL_NAME_STATUS

    RANDOM_STATE = 42
    MAX_LEAF_NODES = 5
    UNSUPERVISED_BINS = [2, 3, 5]
    TREE_THRESHOLD_IGNORE = -2

    RESULT_COL_ATTR = 'Atributas'
    RESULT_COL_METHOD = 'Metodas'
    RESULT_COL_MI = 'Tarpusavio Informacija'

    PLOT_CONFIG = {
        'style': 'seaborn-v0_8-whitegrid',
        'palette': 'viridis',
        'figsize': (14, 8),
        'title': 'Atributų informatyvumo palyginimas',
        'x_label': 'Tiriamas Atributas',
        'y_label': 'Tarpusavio Informacijos Koeficientas',
        'legend_title': 'Diskretizavimo Metodas',
        'x_col': RESULT_COL_ATTR,
        'y_col': RESULT_COL_MI,
        'hue_col': RESULT_COL_METHOD
    }

    data = load_parkinsons_data(UCI_REPO_ID, ATTRIBUTES_TO_TEST)

    print("Duomenų pavyzdys:")
    print(data.head())

    X = data[ATTRIBUTES_TO_TEST]
    y = data[TARGET_COL]
    results = []

    for attribute in ATTRIBUTES_TO_TEST:
        for n_bins in UNSUPERVISED_BINS:
            mi_score = calculate_unsupervised_mi(X, y, attribute, n_bins, RANDOM_STATE)
            results.append({
                RESULT_COL_ATTR: attribute,
                RESULT_COL_METHOD: f'Unsupervised ({n_bins} bins)',
                RESULT_COL_MI: mi_score
            })

    for attribute in ATTRIBUTES_TO_TEST:
        mi_score, num_bins = calculate_supervised_mi(
            X, y, attribute, MAX_LEAF_NODES, RANDOM_STATE, TREE_THRESHOLD_IGNORE
        )
        results.append({
            RESULT_COL_ATTR: attribute,
            RESULT_COL_METHOD: f'Supervised ({num_bins} bins)',
            RESULT_COL_MI: mi_score
        })

    results_df = pd.DataFrame(results)

    SEPARATOR = "\n" + "=" * 40 + "\n"
    print(SEPARATOR)
    print("Duomenų pavyzdys:")
    print(results_df.round(4))

    print(SEPARATOR)
    print("Generuojamas grafikas...")
    generate_results_plot(results_df, PLOT_CONFIG)


if __name__ == "__main__":
    main()
