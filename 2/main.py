# literally to extract arrf file from the repo

from ucimlrepo import fetch_ucirepo

# fetch dataset
chronic_kidney_disease = fetch_ucirepo(id=336)

# data (as pandas dataframes)
X = chronic_kidney_disease.data.features
y = chronic_kidney_disease.data.targets

# metadata
print(chronic_kidney_disease.metadata)

# variable information
print(chronic_kidney_disease.variables)
import pandas as pd

# Combine features and target into a single dataframe
data = pd.concat([X, y], axis=1)

# Save to CSV
data.to_csv("chronic_kidney_disease.csv", index=False)
import pandas as pd

# Combine features and target into a single dataframe
data = pd.concat([X, y], axis=1)

X.to_csv("chronic_kidney_disease_features.csv", index=False)

# Save target
y.to_csv("chronic_kidney_disease_target.csv", index=False)