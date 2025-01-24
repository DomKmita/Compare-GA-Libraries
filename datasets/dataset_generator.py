from sklearn.datasets import make_classification
from scipy.sparse import csr_matrix
import pandas as pd
import numpy as np

# Function to generate datasets with specified parameters
# Make classification is used as it allows for greater control
# over the dataset
def generate_dataset(n_samples, n_features, n_classes, informative, redundant, weights,
                     separability, outliers, feature_type):
    if separability == 'linear':
        class_sep = 2.0  # Linearly separable
    elif separability == 'nonlinear':
        class_sep = 0.5 # Non-linearly separable
    else:
        class_sep = 1.0 # Sci-kit learn default

    if outliers == 'none':
        flip_y = 0.0
    elif outliers == 'moderate':
        flip_y = 0.05
    elif outliers == 'heavy':
        flip_y = 0.2
    else:
        flip_y = 0.1 # Sci-kit learn default

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=informative,
        n_redundant=redundant,
        n_classes=n_classes,
        weights=weights,
        flip_y=flip_y,
        class_sep=class_sep,
        random_state=932024
    )

    if feature_type == 'categorical':
        # Convert continuous features to categorical
        # Were simulating a one-hot encoding for categorical features by rounding continuous values
        X = np.round(X).astype(int)
    elif feature_type == 'mixed':
        # Mix continuous and categorical features
        X_cat = np.round(X[:, :n_features // 2]).astype(int)
        X_cont = X[:, n_features // 2:]
        X = np.hstack((X_cat, X_cont))

    return X, y

# A map defining the different parameters to be passed to the generate_dataset function
# The key is the label that will be appended to the filename. Level refers to the Taguchi method levels.
# I considered using nested maps for readability but the fact that this will only be used once doesn't justify their
# usage.
datasetTypeMap = {
    # default dataset to avoid repetition. This is the baseline. Each other dataset will change one of
    # these values only.
    "default_version": [10, 2, 10, 0, [0.5, 0.5], "linear", "none", "continuous"],

    "dimensions_level_1": [50, 2, 10, 0, [0.5, 0.5], "linear", "none", "continuous"],
    "dimensions_level_2": [100, 2, 10, 0, [0.5, 0.5], "linear", "none", "continuous"],

    "distribution_level_1": [10, 2, 10, 0, [0.75, 0.25], "linear", "none", "continuous"],
    "distribution_level_2": [10, 2, 10, 0, [0.9, 0.1], "linear", "none", "continuous"],

    # the weights for the class counts must also be adjusted to account for their increased count. Maintaining balanced
    # distribution as per default
    "class_count_level_1": [10, 5, 10, 0, [0.2, 0.2, 0.2, 0.2, 0.2], "linear", "none", "continuous"],
    "class_count_level_2": [10, 10, 10, 0, [0.1, 0.1 ,0.1, 0.1, 0.1, 0.1, 0.1 ,0.1, 0.1, 0.1], "linear", "none", "continuous"],

    "types_level_1": [10, 2, 10, 0, [0.5, 0.5], "linear", "none", "categorical"],
    "types_level_2": [10, 2, 10, 0, [0.5, 0.5], "linear", "none", "mixed"],

    "outliers_level_1": [10, 2, 10, 0, [0.5, 0.5], "linear", "moderate", "continuous"],
    "outliers_level_2": [10, 2, 10, 0, [0.5, 0.5], "linear", "heavy", "continuous"],

    "separability_level_1": [10, 2, 10, 0, [0.5, 0.5], "nonlinear", "none", "continuous"],

    "properties_level_1": [10, 2, 5, 5, [0.5, 0.5], "linear", "none", "continuous"],
    "properties_level_2": [10, 2, 5, 0, [0.5, 0.5], "linear", "none", "continuous"],
    "properties_level_3": [10, 2, 10, 0, [0.5, 0.5], "linear", "none", "continuous"], # this dataset is the same as
    # the default but will be transformed to be sparse. This can't be achieved while generating. All datasets are dense
    # by default.
    }

def main():
    for label, dataset_type in datasetTypeMap.items():
        # keeping sample size static for now. Once data is analysed I'll see what data properties to test for at larger
        # dataset sizes
        X, y = generate_dataset(10000, n_features = dataset_type[0], n_classes = dataset_type[1],
                                informative = dataset_type[2], redundant = dataset_type[3],weights = dataset_type[4],
                                separability = dataset_type[5], outliers = dataset_type[6], feature_type = dataset_type[7])

        # convert final dataset to be sparse. This can't be done during generation and therefore it must be transformed
        if label == "properties_level_3":
            X = csr_matrix(X)
            # Append features
            df = pd.DataFrame(X.toarray(), columns=[f"feature_{i}" for i in range(X.shape[1])])
        else:
            df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])

        # Append target classification
        df['target'] = y

        # Save to CSV
        df.to_csv(f"small_dataset_{label}.csv", index=False)
        print(f"Dataset_{label} saved as small_dataset_{label}.csv")

if __name__ == "__main__":
    main()