from sklearn.datasets import make_classification
import pandas as pd

# Function to generate datasets with specified parameters
# Make classification is used as it allows for greateer control
# over the dataset
def generate_dataset(n_samples, n_features, n_classes, informative, redundant,
                     separability, outliers):
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
        flip_y=flip_y,
        class_sep=class_sep,
        random_state=932024
    )

    return X, y

def main():
    X, y = generate_dataset(10000, 10, 2, 10, 0, 2, "none");

    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    df['target'] = y

    # Save to CSV
    df.to_csv(f"small_dataset_{1}.csv", index=False)
    print(f"Dataset {1} saved as small_dataset_{1}.csv")

if __name__ == "__main__":
    main()