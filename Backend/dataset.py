import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification, make_moons, make_circles
def generate_dataset(dataset_type="linear", sample_size=300):
    np.random.seed(42)
    X = np.linspace(-3, 3, sample_size).reshape(-1, 1)
    
    if dataset_type == "linear":
        y = 2 * X.squeeze() + np.random.normal(0, 1, sample_size)
    elif dataset_type == "u_shaped":
        y = X.squeeze() ** 2 + np.random.normal(0, 1, sample_size)
    elif dataset_type == "concentric":
        y = np.sin(X.squeeze()) + np.random.normal(0, 0.2, sample_size)
    else:
        raise ValueError("Invalid dataset type. Choose 'linear', 'u_shaped', or 'concentric'.")
    
    return train_test_split(X, y, test_size=0.2, random_state=42)


# def create_classification_dataset(dataset_type='linear', n_samples=200, noise=0.1, random_state=42):
#     """
#     Create a synthetic classification dataset of a specified type.

#     Parameters:
#     - dataset_type: Type of dataset to create. Options: 'linear', 'moons', 'circles', 'noisy_linear'.
#     - n_samples: Number of samples in the dataset.
#     - noise: Amount of noise to add to the dataset.
#     - random_state: Random seed for reproducibility.

#     Returns:
#     - X: Feature matrix (n_samples, 2).
#     - y: Target labels (n_samples,).
#     """
#     if dataset_type == 'linear':
#         # Linear classification dataset
#         X, y = make_classification(
#             n_samples=n_samples,
#             n_features=2,
#             n_informative=2,
#             n_redundant=0,
#             n_clusters_per_class=1,
#             random_state=random_state
#         )
#     elif dataset_type == 'moons':
#         # U-shaped (moons) dataset
#         X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
#     elif dataset_type == 'circles':
#         # Circular dataset
#         X, y = make_circles(n_samples=n_samples, noise=noise, random_state=random_state)
#     elif dataset_type == 'noisy_linear':
#         # Linear dataset with added noise
#         X, y = make_classification(
#             n_samples=n_samples,
#             n_features=2,
#             n_informative=2,
#             n_redundant=0,
#             n_clusters_per_class=1,
#             random_state=random_state
#         )
#         # Add extra noise
#         X += np.random.normal(scale=noise * 5, size=X.shape)
#     else:
#         raise ValueError(f"Unknown dataset type: {dataset_type}. Choose from 'linear', 'moons', 'circles', 'noisy_linear'.")

#     return train_test_split(X, y, test_size=0.2, random_state=42)
# from sklearn.datasets import make_classification, make_moons, make_circles

def create_classification_dataset(dataset_type='linear', n_samples=200, noise=0.1, random_state=42):
    """
    Create a synthetic classification dataset of a specified type.

    Parameters:
    - dataset_type: Type of dataset to create. Options: 'linear', 'moons', 'circles'.
    - n_samples: Number of samples in the dataset.
    - noise: Amount of noise to add to the dataset.
    - random_state: Random seed for reproducibility.

    Returns:
    - X: Feature matrix (n_samples, 2).
    - y: Target labels (n_samples,).
    """
    if dataset_type == 'linear':
        # Linear classification dataset
        X, y = make_classification(
            n_samples=n_samples,
            n_features=2,
            n_informative=2,
            n_redundant=0,
            n_clusters_per_class=1,
            random_state=random_state
        )
    elif dataset_type == 'moons':
        # U-shaped (moons) dataset
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif dataset_type == 'circles':
        # Circular dataset
        X, y = make_circles(n_samples=n_samples, noise=noise, random_state=random_state)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}. Choose from 'linear', 'moons', 'circles'.")

    return X, y