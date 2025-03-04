import numpy as np
from sklearn.model_selection import train_test_split

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
