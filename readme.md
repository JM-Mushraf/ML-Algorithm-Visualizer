->Create the that holds all the model(functions) with default HyperParameters and it 
->Create a file(function) for graph plotting


from sklearn.datasets import make_regression, make_blobs, make_classification
import numpy as np

def generate_dataset(dataset_type, n_samples=100): 
    if dataset_type == "linear":
        X, y = make_regression(n_samples=n_samples, n_features=1, noise=10)
    elif dataset_type == "random":
        X, y = make_blobs(n_samples=n_samples, centers=3, random_state=42)
    elif dataset_type == "exponential":
        X, y = make_regression(n_samples=n_samples, n_features=1, noise=10)
        y = np.exp(y)  # Transform to exponential data
    elif dataset_type == "classified":
        X, y = make_classification(n_samples=n_samples, n_features=2, n_classes=2, random_state=42)
    else:
        raise ValueError("Invalid dataset type. Choose from 'linear', 'random', 'exponential', or 'classified'.")
    
    return X, y

# Example usage:
dataset_type = "random"  # Change this to 'linear', 'exponential', or 'classified'
X, y = generate_dataset(dataset_type)

print(X[:5])  # Print first 5 data points
print(y[:5])


1.svm

3.knn-k
4.naive bais
5.boosting- xg boost