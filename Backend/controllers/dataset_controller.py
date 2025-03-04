import pandas as pd
import seaborn as sns
from sklearn.datasets import (
    load_diabetes,
    load_digits,
    load_iris,
    load_breast_cancer,
    make_circles,
    make_moons,
    make_blobs,
)
from sklearn.preprocessing import PolynomialFeatures
def get_builtin_dataset(dataset_name):
    """Fetch the dataset from built-in libraries."""
    try:
        if dataset_name == "Linear Regression":
            data = load_diabetes(as_frame=True).frame

        elif dataset_name == "U-Shape (Polynomial Regression)":
            data = load_diabetes(as_frame=True).frame
            poly = PolynomialFeatures(degree=2)
            poly_features = poly.fit_transform(data[["bmi"]])
            df = pd.DataFrame(poly_features, columns=[f"Poly_{i}" for i in range(poly_features.shape[1])])
            df["target"] = data["target"]
            data = df

        elif dataset_name == "Circular":
            X, y = make_circles(n_samples=300, noise=0.1, factor=0.5)
            data = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "Class": y})

        elif dataset_name == "Moons":
            X, y = make_moons(n_samples=300, noise=0.2)
            data = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "Class": y})

        elif dataset_name == "Gaussian Blobs":
            X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.0)
            data = pd.DataFrame({"X1": X[:, 0], "X2": X[:, 1], "Cluster": y})

        elif dataset_name == "High-Dimensional":
            data = load_digits(as_frame=True).frame  

        elif dataset_name == "Time-Series":
            data = sns.load_dataset("flights") 

        elif dataset_name == "Sparse":
            data = load_breast_cancer(as_frame=True).frame

        elif dataset_name == "Binary Classification":
            data = load_breast_cancer(as_frame=True).frame  

        elif dataset_name == "Multi-Class Classification":
            data = load_iris(as_frame=True).frame 

        else:
            return {"error": "Invalid dataset name."}

        return {"message": f"Dataset '{dataset_name}' loaded successfully!", "data": data.to_dict(orient="records")}

    except Exception as e:
        return {"error": str(e)}