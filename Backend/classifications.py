import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from dataset import create_classification_dataset

# Ensure 'static/plots' directory exists
PLOT_DIR = "static/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def plot_decision_boundary(X, y, model, model_type, plot_path):
    """
    Plot the decision boundary of the classification model.

    Parameters:
    - X: Feature matrix (n_samples, 2).
    - y: Target labels (n_samples,).
    - model: Trained classification model.
    - model_type: Type of model (e.g., 'logistic', 'dt', 'rf', 'svm').
    - plot_path: Path to save the plot.
    """
    plt.figure(figsize=(8, 6))

    # Create a meshgrid to plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Predict for each point in the meshgrid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=plt.cm.Paired)
    plt.title(f"{model_type} Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label='Class')

    # Save the plot
    plt.savefig(plot_path)
    plt.close()

def run_classification(model_type="logistic", dataset_type="linear", sample_size=200, noise=0.1, **hyperparams):
    """
    Run the classification model and return accuracy, precision, recall, F1 score, confusion matrix, and plot.

    Parameters:
    - model_type: Type of model. Options: 'logistic', 'dt', 'rf', 'svm', 'nb', 'knn'.
    - dataset_type: Type of dataset. Options: 'linear', 'moons', 'circles'.
    - sample_size: Number of samples in the dataset.
    - noise: Amount of noise in the dataset.
    - hyperparams: Additional hyperparameters for the model.

    Returns:
    - accuracy: Accuracy score of the model.
    - precision: Precision score of the model.
    - recall: Recall score of the model.
    - f1: F1 score of the model.
    - confusion_matrix: Confusion matrix as a list of lists.
    - plot_filename: Filename of the saved plot.
    """
    # Generate dataset using your existing function
    X, y = create_classification_dataset(dataset_type=dataset_type, n_samples=sample_size, noise=noise)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize model with hyperparameters
    if model_type == "logistic":
        model = LogisticRegression(
            C=hyperparams.get("C", 1.0),  # Regularization strength (inverse of lambda)
            penalty=hyperparams.get("penalty", "l2"),  # Regularization type ('l1', 'l2', 'elasticnet')
            solver=hyperparams.get("solver", "lbfgs"),  # Optimization algorithm ('lbfgs', 'liblinear', 'newton-cg')
            max_iter=hyperparams.get("max_iter", 100)  # Maximum number of iterations
        )
    elif model_type == "dt":
        model = DecisionTreeClassifier(
            max_depth=hyperparams.get("max_depth", 5),  
            min_samples_split=hyperparams.get("min_samples_split", 2),  
            min_samples_leaf=hyperparams.get("min_samples_leaf", 1), 
            max_features=hyperparams.get("max_features", "auto"),  
            criterion=hyperparams.get("criterion", "gini") 
        )
    elif model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=hyperparams.get("n_estimators", 100),  # Number of trees in the forest
            max_depth=hyperparams.get("max_depth", None),  # Maximum depth of each tree
            min_samples_split=hyperparams.get("min_samples_split", 2),  # Minimum samples to split a node
            min_samples_leaf=hyperparams.get("min_samples_leaf", 1),  # Minimum samples at a leaf node
            max_features=hyperparams.get("max_features", "auto"),  # Number of features to consider for splitting
            criterion=hyperparams.get("criterion", "gini"),  # Splitting criterion ('gini' or 'entropy')
            random_state=42  # Random seed for reproducibility
        )
    elif model_type == "svm":
        model = SVC(
            kernel=hyperparams.get("kernel", "linear"),  # Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            C=hyperparams.get("C", 1.0),  # Regularization parameter
            gamma=hyperparams.get("gamma", "scale"),  # Kernel coefficient ('scale', 'auto', or float)
            degree=hyperparams.get("degree", 3)  # Degree of polynomial kernel (only for 'poly' kernel)
        )
    elif model_type == "nb":
        model = GaussianNB(
            var_smoothing=hyperparams.get("var_smoothing", 1e-9)  # Smoothing parameter for variance
        )
    elif model_type == "knn":
        model = KNeighborsClassifier(
            n_neighbors=hyperparams.get("n_neighbors", 5),  # Number of neighbors
            weights=hyperparams.get("weights", "uniform"),  # Weight function ('uniform', 'distance')
            algorithm=hyperparams.get("algorithm", "auto"),  # Algorithm used ('auto', 'ball_tree', 'kd_tree', 'brute')
            leaf_size=hyperparams.get("leaf_size", 30),  # Leaf size for tree-based algorithms
            p=hyperparams.get("p", 2)  # Power parameter for Minkowski distance (1 for Manhattan, 2 for Euclidean)
        )
    else:
        raise ValueError("Invalid model type. Choose 'logistic', 'dt', 'rf', 'svm', 'nb', or 'knn'.")

    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred).tolist()

    # Generate and save plot
    plot_filename = f"{model_type}_classification.png"
    plot_path = os.path.join(PLOT_DIR, plot_filename)
    plot_decision_boundary(X_test, y_test, model, model_type, plot_path)

    return accuracy, precision, recall, f1, conf_matrix, plot_filename