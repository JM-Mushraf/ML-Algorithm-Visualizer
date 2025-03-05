import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-GUI backend before importing pyplot
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dataset import create_classification_dataset  # Import your dataset function

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
    Run the classification model and return accuracy and plot.

    Parameters:
    - model_type: Type of model. Options: 'logistic', 'dt', 'rf', 'svm'.
    - dataset_type: Type of dataset. Options: 'linear', 'moons', 'circles'.
    - sample_size: Number of samples in the dataset.
    - noise: Amount of noise in the dataset.
    - hyperparams: Additional hyperparameters for the model.

    Returns:
    - accuracy: Accuracy score of the model.
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
            max_depth=hyperparams.get("max_depth", 5),  # Maximum depth of the tree
            min_samples_split=hyperparams.get("min_samples_split", 2),  # Minimum samples to split a node
            min_samples_leaf=hyperparams.get("min_samples_leaf", 1),  # Minimum samples at a leaf node
            max_features=hyperparams.get("max_features", "auto"),  # Number of features to consider for splitting
            criterion=hyperparams.get("criterion", "gini")  # Splitting criterion ('gini' or 'entropy')
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
    else:
        raise ValueError("Invalid model type. Choose 'logistic', 'dt', 'rf', or 'svm'.")

    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Generate and save plot
    plot_filename = f"{model_type}_classification.png"
    plot_path = os.path.join(PLOT_DIR, plot_filename)
    plot_decision_boundary(X_test, y_test, model, model_type, plot_path)

    return accuracy, plot_filename

#################### USE THE BELOW CODE WHEN UR INTEGRATING FRONTEND, CURRENT CODE IS JUST TO SAVE THE GRAPH TO CHECK API'S

# import numpy as np
# import matplotlib.pyplot as plt
# import io
# import base64
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from dataset import create_classification_dataset

# def plot_decision_boundary(X, y, model, model_type):
#     """Generate a decision boundary plot and return it as a Base64 string."""
#     plt.figure(figsize=(8, 6))

#     # Create a meshgrid to plot the decision boundary
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
#                          np.arange(y_min, y_max, 0.01))

#     # Predict for each point in the meshgrid
#     Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
#     Z = Z.reshape(xx.shape)

#     # Plot the decision boundary
#     plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
#     plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=plt.cm.Paired)
#     plt.title(f"{model_type} Decision Boundary")
#     plt.xlabel("Feature 1")
#     plt.ylabel("Feature 2")
#     plt.colorbar(label='Class')

#     # Save plot to a memory buffer
#     img_buffer = io.BytesIO()
#     plt.savefig(img_buffer, format="png")
#     plt.close()
    
#     # Encode to Base64
#     img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    
#     return img_base64

# def run_classification(model_type="logistic", dataset_type="linear", sample_size=200, noise=0.1, **hyperparams):
#     """
#     Run the classification model and return accuracy and the Base64 image.

#     Parameters:
#     - model_type: Type of model. Options: 'logistic', 'dt', 'rf', 'svm'.
#     - dataset_type: Type of dataset. Options: 'linear', 'moons', 'circles'.
#     - sample_size: Number of samples in the dataset.
#     - noise: Amount of noise in the dataset.
#     - hyperparams: Additional hyperparameters for the model.

#     Returns:
#     - accuracy: Accuracy score of the model.
#     - plot_base64: Base64-encoded image of the decision boundary plot.
#     """
#     # Generate dataset using your existing function
#     X, y = create_classification_dataset(dataset_type=dataset_type, n_samples=sample_size, noise=noise)
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#     # Initialize model
#     if model_type == "logistic":
#         model = LogisticRegression()
#     elif model_type == "dt":
#         max_depth = hyperparams.get("max_depth", 5)
#         model = DecisionTreeClassifier(max_depth=max_depth)
#     elif model_type == "rf":
#         n_estimators = hyperparams.get("n_estimators", 100)
#         model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
#     elif model_type == "svm":
#         kernel = hyperparams.get("kernel", "linear")
#         model = SVC(kernel=kernel)
#     else:
#         raise ValueError("Invalid model type. Choose 'logistic', 'dt', 'rf', or 'svm'.")

#     # Train model
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)

#     # Generate Base64 plot
#     plot_base64 = plot_decision_boundary(X_test, y_test, model, model_type)

#     return accuracy, plot_base64