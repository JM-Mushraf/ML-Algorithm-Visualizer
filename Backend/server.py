from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import io
import base64
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from dataset import create_classification_dataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Ensure 'static/plots' directory exists
PLOT_DIR = "static/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def plot_decision_boundary(X, y, model, model_type, plot_path):
    """Plot the decision boundary of the classification model."""
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
    """Run the classification model and return metrics and plot."""
    X, y = create_classification_dataset(dataset_type=dataset_type, n_samples=sample_size, noise=noise)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize model with hyperparameters
    if model_type == "logistic":
        model = LogisticRegression(
            C=hyperparams.get("C", 1.0),  # Regularization strength
            penalty=hyperparams.get("penalty", "l2"),  # Regularization type
            solver=hyperparams.get("solver", "lbfgs"),  # Optimization algorithm
            max_iter=hyperparams.get("max_iter", 100)  # Maximum iterations
        )
    elif model_type == "dt":
        model = DecisionTreeClassifier(
            max_depth=hyperparams.get("max_depth", 5),  # Maximum depth of the tree
            min_samples_split=hyperparams.get("min_samples_split", 2),  # Minimum samples to split a node
            min_samples_leaf=hyperparams.get("min_samples_leaf", 1),  # Minimum samples at a leaf node
            max_features=hyperparams.get("max_features", None),  # Number of features to consider for splitting
            criterion=hyperparams.get("criterion", "gini")  # Splitting criterion
        )
    elif model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=hyperparams.get("n_estimators", 100),  # Number of trees in the forest
            max_depth=hyperparams.get("max_depth", None),  # Maximum depth of each tree
            min_samples_split=hyperparams.get("min_samples_split", 2),  # Minimum samples to split a node
            min_samples_leaf=hyperparams.get("min_samples_leaf", 1),  # Minimum samples at a leaf node
            max_features=hyperparams.get("max_features", "sqrt"),  # Number of features to consider for splitting
            criterion=hyperparams.get("criterion", "gini"),  # Splitting criterion
            random_state=42  # Random seed for reproducibility
        )
    elif model_type == "svm":
        model = SVC(
            kernel=hyperparams.get("kernel", "linear"),  # Kernel type
            C=hyperparams.get("C", 1.0),  # Regularization parameter
            gamma=hyperparams.get("gamma", "scale"),  # Kernel coefficient
            degree=hyperparams.get("degree", 3)  # Degree of polynomial kernel
        )
    else:
        raise ValueError("Invalid model type. Choose 'logistic', 'dt', 'rf', or 'svm'.")

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

@app.route('/classification', methods=['POST', 'OPTIONS'])
def classification():
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({'message': 'Preflight request received'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    # Handle the actual POST request
    data = request.json
    print("Incoming request payload:", data)  # Log the request payload

    try:
        accuracy, precision, recall, f1, conf_matrix, plot_filename = run_classification(
            model_type=data.get('model_type'),
            dataset_type=data.get('dataset_type'),
            sample_size=data.get('sample_size'),
            noise=data.get('noise'),
            **data.get('hyperparameters', {})
        )
        return jsonify({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': conf_matrix,
            'plot_filename': plot_filename
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/static/plots/<filename>')
def serve_plot(filename):
    """Serve the generated plot image."""
    return send_from_directory(PLOT_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True)