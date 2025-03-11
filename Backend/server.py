from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import os
import io
import base64
import time
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from flask import send_from_directory
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from dataset import create_classification_dataset, generate_dataset

# Set up upload and plot directories
UPLOAD_FOLDER = "uploads"
PLOT_DIR = "static/plots"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Helper function to plot decision boundaries for classification models
def plot_decision_boundary(X, y, model, model_type, plot_path):
    """Plot the decision boundary of a classification model."""
    plt.figure(figsize=(8, 6))
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', cmap=plt.cm.Paired)
    plt.title(f"{model_type} Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.colorbar(label='Class')
    plt.savefig(plot_path)
    plt.close()

# Classification model runner
def run_classification(model_type="logistic", dataset_type="linear", sample_size=200, noise=0.1, **hyperparams):
    """Run classification model and return metrics and plot filename."""
    X, y = create_classification_dataset(dataset_type=dataset_type, n_samples=sample_size, noise=noise)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    models = {
        "logistic": LogisticRegression(C=hyperparams.get("C", 1.0), max_iter=hyperparams.get("max_iter", 100)),
        "dt": DecisionTreeClassifier(max_depth=hyperparams.get("max_depth", 5)),
        "rf": RandomForestClassifier(n_estimators=hyperparams.get("n_estimators", 100), random_state=42),
        "svm": SVC(kernel=hyperparams.get("kernel", "linear"), C=hyperparams.get("C", 1.0))
    }

    if model_type not in models:
        raise ValueError("Invalid model type. Choose 'logistic', 'dt', 'rf', or 'svm'.")

    model = models[model_type]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1": f1_score(y_test, y_pred, average='weighted'),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    plot_filename = f"{model_type}_classification.png"
    plot_path = os.path.join(PLOT_DIR, plot_filename)
    plot_decision_boundary(X_test, y_test, model, model_type, plot_path)

    return metrics, plot_filename

# Helper function to plot regression results
def plot_results(X_test, y_test, y_pred, model_type):
    """Generate a regression plot and return it as a Base64 string."""
    plt.figure(figsize=(6, 4))
    plt.scatter(X_test, y_test, color="blue", label="Actual Data", alpha=0.6)

    # Sort data for a smoother regression line
    sorted_idx = np.argsort(X_test.squeeze())
    X_sorted, y_sorted = X_test[sorted_idx], y_pred[sorted_idx]
    
    plt.plot(X_sorted, y_sorted, color="red", label=f"Best Fit ({model_type})")
    plt.title(f"{model_type} Regression")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()

    # Save plot to a memory buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    plt.close()
    
    # Encode to Base64
    img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    
    return img_base64

# Regression model runner
def run_regression(model_type="linear", dataset_type="linear", sample_size=300, **hyperparams):
    """Runs the regression model and returns R² score, execution time, and the Base64 image."""
    X_train, X_test, y_train, y_test = generate_dataset(dataset_type, sample_size)

    # Start measuring execution time
    start_time = time.time()

    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "polynomial":
        degree = hyperparams.get("degree", 2)
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    elif model_type == "dt":
        max_depth = hyperparams.get("max_depth", 5)
        model = DecisionTreeRegressor(max_depth=max_depth)
    elif model_type == "rf":
        n_estimators = hyperparams.get("n_estimators", 100)
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    else:
        raise ValueError("Invalid model type. Choose 'linear', 'polynomial', 'dt', or 'rf'.")

    # Fit the model
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Calculate R² score
    r2_score = model.score(X_test, y_test)

    # Calculate execution time
    execution_time = time.time() - start_time

    # Generate Base64 plot
    plot_base64 = plot_results(X_test, y_test, y_pred, model_type)

    # Number of iterations (for iterative algorithms like Random Forest)
    if model_type == "rf":
        n_iterations = n_estimators  # Number of trees in Random Forest
    else:
        n_iterations = 1  # Non-iterative models like Linear Regression

    return r2_score, y_pred, plot_base64, n_iterations, execution_time

# File upload and analysis
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    
    # Analyze the file and generate visualizations
    analysis_results = analyze_file(file_path)
    
    return jsonify({"message": "File uploaded successfully", "file_path": file_path, "analysis": analysis_results})

def analyze_file(file_path):
    # Load the file into a pandas DataFrame
    df = pd.read_csv(file_path)  # Assuming the file is a CSV, adjust accordingly for other formats
    
    # Basic information about the data
    info = {
        "num_rows": df.shape[0],
        "num_columns": df.shape[1],
        "columns": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict(),
        "dataset_preview": df.head().to_dict()  # First 5 rows of the dataset
    }
    
    # Generate heatmap (only for numeric data)
    visualization_paths = generate_heatmap(df)
    
    # Find the best algorithm, its accuracy, and best features
    best_algorithm, accuracy, best_features = find_best_algorithm(df)
    
    return {
        "info": info,
        "visualization_paths": visualization_paths,
        "best_algorithm": best_algorithm,
        "accuracy": accuracy,
        "best_features": best_features
    }

def generate_heatmap(df):
    visualization_paths = {}
    
    # Select only numeric columns for the heatmap
    numeric_df = df.select_dtypes(include=['number'])
    
    # Check if there are at least two numeric columns to compute correlation
    if numeric_df.shape[1] > 1:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
        plt.title("Correlation Heatmap")
        heatmap_path = os.path.join(UPLOAD_FOLDER, "heatmap.png")
        plt.savefig(heatmap_path)
        plt.close()
        visualization_paths["heatmap"] = "uploads/heatmap.png"
    else:
        visualization_paths["heatmap"] = "Not enough numeric columns to generate a heatmap."
    
    return visualization_paths

def find_best_algorithm(df):
    # Assuming the last column is the target variable
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Encode categorical columns in X
    X = pd.get_dummies(X)
    
    # Encode the target variable if it's categorical
    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models and parameters for GridSearchCV
    models = {
        "RandomForest": RandomForestClassifier(),
        "SVM": SVC(),
        "LogisticRegression": LogisticRegression()
    }
    
    params = {
        "RandomForest": {"n_estimators": [10, 50, 100]},
        "SVM": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]},
        "LogisticRegression": {"C": [0.1, 1, 10]}
    }
    
    best_algorithm = None
    best_score = 0
    best_accuracy = 0
    best_features = None
    
    # Perform GridSearchCV for each model
    for model_name, model in models.items():
        grid_search = GridSearchCV(model, params[model_name], cv=5)
        grid_search.fit(X_train, y_train)
        score = grid_search.best_score_
        
        if score > best_score:
            best_score = score
            best_algorithm = model_name
            best_accuracy = accuracy_score(y_test, grid_search.predict(X_test))
            
            # Get feature importance or coefficients
            if model_name == "RandomForest":
                best_features = dict(zip(X.columns, grid_search.best_estimator_.feature_importances_))
            elif model_name == "LogisticRegression":
                best_features = dict(zip(X.columns, grid_search.best_estimator_.coef_[0]))
            else:
                best_features = "Feature importance not available for this model."
    
    return best_algorithm, best_accuracy, best_features

# Classification endpoint
@app.route('/classification', methods=['POST', 'OPTIONS'])
def classification():
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'Preflight request received'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    data = request.json
    try:
        metrics, plot_filename = run_classification(
            model_type=data.get('model_type'),
            dataset_type=data.get('dataset_type'),
            sample_size=data.get('sample_size', 200),
            noise=data.get('noise', 0.1),
            **data.get('hyperparameters', {})
        )
        return jsonify({**metrics, 'plot_filename': plot_filename})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/regression', methods=['POST', 'OPTIONS'])
def regression():
    if request.method == 'OPTIONS':
        response = jsonify({'message': 'Preflight request received'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response

    data = request.json
    try:
        r2_score, y_pred, plot_base64, n_iterations, execution_time = run_regression(
            model_type=data.get('model_type'),
            dataset_type=data.get('dataset_type'),
            sample_size=data.get('sample_size', 300),
            **data.get('hyperparameters', {})
        )
        return jsonify({
            'r2_score': r2_score,
            'plot_base64': plot_base64,
            'n_iterations': n_iterations,
            'execution_time': round(execution_time, 4)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Serve static plots
@app.route('/static/plots/<filename>')
def serve_plot(filename):
    return send_from_directory(PLOT_DIR, filename)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)