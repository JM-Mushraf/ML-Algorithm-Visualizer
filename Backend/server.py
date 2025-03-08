from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import numpy as np
import io
import base64
import time  # Import time module to measure execution time
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from dataset import generate_dataset
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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

@app.route('/regression', methods=['POST'])
def regression():
    data = request.json
    model_type = data.get('model_type')
    dataset_type = data.get('dataset_type')
    sample_size = data.get('sample_size')
    hyperparameters = data.get('hyperparameters', {})

    try:
        r2_score, y_pred, plot_base64, n_iterations, execution_time = run_regression(
            model_type=model_type,
            dataset_type=dataset_type,
            sample_size=sample_size,
            **hyperparameters
        )
        return jsonify({
            'r2_score': r2_score,
            'plot_base64': plot_base64,
            'n_iterations': n_iterations,
            'execution_time': round(execution_time, 4)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)