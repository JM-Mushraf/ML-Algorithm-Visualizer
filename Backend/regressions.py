import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from dataset import generate_dataset

# Ensure 'static/plots' directory exists
PLOT_DIR = "static/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

def plot_results(X_test, y_test, y_pred, model_type):
    """Plot results and save as an image."""
    plt.figure(figsize=(6, 4))
    plt.scatter(X_test, y_test, color="blue", label="Actual Data", alpha=0.6)

    # Sort data for better visualization of best fit line
    sorted_idx = np.argsort(X_test.squeeze())
    X_sorted, y_sorted = X_test[sorted_idx], y_pred[sorted_idx]
    
    plt.plot(X_sorted, y_sorted, color="red", label=f"Best Fit ({model_type})")
    plt.title(f"{model_type} Regression")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()

    # Save the plot
    plot_filename = f"{model_type}_regression.png"
    plot_path = os.path.join(PLOT_DIR, plot_filename)
    plt.savefig(plot_path)
    plt.close()
    
    return plot_filename

def run_regression(model_type="linear", dataset_type="linear", sample_size=300, **hyperparams):
    """Runs the regression model and returns R² score and plot."""
    X_train, X_test, y_train, y_test = generate_dataset(dataset_type, sample_size)

    if model_type == "linear":
        model = LinearRegression()
    elif model_type == "polynomial":
        degree = hyperparams.get("degree", 2)
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    elif model_type == "dt":
        model = DecisionTreeRegressor(
            max_depth=hyperparams.get("max_depth", 5),
            min_samples_split=hyperparams.get("min_samples_split", 2),
            min_samples_leaf=hyperparams.get("min_samples_leaf", 1),
            max_features=hyperparams.get("max_features", None)  # Use None or "sqrt"
        )
    elif model_type == "rf":
        model = RandomForestRegressor(
            n_estimators=hyperparams.get("n_estimators", 100),
            max_depth=hyperparams.get("max_depth", None),
            min_samples_split=hyperparams.get("min_samples_split", 2),
            min_samples_leaf=hyperparams.get("min_samples_leaf", 1),
            max_features=hyperparams.get("max_features", "sqrt"),  # Use "sqrt" or another valid value
            random_state=42
        )
    else:
        raise ValueError("Invalid model type. Choose 'linear', 'polynomial', 'dt', or 'rf'.")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2_score = model.score(X_test, y_test)

    # Generate and save plot
    plot_filename = plot_results(X_test, y_test, y_pred, model_type)

    return r2_score, y_pred, plot_filename

#################### USE THE BELOW CODE WHEN UR INTEGRATING FRONTEND, CURRENT CODE IS JUST TO SAVE THE GRAPH TO CHECK API'S


# import numpy as np
# import matplotlib.pyplot as plt
# import io
# import base64
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import make_pipeline
# from dataset import generate_dataset

# def plot_results(X_test, y_test, y_pred, model_type):
#     """Generate a regression plot and return it as a Base64 string."""
#     plt.figure(figsize=(6, 4))
#     plt.scatter(X_test, y_test, color="blue", label="Actual Data", alpha=0.6)

#     # Sort data for a smoother regression line
#     sorted_idx = np.argsort(X_test.squeeze())
#     X_sorted, y_sorted = X_test[sorted_idx], y_pred[sorted_idx]
    
#     plt.plot(X_sorted, y_sorted, color="red", label=f"Best Fit ({model_type})")
#     plt.title(f"{model_type} Regression")
#     plt.xlabel("X")
#     plt.ylabel("y")
#     plt.legend()

#     # Save plot to a memory buffer
#     img_buffer = io.BytesIO()
#     plt.savefig(img_buffer, format="png")
#     plt.close()
    
#     # Encode to Base64
#     img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    
#     return img_base64

# def run_regression(model_type="linear", dataset_type="linear", sample_size=300, **hyperparams):
#     """Runs the regression model and returns R² score and the Base64 image."""
#     X_train, X_test, y_train, y_test = generate_dataset(dataset_type, sample_size)

#     if model_type == "linear":
#         model = LinearRegression()
#     elif model_type == "polynomial":
#         degree = hyperparams.get("degree", 2)
#         model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
#     elif model_type == "dt":
#         max_depth = hyperparams.get("max_depth", 5)
#         model = DecisionTreeRegressor(max_depth=max_depth)
#     elif model_type == "rf":
#         n_estimators = hyperparams.get("n_estimators", 100)
#         model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
#     else:
#         raise ValueError("Invalid model type. Choose 'linear', 'polynomial', 'dt', or 'rf'.")

#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)
#     r2_score = model.score(X_test, y_test)

#     # Generate Base64 plot
#     plot_base64 = plot_results(X_test, y_test, y_pred, model_type)

#     return r2_score, y_pred, plot_base64
