import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

def train_linear_regression(dataset_name, hyperparameters):
    """Train Linear Regression model and return best fit line"""
    
    # Load dataset
    df = sns.load_dataset(dataset_name)
    if df is None:
        return {"error": "Dataset not found"}
    
    # Select first numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) < 2:
        return {"error": "Dataset does not have enough numeric columns"}

    X = df[numeric_cols[0]].values.reshape(-1, 1)  # First numeric column as X
    y = df[numeric_cols[1]].values  # Second numeric column as y

    # Train Linear Regression Model
    model = LinearRegression()
    model.fit(X, y)

    # Generate best fit line
    x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(x_line)

    return {
        "x_values": x_line.flatten().tolist(),
        "y_values": y_line.tolist(),
        "slope": model.coef_[0],
        "intercept": model.intercept_
    }
