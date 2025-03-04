import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

def train_logistic_regression(dataset_name, hyperparameters):
    """Train Logistic Regression and return decision boundary"""
    
    # Load dataset
    df = sns.load_dataset(dataset_name)
    if df is None:
        return {"error": "Dataset not found"}

    # Select first two numeric columns + target
    numeric_cols = df.select_dtypes(include=['number']).columns
    target_col = df.select_dtypes(include=['object', 'category']).columns

    if len(numeric_cols) < 2 or len(target_col) < 1:
        return {"error": "Dataset does not have enough numeric columns or categorical labels"}

    X = df[numeric_cols[:2]].values  # First two numeric columns as features
    y = df[target_col[0]].values  # First categorical column as target

    # Encode categorical labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Train Logistic Regression Model
    model = LogisticRegression()
    model.fit(X, y)

    return {
        "slope": model.coef_.tolist(),
        "intercept": model.intercept_.tolist(),
    }
