import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load Iris dataset
df = sns.load_dataset('iris')

# Convert categorical target ('species') into numerical labels
label_encoder = LabelEncoder()
df["species"] = label_encoder.fit_transform(df["species"])

# Features (X) and target (y)
X = df.iloc[:, :-1].values
y = df["species"].values

# Train Logistic Regression model
LogReg = LogisticRegression(max_iter=200)
LogReg.fit(X, y)
