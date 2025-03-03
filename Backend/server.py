from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from flask_cors import CORS
import seaborn as sns

app = Flask(__name__)
CORS(app)  # Allow requests from frontend

# Load Iris dataset
df = sns.load_dataset('iris')

# Convert categorical target ('species') into numerical labels
label_encoder = LabelEncoder()
df["species"] = label_encoder.fit_transform(df["species"])

# Features (X) and target (y)
X = df.iloc[:, :-1].values  # All feature columns (sepal_length, sepal_width, etc.)
y = df["species"].values  # Encoded species

# Train Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

@app.route('/api/logistic-regression', methods=['POST'])
def logistic_regression():
    try:
        data = request.json
        input_features = np.array(data["features"]).reshape(1, -1)  # Convert input to numpy array
        prediction = model.predict(input_features)
        predicted_class = label_encoder.inverse_transform(prediction)[0]  # Convert back to original species name

        return jsonify({"prediction": predicted_class})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)