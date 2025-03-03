from flask import Flask, request, jsonify
import numpy as np
from algorithms.linear_regression import train_linear_regression
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])

@app.route('/api/linear-regression', methods=['POST'])
def linear_regression():
    data = request.json
    X = np.array(data['X']).reshape(-1, 1)  # Reshape X to 2D
    y = np.array(data['y'])
    model = train_linear_regression(X, y)
    return jsonify({
    "coefficients": model.coef_.tolist(),  # Slope
    "intercept": model.intercept_.tolist()  # Intercept
})


if __name__ == '__main__':
    app.run(debug=True)