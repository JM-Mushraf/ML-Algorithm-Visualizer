from flask import Blueprint, request, jsonify
import numpy as np
from algorithms.logistic_regression import logistic_model, logistic_label_encoder
from algorithms.linear_regression import linear_model

model_blueprint = Blueprint('models', __name__)

@model_blueprint.route('/logistic-regression', methods=['POST'])
def logistic_regression():
    try:
        data = request.json
        input_features = np.array(data["features"]).reshape(1, -1)
        prediction = logistic_model.predict(input_features)
        predicted_class = logistic_label_encoder.inverse_transform(prediction)[0]
        return jsonify({"prediction": predicted_class})
    except Exception as e:
        return jsonify({"error": str(e)})

@model_blueprint.route('/linear-regression', methods=['POST'])
def linear_regression():
    try:
        data = request.json
        input_features = np.array(data["features"]).reshape(1, -1)
        prediction = linear_model.predict(input_features)[0]
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)})
