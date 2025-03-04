from flask import Blueprint, request, jsonify
import numpy as np
from algorithms.logistic_regression import LogReg,label_encoder

api_routes = Blueprint('api_routes', __name__)

@api_routes.route('/logistic-regression', methods=['POST'])
def logistic_regression():
    try:
        data = request.json
        input_features = np.array(data["features"]).reshape(1, -1)  # Convert input to numpy array
        prediction = LogReg.predict(input_features)
        predicted_class = label_encoder.inverse_transform(prediction)[0]  # Convert back to original species name

        return jsonify({"prediction": predicted_class})
    except Exception as e:
        return jsonify({"error": str(e)})
