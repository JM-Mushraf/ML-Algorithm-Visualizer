from flask import Flask, request, jsonify, send_from_directory
from regressions import run_regression
from classifications import run_classification  # Import the classification function
import os

app = Flask(__name__)

# Define static folder for serving images
PLOT_DIR = "static/plots"

@app.route("/")
def home():
    return jsonify({"message": "Welcome to the Regression and Classification API"}), 200

@app.route("/regression", methods=["POST"])
def regression():
    try:
        data = request.get_json()

        model_type = data.get("model_type", "linear")
        dataset_type = data.get("dataset_type", "linear")
        sample_size = int(data.get("sample_size", 300))
        hyperparams = data.get("hyperparams", {})

        r2_score, y_pred, plot_filename = run_regression(model_type, dataset_type, sample_size, **hyperparams)

        return jsonify({
            "model_type": model_type,
            "dataset_type": dataset_type,
            "r2_score": round(r2_score, 4),
            "plot_url": f"http://127.0.0.1:5000/plots/{plot_filename}"  # URL to access the plot
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/classification", methods=["POST"])
def classification():
    try:
        data = request.get_json()

        model_type = data.get("model_type", "logistic")
        dataset_type = data.get("dataset_type", "linear")
        sample_size = int(data.get("sample_size", 200))
        noise = float(data.get("noise", 0.1))
        hyperparams = data.get("hyperparams", {})

        accuracy, plot_filename = run_classification(model_type, dataset_type, sample_size, noise, **hyperparams)

        return jsonify({
            "model_type": model_type,
            "dataset_type": dataset_type,
            "accuracy": round(accuracy, 4),
            "plot_url": f"http://127.0.0.1:5000/plots/{plot_filename}"  # URL to access the plot
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/plots/<filename>")
def get_plot(filename):
    return send_from_directory(PLOT_DIR, filename)

if __name__ == "__main__":
    app.run(debug=True)