from flask import Blueprint, request, jsonify
import seaborn as sns

dataset_blueprint = Blueprint('dataset', __name__)

# Function to fetch dataset
def get_builtin_dataset(dataset_name):
    try:
        if dataset_name not in sns.get_dataset_names():
            return {"error": "Dataset not found"}
        df = sns.load_dataset(dataset_name)
        return {"columns": df.columns.tolist(), "data": df.head(5).to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e)}

@dataset_blueprint.route('/get', methods=['POST'])
def fetch_dataset():
    data = request.json
    dataset_name = data.get("datasetName")
    if not dataset_name:
        return jsonify({"error": "Dataset name is required"}), 400

    result = get_builtin_dataset(dataset_name)
    return jsonify(result)
