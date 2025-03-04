from flask import Blueprint, request, jsonify,Response
import numpy as np
from algorithms.logistic_regression import LogReg,label_encoder
import matplotlib.pyplot as plt
import io
import seaborn as sns

df=sns.load_dataset('iris')
api_routes = Blueprint('api_routes', __name__)

# @api_routes.route('/logistic-regression', methods=['POST'])

@api_routes.route('/visualize', methods=['GET'])
def visualize():
    try:
        # Create a scatter plot
        print(df.head())
        plt.figure(figsize=(8, 6))
        plt.scatter(df['sepal_width'], df['petal_length'], cmap='winter')
        plt.xlabel("Sepal Width")
        plt.ylabel("Petal Length")
        plt.title("Sepal Width vs Petal Length (Iris Dataset)")

        # Save the plot to a bytes buffer
        img = io.BytesIO()
        plt.savefig(img, format='png')  # Save as PNG
        plt.close()  # Close the figure to free memory
        img.seek(0)  # Move cursor to the start of the image file

        # Return the image as a response
        return Response(img.getvalue(), mimetype='image/png')

    except Exception as e:
        return {"error": str(e)}