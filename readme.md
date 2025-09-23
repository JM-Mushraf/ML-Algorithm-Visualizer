# 🚀 AlgoViz – Interactive ML Visualization Tool 🎨📊  

AlgoViz is an **interactive Machine Learning visualization tool** that helps you understand how different ML algorithms work by **visualizing datasets, decision boundaries, regression lines, and performance insights** in real-time.  

Built by **Mushraf JM** & **Himanshu Sahu** 🛠️🚀  

---

## ✨ Features  

🔢 **Algorithm Selection** – Choose from ML models like:  
- Decision Trees 🌳  
- Support Vector Machines (SVM) 📐  
- K-Nearest Neighbors (KNN) 👥  
- Naive Bayes 🎲  
- Boosting (XGBoost) ⚡  

📈 **Regression Visualization** – Plot datasets and see the regression line adapt to your data.  

🎯 **Classification Visualization** – Watch decision boundaries form for different classifiers.  

📂 **Upload Your Dataset** – Drop a CSV file 📑 and AlgoViz will:  
- Analyze your dataset  
- Suggest the best-performing algorithm  
- Display accuracy scores  

⏳ **Performance Insights** – Get execution time, accuracy, and dataset statistics.  

🌡️ **Heatmap Analysis** – Quickly identify correlations between features.  

---

## 📊 Supported Datasets  

You can generate or upload datasets:  
- **Linear Regression** (using `make_regression`)  
- **Random Clusters** (using `make_blobs`)  
- **Exponential Data** (transformed regression)  
- **Classified Data** (using `make_classification`)  

Example snippet to generate dataset:  

```python
from sklearn.datasets import make_regression, make_blobs, make_classification
import numpy as np

def generate_dataset(dataset_type, n_samples=100): 
    if dataset_type == "linear":
        X, y = make_regression(n_samples=n_samples, n_features=1, noise=10)
    elif dataset_type == "random":
        X, y = make_blobs(n_samples=n_samples, centers=3, random_state=42)
    elif dataset_type == "exponential":
        X, y = make_regression(n_samples=n_samples, n_features=1, noise=10)
        y = np.exp(y)
    elif dataset_type == "classified":
        X, y = make_classification(n_samples=n_samples, n_features=2, n_classes=2, random_state=42)
    else:
        raise ValueError("Invalid dataset type.")
    
    return X, y
```
## 🛠️ Installation  

Clone the repository:  

```bash
git clone https://github.com/your-username/algoviz.git
cd algoviz
pip install -r requirements.txt
python app.py
```
## 📌 Project Links  
- 🌐 **Live Demo**: https://algoviz-ichv.onrender.com/
