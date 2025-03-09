"use client";
import { motion } from "framer-motion";
import { Upload, BarChart3, LineChart } from "lucide-react";
import { useState } from "react";
import axios from "axios";
import "./UploadContent.css";

function UploadContent() {
  const [file, setFile] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post("http://localhost:5000/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      console.log("Response:", response.data);
      setAnalysisResults(response.data.analysis);
      setError(null);
    } catch (error) {
      console.error("Error uploading file:", error);
      setError("Error uploading file. Please try again.");
      setAnalysisResults(null);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.5 }}
      className="container"
    >
      <h1 className="section-title gradient-text pink-to-orange">Upload Dataset</h1>

      <div className="upload-container glass-panel">
        <div className="dropzone">
          <Upload className="upload-icon" />
          <h3 className="dropzone-title">Drag and drop your dataset</h3>
          <p className="dropzone-text">Upload CSV, Excel, or JSON files to visualize and analyze your data</p>

          <input
            type="file"
            onChange={handleFileChange}
            style={{ display: "none" }}
            id="file-input"
          />
          <label htmlFor="file-input" className="browse-button">
            Browse Files
          </label>
        </div>
      </div>

      <div className="datasets-grid">
        <div className="sample-datasets glass-panel">
          <h3 className="panel-title">Sample Datasets</h3>
          <ul className="dataset-list">
            <li className="dataset-item">
              <div className="dataset-icon-container pink">
                <BarChart3 className="dataset-icon" />
              </div>
              <div className="dataset-info">
                <div className="dataset-name">Iris Dataset</div>
                <div className="dataset-type">Classification</div>
              </div>
            </li>
            <li className="dataset-item">
              <div className="dataset-icon-container orange">
                <LineChart className="dataset-icon" />
              </div>
              <div className="dataset-info">
                <div className="dataset-name">Boston Housing</div>
                <div className="dataset-type">Regression</div>
              </div>
            </li>
            <li className="dataset-item">
              <div className="dataset-icon-container purple">
                <BarChart3 className="dataset-icon" />
              </div>
              <div className="dataset-info">
                <div className="dataset-name">MNIST Digits</div>
                <div className="dataset-type">Classification</div>
              </div>
            </li>
          </ul>
        </div>

        <button onClick={handleUpload} className="upload-button">
          Upload File
        </button>
      </div>

      {error && <div className="error-message">{error}</div>}

      {analysisResults && (
        <div className="analysis-results">
          <h2 className="analysis-title">Analysis Results</h2>
          <div className="analysis-info">
            <h3>Basic Information</h3>
            <p>Number of Rows: {analysisResults.info.num_rows}</p>
            <p>Number of Columns: {analysisResults.info.num_columns}</p>
            <p>Columns: {analysisResults.info.columns.join(", ")}</p>
            <h3>Missing Values</h3>
            <ul>
              {Object.entries(analysisResults.info.missing_values).map(([col, count]) => (
                <li key={col}>
                  {col}: {count}
                </li>
              ))}
            </ul>
            <h3>Dataset Preview</h3>
            <pre>{JSON.stringify(analysisResults.info.dataset_preview, null, 2)}</pre>
            <h3>Best Algorithm</h3>
            <p>Algorithm: {analysisResults.best_algorithm}</p>
            <p>Accuracy: {analysisResults.accuracy.toFixed(2)}</p>
            <h3>Best Features</h3>
            <pre>{JSON.stringify(analysisResults.best_features, null, 2)}</pre>
          </div>
          <div className="visualizations">
            <h3>Heatmap</h3>
            <img src={`http://localhost:5000/uploads/${analysisResults.visualization_paths.heatmap.split('/').pop()}`} alt="Heatmap" />
          </div>
        </div>
      )}
    </motion.div>
  );
}

export default UploadContent;