"use client"
import { motion } from "framer-motion"
import { Upload, BarChart3, LineChart } from "lucide-react"
import { useState } from "react"
import axios from "axios"
import "./UploadContent.css"

function UploadContent() {
  const [file, setFile] = useState(null)
  const [analysisResults, setAnalysisResults] = useState(null)
  const [error, setError] = useState(null)
  const [isUploading, setIsUploading] = useState(false)

  const handleFileChange = (event) => {
    setFile(event.target.files[0])
  }

  const handleUpload = async () => {
    if (!file) {
      alert("Please select a file first!")
      return
    }

    setIsUploading(true)
    const formData = new FormData()
    formData.append("file", file)

    try {
      const response = await axios.post("https://algovizserver.onrender.com/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      })

      console.log("Response:", response.data)
      setAnalysisResults(response.data.analysis)
      setError(null)
    } catch (error) {
      console.error("Error uploading file:", error)
      setError("Error uploading file. Please try again.")
      setAnalysisResults(null)
    } finally {
      setIsUploading(false)
    }
  }

  const handleSampleDataset = (dataset) => {
    console.log(`Selected ${dataset} dataset`)
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.5 }}
      className="upload-content-container"
    >
      <h1 className="upload-content-title">Upload Dataset</h1>

      <div className="upload-content-grid">
        {/* Main Upload Area */}
        <div className="upload-content-area glass-panel">
          <div className="upload-content-dropzone">
            <Upload className="upload-content-icon" />
            <h3 className="upload-content-dropzone-title">Drag and drop your dataset</h3>
            <p className="upload-content-dropzone-text">Upload CSV, Excel, or JSON files to visualize and analyze your data</p>

            <input type="file" onChange={handleFileChange} style={{ display: "none" }} id="file-input" />
            <label htmlFor="file-input" className="upload-content-browse-button">
              Browse Files
            </label>
            {file && (
              <div className="upload-content-selected-file">
                <p>Selected: {file.name}</p>
                <button onClick={handleUpload} className="upload-content-upload-button" disabled={isUploading}>
                  {isUploading ? "Uploading..." : "Upload File"}
                </button>
              </div>
            )}
          </div>
        </div>

        {/* Sample Datasets Panel */}
        <div className="upload-content-sample-datasets glass-panel">
          <h3 className="upload-content-panel-title">Sample Datasets</h3>
          <ul className="upload-content-dataset-list">
            <li className="upload-content-dataset-item" onClick={() => handleSampleDataset("Iris")}>
              <div className="upload-content-dataset-icon-container upload-content-pink">
                <BarChart3 className="upload-content-dataset-icon" />
              </div>
              <div className="upload-content-dataset-info">
                <div className="upload-content-dataset-name">Iris Dataset</div>
                <div className="upload-content-dataset-type">Classification</div>
              </div>
            </li>
            <li className="upload-content-dataset-item" onClick={() => handleSampleDataset("Boston Housing")}>
              <div className="upload-content-dataset-icon-container upload-content-orange">
                <LineChart className="upload-content-dataset-icon" />
              </div>
              <div className="upload-content-dataset-info">
                <div className="upload-content-dataset-name">Boston Housing</div>
                <div className="upload-content-dataset-type">Regression</div>
              </div>
            </li>
            <li className="upload-content-dataset-item" onClick={() => handleSampleDataset("MNIST")}>
              <div className="upload-content-dataset-icon-container upload-content-purple">
                <BarChart3 className="upload-content-dataset-icon" />
              </div>
              <div className="upload-content-dataset-info">
                <div className="upload-content-dataset-name">MNIST Digits</div>
                <div className="upload-content-dataset-type">Classification</div>
              </div>
            </li>
          </ul>
        </div>
      </div>

      {error && <div className="upload-content-error-message">{error}</div>}

      {analysisResults && (
        <div className="upload-content-analysis-container">
          <h2 className="upload-content-analysis-title">Analysis Results</h2>

          {/* Performance Metrics */}
          <div className="upload-content-metrics-card glass-panel">
            <h3 className="upload-content-card-title">Model Performance</h3>
            <div className="upload-content-metrics-grid">
              <div className="upload-content-metric">
                <span className="upload-content-metric-label">Accuracy</span>
                <span className="upload-content-metric-value">{analysisResults.accuracy.toFixed(2)}</span>
              </div>
              <div className="upload-content-metric">
                <span className="upload-content-metric-label">Precision</span>
                <span className="upload-content-metric-value">0.00</span>
              </div>
              <div className="upload-content-metric">
                <span className="upload-content-metric-label">Recall</span>
                <span className="upload-content-metric-value">0.00</span>
              </div>
              <div className="upload-content-metric">
                <span className="upload-content-metric-label">F1 Score</span>
                <span className="upload-content-metric-value">0.00</span>
              </div>
            </div>
          </div>

          {/* Dataset Information */}
          <div className="upload-content-info-grid">
            <div className="upload-content-info-card glass-panel">
              <h3 className="upload-content-card-title">Dataset Information</h3>

              <div className="upload-content-info-section">
                <h4 className="upload-content-section-title">Basic Information</h4>
                <p className="upload-content-info-item">Number of Rows: {analysisResults.info.num_rows}</p>
                <p className="upload-content-info-item">Number of Columns: {analysisResults.info.num_columns}</p>
                <p className="upload-content-info-item">Columns: {analysisResults.info.columns.join(", ")}</p>
              </div>

              <div className="upload-content-info-section">
                <h4 className="upload-content-section-title">Missing Values</h4>
                <div className="upload-content-missing-values-grid">
                  {Object.entries(analysisResults.info.missing_values).map(([col, count]) => (
                    <div key={col} className="upload-content-missing-value-item">
                      <span className="upload-content-column-name">{col}:</span> {count}
                    </div>
                  ))}
                </div>
              </div>

              <div className="upload-content-info-section">
                <h4 className="upload-content-section-title">Dataset Preview</h4>
                <pre className="upload-content-dataset-preview">
                  {JSON.stringify(analysisResults.info.dataset_preview, null, 2)}
                </pre>
              </div>
            </div>

            <div className="upload-content-algorithm-card glass-panel">
              <div className="upload-content-info-section">
                <h4 className="upload-content-section-title">Best Algorithm</h4>
                <div className="upload-content-algorithm-info">
                  <p className="upload-content-info-item">
                    <span className="upload-content-info-label">Algorithm:</span> {analysisResults.best_algorithm}
                  </p>
                  <p className="upload-content-info-item">
                    <span className="upload-content-info-label">Accuracy:</span> {analysisResults.accuracy.toFixed(2)}
                  </p>
                </div>
              </div>

              <div className="upload-content-info-section">
                <h4 className="upload-content-section-title">Best Features</h4>
                <div className="upload-content-features-info">
                  {analysisResults.best_features ? (
                    Object.entries(analysisResults.best_features).map(([key, value]) => (
                      <div key={key}>
                        <strong>{key}:</strong> {value}
                      </div>
                    ))
                  ) : (
                    <p>Feature importance not available for this model.</p>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* Visualization */}
          <div className="upload-content-visualization-card glass-panel">
            <h3 className="upload-content-card-title">Heatmap</h3>
            <div className="upload-content-visualization-container">
              <img
                src={`https://algovizserver.onrender.com/uploads/${analysisResults.visualization_paths.heatmap.split("/").pop()}`}
                alt="Heatmap"
                className="upload-content-visualization-image"
              />
            </div>
          </div>
        </div>
      )}
    </motion.div>
  )
}

export default UploadContent