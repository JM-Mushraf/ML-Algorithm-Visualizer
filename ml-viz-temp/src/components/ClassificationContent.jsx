"use client"
import { motion } from "framer-motion"
import { BarChart3 } from "lucide-react"
import "./ClassificationContent.css"

function ClassificationContent() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.5 }}
      className="container"
    >
      <h1 className="section-title gradient-text cyan-to-purple">Classification Visualizer</h1>

      <div className="classification-grid">
        <div className="visualization-panel">
          <div className="visualization-placeholder">
            <BarChart3 className="visualization-icon cyan" />
            <p className="placeholder-text">Classification visualization will appear here</p>
          </div>
        </div>

        <div className="parameters-panel glass-panel">
          <h3 className="panel-title">Parameters</h3>

          <div className="parameters-form">
            <div className="form-group">
              <label className="form-label">Algorithm</label>
              <select className="form-select cyan">
                <option>Logistic Regression</option>
                <option>K-Nearest Neighbors</option>
                <option>Support Vector Machine</option>
                <option>Decision Tree</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Regularization</label>
              <input type="range" min="0.01" max="10" step="0.01" defaultValue="1" className="form-range cyan" />
              <div className="range-labels">
                <span>0.01</span>
                <span>1.0</span>
                <span>5.0</span>
                <span>10.0</span>
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">Test Split</label>
              <input type="range" min="0.1" max="0.5" step="0.05" defaultValue="0.2" className="form-range cyan" />
              <div className="range-labels">
                <span>10%</span>
                <span>20%</span>
                <span>35%</span>
                <span>50%</span>
              </div>
            </div>

            <button className="run-button cyan">Run Model</button>
          </div>
        </div>
      </div>

      <div className="metrics-grid">
        <div className="metrics-panel glass-panel">
          <h3 className="panel-title">Model Performance</h3>
          <div className="metrics-list">
            <div className="metric-item">
              <span className="metric-label">Accuracy:</span>
              <span className="metric-value">0.92</span>
            </div>
            <div className="metric-item">
              <span className="metric-label">Precision:</span>
              <span className="metric-value">0.89</span>
            </div>
            <div className="metric-item">
              <span className="metric-label">Recall:</span>
              <span className="metric-value">0.94</span>
            </div>
            <div className="metric-item">
              <span className="metric-label">F1 Score:</span>
              <span className="metric-value">0.91</span>
            </div>
          </div>
        </div>

        <div className="metrics-panel glass-panel">
          <h3 className="panel-title">Confusion Matrix</h3>
          <div className="confusion-matrix">
            <div className="matrix-cell true-positive">
              <div className="cell-value">42</div>
              <div className="cell-label">True Positive</div>
            </div>
            <div className="matrix-cell false-negative">
              <div className="cell-value">5</div>
              <div className="cell-label">False Negative</div>
            </div>
            <div className="matrix-cell false-positive">
              <div className="cell-value">3</div>
              <div className="cell-label">False Positive</div>
            </div>
            <div className="matrix-cell true-negative">
              <div className="cell-value">50</div>
              <div className="cell-label">True Negative</div>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

export default ClassificationContent