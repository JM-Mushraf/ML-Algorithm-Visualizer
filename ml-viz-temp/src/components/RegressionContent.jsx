"use client"
import { motion } from "framer-motion"
import { LineChart } from "lucide-react"
import "./RegressionContent.css"

function RegressionContent() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.5 }}
      className="container"
    >
      <h1 className="section-title gradient-text purple-to-pink">Regression Visualizer</h1>

      <div className="regression-grid">
        <div className="visualization-panel">
          <div className="visualization-placeholder">
            <LineChart className="visualization-icon" />
            <p className="placeholder-text">Regression visualization will appear here</p>
          </div>
        </div>

        <div className="parameters-panel glass-panel">
          <h3 className="panel-title">Parameters</h3>

          <div className="parameters-form">
            <div className="form-group">
              <label className="form-label">Algorithm</label>
              <select className="form-select">
                <option>Linear Regression</option>
                <option>Polynomial Regression</option>
                <option>Ridge Regression</option>
                <option>Lasso Regression</option>
              </select>
            </div>

            <div className="form-group">
              <label className="form-label">Learning Rate</label>
              <input type="range" min="0.001" max="1" step="0.001" defaultValue="0.01" className="form-range" />
              <div className="range-labels">
                <span>0.001</span>
                <span>0.01</span>
                <span>0.1</span>
                <span>1.0</span>
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">Iterations</label>
              <input type="range" min="10" max="1000" step="10" defaultValue="100" className="form-range" />
              <div className="range-labels">
                <span>10</span>
                <span>100</span>
                <span>500</span>
                <span>1000</span>
              </div>
            </div>

            <button className="run-button">Run Model</button>
          </div>
        </div>
      </div>

      <div className="metrics-grid">
        <div className="metrics-panel glass-panel">
          <h3 className="panel-title">Model Performance</h3>
          <div className="metrics-list">
            <div className="metric-item">
              <span className="metric-label">R² Score:</span>
              <span className="metric-value">0.87</span>
            </div>
            <div className="metric-item">
              <span className="metric-label">Mean Squared Error:</span>
              <span className="metric-value">0.0342</span>
            </div>
            <div className="metric-item">
              <span className="metric-label">Mean Absolute Error:</span>
              <span className="metric-value">0.1253</span>
            </div>
          </div>
        </div>

        <div className="metrics-panel glass-panel">
          <h3 className="panel-title">Coefficients</h3>
          <div className="metrics-list">
            <div className="metric-item">
              <span className="metric-label">Intercept:</span>
              <span className="metric-value">2.3451</span>
            </div>
            <div className="metric-item">
              <span className="metric-label">X₁:</span>
              <span className="metric-value">0.7823</span>
            </div>
            <div className="metric-item">
              <span className="metric-label">X₂:</span>
              <span className="metric-value">-0.4217</span>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

export default RegressionContent

