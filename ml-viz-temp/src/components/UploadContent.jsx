"use client"
import { motion } from "framer-motion"
import { Upload, BarChart3, LineChart } from "lucide-react"
import "./UploadContent.css"

function UploadContent() {
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

          <button className="browse-button">Browse Files</button>
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

        <div className="recent-uploads glass-panel">
          <h3 className="panel-title">Recent Uploads</h3>
          <div className="table-container">
            <table className="uploads-table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Type</th>
                  <th>Size</th>
                  <th>Date</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>customer_data.csv</td>
                  <td>CSV</td>
                  <td>2.4 MB</td>
                  <td>Today</td>
                </tr>
                <tr>
                  <td>sales_2023.xlsx</td>
                  <td>Excel</td>
                  <td>4.7 MB</td>
                  <td>Yesterday</td>
                </tr>
                <tr>
                  <td>sensor_readings.json</td>
                  <td>JSON</td>
                  <td>1.2 MB</td>
                  <td>3 days ago</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

export default UploadContent

