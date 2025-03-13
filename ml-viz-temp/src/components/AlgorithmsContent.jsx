"use client"
import { motion } from "framer-motion"
import { LineChart, BarChart3, Brain } from "lucide-react"
import { useNavigate } from "react-router-dom"

import "./AlgorithmsContent.css"

function AlgorithmsContent() {
  const navigate=useNavigate()
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.5 }}
      className="container"
    >
      <h1 className="section-title gradient-text green-to-blue">Learn Algorithms</h1>

      <div className="algorithms-grid">
        <div className="algorithm-card glass-panel green">
          <div className="algorithm-icon-container green">
            <LineChart className="algorithm-icon" />
          </div>

          <h3 className="algorithm-title">Linear Regression</h3>
          <p className="algorithm-description">
            Learn how to predict continuous values using linear relationships between variables.
          </p>
          <div className="algorithm-footer">
            <span className="difficulty-label">Difficulty: Beginner</span>
            <a className="learn-more-button green" href="https://www.geeksforgeeks.org/ml-linear-regression/" target="_blank" >Learn more →</a>
          </div>
        </div>

        <div className="algorithm-card glass-panel blue">
          <div className="algorithm-icon-container blue">
            <BarChart3 className="algorithm-icon" />
          </div>
          <h3 className="algorithm-title">Logistic Regression</h3>
          <p className="algorithm-description">
            Understand how to classify data into discrete categories using probability.
          </p>
          <div className="algorithm-footer">
            <span className="difficulty-label">Difficulty: Intermediate</span>
            <a className="learn-more-button blue" href="https://www.geeksforgeeks.org/understanding-logistic-regression/" target="_blank">Learn more →</a>
          </div>
        </div>

        <div className="algorithm-card glass-panel purple">
          <div className="algorithm-icon-container purple">
            <Brain className="algorithm-icon" />
          </div>
          <h3 className="algorithm-title">Neural Networks</h3>
          <p className="algorithm-description">
            Explore deep learning with multi-layer networks inspired by the human brain.
          </p>
          <div className="algorithm-footer">
            <span className="difficulty-label">Difficulty: Advanced</span>
            <a className="learn-more-button purple" href="https://www.geeksforgeeks.org/neural-networks-a-beginners-guide/" target="_blank">Learn more →</a>
          </div>
        </div>

        <div className="algorithm-card glass-panel cyan">
          <div className="algorithm-icon-container cyan">
            <BarChart3 className="algorithm-icon" />
          </div>
          <h3 className="algorithm-title">K-Means Clustering</h3>
          <p className="algorithm-description">
            Discover how to group similar data points together using unsupervised learning.
          </p>
          <div className="algorithm-footer">
            <span className="difficulty-label">Difficulty: Intermediate</span>
            <a className="learn-more-button cyan" href="https://www.geeksforgeeks.org/k-means-clustering-introduction/" target="_blank">Learn more →</a>
          </div>
        </div>

        <div className="algorithm-card glass-panel pink">
          <div className="algorithm-icon-container pink">
            <Brain className="algorithm-icon" />
          </div>
          <h3 className="algorithm-title">Support Vector Machines</h3>
          <p className="algorithm-description">
            Learn how to classify data by finding the optimal hyperplane between classes.
          </p>
          <div className="algorithm-footer">
            <span className="difficulty-label">Difficulty: Advanced</span>
            <a className="learn-more-button pink" href="https://www.geeksforgeeks.org/support-vector-machine-algorithm/" target="_blank">Learn more →</a>
          </div>
        </div>

        <div className="algorithm-card glass-panel orange">
          <div className="algorithm-icon-container orange">
            <Brain className="algorithm-icon" />
          </div>
          <h3 className="algorithm-title">Decision Trees</h3>
          <p className="algorithm-description">
            Understand how to make decisions by splitting data based on feature values.
          </p>
          <div className="algorithm-footer">
            <span className="difficulty-label">Difficulty: Intermediate</span>
            <a className="learn-more-button orange" href="https://www.geeksforgeeks.org/decision-tree/" >Learn more →</a>
          </div>
        </div>
      </div>

      <div className="learning-path glass-panel">
        <h3 className="panel-title">Learning Path</h3>
        <div className="path-container">
          <div className="path-line"></div>

          <div className="path-steps">
            <div className="path-step">
              <div className="path-marker green"></div>
              <h4 className="path-title">Beginner Level</h4>
              <p className="path-description">Start with the fundamentals of machine learning</p>
              <div className="path-tags">
                <span className="path-tag">Linear Regression</span>
                <span className="path-tag">Data Preprocessing</span>
                <span className="path-tag">Model Evaluation</span>
              </div>
            </div>

            <div className="path-step">
              <div className="path-marker blue"></div>
              <h4 className="path-title">Intermediate Level</h4>
              <p className="path-description">Expand your knowledge with more complex algorithms</p>
              <div className="path-tags">
                <span className="path-tag">Logistic Regression</span>
                <span className="path-tag">Decision Trees</span>
                <span className="path-tag">K-Means Clustering</span>
              </div>
            </div>

            <div className="path-step">
              <div className="path-marker purple"></div>
              <h4 className="path-title">Advanced Level</h4>
              <p className="path-description">Master complex techniques and deep learning</p>
              <div className="path-tags">
                <span className="path-tag">Neural Networks</span>
                <span className="path-tag">Support Vector Machines</span>
                <span className="path-tag">Ensemble Methods</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  )
}

export default AlgorithmsContent

