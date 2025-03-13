"use client"
import { motion } from "framer-motion"
import { BarChart3 } from "lucide-react"
import { useState } from "react"
import "./ClassificationContent.css"
import toast from "react-hot-toast"
function ClassificationContent() {
  const [metrics, setMetrics] = useState({
    accuracy: 0,
    precision: 0,
    recall: 0,
    f1: 0,
    confusionMatrix: [[0, 0], [0, 0]],
    plotUrl: ""
  });

  const [modelType, setModelType] = useState("logistic");
  const [datasetType, setDatasetType] = useState("linear");
  const [sampleSize, setSampleSize] = useState(200);
  const [regularization, setRegularization] = useState(1.0);
  const [testSplit, setTestSplit] = useState(0.2);
  const [maxDepth, setMaxDepth] = useState(5);
  const [maxFeatures, setMaxFeatures] = useState("sqrt");
  const [solver, setSolver] = useState("lbfgs");
  const [penalty, setPenalty] = useState("l2");
  const [maxIter, setMaxIter] = useState(100);
  const [minSamplesSplit, setMinSamplesSplit] = useState(2);
  const [minSamplesLeaf, setMinSamplesLeaf] = useState(1);
  const [criterion, setCriterion] = useState("gini");
  const [nEstimators, setNEstimators] = useState(100);
  const [kernel, setKernel] = useState("linear");
  const [gamma, setGamma] = useState("scale");
  const [degree, setDegree] = useState(3);
  const [nNeighbors, setNNeighbors] = useState(5); // KNN hyperparameter
  const [weights, setWeights] = useState("uniform"); // KNN hyperparameter
  const [algorithm, setAlgorithm] = useState("auto"); // KNN hyperparameter
  const [leafSize, setLeafSize] = useState(30); // KNN hyperparameter
  const [p, setP] = useState(2); // KNN hyperparameter
  const [varSmoothing, setVarSmoothing] = useState(1e-9); // Naive Bayes hyperparameter

  const handleRunModel = async () => {
    try {
      const response = await fetch("https://algovizserver.onrender.com/classification", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model_type: modelType,
          dataset_type: datasetType,
          sample_size: sampleSize,
          noise: 0.1,
          hyperparameters: {
            C: regularization,
            test_split: testSplit,
            max_depth: maxDepth,
            max_features: maxFeatures,
            solver: solver,
            penalty: penalty,
            max_iter: maxIter,
            min_samples_split: minSamplesSplit,
            min_samples_leaf: minSamplesLeaf,
            criterion: criterion,
            n_estimators: nEstimators,
            kernel: kernel,
            gamma: gamma,
            degree: degree,
            n_neighbors: nNeighbors, // KNN hyperparameter
            weights: weights, // KNN hyperparameter
            algorithm: algorithm, // KNN hyperparameter
            leaf_size: leafSize, // KNN hyperparameter
            p: p, // KNN hyperparameter
            var_smoothing: varSmoothing, // Naive Bayes hyperparameter
          }
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`HTTP error! Status: ${response.status}, Message: ${errorData.error}`);
      }
      toast.success("Graph Generated")
      const data = await response.json();
      setMetrics({
        accuracy: data.accuracy,
        precision: data.precision,
        recall: data.recall,
        f1: data.f1,
        confusionMatrix: data.confusion_matrix,
        plotUrl: `https://algovizserver.onrender.com/static/plots/${data.plot_filename}`,
      });
    } catch (error) {
      console.error("Error fetching data:", error);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      transition={{ duration: 0.5 }}
      className="classification-container"
    >
      <h1 className="classification-section-title gradient-text cyan-to-purple">Classification Visualizer</h1>

      <div className="classification-grid">
        <div className="classification-visualization-panel">
          <div className="classification-visualization-placeholder">
            {metrics.plotUrl ? (
              <img src={metrics.plotUrl} alt="Decision Boundary" className="classification-visualization-image" />
            ) : (
              <>
                <BarChart3 className="classification-visualization-icon cyan" />
                <p className="classification-placeholder-text">Classification visualization will appear here</p>
              </>
            )}
          </div>
        </div>

        <div className="classification-parameters-panel glass-panel">
          <h3 className="classification-panel-title">Parameters</h3>

          <div className="classification-parameters-form">
            <div className="classification-form-group">
              <label className="classification-form-label">Algorithm</label>
              <select
                className="classification-form-select cyan"
                value={modelType}
                onChange={(e) => setModelType(e.target.value)}
              >
                <option value="logistic">Logistic Regression</option>
                <option value="dt">Decision Tree</option>
                <option value="rf">Random Forest</option>
                <option value="svm">Support Vector Machine</option>
                <option value="nb">Naive Bayes</option>
                <option value="knn">K-Nearest Neighbors</option>
              </select>
            </div>

            <div className="classification-form-group">
              <label className="classification-form-label">Dataset Configuration</label>
              <select
                className="classification-form-select cyan"
                value={datasetType}
                onChange={(e) => setDatasetType(e.target.value)}
              >
                <option value="linear">Linear Dataset</option>
                <option value="moons">Moons Dataset</option>
                <option value="circles">Circles Dataset</option>
              </select>
            </div>

            <div className="classification-form-group">
              <label className="classification-form-label">Sample Size</label>
              <input
                type="number"
                min="100"
                max="1000"
                step="100"
                value={sampleSize}
                onChange={(e) => setSampleSize(parseInt(e.target.value))}
                className="classification-form-input cyan"
              />
            </div>

            {modelType === "logistic" && (
              <>
                <div className="classification-form-group">
                  <label className="classification-form-label">Regularization</label>
                  <input
                    type="range"
                    min="0.01"
                    max="10"
                    step="0.01"
                    value={regularization}
                    onChange={(e) => setRegularization(parseFloat(e.target.value))}
                    className="classification-form-range cyan"
                  />
                  <div className="classification-range-labels">
                    <span>0.01</span>
                    <span>1.0</span>
                    <span>5.0</span>
                    <span>10.0</span>
                  </div>
                </div>

                <div className="classification-form-group">
                  <label className="classification-form-label">Solver</label>
                  <select
                    className="classification-form-select cyan"
                    value={solver}
                    onChange={(e) => setSolver(e.target.value)}
                  >
                    <option value="lbfgs">lbfgs</option>
                    <option value="liblinear">liblinear</option>
                    <option value="newton-cg">newton-cg</option>
                  </select>
                </div>

                <div className="classification-form-group">
                  <label className="classification-form-label">Penalty</label>
                  <select
                    className="classification-form-select cyan"
                    value={penalty}
                    onChange={(e) => setPenalty(e.target.value)}
                  >
                    <option value="l2">l2</option>
                    <option value="l1">l1</option>
                    <option value="elasticnet">elasticnet</option>
                  </select>
                </div>

                <div className="classification-form-group">
                  <label className="classification-form-label">Max Iterations</label>
                  <input
                    type="number"
                    min="100"
                    max="1000"
                    step="100"
                    value={maxIter}
                    onChange={(e) => setMaxIter(parseInt(e.target.value))}
                    className="classification-form-input cyan"
                  />
                </div>
              </>
            )}

            {(modelType === "dt" || modelType === "rf") && (
              <>
                <div className="classification-form-group">
                  <label className="classification-form-label">Max Depth</label>
                  <input
                    type="number"
                    min="1"
                    max="50"
                    value={maxDepth}
                    onChange={(e) => setMaxDepth(parseInt(e.target.value))}
                    className="classification-form-input cyan"
                  />
                </div>

                <div className="classification-form-group">
                  <label className="classification-form-label">Max Features</label>
                  <select
                    className="classification-form-select cyan"
                    value={maxFeatures}
                    onChange={(e) => setMaxFeatures(e.target.value)}
                  >
                    <option value="sqrt">sqrt</option>
                    <option value="log2">log2</option>
                    <option value={null}>None</option>
                  </select>
                </div>

                <div className="classification-form-group">
                  <label className="classification-form-label">Min Samples Split</label>
                  <input
                    type="number"
                    min="2"
                    max="20"
                    value={minSamplesSplit}
                    onChange={(e) => setMinSamplesSplit(parseInt(e.target.value))}
                    className="classification-form-input cyan"
                  />
                </div>

                <div className="classification-form-group">
                  <label className="classification-form-label">Min Samples Leaf</label>
                  <input
                    type="number"
                    min="1"
                    max="20"
                    value={minSamplesLeaf}
                    onChange={(e) => setMinSamplesLeaf(parseInt(e.target.value))}
                    className="classification-form-input cyan"
                  />
                </div>

                <div className="classification-form-group">
                  <label className="classification-form-label">Criterion</label>
                  <select
                    className="classification-form-select cyan"
                    value={criterion}
                    onChange={(e) => setCriterion(e.target.value)}
                  >
                    <option value="gini">gini</option>
                    <option value="entropy">entropy</option>
                  </select>
                </div>
              </>
            )}

            {modelType === "rf" && (
              <div className="classification-form-group">
                <label className="classification-form-label">Number of Estimators</label>
                <input
                  type="number"
                  min="10"
                  max="500"
                  step="10"
                  value={nEstimators}
                  onChange={(e) => setNEstimators(parseInt(e.target.value))}
                  className="classification-form-input cyan"
                />
              </div>
            )}

            {modelType === "svm" && (
              <>
                <div className="classification-form-group">
                  <label className="classification-form-label">Kernel</label>
                  <select
                    className="classification-form-select cyan"
                    value={kernel}
                    onChange={(e) => setKernel(e.target.value)}
                  >
                    <option value="linear">linear</option>
                    <option value="poly">poly</option>
                    <option value="rbf">rbf</option>
                    <option value="sigmoid">sigmoid</option>
                  </select>
                </div>

                <div className="classification-form-group">
                  <label className="classification-form-label">Gamma</label>
                  <select
                    className="classification-form-select cyan"
                    value={gamma}
                    onChange={(e) => setGamma(e.target.value)}
                  >
                    <option value="scale">scale</option>
                    <option value="auto">auto</option>
                  </select>
                </div>

                <div className="classification-form-group">
                  <label className="classification-form-label">Degree</label>
                  <input
                    type="number"
                    min="1"
                    max="10"
                    value={degree}
                    onChange={(e) => setDegree(parseInt(e.target.value))}
                    className="classification-form-input cyan"
                  />
                </div>
              </>
            )}

            {modelType === "nb" && (
              <div className="classification-form-group">
                <label className="classification-form-label">Var Smoothing</label>
                <input
                  type="number"
                  min="1e-12"
                  max="1e-6"
                  step="1e-9"
                  value={varSmoothing}
                  onChange={(e) => setVarSmoothing(parseFloat(e.target.value))}
                  className="classification-form-input cyan"
                />
              </div>
            )}

            {modelType === "knn" && (
              <>
                <div className="classification-form-group">
                  <label className="classification-form-label">Number of Neighbors</label>
                  <input
                    type="number"
                    min="1"
                    max="50"
                    value={nNeighbors}
                    onChange={(e) => setNNeighbors(parseInt(e.target.value))}
                    className="classification-form-input cyan"
                  />
                </div>

                <div className="classification-form-group">
                  <label className="classification-form-label">Weights</label>
                  <select
                    className="classification-form-select cyan"
                    value={weights}
                    onChange={(e) => setWeights(e.target.value)}
                  >
                    <option value="uniform">uniform</option>
                    <option value="distance">distance</option>
                  </select>
                </div>

                <div className="classification-form-group">
                  <label className="classification-form-label">Algorithm</label>
                  <select
                    className="classification-form-select cyan"
                    value={algorithm}
                    onChange={(e) => setAlgorithm(e.target.value)}
                  >
                    <option value="auto">auto</option>
                    <option value="ball_tree">ball_tree</option>
                    <option value="kd_tree">kd_tree</option>
                    <option value="brute">brute</option>
                  </select>
                </div>

                <div className="classification-form-group">
                  <label className="classification-form-label">Leaf Size</label>
                  <input
                    type="number"
                    min="10"
                    max="100"
                    value={leafSize}
                    onChange={(e) => setLeafSize(parseInt(e.target.value))}
                    className="classification-form-input cyan"
                  />
                </div>

                <div className="classification-form-group">
                  <label className="classification-form-label">Distance Metric (p)</label>
                  <input
                    type="number"
                    min="1"
                    max="2"
                    value={p}
                    onChange={(e) => setP(parseInt(e.target.value))}
                    className="classification-form-input cyan"
                  />
                </div>
              </>
            )}
          </div>

          <div className="classification-sticky-button-container">
            <button className="classification-run-button cyan" onClick={handleRunModel}>
              Run Model
            </button>
          </div>
        </div>
      </div>

      <div className="classification-metrics-grid">
        <div className="classification-metrics-panel glass-panel">
          <h3 className="classification-panel-title">Model Performance</h3>
          <div className="classification-metrics-list">
            <div className="classification-metric-item">
              <span className="classification-metric-label">Accuracy:</span>
              <span className="classification-metric-value">{metrics.accuracy.toFixed(2)}</span>
            </div>
            <div className="classification-metric-item">
              <span className="classification-metric-label">Precision:</span>
              <span className="classification-metric-value">{metrics.precision.toFixed(2)}</span>
            </div>
            <div className="classification-metric-item">
              <span className="classification-metric-label">Recall:</span>
              <span className="classification-metric-value">{metrics.recall.toFixed(2)}</span>
            </div>
            <div className="classification-metric-item">
              <span className="classification-metric-label">F1 Score:</span>
              <span className="classification-metric-value">{metrics.f1.toFixed(2)}</span>
            </div>
          </div>
        </div>

        <div className="classification-metrics-panel glass-panel">
          <h3 className="classification-panel-title">Confusion Matrix</h3>
          <div className="classification-confusion-matrix">
            <div className="classification-matrix-cell classification-true-positive">
              <div className="classification-cell-value">{metrics.confusionMatrix[0][0]}</div>
              <div className="classification-cell-label">True Positive</div>
            </div>
            <div className="classification-matrix-cell classification-false-negative">
              <div className="classification-cell-value">{metrics.confusionMatrix[1][0]}</div>
              <div className="classification-cell-label">False Negative</div>
            </div>
            <div className="classification-matrix-cell classification-false-positive">
              <div className="classification-cell-value">{metrics.confusionMatrix[0][1]}</div>
              <div className="classification-cell-label">False Positive</div>
            </div>
            <div className="classification-matrix-cell classification-true-negative">
              <div className="classification-cell-value">{metrics.confusionMatrix[1][1]}</div>
              <div className="classification-cell-label">True Negative</div>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

export default ClassificationContent;