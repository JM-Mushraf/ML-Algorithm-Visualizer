import { useState } from "react";
import axios from "axios";

export default function RegressionApp({ onBack }) {
  const [modelType, setModelType] = useState("linear");
  const [datasetType, setDatasetType] = useState("linear");
  const [sampleSize, setSampleSize] = useState(300);
  const [plotUrl, setPlotUrl] = useState(null);
  const [r2Score, setR2Score] = useState(null);
  const [loading, setLoading] = useState(false);

  // Hyperparameters state
  const [hyperparams, setHyperparams] = useState({
    degree: 2, // For Polynomial Regression
    max_depth: 5, // For Decision Tree and Random Forest
    n_estimators: 100, // For Random Forest
    max_features: "sqrt", // For Decision Tree and Random Forest
  });

  const runRegression = async () => {
    setLoading(true);
    try {
      const response = await axios.post("http://127.0.0.1:5000/regression", {
        model_type: modelType,
        dataset_type: datasetType,
        sample_size: sampleSize,
        hyperparams: hyperparams, // Pass hyperparameters to the backend
      });
      setPlotUrl(response.data.plot_url);
      setR2Score(response.data.r2_score);
    } catch (error) {
      console.error("Error running regression:", error);
    }
    setLoading(false);
  };

  // Update hyperparameters based on model type
  const handleModelTypeChange = (e) => {
    const selectedModel = e.target.value;
    setModelType(selectedModel);
    // Reset hyperparameters when model type changes
    setHyperparams({
      degree: 2,
      max_depth: 5,
      n_estimators: 100,
      max_features: "sqrt",
    });
  };

  // Render hyperparameter inputs based on the selected model
  const renderHyperparameterInputs = () => {
    switch (modelType) {
      case "polynomial":
        return (
          <div style={{ marginBottom: "10px" }}>
            <label style={{ display: "block", marginBottom: "5px" }}>Degree:</label>
            <input
              type="number"
              value={hyperparams.degree}
              onChange={(e) => setHyperparams({ ...hyperparams, degree: parseInt(e.target.value, 10) || 2 })}
              style={{ width: "100%", padding: "8px" }}
            />
          </div>
        );
      case "dt":
        return (
          <>
            <div style={{ marginBottom: "10px" }}>
              <label style={{ display: "block", marginBottom: "5px" }}>Max Depth:</label>
              <input
                type="number"
                value={hyperparams.max_depth}
                onChange={(e) => setHyperparams({ ...hyperparams, max_depth: parseInt(e.target.value, 10) || 5 })}
                style={{ width: "100%", padding: "8px" }}
              />
            </div>
            <div style={{ marginBottom: "10px" }}>
              <label style={{ display: "block", marginBottom: "5px" }}>Max Features:</label>
              <select
                value={hyperparams.max_features}
                onChange={(e) => setHyperparams({ ...hyperparams, max_features: e.target.value })}
                style={{ width: "100%", padding: "8px" }}
              >
                <option value="sqrt">sqrt</option>
                <option value="log2">log2</option>
                <option value={null}>None</option>
              </select>
            </div>
          </>
        );
      case "rf":
        return (
          <>
            <div style={{ marginBottom: "10px" }}>
              <label style={{ display: "block", marginBottom: "5px" }}>Max Depth:</label>
              <input
                type="number"
                value={hyperparams.max_depth}
                onChange={(e) => setHyperparams({ ...hyperparams, max_depth: parseInt(e.target.value, 10) || 5 })}
                style={{ width: "100%", padding: "8px" }}
              />
            </div>
            <div style={{ marginBottom: "10px" }}>
              <label style={{ display: "block", marginBottom: "5px" }}>Number of Trees:</label>
              <input
                type="number"
                value={hyperparams.n_estimators}
                onChange={(e) => setHyperparams({ ...hyperparams, n_estimators: parseInt(e.target.value, 10) || 100 })}
                style={{ width: "100%", padding: "8px" }}
              />
            </div>
            <div style={{ marginBottom: "10px" }}>
              <label style={{ display: "block", marginBottom: "5px" }}>Max Features:</label>
              <select
                value={hyperparams.max_features}
                onChange={(e) => setHyperparams({ ...hyperparams, max_features: e.target.value })}
                style={{ width: "100%", padding: "8px" }}
              >
                <option value="sqrt">sqrt</option>
                <option value="log2">log2</option>
                <option value={null}>None</option>
              </select>
            </div>
          </>
        );
      default:
        return null;
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: "20px", backgroundColor: "#222", color: "white", minHeight: "100vh" }}>
      <h1 style={{ fontSize: "24px", fontWeight: "bold" }}>Regression Model Visualization</h1>

      <div style={{ padding: "20px", backgroundColor: "#333", borderRadius: "8px", marginTop: "20px", width: "300px", textAlign: "center" }}>
        <label style={{ display: "block", marginBottom: "10px" }}>Select Model:</label>
        <select
          value={modelType}
          onChange={handleModelTypeChange}
          style={{ width: "100%", padding: "8px", marginBottom: "10px" }}
        >
          <option value="linear">Linear Regression</option>
          <option value="polynomial">Polynomial Regression</option>
          <option value="dt">Decision Tree</option>
          <option value="rf">Random Forest</option>
        </select>

        <label style={{ display: "block", marginBottom: "10px" }}>Select Dataset:</label>
        <select
          value={datasetType}
          onChange={(e) => setDatasetType(e.target.value)}
          style={{ width: "100%", padding: "8px", marginBottom: "10px" }}
        >
          <option value="linear">Linear</option>
          <option value="u_shaped">U-Shaped</option>
          <option value="concentric">Concentric</option>
        </select>

        <label style={{ display: "block", marginBottom: "10px" }}>Sample Size:</label>
        <input
          type="number"
          value={sampleSize}
          onChange={(e) => setSampleSize(parseInt(e.target.value, 10) || 0)}
          placeholder="Sample Size"
          style={{ width: "100%", padding: "8px", marginBottom: "10px" }}
        />

        {/* Render hyperparameter inputs */}
        {renderHyperparameterInputs()}

        <button
          onClick={runRegression}
          disabled={loading}
          style={{ padding: "10px", width: "100%", backgroundColor: "#555", color: "white", cursor: "pointer", border: "none" }}
        >
          {loading ? "Running..." : "Run Regression"}
        </button>
      </div>

      {plotUrl && (
        <div style={{ textAlign: "center", marginTop: "20px" }}>
          <h2 style={{ fontSize: "18px", fontWeight: "bold" }}>R² Score: {r2Score}</h2>
          <img src={plotUrl} alt="Regression Plot" style={{ marginTop: "10px", borderRadius: "8px", maxWidth: "100%" }} />
        </div>
      )}

      <button
        onClick={onBack}
        style={{ marginTop: "20px", padding: "10px 20px", backgroundColor: "#555", color: "white", cursor: "pointer", border: "none", borderRadius: "5px" }}
      >
        Back to Task Selection
      </button>
    </div>
  );
}