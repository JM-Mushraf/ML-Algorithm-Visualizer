import { useState } from "react";
import axios from "axios";

export default function ClassificationApp({ onBack }) {
  const [modelType, setModelType] = useState("logistic");
  const [datasetType, setDatasetType] = useState("linear");
  const [sampleSize, setSampleSize] = useState(200);
  const [noise, setNoise] = useState(0.1);
  const [plotUrl, setPlotUrl] = useState(null);
  const [accuracy, setAccuracy] = useState(null);
  const [loading, setLoading] = useState(false);

  const runClassification = async () => {
    setLoading(true);
    try {
      const response = await axios.post("http://127.0.0.1:5000/classification", {
        model_type: modelType,
        dataset_type: datasetType,
        sample_size: sampleSize,
        noise: noise,
      });
      setPlotUrl(response.data.plot_url);
      setAccuracy(response.data.accuracy);
    } catch (error) {
      console.error("Error running classification:", error);
    }
    setLoading(false);
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: "20px", backgroundColor: "#222", color: "white", minHeight: "100vh" }}>
      <h1 style={{ fontSize: "24px", fontWeight: "bold" }}>Classification Model Visualization</h1>

      <div style={{ padding: "20px", backgroundColor: "#333", borderRadius: "8px", marginTop: "20px", width: "300px", textAlign: "center" }}>
        <label style={{ display: "block", marginBottom: "10px" }}>Select Model:</label>
        <select value={modelType} onChange={(e) => setModelType(e.target.value)} style={{ width: "100%", padding: "8px", marginBottom: "10px" }}>
          <option value="logistic">Logistic Regression</option>
          <option value="dt">Decision Tree</option>
          <option value="rf">Random Forest</option>
          <option value="svm">SVM</option>
        </select>

        <label style={{ display: "block", marginBottom: "10px" }}>Select Dataset:</label>
        <select value={datasetType} onChange={(e) => setDatasetType(e.target.value)} style={{ width: "100%", padding: "8px", marginBottom: "10px" }}>
          <option value="linear">Linear</option>
          <option value="moons">Moons</option>
          <option value="circles">Circles</option>
        </select>

        <label style={{ display: "block", marginBottom: "10px" }}>Sample Size:</label>
        <input
          type="number"
          value={sampleSize}
          onChange={(e) => setSampleSize(parseInt(e.target.value, 10) || 0)}
          placeholder="Sample Size"
          style={{ width: "100%", padding: "8px", marginBottom: "10px" }}
        />

        <label style={{ display: "block", marginBottom: "10px" }}>Noise Level:</label>
        <input
          type="number"
          step="0.1"
          value={noise}
          onChange={(e) => setNoise(parseFloat(e.target.value))}
          placeholder="Noise Level"
          style={{ width: "100%", padding: "8px", marginBottom: "10px" }}
        />

        <button onClick={runClassification} disabled={loading} style={{ padding: "10px", width: "100%", backgroundColor: "#555", color: "white", cursor: "pointer", border: "none" }}>
          {loading ? "Running..." : "Run Classification"}
        </button>
      </div>

      {plotUrl && (
        <div style={{ textAlign: "center", marginTop: "20px" }}>
          <h2 style={{ fontSize: "18px", fontWeight: "bold" }}>Accuracy: {accuracy}</h2>
          <img src={plotUrl} alt="Classification Plot" style={{ marginTop: "10px", borderRadius: "8px", maxWidth: "100%" }} />
        </div>
      )}

      <button onClick={onBack} style={{ marginTop: "20px", padding: "10px 20px", backgroundColor: "#555", color: "white", cursor: "pointer", border: "none", borderRadius: "5px" }}>
        Back to Task Selection
      </button>
    </div>
  );
}