import { useState } from "react";
import axios from "axios";
import "./DataSelector.css";

const DatasetSelector = () => {
  const [selectedDataset, setSelectedDataset] = useState("");
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  const datasets = [
    "Linear Regression",
    "U-Shape (Polynomial Regression)",
    "Circular",
    "Moons",
    "Gaussian Blobs",
    "High-Dimensional",
    "Time-Series",
    "Sparse",
    "Binary Classification",
    "Multi-Class Classification",
  ];

  const handleDatasetSelect = async (dataset) => {
    setSelectedDataset(dataset);
    setLoading(true);
    setMessage("");

    try {
      const response = await axios.post("http://127.0.0.1:5000/get-dataset", {
        datasetName: dataset,
      });

      setMessage(`Dataset "${dataset}" loaded successfully!`);
      console.log(response.data);
    } catch (error) {
      setMessage("Error loading dataset. Try again.");
    }

    setLoading(false);
  };

  return (
    <div className="container">
      <h2>Select a Dataset</h2>

      <select onChange={(e) => handleDatasetSelect(e.target.value)}>
        <option value="">-- Choose a Dataset --</option>
        {datasets.map((dataset, index) => (
          <option key={index} value={dataset}>
            {dataset}
          </option>
        ))}
      </select>

      {loading && <p className="loading">Loading dataset...</p>}
      {message && <p className="message">{message}</p>}
    </div>
  );
};

export default DatasetSelector;
