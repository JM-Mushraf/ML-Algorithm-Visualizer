import { useState } from "react";
import axios from "axios";
import DataSelector from "./DataSelector";

const LogisticRegression = () => {
  const [dataset, setDataset] = useState("");
  const [boundaryData, setBoundaryData] = useState(null);

  const trainModel = async () => {
    try {
      const response = await axios.post("http://127.0.0.1:5000/train/logistic-regression", {
        datasetName: dataset,
      });

      console.log("Response from backend:", response.data); // Debugging log

      setBoundaryData(response.data);
    } catch (error) {
      console.error("Error fetching logistic regression model:", error);
    }
  };

  return (
    <div>
      <h2>Logistic Regression</h2>
      <DataSelector onSelect={setDataset} />
      <button onClick={trainModel}>Train Model</button>

      {boundaryData && boundaryData.slope && boundaryData.intercept ? (
        <div>
          <p>Slope: {boundaryData.slope.join(", ")}</p>
          <p>Intercept: {boundaryData.intercept.join(", ")}</p>
        </div>
      ) : (
        <p>No data available. Train the model first.</p>
      )}
    </div>
  );
};

export default LogisticRegression;
