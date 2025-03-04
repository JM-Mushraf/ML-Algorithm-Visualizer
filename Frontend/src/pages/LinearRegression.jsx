import { useState } from "react";
import axios from "axios";
import DataSelector from "../components/DataSelector";
import { Line } from "react-chartjs-2";

const LinearRegression = () => {
  const [dataset, setDataset] = useState("");
  const [graphData, setGraphData] = useState(null);

  const trainModel = async () => {
    const response = await axios.post("http://127.0.0.1:5000/train/linear-regression", {
      datasetName: dataset
    });
    setGraphData(response.data);
  };

  return (
    <div>
      <h2>Linear Regression</h2>
      <DataSelector onSelect={setDataset} />
      <button onClick={trainModel}>Train Model</button>

      {graphData && (
        <Line
          data={{
            labels: graphData.x_values,
            datasets: [{ label: "Best Fit Line", data: graphData.y_values, borderColor: "blue" }]
          }}
        />
      )}
    </div>
  );
};

export default LinearRegression;
