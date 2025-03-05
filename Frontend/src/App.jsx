import React, { useState } from "react";
import RegressionApp from "../components/Regression";
import ClassificationApp from "../components/Classification";

const App = () => {
  const [selectedTask, setSelectedTask] = useState(null);

  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", padding: "20px", backgroundColor: "#222", color: "white", minHeight: "100vh" }}>
      {!selectedTask ? (
        <div style={{ textAlign: "center" }}>
          <h1 style={{ fontSize: "24px", fontWeight: "bold", marginBottom: "20px" }}>Choose a Task</h1>
          <button
            onClick={() => setSelectedTask("regression")}
            style={{ padding: "10px 20px", margin: "10px", backgroundColor: "#555", color: "white", cursor: "pointer", border: "none", borderRadius: "5px" }}
          >
            Regression
          </button>
          <button
            onClick={() => setSelectedTask("classification")}
            style={{ padding: "10px 20px", margin: "10px", backgroundColor: "#555", color: "white", cursor: "pointer", border: "none", borderRadius: "5px" }}
          >
            Classification
          </button>
        </div>
      ) : selectedTask === "regression" ? (
        <RegressionApp onBack={() => setSelectedTask(null)} />
      ) : (
        <ClassificationApp onBack={() => setSelectedTask(null)} />
      )}
    </div>
  );
};

export default App;