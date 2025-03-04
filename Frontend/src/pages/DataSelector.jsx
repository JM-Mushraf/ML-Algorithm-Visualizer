import { useState } from "react";

const DatasetSelector = ({ onSelect }) => {
  const datasets = ["iris", "tips", "diamonds", "penguins"]; // Example datasets
  return (
    <select onChange={(e) => onSelect(e.target.value)}>
      <option value="">Select Dataset</option>
      {datasets.map((ds) => (
        <option key={ds} value={ds}>{ds}</option>
      ))}
    </select>
  );
};

export default DatasetSelector;
