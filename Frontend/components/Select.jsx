// components/Select.jsx
import React from "react";
import "./Select.css";

const Select = ({ options, onChange }) => {
  return (
    <select className="select" onChange={onChange}>
      {options.map((option, index) => (
        <option key={index} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  );
};

export default Select;
