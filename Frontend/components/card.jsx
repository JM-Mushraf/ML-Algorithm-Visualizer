// components/Card.jsx
import React from "react";
import "./Card.css"; // Import CSS for styling

const Card = ({ children }) => {
  return <div className="card">{children}</div>;
};

export default Card;
