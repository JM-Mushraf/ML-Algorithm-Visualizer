import React, { useState } from 'react';
import axios from 'axios';
import Chart from './Chart'; // Import the Chart component

const LinearRegression = () => {
  const [X, setX] = useState([1, 2, 3, 4, 5]);
  const [y, setY] = useState([1, 2, 3, 4, 5]);
  const [result, setResult] = useState(null);
  const [predictedY, setPredictedY] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async () => {
    if (X.length === 0 || y.length === 0) {
      setError('Please provide values for X and y.');
      return;
    }
  
    setLoading(true);
    setError(null);
  
    try {
      const response = await axios.post('http://localhost:5000/api/linear-regression', { X, y });
  
      const { coefficients, intercept } = response.data;
  
      if (!coefficients || coefficients.length === 0) {
        throw new Error("Invalid response from server");
      }
  
      setResult(coefficients);
  
      // Compute predicted Y values using y = mX + b
      const m = coefficients[0]; // Slope
      const b = intercept; // Intercept
      const predictions = X.map((x) => m * x + b);
      setPredictedY(predictions);
    } catch (error) {
      console.error('Error:', error);
      setError('An error occurred while running the algorithm.');
    } finally {
      setLoading(false);
    }
  };
  

  return (
    <div>
      <h2>Linear Regression</h2>

      <div>
        <label>
          X (Features):
          <input
            type="text"
            value={X.join(', ')}
            onChange={(e) => setX(e.target.value.split(',').map(Number))}
          />
        </label>
      </div>
      <div>
        <label>
          y (Target):
          <input
            type="text"
            value={y.join(', ')}
            onChange={(e) => setY(e.target.value.split(',').map(Number))}
          />
        </label>
      </div>

      <button onClick={handleSubmit} disabled={loading}>
        {loading ? 'Running...' : 'Run Algorithm'}
      </button>

      {result && (
        <div>
          <h3>Results:</h3>
          <p>Coefficients: {result.join(', ')}</p>
        </div>
      )}

      {error && <p style={{ color: 'red' }}>{error}</p>}

      {/* Display the Chart */}
      {result && (
        <Chart
          data={{
            labels: X,
            values: y,
            predicted: predictedY,
          }}
        />
      )}
    </div>
  );
};

export default LinearRegression;
