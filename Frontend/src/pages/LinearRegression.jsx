import React, { useState } from 'react';
import axios from 'axios';

const LogisticRegression = () => {
  const [features, setFeatures] = useState([5.1, 3.5, 1.4, 0.2]); 
  const [prediction, setPrediction] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post('http://localhost:5000/api/logistic-regression', {
        features: features.map(Number), // Ensure numeric values
      });
      console.log(response);
      

      setPrediction(response.data.prediction);
    } catch (error) {
      console.error('Error:', error);
      setError('An error occurred while making the prediction.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2>Logistic Regression (Iris Dataset)</h2>

      <div>
        <label>
          Features (Sepal & Petal Length/Width):
          <input
            type="text"
            value={features.join(', ')}
            onChange={(e) => setFeatures(e.target.value.split(',').map(Number))}
          />
        </label>
      </div>

      <button onClick={handleSubmit} disabled={loading}>
        {loading ? 'Predicting...' : 'Make Prediction'}
      </button>

      {prediction && (
        <div>
          <h3>Prediction:</h3>
          <p>Predicted Species: {prediction}</p>
        </div>
      )}

      {error && <p style={{ color: 'red' }}>{error}</p>}
    </div>
  );
};

export default LogisticRegression;
