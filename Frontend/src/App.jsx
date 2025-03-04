import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import './App.css';
// import LogisticRegression from './pages/LogisticRegression';
import LinearRegression from './pages/LinearRegression';
import DataSelector from './pages/DataSelector';

function App() {
  return (
    <Router>
      <div className="container">
        <h1>Algo Visualizer</h1>
        <Link to="/linear-regression">
          <button>LinearRegression</button>
        </Link>
        <Routes>
          <Route path="/datasets" element={<DataSelector />} />
          {/* <Route path="/logistic-regression" element={<LogisticRegression />} /> */}
          <Route path="/linear-regression" element={<LinearRegression />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
