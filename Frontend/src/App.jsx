import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import LinearRegression from "./pages/LinearRegression";
import LogisticRegression from "./pages/LogisticRegression";

const App = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/linear-regression" element={<LinearRegression />} />
        <Route path="/logistic-regression" element={<LogisticRegression />} />
      </Routes>
    </Router>
  );
};

export default App;
