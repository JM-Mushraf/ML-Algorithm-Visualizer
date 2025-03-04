import { Link } from "react-router-dom";

const Home = () => {
  return (
    <div>
      <h1>ML Model Trainer</h1>
      <ul>
        <li><Link to="/linear-regression">Linear Regression</Link></li>
        <li><Link to="/logistic-regression">Logistic Regression</Link></li>
      </ul>
    </div>
  );
};

export default Home;
