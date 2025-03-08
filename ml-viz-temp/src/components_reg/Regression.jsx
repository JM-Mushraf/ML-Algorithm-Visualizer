import { useState } from "react";
import { Select } from "./Select";
import { Slider } from "./Slider";
import { Button } from "./Button";
import { Input } from "./Input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./Tabs";
import { Card, CardContent } from "./Card";
import { Badge } from "./Badge";
import axios from "axios";
import "./Regression.css";

import {
  LineChart,
  ScatterChart,
  Brain,
  Layers,
  Cpu,
  BarChart4,
  GitBranch,
  MessageSquare,
  ChevronRight,
  Sparkles,
} from "./Icons";

function Regression() {
  const [algorithm, setAlgorithm] = useState("kmeans");
  const [dataset, setDataset] = useState("linear");
  const [sampleSize, setSampleSize] = useState(100);
  const [isRunning, setIsRunning] = useState(false);
  const [chatMessage, setChatMessage] = useState("");
  const [chatHistory, setChatHistory] = useState([
    "Welcome to ML Visualizer! How can I help you understand the algorithms?",
  ]);
  const [visualizationData, setVisualizationData] = useState({
    plotBase64: null,
    r2Score: null,
  });

  // Hyperparameters state
  const [kmeansClusters, setKmeansClusters] = useState(3);
  const [maxIterations, setMaxIterations] = useState(100);
  const [learningRate, setLearningRate] = useState(0.01);
  const [regularization, setRegularization] = useState(0.1);
  const [hiddenLayers, setHiddenLayers] = useState(2);
  const [neuronsPerLayer, setNeuronsPerLayer] = useState(64);
  const [activationFunction, setActivationFunction] = useState("relu");

  const runAlgorithm = async () => {
    setIsRunning(true);

    // Prepare hyperparameters based on the selected algorithm
    const hyperparameters = {};
    if (algorithm === "kmeans") {
      hyperparameters.kmeansClusters = kmeansClusters;
      hyperparameters.maxIterations = maxIterations;
    } else if (algorithm === "linear") {
      hyperparameters.learningRate = learningRate;
      hyperparameters.regularization = regularization;
    } else if (algorithm === "neural_network") {
      hyperparameters.hiddenLayers = hiddenLayers;
      hyperparameters.neuronsPerLayer = neuronsPerLayer;
      hyperparameters.activationFunction = activationFunction;
    }

    try {
      console.log(algorithm, dataset, hyperparameters);

      const response = await axios.post("http://127.0.0.1:5000/regression", {
        model_type: algorithm,
        dataset_type: dataset,
        sample_size: sampleSize,
        hyperparameters: hyperparameters,
      });

      // Handle the response (e.g., update the visualization, display results, etc.)
      console.log(response.data);

      // Update visualization data
      setVisualizationData({
        plotBase64: response.data.plot_base64,
        r2Score: response.data.r2_score,
      });
    } catch (error) {
      console.error("Error running algorithm:", error);
    } finally {
      setIsRunning(false);
    }
  };

  const sendMessage = (e) => {
    e.preventDefault();
    if (!chatMessage.trim()) return;

    setChatHistory([...chatHistory, `You: ${chatMessage}`]);

    // Simulate AI response
    setTimeout(() => {
      let response = "";
      if (chatMessage.toLowerCase().includes("explain")) {
        response = `The ${algorithm} algorithm works by clustering data points based on similarity. It's commonly used for unsupervised learning tasks.`;
      } else if (chatMessage.toLowerCase().includes("dataset")) {
        response = `The ${dataset} dataset contains features that are well-suited for classification tasks. It's a good starting point for understanding ML algorithms.`;
      } else {
        response = "I can explain algorithms, datasets, or parameters. What would you like to know more about?";
      }
      setChatHistory((prev) => [...prev, response]);
    }, 1000);

    setChatMessage("");
  };

  return (
    <main className="main">
      <div className="container">
        <header className="header">
          <div className="header-content">
            <div className="logo">
              <Brain className="logo-icon" />
              <h1 className="title">ML Algorithm Visualizer</h1>
            </div>
            <Badge variant="outline" className="badge">
              <Sparkles className="badge-icon" />
              <span>Interactive Learning</span>
            </Badge>
          </div>
        </header>

        <div className="grid">
          {/* Sidebar */}
          <div className="sidebar-rr">
            {/* Algorithm Selection */}
            <Card className="card" id="selc-op">
              <CardContent className="card-content">
                <h2 className="section-title">
                  <Cpu className="section-icon" />
                  Choose Algorithm
                </h2>
                <Select
                  value={algorithm}
                  onValueChange={setAlgorithm}
                  options={[
                    { value: "linear", label: "Linear Regression" },
                    { value: "polynomial", label: "polynomial" },
                    { value: "dt", label: "Decision Tree" },
                    { value: "rf", label: "Random Forest" },
                    { value: "svm", label: "Support Vector Machine" },
                  ]}
                />
              </CardContent>
            </Card>

            {/* Dataset Configuration */}
            <Card className="card">
              <CardContent className="card-content">
                <h2 className="section-title">
                  <BarChart4 className="section-icon" />
                  Dataset Configuration
                </h2>
                <div className="form-group">
                  <div className="form-field" id="selc-op2">
                    <label className="form-label">Choose Dataset</label>
                    <Select
                      value={dataset}
                      onValueChange={setDataset}
                      options={[
                        { value: "linear", label: "Linear dataset" },
                        { value: "u_shaped", label: "u_shaped" },
                        { value: "concentric", label: "concentric" },
                        // { value: "wine", label: "Wine Quality" },
                        // { value: "custom", label: "Custom Dataset" },
                      ]}
                    />
                  </div>

                  <div className="form-field">
                    <label className="form-label">Sample Size</label>
                    <div className="slider-container">
                      <Slider
                        value={[sampleSize]}
                        onValueChange={(values) => setSampleSize(values[0])}
                        min={10}
                        max={500}
                        step={10}
                      />
                      <span className="slider-value">{sampleSize}</span>
                    </div>
                  </div>

                  <Button variant="outline" className="custom-dataset-btn">
                    <GitBranch className="btn-icon" />
                    Create Custom Dataset
                  </Button>
                </div>
              </CardContent>
            </Card>

            {/* Hyperparameters */}
            <Card className="card">
              <CardContent className="card-content">
                <h2 className="section-title">
                  <Layers className="section-icon" />
                  Hyperparameters
                </h2>
                <div className="form-group">
                  {algorithm === "kmeans" && (
                    <>
                      <div className="form-field">
                        <label className="form-label">Number of Clusters (k)</label>
                        <Slider
                          value={[kmeansClusters]}
                          onValueChange={(values) => setKmeansClusters(values[0])}
                          min={2}
                          max={10}
                          step={1}
                        />
                        <div className="slider-range">
                          <span>2</span>
                          <span>10</span>
                        </div>
                        <span className="slider-value">{kmeansClusters}</span>
                      </div>
                      <div className="form-field">
                        <label className="form-label">Max Iterations</label>
                        <Slider
                          value={[maxIterations]}
                          onValueChange={(values) => setMaxIterations(values[0])}
                          min={10}
                          max={300}
                          step={10}
                        />
                        <div className="slider-range">
                          <span>10</span>
                          <span>300</span>
                        </div>
                        <span className="slider-value">{maxIterations}</span>
                      </div>
                    </>
                  )}

                  {algorithm === "linear" && (
                    <>
                      <div className="form-field">
                        <label className="form-label">Learning Rate</label>
                        <Slider
                          value={[learningRate]}
                          onValueChange={(values) => setLearningRate(values[0])}
                          min={0.001}
                          max={0.1}
                          step={0.001}
                        />
                        <div className="slider-range">
                          <span>0.001</span>
                          <span>0.1</span>
                        </div>
                        <span className="slider-value">{learningRate}</span>
                      </div>
                      <div className="form-field">
                        <label className="form-label">Regularization</label>
                        <Slider
                          value={[regularization]}
                          onValueChange={(values) => setRegularization(values[0])}
                          min={0}
                          max={1}
                          step={0.1}
                        />
                        <div className="slider-range">
                          <span>0</span>
                          <span>1.0</span>
                        </div>
                        <span className="slider-value">{regularization}</span>
                      </div>
                    </>
                  )}

                  {algorithm === "neural_network" && (
                    <>
                      <div className="form-field">
                        <label className="form-label">Hidden Layers</label>
                        <Slider
                          value={[hiddenLayers]}
                          onValueChange={(values) => setHiddenLayers(values[0])}
                          min={1}
                          max={5}
                          step={1}
                        />
                        <div className="slider-range">
                          <span>1</span>
                          <span>5</span>
                        </div>
                        <span className="slider-value">{hiddenLayers}</span>
                      </div>
                      <div className="form-field">
                        <label className="form-label">Neurons per Layer</label>
                        <Slider
                          value={[neuronsPerLayer]}
                          onValueChange={(values) => setNeuronsPerLayer(values[0])}
                          min={8}
                          max={128}
                          step={8}
                        />
                        <div className="slider-range">
                          <span>8</span>
                          <span>128</span>
                        </div>
                        <span className="slider-value">{neuronsPerLayer}</span>
                      </div>
                      <div className="form-field">
                        <label className="form-label">Activation Function</label>
                        <Select
                          value={activationFunction}
                          onValueChange={setActivationFunction}
                          options={[
                            { value: "relu", label: "ReLU" },
                            { value: "sigmoid", label: "Sigmoid" },
                            { value: "tanh", label: "Tanh" },
                          ]}
                        />
                      </div>
                    </>
                  )}
                </div>

                <Button className="run-btn" onClick={runAlgorithm} disabled={isRunning}>
                  {isRunning ? (
                    <>
                      <div className="spinner" />
                      Running...
                    </>
                  ) : (
                    "Run Algorithm"
                  )}
                </Button>
              </CardContent>
            </Card>

            {/* Chat Bot */}
            <Card className="card">
              <CardContent className="card-content">
                <h2 className="section-title">
                  <MessageSquare className="section-icon" />
                  ML Assistant
                </h2>
                <div className="chat-container">
                  {chatHistory.map((message, index) => (
                    <div
                      key={index}
                      className={`chat-message ${message.startsWith("You:") ? "user-message" : "bot-message"}`}
                    >
                      {message}
                    </div>
                  ))}
                </div>
                <form onSubmit={sendMessage} className="chat-form">
                  <Input
                    value={chatMessage}
                    onChange={(e) => setChatMessage(e.target.value)}
                    placeholder="Ask about algorithms..."
                    className="chat-input"
                  />
                  <Button type="submit" size="icon" variant="outline" className="chat-submit">
                    <ChevronRight className="submit-icon" />
                  </Button>
                </form>
              </CardContent>
            </Card>
          </div>

          {/* Main Content */}
          <div className="main-content">
            <Tabs defaultValue="visualization" className="tabs">
              <TabsList className="tabs-list">
                <TabsTrigger value="visualization" className="tab">
                  <LineChart className="tab-icon" />
                  Visualization
                </TabsTrigger>
                <TabsTrigger value="animation" className="tab">
                  <ScatterChart className="tab-icon" />
                  Animation
                </TabsTrigger>
              </TabsList>

              <TabsContent value="visualization" className="tab-content">
                <Card className="visualization-card">
                  <CardContent className="visualization-content">
                    <div className="visualization-container">
                      {/* Visualization Canvas */}
                      <div className="canvas" id="visualization-canvas">
                        {visualizationData.plotBase64 ? (
                          <>
                            <img
                              src={`data:image/png;base64,${visualizationData.plotBase64}`}
                              alt="Regression Plot"
                              style={{ width: "100%", height: "auto" }}
                            />
                            <p>R² Score: {visualizationData.r2Score.toFixed(4)}</p>
                          </>
                        ) : (
                          <div className="placeholder">
                            <LineChart className="placeholder-icon" />
                            <h3 className="placeholder-title">Visualization Area</h3>
                            <p className="placeholder-text">
                              Select an algorithm and dataset, then click "Run Algorithm" to see the visualization here.
                            </p>
                            <Button onClick={runAlgorithm} className="run-btn">
                              Run Algorithm
                            </Button>
                          </div>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="animation" className="tab-content">
                <Card className="visualization-card">
                  <CardContent className="visualization-content">
                    <div className="visualization-container">
                      {/* Animation Canvas */}
                      <div className="canvas" id="animation-canvas"></div>

                      {/* Placeholder content when no algorithm is running */}
                      {!isRunning && (
                        <div className="placeholder">
                          <ScatterChart className="placeholder-icon" />
                          <h3 className="placeholder-title">Algorithm Animation</h3>
                          <p className="placeholder-text">
                            Watch the algorithm work step by step with an animated visualization of each iteration.
                          </p>
                          <Button onClick={runAlgorithm} className="run-btn">
                            Run Animation
                          </Button>
                        </div>
                      )}
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>

            {/* Results and Metrics */}
            <Card className="metrics-card">
              <CardContent className="card-content">
                <h2 className="section-title">Results & Metrics</h2>
                <div className="metrics-grid">
                  <div className="metric">
                    <h3 className="metric-label">Accuracy</h3>
                    <div className="metric-value accuracy">
                      {visualizationData.r2Score !== null
                        ? visualizationData.r2Score.toFixed(4) // Display R² score with 4 decimal places
                        : "--"}
                    </div>
                  </div>
                  <div className="metric">
                    <h3 className="metric-label">Iterations</h3>
                    <div className="metric-value iterations">--</div>
                  </div>
                  <div className="metric">
                    <h3 className="metric-label">Execution Time</h3>
                    <div className="metric-value time">--</div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </main>
  );
}

export default Regression;