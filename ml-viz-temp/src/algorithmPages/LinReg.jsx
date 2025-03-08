import React from "react";
import { motion } from "framer-motion";
import { LineChart, Brain, Code, ArrowRight, Github, Linkedin, Youtube, BarChart2, Lightbulb, AlertCircle } from "lucide-react";
import "./LinReg.css";

const LinearRegressionPage = () => {
  return (
    <div className="linear-regression-page min-h-screen bg-gradient-to-br from-gray-900 to-black text-white font-sans">
      {/* Hero Section */}
      <motion.section
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1 }}
        className="linear-hero flex flex-col items-center justify-center h-screen text-center px-4"
      >
        <h1 className="linear-hero-title text-6xl font-bold text-blue-400 mb-4">
          Linear Regression
        </h1>
        <p className="linear-hero-subtitle text-xl text-gray-300 mb-8">
          The backbone of predictive analytics and machine learning.
        </p>
        <motion.button
          whileHover={{ scale: 1.1 }}
          whileTap={{ scale: 0.9 }}
          className="linear-hero-button flex items-center bg-blue-600 text-white px-6 py-3 rounded-lg shadow-lg hover:bg-blue-500 transition-all"
        >
          <span>Explore More</span>
          <ArrowRight className="ml-2" />
        </motion.button>
      </motion.section>

      {/* What is Linear Regression? */}
      <motion.section
        initial={{ opacity: 0, y: 50 }}
        whileInView={{ opacity: 1, y: 0 }}
        transition={{ duration: 1 }}
        className="linear-what-is py-20 px-4 bg-gray-800"
      >
        <div className="linear-what-is-container max-w-5xl mx-auto text-center">
          <h2 className="text-4xl font-bold text-blue-400 mb-8">What is Linear Regression?</h2>
          <p className="text-lg text-gray-300 mb-8">
            Linear regression is a statistical technique used to model the relationship between a dependent variable and one or more independent variables. It is fundamental in predictive modeling and data science.
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <motion.div
              whileHover={{ scale: 1.05 }}
              className="p-6 bg-gray-900 rounded-lg shadow-md"
            >
              <LineChart className="w-12 h-12 mx-auto mb-4 text-blue-400" />
              <h3 className="text-xl font-semibold text-white mb-2">Prediction</h3>
              <p className="text-gray-400">Forecast future trends with data.</p>
            </motion.div>
            <motion.div
              whileHover={{ scale: 1.05 }}
              className="p-6 bg-gray-900 rounded-lg shadow-md"
            >
              <Brain className="w-12 h-12 mx-auto mb-4 text-blue-400" />
              <h3 className="text-xl font-semibold text-white mb-2">Data Modeling</h3>
              <p className="text-gray-400">Understand relationships between variables.</p>
            </motion.div>
            <motion.div
              whileHover={{ scale: 1.05 }}
              className="p-6 bg-gray-900 rounded-lg shadow-md"
            >
              <Code className="w-12 h-12 mx-auto mb-4 text-blue-400" />
              <h3 className="text-xl font-semibold text-white mb-2">Easy to Implement</h3>
              <p className="text-gray-400">Simple yet powerful in real-world applications.</p>
            </motion.div>
          </div>
        </div>
      </motion.section>

      {/* Real-World Applications */}
      <motion.section className="py-20 px-4 bg-gray-900">
        <div className="max-w-5xl mx-auto text-center">
          <h2 className="text-4xl font-bold text-blue-400 mb-8">Real-World Applications</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <motion.div whileHover={{ scale: 1.02 }} className="p-6 bg-gray-800 rounded-lg shadow-md">
              <BarChart2 className="w-12 h-12 mx-auto mb-4 text-green-400" />
              <h3 className="text-xl font-semibold text-white mb-2">Stock Market Prediction</h3>
              <p className="text-gray-400">Used to analyze trends and forecast prices.</p>
            </motion.div>
            <motion.div whileHover={{ scale: 1.02 }} className="p-6 bg-gray-800 rounded-lg shadow-md">
              <Lightbulb className="w-12 h-12 mx-auto mb-4 text-yellow-400" />
              <h3 className="text-xl font-semibold text-white mb-2">Energy Consumption</h3>
              <p className="text-gray-400">Predict energy demand based on historical data.</p>
            </motion.div>
            <motion.div whileHover={{ scale: 1.02 }} className="p-6 bg-gray-800 rounded-lg shadow-md">
              <AlertCircle className="w-12 h-12 mx-auto mb-4 text-red-400" />
              <h3 className="text-xl font-semibold text-white mb-2">Risk Assessment</h3>
              <p className="text-gray-400">Used in insurance and finance for risk predictions.</p>
            </motion.div>
            <motion.div whileHover={{ scale: 1.02 }} className="p-6 bg-gray-800 rounded-lg shadow-md">
              <Brain className="w-12 h-12 mx-auto mb-4 text-purple-400" />
              <h3 className="text-xl font-semibold text-white mb-2">AI & Machine Learning</h3>
              <p className="text-gray-400">Forms the basis of regression models in ML.</p>
            </motion.div>
          </div>
        </div>
      </motion.section>

      {/* Visualization */}
      <motion.section className="py-20 px-4 bg-gray-800">
        <div className="max-w-5xl mx-auto text-center">
          <h2 className="text-4xl font-bold text-blue-400 mb-8">Visualization</h2>
          <div className="p-8 rounded-lg shadow-md">
            <img
              src="https://miro.medium.com/v2/resize:fit:1400/1*TpbxEQy4ckB-g31PwUQPlg.png"
              alt="Linear Regression Visualization"
              className="w-full rounded-lg"
            />
          </div>
        </div>
      </motion.section>

      {/* Footer */}
      <footer className="py-10 bg-black text-gray-400 text-center">
        <div className="flex justify-center space-x-6 mb-6">
          <a href="https://github.com" target="_blank" rel="noopener noreferrer">
            <Github className="w-6 h-6 hover:text-blue-400 transition-all" />
          </a>
          <a href="https://linkedin.com" target="_blank" rel="noopener noreferrer">
            <Linkedin className="w-6 h-6 hover:text-blue-400 transition-all" />
          </a>
          <a href="https://youtube.com" target="_blank" rel="noopener noreferrer">
            <Youtube className="w-6 h-6 hover:text-blue-400 transition-all" />
          </a>
        </div>
        <p className="text-sm">&copy; 2025 Linear Regression Guide. All rights reserved.</p>
      </footer>
    </div>
  );
};

export default LinearRegressionPage;
