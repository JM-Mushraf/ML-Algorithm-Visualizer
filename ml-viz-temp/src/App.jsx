import { useState } from "react"
import { AnimatePresence } from "framer-motion"
import ParticleBackground from "./components/ParticleBackground"
import Sidebar from "./components/Sidebar"
import HomeContent from "./components/HomeContent"
// import RegressionContent from "./components/RegressionContent"
import Regression from "./components_reg/Regression"
import ClassificationContent from "./components/ClassificationContent"
import UploadContent from "./components/UploadContent"
import AlgorithmsContent from "./components/AlgorithmsContent"
import { Menu, X } from "lucide-react"
import "./App.css"

function App() {
  const [activeTab, setActiveTab] = useState("home")
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  const menuItems = [
    { id: "regression", label: "Regression Visualizer", icon: "line-chart" },
    { id: "classification", label: "Classification Visualizer", icon: "bar-chart" },
    { id: "upload", label: "Upload Dataset", icon: "upload" },
    { id: "algorithms", label: "Learn Algorithms", icon: "brain" },
  ]

  return (
    <div className="app">
      {/* Animated background */}
      <ParticleBackground />

      {/* Mobile menu button */}
      <div className="mobile-menu-button">
        <button onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)} className="menu-button">
          {isMobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
        </button>
      </div>

      {/* Sidebar */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        menuItems={menuItems}
        isMobileMenuOpen={isMobileMenuOpen}
        setIsMobileMenuOpen={setIsMobileMenuOpen}
      />

      {/* Main content */}
      <main className={`main-content ${isMobileMenuOpen ? "blur" : ""}`}>
        <AnimatePresence mode="wait">
          {activeTab === "home" && <HomeContent key="home" />}

          {activeTab === "regression" && <Regression key="regression" />}

          {activeTab === "classification" && <ClassificationContent key="classification" />}

          {activeTab === "upload" && <UploadContent key="upload" />}

          {activeTab === "algorithms" && <AlgorithmsContent key="algorithms" />}
        </AnimatePresence>
      </main>
    </div>
  )
}

export default App

