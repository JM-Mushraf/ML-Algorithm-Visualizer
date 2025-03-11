import { useState } from "react"
import { AnimatePresence } from "framer-motion"
import ParticleBackground from "./components/ParticleBackground"
import Sidebar from "./components/Sidebar"
import HomeContent from "./components/HomeContent"
import Regression from "./components_reg/Regression"
import ClassificationContent from "./components/ClassificationContent"
import UploadContent from "./components/UploadContent"
import AlgorithmsContent from "./components/AlgorithmsContent"
import { Menu, X } from "lucide-react"
import { BrowserRouter as Router, Routes, Route } from "react-router-dom"
import "./App.css"
import LinReg from "./algorithmPages/LinReg"
import Chatbot from "./chatbot/Chatbot"

function App() {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)

  return (
    <Router>
      <div className="app">
        <ParticleBackground />

        {/* Mobile menu button */}
        <div className="mobile-menu-button">
          <button onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)} className="menu-button">
            {isMobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
          </button>
        </div>

        {/* Sidebar */}
        <Sidebar isMobileMenuOpen={isMobileMenuOpen} setIsMobileMenuOpen={setIsMobileMenuOpen} />

        {/* Main content */}
        <main className={`main-content ${isMobileMenuOpen ? "blur" : ""}`}>
          <AnimatePresence mode="wait">
            <Routes>
              <Route path="/" element={<HomeContent />} />
              <Route path="/regression" element={<Regression />} />
              <Route path="/classification" element={<ClassificationContent />} />
              <Route path="/upload" element={<UploadContent />} />
              <Route path="/algorithms" element={<AlgorithmsContent />} />
              <Route path="/linear-regression" element={<LinReg/>} />
            </Routes>
            <Chatbot/>
          </AnimatePresence>
        </main>
      </div>
    </Router>
  )
}

export default App
