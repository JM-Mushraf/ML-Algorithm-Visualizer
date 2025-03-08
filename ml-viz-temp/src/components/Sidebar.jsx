"use client"

import { NavLink } from "react-router-dom"
import { ChevronRight, LineChart, BarChart3, Upload, Brain, Info } from "lucide-react"
import "./Sidebar.css"

function Sidebar({ isMobileMenuOpen, setIsMobileMenuOpen }) {
  const menuItems = [
    { path: "/", label: "Home", icon: <LineChart className="menu-icon" /> },
    { path: "/regression", label: "Regression Visualizer", icon: <LineChart className="menu-icon" /> },
    { path: "/classification", label: "Classification Visualizer", icon: <BarChart3 className="menu-icon" /> },
    { path: "/upload", label: "Upload Dataset", icon: <Upload className="menu-icon" /> },
    { path: "/algorithms", label: "Learn Algorithms", icon: <Brain className="menu-icon" /> },
  ]

  return (
    <div className={`sidebar ${isMobileMenuOpen ? "sidebar-open" : "sidebar-closed"}`}>
      <div className="sidebar-container">
        <div className="sidebar-header">
          <h2 className="sidebar-title gradient-text purple-to-cyan">ML Visualizer</h2>
        </div>

        <nav className="sidebar-nav">
          <ul className="sidebar-menu">
            {menuItems.map((item) => (
              <li key={item.path}>
                <NavLink
                  to={item.path}
                  className={({ isActive }) => `sidebar-menu-item ${isActive ? "active" : ""}`}
                  onClick={() => setIsMobileMenuOpen(false)}
                >
                  <span className="sidebar-icon">{item.icon}</span>
                  <span>{item.label}</span>
                  {/** Show Chevron only if active */}
                  <ChevronRight className="chevron-icon" />
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>

        <div className="sidebar-footer">
          <button className="about-button">
            <Info className="info-icon" />
            <span>About</span>
          </button>
        </div>
      </div>
    </div>
  )
}

export default Sidebar
