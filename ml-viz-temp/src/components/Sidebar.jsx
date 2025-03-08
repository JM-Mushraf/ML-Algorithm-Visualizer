"use client"
import { ChevronRight, LineChart, BarChart3, Upload, Brain, Info } from "lucide-react"
import "./Sidebar.css"

function Sidebar({ activeTab, setActiveTab, menuItems, isMobileMenuOpen, setIsMobileMenuOpen }) {
  const getIcon = (iconName) => {
    switch (iconName) {
      case "line-chart":
        return <LineChart className="menu-icon" />
      case "bar-chart":
        return <BarChart3 className="menu-icon" />
      case "upload":
        return <Upload className="menu-icon" />
      case "brain":
        return <Brain className="menu-icon" />
      default:
        return null
    }
  }

  return (
    <div className={`sidebar ${isMobileMenuOpen ? "sidebar-open" : "sidebar-closed"}`}>
      <div className="sidebar-container">
        <div className="sidebar-header">
          <h2 className="sidebar-title gradient-text purple-to-cyan">ML Visualizer</h2>
        </div>

        <nav className="sidebar-nav">
          <ul className="sidebar-menu">
            {menuItems.map((item) => (
              <li key={item.id}>
                <button
                  onClick={() => {
                    setActiveTab(item.id)
                    setIsMobileMenuOpen(false)
                  }}
                  className={`sidebar-menu-item ${activeTab === item.id ? "active" : ""}`}
                >
                  <span className={`sidebar-icon ${activeTab === item.id ? "active-icon" : ""}`}>
                    {getIcon(item.icon)}
                  </span>
                  <span>{item.label}</span>
                  {activeTab === item.id && <ChevronRight className="chevron-icon" />}
                </button>
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

