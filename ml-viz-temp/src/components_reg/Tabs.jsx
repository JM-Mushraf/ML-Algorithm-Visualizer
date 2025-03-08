import { createContext, useContext, useState } from "react";
import "./Tabs.css";

const TabsContext = createContext(undefined);

export function Tabs({ defaultValue, children, className = "" }) {
  const [selectedTab, setSelectedTab] = useState(defaultValue);

  return (
    <TabsContext.Provider value={{ selectedTab, setSelectedTab }}>
      <div className={`tabs ${className}`}>{children}</div>
    </TabsContext.Provider>
  );
}

export function TabsList({ children, className = "" }) {
  return <div className={`tabs-list ${className}`}>{children}</div>;
}

export function TabsTrigger({ value, children, className = "" }) {
  const context = useContext(TabsContext);

  if (!context) {
    throw new Error("TabsTrigger must be used within a Tabs component");
  }

  const { selectedTab, setSelectedTab } = context;
  const isSelected = selectedTab === value;

  return (
    <button
      type="button"
      role="tab"
      aria-selected={isSelected}
      data-state={isSelected ? "active" : "inactive"}
      onClick={() => setSelectedTab(value)}
      className={`tabs-trigger ${isSelected ? "active" : ""} ${className}`}
    >
      {children}
    </button>
  );
}

export function TabsContent({ value, children, className = "" }) {
  const context = useContext(TabsContext);

  if (!context) {
    throw new Error("TabsContent must be used within a Tabs component");
  }

  const { selectedTab } = context;

  if (selectedTab !== value) {
    return null;
  }

  return <div className={`tabs-content ${className}`}>{children}</div>;
}
