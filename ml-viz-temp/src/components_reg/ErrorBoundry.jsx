import React, { Component } from "react";
import { toast } from "react-hot-toast";

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render shows the fallback UI
    return { hasError: true };
  }

  componentDidCatch(error, errorInfo) {
    // Log the error to an error reporting service (optional)
    console.error("Error caught by error boundary:", error, errorInfo);

    // Show a toast notification
    toast.error("An error occurred. Please try again.");
  }

  render() {
    if (this.state.hasError) {
      // You can render a fallback UI here if needed
      return <h1>Something went wrong.</h1>;
    }

    return this.props.children;
  }
}

export default ErrorBoundary;