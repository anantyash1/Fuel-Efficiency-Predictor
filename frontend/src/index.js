import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css'; // Global styles for body, reset, etc.
import App from './app'; // Your main application component
import { reportWebVitals } from './reportWebVitals';

// Global error boundary for robust application
class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI.
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    // You can also log the error to an error reporting service
    console.error('Error caught by boundary:', error, errorInfo);
    
    // In a real-world app, you'd send this to Sentry, Crashlytics, etc.
    if (process.env.NODE_ENV === 'production') {
      // window.Sentry?.captureException(error); // Example Sentry integration
    }
  }

  render() {
    if (this.state.hasError) {
      // Fallback UI when an error occurs
      return (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          height: '100vh',
          // Matching the app's dark theme
          background: 'linear-gradient(135deg, #0a0f1c 0%, #111827 100%)',
          color: '#f9fafb', // Primary text color
          fontFamily: 'Inter, sans-serif',
          textAlign: 'center',
          padding: '20px',
          boxShadow: 'inset 0 0 50px rgba(0,0,0,0.5)' // Added a subtle inner shadow
        }}>
          <h1 style={{ color: '#ef4444', marginBottom: '15px' }}>🚫 Something went wrong!</h1>
          <p style={{ fontSize: '1.1rem', marginBottom: '25px', maxWidth: '600px', lineHeight: '1.5' }}>
            We're sorry, but the GreenAI Fuel Efficiency Predictor encountered an unexpected error.
          </p>
          <details style={{ 
            marginTop: '20px', 
            padding: '15px 20px', 
            background: 'rgba(31, 41, 55, 0.7)', // bg-tertiary with transparency
            borderRadius: '12px', 
            maxWidth: '90%', 
            maxHeight: '300px', 
            overflow: 'auto',
            border: '1px solid rgba(75, 85, 99, 0.3)', // border-color
            boxShadow: '0 5px 15px rgba(0,0,0,0.3)'
          }}>
            <summary style={{ cursor: 'pointer', marginBottom: '10px', fontWeight: 'bold', color: '#6ee7b7' }}>Error Details (Click to expand)</summary>
            <pre style={{ 
              textAlign: 'left', 
              fontSize: '12px', 
              overflow: 'auto', 
              color: '#d1d5db', // text-secondary
              whiteSpace: 'pre-wrap', // Ensures long lines wrap
              wordBreak: 'break-word'
            }}>
              {this.state.error && this.state.error.stack ? this.state.error.stack : this.state.error.toString()}
            </pre>
          </details>
          <button 
            onClick={() => window.location.reload()} 
            style={{
              marginTop: '30px',
              padding: '12px 25px',
              background: 'linear-gradient(90deg, #10b981, #34d399)', // green-primary to green-secondary
              color: 'white',
              border: 'none',
              borderRadius: '10px',
              cursor: 'pointer',
              fontSize: '17px',
              fontWeight: '600',
              boxShadow: '0 5px 20px rgba(16, 185, 129, 0.4)',
              transition: 'all 0.3s ease'
            }}
            onMouseEnter={(e) => e.currentTarget.style.transform = 'translateY(-2px)'}
            onMouseLeave={(e) => e.currentTarget.style.transform = 'translateY(0)'}
          >
            🔄 Reload Application
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

// Create root and render app
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>
);

// Performance measurement
reportWebVitals(console.log);

// Hot module replacement for development
if (module.hot) {
  module.hot.accept();
}