
import React, { useState, useEffect } from "react";
import "./app.css";

// --- Reusable Components ---

const Header = () => (
  <header className="app-header">
    <div className="header-content">
      <div className="header-animation">
        <div className="floating-icon">🌱</div>
        <div className="floating-icon delay-1">⚡</div>
        <div className="floating-icon delay-2">🚗</div>
      </div>
      <h1>GreenAI Fuel Efficiency Predictor</h1>
      <p>Predict fuel efficiency and reduce your carbon footprint with AI</p>
    </div>
  </header>
);

const StatCard = ({ title, value, subtitle, icon, isAnimated, highlight }) => (
  <div
    className={`stat-card ${isAnimated ? "animate-in" : ""} ${
      highlight ? "highlight-card" : ""
    }`}
  >
    <div className="stat-icon">{icon}</div>
    <div className="stat-content">
      <h3>{title}</h3>
      <div className="stat-value">{value}</div>
      <p className="stat-subtitle">{subtitle}</p>
    </div>
  </div>
);

const LoadingSpinner = () => (
  <div className="loading-spinner">
    <div className="spinner"></div>
    <p>Analyzing vehicle data...</p>
  </div>
);

const AnimatedDropdown = ({
  id,
  name,
  value,
  onChange,
  options,
  placeholder,
  disabled,
  error,
  loading,
}) => (
  <div className="form-group">
    <label htmlFor={id}>{placeholder}</label>
    <div className="dropdown-wrapper">
      <select
        id={id}
        name={name}
        value={value}
        onChange={onChange}
        required
        disabled={disabled || loading}
        className={error ? "error" : ""}
      >
        <option value="">
          {loading ? "Loading..." : `Select ${placeholder}`}
        </option>
        {options.map((option, index) => (
          <option key={index} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
      <div className="dropdown-arrow">▼</div>
    </div>
    {error && <span className="error-text">{error}</span>}
    {loading && <div className="loading-indicator">🔄</div>}
  </div>
);

// --- New Sections / Components ---

const HeroSection = () => (
  <section className="hero-section">
    <div className="hero-content">
      <h2 className="hero-title">Drive Smarter, Live Greener</h2>
      {/* <p className="hero-subtitle">
        Harness the power of AI to understand and improve your vehicle's
        environmental impact.
      </p> */}
      <div className="hero-buttons">
        <button
          className="btn btn-primary"
          onClick={() =>
            document
              .getElementById("prediction-section")
              .scrollIntoView({ behavior: "smooth" })
          }
        >
          Get Started <span className="icon">🚀</span>
        </button>
        <button className="btn btn-secondary">
          Learn More <span className="icon">📚</span>
        </button>
      </div>
    </div>
    <div className="hero-image">
      {/* Placeholder for an engaging image or animation */}
      <img
        src="https://via.placeholder.com/600x400/1e293b/34d399?text=AI-Powered+Green+Driving"
        alt="Green AI driving concept"
      />
    </div>
  </section>
);

const AboutSection = () => (
  <section className="about-section">
    <div className="about-content">
      <h2>About GreenAI Predictor</h2>
      <p>
        The GreenAI Fuel Efficiency Predictor utilizes advanced Machine Learning
        and Deep Learning models to accurately estimate your vehicle's fuel
        efficiency (MPG) and its corresponding environmental impact. Our goal is
        to empower vehicle owners with actionable insights to make more
        eco-conscious decisions.
      </p>
      <p>
        By providing data on vehicle make, model, engine size, and year, you can
        instantly see a comprehensive breakdown of estimated MPG, annual CO₂
        emissions, the number of trees needed to offset those emissions, and
        projected annual fuel costs. Drive towards a sustainable future with
        GreenAI!
      </p>
      <button className="btn btn-primary-outline">
        Our Mission <span className="icon">🌍</span>
      </button>
    </div>
    <div className="about-image">
      <img
        src="/pexels-enginakyurt-20500733.jpg"
        alt="Sustainable future with Green AI"
        style={{
          width: "600px",
          height: "auto",
          borderRadius: "10px",
          filter: "grayscale(50%)",
        }}
      />
    </div>
  </section>
);

const FeatureSection = () => (
  <section className="feature-section">
    <h2>Key Features</h2>
    <div className="feature-grid">
      <div className="feature-item">
        <span className="feature-icon">✨</span>
        <h3>Accurate AI Prediction</h3>
        <p>Leveraging cutting-edge ML/DL models for precise MPG estimations.</p>
      </div>
      <div className="feature-item">
        <span className="feature-icon">📊</span>
        <h3>Detailed Impact Reports</h3>
        <p>Understand CO₂, fuel cost, and tree offset for your vehicle.</p>
      </div>
      <div className="feature-item">
        <span className="feature-icon">💡</span>
        <h3>Personalized Recommendations</h3>
        <p>Tips to improve efficiency and reduce your carbon footprint.</p>
      </div>
      <div className="feature-item">
        <span className="feature-icon">🔄</span>
        <h3>Real-time Data</h3>
        <p>
          Access to a constantly updated database of vehicle specifications.
        </p>
      </div>
    </div>
  </section>
);

const CallToAction = () => (
  <section className="cta-section">
    <div className="cta-content">
      <h2>Ready to Discover Your Green Score?</h2>
      <p>
        Join thousands of users making informed decisions for a healthier
        planet.
      </p>
      <button
        className="btn btn-primary cta-button"
        onClick={() =>
          document
            .getElementById("prediction-section")
            .scrollIntoView({ behavior: "smooth" })
        }
      >
        Predict Now <span className="icon">🚀</span>
      </button>
    </div>
  </section>
);

const TrendingVehicles = () => {
  const [trending, setTrending] = useState([]);
  const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:5000/api";

  useEffect(() => {
    const fetchTrending = async () => {
      try {
        const response = await fetch(`${API_BASE}/vehicles/trending`);
        if (response.ok) {
          const data = await response.json();
          setTrending(data);
        }
      } catch (error) {
        console.error("Failed to fetch trending vehicles:", error);
        // Fallback to dummy data if API fails
        setTrending([
          {
            make: "Toyota",
            model: "Prius",
            year: 2022,
            engine_size: 1.8,
            mpg: 56,
          },
          {
            make: "Tesla",
            model: "Model 3",
            year: 2023,
            engine_size: 0,
            mpg: 132,
          },
          {
            make: "Honda",
            model: "CR-V Hybrid",
            year: 2023,
            engine_size: 2.0,
            mpg: 40,
          },
          {
            make: "Hyundai",
            model: "Elantra Hybrid",
            year: 2024,
            engine_size: 1.6,
            mpg: 54,
          },
        ]);
      }
    };
    fetchTrending();
  }, [API_BASE]);

  return (
    <section className="trending-section">
      <h2>🔥 Trending Eco-Vehicles</h2>
      <div className="trending-grid">
        {trending.map((vehicle, index) => (
          <div key={index} className="trending-card">
            <span className="trending-icon">✨</span>
            <div className="trending-info">
              <h3>
                {vehicle.make} {vehicle.model}
              </h3>
              <p>
                {vehicle.year}{" "}
                {parseFloat(vehicle.engine_size) === 0
                  ? "Electric"
                  : `${vehicle.engine_size}L`}
              </p>
            </div>
            <div className="trending-mpg">
              <span className="value">{vehicle.mpg}</span>
              <span className="unit">MPG</span>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
};

// --- Main Prediction Form & Result Components ---

const PredictionForm = ({ onSubmit, loading }) => {
  const [formData, setFormData] = useState({
    make: "",
    model: "",
    engine_size: "",
    year: "",
  });

  const [dropdownOptions, setDropdownOptions] = useState({
    makes: [],
    models: [],
    engineSizes: [],
    years: [],
  });

  const [dropdownLoading, setDropdownLoading] = useState({
    makes: false,
    models: false,
    engineSizes: false,
    years: false,
  });

  const [errors, setErrors] = useState({});
  const [totalVehicles, setTotalVehicles] = useState(0);

  const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:5000/api";

  useEffect(() => {
    loadMakes();
  }, []);

  const loadMakes = async () => {
    setDropdownLoading((prev) => ({ ...prev, makes: true }));
    try {
      const response = await fetch(`${API_BASE}/vehicles/makes`);
      if (response.ok) {
        const makes = await response.json();
        setDropdownOptions((prev) => ({ ...prev, makes }));

        const allVehiclesResponse = await fetch(`${API_BASE}/vehicles/all`);
        if (allVehiclesResponse.ok) {
          const allVehicles = await allVehiclesResponse.json();
          setTotalVehicles(allVehicles.length);
        }
      }
    } catch (error) {
      console.error("Error loading makes:", error);
    } finally {
      setDropdownLoading((prev) => ({ ...prev, makes: false }));
    }
  };

  const loadModels = async (make) => {
    if (!make) return;

    setDropdownLoading((prev) => ({ ...prev, models: true }));
    try {
      const response = await fetch(
        `${API_BASE}/vehicles/models/${encodeURIComponent(make)}`
      );
      if (response.ok) {
        const models = await response.json();
        setDropdownOptions((prev) => ({
          ...prev,
          models,
          engineSizes: [],
          years: [],
        }));
      }
    } catch (error) {
      console.error("Error loading models:", error);
    } finally {
      setDropdownLoading((prev) => ({ ...prev, models: false }));
    }
  };

  const loadEngineSizes = async (make, model) => {
    if (!make || !model) return;

    setDropdownLoading((prev) => ({ ...prev, engineSizes: true }));
    try {
      const response = await fetch(
        `${API_BASE}/vehicles/engine_sizes/${encodeURIComponent(
          make
        )}/${encodeURIComponent(model)}`
      );
      if (response.ok) {
        const engineSizes = await response.json();
        setDropdownOptions((prev) => ({
          ...prev,
          engineSizes,
          years: [],
        }));
      }
    } catch (error) {
      console.error("Error loading engine sizes:", error);
    } finally {
      setDropdownLoading((prev) => ({ ...prev, engineSizes: false }));
    }
  };

  const loadYears = async (make, model, engineSize) => {
    if (!make || !model || !engineSize) return;

    setDropdownLoading((prev) => ({ ...prev, years: true }));
    try {
      const response = await fetch(
        `${API_BASE}/vehicles/years/${encodeURIComponent(
          make
        )}/${encodeURIComponent(model)}/${engineSize}`
      );
      if (response.ok) {
        const years = await response.json();
        setDropdownOptions((prev) => ({ ...prev, years }));
      }
    } catch (error) {
      console.error("Error loading years:", error);
    } finally {
      setDropdownLoading((prev) => ({ ...prev, years: false }));
    }
  };

  useEffect(() => {
    if (formData.make) {
      loadModels(formData.make);
      setFormData((prev) => ({
        ...prev,
        model: "",
        engine_size: "",
        year: "",
      }));
    }
  }, [formData.make]);

  useEffect(() => {
    if (formData.make && formData.model) {
      loadEngineSizes(formData.make, formData.model);
      loadAllYearsForModel(formData.make, formData.model);
      setFormData((prev) => ({
        ...prev,
        engine_size: "",
        year: "",
      }));
    }
  }, [formData.make, formData.model]);

  const loadAllYearsForModel = async (make, model) => {
    if (!make || !model) return;

    setDropdownLoading((prev) => ({ ...prev, years: true }));
    try {
      const response = await fetch(`${API_BASE}/vehicles/all`);
      if (response.ok) {
        const allVehicles = await response.json();
        const yearsForModel = [
          ...new Set(
            allVehicles
              .filter((v) => v.make === make && v.model === model)
              .map((v) => v.year)
          ),
        ].sort((a, b) => b - a);

        const yearOptions = yearsForModel.map((year) => ({
          value: year,
          label: year.toString(),
        }));

        setDropdownOptions((prev) => ({ ...prev, years: yearOptions }));
      }
    } catch (error) {
      console.error("Error loading years for model:", error);
    } finally {
      setDropdownLoading((prev) => ({ ...prev, years: false }));
    }
  };

  useEffect(() => {
    if (formData.make && formData.model && formData.engine_size) {
      loadYears(
        formData.make,
        formData.model,
        parseFloat(formData.engine_size)
      );
      setFormData((prev) => ({
        ...prev,
        year: "",
      }));
    }
  }, [formData.make, formData.model, formData.engine_size]);

  const validateConfiguration = async () => {
    if (!formData.make || !formData.model || !formData.engine_size) {
      return {
        valid: false,
        error: "Please select make, model, and engine size",
      };
    }

    try {
      const response = await fetch(`${API_BASE}/vehicles/validate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          make: formData.make,
          model: formData.model,
          engine_size: parseFloat(formData.engine_size),
          year: formData.year ? parseInt(formData.year) : null,
        }),
      });

      if (response.ok) {
        const result = await response.json();
        return result;
      } else {
        return { valid: false, error: "Validation failed" };
      }
    } catch (error) {
      console.error("Validation error:", error);
      return { valid: false, error: "Network error during validation" };
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));

    if (errors[name]) {
      setErrors((prev) => ({ ...prev, [name]: null, general: null }));
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const validation = await validateConfiguration();

    if (!validation.valid) {
      setErrors({
        general: validation.error || "Invalid vehicle configuration",
      });
      return;
    }

    const vehicleData = validation.vehicle;
    onSubmit(vehicleData);
  };

  const isFormValid = formData.make && formData.model && formData.engine_size;

  return (
    <form className="prediction-form glass-effect" onSubmit={handleSubmit}>
      <div className="form-header">
        <p className="form-description">
          Select your vehicle from our database of{" "}
          {totalVehicles.toLocaleString()} vehicles
        </p>
      </div>

      <div className="form-grid">
        <AnimatedDropdown
          id="make"
          name="make"
          value={formData.make}
          onChange={handleChange}
          options={dropdownOptions.makes}
          placeholder="Vehicle Brand"
          loading={dropdownLoading.makes}
          error={errors.make}
        />

        <AnimatedDropdown
          id="model"
          name="model"
          value={formData.model}
          onChange={handleChange}
          options={dropdownOptions.models}
          placeholder="Vehicle Model"
          disabled={!formData.make}
          loading={dropdownLoading.models}
          error={errors.model}
        />

        <AnimatedDropdown
          id="engine_size"
          name="engine_size"
          value={formData.engine_size}
          onChange={handleChange}
          options={dropdownOptions.engineSizes}
          placeholder="Engine Size"
          disabled={!formData.model}
          loading={dropdownLoading.engineSizes}
          error={errors.engine_size}
        />

        <AnimatedDropdown
          id="year"
          name="year"
          value={formData.year}
          onChange={handleChange}
          options={dropdownOptions.years}
          placeholder="Year (Optional)"
          disabled={!formData.make || !formData.model}
          loading={dropdownLoading.years}
          error={errors.year}
        />
      </div>

      {process.env.NODE_ENV === "development" && (
        <div className="debug-info">
          <strong>Debug Info:</strong>
          <br />
          Makes: {dropdownOptions.makes.length}
          <br />
          Models: {dropdownOptions.models.length}
          <br />
          Engine Sizes: {dropdownOptions.engineSizes.length}
          <br />
          Years: {dropdownOptions.years.length}
          <br />
          Selected: {formData.make} | {formData.model} | {formData.engine_size}{" "}
          | {formData.year}
        </div>
      )}

      {errors.general && (
        <div className="error-message animate-in">❌ {errors.general}</div>
      )}

      <div className="form-status">
        {formData.make && formData.model && formData.engine_size && (
          <div className="selected-vehicle animate-in">
            <span className="vehicle-icon">🚗</span>
            <span className="vehicle-info">
              {formData.make} {formData.model}
              {formData.engine_size &&
                ` (${
                  parseFloat(formData.engine_size) === 0
                    ? "Electric"
                    : formData.engine_size + "L"
                })`}
              {formData.year && ` - ${formData.year}`}
            </span>
          </div>
        )}
      </div>

      <button
        type="submit"
        className={`predict-button ${loading ? "loading" : ""}`}
        disabled={loading || !isFormValid}
      >
        {loading ? <LoadingSpinner /> : "🚗 Predict Fuel Efficiency"}
      </button>
    </form>
  );
};

const PredictionResult = ({ result }) => {
  if (!result) return null;

  const getGradeFromScore = (score) => {
    if (score >= 90) return { grade: "A+", color: "#10b981" }; // green
    if (score >= 80) return { grade: "A", color: "#22c55e" }; // lighter green
    if (score >= 70) return { grade: "B+", color: "#84cc16" }; // lime green
    if (score >= 60) return { grade: "B", color: "#eab308" }; // yellow
    if (score >= 50) return { grade: "C+", color: "#f97316" }; // orange
    if (score >= 40) return { grade: "C", color: "#f59e0b" }; // dark orange
    return { grade: "D", color: "#ef4444" }; // red
  };

  const gradeInfo = getGradeFromScore(result.green_score);

  return (
    <div className="prediction-result animate-in glass-effect">
      <div className="result-header">
        <h2>🎯 Prediction Results</h2>
        <div
          className="green-score animate-pulse"
          style={{ backgroundColor: gradeInfo.color }}
        >
          <span className="score-grade">{gradeInfo.grade}</span>
          <span className="score-value">
            {Math.round(result.green_score)}/100
          </span>
        </div>
      </div>

      <div className="result-grid">
        <StatCard
          title="Predicted MPG"
          value={`${result.predicted_mpg} MPG`}
          subtitle={
            result.comparison.better_than_average
              ? `✅ Better than avg (${result.comparison.average_mpg} MPG)`
              : `⚠️ Below average (${result.comparison.average_mpg} MPG)`
          }
          icon="⛽"
          isAnimated={true}
          highlight={result.comparison.better_than_average}
        />
        <StatCard
          title="Annual CO₂ Impact"
          value={`${result.environmental_impact.annual_co2_tons} tons`}
          subtitle={`${result.environmental_impact.annual_co2_pounds} lbs/year`}
          icon="🌍"
          isAnimated={true}
        />
        <StatCard
          title="Trees to Offset"
          value={`${result.environmental_impact.trees_to_offset} trees`}
          subtitle="To neutralize annual emissions"
          icon="🌳"
          isAnimated={true}
        />

        <StatCard
          title="Annual Fuel Cost"
          value={`$${Math.round(
            result.environmental_impact.annual_gallons * 3.5
          ).toLocaleString()}`}
          subtitle={`${Math.round(
            result.environmental_impact.annual_gallons
          ).toLocaleString()} gallons/year`}
          icon="💰"
          isAnimated={true}
        />
      </div>

      <div className="model-comparison animate-in glass-effect">
        <h3>🤖 AI Model Predictions</h3>
        <div className="model-grid">
          <div className="model-result">
            <span className="model-name">Random Forest</span>
            <span className="model-value">{result.ml_prediction} MPG</span>
          </div>
          <div className="model-result">
            <span className="model-name">Deep Learning</span>
            <span className="model-value">{result.dl_prediction} MPG</span>
          </div>
        </div>
      </div>

      <div className="recommendations animate-in glass-effect">
        <h3>🌱 Green Recommendations</h3>
        <ul className="recommendation-list">
          {result.recommendations.map((rec, index) => (
            <li
              key={index}
              className="recommendation-item animate-in"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <span className="rec-icon">💡</span>
              {rec}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

const Statistics = ({ stats }) => {
  if (!stats) return null;

  return (
    <div className="statistics-section animate-in glass-effect">
      <h2>📊 Global Impact Statistics</h2>
      <div className="stats-grid">
        <StatCard
          title="Total Predictions"
          value={stats.total_predictions.toLocaleString()}
          subtitle="Vehicles analyzed"
          icon="🚗"
          isAnimated={true}
        />
        <StatCard
          title="Average Efficiency"
          value={`${stats.average_predicted_mpg} MPG`}
          subtitle="Across all predictions"
          icon="📈"
          isAnimated={true}
        />
        <StatCard
          title="CO₂ Awareness"
          value={`${stats.total_co2_impact_tons} tons`}
          subtitle="Environmental impact calculated"
          icon="🌍"
          isAnimated={true}
        />
        <StatCard
          title="Tree Equivalent"
          value={`${stats.trees_equivalent.toLocaleString()} trees`}
          subtitle="Needed for offset"
          icon="🌳"
          isAnimated={true}
        />
      </div>
    </div>
  );
};

function App() {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [statistics, setStatistics] = useState(null);
  const [error, setError] = useState(null);

  const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:5000/api";

  useEffect(() => {
    fetchStatistics();
  }, []);

  const fetchStatistics = async () => {
    try {
      const response = await fetch(`${API_BASE}/stats`);
      if (response.ok) {
        const data = await response.json();
        setStatistics(data);
      }
    } catch (err) {
      console.error("Failed to fetch statistics:", err);
      // Fallback dummy data for stats if API fails
      setStatistics({
        total_predictions: 123456,
        average_predicted_mpg: 32.1,
        total_co2_impact_tons: 7890,
        trees_equivalent: 345678,
      });
    }
  };

  const handlePrediction = async (vehicleData) => {
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(vehicleData),
      });

      if (!response.ok) {
        const errorData = await response
          .json()
          .catch(() => ({ error: "Unknown error" }));
        throw new Error(
          errorData.error || `HTTP error! status: ${response.status}`
        );
      }

      const result = await response.json();

      setPrediction({
        ...result,
        comparison: {
          average_mpg: 25.4, // Assuming a global average for comparison
          better_than_average: result.predicted_mpg > 25.4,
        },
        recommendations: [
          "Check tire pressure regularly to improve fuel economy by up to 3%.",
          "Remove unnecessary weight from your vehicle.",
          "Avoid aggressive driving; smooth acceleration and braking saves fuel.",
          "Use cruise control on highways to maintain a consistent speed.",
          "Consider carpooling or using public transport for short trips.",
        ],
      });
      fetchStatistics();
    } catch (err) {
      const errorMessage = `Failed to get prediction: ${err.message}`;
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App">
      <Header />

      <main className="main-content">
        <div className="container">
          <HeroSection />
          <AboutSection />
          <FeatureSection />

          {error && <div className="error-message animate-in">❌ {error}</div>}

          <div
            id="prediction-section"
            className="prediction-section animate-in"
          >
            <h2>🔍 Vehicle Analysis</h2>
            <PredictionForm onSubmit={handlePrediction} loading={loading} />
          </div>

          <PredictionResult result={prediction} />

          <TrendingVehicles />
          <Statistics stats={statistics} />
          <CallToAction />
        </div>
      </main>

      <footer className="app-footer glass-effect">
        <p>
          🌱 Built with GreenAI principles • Powered by Machine Learning & Deep
          Learning
        </p>
        <p>Helping reduce carbon emissions one prediction at a time</p>
        <div className="footer-links">
          <a href="#privacy">Privacy Policy</a>
          <a href="#terms">Terms of Service</a>
          <a href="#contact">Contact Us</a>
        </div>
      </footer>
    </div>
  );
}

export default App;
