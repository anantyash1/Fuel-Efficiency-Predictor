-- Database initialization script for Fuel Efficiency Predictor

-- Create extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Vehicles table for reference data
CREATE TABLE vehicles (
    id SERIAL PRIMARY KEY,
    make VARCHAR(100) NOT NULL,
    model VARCHAR(100) NOT NULL,
    year INTEGER NOT NULL CHECK (year >= 1990 AND year <= 2030),
    engine_size DECIMAL(3,1) NOT NULL CHECK (engine_size > 0),
    cylinders INTEGER NOT NULL CHECK (cylinders IN (3,4,5,6,8,10,12)),
    transmission VARCHAR(50) NOT NULL,
    fuel_type VARCHAR(50) NOT NULL,
    drivetrain VARCHAR(10),
    weight INTEGER,
    city_mpg DECIMAL(4,1),
    highway_mpg DECIMAL(4,1),
    combined_mpg DECIMAL(4,1) NOT NULL,
    co2_emissions DECIMAL(6,1),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(make, model, year, engine_size)
);

-- Prediction history table
CREATE TABLE prediction_history (
    id SERIAL PRIMARY KEY,
    session_id UUID DEFAULT uuid_generate_v4(),
    user_input JSONB NOT NULL,
    predicted_mpg DECIMAL(4,1) NOT NULL,
    environmental_impact DECIMAL(8,2) NOT NULL,
    recommendations JSONB NOT NULL,
    model_predictions JSONB, -- Store individual model predictions
    green_score INTEGER CHECK (green_score >= 0 AND green_score <= 100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ip_address INET,
    user_agent TEXT
);

-- User feedback table
CREATE TABLE user_feedback (
    id SERIAL PRIMARY KEY,
    prediction_id INTEGER REFERENCES prediction_history(id),
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    feedback_text TEXT,
    actual_mpg DECIMAL(4,1), -- If user provides actual MPG
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Environmental impact tracking
CREATE TABLE environmental_stats (
    id SERIAL PRIMARY KEY,
    date DATE DEFAULT CURRENT_DATE,
    total_predictions INTEGER DEFAULT 0,
    avg_predicted_mpg DECIMAL(4,1),
    total_co2_calculated DECIMAL(10,2),
    trees_equivalent INTEGER,
    fuel_savings_gallons DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date)
);

-- Model performance tracking
CREATE TABLE model_performance (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(50) NOT NULL,
    version VARCHAR(20) NOT NULL,
    rmse DECIMAL(6,3),
    mae DECIMAL(6,3),
    r2_score DECIMAL(4,3),
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT FALSE,
    UNIQUE(model_name, version)
);

-- Insert sample vehicle data
INSERT INTO vehicles (make, model, year, engine_size, cylinders, transmission, fuel_type, drivetrain, weight, city_mpg, highway_mpg, combined_mpg, co2_emissions) VALUES
-- Hybrid/Electric vehicles
('Toyota', 'Prius', 2023, 1.8, 4, 'CVT', 'Hybrid', 'FWD', 3142, 58, 53, 56, 96),
('Toyota', 'Prius Prime', 2023, 1.8, 4, 'CVT', 'Plug-in Hybrid', 'FWD', 3365, 127, 108, 114, 28),
('Tesla', 'Model 3', 2023, 0.0, 0, 'Single-Speed', 'Electric', 'RWD', 4034, 132, 126, 129, 0),
('Tesla', 'Model S', 2023, 0.0, 0, 'Single-Speed', 'Electric', 'AWD', 4561, 120, 115, 118, 0),
('Honda', 'Insight', 2022, 1.5, 4, 'CVT', 'Hybrid', 'FWD', 2987, 55, 49, 52, 110),

-- Efficient gasoline vehicles
('Honda', 'Civic', 2023, 2.0, 4, 'CVT', 'Gasoline', 'FWD', 2906, 32, 42, 36, 262),
('Toyota', 'Corolla', 2023, 2.0, 4, 'CVT', 'Gasoline', 'FWD', 2955, 32, 41, 36, 258),
('Nissan', 'Versa', 2023, 1.6, 4, 'CVT', 'Gasoline', 'FWD', 2599, 32, 40, 35, 267),
('Mitsubishi', 'Mirage', 2023, 1.2, 3, 'CVT', 'Gasoline', 'FWD', 2018, 36, 43, 39, 238),

-- Mid-range vehicles
('Honda', 'Accord', 2023, 1.5, 4, 'CVT', 'Gasoline', 'FWD', 3131, 32, 42, 36, 267),
('Toyota', 'Camry', 2023, 2.5, 4, '8-Speed Automatic', 'Gasoline', 'FWD', 3310, 28, 39, 32, 296),
('Mazda', 'Mazda3', 2023, 2.5, 4, '6-Speed Automatic', 'Gasoline', 'FWD', 3064, 26, 35, 30, 317),

-- SUVs
('Toyota', 'RAV4', 2023, 2.5, 4, '8-Speed Automatic', 'Gasoline', 'AWD', 3605, 27, 35, 30, 318),
('Honda', 'CR-V', 2023, 1.5, 4, 'CVT', 'Gasoline', 'AWD', 3433, 27, 32, 29, 328),
('Subaru', 'Forester', 2023, 2.5, 4, 'CVT', 'Gasoline', 'AWD', 3449, 26, 33, 29, 330),

-- Trucks
('Ford', 'F-150', 2023, 3.3, 6, '10-Speed Automatic', 'Gasoline', 'RWD', 4021, 19, 24, 21, 555),
('Chevrolet', 'Silverado 1500', 2023, 2.7, 4, '8-Speed Automatic', 'Gasoline', 'RWD', 4520, 20, 23, 21, 567),
('Ram', '1500', 2023, 3.6, 6, '8-Speed Automatic', 'Gasoline', 'RWD', 4798, 20, 25, 22, 533);

-- Create indexes for better performance
CREATE INDEX idx_vehicles_make_model ON vehicles(make, model);
CREATE INDEX idx_vehicles_year ON vehicles(year);
CREATE INDEX idx_vehicles_mpg ON vehicles(combined_mpg);
CREATE INDEX idx_prediction_history_created_at ON prediction_history(created_at);
CREATE INDEX idx_prediction_history_mpg ON prediction_history(predicted_mpg);
CREATE INDEX idx_environmental_stats_date ON environmental_stats(date);

-- Create functions for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_modified_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for automatic timestamp updates
CREATE TRIGGER update_vehicles_modtime 
    BEFORE UPDATE ON vehicles 
    FOR EACH ROW 
    EXECUTE FUNCTION update_modified_column();

-- Function to update daily environmental stats
CREATE OR REPLACE FUNCTION update_environmental_stats()
RETURNS VOID AS $$
DECLARE
    today DATE := CURRENT_DATE;
    pred_count INTEGER;
    avg_mpg DECIMAL(4,1);
    total_co2 DECIMAL(10,2);
    trees_needed INTEGER;
    fuel_saved DECIMAL(10,2);
BEGIN
    -- Calculate today's statistics
    SELECT 
        COUNT(*),
        AVG(predicted_mpg),
        SUM(environmental_impact),
        SUM(environmental_impact * 16)
    INTO pred_count, avg_mpg, total_co2, trees_needed
    FROM prediction_history 
    WHERE DATE(created_at) = today;
    
    -- Calculate fuel savings compared to average vehicle (25.4 MPG)
    SELECT 
        SUM((predicted_mpg - 25.4) * 12000 / 25.4 / predicted_mpg)
    INTO fuel_saved
    FROM prediction_history 
    WHERE DATE(created_at) = today AND predicted_mpg > 25.4;
    
    -- Insert or update today's stats
    INSERT INTO environmental_stats (date, total_predictions, avg_predicted_mpg, total_co2_calculated, trees_equivalent, fuel_savings_gallons)
    VALUES (today, pred_count, avg_mpg, total_co2, trees_needed, COALESCE(fuel_saved, 0))
    ON CONFLICT (date) 
    DO UPDATE SET
        total_predictions = pred_count,
        avg_predicted_mpg = avg_mpg,
        total_co2_calculated = total_co2,
        trees_equivalent = trees_needed,
        fuel_savings_gallons = COALESCE(fuel_saved, 0);
END;
$$ LANGUAGE plpgsql;

-- Create a view for dashboard statistics
CREATE VIEW dashboard_stats AS
SELECT 
    COUNT(*) as total_predictions,
    AVG(predicted_mpg) as avg_predicted_mpg,
    SUM(environmental_impact) as total_co2_impact,
    SUM(environmental_impact * 16) as trees_equivalent,
    COUNT(DISTINCT DATE(created_at)) as active_days,
    MAX(created_at) as last_prediction
FROM prediction_history;

-- Create a view for recent high-efficiency predictions
CREATE VIEW efficient_predictions AS
SELECT 
    user_input->>'make' as make,
    user_input->>'model' as model,
    predicted_mpg,
    green_score,
    created_at
FROM prediction_history 
WHERE predicted_mpg > 35 
ORDER BY predicted_mpg DESC, created_at DESC 
LIMIT 50;

-- Insert initial environmental stats
INSERT INTO environmental_stats (date, total_predictions, avg_predicted_mpg, total_co2_calculated, trees_equivalent)
VALUES (CURRENT_DATE, 0, 0, 0, 0);

COMMIT;