# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from flask_sqlalchemy import SQLAlchemy
# import pandas as pd
# import numpy as np
# import joblib
# import requests
# import os
# from datetime import datetime
# import tensorflow as tf
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestRegressor
# import json
# import fcntl

# app = Flask(__name__)

# # Production configuration
# if os.environ.get('FLASK_ENV') == 'production':
#     app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', '').replace('postgres://', 'postgresql://')
#     app.config['DEBUG'] = False
# else:
#     app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fuel_efficiency.db'
#     app.config['DEBUG'] = True

# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# # CORS configuration for production
# CORS(app, origins=[
#     "http://localhost:3000",
#     "https://fuel-predictor-frontend.onrender.com",
#     "https://*.onrender.com"
# ])

# db = SQLAlchemy(app)





# CORS(app)

# # Database configuration
# app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fuel_efficiency.db'
# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# db = SQLAlchemy(app)

# # Vehicle Model
# class Vehicle(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     make = db.Column(db.String(100), nullable=False)
#     model = db.Column(db.String(100), nullable=False)
#     year = db.Column(db.Integer, nullable=False)
#     engine_size = db.Column(db.Float, nullable=False)
#     cylinders = db.Column(db.Integer, nullable=False)
#     transmission = db.Column(db.String(50), nullable=False)
#     fuel_type = db.Column(db.String(50), nullable=False)
#     city_mpg = db.Column(db.Float, nullable=False)
#     highway_mpg = db.Column(db.Float, nullable=False)
#     combined_mpg = db.Column(db.Float, nullable=False)
#     co2_emissions = db.Column(db.Float, nullable=False)
#     created_at = db.Column(db.DateTime, default=datetime.now)

# # Prediction History Model
# class PredictionHistory(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     user_input = db.Column(db.Text, nullable=False)
#     predicted_mpg = db.Column(db.Float, nullable=False)
#     environmental_impact = db.Column(db.Float, nullable=False)
#     recommendations = db.Column(db.Text, nullable=False)
#     created_at = db.Column(db.DateTime, default=datetime.now)

# # Initialize ML models
# ml_model = None
# scaler = None
# deep_learning_model = None

# def load_ml_models():
#     global ml_model, scaler, deep_learning_model
#     try:
#         # Check if model files exist
#         model_files = [
#             'ml_models/fuel_efficiency_model.pkl',
#             'ml_models/scaler.pkl',
#             'ml_models/deep_learning_model.h5'
#         ]
        
#         missing_files = [f for f in model_files if not os.path.exists(f)]
        
#         if missing_files:
#             print(f"⚠️ Missing model files: {missing_files}")
#             print("🔄 Training new models...")
#             train_models()
        
#         # Load models
#         ml_model = joblib.load('ml_models/fuel_efficiency_model.pkl')
#         scaler = joblib.load('ml_models/scaler.pkl')
#         deep_learning_model = tf.keras.models.load_model('ml_models/deep_learning_model.h5')
#         print("✅ All models loaded successfully!")
        
#     except Exception as e:
#         print(f"❌ Error loading models: {e}")
#         print("🔄 Training new models...")
#         train_models()

# def train_models():
#     global ml_model, scaler, deep_learning_model
    
#     # Generate sample training data (replace with real data)
#     data = generate_sample_data()
    
#     # Features and target
#     features = ['engine_size', 'cylinders', 'year', 'transmission_encoded', 'fuel_type_encoded']
#     X = data[features]
#     y = data['combined_mpg']
    
#     # Scale features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
    
#     # Train Random Forest
#     ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
#     ml_model.fit(X_scaled, y)
    
#     # Train Deep Learning Model
#     deep_learning_model = tf.keras.Sequential([
#         tf.keras.layers.Dense(128, activation='relu', input_shape=(len(features),)),
#         tf.keras.layers.Dropout(0.3),
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.Dropout(0.3),
#         tf.keras.layers.Dense(32, activation='relu'),
#         tf.keras.layers.Dense(1)
#     ])
    
#     deep_learning_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
#     deep_learning_model.fit(X_scaled, y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    
#     # Save models
#     os.makedirs('ml_models', exist_ok=True)
#     joblib.dump(ml_model, 'ml_models/fuel_efficiency_model.pkl')
#     joblib.dump(scaler, 'ml_models/scaler.pkl')
#     deep_learning_model.save('ml_models/deep_learning_model.h5')

# def generate_sample_data():
#     np.random.seed(42)
#     n_samples = 1000
    
#     data = {
#         'engine_size': np.random.normal(2.5, 0.8, n_samples),
#         'cylinders': np.random.choice([4, 6, 8], n_samples),
#         'year': np.random.randint(2015, 2025, n_samples),
#         'transmission_encoded': np.random.choice([0, 1], n_samples),  # 0: Manual, 1: Automatic
#         'fuel_type_encoded': np.random.choice([0, 1, 2], n_samples),  # 0: Gasoline, 1: Diesel, 2: Hybrid
#     }
    
#     # Generate realistic MPG based on features
#     mpg = (30 - data['engine_size'] * 3 + 
#            (data['cylinders'] == 4) * 5 - 
#            (data['cylinders'] == 8) * 8 +
#            (data['year'] - 2015) * 0.5 +
#            data['transmission_encoded'] * -2 +
#            data['fuel_type_encoded'] * 3 +
#            np.random.normal(0, 3, n_samples))
    
#     data['combined_mpg'] = np.clip(mpg, 15, 50)
    
#     return pd.DataFrame(data)

# def calculate_environmental_impact(mpg, annual_miles=12000):
#     """Calculate CO2 emissions and environmental impact"""
#     if mpg <= 0:
#         mpg = 1  # Avoid division by zero
        
#     gallons_per_year = annual_miles / mpg
#     co2_per_gallon = 19.6  # pounds of CO2 per gallon of gasoline
#     annual_co2 = gallons_per_year * co2_per_gallon
    
#     # Convert to tons
#     co2_tons = annual_co2 / 2000
    
#     return {
#         'annual_gallons': round(float(gallons_per_year), 2),
#         'annual_co2_pounds': round(float(annual_co2), 2),
#         'annual_co2_tons': round(float(co2_tons), 2),
#         'trees_to_offset': int(round(co2_tons * 16, 0))  # ~16 trees per ton of CO2
#     }

# def get_green_recommendations(mpg, vehicle_data):
#     """Generate eco-friendly recommendations"""
#     recommendations = []
    
#     if mpg < 25:
#         recommendations.extend([
#             "Consider hybrid or electric alternatives",
#             "Regular maintenance can improve fuel efficiency by 4-10%",
#             "Proper tire inflation can improve MPG by up to 3%"
#         ])
    
#     if mpg < 30:
#         recommendations.extend([
#             "Avoid aggressive driving - can improve MPG by 15-30%",
#             "Remove excess weight from your vehicle",
#             "Use cruise control on highways"
#         ])
    
#     recommendations.extend([
#         "Consider carpooling or public transportation",
#         "Plan and combine trips to reduce overall driving",
#         "Keep your vehicle well-maintained"
#     ])
    
#     # Ensure all recommendations are strings
#     return [str(rec) for rec in recommendations]

# @app.route('/api/predict', methods=['POST'])
# def predict_fuel_efficiency():
#     try:
#         # Check if models are loaded
#         if ml_model is None or scaler is None or deep_learning_model is None:
#             print("❌ Models not loaded, attempting to load...")
#             load_ml_models()
            
#         if ml_model is None:
#             return jsonify({'error': 'ML models not available. Please try again later.'}), 503
            
#         data = request.json
#         print(f"📥 Received prediction request: {data}")
        
#         # Extract features with validation
#         try:
#             features = {
#                 'engine_size': float(data.get('engine_size', 2.5)),
#                 'cylinders': int(data.get('cylinders', 4)),
#                 'year': int(data.get('year', 2020)),
#                 'transmission_encoded': 1 if data.get('transmission', 'automatic').lower() == 'automatic' else 0,
#                 'fuel_type_encoded': {'gasoline': 0, 'diesel': 1, 'hybrid': 2, 'electric': 3}.get(
#                     data.get('fuel_type', 'gasoline').lower(), 0)
#             }
#             print(f"🔧 Processed features: {features}")
#         except (ValueError, TypeError) as e:
#             return jsonify({'error': f'Invalid input data: {str(e)}'}), 400
        
#         # Prepare features for prediction
#         feature_array = np.array([[features['engine_size'], features['cylinders'], 
#                                  features['year'], features['transmission_encoded'], 
#                                  features['fuel_type_encoded']]])
        
#         print(f"📊 Feature array shape: {feature_array.shape}")
#         print(f"📊 Feature array: {feature_array}")
        
#         # Scale features
#         try:
#             feature_scaled = scaler.transform(feature_array)
#             print(f"📈 Scaled features: {feature_scaled}")
#         except Exception as e:
#             print(f"❌ Scaling error: {e}")
#             return jsonify({'error': 'Feature scaling failed'}), 500
        
#         # Make predictions
#         try:
#             ml_prediction = ml_model.predict(feature_scaled)[0]
#             print(f"🤖 ML prediction: {ml_prediction}")
#         except Exception as e:
#             print(f"❌ ML prediction error: {e}")
#             ml_prediction = 25.0  # Fallback
            
#         try:
#             dl_prediction = deep_learning_model.predict(feature_scaled, verbose=0)[0][0]
#             print(f"🧠 DL prediction: {dl_prediction}")
#         except Exception as e:
#             print(f"❌ DL prediction error: {e}")
#             dl_prediction = 25.0  # Fallback
        
#         # Ensemble prediction (average of both models)
#         final_prediction = (ml_prediction + dl_prediction) / 2
#         final_prediction = max(15, min(50, final_prediction))  # Clamp between realistic values
        
#         print(f"🎯 Final prediction: {final_prediction}")
        
#         # Calculate environmental impact
#         env_impact = calculate_environmental_impact(final_prediction)
        
#         # Get recommendations
#         recommendations = get_green_recommendations(final_prediction, data)
        
#         # Save to database
#         try:
#             prediction_record = PredictionHistory(
#                 user_input=json.dumps(data),
#                 predicted_mpg=final_prediction,
#                 environmental_impact=env_impact['annual_co2_tons'],
#                 recommendations=json.dumps(recommendations)
#             )
#             db.session.add(prediction_record)
#             db.session.commit()
#             print("💾 Prediction saved to database")
#         except Exception as e:
#             print(f"⚠️ Database save error: {e}")
#             # Continue without saving to database
        
#         response = {
#             'predicted_mpg': round(final_prediction, 2),
#             'ml_prediction': round(float(ml_prediction), 2),
#             'dl_prediction': round(float(dl_prediction), 2),
#             'environmental_impact': env_impact,
#             'recommendations': recommendations,
#             'green_score': round(min(100, max(0, (final_prediction - 15) * 100 / 35)), 2),  # Score out of 100
#             'comparison': {
#                 'average_vehicle_mpg': 25.4,
#                 'better_than_average': bool(final_prediction > 25.4)  # Explicitly convert to bool
#             }
#         }
        
#         # Debug: Test JSON serialization before returning
#         try:
#             import json
#             json.dumps(response)
#             print("✅ JSON serialization test passed")
#         except Exception as json_err:
#             print(f"❌ JSON serialization test failed: {json_err}")
#             print(f"🔍 Response data types: {[(k, type(v)) for k, v in response.items()]}")
#             # Return a simpler response if serialization fails
#             response = {
#                 'predicted_mpg': float(final_prediction),
#                 'error': 'Simplified response due to serialization issue'
#             }
        
#         print(f"✅ Prediction successful: {response['predicted_mpg']} MPG")
#         return jsonify(response)
        
#     except Exception as e:
#         error_msg = f'Prediction failed: {str(e)}'
#         print(f"❌ {error_msg}")
#         import traceback
#         traceback.print_exc()  # This will show the full error traceback
#         return jsonify({'error': error_msg}), 500

# @app.route('/api/vehicles/search', methods=['GET'])
# def search_vehicles():
#     """Search vehicles by make and model"""
#     make = request.args.get('make', '')
#     model = request.args.get('model', '')
    
#     query = Vehicle.query
#     if make:
#         query = query.filter(Vehicle.make.ilike(f'%{make}%'))
#     if model:
#         query = query.filter(Vehicle.model.ilike(f'%{model}%'))
    
#     vehicles = query.limit(20).all()
    
#     result = []
#     for vehicle in vehicles:
#         result.append({
#             'id': vehicle.id,
#             'make': vehicle.make,
#             'model': vehicle.model,
#             'year': vehicle.year,
#             'engine_size': vehicle.engine_size,
#             'cylinders': vehicle.cylinders,
#             'transmission': vehicle.transmission,
#             'fuel_type': vehicle.fuel_type,
#             'combined_mpg': vehicle.combined_mpg,
#             'co2_emissions': vehicle.co2_emissions
#         })
    
#     return jsonify(result)

# @app.route('/api/history', methods=['GET'])
# def get_prediction_history():
#     """Get recent prediction history"""
#     history = PredictionHistory.query.order_by(PredictionHistory.created_at.desc()).limit(10).all()
    
#     result = []
#     for record in history:
#         result.append({
#             'id': record.id,
#             'predicted_mpg': record.predicted_mpg,
#             'environmental_impact': record.environmental_impact,
#             'created_at': record.created_at.isoformat()
#         })
    
#     return jsonify(result)

# @app.route('/api/stats', methods=['GET'])
# def get_statistics():
#     """Get application statistics"""
#     total_predictions = PredictionHistory.query.count()
#     avg_mpg = db.session.query(db.func.avg(PredictionHistory.predicted_mpg)).scalar() or 0
#     total_co2_saved = db.session.query(db.func.sum(PredictionHistory.environmental_impact)).scalar() or 0
    
#     return jsonify({
#         'total_predictions': total_predictions,
#         'average_predicted_mpg': round(float(avg_mpg), 2),
#         'total_co2_impact_tons': round(float(total_co2_saved), 2),
#         'trees_equivalent': round(float(total_co2_saved) * 16, 0)
#     })

# @app.route('/', methods=['GET'])
# def index():
#     """Root endpoint"""
#     return jsonify({
#         'message': 'GreenAI Fuel Efficiency Predictor API',
#         'status': 'running',
#         'version': '1.0.0',
#         'endpoints': {
#             'predict': '/api/predict',
#             'vehicles': '/api/vehicles/search',
#             'stats': '/api/stats',
#             'history': '/api/history',
#             'health': '/api/health'
#         }
#     })

# @app.route('/api/health', methods=['GET'])
# def health_check():
#     return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})


# # Add these new endpoints to your existing app.py

# @app.route('/api/vehicles/makes', methods=['GET'])
# def get_vehicle_makes():
#     """Get unique vehicle makes (brands) for dropdown"""
#     try:
#         makes = db.session.query(Vehicle.make).distinct().order_by(Vehicle.make).all()
#         result = [{'value': make[0], 'label': make[0]} for make in makes]
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/vehicles/models/<make>', methods=['GET'])
# def get_models_for_make(make):
#     """Get models for a specific make"""
#     try:
#         models = db.session.query(Vehicle.model).filter(
#             Vehicle.make == make
#         ).distinct().order_by(Vehicle.model).all()
        
#         result = [{'value': model[0], 'label': model[0]} for model in models]
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/vehicles/engine_sizes/<make>/<model>', methods=['GET'])
# def get_engine_sizes_for_model(make, model):
#     """Get engine sizes for a specific make and model"""
#     try:
#         engine_sizes = db.session.query(Vehicle.engine_size).filter(
#             Vehicle.make == make,
#             Vehicle.model == model
#         ).distinct().order_by(Vehicle.engine_size).all()
        
#         result = []
#         for size in engine_sizes:
#             size_value = size[0]
#             if size_value == 0:
#                 label = "Electric (0L)"
#             else:
#                 label = f"{size_value}L"
#             result.append({'value': size_value, 'label': label})
        
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/vehicles/years/<make>/<model>/<float:engine_size>', methods=['GET'])
# def get_years_for_vehicle(make, model, engine_size):
#     """Get available years for a specific vehicle configuration"""
#     try:
#         years = db.session.query(Vehicle.year).filter(
#             Vehicle.make == make,
#             Vehicle.model == model,
#             Vehicle.engine_size == engine_size
#         ).distinct().order_by(Vehicle.year.desc()).all()
        
#         result = [{'value': year[0], 'label': str(year[0])} for year in years]
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/vehicles/validate', methods=['POST'])
# def validate_vehicle_configuration():
#     """Validate if a specific vehicle configuration exists in database"""
#     try:
#         data = request.json
#         make = data.get('make')
#         model = data.get('model')
#         engine_size = data.get('engine_size')
#         year = data.get('year')
        
#         # Check if exact configuration exists
#         vehicle = Vehicle.query.filter_by(
#             make=make,
#             model=model,
#             engine_size=engine_size,
#             year=year
#         ).first()
        
#         if vehicle:
#             return jsonify({
#                 'valid': True,
#                 'vehicle': {
#                     'make': vehicle.make,
#                     'model': vehicle.model,
#                     'year': vehicle.year,
#                     'engine_size': vehicle.engine_size,
#                     'cylinders': vehicle.cylinders,
#                     'transmission': vehicle.transmission,
#                     'fuel_type': vehicle.fuel_type,
#                     'combined_mpg': vehicle.combined_mpg
#                 }
#             })
#         else:
#             # Try to find closest match without year requirement
#             closest = Vehicle.query.filter_by(
#                 make=make,
#                 model=model,
#                 engine_size=engine_size
#             ).first()
            
#             if closest:
#                 return jsonify({
#                     'valid': True,
#                     'vehicle': {
#                         'make': closest.make,
#                         'model': closest.model,
#                         'year': closest.year,
#                         'engine_size': closest.engine_size,
#                         'cylinders': closest.cylinders,
#                         'transmission': closest.transmission,
#                         'fuel_type': closest.fuel_type,
#                         'combined_mpg': closest.combined_mpg
#                     },
#                     'note': 'Using closest available year'
#                 })
#             else:
#                 return jsonify({'valid': False, 'error': 'Vehicle configuration not found'})
        
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500
    
    
# @app.route('/api/vehicles/all', methods=['GET'])
# def get_all_vehicles():
#     """Get all vehicles in database"""
#     try:
#         vehicles = Vehicle.query.all()
#         result = []
        
#         for vehicle in vehicles:
#             result.append({
#                 'id': vehicle.id,
#                 'make': vehicle.make,
#                 'model': vehicle.model,
#                 'year': vehicle.year,
#                 'engine_size': vehicle.engine_size,
#                 'cylinders': vehicle.cylinders,
#                 'transmission': vehicle.transmission,
#                 'fuel_type': vehicle.fuel_type,
#                 'combined_mpg': vehicle.combined_mpg,
#                 'city_mpg': vehicle.city_mpg,
#                 'highway_mpg': vehicle.highway_mpg,
#                 'co2_emissions': vehicle.co2_emissions
#             })
        
#         return jsonify(result)
        
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500    
    
    
# @app.route('/api/vehicles/years_for_model/<make>/<model>', methods=['GET'])
# def get_years_for_model_only(make, model):
#     """Get all available years for a specific make and model (regardless of engine size)"""
#     try:
#         years = db.session.query(Vehicle.year).filter(
#             Vehicle.make == make,
#             Vehicle.model == model
#         ).distinct().order_by(Vehicle.year.desc()).all()
        
#         result = [{'value': year[0], 'label': str(year[0])} for year in years]
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500


# if __name__ == '__main__':
#     with app.app_context():
#         db.create_all()
#         # Populate sample data
#         if Vehicle.query.count() == 0:
#             sample_vehicles = [
#                 Vehicle(make='Toyota', model='Prius', year=2022, engine_size=1.8, cylinders=4, 
#                        transmission='CVT', fuel_type='Hybrid', city_mpg=58, highway_mpg=53, 
#                        combined_mpg=56, co2_emissions=96),
#                 Vehicle(make='Honda', model='Civic', year=2022, engine_size=2.0, cylinders=4, 
#                        transmission='Manual', fuel_type='Gasoline', city_mpg=32, highway_mpg=42, 
#                        combined_mpg=36, co2_emissions=262),
#                 Vehicle(make='Ford', model='F-150', year=2022, engine_size=3.3, cylinders=6, 
#                        transmission='Automatic', fuel_type='Gasoline', city_mpg=19, highway_mpg=24, 
#                        combined_mpg=21, co2_emissions=555),
#             ]
#             for vehicle in sample_vehicles:
#                 db.session.add(vehicle)
#             db.session.commit()
        
#         load_ml_models()
#         print("🌱 GreenAI Fuel Efficiency Predictor API is running!")
#         print("📍 Backend: http://localhost:5000")
#         print("📍 API Health: http://localhost:5000/api/health")
#         print("📍 API Docs: http://localhost:5000/")
    
#     app.run(debug=True, port=5000, host='0.0.0.0')






# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from flask_sqlalchemy import SQLAlchemy
# import pandas as pd
# import numpy as np
# import joblib
# import os
# from datetime import datetime
# import tensorflow as tf
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestRegressor
# import json
# import sys

# # Platform-specific imports
# try:
#     import fcntl
#     HAS_FCNTL = True
# except ImportError:
#     HAS_FCNTL = False  # Windows doesn't have fcntl
#     print("⚠️ fcntl not available (Windows system) - file locking disabled")

# app = Flask(__name__)

# # Production configuration
# if os.environ.get('FLASK_ENV') == 'production':
#     app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', '').replace('postgres://', 'postgresql://')
#     app.config['DEBUG'] = False
# else:
#     app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fuel_efficiency.db'
#     app.config['DEBUG'] = True

# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# # CORS configuration for production
# CORS(app, origins=[
#     "http://localhost:3000",
#     "https://fuel-predictor-frontend.onrender.com",
#     "https://*.onrender.com"
# ])

# db = SQLAlchemy(app)

# # Vehicle Model
# class Vehicle(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     make = db.Column(db.String(100), nullable=False)
#     model = db.Column(db.String(100), nullable=False)
#     year = db.Column(db.Integer, nullable=False)
#     engine_size = db.Column(db.Float, nullable=False)
#     cylinders = db.Column(db.Integer, nullable=False)
#     transmission = db.Column(db.String(50), nullable=False)
#     fuel_type = db.Column(db.String(50), nullable=False)
#     city_mpg = db.Column(db.Float, nullable=False)
#     highway_mpg = db.Column(db.Float, nullable=False)
#     combined_mpg = db.Column(db.Float, nullable=False)
#     co2_emissions = db.Column(db.Float, nullable=False)
#     created_at = db.Column(db.DateTime, default=datetime.now)

# # Prediction History Model
# class PredictionHistory(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     user_input = db.Column(db.Text, nullable=False)
#     predicted_mpg = db.Column(db.Float, nullable=False)
#     environmental_impact = db.Column(db.Float, nullable=False)
#     recommendations = db.Column(db.Text, nullable=False)
#     created_at = db.Column(db.DateTime, default=datetime.now)

# # Initialize ML models
# ml_model = None
# scaler = None
# deep_learning_model = None

# def safe_file_operation(file_path, mode, operation):
#     """Safely perform file operations with proper locking on Unix systems"""
#     try:
#         with open(file_path, mode) as f:
#             if HAS_FCNTL and mode != 'r':
#                 fcntl.flock(f.fileno(), fcntl.LOCK_EX)
#             result = operation(f)
#             if HAS_FCNTL and mode != 'r':
#                 fcntl.flock(f.fileno(), fcntl.LOCK_UN)
#             return result
#     except Exception as e:
#         print(f"File operation error: {str(e)}")
#         raise

# def load_ml_models():
#     global ml_model, scaler, deep_learning_model
#     try:
#         # Check if model files exist
#         model_files = [
#             'ml_models/fuel_efficiency_model.pkl',
#             'ml_models/scaler.pkl',
#             'ml_models/deep_learning_model.h5'
#         ]
        
#         missing_files = [f for f in model_files if not os.path.exists(f)]
        
#         if missing_files:
#             print(f"⚠️ Missing model files: {missing_files}")
#             print("🔄 Training new models...")
#             train_models()
        
#         # Load models
#         ml_model = joblib.load('ml_models/fuel_efficiency_model.pkl')
#         scaler = joblib.load('ml_models/scaler.pkl')
#         deep_learning_model = tf.keras.models.load_model('ml_models/deep_learning_model.h5')
#         print("✅ All models loaded successfully!")
        
#     except Exception as e:
#         print(f"❌ Error loading models: {e}")
#         print("🔄 Training new models...")
#         train_models()

# def train_models():
#     global ml_model, scaler, deep_learning_model
    
#     # Generate sample training data (replace with real data)
#     data = generate_sample_data()
    
#     # Features and target
#     features = ['engine_size', 'cylinders', 'year', 'transmission_encoded', 'fuel_type_encoded']
#     X = data[features]
#     y = data['combined_mpg']
    
#     # Scale features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
    
#     # Train Random Forest
#     ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
#     ml_model.fit(X_scaled, y)
    
#     # Train Deep Learning Model
#     deep_learning_model = tf.keras.Sequential([
#         tf.keras.layers.Dense(128, activation='relu', input_shape=(len(features),)),
#         tf.keras.layers.Dropout(0.3),
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.Dropout(0.3),
#         tf.keras.layers.Dense(32, activation='relu'),
#         tf.keras.layers.Dense(1)
#     ])
    
#     deep_learning_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
#     deep_learning_model.fit(X_scaled, y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    
#     # Save models
#     os.makedirs('ml_models', exist_ok=True)
#     joblib.dump(ml_model, 'ml_models/fuel_efficiency_model.pkl')
#     joblib.dump(scaler, 'ml_models/scaler.pkl')
#     deep_learning_model.save('ml_models/deep_learning_model.h5')
#     print("✅ Models trained and saved successfully!")

# def generate_sample_data():
#     np.random.seed(42)
#     n_samples = 1000
    
#     data = {
#         'engine_size': np.random.normal(2.5, 0.8, n_samples),
#         'cylinders': np.random.choice([4, 6, 8], n_samples),
#         'year': np.random.randint(2015, 2025, n_samples),
#         'transmission_encoded': np.random.choice([0, 1], n_samples),  # 0: Manual, 1: Automatic
#         'fuel_type_encoded': np.random.choice([0, 1, 2], n_samples),  # 0: Gasoline, 1: Diesel, 2: Hybrid
#     }
    
#     # Generate realistic MPG based on features
#     mpg = (30 - data['engine_size'] * 3 + 
#            (data['cylinders'] == 4) * 5 - 
#            (data['cylinders'] == 8) * 8 +
#            (data['year'] - 2015) * 0.5 +
#            data['transmission_encoded'] * -2 +
#            data['fuel_type_encoded'] * 3 +
#            np.random.normal(0, 3, n_samples))
    
#     data['combined_mpg'] = np.clip(mpg, 15, 50)
    
#     return pd.DataFrame(data)

# def calculate_environmental_impact(mpg, annual_miles=12000):
#     """Calculate CO2 emissions and environmental impact"""
#     if mpg <= 0:
#         mpg = 1  # Avoid division by zero
        
#     gallons_per_year = annual_miles / mpg
#     co2_per_gallon = 19.6  # pounds of CO2 per gallon of gasoline
#     annual_co2 = gallons_per_year * co2_per_gallon
    
#     # Convert to tons
#     co2_tons = annual_co2 / 2000
    
#     return {
#         'annual_gallons': round(float(gallons_per_year), 2),
#         'annual_co2_pounds': round(float(annual_co2), 2),
#         'annual_co2_tons': round(float(co2_tons), 2),
#         'trees_to_offset': int(round(co2_tons * 16, 0))  # ~16 trees per ton of CO2
#     }

# def get_green_recommendations(mpg, vehicle_data):
#     """Generate eco-friendly recommendations"""
#     recommendations = []
    
#     if mpg < 25:
#         recommendations.extend([
#             "Consider hybrid or electric alternatives",
#             "Regular maintenance can improve fuel efficiency by 4-10%",
#             "Proper tire inflation can improve MPG by up to 3%"
#         ])
    
#     if mpg < 30:
#         recommendations.extend([
#             "Avoid aggressive driving - can improve MPG by 15-30%",
#             "Remove excess weight from your vehicle",
#             "Use cruise control on highways"
#         ])
    
#     recommendations.extend([
#         "Consider carpooling or public transportation",
#         "Plan and combine trips to reduce overall driving",
#         "Keep your vehicle well-maintained"
#     ])
    
#     # Ensure all recommendations are strings
#     return [str(rec) for rec in recommendations]

# @app.route('/api/predict', methods=['POST'])
# def predict_fuel_efficiency():
#     try:
#         # Check if models are loaded
#         if ml_model is None or scaler is None or deep_learning_model is None:
#             print("❌ Models not loaded, attempting to load...")
#             load_ml_models()
            
#         if ml_model is None:
#             return jsonify({'error': 'ML models not available. Please try again later.'}), 503
            
#         data = request.json
#         print(f"📥 Received prediction request: {data}")
        
#         # Extract features with validation
#         try:
#             features = {
#                 'engine_size': float(data.get('engine_size', 2.5)),
#                 'cylinders': int(data.get('cylinders', 4)),
#                 'year': int(data.get('year', 2020)),
#                 'transmission_encoded': 1 if data.get('transmission', 'automatic').lower() == 'automatic' else 0,
#                 'fuel_type_encoded': {'gasoline': 0, 'diesel': 1, 'hybrid': 2, 'electric': 3}.get(
#                     data.get('fuel_type', 'gasoline').lower(), 0)
#             }
#             print(f"🔧 Processed features: {features}")
#         except (ValueError, TypeError) as e:
#             return jsonify({'error': f'Invalid input data: {str(e)}'}), 400
        
#         # Prepare features for prediction
#         feature_array = np.array([[features['engine_size'], features['cylinders'], 
#                                  features['year'], features['transmission_encoded'], 
#                                  features['fuel_type_encoded']]])
        
#         print(f"📊 Feature array shape: {feature_array.shape}")
#         print(f"📊 Feature array: {feature_array}")
        
#         # Scale features
#         try:
#             feature_scaled = scaler.transform(feature_array)
#             print(f"📈 Scaled features: {feature_scaled}")
#         except Exception as e:
#             print(f"❌ Scaling error: {e}")
#             return jsonify({'error': 'Feature scaling failed'}), 500
        
#         # Make predictions
#         try:
#             ml_prediction = ml_model.predict(feature_scaled)[0]
#             print(f"🤖 ML prediction: {ml_prediction}")
#         except Exception as e:
#             print(f"❌ ML prediction error: {e}")
#             ml_prediction = 25.0  # Fallback
            
#         try:
#             dl_prediction = deep_learning_model.predict(feature_scaled, verbose=0)[0][0]
#             print(f"🧠 DL prediction: {dl_prediction}")
#         except Exception as e:
#             print(f"❌ DL prediction error: {e}")
#             dl_prediction = 25.0  # Fallback
        
#         # Ensemble prediction (average of both models)
#         final_prediction = (ml_prediction + dl_prediction) / 2
#         final_prediction = max(15, min(50, final_prediction))  # Clamp between realistic values
        
#         print(f"🎯 Final prediction: {final_prediction}")
        
#         # Calculate environmental impact
#         env_impact = calculate_environmental_impact(final_prediction)
        
#         # Get recommendations
#         recommendations = get_green_recommendations(final_prediction, data)
        
#         # Save to database
#         try:
#             prediction_record = PredictionHistory(
#                 user_input=json.dumps(data),
#                 predicted_mpg=final_prediction,
#                 environmental_impact=env_impact['annual_co2_tons'],
#                 recommendations=json.dumps(recommendations)
#             )
#             db.session.add(prediction_record)
#             db.session.commit()
#             print("💾 Prediction saved to database")
#         except Exception as e:
#             print(f"⚠️ Database save error: {e}")
#             # Continue without saving to database
        
#         response = {
#             'predicted_mpg': round(final_prediction, 2),
#             'ml_prediction': round(float(ml_prediction), 2),
#             'dl_prediction': round(float(dl_prediction), 2),
#             'environmental_impact': env_impact,
#             'recommendations': recommendations,
#             'green_score': round(min(100, max(0, (final_prediction - 15) * 100 / 35)), 2),  # Score out of 100
#             'comparison': {
#                 'average_vehicle_mpg': 25.4,
#                 'better_than_average': bool(final_prediction > 25.4)  # Explicitly convert to bool
#             }
#         }
        
#         # Debug: Test JSON serialization before returning
#         try:
#             json.dumps(response)
#             print("✅ JSON serialization test passed")
#         except Exception as json_err:
#             print(f"❌ JSON serialization test failed: {json_err}")
#             print(f"🔍 Response data types: {[(k, type(v)) for k, v in response.items()]}")
#             # Return a simpler response if serialization fails
#             response = {
#                 'predicted_mpg': float(final_prediction),
#                 'error': 'Simplified response due to serialization issue'
#             }
        
#         print(f"✅ Prediction successful: {response['predicted_mpg']} MPG")
#         return jsonify(response)
        
#     except Exception as e:
#         error_msg = f'Prediction failed: {str(e)}'
#         print(f"❌ {error_msg}")
#         import traceback
#         traceback.print_exc()  # This will show the full error traceback
#         return jsonify({'error': error_msg}), 500

# @app.route('/api/vehicles/search', methods=['GET'])
# def search_vehicles():
#     """Search vehicles by make and model"""
#     make = request.args.get('make', '')
#     model = request.args.get('model', '')
    
#     query = Vehicle.query
#     if make:
#         query = query.filter(Vehicle.make.ilike(f'%{make}%'))
#     if model:
#         query = query.filter(Vehicle.model.ilike(f'%{model}%'))
    
#     vehicles = query.limit(20).all()
    
#     result = []
#     for vehicle in vehicles:
#         result.append({
#             'id': vehicle.id,
#             'make': vehicle.make,
#             'model': vehicle.model,
#             'year': vehicle.year,
#             'engine_size': vehicle.engine_size,
#             'cylinders': vehicle.cylinders,
#             'transmission': vehicle.transmission,
#             'fuel_type': vehicle.fuel_type,
#             'combined_mpg': vehicle.combined_mpg,
#             'co2_emissions': vehicle.co2_emissions
#         })
    
#     return jsonify(result)

# @app.route('/api/history', methods=['GET'])
# def get_prediction_history():
#     """Get recent prediction history"""
#     history = PredictionHistory.query.order_by(PredictionHistory.created_at.desc()).limit(10).all()
    
#     result = []
#     for record in history:
#         result.append({
#             'id': record.id,
#             'predicted_mpg': record.predicted_mpg,
#             'environmental_impact': record.environmental_impact,
#             'created_at': record.created_at.isoformat()
#         })
    
#     return jsonify(result)

# @app.route('/api/stats', methods=['GET'])
# def get_statistics():
#     """Get application statistics"""
#     total_predictions = PredictionHistory.query.count()
#     avg_mpg = db.session.query(db.func.avg(PredictionHistory.predicted_mpg)).scalar() or 0
#     total_co2_saved = db.session.query(db.func.sum(PredictionHistory.environmental_impact)).scalar() or 0
    
#     return jsonify({
#         'total_predictions': total_predictions,
#         'average_predicted_mpg': round(float(avg_mpg), 2),
#         'total_co2_impact_tons': round(float(total_co2_saved), 2),
#         'trees_equivalent': round(float(total_co2_saved) * 16, 0)
#     })

# @app.route('/', methods=['GET'])
# def index():
#     """Root endpoint"""
#     return jsonify({
#         'message': 'GreenAI Fuel Efficiency Predictor API',
#         'status': 'running',
#         'version': '1.0.0',
#         'endpoints': {
#             'predict': '/api/predict',
#             'vehicles': '/api/vehicles/search',
#             'stats': '/api/stats',
#             'history': '/api/history',
#             'health': '/api/health'
#         }
#     })

# @app.route('/api/health', methods=['GET'])
# def health_check():
#     return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

# @app.route('/api/vehicles/makes', methods=['GET'])
# def get_vehicle_makes():
#     """Get unique vehicle makes (brands) for dropdown"""
#     try:
#         makes = db.session.query(Vehicle.make).distinct().order_by(Vehicle.make).all()
#         result = [{'value': make[0], 'label': make[0]} for make in makes]
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/vehicles/models/<make>', methods=['GET'])
# def get_models_for_make(make):
#     """Get models for a specific make"""
#     try:
#         models = db.session.query(Vehicle.model).filter(
#             Vehicle.make == make
#         ).distinct().order_by(Vehicle.model).all()
        
#         result = [{'value': model[0], 'label': model[0]} for model in models]
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/vehicles/engine_sizes/<make>/<model>', methods=['GET'])
# def get_engine_sizes_for_model(make, model):
#     """Get engine sizes for a specific make and model"""
#     try:
#         engine_sizes = db.session.query(Vehicle.engine_size).filter(
#             Vehicle.make == make,
#             Vehicle.model == model
#         ).distinct().order_by(Vehicle.engine_size).all()
        
#         result = []
#         for size in engine_sizes:
#             size_value = size[0]
#             if size_value == 0:
#                 label = "Electric (0L)"
#             else:
#                 label = f"{size_value}L"
#             result.append({'value': size_value, 'label': label})
        
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/vehicles/years/<make>/<model>/<float:engine_size>', methods=['GET'])
# def get_years_for_vehicle(make, model, engine_size):
#     """Get available years for a specific vehicle configuration"""
#     try:
#         years = db.session.query(Vehicle.year).filter(
#             Vehicle.make == make,
#             Vehicle.model == model,
#             Vehicle.engine_size == engine_size
#         ).distinct().order_by(Vehicle.year.desc()).all()
        
#         result = [{'value': year[0], 'label': str(year[0])} for year in years]
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/vehicles/validate', methods=['POST'])
# def validate_vehicle_configuration():
#     """Validate if a specific vehicle configuration exists in database"""
#     try:
#         data = request.json
#         make = data.get('make')
#         model = data.get('model')
#         engine_size = data.get('engine_size')
#         year = data.get('year')
        
#         # Check if exact configuration exists
#         vehicle = Vehicle.query.filter_by(
#             make=make,
#             model=model,
#             engine_size=engine_size,
#             year=year
#         ).first()
        
#         if vehicle:
#             return jsonify({
#                 'valid': True,
#                 'vehicle': {
#                     'make': vehicle.make,
#                     'model': vehicle.model,
#                     'year': vehicle.year,
#                     'engine_size': vehicle.engine_size,
#                     'cylinders': vehicle.cylinders,
#                     'transmission': vehicle.transmission,
#                     'fuel_type': vehicle.fuel_type,
#                     'combined_mpg': vehicle.combined_mpg
#                 }
#             })
#         else:
#             # Try to find closest match without year requirement
#             closest = Vehicle.query.filter_by(
#                 make=make,
#                 model=model,
#                 engine_size=engine_size
#             ).first()
            
#             if closest:
#                 return jsonify({
#                     'valid': True,
#                     'vehicle': {
#                         'make': closest.make,
#                         'model': closest.model,
#                         'year': closest.year,
#                         'engine_size': closest.engine_size,
#                         'cylinders': closest.cylinders,
#                         'transmission': closest.transmission,
#                         'fuel_type': closest.fuel_type,
#                         'combined_mpg': closest.combined_mpg
#                     },
#                     'note': 'Using closest available year'
#                 })
#             else:
#                 return jsonify({'valid': False, 'error': 'Vehicle configuration not found'})
        
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/vehicles/all', methods=['GET'])
# def get_all_vehicles():
#     """Get all vehicles in database"""
#     try:
#         vehicles = Vehicle.query.all()
#         result = []
        
#         for vehicle in vehicles:
#             result.append({
#                 'id': vehicle.id,
#                 'make': vehicle.make,
#                 'model': vehicle.model,
#                 'year': vehicle.year,
#                 'engine_size': vehicle.engine_size,
#                 'cylinders': vehicle.cylinders,
#                 'transmission': vehicle.transmission,
#                 'fuel_type': vehicle.fuel_type,
#                 'combined_mpg': vehicle.combined_mpg,
#                 'city_mpg': vehicle.city_mpg,
#                 'highway_mpg': vehicle.highway_mpg,
#                 'co2_emissions': vehicle.co2_emissions
#             })
        
#         return jsonify(result)
        
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/vehicles/years_for_model/<make>/<model>', methods=['GET'])
# def get_years_for_model_only(make, model):
#     """Get all available years for a specific make and model (regardless of engine size)"""
#     try:
#         years = db.session.query(Vehicle.year).filter(
#             Vehicle.make == make,
#             Vehicle.model == model
#         ).distinct().order_by(Vehicle.year.desc()).all()
        
#         result = [{'value': year[0], 'label': str(year[0])} for year in years]
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     with app.app_context():
#         db.create_all()
#         print("✅ Database tables created")
        
#         # Check if we have data
#         vehicle_count = Vehicle.query.count()
#         if vehicle_count == 0:
#             print("⚠️ No vehicles in database. Run csv_data_processor.py to load data.")
#         else:
#             print(f"✅ Found {vehicle_count} vehicles in database")
        
#         # Load ML models
#         load_ml_models()
        
#         print("\n" + "="*60)
#         print("🌱 GreenAI Fuel Efficiency Predictor API is running!")
#         print("="*60)
#         print("📍 Backend: http://localhost:5000")
#         print("📍 API Health: http://localhost:5000/api/health")
#         print("📍 API Docs: http://localhost:5000/")
#         print("="*60 + "\n")
    
#     app.run(debug=True, port=5000, host='0.0.0.0')
    
    
#########################################################    
    
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from flask_sqlalchemy import SQLAlchemy
# import pandas as pd
# import numpy as np
# import joblib
# import os
# from datetime import datetime
# import tensorflow as tf
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestRegressor
# import json
# import sys

# # Platform-specific imports
# try:
#     import fcntl
#     HAS_FCNTL = True
# except ImportError:
#     HAS_FCNTL = False  # Windows doesn't have fcntl
#     print("⚠️ fcntl not available (Windows system) - file locking disabled")

# app = Flask(__name__)

# # Production configuration
# if os.environ.get('FLASK_ENV') == 'production' or os.environ.get('DATABASE_URL'):
#     database_url = os.environ.get('DATABASE_URL', '')
#     if database_url.startswith('postgres://'):
#         database_url = database_url.replace('postgres://', 'postgresql://', 1)
#     app.config['SQLALCHEMY_DATABASE_URI'] = database_url
#     app.config['DEBUG'] = False
#     print(f"✅ Using PostgreSQL database")
# else:
#     app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fuel_efficiency.db'
#     app.config['DEBUG'] = True
#     print(f"✅ Using SQLite database")

# app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# # CORS configuration for production
# CORS(app, origins=[
#     "http://localhost:3000",
#     "https://fuel-predictor-frontend.onrender.com",
#     "https://*.onrender.com"
# ])

# db = SQLAlchemy(app)

# # Vehicle Model
# class Vehicle(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     make = db.Column(db.String(100), nullable=False)
#     model = db.Column(db.String(100), nullable=False)
#     year = db.Column(db.Integer, nullable=False)
#     engine_size = db.Column(db.Float, nullable=False)
#     cylinders = db.Column(db.Integer, nullable=False)
#     transmission = db.Column(db.String(50), nullable=False)
#     fuel_type = db.Column(db.String(50), nullable=False)
#     city_mpg = db.Column(db.Float, nullable=False)
#     highway_mpg = db.Column(db.Float, nullable=False)
#     combined_mpg = db.Column(db.Float, nullable=False)
#     co2_emissions = db.Column(db.Float, nullable=False)
#     created_at = db.Column(db.DateTime, default=datetime.now)

# # Prediction History Model
# class PredictionHistory(db.Model):
#     id = db.Column(db.Integer, primary_key=True)
#     user_input = db.Column(db.Text, nullable=False)
#     predicted_mpg = db.Column(db.Float, nullable=False)
#     environmental_impact = db.Column(db.Float, nullable=False)
#     recommendations = db.Column(db.Text, nullable=False)
#     created_at = db.Column(db.DateTime, default=datetime.now)

# # Initialize ML models
# ml_model = None
# scaler = None
# deep_learning_model = None

# def safe_file_operation(file_path, mode, operation):
#     """Safely perform file operations with proper locking on Unix systems"""
#     try:
#         with open(file_path, mode) as f:
#             if HAS_FCNTL and mode != 'r':
#                 fcntl.flock(f.fileno(), fcntl.LOCK_EX)
#             result = operation(f)
#             if HAS_FCNTL and mode != 'r':
#                 fcntl.flock(f.fileno(), fcntl.LOCK_UN)
#             return result
#     except Exception as e:
#         print(f"File operation error: {str(e)}")
#         raise

# def initialize_database():
#     """Initialize database tables"""
#     try:
#         with app.app_context():
#             # Create all tables
#             db.create_all()
#             print("✅ Database tables created successfully")
            
#             # Check if we have data
#             vehicle_count = Vehicle.query.count()
#             if vehicle_count == 0:
#                 print("⚠️ No vehicles in database. Adding sample data...")
#                 add_sample_data()
#             else:
#                 print(f"✅ Found {vehicle_count} vehicles in database")
            
#             return True
#     except Exception as e:
#         print(f"❌ Error initializing database: {e}")
#         import traceback
#         traceback.print_exc()
#         return False

# def add_sample_data():
#     """Add sample vehicle data if database is empty"""
#     try:
#         sample_vehicles = [
#             Vehicle(make='Toyota', model='Prius', year=2022, engine_size=1.8, cylinders=4, 
#                    transmission='CVT', fuel_type='Hybrid', city_mpg=58, highway_mpg=53, 
#                    combined_mpg=56, co2_emissions=96),
#             Vehicle(make='Honda', model='Civic', year=2023, engine_size=2.0, cylinders=4, 
#                    transmission='Manual', fuel_type='Gasoline', city_mpg=32, highway_mpg=42, 
#                    combined_mpg=36, co2_emissions=262),
#             Vehicle(make='Ford', model='F-150', year=2023, engine_size=3.3, cylinders=6, 
#                    transmission='Automatic', fuel_type='Gasoline', city_mpg=19, highway_mpg=24, 
#                    combined_mpg=21, co2_emissions=555),
#             Vehicle(make='Tesla', model='Model 3', year=2023, engine_size=0.0, cylinders=0, 
#                    transmission='Automatic', fuel_type='Electric', city_mpg=142, highway_mpg=132, 
#                    combined_mpg=137, co2_emissions=0),
#             Vehicle(make='Chevrolet', model='Malibu', year=2022, engine_size=1.5, cylinders=4, 
#                    transmission='Automatic', fuel_type='Gasoline', city_mpg=29, highway_mpg=36, 
#                    combined_mpg=32, co2_emissions=277),
#         ]
        
#         for vehicle in sample_vehicles:
#             db.session.add(vehicle)
        
#         db.session.commit()
#         print(f"✅ Added {len(sample_vehicles)} sample vehicles")
        
#     except Exception as e:
#         print(f"❌ Error adding sample data: {e}")
#         db.session.rollback()

# def load_ml_models():
#     global ml_model, scaler, deep_learning_model
#     try:
#         # Check if model files exist
#         model_files = [
#             'ml_models/fuel_efficiency_model.pkl',
#             'ml_models/scaler.pkl',
#             'ml_models/deep_learning_model.h5'
#         ]
        
#         missing_files = [f for f in model_files if not os.path.exists(f)]
        
#         if missing_files:
#             print(f"⚠️ Missing model files: {missing_files}")
#             print("🔄 Training new models...")
#             train_models()
        
#         # Load models
#         ml_model = joblib.load('ml_models/fuel_efficiency_model.pkl')
#         scaler = joblib.load('ml_models/scaler.pkl')
#         deep_learning_model = tf.keras.models.load_model('ml_models/deep_learning_model.h5')
#         print("✅ All models loaded successfully!")
        
#     except Exception as e:
#         print(f"❌ Error loading models: {e}")
#         print("🔄 Training new models...")
#         train_models()

# def train_models():
#     global ml_model, scaler, deep_learning_model
    
#     # Generate sample training data (replace with real data)
#     data = generate_sample_data()
    
#     # Features and target
#     features = ['engine_size', 'cylinders', 'year', 'transmission_encoded', 'fuel_type_encoded']
#     X = data[features]
#     y = data['combined_mpg']
    
#     # Scale features
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
    
#     # Train Random Forest
#     ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
#     ml_model.fit(X_scaled, y)
    
#     # Train Deep Learning Model
#     deep_learning_model = tf.keras.Sequential([
#         tf.keras.layers.Dense(128, activation='relu', input_shape=(len(features),)),
#         tf.keras.layers.Dropout(0.3),
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.Dropout(0.3),
#         tf.keras.layers.Dense(32, activation='relu'),
#         tf.keras.layers.Dense(1)
#     ])
    
#     deep_learning_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
#     deep_learning_model.fit(X_scaled, y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    
#     # Save models
#     os.makedirs('ml_models', exist_ok=True)
#     joblib.dump(ml_model, 'ml_models/fuel_efficiency_model.pkl')
#     joblib.dump(scaler, 'ml_models/scaler.pkl')
#     deep_learning_model.save('ml_models/deep_learning_model.h5')
#     print("✅ Models trained and saved successfully!")

# def generate_sample_data():
#     np.random.seed(42)
#     n_samples = 1000
    
#     data = {
#         'engine_size': np.random.normal(2.5, 0.8, n_samples),
#         'cylinders': np.random.choice([4, 6, 8], n_samples),
#         'year': np.random.randint(2015, 2025, n_samples),
#         'transmission_encoded': np.random.choice([0, 1], n_samples),
#         'fuel_type_encoded': np.random.choice([0, 1, 2], n_samples),
#     }
    
#     mpg = (30 - data['engine_size'] * 3 + 
#            (data['cylinders'] == 4) * 5 - 
#            (data['cylinders'] == 8) * 8 +
#            (data['year'] - 2015) * 0.5 +
#            data['transmission_encoded'] * -2 +
#            data['fuel_type_encoded'] * 3 +
#            np.random.normal(0, 3, n_samples))
    
#     data['combined_mpg'] = np.clip(mpg, 15, 50)
    
#     return pd.DataFrame(data)

# def calculate_environmental_impact(mpg, annual_miles=12000):
#     """Calculate CO2 emissions and environmental impact"""
#     if mpg <= 0:
#         mpg = 1
        
#     gallons_per_year = annual_miles / mpg
#     co2_per_gallon = 19.6
#     annual_co2 = gallons_per_year * co2_per_gallon
#     co2_tons = annual_co2 / 2000
    
#     return {
#         'annual_gallons': round(float(gallons_per_year), 2),
#         'annual_co2_pounds': round(float(annual_co2), 2),
#         'annual_co2_tons': round(float(co2_tons), 2),
#         'trees_to_offset': int(round(co2_tons * 16, 0))
#     }

# def get_green_recommendations(mpg, vehicle_data):
#     """Generate eco-friendly recommendations"""
#     recommendations = []
    
#     if mpg < 25:
#         recommendations.extend([
#             "Consider hybrid or electric alternatives",
#             "Regular maintenance can improve fuel efficiency by 4-10%",
#             "Proper tire inflation can improve MPG by up to 3%"
#         ])
    
#     if mpg < 30:
#         recommendations.extend([
#             "Avoid aggressive driving - can improve MPG by 15-30%",
#             "Remove excess weight from your vehicle",
#             "Use cruise control on highways"
#         ])
    
#     recommendations.extend([
#         "Consider carpooling or public transportation",
#         "Plan and combine trips to reduce overall driving",
#         "Keep your vehicle well-maintained"
#     ])
    
#     return [str(rec) for rec in recommendations]

# @app.route('/api/predict', methods=['POST'])
# def predict_fuel_efficiency():
#     try:
#         if ml_model is None or scaler is None or deep_learning_model is None:
#             print("❌ Models not loaded, attempting to load...")
#             load_ml_models()
            
#         if ml_model is None:
#             return jsonify({'error': 'ML models not available. Please try again later.'}), 503
            
#         data = request.json
#         print(f"📥 Received prediction request: {data}")
        
#         try:
#             features = {
#                 'engine_size': float(data.get('engine_size', 2.5)),
#                 'cylinders': int(data.get('cylinders', 4)),
#                 'year': int(data.get('year', 2020)),
#                 'transmission_encoded': 1 if data.get('transmission', 'automatic').lower() == 'automatic' else 0,
#                 'fuel_type_encoded': {'gasoline': 0, 'diesel': 1, 'hybrid': 2, 'electric': 3}.get(
#                     data.get('fuel_type', 'gasoline').lower(), 0)
#             }
#         except (ValueError, TypeError) as e:
#             return jsonify({'error': f'Invalid input data: {str(e)}'}), 400
        
#         feature_array = np.array([[features['engine_size'], features['cylinders'], 
#                                  features['year'], features['transmission_encoded'], 
#                                  features['fuel_type_encoded']]])
        
#         try:
#             feature_scaled = scaler.transform(feature_array)
#         except Exception as e:
#             print(f"❌ Scaling error: {e}")
#             return jsonify({'error': 'Feature scaling failed'}), 500
        
#         try:
#             ml_prediction = ml_model.predict(feature_scaled)[0]
#         except Exception as e:
#             print(f"❌ ML prediction error: {e}")
#             ml_prediction = 25.0
            
#         try:
#             dl_prediction = deep_learning_model.predict(feature_scaled, verbose=0)[0][0]
#         except Exception as e:
#             print(f"❌ DL prediction error: {e}")
#             dl_prediction = 25.0
        
#         final_prediction = (ml_prediction + dl_prediction) / 2
#         final_prediction = max(15, min(50, final_prediction))
        
#         env_impact = calculate_environmental_impact(final_prediction)
#         recommendations = get_green_recommendations(final_prediction, data)
        
#         try:
#             prediction_record = PredictionHistory(
#                 user_input=json.dumps(data),
#                 predicted_mpg=final_prediction,
#                 environmental_impact=env_impact['annual_co2_tons'],
#                 recommendations=json.dumps(recommendations)
#             )
#             db.session.add(prediction_record)
#             db.session.commit()
#         except Exception as e:
#             print(f"⚠️ Database save error: {e}")
        
#         response = {
#             'predicted_mpg': round(final_prediction, 2),
#             'ml_prediction': round(float(ml_prediction), 2),
#             'dl_prediction': round(float(dl_prediction), 2),
#             'environmental_impact': env_impact,
#             'recommendations': recommendations,
#             'green_score': round(min(100, max(0, (final_prediction - 15) * 100 / 35)), 2),
#             'comparison': {
#                 'average_vehicle_mpg': 25.4,
#                 'better_than_average': bool(final_prediction > 25.4)
#             }
#         }
        
#         return jsonify(response)
        
#     except Exception as e:
#         error_msg = f'Prediction failed: {str(e)}'
#         print(f"❌ {error_msg}")
#         import traceback
#         traceback.print_exc()
#         return jsonify({'error': error_msg}), 500

# @app.route('/api/vehicles/search', methods=['GET'])
# def search_vehicles():
#     make = request.args.get('make', '')
#     model = request.args.get('model', '')
    
#     query = Vehicle.query
#     if make:
#         query = query.filter(Vehicle.make.ilike(f'%{make}%'))
#     if model:
#         query = query.filter(Vehicle.model.ilike(f'%{model}%'))
    
#     vehicles = query.limit(20).all()
    
#     result = []
#     for vehicle in vehicles:
#         result.append({
#             'id': vehicle.id,
#             'make': vehicle.make,
#             'model': vehicle.model,
#             'year': vehicle.year,
#             'engine_size': vehicle.engine_size,
#             'cylinders': vehicle.cylinders,
#             'transmission': vehicle.transmission,
#             'fuel_type': vehicle.fuel_type,
#             'combined_mpg': vehicle.combined_mpg,
#             'co2_emissions': vehicle.co2_emissions
#         })
    
#     return jsonify(result)

# @app.route('/api/history', methods=['GET'])
# def get_prediction_history():
#     history = PredictionHistory.query.order_by(PredictionHistory.created_at.desc()).limit(10).all()
    
#     result = []
#     for record in history:
#         result.append({
#             'id': record.id,
#             'predicted_mpg': record.predicted_mpg,
#             'environmental_impact': record.environmental_impact,
#             'created_at': record.created_at.isoformat()
#         })
    
#     return jsonify(result)

# @app.route('/api/stats', methods=['GET'])
# def get_statistics():
#     total_predictions = PredictionHistory.query.count()
#     avg_mpg = db.session.query(db.func.avg(PredictionHistory.predicted_mpg)).scalar() or 0
#     total_co2_saved = db.session.query(db.func.sum(PredictionHistory.environmental_impact)).scalar() or 0
    
#     return jsonify({
#         'total_predictions': total_predictions,
#         'average_predicted_mpg': round(float(avg_mpg), 2),
#         'total_co2_impact_tons': round(float(total_co2_saved), 2),
#         'trees_equivalent': round(float(total_co2_saved) * 16, 0)
#     })

# @app.route('/', methods=['GET'])
# def index():
#     return jsonify({
#         'message': 'GreenAI Fuel Efficiency Predictor API',
#         'status': 'running',
#         'version': '1.0.0',
#         'endpoints': {
#             'predict': '/api/predict',
#             'vehicles': '/api/vehicles/search',
#             'stats': '/api/stats',
#             'history': '/api/history',
#             'health': '/api/health'
#         }
#     })

# @app.route('/api/health', methods=['GET'])
# def health_check():
#     return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

# @app.route('/api/vehicles/makes', methods=['GET'])
# def get_vehicle_makes():
#     try:
#         makes = db.session.query(Vehicle.make).distinct().order_by(Vehicle.make).all()
#         result = [{'value': make[0], 'label': make[0]} for make in makes]
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/vehicles/models/<make>', methods=['GET'])
# def get_models_for_make(make):
#     try:
#         models = db.session.query(Vehicle.model).filter(
#             Vehicle.make == make
#         ).distinct().order_by(Vehicle.model).all()
        
#         result = [{'value': model[0], 'label': model[0]} for model in models]
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/vehicles/engine_sizes/<make>/<model>', methods=['GET'])
# def get_engine_sizes_for_model(make, model):
#     try:
#         engine_sizes = db.session.query(Vehicle.engine_size).filter(
#             Vehicle.make == make,
#             Vehicle.model == model
#         ).distinct().order_by(Vehicle.engine_size).all()
        
#         result = []
#         for size in engine_sizes:
#             size_value = size[0]
#             if size_value == 0:
#                 label = "Electric (0L)"
#             else:
#                 label = f"{size_value}L"
#             result.append({'value': size_value, 'label': label})
        
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/vehicles/years/<make>/<model>/<float:engine_size>', methods=['GET'])
# def get_years_for_vehicle(make, model, engine_size):
#     try:
#         years = db.session.query(Vehicle.year).filter(
#             Vehicle.make == make,
#             Vehicle.model == model,
#             Vehicle.engine_size == engine_size
#         ).distinct().order_by(Vehicle.year.desc()).all()
        
#         result = [{'value': year[0], 'label': str(year[0])} for year in years]
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/vehicles/validate', methods=['POST'])
# def validate_vehicle_configuration():
#     try:
#         data = request.json
#         make = data.get('make')
#         model = data.get('model')
#         engine_size = data.get('engine_size')
#         year = data.get('year')
        
#         vehicle = Vehicle.query.filter_by(
#             make=make,
#             model=model,
#             engine_size=engine_size,
#             year=year
#         ).first()
        
#         if vehicle:
#             return jsonify({
#                 'valid': True,
#                 'vehicle': {
#                     'make': vehicle.make,
#                     'model': vehicle.model,
#                     'year': vehicle.year,
#                     'engine_size': vehicle.engine_size,
#                     'cylinders': vehicle.cylinders,
#                     'transmission': vehicle.transmission,
#                     'fuel_type': vehicle.fuel_type,
#                     'combined_mpg': vehicle.combined_mpg
#                 }
#             })
#         else:
#             closest = Vehicle.query.filter_by(
#                 make=make,
#                 model=model,
#                 engine_size=engine_size
#             ).first()
            
#             if closest:
#                 return jsonify({
#                     'valid': True,
#                     'vehicle': {
#                         'make': closest.make,
#                         'model': closest.model,
#                         'year': closest.year,
#                         'engine_size': closest.engine_size,
#                         'cylinders': closest.cylinders,
#                         'transmission': closest.transmission,
#                         'fuel_type': closest.fuel_type,
#                         'combined_mpg': closest.combined_mpg
#                     },
#                     'note': 'Using closest available year'
#                 })
#             else:
#                 return jsonify({'valid': False, 'error': 'Vehicle configuration not found'})
        
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/vehicles/all', methods=['GET'])
# def get_all_vehicles():
#     try:
#         vehicles = Vehicle.query.all()
#         result = []
        
#         for vehicle in vehicles:
#             result.append({
#                 'id': vehicle.id,
#                 'make': vehicle.make,
#                 'model': vehicle.model,
#                 'year': vehicle.year,
#                 'engine_size': vehicle.engine_size,
#                 'cylinders': vehicle.cylinders,
#                 'transmission': vehicle.transmission,
#                 'fuel_type': vehicle.fuel_type,
#                 'combined_mpg': vehicle.combined_mpg,
#                 'city_mpg': vehicle.city_mpg,
#                 'highway_mpg': vehicle.highway_mpg,
#                 'co2_emissions': vehicle.co2_emissions
#             })
        
#         return jsonify(result)
        
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/api/vehicles/years_for_model/<make>/<model>', methods=['GET'])
# def get_years_for_model_only(make, model):
#     try:
#         years = db.session.query(Vehicle.year).filter(
#             Vehicle.make == make,
#             Vehicle.model == model
#         ).distinct().order_by(Vehicle.year.desc()).all()
        
#         result = [{'value': year[0], 'label': str(year[0])} for year in years]
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# # Initialize database on startup
# print("\n" + "="*60)
# print("🚀 Starting Fuel Efficiency Predictor API")
# print("="*60)
# initialize_database()
# load_ml_models()
# print("="*60 + "\n")

# if __name__ == '__main__':
#     port = int(os.environ.get('PORT', 5000))
#     app.run(debug=app.config['DEBUG'], port=port, host='0.0.0.0')


#################################################


from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import json
import sys

# Platform-specific imports
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    HAS_FCNTL = False
    print("⚠️ fcntl not available (Windows system) - file locking disabled")

app = Flask(__name__)

# Production configuration
if os.environ.get('FLASK_ENV') == 'production' or os.environ.get('DATABASE_URL'):
    database_url = os.environ.get('DATABASE_URL', '')
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    app.config['DEBUG'] = False
    print(f"✅ Using PostgreSQL database")
else:
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fuel_efficiency.db'
    app.config['DEBUG'] = True
    print(f"✅ Using SQLite database")

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
}

# CRITICAL: Database Configuration
database_url = os.environ.get('DATABASE_URL')

if not database_url:
    print("❌ WARNING: DATABASE_URL not found!")
    print("🔄 Using SQLite as fallback (not recommended for production)")
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fuel_efficiency.db'
else:
    print(f"✅ DATABASE_URL found: {database_url[:40]}...")
    
    # Render uses postgres:// but SQLAlchemy needs postgresql://
    if database_url.startswith('postgres://'):
        database_url = database_url.replace('postgres://', 'postgresql://', 1)
        print("✓ Converted postgres:// to postgresql://")
    
    app.config['SQLALCHEMY_DATABASE_URI'] = database_url
    print("✅ PostgreSQL configured")

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
}

# CRITICAL: CORS Configuration for Frontend
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:3000",
            "http://localhost:3001", 
            "https://fuel-efficiency-predictor-3.onrender.com",  # REPLACE WITH YOUR FRONTEND URL
            "https://*.onrender.com"
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})

db = SQLAlchemy(app)
# Vehicle Model
class Vehicle(db.Model):
    __tablename__ = 'vehicle'
    id = db.Column(db.Integer, primary_key=True)
    make = db.Column(db.String(100), nullable=False)
    model = db.Column(db.String(100), nullable=False)
    year = db.Column(db.Integer, nullable=False)
    engine_size = db.Column(db.Float, nullable=False)
    cylinders = db.Column(db.Integer, nullable=False)
    transmission = db.Column(db.String(50), nullable=False)
    fuel_type = db.Column(db.String(50), nullable=False)
    city_mpg = db.Column(db.Float, nullable=False)
    highway_mpg = db.Column(db.Float, nullable=False)
    combined_mpg = db.Column(db.Float, nullable=False)
    co2_emissions = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)

# Prediction History Model
class PredictionHistory(db.Model):
    __tablename__ = 'prediction_history'
    id = db.Column(db.Integer, primary_key=True)
    user_input = db.Column(db.Text, nullable=False)
    predicted_mpg = db.Column(db.Float, nullable=False)
    environmental_impact = db.Column(db.Float, nullable=False)
    recommendations = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)

# Initialize ML models
ml_model = None
scaler = None
deep_learning_model = None

def safe_db_operation(operation, default_return=None):
    """Safely execute database operations with error handling"""
    try:
        return operation()
    except Exception as e:
        print(f"⚠️ Database operation failed: {e}")
        return default_return

def initialize_database():
    """Initialize database tables and check data"""
    try:
        with app.app_context():
            # Create all tables
            db.create_all()
            print("✅ Database tables created/verified")
            
            # Safely check if we have data
            vehicle_count = safe_db_operation(lambda: Vehicle.query.count(), 0)
            
            if vehicle_count == 0:
                print("⚠️ No vehicles in database. Adding minimal sample data...")
                add_minimal_sample_data()
                vehicle_count = safe_db_operation(lambda: Vehicle.query.count(), 0)
            
            print(f"✅ Database ready with {vehicle_count} vehicles")
            return True
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        import traceback
        traceback.print_exc()
        return False

def add_minimal_sample_data():
    """Add minimal sample data if database is empty"""
    try:
        sample_vehicles = [
            Vehicle(make='Toyota', model='Prius', year=2023, engine_size=1.8, cylinders=4, 
                   transmission='CVT', fuel_type='Hybrid', city_mpg=58, highway_mpg=53, 
                   combined_mpg=56, co2_emissions=96),
            Vehicle(make='Honda', model='Civic', year=2023, engine_size=2.0, cylinders=4, 
                   transmission='Manual', fuel_type='Gasoline', city_mpg=32, highway_mpg=42, 
                   combined_mpg=36, co2_emissions=262),
            Vehicle(make='Ford', model='F-150', year=2023, engine_size=3.3, cylinders=6, 
                   transmission='Automatic', fuel_type='Gasoline', city_mpg=19, highway_mpg=24, 
                   combined_mpg=21, co2_emissions=555),
            Vehicle(make='Tesla', model='Model 3', year=2023, engine_size=0.0, cylinders=0, 
                   transmission='Automatic', fuel_type='Electric', city_mpg=142, highway_mpg=132, 
                   combined_mpg=137, co2_emissions=0),
            Vehicle(make='Chevrolet', model='Malibu', year=2022, engine_size=1.5, cylinders=4, 
                   transmission='Automatic', fuel_type='Gasoline', city_mpg=29, highway_mpg=36, 
                   combined_mpg=32, co2_emissions=277),
        ]
        
        for vehicle in sample_vehicles:
            db.session.add(vehicle)
        
        db.session.commit()
        print(f"✅ Added {len(sample_vehicles)} sample vehicles")
    except Exception as e:
        print(f"❌ Error adding sample data: {e}")
        db.session.rollback()

def load_ml_models():
    global ml_model, scaler, deep_learning_model
    try:
        model_files = [
            'ml_models/fuel_efficiency_model.pkl',
            'ml_models/scaler.pkl',
            'ml_models/deep_learning_model.h5'
        ]
        
        missing_files = [f for f in model_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"⚠️ Missing model files: {missing_files}")
            print("🔄 Training new models...")
            train_models()
        
        ml_model = joblib.load('ml_models/fuel_efficiency_model.pkl')
        scaler = joblib.load('ml_models/scaler.pkl')
        deep_learning_model = tf.keras.models.load_model('ml_models/deep_learning_model.h5')
        print("✅ All models loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("🔄 Training new models...")
        train_models()

def train_models():
    global ml_model, scaler, deep_learning_model
    
    data = generate_sample_data()
    features = ['engine_size', 'cylinders', 'year', 'transmission_encoded', 'fuel_type_encoded']
    X = data[features]
    y = data['combined_mpg']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    ml_model = RandomForestRegressor(n_estimators=100, random_state=42)
    ml_model.fit(X_scaled, y)
    
    deep_learning_model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(len(features),)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    deep_learning_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    deep_learning_model.fit(X_scaled, y, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    
    os.makedirs('ml_models', exist_ok=True)
    joblib.dump(ml_model, 'ml_models/fuel_efficiency_model.pkl')
    joblib.dump(scaler, 'ml_models/scaler.pkl')
    deep_learning_model.save('ml_models/deep_learning_model.h5')
    print("✅ Models trained and saved successfully!")

def generate_sample_data():
    np.random.seed(42)
    n_samples = 1000
    data = {
        'engine_size': np.random.normal(2.5, 0.8, n_samples),
        'cylinders': np.random.choice([4, 6, 8], n_samples),
        'year': np.random.randint(2015, 2025, n_samples),
        'transmission_encoded': np.random.choice([0, 1], n_samples),
        'fuel_type_encoded': np.random.choice([0, 1, 2], n_samples),
    }
    mpg = (30 - data['engine_size'] * 3 + 
           (data['cylinders'] == 4) * 5 - 
           (data['cylinders'] == 8) * 8 +
           (data['year'] - 2015) * 0.5 +
           data['transmission_encoded'] * -2 +
           data['fuel_type_encoded'] * 3 +
           np.random.normal(0, 3, n_samples))
    data['combined_mpg'] = np.clip(mpg, 15, 50)
    return pd.DataFrame(data)

def calculate_environmental_impact(mpg, annual_miles=12000):
    if mpg <= 0:
        mpg = 1
    gallons_per_year = annual_miles / mpg
    co2_per_gallon = 19.6
    annual_co2 = gallons_per_year * co2_per_gallon
    co2_tons = annual_co2 / 2000
    return {
        'annual_gallons': round(float(gallons_per_year), 2),
        'annual_co2_pounds': round(float(annual_co2), 2),
        'annual_co2_tons': round(float(co2_tons), 2),
        'trees_to_offset': int(round(co2_tons * 16, 0))
    }

def get_green_recommendations(mpg, vehicle_data):
    recommendations = []
    if mpg < 25:
        recommendations.extend([
            "Consider hybrid or electric alternatives",
            "Regular maintenance can improve fuel efficiency by 4-10%",
            "Proper tire inflation can improve MPG by up to 3%"
        ])
    if mpg < 30:
        recommendations.extend([
            "Avoid aggressive driving - can improve MPG by 15-30%",
            "Remove excess weight from your vehicle",
            "Use cruise control on highways"
        ])
    recommendations.extend([
        "Consider carpooling or public transportation",
        "Plan and combine trips to reduce overall driving",
        "Keep your vehicle well-maintained"
    ])
    return [str(rec) for rec in recommendations]

@app.route('/api/predict', methods=['POST'])
def predict_fuel_efficiency():
    try:
        if ml_model is None or scaler is None or deep_learning_model is None:
            load_ml_models()
            if ml_model is None:
                return jsonify({'error': 'ML models not available'}), 503
        
        data = request.json
        features = {
            'engine_size': float(data.get('engine_size', 2.5)),
            'cylinders': int(data.get('cylinders', 4)),
            'year': int(data.get('year', 2020)),
            'transmission_encoded': 1 if data.get('transmission', 'automatic').lower() == 'automatic' else 0,
            'fuel_type_encoded': {'gasoline': 0, 'diesel': 1, 'hybrid': 2, 'electric': 3}.get(
                data.get('fuel_type', 'gasoline').lower(), 0)
        }
        
        feature_array = np.array([[features['engine_size'], features['cylinders'], 
                                 features['year'], features['transmission_encoded'], 
                                 features['fuel_type_encoded']]])
        
        feature_scaled = scaler.transform(feature_array)
        ml_prediction = float(ml_model.predict(feature_scaled)[0])
        dl_prediction = float(deep_learning_model.predict(feature_scaled, verbose=0)[0][0])
        final_prediction = (ml_prediction + dl_prediction) / 2
        final_prediction = max(15, min(50, final_prediction))
        
        env_impact = calculate_environmental_impact(final_prediction)
        recommendations = get_green_recommendations(final_prediction, data)
        
        try:
            prediction_record = PredictionHistory(
                user_input=json.dumps(data),
                predicted_mpg=final_prediction,
                environmental_impact=env_impact['annual_co2_tons'],
                recommendations=json.dumps(recommendations)
            )
            db.session.add(prediction_record)
            db.session.commit()
        except:
            pass
        
        response = {
            'predicted_mpg': round(final_prediction, 2),
            'ml_prediction': round(ml_prediction, 2),
            'dl_prediction': round(dl_prediction, 2),
            'environmental_impact': env_impact,
            'recommendations': recommendations,
            'green_score': round(min(100, max(0, (final_prediction - 15) * 100 / 35)), 2),
            'comparison': {
                'average_vehicle_mpg': 25.4,
                'better_than_average': bool(final_prediction > 25.4)
            }
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/vehicles/search', methods=['GET'])
def search_vehicles():
    make = request.args.get('make', '')
    model = request.args.get('model', '')
    query = Vehicle.query
    if make:
        query = query.filter(Vehicle.make.ilike(f'%{make}%'))
    if model:
        query = query.filter(Vehicle.model.ilike(f'%{model}%'))
    vehicles = query.limit(20).all()
    return jsonify([{
        'id': v.id, 'make': v.make, 'model': v.model, 'year': v.year,
        'engine_size': v.engine_size, 'cylinders': v.cylinders,
        'transmission': v.transmission, 'fuel_type': v.fuel_type,
        'combined_mpg': v.combined_mpg, 'co2_emissions': v.co2_emissions
    } for v in vehicles])

@app.route('/api/history', methods=['GET'])
def get_prediction_history():
    history = safe_db_operation(
        lambda: PredictionHistory.query.order_by(PredictionHistory.created_at.desc()).limit(10).all(),
        []
    )
    return jsonify([{
        'id': r.id, 'predicted_mpg': r.predicted_mpg,
        'environmental_impact': r.environmental_impact,
        'created_at': r.created_at.isoformat()
    } for r in history])

@app.route('/api/stats', methods=['GET'])
def get_statistics():
    total_predictions = safe_db_operation(lambda: PredictionHistory.query.count(), 0)
    avg_mpg = safe_db_operation(lambda: db.session.query(db.func.avg(PredictionHistory.predicted_mpg)).scalar(), 0) or 0
    total_co2_saved = safe_db_operation(lambda: db.session.query(db.func.sum(PredictionHistory.environmental_impact)).scalar(), 0) or 0
    
    return jsonify({
        'total_predictions': total_predictions,
        'average_predicted_mpg': round(float(avg_mpg), 2),
        'total_co2_impact_tons': round(float(total_co2_saved), 2),
        'trees_equivalent': round(float(total_co2_saved) * 16, 0)
    })

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'message': 'GreenAI Fuel Efficiency Predictor API',
        'status': 'running',
        'version': '1.0.0',
        'endpoints': {
            'predict': '/api/predict',
            'vehicles': '/api/vehicles/search',
            'stats': '/api/stats',
            'history': '/api/history',
            'health': '/api/health'
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    vehicle_count = safe_db_operation(lambda: Vehicle.query.count(), 0)
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'database': 'connected',
        'vehicles_in_db': vehicle_count,
        'models_loaded': ml_model is not None
    })

@app.route('/api/vehicles/makes', methods=['GET'])
def get_vehicle_makes():
    try:
        makes = safe_db_operation(
            lambda: db.session.query(Vehicle.make).distinct().order_by(Vehicle.make).all(),
            []
        )
        return jsonify([{'value': make[0], 'label': make[0]} for make in makes])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/vehicles/models/<make>', methods=['GET'])
def get_models_for_make(make):
    try:
        models = safe_db_operation(
            lambda: db.session.query(Vehicle.model).filter(Vehicle.make == make).distinct().order_by(Vehicle.model).all(),
            []
        )
        return jsonify([{'value': model[0], 'label': model[0]} for model in models])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/vehicles/engine_sizes/<make>/<model>', methods=['GET'])
def get_engine_sizes_for_model(make, model):
    try:
        engine_sizes = safe_db_operation(
            lambda: db.session.query(Vehicle.engine_size).filter(
                Vehicle.make == make, Vehicle.model == model
            ).distinct().order_by(Vehicle.engine_size).all(),
            []
        )
        result = []
        for size in engine_sizes:
            size_value = size[0]
            label = "Electric (0L)" if size_value == 0 else f"{size_value}L"
            result.append({'value': size_value, 'label': label})
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/vehicles/years/<make>/<model>/<float:engine_size>', methods=['GET'])
def get_years_for_vehicle(make, model, engine_size):
    try:
        years = safe_db_operation(
            lambda: db.session.query(Vehicle.year).filter(
                Vehicle.make == make, Vehicle.model == model, Vehicle.engine_size == engine_size
            ).distinct().order_by(Vehicle.year.desc()).all(),
            []
        )
        return jsonify([{'value': year[0], 'label': str(year[0])} for year in years])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/vehicles/validate', methods=['POST'])
def validate_vehicle_configuration():
    try:
        data = request.json
        vehicle = safe_db_operation(
            lambda: Vehicle.query.filter_by(
                make=data.get('make'),
                model=data.get('model'),
                engine_size=data.get('engine_size'),
                year=data.get('year')
            ).first(),
            None
        )
        
        if vehicle:
            return jsonify({
                'valid': True,
                'vehicle': {
                    'make': vehicle.make, 'model': vehicle.model, 'year': vehicle.year,
                    'engine_size': vehicle.engine_size, 'cylinders': vehicle.cylinders,
                    'transmission': vehicle.transmission, 'fuel_type': vehicle.fuel_type,
                    'combined_mpg': vehicle.combined_mpg
                }
            })
        return jsonify({'valid': False, 'error': 'Vehicle configuration not found'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/vehicles/all', methods=['GET'])
def get_all_vehicles():
    try:
        vehicles = safe_db_operation(lambda: Vehicle.query.limit(1000).all(), [])
        return jsonify([{
            'id': v.id, 'make': v.make, 'model': v.model, 'year': v.year,
            'engine_size': v.engine_size, 'cylinders': v.cylinders,
            'transmission': v.transmission, 'fuel_type': v.fuel_type,
            'combined_mpg': v.combined_mpg, 'city_mpg': v.city_mpg,
            'highway_mpg': v.highway_mpg, 'co2_emissions': v.co2_emissions
        } for v in vehicles])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/vehicles/years_for_model/<make>/<model>', methods=['GET'])
def get_years_for_model_only(make, model):
    try:
        years = safe_db_operation(
            lambda: db.session.query(Vehicle.year).filter(
                Vehicle.make == make, Vehicle.model == model
            ).distinct().order_by(Vehicle.year.desc()).all(),
            []
        )
        return jsonify([{'value': year[0], 'label': str(year[0])} for year in years])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Initialize on startup
print("\n" + "="*60)
print("🚀 Starting Fuel Efficiency Predictor API")
print("="*60)
initialize_database()
load_ml_models()
print("="*60 + "\n")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=app.config['DEBUG'], port=port, host='0.0.0.0')