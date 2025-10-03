import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class GreenAIFuelEfficiencyPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_importance = {}
        self.model_performance = {}
        
    def load_and_prepare_data(self):
        """Load and prepare comprehensive fuel efficiency dataset"""
        np.random.seed(42)
        
        # Generate realistic synthetic dataset (replace with real data in production)
        n_samples = 5000
        
        # Vehicle makes and models with realistic distributions
        makes = ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'Nissan', 'BMW', 'Mercedes', 'Audi']
        models = ['Sedan', 'SUV', 'Hatchback', 'Coupe', 'Truck', 'Hybrid', 'Electric']
        
        data = {
            'make': np.random.choice(makes, n_samples),
            'model_type': np.random.choice(models, n_samples),
            'year': np.random.randint(2010, 2025, n_samples),
            'engine_size': np.random.normal(2.5, 0.8, n_samples),
            'cylinders': np.random.choice([4, 6, 8], n_samples, p=[0.6, 0.3, 0.1]),
            'transmission': np.random.choice(['Manual', 'Automatic', 'CVT'], n_samples, p=[0.2, 0.6, 0.2]),
            'fuel_type': np.random.choice(['Gasoline', 'Diesel', 'Hybrid', 'Electric'], 
                                        n_samples, p=[0.65, 0.15, 0.15, 0.05]),
            'weight': np.random.normal(3500, 800, n_samples),  # Vehicle weight in lbs
            'drivetrain': np.random.choice(['FWD', 'RWD', 'AWD'], n_samples, p=[0.5, 0.3, 0.2]),
            'turbo': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # Turbocharged or not
        }
        
        df = pd.DataFrame(data)
        
        # Ensure realistic ranges
        df['engine_size'] = np.clip(df['engine_size'], 1.0, 8.0)
        df['weight'] = np.clip(df['weight'], 2000, 7000)
        
        # Generate realistic MPG based on complex relationships
        base_mpg = 35  # Base efficiency
        
        # Engine size effect (larger engines = lower efficiency)
        mpg = base_mpg - (df['engine_size'] - 2.0) * 4
        
        # Cylinder effect
        mpg -= (df['cylinders'] - 4) * 2
        
        # Year effect (newer = more efficient)
        mpg += (df['year'] - 2010) * 0.3
        
        # Weight effect (heavier = less efficient)
        mpg -= (df['weight'] - 3500) / 200
        
        # Fuel type effect
        fuel_multipliers = {'Electric': 2.5, 'Hybrid': 1.6, 'Diesel': 1.1, 'Gasoline': 1.0}
        for fuel, multiplier in fuel_multipliers.items():
            mpg.loc[df['fuel_type'] == fuel] *= multiplier
        
        # Transmission effect
        transmission_effect = {'Manual': 1.05, 'CVT': 1.03, 'Automatic': 1.0}
        for trans, effect in transmission_effect.items():
            mpg.loc[df['transmission'] == trans] *= effect
        
        # Drivetrain effect
        drivetrain_effect = {'FWD': 1.0, 'RWD': 0.95, 'AWD': 0.88}
        for drive, effect in drivetrain_effect.items():
            mpg.loc[df['drivetrain'] == drive] *= effect
        
        # Turbo effect
        mpg.loc[df['turbo'] == 1] *= 0.92
        
        # Add some noise and ensure realistic bounds
        mpg += np.random.normal(0, 2, n_samples)
        mpg = np.clip(mpg, 8, 150)  # Realistic MPG range
        
        df['combined_mpg'] = mpg
        
        # Calculate CO2 emissions (inverse relationship with MPG)
        df['co2_emissions'] = (19.6 * 12000) / df['combined_mpg']  # grams per mile
        
        return df
    
    def encode_categorical_features(self, df):
        """Encode categorical variables"""
        categorical_columns = ['make', 'model_type', 'transmission', 'fuel_type', 'drivetrain']
        
        for col in categorical_columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col])
            else:
                df[f'{col}_encoded'] = self.encoders[col].transform(df[col])
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for training"""
        df = self.encode_categorical_features(df)
        
        feature_columns = [
            'year', 'engine_size', 'cylinders', 'weight', 'turbo',
            'make_encoded', 'model_type_encoded', 'transmission_encoded',
            'fuel_type_encoded', 'drivetrain_encoded'
        ]
        
        X = df[feature_columns].copy()
        y = df['combined_mpg'].copy()
        
        return X, y, feature_columns
    
    def train_random_forest(self, X_train, y_train):
        """Train optimized Random Forest model"""
        print("Training Random Forest...")
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        rf = RandomForestRegressor(random_state=42)
        rf_grid = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        rf_grid.fit(X_train, y_train)
        
        self.models['random_forest'] = rf_grid.best_estimator_
        return rf_grid.best_estimator_
    
    def train_gradient_boosting(self, X_train, y_train):
        """Train Gradient Boosting model"""
        print("Training Gradient Boosting...")
        
        gb = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        gb.fit(X_train, y_train)
        
        self.models['gradient_boosting'] = gb
        return gb
    
    def build_deep_learning_model(self, input_dim):
        """Build optimized deep learning model"""
        print("Building Deep Learning model...")
        
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            
            tf.keras.layers.Dense(1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_deep_learning(self, X_train, y_train, X_val, y_val):
        """Train deep learning model with callbacks"""
        model = self.build_deep_learning_model(X_train.shape[1])
        
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5),
        ]
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        
        self.models['deep_learning'] = model
        return model, history
    
    def evaluate_models(self, X_test, y_test, feature_names):
        """Evaluate all trained models"""
        print("\nEvaluating models...")
        
        for name, model in self.models.items():
            if name == 'deep_learning':
                predictions = model.predict(X_test).flatten()
            else:
                predictions = model.predict(X_test)
            
            mse = mean_squared_error(y_test, predictions)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            
            self.model_performance[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2
            }
            
            print(f"\n{name.replace('_', ' ').title()} Performance:")
            print(f"  RMSE: {rmse:.3f}")
            print(f"  MAE: {mae:.3f}")
            print(f"  R²: {r2:.3f}")
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                self.feature_importance[name] = importance
                print(f"  Top 3 Features: {', '.join(importance.head(3)['feature'].tolist())}")
    
    def save_models(self):
        """Save all trained models"""
        import os
        os.makedirs('ml_models', exist_ok=True)
        
        # Save scikit-learn models
        for name, model in self.models.items():
            if name != 'deep_learning':
                joblib.dump(model, f'ml_models/{name}_model.pkl')
        
        # Save deep learning model
        if 'deep_learning' in self.models:
            self.models['deep_learning'].save('ml_models/deep_learning_model.h5')
        
        # Save scaler and encoders
        joblib.dump(self.scaler, 'ml_models/scaler.pkl')
        joblib.dump(self.encoders, 'ml_models/encoders.pkl')
        
        # Save performance metrics
        joblib.dump(self.model_performance, 'ml_models/performance_metrics.pkl')
        joblib.dump(self.feature_importance, 'ml_models/feature_importance.pkl')
        
        print("\nAll models saved successfully!")
    
    def create_ensemble_model(self, X_test, y_test):
        """Create ensemble model combining all approaches"""
        print("\nCreating ensemble model...")
        
        predictions = {}
        for name, model in self.models.items():
            if name == 'deep_learning':
                predictions[name] = model.predict(X_test).flatten()
            else:
                predictions[name] = model.predict(X_test)
        
        # Weighted ensemble based on R² scores
        weights = {}
        total_r2 = sum([perf['R2'] for perf in self.model_performance.values()])
        
        for name, perf in self.model_performance.items():
            weights[name] = perf['R2'] / total_r2
        
        ensemble_pred = np.zeros(len(y_test))
        for name, pred in predictions.items():
            ensemble_pred += weights[name] * pred
        
        ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        
        print(f"Ensemble Model Performance:")
        print(f"  RMSE: {ensemble_rmse:.3f}")
        print(f"  R²: {ensemble_r2:.3f}")
        
        # Save ensemble weights
        joblib.dump(weights, 'ml_models/ensemble_weights.pkl')
        
        return weights
    
    def train_all_models(self):
        """Complete training pipeline"""
        print("🌱 Starting GreenAI Fuel Efficiency Model Training...")
        print(f"Training started at: {datetime.now()}")
        
        # Load and prepare data
        df = self.load_and_prepare_data()
        print(f"Dataset loaded: {len(df)} samples")
        
        X, y, feature_names = self.prepare_features(df)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        # Train all models
        self.train_random_forest(X_train_scaled, y_train)
        self.train_gradient_boosting(X_train_scaled, y_train)
        self.train_deep_learning(X_train_scaled, y_train, X_val_scaled, y_val)
        
        # Evaluate models
        self.evaluate_models(X_test_scaled, y_test, feature_names)
        
        # Create ensemble
        ensemble_weights = self.create_ensemble_model(X_test_scaled, y_test)
        
        # Save everything
        self.save_models()
        
        print("\n✅ Training completed successfully!")
        print("🌱 GreenAI models are ready for deployment!")
        
        return {
            'feature_names': feature_names,
            'performance': self.model_performance,
            'ensemble_weights': ensemble_weights
        }

if __name__ == "__main__":
    trainer = GreenAIFuelEfficiencyPredictor()
    results = trainer.train_all_models()