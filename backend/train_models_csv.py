import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import GridSearchCV
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from app import app, Vehicle
from datetime import datetime
import warnings
import os
warnings.filterwarnings('ignore')

class CSVModelTrainer:
    """Train ML models using CSV vehicle data"""
    
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.encoders = {}
        self.feature_names = []
        self.training_data = None
        self.performance_metrics = {}
        
    def load_data_from_database(self):
        """Load vehicle data from database"""
        print("📊 Loading vehicle data from database...")
        
        with app.app_context():
            vehicles = Vehicle.query.all()
            
            if not vehicles:
                print("❌ No vehicles found in database")
                return False
            
            # Convert to DataFrame
            data = []
            for vehicle in vehicles:
                data.append({
                    'make': vehicle.make,
                    'model': vehicle.model,
                    'year': vehicle.year,
                    'engine_size': vehicle.engine_size,
                    'cylinders': vehicle.cylinders,
                    'transmission': vehicle.transmission,
                    'fuel_type': vehicle.fuel_type,
                    'city_mpg': vehicle.city_mpg,
                    'highway_mpg': vehicle.highway_mpg,
                    'combined_mpg': vehicle.combined_mpg,
                    'co2_emissions': vehicle.co2_emissions
                })
            
            self.training_data = pd.DataFrame(data)
            print(f"✅ Loaded {len(self.training_data):,} vehicles")
            
            # Display basic statistics
            self._display_data_overview()
            
            return True
    
    def _display_data_overview(self):
        """Display overview of training data"""
        print("\n📋 Training Data Overview:")
        print(f"  Total vehicles: {len(self.training_data):,}")
        print(f"  Year range: {self.training_data['year'].min()} - {self.training_data['year'].max()}")
        print(f"  MPG range: {self.training_data['combined_mpg'].min():.1f} - {self.training_data['combined_mpg'].max():.1f}")
        print(f"  Average MPG: {self.training_data['combined_mpg'].mean():.1f}")
        
        print(f"\n🏭 Manufacturers ({self.training_data['make'].nunique()}):")
        top_makes = self.training_data['make'].value_counts().head(10)
        for make, count in top_makes.items():
            print(f"  {make}: {count:,}")
        
        print(f"\n⛽ Fuel Types:")
        fuel_counts = self.training_data['fuel_type'].value_counts()
        for fuel, count in fuel_counts.items():
            print(f"  {fuel}: {count:,}")
    
    def prepare_features(self):
        """Prepare features for training"""
        print("\n🔧 Preparing features...")
        
        df = self.training_data.copy()
        
        # Encode categorical variables
        categorical_columns = ['make', 'transmission', 'fuel_type']
        
        for col in categorical_columns:
            if col not in self.encoders:
                self.encoders[col] = LabelEncoder()
            
            df[f'{col}_encoded'] = self.encoders[col].fit_transform(df[col])
        
        # Create additional features
        df['age'] = 2024 - df['year']  # Vehicle age
        df['power_to_weight'] = df['engine_size'] / (df['cylinders'] * 0.5)  # Rough power-to-weight
        df['is_hybrid'] = (df['fuel_type'] == 'Hybrid').astype(int)
        df['is_electric'] = (df['fuel_type'] == 'Electric').astype(int)
        df['is_premium_brand'] = df['make'].isin(['BMW', 'Mercedes-Benz', 'Audi', 'Lexus']).astype(int)
        
        # Select features
        self.feature_names = [
            'year', 'engine_size', 'cylinders', 'age',
            'make_encoded', 'transmission_encoded', 'fuel_type_encoded',
            'power_to_weight', 'is_hybrid', 'is_electric', 'is_premium_brand'
        ]
        
        X = df[self.feature_names].copy()
        y = df['combined_mpg'].copy()
        
        print(f"✅ Features prepared: {len(self.feature_names)} features, {len(X)} samples")
        
        return X, y
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train optimized Random Forest model"""
        print("\n🌲 Training Random Forest...")
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [200, 300],
            'max_depth': [15, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
        
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        rf_grid = GridSearchCV(
            rf, param_grid, 
            cv=5, 
            scoring='neg_mean_squared_error', 
            n_jobs=-1,
            verbose=1
        )
        
        rf_grid.fit(X_train, y_train)
        
        best_rf = rf_grid.best_estimator_
        self.models['random_forest'] = best_rf
        
        # Evaluate
        val_pred = best_rf.predict(X_val)
        val_score = r2_score(y_val, val_pred)
        
        print(f"  ✅ Best parameters: {rf_grid.best_params_}")
        print(f"  📊 Validation R²: {val_score:.4f}")
        
        return best_rf
    
    def train_gradient_boosting(self, X_train, y_train, X_val, y_val):
        """Train Gradient Boosting model"""
        print("\n🚀 Training Gradient Boosting...")
        
        gb = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        )
        
        gb.fit(X_train, y_train)
        self.models['gradient_boosting'] = gb
        
        # Evaluate
        val_pred = gb.predict(X_val)
        val_score = r2_score(y_val, val_pred)
        
        print(f"  📊 Validation R²: {val_score:.4f}")
        
        return gb
    
    def train_extra_trees(self, X_train, y_train, X_val, y_val):
        """Train Extra Trees model"""
        print("\n🌳 Training Extra Trees...")
        
        et = ExtraTreesRegressor(
            n_estimators=200,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        
        et.fit(X_train, y_train)
        self.models['extra_trees'] = et
        
        # Evaluate
        val_pred = et.predict(X_val)
        val_score = r2_score(y_val, val_pred)
        
        print(f"  📊 Validation R²: {val_score:.4f}")
        
        return et
    
    def train_deep_learning(self, X_train, y_train, X_val, y_val):
        """Train deep learning model"""
        print("\n🧠 Training Deep Learning Model...")
        
        # Build model
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(len(self.feature_names),)),
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
        
        # Compile
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=8),
            tf.keras.callbacks.ModelCheckpoint('best_model.h5', save_best_only=True)
        ]
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        self.models['deep_learning'] = model
        
        # Evaluate
        val_pred = model.predict(X_val).flatten()
        val_score = r2_score(y_val, val_pred)
        
        print(f"  📊 Validation R²: {val_score:.4f}")
        
        return model, history
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all trained models"""
        print("\n📊 Evaluating All Models:")
        print("=" * 50)
        
        for name, model in self.models.items():
            print(f"\n🔍 {name.replace('_', ' ').title()}:")
            
            # Make predictions
            if name == 'deep_learning':
                y_pred = model.predict(X_test).flatten()
            else:
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            
            # Store metrics
            self.performance_metrics[name] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'MAPE': mape
            }
            
            print(f"  RMSE: {rmse:.3f}")
            print(f"  MAE: {mae:.3f}")
            print(f"  R²: {r2:.4f}")
            print(f"  MAPE: {mape:.3f}")
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(f"  Top 3 Features:")
                for i, (_, row) in enumerate(importance_df.head(3).iterrows()):
                    print(f"    {i+1}. {row['feature']}: {row['importance']:.3f}")
    
    def create_ensemble_model(self, X_test, y_test):
        """Create weighted ensemble model"""
        print("\n🎭 Creating Ensemble Model...")
        
        if len(self.models) < 2:
            print("❌ Need at least 2 models for ensemble")
            return None
        
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            if name == 'deep_learning':
                predictions[name] = model.predict(X_test).flatten()
            else:
                predictions[name] = model.predict(X_test)
        
        # Calculate weights based on R² scores (higher R² = higher weight)
        weights = {}
        total_r2 = sum([metrics['R2'] for metrics in self.performance_metrics.values()])
        
        for name in self.models.keys():
            if total_r2 > 0:
                weights[name] = self.performance_metrics[name]['R2'] / total_r2
            else:
                weights[name] = 1 / len(self.models)
        
        print("  Model weights:")
        for name, weight in weights.items():
            print(f"    {name}: {weight:.3f}")
        
        # Create ensemble prediction
        ensemble_pred = np.zeros(len(X_test))
        for name, pred in predictions.items():
            ensemble_pred += weights[name] * pred
        
        # Evaluate ensemble
        ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        
        print(f"  📊 Ensemble Performance:")
        print(f"    RMSE: {ensemble_rmse:.3f}")
        print(f"    MAE: {ensemble_mae:.3f}")
        print(f"    R²: {ensemble_r2:.4f}")
        
        # Save ensemble weights
        self.models['ensemble_weights'] = weights
        
        return weights, ensemble_pred
    
    def save_models(self):
        """Save all models and preprocessing objects"""
        print("\n💾 Saving Models...")
        
        # Create models directory
        os.makedirs('ml_models', exist_ok=True)
        
        # Save scikit-learn models
        for name, model in self.models.items():
            if name not in ['deep_learning', 'ensemble_weights']:
                joblib.dump(model, f'ml_models/{name}_model.pkl')
                print(f"  ✅ Saved {name}")
        
        # Save deep learning model
        if 'deep_learning' in self.models:
            self.models['deep_learning'].save('ml_models/deep_learning_model.h5')
            print(f"  ✅ Saved deep_learning")
        
        # Save preprocessing objects
        joblib.dump(self.scaler, 'ml_models/scaler.pkl')
        joblib.dump(self.encoders, 'ml_models/encoders.pkl')
        joblib.dump(self.feature_names, 'ml_models/feature_names.pkl')
        
        # Save ensemble weights if available
        if 'ensemble_weights' in self.models:
            joblib.dump(self.models['ensemble_weights'], 'ml_models/ensemble_weights.pkl')
        
        # Save performance metrics
        joblib.dump(self.performance_metrics, 'ml_models/performance_metrics.pkl')
        
        print("  ✅ Saved preprocessing objects and metrics")
    
    def plot_results(self, X_test, y_test):
        """Create visualizations of model performance"""
        print("\n📊 Creating Performance Visualizations...")
        
        try:
            import matplotlib.pyplot as plt
            
            # Set up the plot
            n_models = len([name for name in self.models.keys() if name != 'ensemble_weights'])
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Model Performance Analysis', fontsize=16)
            
            # 1. R² Score Comparison
            model_names = []
            r2_scores = []
            
            for name, metrics in self.performance_metrics.items():
                model_names.append(name.replace('_', ' ').title())
                r2_scores.append(metrics['R2'])
            
            axes[0, 0].bar(model_names, r2_scores, color='skyblue', alpha=0.7)
            axes[0, 0].set_title('R² Score Comparison')
            axes[0, 0].set_ylabel('R² Score')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. RMSE Comparison
            rmse_scores = [metrics['RMSE'] for metrics in self.performance_metrics.values()]
            axes[0, 1].bar(model_names, rmse_scores, color='lightcoral', alpha=0.7)
            axes[0, 1].set_title('RMSE Comparison')
            axes[0, 1].set_ylabel('RMSE')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # 3. Prediction vs Actual (Best model)
            best_model_name = max(self.performance_metrics.items(), key=lambda x: x[1]['R2'])[0]
            best_model = self.models[best_model_name]
            
            if best_model_name == 'deep_learning':
                y_pred_best = best_model.predict(X_test).flatten()
            else:
                y_pred_best = best_model.predict(X_test)
            
            axes[1, 0].scatter(y_test, y_pred_best, alpha=0.6, color='green')
            axes[1, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[1, 0].set_xlabel('Actual MPG')
            axes[1, 0].set_ylabel('Predicted MPG')
            axes[1, 0].set_title(f'Prediction vs Actual ({best_model_name.replace("_", " ").title()})')
            
            # 4. Feature Importance (if available)
            if hasattr(best_model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': best_model.feature_importances_
                }).sort_values('importance', ascending=True)
                
                axes[1, 1].barh(importance_df['feature'], importance_df['importance'])
                axes[1, 1].set_title(f'Feature Importance ({best_model_name.replace("_", " ").title()})')
                axes[1, 1].set_xlabel('Importance')
            else:
                axes[1, 1].text(0.5, 0.5, 'Feature importance\nnot available for\nthis model type', 
                               ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)
                axes[1, 1].set_title('Feature Importance')
            
            plt.tight_layout()
            plt.savefig('ml_models/model_performance.png', dpi=300, bbox_inches='tight')
            print("  ✅ Saved performance plots to ml_models/model_performance.png")
            
        except Exception as e:
            print(f"  ⚠️ Could not create plots: {e}")
    
    def generate_training_report(self):
        """Generate comprehensive training report"""
        report = {
            'training_date': datetime.now().isoformat(),
            'dataset_info': {
                'total_samples': len(self.training_data),
                'features_used': self.feature_names,
                'target_variable': 'combined_mpg',
                'year_range': [int(self.training_data['year'].min()), int(self.training_data['year'].max())],
                'mpg_range': [float(self.training_data['combined_mpg'].min()), float(self.training_data['combined_mpg'].max())],
                'unique_makes': int(self.training_data['make'].nunique()),
                'unique_models': int(self.training_data['model'].nunique())
            },
            'model_performance': self.performance_metrics,
            'best_model': max(self.performance_metrics.items(), key=lambda x: x[1]['R2'])[0] if self.performance_metrics else None
        }
        
        # Save report
        with open('ml_models/training_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print("  ✅ Saved training report to ml_models/training_report.json")
        
        return report
    
    def train_all_models(self):
        """Complete training pipeline"""
        print("🚀 Starting Comprehensive Model Training")
        print("=" * 60)
        print(f"Training started at: {datetime.now()}")
        
        # Load data
        if not self.load_data_from_database():
            return False
        
        # Prepare features
        X, y = self.prepare_features()
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=42, stratify=None
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\n📊 Data Split:")
        print(f"  Training: {X_train.shape[0]:,} samples")
        print(f"  Validation: {X_val.shape[0]:,} samples")
        print(f"  Test: {X_test.shape[0]:,} samples")
        
        # Train all models
        try:
            self.train_random_forest(X_train_scaled, y_train, X_val_scaled, y_val)
            self.train_gradient_boosting(X_train_scaled, y_train, X_val_scaled, y_val)
            self.train_extra_trees(X_train_scaled, y_train, X_val_scaled, y_val)
            self.train_deep_learning(X_train_scaled, y_train, X_val_scaled, y_val)
        except Exception as e:
            print(f"⚠️ Some models failed to train: {e}")
        
        # Evaluate all models
        if self.models:
            self.evaluate_all_models(X_test_scaled, y_test)
            
            # Create ensemble
            ensemble_weights, ensemble_pred = self.create_ensemble_model(X_test_scaled, y_test)
            
            # Save everything
            self.save_models()
            
            # Create visualizations
            self.plot_results(X_test_scaled, y_test)
            
            # Generate report
            report = self.generate_training_report()
            
            print(f"\n🎉 Training Complete!")
            print(f"📊 Best model: {report['best_model']} (R² = {self.performance_metrics[report['best_model']]['R2']:.4f})")
            print(f"💾 Models saved to ml_models/ directory")
            
            return True
        else:
            print("❌ No models were successfully trained")
            return False

def train_models_with_csv():
    """Main function to train models with CSV data"""
    trainer = CSVModelTrainer()
    success = trainer.train_all_models()
    
    if success:
        print("\n✅ Model training successful!")
        print("\n📋 Next steps:")
        print("1. Start the backend: python app.py")
        print("2. Start the frontend: npm start (in frontend directory)")
        print("3. Test predictions with your vehicle data!")
    else:
        print("\n❌ Model training failed!")
        print("Make sure you have:")
        print("1. Processed your CSV data: python csv_data_processor.py")
        print("2. Verified database has vehicles: check with admin panel")

if __name__ == "__main__":
    import json
    
    print("🤖 CSV-Based Model Training")
    print("=" * 40)
    
    # Check if we have data
    with app.app_context():
        vehicle_count = Vehicle.query.count()
        
    if vehicle_count == 0:
        print("❌ No vehicles found in database!")
        print("\n📋 Please first process your CSV file:")
        print("python csv_data_processor.py")
        exit(1)
    
    print(f"📊 Found {vehicle_count:,} vehicles in database")
    
    # Confirm training
    response = input("\nProceed with model training? (y/n): ").lower().strip()
    if response == 'y':
        train_models_with_csv()
    else:
        print("Training cancelled.")