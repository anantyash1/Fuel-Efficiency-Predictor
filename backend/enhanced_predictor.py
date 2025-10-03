import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_percentage_error
import tensorflow as tf
import joblib
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class EnhancedGreenAIPredictor:
    """Advanced AI predictor with explainability and online learning"""
    
    def __init__(self):
        self.models = {}
        self.explainer = None
        self.performance_history = []
        self.model_weights = {}
        self.feature_names = []
        self.scaler = None
        
    def load_models(self):
        """Load all trained models"""
        try:
            self.models['rf'] = joblib.load('ml_models/random_forest_model.pkl')
            self.models['gb'] = joblib.load('ml_models/gradient_boosting_model.pkl')
            self.models['dl'] = tf.keras.models.load_model('ml_models/deep_learning_model.h5')
            self.scaler = joblib.load('ml_models/scaler.pkl')
            self.model_weights = joblib.load('ml_models/ensemble_weights.pkl')
            
            # Initialize SHAP explainer for Random Forest (fastest for real-time explanations)
            self.explainer = shap.TreeExplainer(self.models['rf'])
            
            print("✅ All models loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return False
    
    def create_advanced_ensemble(self, X_train, y_train):
        """Create advanced ensemble with stacking"""
        
        # Base models
        base_models = [
            ('rf', self.models['rf']),
            ('gb', self.models['gb'])
        ]
        
        # Meta-learner (Ridge regression for stability)
        meta_learner = Ridge(alpha=1.0)
        
        # Create stacking ensemble
        stacking_model = StackingRegressor(
            estimators=base_models,
            final_estimator=meta_learner,
            cv=5
        )
        
        stacking_model.fit(X_train, y_train)
        self.models['stacking'] = stacking_model
        
        # Create voting ensemble
        voting_model = VotingRegressor(
            estimators=base_models,
            weights=[self.model_weights.get('random_forest', 0.5), 
                    self.model_weights.get('gradient_boosting', 0.5)]
        )
        
        voting_model.fit(X_train, y_train)
        self.models['voting'] = voting_model
        
        return stacking_model, voting_model
    
    def predict_with_uncertainty(self, features):
        """Make predictions with uncertainty quantification"""
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        predictions = {}
        
        # Get predictions from all models
        predictions['rf'] = self.models['rf'].predict(features_scaled)[0]
        predictions['gb'] = self.models['gb'].predict(features_scaled)[0] 
        predictions['dl'] = float(self.models['dl'].predict(features_scaled)[0][0])
        
        if 'stacking' in self.models:
            predictions['stacking'] = self.models['stacking'].predict(features_scaled)[0]
        if 'voting' in self.models:
            predictions['voting'] = self.models['voting'].predict(features_scaled)[0]
        
        # Calculate ensemble prediction and uncertainty
        pred_values = list(predictions.values())
        ensemble_pred = np.mean(pred_values)
        uncertainty = np.std(pred_values)
        
        # Confidence interval (assuming normal distribution)
        confidence_interval = {
            'lower': ensemble_pred - 1.96 * uncertainty,
            'upper': ensemble_pred + 1.96 * uncertainty,
            'uncertainty': uncertainty
        }
        
        return ensemble_pred, confidence_interval, predictions
    
    def explain_prediction(self, features, feature_names):
        """Generate SHAP explanations for predictions"""
        try:
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            # Get SHAP values
            shap_values = self.explainer.shap_values(features_scaled)
            
            # Create explanation dictionary
            explanations = {}
            for i, feature_name in enumerate(feature_names):
                explanations[feature_name] = {
                    'shap_value': float(shap_values[0][i]),
                    'feature_value': float(features[i]),
                    'impact': 'positive' if shap_values[0][i] > 0 else 'negative'
                }
            
            # Sort by absolute impact
            sorted_explanations = dict(
                sorted(explanations.items(), 
                      key=lambda x: abs(x[1]['shap_value']), 
                      reverse=True)
            )
            
            return sorted_explanations
            
        except Exception as e:
            print(f"Error generating explanations: {e}")
            return {}
    
    def calculate_environmental_impact_detailed(self, mpg, annual_miles=12000, fuel_type='gasoline'):
        """Calculate detailed environmental impact"""
        
        # Fuel-specific carbon emissions (kg CO2 per gallon)
        emission_factors = {
            'gasoline': 8.887,
            'diesel': 10.180,
            'hybrid': 4.444,  # Roughly half of gasoline
            'electric': 0.0   # Zero direct emissions
        }
        
        emission_factor = emission_factors.get(fuel_type.lower(), 8.887)
        
        if mpg <= 0:
            return {}
        
        gallons_per_year = annual_miles / mpg
        co2_kg_annual = gallons_per_year * emission_factor
        co2_tons_annual = co2_kg_annual / 1000
        
        # Cost calculations (average fuel price per gallon)
        fuel_prices = {
            'gasoline': 3.50,
            'diesel': 3.80,
            'hybrid': 3.50,
            'electric': 0.12  # per kWh equivalent
        }
        
        fuel_price = fuel_prices.get(fuel_type.lower(), 3.50)
        annual_fuel_cost = gallons_per_year * fuel_price
        
        # Environmental equivalents
        trees_to_offset = co2_tons_annual * 16  # Trees needed to offset CO2
        coal_equivalent = co2_kg_annual / 2.86  # kg of coal burned equivalent
        
        # Comparison with average vehicle (25.4 MPG)
        avg_mpg = 25.4
        avg_gallons = annual_miles / avg_mpg
        gallons_saved = avg_gallons - gallons_per_year
        co2_saved = gallons_saved * emission_factor
        cost_saved = gallons_saved * fuel_price
        
        return {
            'annual_gallons': round(gallons_per_year, 2),
            'annual_co2_kg': round(co2_kg_annual, 2),
            'annual_co2_tons': round(co2_tons_annual, 3),
            'annual_fuel_cost': round(annual_fuel_cost, 2),
            'trees_to_offset': round(trees_to_offset, 1),
            'coal_equivalent_kg': round(coal_equivalent, 2),
            'vs_average': {
                'gallons_saved_vs_avg': round(gallons_saved, 2),
                'co2_saved_vs_avg_kg': round(co2_saved, 2),
                'cost_saved_vs_avg': round(cost_saved, 2),
                'is_better_than_average': mpg > avg_mpg
            },
            'fuel_type': fuel_type,
            'emission_factor': emission_factor
        }
    
    def generate_smart_recommendations(self, mpg, vehicle_data, explanations):
        """Generate AI-powered smart recommendations"""
        
        recommendations = []
        priority_scores = []
        
        # Base recommendations based on efficiency
        if mpg < 20:
            recommendations.extend([
                {
                    'category': 'Vehicle Choice',
                    'recommendation': 'Consider upgrading to a hybrid or electric vehicle for significant efficiency gains',
                    'impact': 'High',
                    'co2_reduction': 'Up to 50% reduction in emissions'
                },
                {
                    'category': 'Maintenance',
                    'recommendation': 'Ensure regular engine tune-ups and air filter replacement',
                    'impact': 'Medium',
                    'co2_reduction': '5-10% improvement possible'
                }
            ])
            priority_scores.extend([0.9, 0.7])
            
        elif mpg < 30:
            recommendations.extend([
                {
                    'category': 'Driving Habits',
                    'recommendation': 'Avoid aggressive acceleration and maintain steady speeds',
                    'impact': 'Medium',
                    'co2_reduction': '10-15% improvement in fuel economy'
                },
                {
                    'category': 'Vehicle Care',
                    'recommendation': 'Keep tires properly inflated and aligned',
                    'impact': 'Low-Medium',
                    'co2_reduction': '3-5% improvement in efficiency'
                }
            ])
            priority_scores.extend([0.6, 0.5])
        
        # Smart recommendations based on SHAP explanations
        if explanations:
            top_negative_features = [
                (feature, data) for feature, data in explanations.items()
                if data['impact'] == 'negative' and abs(data['shap_value']) > 0.5
            ][:3]
            
            for feature, data in top_negative_features:
                if 'engine_size' in feature and data['feature_value'] > 3.0:
                    recommendations.append({
                        'category': 'AI Insight',
                        'recommendation': f'Large engine size ({data["feature_value"]:.1f}L) significantly impacts efficiency. Consider smaller engines for daily driving.',
                        'impact': 'High',
                        'co2_reduction': 'Potential 20-30% improvement with smaller engine'
                    })
                    priority_scores.append(0.8)
                
                elif 'cylinders' in feature and data['feature_value'] > 6:
                    recommendations.append({
                        'category': 'AI Insight', 
                        'recommendation': f'{int(data["feature_value"])} cylinders reduce efficiency. 4-cylinder engines offer better fuel economy.',
                        'impact': 'Medium-High',
                        'co2_reduction': '15-25% improvement possible'
                    })
                    priority_scores.append(0.75)
                
                elif 'weight' in feature and data['feature_value'] > 4000:
                    recommendations.append({
                        'category': 'AI Insight',
                        'recommendation': f'Vehicle weight ({data["feature_value"]:.0f} lbs) impacts efficiency. Remove unnecessary items.',
                        'impact': 'Low-Medium',
                        'co2_reduction': '2-5% improvement per 100 lbs removed'
                    })
                    priority_scores.append(0.4)
        
        # Fuel-type specific recommendations
        fuel_type = vehicle_data.get('fuel_type', 'gasoline').lower()
        if fuel_type == 'gasoline':
            recommendations.append({
                'category': 'Fuel Choice',
                'recommendation': 'Consider premium fuel for high-performance engines or ethanol blends for environmental benefits',
                'impact': 'Low',
                'co2_reduction': '2-3% potential improvement'
            })
            priority_scores.append(0.3)
        
        elif fuel_type == 'hybrid':
            recommendations.append({
                'category': 'Hybrid Optimization',
                'recommendation': 'Maximize electric mode usage in city driving and maintain battery health',
                'impact': 'Medium',
                'co2_reduction': '10-20% additional improvement'
            })
            priority_scores.append(0.6)
        
        # Environmental impact recommendations
        recommendations.extend([
            {
                'category': 'Lifestyle',
                'recommendation': 'Combine errands into single trips and use public transportation when possible',
                'impact': 'Variable',
                'co2_reduction': 'Reduce overall transportation emissions by 15-30%'
            },
            {
                'category': 'Technology',
                'recommendation': 'Use eco-driving apps and real-time fuel efficiency monitoring',
                'impact': 'Medium',
                'co2_reduction': '5-15% improvement through better driving awareness'
            }
        ])
        priority_scores.extend([0.5, 0.6])
        
        # Sort recommendations by priority
        recommendations_with_priority = list(zip(recommendations, priority_scores))
        recommendations_with_priority.sort(key=lambda x: x[1], reverse=True)
        
        return [rec for rec, _ in recommendations_with_priority]
    
    def calculate_green_score(self, mpg, vehicle_data, environmental_impact):
        """Calculate comprehensive green score (0-100)"""
        
        # Base score from MPG (0-50 points)
        mpg_score = min(50, (mpg - 10) * 50 / 40)  # 10 MPG = 0, 50+ MPG = 50
        
        # Fuel type bonus (0-25 points)
        fuel_type = vehicle_data.get('fuel_type', 'gasoline').lower()
        fuel_bonuses = {
            'electric': 25,
            'hybrid': 15,
            'diesel': 5,
            'gasoline': 0
        }
        fuel_score = fuel_bonuses.get(fuel_type, 0)
        
        # Environmental impact score (0-25 points)
        co2_tons = environmental_impact.get('annual_co2_tons', 5)
        env_score = max(0, 25 - co2_tons * 5)  # Lower emissions = higher score
        
        total_score = mpg_score + fuel_score + env_score
        return min(100, max(0, total_score))
    
    def online_learning_update(self, new_data, learning_rate=0.01):
        """Simulate online learning for model adaptation"""
        
        # This is a simplified simulation - in production, you'd use
        # proper online learning algorithms like SGD or incremental learning
        
        try:
            feedback_data = pd.DataFrame(new_data)
            
            if len(feedback_data) < 10:  # Need minimum samples
                return False
            
            # Update performance tracking
            timestamp = datetime.now()
            self.performance_history.append({
                'timestamp': timestamp,
                'samples_processed': len(feedback_data),
                'avg_actual_mpg': feedback_data['actual_mpg'].mean(),
                'avg_predicted_mpg': feedback_data['predicted_mpg'].mean()
            })
            
            # Calculate model drift
            mae = mean_absolute_percentage_error(
                feedback_data['actual_mpg'], 
                feedback_data['predicted_mpg']
            )
            
            # If significant drift detected, flag for retraining
            if mae > 0.15:  # 15% error threshold
                print(f"⚠️ Model drift detected (MAE: {mae:.3f}). Consider retraining.")
                return {'retrain_needed': True, 'mae': mae}
            
            print(f"✅ Online learning update completed. MAE: {mae:.3f}")
            return {'retrain_needed': False, 'mae': mae}
            
        except Exception as e:
            print(f"❌ Online learning update failed: {e}")
            return False
    
    def get_model_performance_insights(self):
        """Get insights into model performance over time"""
        
        if not self.performance_history:
            return {}
        
        recent_performance = self.performance_history[-10:]  # Last 10 updates
        
        avg_mae = np.mean([p.get('mae', 0) for p in recent_performance if 'mae' in p])
        trend = 'improving' if len(recent_performance) > 1 and recent_performance[-1].get('mae', 1) < recent_performance[0].get('mae', 1) else 'stable'
        
        return {
            'total_updates': len(self.performance_history),
            'recent_average_mae': round(avg_mae, 3),
            'performance_trend': trend,
            'last_update': self.performance_history[-1]['timestamp'].isoformat() if self.performance_history else None,
            'samples_processed_total': sum([p['samples_processed'] for p in self.performance_history])
        }
    
    def comprehensive_prediction(self, vehicle_data, feature_names):
        """Make comprehensive prediction with all advanced features"""
        
        try:
            # Prepare features
            features = np.array([
                vehicle_data.get('year', 2020),
                vehicle_data.get('engine_size', 2.5),
                vehicle_data.get('cylinders', 4),
                vehicle_data.get('weight', 3500),
                vehicle_data.get('turbo', 0),
                vehicle_data.get('make_encoded', 0),
                vehicle_data.get('model_type_encoded', 0),
                vehicle_data.get('transmission_encoded', 1),
                vehicle_data.get('fuel_type_encoded', 0),
                vehicle_data.get('drivetrain_encoded', 0)
            ])
            
            # Make prediction with uncertainty
            prediction, confidence_interval, individual_predictions = self.predict_with_uncertainty(features)
            
            # Generate explanations
            explanations = self.explain_prediction(features, feature_names)
            
            # Calculate environmental impact
            environmental_impact = self.calculate_environmental_impact_detailed(
                prediction, 
                fuel_type=vehicle_data.get('fuel_type', 'gasoline')
            )
            
            # Generate smart recommendations
            recommendations = self.generate_smart_recommendations(
                prediction, vehicle_data, explanations
            )
            
            # Calculate green score
            green_score = self.calculate_green_score(
                prediction, vehicle_data, environmental_impact
            )
            
            return {
                'predicted_mpg': round(prediction, 2),
                'confidence_interval': confidence_interval,
                'individual_model_predictions': individual_predictions,
                'explanations': explanations,
                'environmental_impact': environmental_impact,
                'recommendations': recommendations,
                'green_score': round(green_score, 1),
                'model_insights': self.get_model_performance_insights()
            }
            
        except Exception as e:
            print(f"❌ Comprehensive prediction failed: {e}")
            return None

# Usage example
if __name__ == "__main__":
    predictor = EnhancedGreenAIPredictor()
    
    # Load models
    if predictor.load_models():
        # Example prediction
        vehicle_data = {
            'make': 'Toyota',
            'model': 'Prius',
            'year': 2022,
            'engine_size': 1.8,
            'cylinders': 4,
            'transmission': 'automatic',
            'fuel_type': 'hybrid',
            'weight': 3200,
            'turbo': 0,
            'make_encoded': 5,
            'model_type_encoded': 2,
            'transmission_encoded': 1,
            'fuel_type_encoded': 2,
            'drivetrain_encoded': 0
        }
        
        feature_names = [
            'year', 'engine_size', 'cylinders', 'weight', 'turbo',
            'make_encoded', 'model_type_encoded', 'transmission_encoded',
            'fuel_type_encoded', 'drivetrain_encoded'
        ]
        
        result = predictor.comprehensive_prediction(vehicle_data, feature_names)
        
        if result:
            print("\n🚗 Enhanced Prediction Results:")
            print(f"Predicted MPG: {result['predicted_mpg']}")
            print(f"Green Score: {result['green_score']}/100")
            print(f"Annual CO2: {result['environmental_impact']['annual_co2_tons']} tons")
            print(f"\nTop Recommendations:")
            for i, rec in enumerate(result['recommendations'][:3], 1):
                print(f"{i}. {rec['recommendation']} ({rec['impact']} impact)")
    else:
        print("❌ Failed to load models. Please run train_models.py first.")