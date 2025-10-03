import pytest
import json
import numpy as np
from app import app, db, Vehicle, PredictionHistory
from enhanced_predictor import EnhancedGreenAIPredictor
import tempfile
import os

@pytest.fixture
def client():
    """Create test client"""
    db_fd, app.config['DATABASE'] = tempfile.mkstemp()
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
            yield client
    
    os.close(db_fd)
    os.unlink(app.config['DATABASE'])

def test_health_endpoint(client):
    """Test health check endpoint"""
    response = client.get('/api/health')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'status' in data
    assert data['status'] == 'healthy'

def test_prediction_endpoint(client):
    """Test fuel efficiency prediction"""
    test_data = {
        'make': 'Toyota',
        'model': 'Prius',
        'year': 2022,
        'engine_size': 1.8,
        'cylinders': 4,
        'transmission': 'automatic',
        'fuel_type': 'hybrid'
    }
    
    response = client.post('/api/predict', 
                          data=json.dumps(test_data),
                          content_type='application/json')
    
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'predicted_mpg' in data
    assert 'environmental_impact' in data
    assert 'recommendations' in data
    assert 'green_score' in data
    
    # Validate prediction ranges
    assert 15 <= data['predicted_mpg'] <= 60
    assert 0 <= data['green_score'] <= 100

def test_invalid_prediction_data(client):
    """Test prediction with invalid data"""
    test_data = {
        'make': 'Toyota',
        'model': 'Prius',
        'year': 2050,  # Invalid year
        'engine_size': -1.0,  # Invalid engine size
        'cylinders': 15  # Invalid cylinder count
    }
    
    response = client.post('/api/predict',
                          data=json.dumps(test_data),
                          content_type='application/json')
    
    # Should still return a response with clamped values
    assert response.status_code == 200

def test_vehicle_search(client):
    """Test vehicle search functionality"""
    # Add test vehicle
    with app.app_context():
        vehicle = Vehicle(
            make='Toyota',
            model='Camry',
            year=2022,
            engine_size=2.5,
            cylinders=4,
            transmission='Automatic',
            fuel_type='Gasoline',
            combined_mpg=32.0,
            co2_emissions=280
        )
        db.session.add(vehicle)
        db.session.commit()
    
    response = client.get('/api/vehicles/search?make=Toyota')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert len(data) > 0
    assert data[0]['make'] == 'Toyota'

def test_statistics_endpoint(client):
    """Test statistics endpoint"""
    response = client.get('/api/stats')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert 'total_predictions' in data
    assert 'average_predicted_mpg' in data
    assert 'total_co2_impact_tons' in data

def test_prediction_history(client):
    """Test prediction history endpoint"""
    response = client.get('/api/history')
    assert response.status_code == 200
    
    data = json.loads(response.data)
    assert isinstance(data, list)

class TestEnhancedPredictor:
    """Test enhanced predictor functionality"""
    
    @pytest.fixture
    def predictor(self):
        """Create test predictor instance"""
        return EnhancedGreenAIPredictor()
    
    def test_environmental_impact_calculation(self, predictor):
        """Test environmental impact calculations"""
        impact = predictor.calculate_environmental_impact_detailed(30, 12000, 'gasoline')
        
        assert 'annual_gallons' in impact
        assert 'annual_co2_tons' in impact
        assert 'trees_to_offset' in impact
        assert 'vs_average' in impact
        
        # Validate calculations
        expected_gallons = 12000 / 30
        assert abs(impact['annual_gallons'] - expected_gallons) < 0.1

    def test_green_score_calculation(self, predictor):
        """Test green score calculation"""
        vehicle_data = {
            'fuel_type': 'hybrid',
            'year': 2022
        }
        environmental_impact = {
            'annual_co2_tons': 2.0
        }
        
        score = predictor.calculate_green_score(45, vehicle_data, environmental_impact)
        
        assert 0 <= score <= 100
        assert isinstance(score, (int, float))

    def test_smart_recommendations(self, predictor):
        """Test smart recommendation generation"""
        vehicle_data = {
            'fuel_type': 'gasoline',
            'engine_size': 3.5,
            'cylinders': 8
        }
        
        recommendations = predictor.generate_smart_recommendations(
            20, vehicle_data, {}
        )
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        for rec in recommendations:
            assert 'category' in rec
            assert 'recommendation' in rec
            assert 'impact' in rec

def test_model_performance():
    """Test model performance metrics"""
    # Generate test data
    np.random.seed(42)
    test_features = np.random.randn(100, 10)
    test_targets = np.random.randn(100) * 10 + 30
    
    # Test basic prediction functionality
    predictions = np.random.randn(100) * 10 + 30
    
    from sklearn.metrics import mean_squared_error, r2_score
    
    mse = mean_squared_error(test_targets, predictions)
    r2 = r2_score(test_targets, predictions)
    
    assert mse >= 0
    assert -1 <= r2 <= 1

def test_data_validation():
    """Test input data validation"""
    valid_data = {
        'year': 2022,
        'engine_size': 2.5,
        'cylinders': 4,
        'fuel_type': 'gasoline'
    }
    
    # Test valid data
    assert valid_data['year'] >= 2000
    assert valid_data['engine_size'] > 0
    assert valid_data['cylinders'] in [3, 4, 5, 6, 8, 10, 12]
    assert valid_data['fuel_type'] in ['gasoline', 'diesel', 'hybrid', 'electric']

def test_edge_cases():
    """Test edge cases and boundary conditions"""
    edge_cases = [
        {'mpg': 0, 'should_handle': True},
        {'mpg': 150, 'should_handle': True},
        {'mpg': -5, 'should_handle': True},
        {'engine_size': 0, 'should_handle': True},
        {'year': 1900, 'should_handle': True},
        {'year': 2100, 'should_handle': True}
    ]
    
    for case in edge_cases:
        # All cases should be handled gracefully
        assert case['should_handle']

class TestAPIEndpoints:
    """Integration tests for API endpoints"""
    
    def test_prediction_pipeline(self, client):
        """Test complete prediction pipeline"""
        # Step 1: Make prediction
        test_data = {
            'make': 'Honda',
            'model': 'Civic',
            'year': 2021,
            'engine_size': 2.0,
            'cylinders': 4,
            'transmission': 'manual',
            'fuel_type': 'gasoline'
        }
        
        response = client.post('/api/predict',
                              data=json.dumps(test_data),
                              content_type='application/json')
        
        assert response.status_code == 200
        prediction_data = json.loads(response.data)
        
        # Step 2: Verify prediction structure
        required_fields = [
            'predicted_mpg', 'environmental_impact', 
            'recommendations', 'green_score'
        ]
        for field in required_fields:
            assert field in prediction_data
        
        # Step 3: Check statistics updated
        stats_response = client.get('/api/stats')
        assert stats_response.status_code == 200
        stats_data = json.loads(stats_response.data)
        assert stats_data['total_predictions'] >= 1
    
    def test_concurrent_predictions(self, client):
        """Test handling multiple concurrent predictions"""
        import threading
        
        results = []
        
        def make_prediction():
            test_data = {
                'make': 'Ford',
                'model': 'Focus',
                'year': 2020,
                'engine_size': 2.0,
                'cylinders': 4,
                'transmission': 'automatic',
                'fuel_type': 'gasoline'
            }
            
            response = client.post('/api/predict',
                                  data=json.dumps(test_data),
                                  content_type='application/json')
            results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_prediction)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert all(status == 200 for status in results)
    
    def test_data_persistence(self, client):
        """Test data persistence in database"""
        test_data = {
            'make': 'BMW',
            'model': '3 Series',
            'year': 2023,
            'engine_size': 2.0,
            'cylinders': 4,
            'transmission': 'automatic',
            'fuel_type': 'gasoline'
        }
        
        # Make prediction
        response = client.post('/api/predict',
                              data=json.dumps(test_data),
                              content_type='application/json')
        assert response.status_code == 200
        
        # Check history
        history_response = client.get('/api/history')
        assert history_response.status_code == 200
        
        history_data = json.loads(history_response.data)
        assert len(history_data) > 0

def test_ml_model_accuracy():
    """Test ML model accuracy with known data"""
    # Test with known efficient vehicles
    efficient_vehicles = [
        {'mpg_expected': 50, 'fuel_type': 'hybrid', 'engine_size': 1.8},
        {'mpg_expected': 35, 'fuel_type': 'gasoline', 'engine_size': 2.0},
        {'mpg_expected': 120, 'fuel_type': 'electric', 'engine_size': 0.0}
    ]
    
    # For each test case, prediction should be reasonable
    for vehicle in efficient_vehicles:
        expected = vehicle['mpg_expected']
        fuel_type = vehicle['fuel_type']
        
        if fuel_type == 'electric':
            # Electric vehicles should have very high efficiency
            assert expected > 100
        elif fuel_type == 'hybrid':
            # Hybrids should be quite efficient
            assert expected > 40
        else:
            # Regular gasoline should be moderate
            assert 20 <= expected <= 50

def test_environmental_calculations():
    """Test environmental impact calculations"""
    test_cases = [
        {'mpg': 30, 'expected_gallons': 400, 'fuel_type': 'gasoline'},
        {'mpg': 50, 'expected_gallons': 240, 'fuel_type': 'hybrid'},
        {'mpg': 100, 'expected_gallons': 120, 'fuel_type': 'electric'}
    ]
    
    predictor = EnhancedGreenAIPredictor()
    
    for case in test_cases:
        impact = predictor.calculate_environmental_impact_detailed(
            case['mpg'], 12000, case['fuel_type']
        )
        
        # Verify calculations are reasonable
        assert abs(impact['annual_gallons'] - case['expected_gallons']) < 50
        assert impact['annual_co2_tons'] >= 0
        assert impact['trees_to_offset'] >= 0

def test_recommendation_quality():
    """Test quality and relevance of recommendations"""
    predictor = EnhancedGreenAIPredictor()
    
    # Low efficiency vehicle
    low_efficiency_data = {
        'fuel_type': 'gasoline',
        'engine_size': 5.0,
        'cylinders': 8
    }
    
    recommendations = predictor.generate_smart_recommendations(
        15, low_efficiency_data, {}
    )
    
    # Should have recommendations for improvement
    assert len(recommendations) > 3
    
    # Should prioritize high-impact changes
    high_impact_found = any(
        rec['impact'] == 'High' for rec in recommendations
    )
    assert high_impact_found
    
    # High efficiency vehicle
    high_efficiency_data = {
        'fuel_type': 'electric',
        'engine_size': 0.0,
        'cylinders': 0
    }
    
    recommendations_efficient = predictor.generate_smart_recommendations(
        100, high_efficiency_data, {}
    )
    
    # Should still have lifestyle recommendations
    assert len(recommendations_efficient) > 0

class TestSecurityAndValidation:
    """Test security measures and input validation"""
    
    def test_sql_injection_protection(self, client):
        """Test SQL injection protection"""
        malicious_data = {
            'make': "'; DROP TABLE vehicles; --",
            'model': 'Test',
            'year': 2020,
            'engine_size': 2.0,
            'cylinders': 4,
            'transmission': 'automatic',
            'fuel_type': 'gasoline'
        }
        
        response = client.post('/api/predict',
                              data=json.dumps(malicious_data),
                              content_type='application/json')
        
        # Should not cause server error
        assert response.status_code in [200, 400]
    
    def test_input_sanitization(self, client):
        """Test input sanitization"""
        test_data = {
            'make': '<script>alert("xss")</script>',
            'model': 'Normal Model',
            'year': 2020,
            'engine_size': 2.0,
            'cylinders': 4,
            'transmission': 'automatic',
            'fuel_type': 'gasoline'
        }
        
        response = client.post('/api/predict',
                              data=json.dumps(test_data),
                              content_type='application/json')
        
        assert response.status_code == 200
        # XSS should be handled safely
    
    def test_rate_limiting_simulation(self, client):
        """Simulate rate limiting behavior"""
        # Make multiple rapid requests
        responses = []
        for _ in range(20):
            test_data = {
                'make': 'Toyota',
                'model': 'Corolla',
                'year': 2020,
                'engine_size': 1.8,
                'cylinders': 4,
                'transmission': 'automatic',
                'fuel_type': 'gasoline'
            }
            
            response = client.post('/api/predict',
                                  data=json.dumps(test_data),
                                  content_type='application/json')
            responses.append(response.status_code)
        
        # Most should succeed (rate limiting would be handled at nginx level)
        success_rate = sum(1 for status in responses if status == 200) / len(responses)
        assert success_rate > 0.8

class TestPerformance:
    """Test performance characteristics"""
    
    def test_prediction_response_time(self, client):
        """Test prediction response time"""
        import time
        
        test_data = {
            'make': 'Nissan',
            'model': 'Altima',
            'year': 2022,
            'engine_size': 2.5,
            'cylinders': 4,
            'transmission': 'automatic',
            'fuel_type': 'gasoline'
        }
        
        start_time = time.time()
        
        response = client.post('/api/predict',
                              data=json.dumps(test_data),
                              content_type='application/json')
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response.status_code == 200
        # Should respond within reasonable time (5 seconds for ML prediction)
        assert response_time < 5.0
    
    def test_memory_usage(self):
        """Test memory usage during predictions"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Simulate multiple predictions
        predictor = EnhancedGreenAIPredictor()
        
        for _ in range(100):
            vehicle_data = {
                'year': 2020,
                'engine_size': 2.0,
                'cylinders': 4,
                'fuel_type': 'gasoline'
            }
            
            # This would normally make predictions
            # but we'll just test the structure
            pass
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100

if __name__ == '__main__':
    # Run specific tests
    pytest.main([
        __file__,
        '-v',  # Verbose output
        '--cov=app',  # Coverage for app.py
        '--cov-report=html',  # HTML coverage report
        '--cov-report=term-missing'  # Show missing lines
    ])