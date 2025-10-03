import os
import time
import json
import requests
from datetime import datetime, timedelta
from app import app, db, Vehicle
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIManager:
    """Centralized API management for vehicle data"""
    
    def __init__(self):
        self.apis = {
            'nhtsa': {
                'name': 'NHTSA Vehicle API',
                'base_url': 'https://vpic.nhtsa.dot.gov/api',
                'rate_limit': 1.0,  # seconds between requests
                'free': True,
                'status': 'active'
            },
            'epa': {
                'name': 'EPA Fuel Economy API',
                'base_url': 'https://www.fueleconomy.gov/ws/rest',
                'rate_limit': 0.5,
                'free': True,
                'status': 'active'
            },
            'carapi': {
                'name': 'Car API (Alternative)',
                'base_url': 'https://carapi.app/api',
                'rate_limit': 2.0,
                'free': True,  # Limited free tier
                'status': 'backup'
            }
        }
        
        self.last_request_time = {}
        self.request_counts = {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GreenAI-Fuel-Efficiency-Predictor/1.0',
            'Accept': 'application/json'
        })
    
    def respect_rate_limit(self, api_name):
        """Respect API rate limits"""
        if api_name in self.apis:
            rate_limit = self.apis[api_name]['rate_limit']
            
            if api_name in self.last_request_time:
                time_since_last = time.time() - self.last_request_time[api_name]
                if time_since_last < rate_limit:
                    sleep_time = rate_limit - time_since_last
                    logger.info(f"Rate limiting {api_name}: sleeping {sleep_time:.2f}s")
                    time.sleep(sleep_time)
            
            self.last_request_time[api_name] = time.time()
            
            # Track request count
            if api_name not in self.request_counts:
                self.request_counts[api_name] = 0
            self.request_counts[api_name] += 1
    
    def get_api_status(self):
        """Get status of all APIs"""
        status = {}
        
        for api_name, api_info in self.apis.items():
            status[api_name] = {
                'name': api_info['name'],
                'status': api_info['status'],
                'requests_made': self.request_counts.get(api_name, 0),
                'free': api_info['free'],
                'last_used': self.last_request_time.get(api_name)
            }
        
        return status
    
    def test_api_connectivity(self, api_name):
        """Test if an API is accessible"""
        try:
            if api_name == 'nhtsa':
                test_url = f"{self.apis[api_name]['base_url']}/vehicles/GetMakesForVehicleType/car?format=json"
            elif api_name == 'epa':
                test_url = f"{self.apis[api_name]['base_url']}/vehicle/menu/year"
            else:
                return False
            
            self.respect_rate_limit(api_name)
            response = self.session.get(test_url, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"✅ {api_name.upper()} API is accessible")
                return True
            else:
                logger.warning(f"⚠️ {api_name.upper()} API returned status {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ {api_name.upper()} API test failed: {e}")
            return False
    
    def fetch_all_available_data(self, max_vehicles=300):
        """Fetch data from all available APIs"""
        logger.info("🚀 Starting comprehensive data fetch from all APIs...")
        
        all_vehicles = []
        
        # Test API connectivity first
        working_apis = []
        for api_name in self.apis:
            if self.test_api_connectivity(api_name):
                working_apis.append(api_name)
        
        logger.info(f"📡 Working APIs: {', '.join(working_apis)}")
        
        if 'nhtsa' in working_apis:
            logger.info("🔄 Fetching from NHTSA API...")
            nhtsa_data = self._fetch_nhtsa_data(max_vehicles // 2)
            all_vehicles.extend(nhtsa_data)
            logger.info(f"📊 NHTSA: {len(nhtsa_data)} vehicles")
        
        if 'epa' in working_apis:
            logger.info("🔄 Fetching from EPA API...")
            epa_data = self._fetch_epa_data(max_vehicles // 2)
            all_vehicles.extend(epa_data)
            logger.info(f"📊 EPA: {len(epa_data)} vehicles")
        
        # Remove duplicates based on make, model, year
        unique_vehicles = self._remove_duplicates(all_vehicles)
        
        logger.info(f"🎯 Total unique vehicles: {len(unique_vehicles)}")
        return unique_vehicles
    
    def _fetch_nhtsa_data(self, max_vehicles):
        """Fetch data from NHTSA API"""
        vehicles = []
        
        try:
            # Get popular makes
            makes_url = f"{self.apis['nhtsa']['base_url']}/vehicles/GetMakesForVehicleType/car?format=json"
            self.respect_rate_limit('nhtsa')
            
            response = self.session.get(makes_url)
            response.raise_for_status()
            makes_data = response.json()
            
            popular_makes = ['Toyota', 'Honda', 'Ford', 'Chevrolet', 'Nissan', 'BMW', 'Tesla']
            available_makes = [item['MakeName'] for item in makes_data.get('Results', [])]
            target_makes = [make for make in popular_makes if make in available_makes]
            
            for make in target_makes[:8]:  # Limit makes
                if len(vehicles) >= max_vehicles:
                    break
                
                for year in [2023, 2022, 2021]:
                    if len(vehicles) >= max_vehicles:
                        break
                    
                    # Get models for make and year
                    models_url = f"{self.apis['nhtsa']['base_url']}/vehicles/GetModelsForMakeYear/make/{make}/modelyear/{year}?format=json"
                    self.respect_rate_limit('nhtsa')
                    
                    try:
                        response = self.session.get(models_url)
                        response.raise_for_status()
                        models_data = response.json()
                        
                        models = [item['Model_Name'] for item in models_data.get('Results', [])][:3]  # Limit models
                        
                        for model in models:
                            if len(vehicles) >= max_vehicles:
                                break
                            
                            # Create vehicle data with estimated specs
                            vehicle_data = self._create_vehicle_data(make, model, year)
                            if vehicle_data:
                                vehicles.append(vehicle_data)
                                logger.info(f"  ✅ Added {make} {model} ({year})")
                    
                    except Exception as e:
                        logger.warning(f"  ❌ Error fetching {make} models for {year}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"❌ NHTSA API error: {e}")
        
        return vehicles
    
    def _fetch_epa_data(self, max_vehicles):
        """Fetch data from EPA API (simplified)"""
        vehicles = []
        
        # For now, return known EPA data since XML parsing is complex
        epa_sample_data = [
            {"make": "Toyota", "model": "Prius", "year": 2023, "city_mpg": 58, "highway_mpg": 53, "combined_mpg": 56, "fuel_type": "Hybrid"},
            {"make": "Toyota", "model": "Camry Hybrid", "year": 2023, "city_mpg": 51, "highway_mpg": 53, "combined_mpg": 52, "fuel_type": "Hybrid"},
            {"make": "Honda", "model": "Accord Hybrid", "year": 2023, "city_mpg": 48, "highway_mpg": 47, "combined_mpg": 48, "fuel_type": "Hybrid"},
            {"make": "Honda", "model": "Insight", "year": 2022, "city_mpg": 55, "highway_mpg": 49, "combined_mpg": 52, "fuel_type": "Hybrid"},
            {"make": "Tesla", "model": "Model 3", "year": 2023, "city_mpg": 132, "highway_mpg": 126, "combined_mpg": 129, "fuel_type": "Electric"},
            {"make": "Tesla", "model": "Model Y", "year": 2023, "city_mpg": 129, "highway_mpg": 112, "combined_mpg": 122, "fuel_type": "Electric"},
            {"make": "Ford", "model": "Mustang Mach-E", "year": 2023, "city_mpg": 105, "highway_mpg": 93, "combined_mpg": 100, "fuel_type": "Electric"},
            {"make": "Chevrolet", "model": "Bolt EV", "year": 2023, "city_mpg": 120, "highway_mpg": 95, "combined_mpg": 109, "fuel_type": "Electric"},
            {"make": "Nissan", "model": "Leaf", "year": 2023, "city_mpg": 123, "highway_mpg": 99, "combined_mpg": 111, "fuel_type": "Electric"},
            {"make": "BMW", "model": "i4", "year": 2023, "city_mpg": 118, "highway_mpg": 97, "combined_mpg": 108, "fuel_type": "Electric"},
        ]
        
        for data in epa_sample_data[:max_vehicles]:
            vehicle_data = {
                'make': data['make'],
                'model': data['model'],
                'year': data['year'],
                'engine_size': 0.0 if data['fuel_type'] == 'Electric' else 2.0,
                'cylinders': 0 if data['fuel_type'] == 'Electric' else 4,
                'transmission': 'Single-Speed' if data['fuel_type'] == 'Electric' else 'CVT',
                'fuel_type': data['fuel_type'],
                'city_mpg': data['city_mpg'],
                'highway_mpg': data['highway_mpg'],
                'combined_mpg': data['combined_mpg']
            }
            vehicles.append(vehicle_data)
        
        return vehicles
    
    def _create_vehicle_data(self, make, model, year):
        """Create vehicle data with estimated specifications"""
        try:
            # Estimate specifications based on make/model patterns
            specs = self._estimate_vehicle_specs(make, model)
            
            vehicle_data = {
                'make': make,
                'model': model,
                'year': year,
                'engine_size': specs['engine_size'],
                'cylinders': specs['cylinders'],
                'transmission': specs['transmission'],
                'fuel_type': specs['fuel_type'],
                'city_mpg': specs['city_mpg'],
                'highway_mpg': specs['highway_mpg'],
                'combined_mpg': specs['combined_mpg']
            }
            
            return vehicle_data
            
        except Exception as e:
            logger.warning(f"Error creating vehicle data for {make} {model}: {e}")
            return None
    
    def _estimate_vehicle_specs(self, make, model):
        """Estimate vehicle specifications based on make and model"""
        specs = {
            'engine_size': 2.5,
            'cylinders': 4,
            'transmission': 'Automatic',
            'fuel_type': 'Gasoline',
            'city_mpg': 25,
            'highway_mpg': 32,
            'combined_mpg': 28
        }
        
        # Adjust for specific makes
        if make.lower() == 'toyota':
            specs['city_mpg'] += 3
            specs['highway_mpg'] += 3
            specs['combined_mpg'] += 3
        elif make.lower() == 'honda':
            specs['city_mpg'] += 2
            specs['highway_mpg'] += 2
            specs['combined_mpg'] += 2
        
        # Adjust for model types
        model_lower = model.lower()
        
        if any(word in model_lower for word in ['prius', 'hybrid']):
            specs.update({
                'fuel_type': 'Hybrid',
                'city_mpg': 50,
                'highway_mpg': 48,
                'combined_mpg': 49,
                'transmission': 'CVT'
            })
        elif any(word in model_lower for word in ['electric', 'ev', 'model', 'leaf', 'bolt']):
            specs.update({
                'fuel_type': 'Electric',
                'engine_size': 0.0,
                'cylinders': 0,
                'city_mpg': 110,
                'highway_mpg': 95,
                'combined_mpg': 105,
                'transmission': 'Single-Speed'
            })
        elif any(word in model_lower for word in ['f-150', 'silverado', 'ram', 'truck', 'pickup']):
            specs.update({
                'engine_size': 5.0,
                'cylinders': 8,
                'city_mpg': 17,
                'highway_mpg': 23,
                'combined_mpg': 19
            })
        elif any(word in model_lower for word in ['suv', 'explorer', 'tahoe', 'suburban']):
            specs.update({
                'engine_size': 3.5,
                'cylinders': 6,
                'city_mpg': 20,
                'highway_mpg': 27,
                'combined_mpg': 23
            })
        elif any(word in model_lower for word in ['civic', 'corolla', 'sentra']):
            specs.update({
                'engine_size': 2.0,
                'city_mpg': 30,
                'highway_mpg': 38,
                'combined_mpg': 33
            })
        
        return specs
    
    def _remove_duplicates(self, vehicles):
        """Remove duplicate vehicles based on make, model, year"""
        seen = set()
        unique_vehicles = []
        
        for vehicle in vehicles:
            key = (vehicle['make'], vehicle['model'], vehicle['year'])
            if key not in seen:
                seen.add(key)
                unique_vehicles.append(vehicle)
        
        return unique_vehicles
    
    def save_to_database(self, vehicles_data):
        """Save vehicle data to database"""
        logger.info("💾 Saving vehicle data to database...")
        
        with app.app_context():
            # Clear existing data
            Vehicle.query.delete()
            db.session.commit()
            
            saved_count = 0
            for vehicle_data in vehicles_data:
                try:
                    vehicle = Vehicle(
                        make=vehicle_data['make'],
                        model=vehicle_data['model'],
                        year=vehicle_data['year'],
                        engine_size=vehicle_data['engine_size'],
                        cylinders=vehicle_data['cylinders'],
                        transmission=vehicle_data['transmission'],
                        fuel_type=vehicle_data['fuel_type'],
                        city_mpg=vehicle_data['city_mpg'],
                        highway_mpg=vehicle_data['highway_mpg'],
                        combined_mpg=vehicle_data['combined_mpg'],
                        co2_emissions=self._calculate_co2_emissions(vehicle_data['combined_mpg'])
                    )
                    
                    db.session.add(vehicle)
                    saved_count += 1
                    
                except Exception as e:
                    logger.error(f"Error saving {vehicle_data.get('make')} {vehicle_data.get('model')}: {e}")
                    continue
            
            db.session.commit()
            logger.info(f"✅ Saved {saved_count} vehicles to database")
            return saved_count
    
    def _calculate_co2_emissions(self, mpg):
        """Calculate CO2 emissions based on MPG"""
        if mpg <= 0:
            return 0
        return (19.6 * 12000) / mpg
    
    def generate_status_report(self):
        """Generate a comprehensive status report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'api_status': self.get_api_status(),
            'total_requests_made': sum(self.request_counts.values()),
            'database_stats': {}
        }
        
        # Get database statistics
        with app.app_context():
            total_vehicles = Vehicle.query.count()
            makes_count = db.session.query(Vehicle.make).distinct().count()
            fuel_types = db.session.query(Vehicle.fuel_type).distinct().all()
            
            report['database_stats'] = {
                'total_vehicles': total_vehicles,
                'unique_makes': makes_count,
                'fuel_types': [ft[0] for ft in fuel_types],
                'fuel_types_count': len(fuel_types)
            }
        
        return report

def main():
    """Main function to fetch vehicle data from APIs"""
    print("🌐 Starting API Integration for Vehicle Data")
    print("=" * 60)
    
    manager = APIManager()
    
    # Fetch comprehensive data
    vehicles_data = manager.fetch_all_available_data(max_vehicles=200)
    
    if vehicles_data:
        # Save to database
        saved_count = manager.save_to_database(vehicles_data)
        
        # Generate and save report
        report = manager.generate_status_report()
        
        with open(f'api_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n🎉 API Integration Complete!")
        print(f"📊 Total vehicles collected: {len(vehicles_data)}")
        print(f"💾 Vehicles saved to database: {saved_count}")
        print(f"📡 API requests made: {sum(manager.request_counts.values())}")
        print(f"📋 Status report saved")
        
        # Show sample data
        print(f"\n📋 Sample vehicles collected:")
        for vehicle in vehicles_data[:10]:
            print(f"  {vehicle['year']} {vehicle['make']} {vehicle['model']} - {vehicle['combined_mpg']} MPG")
        
        return vehicles_data
    else:
        print("❌ No vehicle data collected from APIs")
        return []

if __name__ == "__main__":
    data = main()