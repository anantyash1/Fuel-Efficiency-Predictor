import requests
import json
import time
from app import app, db, Vehicle
import pandas as pd
from datetime import datetime

class VehicleAPIIntegrator:
    """Integration with various vehicle APIs"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GreenAI-Fuel-Efficiency-Predictor/1.0'
        })
    
    def fetch_nhtsa_makes(self):
        """Fetch all vehicle makes from NHTSA API"""
        print("🚗 Fetching vehicle makes from NHTSA API...")
        
        url = "https://vpic.nhtsa.dot.gov/api/vehicles/GetMakesForVehicleType/car?format=json"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            makes = []
            for item in data.get('Results', []):
                make_name = item.get('MakeName', '').strip()
                if make_name and len(make_name) > 1:
                    makes.append(make_name)
            
            print(f"✅ Found {len(makes)} vehicle makes")
            return sorted(set(makes))  # Remove duplicates and sort
            
        except Exception as e:
            print(f"❌ Error fetching makes: {e}")
            return []
    
    def fetch_nhtsa_models_for_make(self, make, year=2023):
        """Fetch models for a specific make and year"""
        print(f"📋 Fetching models for {make} ({year})...")
        
        # URL encode the make name
        make_encoded = requests.utils.quote(make)
        url = f"https://vpic.nhtsa.dot.gov/api/vehicles/GetModelsForMakeYear/make/{make_encoded}/modelyear/{year}?format=json"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            models = []
            for item in data.get('Results', []):
                model_name = item.get('Model_Name', '').strip()
                if model_name and len(model_name) > 1:
                    models.append(model_name)
            
            print(f"  Found {len(models)} models for {make}")
            return sorted(set(models))
            
        except Exception as e:
            print(f"❌ Error fetching models for {make}: {e}")
            return []
    
    def fetch_epa_fuel_economy_data(self, year=2023):
        """Fetch fuel economy data from EPA API"""
        print(f"⛽ Fetching EPA fuel economy data for {year}...")
        
        # EPA Fuel Economy API
        url = f"https://www.fueleconomy.gov/ws/rest/vehicle/menu/year/{year}"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            # EPA returns XML, we'll need to parse it
            # For now, let's use a simplified approach with known data
            return self._get_epa_sample_data(year)
            
        except Exception as e:
            print(f"❌ Error fetching EPA data: {e}")
            return []
    
    def _get_epa_sample_data(self, year):
        """Get sample EPA data (placeholder for real EPA integration)"""
        # This would normally parse EPA XML data
        # For now, returning structured sample data
        return [
            {"make": "Toyota", "model": "Prius", "year": year, "city_mpg": 58, "highway_mpg": 53, "combined_mpg": 56},
            {"make": "Honda", "model": "Civic", "year": year, "city_mpg": 32, "highway_mpg": 42, "combined_mpg": 36},
            {"make": "Ford", "model": "F-150", "year": year, "city_mpg": 19, "highway_mpg": 24, "combined_mpg": 21},
        ]
    
    def fetch_vehicle_details_nhtsa(self, make, model, year):
        """Fetch detailed vehicle specifications from NHTSA"""
        print(f"🔍 Fetching details for {year} {make} {model}...")
        
        # First, get the vehicle ID
        make_encoded = requests.utils.quote(make)
        model_encoded = requests.utils.quote(model)
        
        url = f"https://vpic.nhtsa.dot.gov/api/vehicles/GetVehicleVariableValuesList/"
        params = {
            'format': 'json',
            'years': year,
            'make': make,
            'model': model
        }
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Extract vehicle details
            vehicle_details = {}
            for item in data.get('Results', []):
                variable = item.get('Variable', '')
                value = item.get('Value', '')
                
                if 'Engine' in variable and 'Displacement' in variable:
                    try:
                        # Extract engine size from displacement
                        engine_size = float(''.join(filter(str.isdigit, value))) / 1000
                        vehicle_details['engine_size'] = engine_size
                    except:
                        pass
                
                elif 'Engine Number of Cylinders' in variable:
                    try:
                        vehicle_details['cylinders'] = int(value)
                    except:
                        pass
                
                elif 'Transmission' in variable:
                    vehicle_details['transmission'] = value
                
                elif 'Fuel Type' in variable:
                    vehicle_details['fuel_type'] = value
            
            return vehicle_details
            
        except Exception as e:
            print(f"❌ Error fetching vehicle details: {e}")
            return {}
    
    def fetch_comprehensive_vehicle_data(self, max_makes=10, years=[2023, 2022, 2021]):
        """Fetch comprehensive vehicle data from multiple sources"""
        print("🚀 Starting comprehensive vehicle data fetch...")
        
        all_vehicles = []
        
        # Get popular makes
        makes = self.fetch_nhtsa_makes()
        popular_makes = [
            "Toyota", "Honda", "Ford", "Chevrolet", "Nissan", 
            "BMW", "Mercedes-Benz", "Audi", "Hyundai", "Subaru",
            "Mazda", "Tesla", "Volkswagen", "Kia", "Jeep"
        ]
        
        # Use popular makes that are available in NHTSA
        target_makes = [make for make in popular_makes if make in makes][:max_makes]
        
        print(f"📊 Processing {len(target_makes)} manufacturers: {', '.join(target_makes)}")
        
        for make in target_makes:
            print(f"\n🔄 Processing {make}...")
            
            for year in years:
                models = self.fetch_nhtsa_models_for_make(make, year)
                
                # Limit models per make to avoid too much data
                models = models[:5] if len(models) > 5 else models
                
                for model in models:
                    try:
                        # Get vehicle specifications
                        details = self.fetch_vehicle_details_nhtsa(make, model, year)
                        
                        # Add estimated fuel economy (in real implementation, get from EPA)
                        fuel_economy = self._estimate_fuel_economy(make, model, details)
                        
                        vehicle_data = {
                            'make': make,
                            'model': model,
                            'year': year,
                            'engine_size': details.get('engine_size', 2.5),
                            'cylinders': details.get('cylinders', 4),
                            'transmission': details.get('transmission', 'Automatic'),
                            'fuel_type': details.get('fuel_type', 'Gasoline'),
                            'city_mpg': fuel_economy['city'],
                            'highway_mpg': fuel_economy['highway'],
                            'combined_mpg': fuel_economy['combined']
                        }
                        
                        all_vehicles.append(vehicle_data)
                        print(f"  ✅ Added {make} {model} ({year}) - {fuel_economy['combined']} MPG")
                        
                        # Rate limiting to be respectful to APIs
                        time.sleep(0.5)
                        
                    except Exception as e:
                        print(f"  ❌ Error processing {make} {model}: {e}")
                        continue
        
        print(f"\n✅ Collected {len(all_vehicles)} vehicle records")
        return all_vehicles
    
    def _estimate_fuel_economy(self, make, model, details):
        """Estimate fuel economy based on vehicle characteristics"""
        # This is a simplified estimation - in real implementation, 
        # you'd get actual data from EPA API
        
        base_mpg = 25
        
        # Adjust for engine size
        engine_size = details.get('engine_size', 2.5)
        if engine_size < 2.0:
            base_mpg += 8
        elif engine_size > 3.0:
            base_mpg -= 5
        
        # Adjust for cylinders
        cylinders = details.get('cylinders', 4)
        if cylinders <= 4:
            base_mpg += 3
        elif cylinders >= 8:
            base_mpg -= 8
        
        # Adjust for make (known efficiency leaders)
        if make.lower() in ['toyota', 'honda', 'nissan']:
            base_mpg += 4
        elif make.lower() in ['bmw', 'mercedes-benz', 'audi']:
            base_mpg -= 2
        
        # Adjust for model type
        if any(word in model.lower() for word in ['prius', 'insight', 'hybrid']):
            base_mpg += 20
        elif any(word in model.lower() for word in ['f-150', 'silverado', 'ram', 'truck']):
            base_mpg -= 8
        elif any(word in model.lower() for word in ['suv', 'tahoe', 'suburban', 'explorer']):
            base_mpg -= 5
        
        # Ensure realistic range
        combined_mpg = max(12, min(60, base_mpg))
        city_mpg = max(10, combined_mpg - 3)
        highway_mpg = min(65, combined_mpg + 5)
        
        return {
            'city': city_mpg,
            'highway': highway_mpg,
            'combined': combined_mpg
        }
    
    def save_to_database(self, vehicles_data):
        """Save fetched vehicle data to database"""
        print("💾 Saving vehicle data to database...")
        
        with app.app_context():
            # Clear existing data
            Vehicle.query.delete()
            db.session.commit()
            
            added_count = 0
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
                    added_count += 1
                    
                except Exception as e:
                    print(f"❌ Error adding {vehicle_data.get('make')} {vehicle_data.get('model')}: {e}")
                    continue
            
            db.session.commit()
            print(f"✅ Successfully added {added_count} vehicles to database")
            return added_count
    
    def _calculate_co2_emissions(self, mpg):
        """Calculate CO2 emissions based on MPG"""
        if mpg <= 0:
            return 0
        # 19.6 lbs CO2 per gallon, 12,000 miles per year
        return (19.6 * 12000) / mpg
    
    def export_to_csv(self, vehicles_data, filename='api_vehicle_data.csv'):
        """Export vehicle data to CSV"""
        print(f"📁 Exporting data to {filename}...")
        
        df = pd.DataFrame(vehicles_data)
        df.to_csv(filename, index=False)
        
        print(f"✅ Data exported to {filename}")
        print(f"📊 Dataset shape: {df.shape}")
        
        return filename

def fetch_vehicle_data_from_apis():
    """Main function to fetch vehicle data from APIs"""
    print("🌐 Starting vehicle data collection from APIs...")
    print("=" * 60)
    
    integrator = VehicleAPIIntegrator()
    
    # Fetch comprehensive data
    vehicles_data = integrator.fetch_comprehensive_vehicle_data(
        max_makes=8,  # Adjust based on how much data you want
        years=[2023, 2022, 2021]
    )
    
    if vehicles_data:
        # Save to database
        count = integrator.save_to_database(vehicles_data)
        
        # Export to CSV for backup/analysis
        csv_file = integrator.export_to_csv(vehicles_data)
        
        print(f"\n🎉 API Integration Complete!")
        print(f"📊 Total vehicles collected: {len(vehicles_data)}")
        print(f"💾 Vehicles saved to database: {count}")
        print(f"📁 Data exported to: {csv_file}")
        
        return vehicles_data
    else:
        print("❌ No vehicle data collected")
        return []

if __name__ == "__main__":
    vehicles = fetch_vehicle_data_from_apis()