import requests
import xml.etree.ElementTree as ET
import pandas as pd
import json
from app import app, db, Vehicle

class EPAFuelEconomyAPI:
    """Integration with EPA Fuel Economy API"""
    
    def __init__(self):
        self.base_url = "https://www.fueleconomy.gov/ws/rest"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GreenAI-Fuel-Efficiency-Predictor/1.0'
        })
    
    def get_vehicle_years(self):
        """Get available years from EPA database"""
        print("📅 Fetching available years from EPA...")
        
        try:
            response = self.session.get(f"{self.base_url}/vehicle/menu/year")
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            years = []
            
            for year_elem in root.findall('.//value'):
                year = year_elem.text
                if year and year.isdigit():
                    years.append(int(year))
            
            years = sorted(years, reverse=True)
            print(f"✅ Found years: {years[:10]}...")  # Show first 10
            return years
            
        except Exception as e:
            print(f"❌ Error fetching years: {e}")
            return list(range(2015, 2024))  # Fallback years
    
    def get_makes_for_year(self, year):
        """Get available makes for a specific year"""
        print(f"🏭 Fetching makes for {year}...")
        
        try:
            response = self.session.get(f"{self.base_url}/vehicle/menu/make?year={year}")
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            makes = []
            
            for make_elem in root.findall('.//value'):
                make = make_elem.text
                if make and len(make.strip()) > 1:
                    makes.append(make.strip())
            
            makes = sorted(set(makes))
            print(f"  Found {len(makes)} makes")
            return makes
            
        except Exception as e:
            print(f"❌ Error fetching makes for {year}: {e}")
            return []
    
    def get_models_for_make_year(self, make, year):
        """Get available models for make and year"""
        try:
            make_encoded = requests.utils.quote(make)
            response = self.session.get(f"{self.base_url}/vehicle/menu/model?year={year}&make={make_encoded}")
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            models = []
            
            for model_elem in root.findall('.//value'):
                model = model_elem.text
                if model and len(model.strip()) > 1:
                    models.append(model.strip())
            
            return sorted(set(models))
            
        except Exception as e:
            print(f"❌ Error fetching models for {make} {year}: {e}")
            return []
    
    def get_vehicle_options(self, year, make, model):
        """Get vehicle options (different configurations)"""
        try:
            make_encoded = requests.utils.quote(make)
            model_encoded = requests.utils.quote(model)
            
            url = f"{self.base_url}/vehicle/menu/options?year={year}&make={make_encoded}&model={model_encoded}"
            response = self.session.get(url)
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            options = []
            
            for option_elem in root.findall('.//menuItem'):
                option_text = option_elem.find('text')
                option_value = option_elem.find('value')
                
                if option_text is not None and option_value is not None:
                    options.append({
                        'text': option_text.text,
                        'value': option_value.text
                    })
            
            return options
            
        except Exception as e:
            print(f"❌ Error fetching options for {year} {make} {model}: {e}")
            return []
    
    def get_vehicle_details(self, vehicle_id):
        """Get detailed vehicle information by ID"""
        try:
            response = self.session.get(f"{self.base_url}/vehicle/{vehicle_id}")
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            
            vehicle_data = {}
            
            # Extract various fields
            fields_mapping = {
                'year': 'year',
                'make': 'make',
                'model': 'model',
                'displ': 'engine_size',
                'cylinders': 'cylinders',
                'trans': 'transmission',
                'drive': 'drivetrain',
                'fuel': 'fuel_type',
                'city08': 'city_mpg',
                'highway08': 'highway_mpg',
                'comb08': 'combined_mpg',
                'co2TailpipeGpm': 'co2_emissions'
            }
            
            for xml_field, db_field in fields_mapping.items():
                elem = root.find(f'.//{xml_field}')
                if elem is not None and elem.text:
                    try:
                        # Convert to appropriate type
                        if db_field in ['year', 'cylinders', 'city_mpg', 'highway_mpg', 'combined_mpg']:
                            vehicle_data[db_field] = int(elem.text)
                        elif db_field in ['engine_size', 'co2_emissions']:
                            vehicle_data[db_field] = float(elem.text)
                        else:
                            vehicle_data[db_field] = elem.text.strip()
                    except ValueError:
                        pass
            
            return vehicle_data
            
        except Exception as e:
            print(f"❌ Error fetching vehicle details for ID {vehicle_id}: {e}")
            return {}
    
    def fetch_comprehensive_epa_data(self, years=[2023, 2022, 2021], max_vehicles_per_year=200):
        """Fetch comprehensive EPA fuel economy data"""
        print("🌍 Starting comprehensive EPA data collection...")
        
        all_vehicles = []
        
        for year in years:
            print(f"\n📅 Processing year {year}...")
            
            makes = self.get_makes_for_year(year)
            
            # Focus on popular makes to avoid overwhelming the API
            popular_makes = [
                "Toyota", "Honda", "Ford", "Chevrolet", "Nissan", 
                "BMW", "Mercedes-Benz", "Audi", "Hyundai", "Subaru",
                "Mazda", "Volkswagen", "Kia", "Jeep", "Tesla"
            ]
            
            target_makes = [make for make in makes if make in popular_makes][:10]
            vehicles_this_year = 0
            
            for make in target_makes:
                if vehicles_this_year >= max_vehicles_per_year:
                    break
                
                print(f"  🔄 Processing {make}...")
                models = self.get_models_for_make_year(make, year)
                
                # Limit models per make
                models = models[:3]
                
                for model in models:
                    if vehicles_this_year >= max_vehicles_per_year:
                        break
                    
                    try:
                        options = self.get_vehicle_options(year, make, model)
                        
                        # Get details for each option (usually different engine/transmission combos)
                        for option in options[:2]:  # Limit to 2 options per model
                            if vehicles_this_year >= max_vehicles_per_year:
                                break
                            
                            vehicle_id = option.get('value')
                            if vehicle_id:
                                vehicle_details = self.get_vehicle_details(vehicle_id)
                                
                                if vehicle_details and vehicle_details.get('combined_mpg', 0) > 0:
                                    all_vehicles.append(vehicle_details)
                                    vehicles_this_year += 1
                                    
                                    mpg = vehicle_details.get('combined_mpg', 'N/A')
                                    print(f"    ✅ {make} {model} - {mpg} MPG")
                                
                                # Rate limiting
                                import time
                                time.sleep(0.3)
                    
                    except Exception as e:
                        print(f"    ❌ Error with {make} {model}: {e}")
                        continue
            
            print(f"  📊 Collected {vehicles_this_year} vehicles for {year}")
        
        print(f"\n🎉 Total vehicles collected: {len(all_vehicles)}")
        return all_vehicles
    
    def save_epa_data_to_database(self, vehicles_data):
        """Save EPA data to database"""
        print("💾 Saving EPA data to database...")
        
        with app.app_context():
            saved_count = 0
            
            for vehicle_data in vehicles_data:
                try:
                    # Check if vehicle already exists
                    existing = Vehicle.query.filter_by(
                        make=vehicle_data.get('make'),
                        model=vehicle_data.get('model'),
                        year=vehicle_data.get('year'),
                        engine_size=vehicle_data.get('engine_size', 0)
                    ).first()
                    
                    if not existing:
                        vehicle = Vehicle(
                            make=vehicle_data.get('make', ''),
                            model=vehicle_data.get('model', ''),
                            year=vehicle_data.get('year', 2020),
                            engine_size=vehicle_data.get('engine_size', 2.5),
                            cylinders=vehicle_data.get('cylinders', 4),
                            transmission=vehicle_data.get('transmission', 'Automatic'),
                            fuel_type=vehicle_data.get('fuel_type', 'Gasoline'),
                            city_mpg=vehicle_data.get('city_mpg', 25),
                            highway_mpg=vehicle_data.get('highway_mpg', 30),
                            combined_mpg=vehicle_data.get('combined_mpg', 27),
                            co2_emissions=vehicle_data.get('co2_emissions', 400)
                        )
                        
                        db.session.add(vehicle)
                        saved_count += 1
                
                except Exception as e:
                    print(f"❌ Error saving vehicle: {e}")
                    continue
            
            db.session.commit()
            print(f"✅ Saved {saved_count} new vehicles to database")
            return saved_count

def fetch_real_epa_data():
    """Main function to fetch real EPA data"""
    print("🌐 Fetching real EPA fuel economy data...")
    print("=" * 50)
    
    epa_api = EPAFuelEconomyAPI()
    
    # Fetch data for recent years
    vehicles_data = epa_api.fetch_comprehensive_epa_data(
        years=[2023, 2022, 2021], 
        max_vehicles_per_year=100
    )
    
    if vehicles_data:
        # Save to database
        count = epa_api.save_epa_data_to_database(vehicles_data)
        
        # Save to CSV for backup
        df = pd.DataFrame(vehicles_data)
        csv_filename = f'epa_vehicle_data_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv'
        df.to_csv(csv_filename, index=False)
        
        print(f"\n🎉 EPA Data Collection Complete!")
        print(f"📊 Total vehicles collected: {len(vehicles_data)}")
        print(f"💾 New vehicles saved to database: {count}")
        print(f"📁 Data backed up to: {csv_filename}")
        
        # Show sample data
        print(f"\n📋 Sample of collected data:")
        for vehicle in vehicles_data[:5]:
            make = vehicle.get('make', 'Unknown')
            model = vehicle.get('model', 'Unknown')
            year = vehicle.get('year', 'N/A')
            mpg = vehicle.get('combined_mpg', 'N/A')
            fuel = vehicle.get('fuel_type', 'N/A')
            print(f"  {year} {make} {model} - {mpg} MPG ({fuel})")
        
        return vehicles_data
    else:
        print("❌ No EPA data collected")
        return []

if __name__ == "__main__":
    data = fetch_real_epa_data()