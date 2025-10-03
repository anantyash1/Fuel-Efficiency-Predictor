import pandas as pd
import numpy as np
import requests
import json
from app import app, db, Vehicle
import os

# Comprehensive vehicle database with real-world data
COMPREHENSIVE_VEHICLE_DATA = [
    # Toyota Models
    {"make": "Toyota", "model": "Prius", "year": 2023, "engine_size": 1.8, "cylinders": 4, "transmission": "CVT", "fuel_type": "Hybrid", "city_mpg": 58, "highway_mpg": 53, "combined_mpg": 56},
    {"make": "Toyota", "model": "Prius", "year": 2022, "engine_size": 1.8, "cylinders": 4, "transmission": "CVT", "fuel_type": "Hybrid", "city_mpg": 58, "highway_mpg": 53, "combined_mpg": 56},
    {"make": "Toyota", "model": "Prius", "year": 2021, "engine_size": 1.8, "cylinders": 4, "transmission": "CVT", "fuel_type": "Hybrid", "city_mpg": 56, "highway_mpg": 51, "combined_mpg": 54},
    {"make": "Toyota", "model": "Camry", "year": 2023, "engine_size": 2.5, "cylinders": 4, "transmission": "8-Speed Automatic", "fuel_type": "Gasoline", "city_mpg": 28, "highway_mpg": 39, "combined_mpg": 32},
    {"make": "Toyota", "model": "Camry Hybrid", "year": 2023, "engine_size": 2.5, "cylinders": 4, "transmission": "CVT", "fuel_type": "Hybrid", "city_mpg": 51, "highway_mpg": 53, "combined_mpg": 52},
    {"make": "Toyota", "model": "Corolla", "year": 2023, "engine_size": 2.0, "cylinders": 4, "transmission": "CVT", "fuel_type": "Gasoline", "city_mpg": 32, "highway_mpg": 41, "combined_mpg": 36},
    {"make": "Toyota", "model": "RAV4", "year": 2023, "engine_size": 2.5, "cylinders": 4, "transmission": "8-Speed Automatic", "fuel_type": "Gasoline", "city_mpg": 27, "highway_mpg": 35, "combined_mpg": 30},
    {"make": "Toyota", "model": "RAV4 Hybrid", "year": 2023, "engine_size": 2.5, "cylinders": 4, "transmission": "CVT", "fuel_type": "Hybrid", "city_mpg": 41, "highway_mpg": 38, "combined_mpg": 40},
    {"make": "Toyota", "model": "Highlander", "year": 2023, "engine_size": 3.5, "cylinders": 6, "transmission": "8-Speed Automatic", "fuel_type": "Gasoline", "city_mpg": 21, "highway_mpg": 29, "combined_mpg": 24},
    
    # Honda Models
    {"make": "Honda", "model": "Civic", "year": 2023, "engine_size": 2.0, "cylinders": 4, "transmission": "CVT", "fuel_type": "Gasoline", "city_mpg": 32, "highway_mpg": 42, "combined_mpg": 36},
    {"make": "Honda", "model": "Civic", "year": 2022, "engine_size": 2.0, "cylinders": 4, "transmission": "Manual", "fuel_type": "Gasoline", "city_mpg": 31, "highway_mpg": 40, "combined_mpg": 35},
    {"make": "Honda", "model": "Accord", "year": 2023, "engine_size": 1.5, "cylinders": 4, "transmission": "CVT", "fuel_type": "Gasoline", "city_mpg": 32, "highway_mpg": 42, "combined_mpg": 36},
    {"make": "Honda", "model": "Accord Hybrid", "year": 2023, "engine_size": 2.0, "cylinders": 4, "transmission": "CVT", "fuel_type": "Hybrid", "city_mpg": 48, "highway_mpg": 47, "combined_mpg": 48},
    {"make": "Honda", "model": "CR-V", "year": 2023, "engine_size": 1.5, "cylinders": 4, "transmission": "CVT", "fuel_type": "Gasoline", "city_mpg": 27, "highway_mpg": 32, "combined_mpg": 29},
    {"make": "Honda", "model": "Pilot", "year": 2023, "engine_size": 3.5, "cylinders": 6, "transmission": "9-Speed Automatic", "fuel_type": "Gasoline", "city_mpg": 20, "highway_mpg": 27, "combined_mpg": 23},
    {"make": "Honda", "model": "Insight", "year": 2022, "engine_size": 1.5, "cylinders": 4, "transmission": "CVT", "fuel_type": "Hybrid", "city_mpg": 55, "highway_mpg": 49, "combined_mpg": 52},
    
    # Ford Models
    {"make": "Ford", "model": "F-150", "year": 2023, "engine_size": 3.3, "cylinders": 6, "transmission": "10-Speed Automatic", "fuel_type": "Gasoline", "city_mpg": 19, "highway_mpg": 24, "combined_mpg": 21},
    {"make": "Ford", "model": "F-150", "year": 2023, "engine_size": 2.7, "cylinders": 6, "transmission": "10-Speed Automatic", "fuel_type": "Gasoline", "city_mpg": 20, "highway_mpg": 26, "combined_mpg": 22},
    {"make": "Ford", "model": "F-150", "year": 2023, "engine_size": 5.0, "cylinders": 8, "transmission": "10-Speed Automatic", "fuel_type": "Gasoline", "city_mpg": 17, "highway_mpg": 24, "combined_mpg": 20},
    {"make": "Ford", "model": "Escape", "year": 2023, "engine_size": 1.5, "cylinders": 3, "transmission": "8-Speed Automatic", "fuel_type": "Gasoline", "city_mpg": 28, "highway_mpg": 34, "combined_mpg": 30},
    {"make": "Ford", "model": "Mustang", "year": 2023, "engine_size": 2.3, "cylinders": 4, "transmission": "Manual", "fuel_type": "Gasoline", "city_mpg": 21, "highway_mpg": 32, "combined_mpg": 25},
    {"make": "Ford", "model": "Mustang", "year": 2023, "engine_size": 5.0, "cylinders": 8, "transmission": "Manual", "fuel_type": "Gasoline", "city_mpg": 16, "highway_mpg": 25, "combined_mpg": 19},
    {"make": "Ford", "model": "Fusion Hybrid", "year": 2020, "engine_size": 2.0, "cylinders": 4, "transmission": "CVT", "fuel_type": "Hybrid", "city_mpg": 43, "highway_mpg": 41, "combined_mpg": 42},
    
    # Chevrolet Models
    {"make": "Chevrolet", "model": "Silverado 1500", "year": 2023, "engine_size": 2.7, "cylinders": 4, "transmission": "8-Speed Automatic", "fuel_type": "Gasoline", "city_mpg": 20, "highway_mpg": 23, "combined_mpg": 21},
    {"make": "Chevrolet", "model": "Silverado 1500", "year": 2023, "engine_size": 5.3, "cylinders": 8, "transmission": "8-Speed Automatic", "fuel_type": "Gasoline", "city_mpg": 16, "highway_mpg": 22, "combined_mpg": 18},
    {"make": "Chevrolet", "model": "Malibu", "year": 2023, "engine_size": 1.5, "cylinders": 4, "transmission": "CVT", "fuel_type": "Gasoline", "city_mpg": 29, "highway_mpg": 36, "combined_mpg": 32},
    {"make": "Chevrolet", "model": "Equinox", "year": 2023, "engine_size": 1.5, "cylinders": 4, "transmission": "6-Speed Automatic", "fuel_type": "Gasoline", "city_mpg": 26, "highway_mpg": 31, "combined_mpg": 28},
    {"make": "Chevrolet", "model": "Tahoe", "year": 2023, "engine_size": 5.3, "cylinders": 8, "transmission": "10-Speed Automatic", "fuel_type": "Gasoline", "city_mpg": 14, "highway_mpg": 19, "combined_mpg": 16},
    
    # Nissan Models
    {"make": "Nissan", "model": "Altima", "year": 2023, "engine_size": 2.5, "cylinders": 4, "transmission": "CVT", "fuel_type": "Gasoline", "city_mpg": 28, "highway_mpg": 39, "combined_mpg": 32},
    {"make": "Nissan", "model": "Sentra", "year": 2023, "engine_size": 2.0, "cylinders": 4, "transmission": "CVT", "fuel_type": "Gasoline", "city_mpg": 29, "highway_mpg": 39, "combined_mpg": 33},
    {"make": "Nissan", "model": "Rogue", "year": 2023, "engine_size": 2.5, "cylinders": 4, "transmission": "CVT", "fuel_type": "Gasoline", "city_mpg": 27, "highway_mpg": 35, "combined_mpg": 30},
    {"make": "Nissan", "model": "Pathfinder", "year": 2023, "engine_size": 3.5, "cylinders": 6, "transmission": "9-Speed Automatic", "fuel_type": "Gasoline", "city_mpg": 21, "highway_mpg": 27, "combined_mpg": 24},
    {"make": "Nissan", "model": "Leaf", "year": 2023, "engine_size": 0.0, "cylinders": 0, "transmission": "Single-Speed", "fuel_type": "Electric", "city_mpg": 123, "highway_mpg": 99, "combined_mpg": 111},
    
    # BMW Models
    {"make": "BMW", "model": "3 Series", "year": 2023, "engine_size": 2.0, "cylinders": 4, "transmission": "8-Speed Automatic", "fuel_type": "Gasoline", "city_mpg": 26, "highway_mpg": 36, "combined_mpg": 30},
    {"make": "BMW", "model": "X3", "year": 2023, "engine_size": 2.0, "cylinders": 4, "transmission": "8-Speed Automatic", "fuel_type": "Gasoline", "city_mpg": 23, "highway_mpg": 29, "combined_mpg": 25},
    {"make": "BMW", "model": "X5", "year": 2023, "engine_size": 3.0, "cylinders": 6, "transmission": "8-Speed Automatic", "fuel_type": "Gasoline", "city_mpg": 21, "highway_mpg": 26, "combined_mpg": 23},
    {"make": "BMW", "model": "i4", "year": 2023, "engine_size": 0.0, "cylinders": 0, "transmission": "Single-Speed", "fuel_type": "Electric", "city_mpg": 118, "highway_mpg": 97, "combined_mpg": 108},
    
    # Mercedes-Benz Models
    {"make": "Mercedes-Benz", "model": "C-Class", "year": 2023, "engine_size": 2.0, "cylinders": 4, "transmission": "9-Speed Automatic", "fuel_type": "Gasoline", "city_mpg": 23, "highway_mpg": 32, "combined_mpg": 26},
    {"make": "Mercedes-Benz", "model": "GLC", "year": 2023, "engine_size": 2.0, "cylinders": 4, "transmission": "9-Speed Automatic", "fuel_type": "Gasoline", "city_mpg": 22, "highway_mpg": 29, "combined_mpg": 25},
    {"make": "Mercedes-Benz", "model": "EQS", "year": 2023, "engine_size": 0.0, "cylinders": 0, "transmission": "Single-Speed", "fuel_type": "Electric", "city_mpg": 122, "highway_mpg": 90, "combined_mpg": 107},
    
    # Tesla Models
    {"make": "Tesla", "model": "Model 3", "year": 2023, "engine_size": 0.0, "cylinders": 0, "transmission": "Single-Speed", "fuel_type": "Electric", "city_mpg": 132, "highway_mpg": 126, "combined_mpg": 129},
    {"make": "Tesla", "model": "Model Y", "year": 2023, "engine_size": 0.0, "cylinders": 0, "transmission": "Single-Speed", "fuel_type": "Electric", "city_mpg": 129, "highway_mpg": 112, "combined_mpg": 122},
    {"make": "Tesla", "model": "Model S", "year": 2023, "engine_size": 0.0, "cylinders": 0, "transmission": "Single-Speed", "fuel_type": "Electric", "city_mpg": 120, "highway_mpg": 115, "combined_mpg": 118},
    {"make": "Tesla", "model": "Model X", "year": 2023, "engine_size": 0.0, "cylinders": 0, "transmission": "Single-Speed", "fuel_type": "Electric", "city_mpg": 102, "highway_mpg": 97, "combined_mpg": 100},
    
    # Hyundai Models
    {"make": "Hyundai", "model": "Elantra", "year": 2023, "engine_size": 2.0, "cylinders": 4, "transmission": "CVT", "fuel_type": "Gasoline", "city_mpg": 31, "highway_mpg": 41, "combined_mpg": 35},
    {"make": "Hyundai", "model": "Sonata Hybrid", "year": 2023, "engine_size": 2.0, "cylinders": 4, "transmission": "6-Speed Automatic", "fuel_type": "Hybrid", "city_mpg": 45, "highway_mpg": 51, "combined_mpg": 47},
    {"make": "Hyundai", "model": "Tucson", "year": 2023, "engine_size": 2.5, "cylinders": 4, "transmission": "8-Speed Automatic", "fuel_type": "Gasoline", "city_mpg": 26, "highway_mpg": 33, "combined_mpg": 29},
    {"make": "Hyundai", "model": "IONIQ 5", "year": 2023, "engine_size": 0.0, "cylinders": 0, "transmission": "Single-Speed", "fuel_type": "Electric", "city_mpg": 114, "highway_mpg": 94, "combined_mpg": 105},
    
    # Audi Models
    {"make": "Audi", "model": "A4", "year": 2023, "engine_size": 2.0, "cylinders": 4, "transmission": "7-Speed Automatic", "fuel_type": "Gasoline", "city_mpg": 24, "highway_mpg": 31, "combined_mpg": 27},
    {"make": "Audi", "model": "Q5", "year": 2023, "engine_size": 2.0, "cylinders": 4, "transmission": "7-Speed Automatic", "fuel_type": "Gasoline", "city_mpg": 22, "highway_mpg": 28, "combined_mpg": 25},
    {"make": "Audi", "model": "e-tron", "year": 2023, "engine_size": 0.0, "cylinders": 0, "transmission": "Single-Speed", "fuel_type": "Electric", "city_mpg": 78, "highway_mpg": 77, "combined_mpg": 78},
    
    # Subaru Models
    {"make": "Subaru", "model": "Outback", "year": 2023, "engine_size": 2.5, "cylinders": 4, "transmission": "CVT", "fuel_type": "Gasoline", "city_mpg": 26, "highway_mpg": 33, "combined_mpg": 29},
    {"make": "Subaru", "model": "Forester", "year": 2023, "engine_size": 2.5, "cylinders": 4, "transmission": "CVT", "fuel_type": "Gasoline", "city_mpg": 26, "highway_mpg": 33, "combined_mpg": 29},
    {"make": "Subaru", "model": "Impreza", "year": 2023, "engine_size": 2.0, "cylinders": 4, "transmission": "CVT", "fuel_type": "Gasoline", "city_mpg": 28, "highway_mpg": 36, "combined_mpg": 31},
    
    # Mazda Models
    {"make": "Mazda", "model": "Mazda3", "year": 2023, "engine_size": 2.5, "cylinders": 4, "transmission": "6-Speed Automatic", "fuel_type": "Gasoline", "city_mpg": 26, "highway_mpg": 35, "combined_mpg": 30},
    {"make": "Mazda", "model": "CX-5", "year": 2023, "engine_size": 2.5, "cylinders": 4, "transmission": "6-Speed Automatic", "fuel_type": "Gasoline", "city_mpg": 24, "highway_mpg": 30, "combined_mpg": 27},
    {"make": "Mazda", "model": "MX-30", "year": 2023, "engine_size": 0.0, "cylinders": 0, "transmission": "Single-Speed", "fuel_type": "Electric", "city_mpg": 105, "highway_mpg": 92, "combined_mpg": 100},
]

def populate_comprehensive_database():
    """Populate database with comprehensive vehicle data"""
    print("🚗 Populating comprehensive vehicle database...")
    
    with app.app_context():
        # Clear existing data
        Vehicle.query.delete()
        db.session.commit()
        
        # Add comprehensive vehicle data
        for vehicle_data in COMPREHENSIVE_VEHICLE_DATA:
            vehicle = Vehicle(
                make=vehicle_data["make"],
                model=vehicle_data["model"],
                year=vehicle_data["year"],
                engine_size=vehicle_data["engine_size"],
                cylinders=vehicle_data["cylinders"],
                transmission=vehicle_data["transmission"],
                fuel_type=vehicle_data["fuel_type"],
                city_mpg=vehicle_data["city_mpg"],
                highway_mpg=vehicle_data["highway_mpg"],
                combined_mpg=vehicle_data["combined_mpg"],
                co2_emissions=calculate_co2_emissions(vehicle_data["combined_mpg"])
            )
            db.session.add(vehicle)
        
        db.session.commit()
        
        total_vehicles = Vehicle.query.count()
        print(f"✅ Added {total_vehicles} vehicles to database")
        
        # Show breakdown by make
        makes = db.session.query(Vehicle.make, db.func.count(Vehicle.id)).group_by(Vehicle.make).all()
        print("\n📊 Vehicles by manufacturer:")
        for make, count in makes:
            print(f"  {make}: {count} models")
        
        return total_vehicles

def calculate_co2_emissions(mpg):
    """Calculate CO2 emissions based on MPG"""
    if mpg <= 0:
        return 0
    # 19.6 lbs CO2 per gallon, 12,000 miles per year
    return (19.6 * 12000) / mpg

def generate_expanded_training_data():
    """Generate expanded training dataset with variations"""
    print("🔄 Generating expanded training data...")
    
    base_data = []
    
    with app.app_context():
        vehicles = Vehicle.query.all()
        
        for vehicle in vehicles:
            # Add base vehicle data
            base_data.append({
                'make': vehicle.make,
                'model': vehicle.model,
                'year': vehicle.year,
                'engine_size': vehicle.engine_size,
                'cylinders': vehicle.cylinders,
                'transmission': vehicle.transmission,
                'fuel_type': vehicle.fuel_type,
                'combined_mpg': vehicle.combined_mpg
            })
            
            # Add variations for different years (if newer than 2020)
            if vehicle.year >= 2020:
                for year_offset in [-1, -2, -3]:
                    new_year = vehicle.year + year_offset
                    if new_year >= 2010:
                        # Slightly adjust MPG for older years (typically less efficient)
                        mpg_adjustment = year_offset * 0.5  # 0.5 MPG decrease per year back
                        new_mpg = max(10, vehicle.combined_mpg + mpg_adjustment)
                        
                        base_data.append({
                            'make': vehicle.make,
                            'model': vehicle.model,
                            'year': new_year,
                            'engine_size': vehicle.engine_size,
                            'cylinders': vehicle.cylinders,
                            'transmission': vehicle.transmission,
                            'fuel_type': vehicle.fuel_type,
                            'combined_mpg': new_mpg
                        })
    
    # Add synthetic variations
    np.random.seed(42)
    synthetic_data = []
    
    for base_vehicle in base_data[:50]:  # Take first 50 for variations
        for _ in range(5):  # 5 variations per vehicle
            # Add slight variations
            new_vehicle = base_vehicle.copy()
            
            # Add noise to continuous variables
            new_vehicle['engine_size'] += np.random.normal(0, 0.1)
            new_vehicle['engine_size'] = max(1.0, min(8.0, new_vehicle['engine_size']))
            
            # Adjust MPG based on variations
            mpg_noise = np.random.normal(0, 2)  # ±2 MPG variation
            new_vehicle['combined_mpg'] += mpg_noise
            new_vehicle['combined_mpg'] = max(10, min(150, new_vehicle['combined_mpg']))
            
            synthetic_data.append(new_vehicle)
    
    all_data = base_data + synthetic_data
    
    print(f"📈 Generated {len(all_data)} training samples")
    print(f"  • Base vehicles: {len(base_data)}")
    print(f"  • Synthetic variations: {len(synthetic_data)}")
    
    return pd.DataFrame(all_data)

def analyze_dataset():
    """Analyze the current dataset"""
    print("📊 Analyzing dataset...")
    
    with app.app_context():
        total_vehicles = Vehicle.query.count()
        print(f"Total vehicles: {total_vehicles}")
        
        # Efficiency ranges
        ranges = [
            ("Ultra High Efficiency (50+ MPG)", Vehicle.query.filter(Vehicle.combined_mpg >= 50).count()),
            ("High Efficiency (35-49 MPG)", Vehicle.query.filter(Vehicle.combined_mpg.between(35, 49)).count()),
            ("Good Efficiency (25-34 MPG)", Vehicle.query.filter(Vehicle.combined_mpg.between(25, 34)).count()),
            ("Average Efficiency (20-24 MPG)", Vehicle.query.filter(Vehicle.combined_mpg.between(20, 24)).count()),
            ("Low Efficiency (<20 MPG)", Vehicle.query.filter(Vehicle.combined_mpg < 20).count())
        ]
        
        print("\n🎯 Efficiency Distribution:")
        for range_name, count in ranges:
            percentage = (count / total_vehicles) * 100 if total_vehicles > 0 else 0
            print(f"  {range_name}: {count} ({percentage:.1f}%)")
        
        # Fuel type distribution
        fuel_types = db.session.query(Vehicle.fuel_type, db.func.count(Vehicle.id)).group_by(Vehicle.fuel_type).all()
        print("\n⛽ Fuel Type Distribution:")
        for fuel_type, count in fuel_types:
            percentage = (count / total_vehicles) * 100 if total_vehicles > 0 else 0
            print(f"  {fuel_type}: {count} ({percentage:.1f}%)")
        
        # Top and bottom performers
        print("\n🏆 Top 5 Most Efficient:")
        top_efficient = Vehicle.query.order_by(Vehicle.combined_mpg.desc()).limit(5).all()
        for vehicle in top_efficient:
            print(f"  {vehicle.make} {vehicle.model} ({vehicle.year}) - {vehicle.combined_mpg} MPG")
        
        print("\n⚠️ Bottom 5 Least Efficient:")
        bottom_efficient = Vehicle.query.order_by(Vehicle.combined_mpg.asc()).limit(5).all()
        for vehicle in bottom_efficient:
            print(f"  {vehicle.make} {vehicle.model} ({vehicle.year}) - {vehicle.combined_mpg} MPG")

def save_training_data_csv():
    """Save training data to CSV for analysis"""
    print("💾 Saving training data to CSV...")
    
    training_df = generate_expanded_training_data()
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    csv_path = 'data/comprehensive_vehicle_training_data.csv'
    training_df.to_csv(csv_path, index=False)
    
    print(f"✅ Training data saved to {csv_path}")
    print(f"📊 Dataset shape: {training_df.shape}")
    
    # Show sample data
    print("\n📋 Sample training data:")
    print(training_df.head(10).to_string())
    
    return csv_path

def check_all_vehicle_efficiency():
    """Check efficiency of all vehicles in database"""
    print("🔍 Checking all vehicle efficiency...")
    
    with app.app_context():
        # Get all vehicles ordered by efficiency
        all_vehicles = Vehicle.query.order_by(Vehicle.combined_mpg.desc()).all()
        
        print(f"\n📊 All {len(all_vehicles)} vehicles by efficiency:")
        print("="*80)
        
        for i, vehicle in enumerate(all_vehicles, 1):
            efficiency_category = get_efficiency_category(vehicle.combined_mpg)
            print(f"{i:3d}. {vehicle.make:12} {vehicle.model:15} ({vehicle.year}) - {vehicle.combined_mpg:5.1f} MPG [{efficiency_category}] {vehicle.fuel_type}")
        
        return all_vehicles

def get_efficiency_category(mpg):
    """Categorize vehicle efficiency"""
    if mpg >= 50:
        return "🟢 ULTRA"
    elif mpg >= 35:
        return "🟡 HIGH "
    elif mpg >= 25:
        return "🟠 GOOD "
    elif mpg >= 20:
        return "🔴 AVG  "
    else:
        return "⚫ LOW  "

def create_vehicle_search_api():
    """Create advanced vehicle search functionality"""
    print("🔍 Creating vehicle search functionality...")
    
    def search_vehicles_by_criteria(make=None, fuel_type=None, min_mpg=None, max_mpg=None, year_range=None):
        """Advanced vehicle search"""
        with app.app_context():
            query = Vehicle.query
            
            if make:
                query = query.filter(Vehicle.make.ilike(f'%{make}%'))
            
            if fuel_type:
                query = query.filter(Vehicle.fuel_type.ilike(f'%{fuel_type}%'))
            
            if min_mpg:
                query = query.filter(Vehicle.combined_mpg >= min_mpg)
            
            if max_mpg:
                query = query.filter(Vehicle.combined_mpg <= max_mpg)
            
            if year_range:
                start_year, end_year = year_range
                query = query.filter(Vehicle.year.between(start_year, end_year))
            
            results = query.order_by(Vehicle.combined_mpg.desc()).all()
            
            return results
    
    # Example searches
    print("\n🔍 Example searches:")
    
    # Most efficient vehicles
    efficient_vehicles = search_vehicles_by_criteria(min_mpg=40)
    print(f"\n🟢 Most efficient vehicles (40+ MPG): {len(efficient_vehicles)}")
    for vehicle in efficient_vehicles[:10]:
        print(f"  {vehicle.make} {vehicle.model} - {vehicle.combined_mpg} MPG")
    
    # Electric vehicles
    electric_vehicles = search_vehicles_by_criteria(fuel_type="Electric")
    print(f"\n⚡ Electric vehicles: {len(electric_vehicles)}")
    for vehicle in electric_vehicles:
        print(f"  {vehicle.make} {vehicle.model} - {vehicle.combined_mpg} MPGe")
    
    # Toyota vehicles
    toyota_vehicles = search_vehicles_by_criteria(make="Toyota")
    print(f"\n🚗 Toyota vehicles: {len(toyota_vehicles)}")
    for vehicle in toyota_vehicles:
        print(f"  {vehicle.model} ({vehicle.year}) - {vehicle.combined_mpg} MPG")
    
    # Recent vehicles (2022+)
    recent_vehicles = search_vehicles_by_criteria(year_range=(2022, 2024))
    print(f"\n📅 Recent vehicles (2022+): {len(recent_vehicles)}")
    for vehicle in recent_vehicles[:15]:
        print(f"  {vehicle.make} {vehicle.model} ({vehicle.year}) - {vehicle.combined_mpg} MPG")

if __name__ == "__main__":
    print("🚀 Comprehensive Vehicle Database Setup")
    print("="*50)
    
    # Step 1: Populate database
    total_vehicles = populate_comprehensive_database()
    
    # Step 2: Analyze dataset
    analyze_dataset()
    
    # Step 3: Save training data
    csv_path = save_training_data_csv()
    
    # Step 4: Check all vehicles
    all_vehicles = check_all_vehicle_efficiency()
    
    # Step 5: Demo search functionality
    create_vehicle_search_api()
    
    print(f"\n✅ Setup complete!")
    print(f"📊 Total vehicles in database: {total_vehicles}")
    print(f"💾 Training data saved to: {csv_path}")
    print(f"🔍 Use search functions to find specific vehicles")
    
    # Instructions for next steps
    print(f"\n📋 Next steps:")
    print(f"1. Run: python train_models.py  # Re-train with new data")
    print(f"2. Run: python app.py          # Start the backend")
    print(f"3. Test predictions with diverse vehicle types")