import pandas as pd
import numpy as np

def diagnose_csv(csv_path):
    """Diagnose CSV file issues before processing"""
    print("🔍 CSV File Diagnostic Tool")
    print("=" * 50)
    
    try:
        # Try to load the CSV
        print(f"📁 Loading: {csv_path}")
        
        # Try different encodings
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        df = None
        
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                print(f"✅ Loaded successfully with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue
        
        if df is None:
            print("❌ Could not load CSV with any encoding")
            return False
        
        # Basic info
        print(f"\n📊 Basic Information:")
        print(f"  Shape: {df.shape}")
        print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        
        # Show all columns
        print(f"\n📋 All Columns ({len(df.columns)}):")
        for i, col in enumerate(df.columns, 1):
            print(f"  {i:2d}. {col}")
        
        # Check key columns we need
        key_columns = {
            'Make': 'Vehicle Manufacturer',
            'Modle': 'Vehicle Model (note: typo in CSV)', 
            'Year_from': 'Start Year',
            'Year_to': 'End Year',
            'capacity_cm3': 'Engine Size (cm³)',
            'number_of_cylinders': 'Number of Cylinders',
            'mixed_fuel_consumption_per_100_km_l': 'Fuel Consumption (L/100km)',
            'city_fuel_per_100km_l': 'City Fuel Consumption',
            'highway_fuel_per_100km_l': 'Highway Fuel Consumption',
            'transmission': 'Transmission Type'
        }
        
        print(f"\n🔍 Key Column Analysis:")
        for col, desc in key_columns.items():
            if col in df.columns:
                print(f"  ✅ {col} ({desc})")
                
                # Analyze the column
                series = df[col]
                print(f"      Data type: {series.dtype}")
                print(f"      Non-null count: {series.notna().sum():,} / {len(series):,}")
                print(f"      Unique values: {series.nunique():,}")
                
                # Show sample values
                sample_values = series.dropna().head(5).tolist()
                print(f"      Sample values: {sample_values}")
                
                # Check for mixed types in supposedly numeric columns
                if col in ['Year_from', 'Year_to', 'capacity_cm3', 'number_of_cylinders', 
                          'mixed_fuel_consumption_per_100_km_l']:
                    # Try converting to numeric
                    numeric_series = pd.to_numeric(series, errors='coerce')
                    conversion_issues = series.notna().sum() - numeric_series.notna().sum()
                    
                    if conversion_issues > 0:
                        print(f"      ⚠️ {conversion_issues:,} values cannot be converted to numeric")
                        
                        # Show problematic values
                        problematic = series[series.notna() & numeric_series.isna()].head(5)
                        if len(problematic) > 0:
                            print(f"      Problem values: {problematic.tolist()}")
                    else:
                        if numeric_series.notna().sum() > 0:
                            print(f"      Range: {numeric_series.min()} - {numeric_series.max()}")
                
                print()  # Empty line for readability
            else:
                print(f"  ❌ {col} ({desc}) - NOT FOUND")
        
        # Check for obvious data quality issues
        print(f"\n⚠️ Data Quality Check:")
        
        # Check Make column
        if 'Make' in df.columns:
            makes = df['Make'].value_counts()
            print(f"  Manufacturers: {len(makes)} unique")
            print(f"  Top 5: {makes.head().to_dict()}")
            
            # Check for unusual values
            unusual_makes = makes[makes.index.str.len() < 2]
            if len(unusual_makes) > 0:
                print(f"  ⚠️ Unusual make names (too short): {unusual_makes.to_dict()}")
        
        # Check fuel consumption
        if 'mixed_fuel_consumption_per_100_km_l' in df.columns:
            fuel_data = pd.to_numeric(df['mixed_fuel_consumption_per_100_km_l'], errors='coerce')
            valid_fuel = fuel_data.dropna()
            
            if len(valid_fuel) > 0:
                print(f"  Fuel consumption (L/100km): {valid_fuel.min():.1f} - {valid_fuel.max():.1f}")
                
                # Check for unrealistic values
                unrealistic = valid_fuel[(valid_fuel < 3) | (valid_fuel > 50)]
                if len(unrealistic) > 0:
                    print(f"  ⚠️ {len(unrealistic)} unrealistic fuel consumption values")
            else:
                print(f"  ❌ No valid fuel consumption data found!")
        
        # Check engine sizes
        if 'capacity_cm3' in df.columns:
            engine_data = pd.to_numeric(df['capacity_cm3'], errors='coerce')
            valid_engines = engine_data.dropna()
            
            if len(valid_engines) > 0:
                print(f"  Engine sizes (cm³): {int(valid_engines.min())} - {int(valid_engines.max())}")
                
                # Convert to liters for display
                engines_liters = valid_engines / 1000
                print(f"  Engine sizes (L): {engines_liters.min():.1f}L - {engines_liters.max():.1f}L")
                
                # Check for unrealistic values
                unrealistic = valid_engines[(valid_engines < 500) | (valid_engines > 10000)]
                if len(unrealistic) > 0:
                    print(f"  ⚠️ {len(unrealistic)} unrealistic engine sizes")
            else:
                print(f"  ❌ No valid engine size data found!")
        
        # Estimate processable rows
        processable = df.copy()
        
        # Must have make and model
        if 'Make' in processable.columns and 'Modle' in processable.columns:
            processable = processable[processable['Make'].notna() & processable['Modle'].notna()]
        
        # Must have valid fuel consumption
        if 'mixed_fuel_consumption_per_100_km_l' in processable.columns:
            fuel_numeric = pd.to_numeric(processable['mixed_fuel_consumption_per_100_km_l'], errors='coerce')
            processable = processable[fuel_numeric.between(3, 50, inclusive='both')]
        
        # Must have valid engine size
        if 'capacity_cm3' in processable.columns:
            engine_numeric = pd.to_numeric(processable['capacity_cm3'], errors='coerce')
            processable = processable[engine_numeric.between(500, 10000, inclusive='both')]
        
        # Must have reasonable year
        if 'Year_from' in processable.columns:
            year_numeric = pd.to_numeric(processable['Year_from'], errors='coerce')
            processable = processable[year_numeric.between(1990, 2024, inclusive='both')]
        
        print(f"\n📈 Processing Estimate:")
        print(f"  Original rows: {len(df):,}")
        print(f"  Potentially processable: {len(processable):,}")
        print(f"  Success rate: {len(processable)/len(df)*100:.1f}%")
        
        if len(processable) < 1000:
            print(f"  ⚠️ Low processable count - check data quality")
        elif len(processable) > 10000:
            print(f"  ✅ Good amount of processable data")
        else:
            print(f"  ✅ Reasonable amount of processable data")
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing CSV: {e}")
        return False

if __name__ == "__main__":
    csv_path = "vehicles.csv"
    
    if not csv_path:
        csv_path = "vehicles.csv"
    
    success = diagnose_csv(csv_path)
    
    if success:
        print(f"\n✅ Diagnostic complete!")
        print(f"\nNext step: Run 'python csv_data_processor.py' to process the data.")
    else:
        print(f"\n❌ Diagnostic failed. Please check your CSV file.")