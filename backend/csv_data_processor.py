# import pandas as pd
# import numpy as np
# from app import app, db, Vehicle
# import os
# import json
# from datetime import datetime

# class KaggleCSVProcessor:
#     """Process Kaggle vehicle CSV dataset for fuel efficiency prediction"""
    
#     def __init__(self):
#         self.df = None
#         self.processed_data = []
#         self.stats = {}
    
#     def load_csv_file(self, csv_path):
#         """Load CSV file and display basic information"""
#         print(f"📁 Loading CSV file: {csv_path}")
        
#         if not os.path.exists(csv_path):
#             print(f"❌ File not found: {csv_path}")
#             return False
        
#         try:
#             # Try different encodings
#             encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
#             for encoding in encodings:
#                 try:
#                     self.df = pd.read_csv(csv_path, encoding=encoding)
#                     print(f"✅ Successfully loaded with {encoding} encoding")
#                     break
#                 except UnicodeDecodeError:
#                     continue
            
#             if self.df is None:
#                 print("❌ Could not read CSV file with any encoding")
#                 return False
            
#             print(f"📊 Dataset shape: {self.df.shape}")
#             print(f"📋 Key columns found:")
            
#             # Check for our required columns
#             required_columns = {
#                 'Make': 'Make',
#                 'Model': 'Modle',  # Note: typo in original
#                 'Year': ['Year_from', 'Year_to'], 
#                 'Engine Size': 'capacity_cm3',
                
            
#             }
            
#             for desc, col in required_columns.items():
#                 if isinstance(col, list):
#                     available = [c for c in col if c in self.df.columns]
#                     if available:
#                         print(f"  ✅ {desc}: {available}")
#                     else:
#                         print(f"  ❌ {desc}: Not found")
#                 elif col in self.df.columns:
#                     print(f"  ✅ {desc}: {col}")
#                 else:
#                     print(f"  ❌ {desc}: Not found")
            
#             # Display first few rows
#             print("\n🔍 Sample data:")
#             sample_cols = ['Make', 'Modle', 'Year_from', 'capacity_cm3', 'mixed_fuel_consumption_per_100_km_l']
#             available_cols = [col for col in sample_cols if col in self.df.columns]
#             if available_cols:
#                 print(self.df[available_cols].head())
            
#             return True
            
#         except Exception as e:
#             print(f"❌ Error loading CSV: {e}")
#             return False
    
#     def analyze_dataset(self):
#         """Analyze the dataset structure and content"""
#         if self.df is None:
#             print("❌ No data loaded")
#             return
        
#         print("\n📊 Dataset Analysis:")
#         print("=" * 50)
        
#         # Basic info
#         print(f"Total rows: {len(self.df):,}")
#         print(f"Total columns: {len(self.df.columns)}")
        
#         # Analyze key columns
#         if 'Make' in self.df.columns:
#             unique_makes = self.df['Make'].nunique()
#             print(f"\n🏭 Manufacturers ({unique_makes}):")
#             makes = self.df['Make'].value_counts().head(15)
#             for make, count in makes.items():
#                 print(f"  {make}: {count:,}")
        
#         if 'Year_from' in self.df.columns:
#             # Clean year data first
#             year_data = pd.to_numeric(self.df['Year_from'], errors='coerce').dropna()
#             if len(year_data) > 0:
#                 year_range = f"{int(year_data.min())} - {int(year_data.max())}"
#                 print(f"\n📅 Year range: {year_range}")
#             else:
#                 print(f"\n📅 Year range: No valid year data found")
        
#         if 'capacity_cm3' in self.df.columns:
#             # Clean engine size data first
#             engines = pd.to_numeric(self.df['capacity_cm3'], errors='coerce').dropna()
#             if len(engines) > 0:
#                 print(f"\n🔧 Engine sizes (cm³): {int(engines.min())} - {int(engines.max())}")
#                 print(f"  Valid engine entries: {len(engines):,} out of {len(self.df):,}")
#             else:
#                 print(f"\n🔧 Engine sizes: No valid engine data found")
        
#         # Check fuel consumption data
#         if 'mixed_fuel_consumption_per_100_km_l' in self.df.columns:
#             fuel_data = pd.to_numeric(self.df['mixed_fuel_consumption_per_100_km_l'], errors='coerce').dropna()
#             if len(fuel_data) > 0:
#                 print(f"\n⛽ Fuel consumption (L/100km): {fuel_data.min():.1f} - {fuel_data.max():.1f}")
#                 print(f"  Valid fuel entries: {len(fuel_data):,} out of {len(self.df):,}")
#             else:
#                 print(f"\n⛽ Fuel consumption: No valid fuel data found")
        
#         # Check for missing values in key columns
#         key_columns = ['Make', 'Modle', 'Year_from', 'capacity_cm3', 'mixed_fuel_consumption_per_100_km_l']
#         existing_columns = [col for col in key_columns if col in self.df.columns]
        
#         print(f"\n⚠️ Missing Values in Key Columns:")
#         for col in existing_columns:
#             missing = self.df[col].isnull().sum()
#             percentage = (missing / len(self.df)) * 100
            
#             # Also check for non-numeric data in numeric columns
#             if col in ['capacity_cm3', 'mixed_fuel_consumption_per_100_km_l', 'Year_from']:
#                 numeric_data = pd.to_numeric(self.df[col], errors='coerce')
#                 non_numeric = numeric_data.isnull().sum() - missing  # Additional nulls from conversion
#                 if non_numeric > 0:
#                     print(f"  {col}: {missing:,} missing + {non_numeric:,} non-numeric ({percentage:.1f}%)")
#                 else:
#                     print(f"  {col}: {missing:,} ({percentage:.1f}%)")
#             else:
#                 print(f"  {col}: {missing:,} ({percentage:.1f}%)")
        
#         # Show sample of problematic data
#         if 'capacity_cm3' in self.df.columns:
#             print(f"\n🔍 Sample engine size data:")
#             sample_engines = self.df['capacity_cm3'].dropna().head(10)
#             for i, value in enumerate(sample_engines):
#                 value_type = type(value).__name__
#                 print(f"  {i+1}. '{value}' (type: {value_type})")
        
#         # Check data types
#         print(f"\n📋 Data Types for Key Columns:")
#         for col in existing_columns:
#             if col in self.df.columns:
#                 dtype = self.df[col].dtype
#                 print(f"  {col}: {dtype}")
    
#     def _safe_float(self, row, column, default):
#         """Safely extract float value with better error handling"""
#         try:
#             if column not in row:
#                 return default
            
#             value = row[column]
            
#             # Handle None/NaN values
#             if pd.isna(value):
#                 return default
            
#             # Convert to numeric, handling strings
#             numeric_value = pd.to_numeric(value, errors='coerce')
            
#             if pd.isna(numeric_value):
#                 return default
            
#             return float(numeric_value) if numeric_value > 0 else default
            
#         except Exception as e:
#             print(f"  Warning: Error processing {column} value '{value}': {e}")
#             return default
    
#     def _safe_int(self, row, column, default):
#         """Safely extract int value with better error handling"""
#         try:
#             if column not in row:
#                 return default
            
#             value = row[column]
            
#             # Handle None/NaN values
#             if pd.isna(value):
#                 return default
            
#             # Convert to numeric, handling strings
#             numeric_value = pd.to_numeric(value, errors='coerce')
            
#             if pd.isna(numeric_value):
#                 return default
            
#             return int(numeric_value) if numeric_value > 0 else default
            
#         except Exception as e:
#             print(f"  Warning: Error processing {column} value '{value}': {e}")
#             return default
#     ############################################################################################
    
    
    
#     # def clean_and_process_data(self):
#     #     """Clean and process the data for training"""
#     #     if self.df is None:
#     #         print("❌ No data loaded")
#     #         return False
        
#     #     print("\n🧹 Cleaning and processing data...")
        
#     #     # Create a copy for processing
#     #     processed_df = self.df.copy()
        
#     #     # Clean numeric columns first
#     #     print("🔧 Cleaning numeric columns...")
#     #     numeric_columns = ['Year_from', 'Year_to', 'capacity_cm3', 'number_of_cylinders', 
#     #                       'mixed_fuel_consumption_per_100_km_l', 'city_fuel_per_100km_l', 
#     #                       'highway_fuel_per_100km_l', 'curb_weight_kg', 'engine_hp']
        
#     #     for col in numeric_columns:
#     #         if col in processed_df.columns:
#     #             # Convert to numeric, replacing non-numeric with NaN
#     #             original_count = len(processed_df[col].dropna())
#     #             processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
#     #             new_count = len(processed_df[col].dropna())
                
#     #             if original_count != new_count:
#     #                 print(f"  {col}: Cleaned {original_count - new_count:,} non-numeric values")
        
#     #     # Filter out rows with missing critical data
#     #     critical_columns = ['Make', 'Modle']
#     #     for col in critical_columns:
#     #         if col in processed_df.columns:
#     #             before_count = len(processed_df)
#     #             processed_df = processed_df[processed_df[col].notna()]
#     #             after_count = len(processed_df)
#     #             if before_count != after_count:
#     #                 print(f"🗑️ Removed {before_count - after_count:,} rows missing {col}")
        
#     #     # Filter by year if available
#     #     if 'Year_from' in processed_df.columns:
#     #         print(f"📅 Filtering years 1990-2024...")
#     #         before_count = len(processed_df)
#     #         processed_df = processed_df[
#     #             (processed_df['Year_from'] >= 1990) & 
#     #             (processed_df['Year_from'] <= 2024) &
#     #             (processed_df['Year_from'].notna())
#     #         ]
#     #         after_count = len(processed_df)
#     #         print(f"  Remaining rows after year filter: {after_count:,} (removed {before_count - after_count:,})")
        
#     #     # Filter for valid fuel consumption data
#     #     if 'mixed_fuel_consumption_per_100_km_l' in processed_df.columns:
#     #         print(f"⛽ Filtering valid fuel consumption data...")
#     #         before_count = len(processed_df)
#     #         processed_df = processed_df[
#     #             (processed_df['mixed_fuel_consumption_per_100_km_l'] > 0) &
#     #             (processed_df['mixed_fuel_consumption_per_100_km_l'] < 50) &  # Reasonable range
#     #             (processed_df['mixed_fuel_consumption_per_100_km_l'].notna())
#     #         ]
#     #         after_count = len(processed_df)
#     #         print(f"  Remaining rows after fuel filter: {after_count:,} (removed {before_count - after_count:,})")
        
#     #     # Filter for valid engine size data
#     #     if 'capacity_cm3' in processed_df.columns:
#     #         print(f"🔧 Filtering valid engine size data...")
#     #         before_count = len(processed_df)
#     #         processed_df = processed_df[
#     #             (processed_df['capacity_cm3'] >= 500) &  # Minimum 0.5L
#     #             (processed_df['capacity_cm3'] <= 10000) &  # Maximum 10L
#     #             (processed_df['capacity_cm3'].notna())
#     #         ]
#     #         after_count = len(processed_df)
#     #         print(f"  Remaining rows after engine filter: {after_count:,} (removed {before_count - after_count:,})")
        
#     #     if len(processed_df) == 0:
#     #         print("❌ No data remaining after filtering!")
#     #         return False
        
#     #     # Process each row
#     #     self.processed_data = []
#     #     processed_count = 0
#     #     error_count = 0
        
#     #     print("🔄 Processing vehicle data...")
        
#     #     for index, row in processed_df.iterrows():
#     #         try:
#     #             vehicle_data = self._extract_vehicle_data(row)
#     #             if vehicle_data:
#     #                 self.processed_data.append(vehicle_data)
#     #                 processed_count += 1
                    
#     #                 if processed_count % 1000 == 0:
#     #                     print(f"  Processed {processed_count:,} vehicles...")
#     #             else:
#     #                 error_count += 1
                        
#     #         except Exception as e:
#     #             error_count += 1
#     #             if error_count < 10:  # Show first 10 errors
#     #                 print(f"  Warning: Error processing row {index}: {e}")
#     #             continue
        
#     #     if error_count > 10:
#     #         print(f"  ... and {error_count - 10} more errors")
        
#     #     print(f"✅ Successfully processed {len(self.processed_data):,} vehicles")
#     #     print(f"⚠️ Skipped {error_count:,} problematic rows")
        
#     #     # Generate statistics
#     #     self._generate_statistics()
        
#     #     return len(self.processed_data) > 0
    
#     # def _extract_vehicle_data(self, row):
#     #     """Extract and clean vehicle data from a row"""
#     #     try:
#     #         # Extract basic info - handle missing Make/Model
#     #         make = str(row.get('Make', 'Unknown')).strip()
#     #         model = str(row.get('Modle', 'Unknown')).strip()  # Note: typo in CSV
            
#     #         if make in ['Unknown', 'nan', ''] or model in ['Unknown', 'nan', '']:
#     #             return None
            
#     #         # Clean make and model names
#     #         make = self._clean_text(make)
#     #         model = self._clean_text(model)
            
#     #         if not make or not model:
#     #             return None
            
#     #         # Extract year - use Year_from as primary, fallback to Year_to
#     #         year = self._safe_int(row, 'Year_from', None)
#     #         if not year or year < 1990 or year > 2024:
#     #             year = self._safe_int(row, 'Year_to', None)
#     #         if not year or year < 1990 or year > 2024:
#     #             return None
            
#     #         # Extract engine size (convert cm³ to liters)
#     #         engine_capacity_cm3 = self._safe_float(row, 'capacity_cm3', None)
#     #         if engine_capacity_cm3 and engine_capacity_cm3 > 0:
#     #             engine_size = round(engine_capacity_cm3 / 1000, 1)  # Convert to liters
#     #             if engine_size < 0.5 or engine_size > 10.0:  # Reasonable range
#     #                 engine_size = 2.0  # Default
#     #         else:
#     #             engine_size = 2.0  # Default
            
#     #         # Extract cylinders
#     #         cylinders = self._safe_int(row, 'number_of_cylinders', 4)
#     #         if cylinders < 1 or cylinders > 16:
#     #             cylinders = 4
            
#     #         # Calculate MPG from fuel consumption (L/100km to MPG)
#     #         combined_mpg = self._calculate_mpg_from_consumption(row, 'mixed_fuel_consumption_per_100_km_l')
#     #         city_mpg = self._calculate_mpg_from_consumption(row, 'city_fuel_per_100km_l')
#     #         highway_mpg = self._calculate_mpg_from_consumption(row, 'highway_fuel_per_100km_l')
            
#     #         # If we don't have combined MPG, skip this vehicle
#     #         if not combined_mpg or combined_mpg <= 0:
#     #             return None
            
#     #         # Estimate city/highway if missing
#     #         if not city_mpg:
#     #             city_mpg = combined_mpg * 0.85  # Typically lower
#     #         if not highway_mpg:
#     #             highway_mpg = combined_mpg * 1.15  # Typically higher
            
#     #         # Extract transmission
#     #         transmission = self._clean_transmission(row)
            
#     #         # Determine fuel type based on available data
#     #         fuel_type = self._determine_fuel_type(row)
            
#     #         vehicle_data = {
#     #             'make': make,
#     #             'model': model,
#     #             'year': year,
#     #             'engine_size': engine_size,
               
#     #         }
            
#     #         return vehicle_data
            
#     #     except Exception as e:
#     #         return None
    
#     # def _clean_text(self, text):
#     #     """Clean and standardize text"""
#     #     if pd.isna(text) or str(text).lower() in ['nan', 'none', '', 'unknown']:
#     #         return None
        
#     #     text = str(text).strip()
#     #     # Remove extra spaces and standardize
#     #     text = ' '.join(text.split())
        
#     #     # Capitalize properly
#     #     if text:
#     #         text = text.title()
            
#     #     return text if len(text) > 1 else None
    
#     # def _safe_float(self, row, column, default):
#     #     """Safely extract float value"""
#     #     try:
#     #         if column in row:
#     #             value = pd.to_numeric(row[column], errors='coerce')
#     #             return float(value) if not pd.isna(value) and value > 0 else default
#     #         return default
#     #     except:
#     #         return default
    
#     # def _safe_int(self, row, column, default):
#     #     """Safely extract int value"""
#     #     try:
#     #         if column in row:
#     #             value = pd.to_numeric(row[column], errors='coerce')
#     #             return int(value) if not pd.isna(value) and value > 0 else default
#     #         return default
#     #     except:
#     #         return default
    
#     # def _calculate_mpg_from_consumption(self, row, column):
#     #     """Convert L/100km to MPG"""
#     #     try:
#     #         consumption = self._safe_float(row, column, None)
#     #         if consumption and consumption > 0:
#     #             # Formula: MPG = 235.214 / (L/100km)
#     #             mpg = 235.214 / consumption
#     #             return mpg if 5 <= mpg <= 150 else None
#     #         return None
#     #     except:
#     #         return None
    
#     # def _clean_transmission(self, row):
#     #     """Clean and standardize transmission data"""
#     #     transmission_col = 'transmission'
        
#     #     if transmission_col not in row or pd.isna(row[transmission_col]):
#     #         return 'Automatic'  # Default
        
#     #     try:
#     #         trans = str(row[transmission_col]).lower()
            
#     #         if any(word in trans for word in ['manual', 'man', 'mt', 'm6', 'm5', 'stick']):
#     #             return 'Manual'
#     #         elif any(word in trans for word in ['cvt', 'variable', 'continuously']):
#     #             return 'CVT'
#     #         elif any(word in trans for word in ['auto', 'automatic', 'at', 'a6', 'a8', 'a4', 'tiptronic']):
#     #             return 'Automatic'
#     #         else:
#     #             return 'Automatic'  # Default for unknown
#     #     except:
#     #         return 'Automatic'
    
#     # def _determine_fuel_type(self, row):
#     #     """Determine fuel type from available data"""
#     #     # Check for electric indicators
#     #     if ('battery_capacity_KW_per_h' in row and 
#     #         self._safe_float(row, 'battery_capacity_KW_per_h', 0) > 0):
#     #         return 'Electric'
        
#     #     # Check engine type if available
#     #     if 'engine_type' in row and not pd.isna(row['engine_type']):
#     #         engine_type = str(row['engine_type']).lower()
#     #         if 'electric' in engine_type or 'ev' in engine_type:
#     #             return 'Electric'
#     #         elif 'hybrid' in engine_type or 'hev' in engine_type:
#     #             return 'Hybrid'
#     #         elif 'diesel' in engine_type:
#     #             return 'Diesel'
        
#     #     # Check fuel grade
#     #     if 'fuel_grade' in row and not pd.isna(row['fuel_grade']):
#     #         fuel_grade = str(row['fuel_grade']).lower()
#     #         if 'diesel' in fuel_grade:
#     #             return 'Diesel'
#     #         elif 'electric' in fuel_grade:
#     #             return 'Electric'
        
#     #     # Default based on engine size and year
#     #     engine_size = self._safe_float(row, 'capacity_cm3', 2000)
#     #     year = self._safe_int(row, 'Year_from', 2000)
        
#     #     # Very small engines in recent years might be hybrid
#     #     if engine_size < 1500 and year > 2010:
#     #         return 'Hybrid'
#     #     elif engine_size > 3000:
#     #         return 'Gasoline'  # Larger engines typically gasoline
#     #     else:
#     #         return 'Gasoline'  # Default
    
#     # def _calculate_co2_emissions(self, mpg):
#     #     """Calculate CO2 emissions based on MPG"""
#     #     if mpg <= 0:
#     #         return 500
#     #     # Formula: CO2 (g/mile) = 19.6 lbs/gallon * 453.592 g/lb / mpg
#     #     return (19.6 * 453.592) / mpg
    
#     # def _generate_statistics(self):
#     #     """Generate statistics about the processed data"""
#     #     if not self.processed_data:
#     #         return
        
#     #     df_stats = pd.DataFrame(self.processed_data)
        
#     #     self.stats = {
#     #         'total_vehicles': len(self.processed_data),
#     #         'unique_makes': df_stats['make'].nunique(),
#     #         'unique_models': df_stats['model'].nunique(),
#     #         'year_range': (int(df_stats['year'].min()), int(df_stats['year'].max())),
#     #         'mpg_range': (round(df_stats['combined_mpg'].min(), 1), round(df_stats['combined_mpg'].max(), 1)),
#     #         'avg_mpg': round(df_stats['combined_mpg'].mean(), 1),
#     #         'engine_size_range': (round(df_stats['engine_size'].min(), 1), round(df_stats['engine_size'].max(), 1)),
#     #         'makes_distribution': df_stats['make'].value_counts().head(15).to_dict(),
#     #         'fuel_types': df_stats['fuel_type'].value_counts().to_dict(),
#     #         'transmission_types': df_stats['transmission'].value_counts().to_dict(),
#     #         'year_distribution': df_stats['year'].value_counts().sort_index().tail(10).to_dict()
#     #     }
    
#     # def save_to_database(self):
#     #     """Save processed data to database"""
#     #     if not self.processed_data:
#     #         print("❌ No processed data to save")
#     #         return False
        
#     #     print("💾 Saving data to database...")
        
#     #     with app.app_context():
#     #         # Clear existing data
#     #         Vehicle.query.delete()
#     #         db.session.commit()
            
#     #         saved_count = 0
#     #         duplicate_count = 0
            
#     #         for vehicle_data in self.processed_data:
#     #             try:
#     #                 # Check for duplicates based on make, model, year, engine_size
#     #                 existing = Vehicle.query.filter_by(
#     #                     make=vehicle_data['make'],
#     #                     model=vehicle_data['model'],
#     #                     year=vehicle_data['year'],
#     #                     engine_size=vehicle_data['engine_size']
#     #                 ).first()
                    
#     #                 if not existing:
#     #                     vehicle = Vehicle(**vehicle_data)
#     #                     db.session.add(vehicle)
#     #                     saved_count += 1
                        
#     #                     if saved_count % 1000 == 0:
#     #                         print(f"  Saved {saved_count:,} vehicles...")
#     #                         db.session.commit()
#     #                 else:
#     #                     duplicate_count += 1
                
#     #             except Exception as e:
#     #                 print(f"  ❌ Error saving vehicle: {e}")
#     #                 continue
            
#     #         # Final commit
#     #         db.session.commit()
            
#     #         print(f"✅ Successfully saved {saved_count:,} vehicles to database")
#     #         if duplicate_count > 0:
#     #             print(f"📋 Skipped {duplicate_count:,} duplicates")
            
#     #         return saved_count
    
#     # def export_processed_data(self, filename='processed_vehicle_data.csv'):
#     #     """Export processed data to CSV"""
#     #     if not self.processed_data:
#     #         print("❌ No processed data to export")
#     #         return False
        
#     #     df_export = pd.DataFrame(self.processed_data)
#     #     df_export.to_csv(filename, index=False)
        
#     #     print(f"📁 Exported {len(self.processed_data):,} vehicles to {filename}")
#     #     return filename
    
#     # def print_statistics(self):
#     #     """Print detailed statistics"""
#     #     if not self.stats:
#     #         print("❌ No statistics available")
#     #         return
        
#     #     print("\n📊 Dataset Statistics:")
#     #     print("=" * 60)
#     #     print(f"Total vehicles: {self.stats['total_vehicles']:,}")
#     #     print(f"Unique makes: {self.stats['unique_makes']:,}")
#     #     print(f"Unique models: {self.stats['unique_models']:,}")
#     #     print(f"Year range: {self.stats['year_range'][0]} - {self.stats['year_range'][1]}")
#     #     print(f"MPG range: {self.stats['mpg_range'][0]} - {self.stats['mpg_range'][1]}")
#     #     print(f"Average MPG: {self.stats['avg_mpg']}")
#     #     print(f"Engine size range: {self.stats['engine_size_range'][0]}L - {self.stats['engine_size_range'][1]}L")
        
#     #     print(f"\n🏭 Top Manufacturers:")
#     #     for make, count in self.stats['makes_distribution'].items():
#     #         percentage = (count / self.stats['total_vehicles']) * 100
#     #         print(f"  {make}: {count:,} ({percentage:.1f}%)")
        
#     #     print(f"\n⛽ Fuel Types:")
#     #     for fuel, count in self.stats['fuel_types'].items():
#     #         percentage = (count / self.stats['total_vehicles']) * 100
#     #         print(f"  {fuel}: {count:,} ({percentage:.1f}%)")
        
#     #     print(f"\n📅 Recent Years Distribution:")
#     #     for year, count in self.stats['year_distribution'].items():
#     #         print(f"  {year}: {count:,}")

# # def process_kaggle_csv(csv_path):
# #     """Main function to process Kaggle CSV file"""
# #     print("🚀 Kaggle Vehicle CSV Data Processing Started")
# #     print("=" * 70)
    
# #     processor = KaggleCSVProcessor()
    
# #     # Step 1: Load CSV
# #     if not processor.load_csv_file(csv_path):
# #         return False
    
# #     # Step 2: Analyze structure
# #     processor.analyze_dataset()
    
# #     # Step 3: Process data
# #     if not processor.clean_and_process_data():
# #         print("❌ Data processing failed")
# #         return False
    
# #     # Step 4: Show statistics
# #     processor.print_statistics()
    
# #     # Step 5: Save to database
# #     saved_count = processor.save_to_database()
    
# #     # Step 6: Export processed data
# #     export_file = processor.export_processed_data()
    
# #     print(f"\n🎉 Processing Complete!")
# #     print(f"📊 Processed {len(processor.processed_data):,} vehicles")
# #     print(f"💾 Saved {saved_count:,} to database")
# #     print(f"📁 Exported to {export_file}")
    
# #     return True


#  ###############################################################################################################   
#     def clean_and_process_data(self):
#         """Clean and process the data for training"""
#         if self.df is None or not self.column_mapping:
#             print("❌ No data or column mapping available")
#             return False
        
#         print("\n🧹 Cleaning and processing data...")
        
#         # Create a copy for processing
#         processed_df = self.df.copy()
        
#         # 1. Filter by year (focus on newer vehicles)
#         if 'year' in self.column_mapping:
#             year_col = self.column_mapping['year']
#             print(f"📅 Filtering years 1990-2020...")
#             processed_df = processed_df[
#                 (processed_df[year_col] >= 1990) & 
#                 (processed_df[year_col] <= 2020)
#             ]
#             print(f"  Remaining rows: {len(processed_df):,}")
        
#         # 2. Remove rows with missing critical data
#         critical_columns = []
#         for key in ['make', 'model', 'year', 'combined_mpg']:
#             if key in self.column_mapping:
#                 critical_columns.append(self.column_mapping[key])
        
#         if critical_columns:
#             before_count = len(processed_df)
#             processed_df = processed_df.dropna(subset=critical_columns)
#             after_count = len(processed_df)
#             print(f"🗑️ Removed {before_count - after_count:,} rows with missing critical data")
        
#         # 3. Clean and standardize data
#         self.processed_data = []
        
#         for index, row in processed_df.iterrows():
#             try:
#                 vehicle_data = self._extract_vehicle_data(row)
#                 if vehicle_data:
#                     self.processed_data.append(vehicle_data)
#             except Exception as e:
#                 continue  # Skip problematic rows
        
#         print(f"✅ Successfully processed {len(self.processed_data):,} vehicles")
        
#         # Generate statistics
#         self._generate_statistics()
        
#         return len(self.processed_data) > 0
    
#     def _extract_vehicle_data(self, row):
#         """Extract and clean vehicle data from a row"""
#         try:
#             # Extract basic info
#             make = str(row[self.column_mapping.get('make', '')]).strip().title()
#             model = str(row[self.column_mapping.get('model', '')]).strip().title()
#             year = int(row[self.column_mapping.get('year', 2000)])
            
#             # Extract MPG data
#             combined_mpg = None
#             if 'combined_mpg' in self.column_mapping:
#                 combined_mpg = float(row[self.column_mapping['combined_mpg']])
#             elif 'city_mpg' in self.column_mapping and 'highway_mpg' in self.column_mapping:
#                 city = float(row[self.column_mapping['city_mpg']])
#                 highway = float(row[self.column_mapping['highway_mpg']])
#                 combined_mpg = (city * 0.55) + (highway * 0.45)  # EPA formula approximation
            
#             if not combined_mpg or combined_mpg <= 0 or combined_mpg > 150:
#                 return None
            
#             # Extract other attributes with defaults
#             engine_size = self._safe_float(row, 'engine_size', 2.5)
#             cylinders = self._safe_int(row, 'cylinders', 4)
            
#             # Clean transmission
#             transmission = self._clean_transmission(row)
            
#             # Clean fuel type
#             fuel_type = self._clean_fuel_type(row)
            
#             # Calculate city and highway if not available
#             city_mpg = self._safe_float(row, 'city_mpg', combined_mpg * 0.9)
#             highway_mpg = self._safe_float(row, 'highway_mpg', combined_mpg * 1.1)
            
#             vehicle_data = {
#                 'make': make,
#                 'model': model,
#                 'year': year,
#                 'engine_size': engine_size,
#                 # 'cylinders': cylinders,
#                 # 'transmission': transmission,
#                 # 'fuel_type': fuel_type,
#                 # 'city_mpg': round(city_mpg, 1),
#                 # 'highway_mpg': round(highway_mpg, 1),
#                 # 'combined_mpg': round(combined_mpg, 1),
#                 # 'co2_emissions': self._calculate_co2_emissions(combined_mpg)
#             }
            
#             return vehicle_data
            
#         except Exception as e:
#             return None
    
#     def _safe_float(self, row, key, default):
#         """Safely extract float value"""
#         try:
#             if key in self.column_mapping:
#                 value = float(row[self.column_mapping[key]])
#                 return value if value > 0 else default
#             return default
#         except:
#             return default
    
#     def _safe_int(self, row, key, default):
#         """Safely extract int value"""
#         try:
#             if key in self.column_mapping:
#                 value = int(row[self.column_mapping[key]])
#                 return value if value > 0 else default
#             return default
#         except:
#             return default
    
#     def _clean_transmission(self, row):
#         """Clean and standardize transmission data"""
#         if 'transmission' not in self.column_mapping:
#             return 'Automatic'
        
#         try:
#             trans = str(row[self.column_mapping['transmission']]).lower()
            
#             if any(word in trans for word in ['manual', 'man', 'mt', 'm6', 'm5']):
#                 return 'Manual'
#             elif any(word in trans for word in ['cvt', 'variable']):
#                 return 'CVT'
#             elif any(word in trans for word in ['auto', 'automatic', 'at', 'a6', 'a8', 'a4']):
#                 return 'Automatic'
#             else:
#                 return 'Automatic'
#         except:
#             return 'Automatic'
    
#     def _clean_fuel_type(self, row):
#         """Clean and standardize fuel type data"""
#         if 'fuel_type' not in self.column_mapping:
#             return 'Gasoline'
        
#         try:
#             fuel = str(row[self.column_mapping['fuel_type']]).lower()
            
#             if any(word in fuel for word in ['electric', 'electricity', 'ev']):
#                 return 'Electric'
#             elif any(word in fuel for word in ['hybrid', 'hev']):
#                 return 'Hybrid'
#             elif any(word in fuel for word in ['diesel', 'biodiesel']):
#                 return 'Diesel'
#             elif any(word in fuel for word in ['gasoline', 'gas', 'petrol', 'regular']):
#                 return 'Gasoline'
#             else:
#                 return 'Gasoline'
#         except:
#             return 'Gasoline'
    
#     def _calculate_co2_emissions(self, mpg):
#         """Calculate CO2 emissions based on MPG"""
#         if mpg <= 0:
#             return 500
#         return (19.6 * 12000) / mpg
    
#     def _generate_statistics(self):
#         """Generate statistics about the processed data"""
#         if not self.processed_data:
#             return
        
#         df_stats = pd.DataFrame(self.processed_data)
        
#         self.stats = {
#             'total_vehicles': len(self.processed_data),
#             'unique_makes': df_stats['make'].nunique(),
#             'unique_models': df_stats['model'].nunique(),
#             'year_range': (df_stats['year'].min(), df_stats['year'].max()),
#             'mpg_range': (df_stats['combined_mpg'].min(), df_stats['combined_mpg'].max()),
#             'avg_mpg': df_stats['combined_mpg'].mean(),
#             'makes_distribution': df_stats['make'].value_counts().head(10).to_dict(),
#             'fuel_types': df_stats['fuel_type'].value_counts().to_dict(),
#             'transmission_types': df_stats['transmission'].value_counts().to_dict()
#         }
    
#     def save_to_database(self):
#         """Save processed data to database"""
#         if not self.processed_data:
#             print("❌ No processed data to save")
#             return False
        
#         print("💾 Saving data to database...")
        
#         with app.app_context():
#             # Clear existing data
#             Vehicle.query.delete()
#             db.session.commit()
            
#             saved_count = 0
#             for vehicle_data in self.processed_data:
#                 try:
#                     # Check for duplicates
#                     existing = Vehicle.query.filter_by(
#                         make=vehicle_data['make'],
#                         model=vehicle_data['model'],
#                         year=vehicle_data['year'],
#                         engine_size=vehicle_data['engine_size']
#                     ).first()
                    
#                     if not existing:
#                         vehicle = Vehicle(**vehicle_data)
#                         db.session.add(vehicle)
#                         saved_count += 1
                        
#                         if saved_count % 1000 == 0:
#                             print(f"  Saved {saved_count:,} vehicles...")
#                             db.session.commit()
                
#                 except Exception as e:
#                     print(f"  ❌ Error saving vehicle: {e}")
#                     continue
            
#             # Final commit
#             db.session.commit()
            
#             print(f"✅ Successfully saved {saved_count:,} vehicles to database")
#             return saved_count
    
#     def export_processed_data(self, filename='processed_vehicle_data.csv'):
#         """Export processed data to CSV"""
#         if not self.processed_data:
#             print("❌ No processed data to export")
#             return False
        
#         df_export = pd.DataFrame(self.processed_data)
#         df_export.to_csv(filename, index=False)
        
#         print(f"📁 Exported {len(self.processed_data):,} vehicles to {filename}")
#         return filename
    
#     def print_statistics(self):
#         """Print detailed statistics"""
#         if not self.stats:
#             print("❌ No statistics available")
#             return
        
#         print("\n📊 Dataset Statistics:")
#         print("=" * 50)
#         print(f"Total vehicles: {self.stats['total_vehicles']:,}")
#         print(f"Unique makes: {self.stats['unique_makes']:,}")
#         print(f"Unique models: {self.stats['unique_models']:,}")
#         print(f"Year range: {self.stats['year_range'][0]} - {self.stats['year_range'][1]}")
#         print(f"MPG range: {self.stats['mpg_range'][0]:.1f} - {self.stats['mpg_range'][1]:.1f}")
#         print(f"Average MPG: {self.stats['avg_mpg']:.1f}")
        
#         print(f"\n🏭 Top Manufacturers:")
#         for make, count in self.stats['makes_distribution'].items():
#             percentage = (count / self.stats['total_vehicles']) * 100
#             print(f"  {make}: {count:,} ({percentage:.1f}%)")
        
#         print(f"\n⛽ Fuel Types:")
#         for fuel, count in self.stats['fuel_types'].items():
#             percentage = (count / self.stats['total_vehicles']) * 100
#             print(f"  {fuel}: {count:,} ({percentage:.1f}%)")

# def process_kaggle_csv(csv_path):
#     """Main function to process Kaggle CSV file"""
#     print("🚀 Kaggle CSV Data Processing Started")
#     print("=" * 60)
    
#     processor = KaggleCSVProcessor()
    
#     # Step 1: Load CSV
#     if not processor.load_csv_file(csv_path):
#         return False
    
#     # Step 2: Analyze structure
#     column_mapping = processor.analyze_dataset()
#     if not column_mapping:
#         print("❌ Could not identify required columns")
#         return False
    
#     # Step 3: Process data
#     if not processor.clean_and_process_data():
#         print("❌ Data processing failed")
#         return False
    
#     # Step 4: Show statistics
#     processor.print_statistics()
    
#     # Step 5: Save to database
#     saved_count = processor.save_to_database()
    
#     # Step 6: Export processed data
#     export_file = processor.export_processed_data()
    
#     print(f"\n🎉 Processing Complete!")
#     print(f"📊 Processed {len(processor.processed_data):,} vehicles")
#     print(f"💾 Saved {saved_count:,} to database")
#     print(f"📁 Exported to {export_file}")
    
#     return True

# if __name__ == "__main__":
#     # Get CSV file path from user
#     csv_path = "vehicles.csv"
    
#     if not csv_path:
#         csv_path = "vehicles.csv"  # Default name
#         print(f"Using default filename: {csv_path}")
    
#     success = process_kaggle_csv(csv_path)
    
#     if success:
#         print("\n✅ Ready for model training!")
#         print("Next steps:")
#         print("1. Run: python train_models.py")
#         print("2. Run: python app.py")
#     else:
#         print("\n❌ Processing failed. Please check your CSV file and try again.")




# import pandas as pd
# import os
# import sys
# import re

# # Add the backend directory to the path
# sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# # Import after path is set
# from flask import Flask
# from flask_sqlalchemy import SQLAlchemy
# from datetime import datetime

# # Create a separate Flask app instance for data loading
# data_app = Flask(__name__)

# # Database configuration
# if os.environ.get('DATABASE_URL'):
#     database_url = os.environ.get('DATABASE_URL')
#     if database_url.startswith('postgres://'):
#         database_url = database_url.replace('postgres://', 'postgresql://', 1)
#     data_app.config['SQLALCHEMY_DATABASE_URI'] = database_url
#     print("✅ Using PostgreSQL database")
# else:
#     data_app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fuel_efficiency.db'
#     print("✅ Using SQLite database")

# if os.environ.get('FLASK_ENV') != 'production':  # Skip DB in build
#     saved_count = processor.save_to_database()
# else:
#     print("⚠️ Skipping DB save in production build")
#     saved_count = 0

# data_app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# # Initialize database
# data_db = SQLAlchemy(data_app)

# # Define Vehicle model
# class Vehicle(data_db.Model):
#     __tablename__ = 'vehicle'
    
#     id = data_db.Column(data_db.Integer, primary_key=True)
#     make = data_db.Column(data_db.String(100), nullable=False)
#     model = data_db.Column(data_db.String(100), nullable=False)
#     year = data_db.Column(data_db.Integer, nullable=False)
#     engine_size = data_db.Column(data_db.Float, nullable=False)
#     cylinders = data_db.Column(data_db.Integer, nullable=False)
#     transmission = data_db.Column(data_db.String(50), nullable=False)
#     fuel_type = data_db.Column(data_db.String(50), nullable=False)
#     city_mpg = data_db.Column(data_db.Float, nullable=False)
#     highway_mpg = data_db.Column(data_db.Float, nullable=False)
#     combined_mpg = data_db.Column(data_db.Float, nullable=False)
#     co2_emissions = data_db.Column(data_db.Float, nullable=False)
#     created_at = data_db.Column(data_db.DateTime, default=datetime.utcnow)

# def clean_column_names(df):
#     """Clean and standardize column names"""
#     df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('-', '_')
#     return df

# def convert_engine_size(value):
#     """Convert engine size from cm3 to liters"""
#     try:
#         if pd.isna(value):
#             return None
#         # Remove any non-numeric characters except decimal point
#         cleaned = re.sub(r'[^\d.]', '', str(value))
#         if not cleaned:
#             return None
#         # Convert from cm3 to liters (divide by 1000)
#         cm3 = float(cleaned)
#         liters = cm3 / 1000.0
#         return round(liters, 1)
#     except:
#         return None

# def convert_fuel_consumption(value):
#     """Convert L/100km to MPG"""
#     try:
#         if pd.isna(value) or value == 0:
#             return None
#         l_per_100km = float(value)
#         # Formula: MPG = 235.215 / (L/100km)
#         mpg = 235.215 / l_per_100km
#         return round(mpg, 1)
#     except:
#         return None

# def estimate_cylinders(engine_size):
#     """Estimate number of cylinders based on engine size"""
#     if engine_size is None or engine_size == 0:
#         return 0  # Electric
#     elif engine_size < 1.0:
#         return 3
#     elif engine_size < 2.0:
#         return 4
#     elif engine_size < 3.5:
#         return 6
#     else:
#         return 8

# def determine_fuel_type(fuel_consumption):
#     """Determine fuel type based on consumption"""
#     if fuel_consumption is None or fuel_consumption > 100:
#         return 'Electric'
#     elif fuel_consumption > 45:
#         return 'Hybrid'
#     else:
#         return 'Gasoline'

# def load_csv_to_database(csv_file='vehicles.csv'):
#     """
#     Load data from CSV file into the database.
#     Handles your specific CSV format with columns:
#     - Make
#     - Modle (Model)
#     - Year_from
#     - capacity_cm3
#     - mixed_fuel_consumption_per_100_km_l
#     """
    
#     # Check if file exists
#     if not os.path.exists(csv_file):
#         print(f"❌ Error: File '{csv_file}' not found!")
#         print(f"📁 Current directory: {os.getcwd()}")
#         print(f"📂 Files in directory: {os.listdir('.')}")
#         return False
    
#     try:
#         print("="*60)
#         print("📊 VEHICLE DATA PROCESSOR")
#         print("="*60)
#         print(f"\n📥 Reading CSV file: {csv_file}")
        
#         # Read CSV
#         df = pd.read_csv(csv_file, low_memory=False)
#         print(f"✅ Successfully read {len(df)} rows from CSV")
        
#         # Clean column names
#         df = clean_column_names(df)
#         print(f"\n📋 Columns found: {list(df.columns)}")
        
#         # Create column mapping for your CSV format
#         column_mapping = {
#             'make': 'make',
#             'modle': 'model',  # Note: Your CSV has 'Modle' (typo)
#             'model': 'model',
#             'year_from': 'year',
#             'capacity_cm3': 'engine_size_cm3',
#             'mixed_fuel_consumption_per_100_km_l': 'fuel_consumption_l100km'
#         }
        
#         # Rename columns
#         rename_dict = {}
#         for csv_col in df.columns:
#             for std_col, target_col in column_mapping.items():
#                 if std_col in csv_col.lower():
#                     rename_dict[csv_col] = target_col
#                     break
        
#         df = df.rename(columns=rename_dict)
#         print(f"\n📝 Mapped columns: {rename_dict}")
        
#         # Check required columns
#         required_base_columns = ['make', 'model', 'year', 'engine_size_cm3', 'fuel_consumption_l100km']
#         missing = [col for col in required_base_columns if col not in df.columns]
        
#         if missing:
#             print(f"\n❌ Missing required columns: {missing}")
#             print(f"📋 Available columns: {list(df.columns)}")
#             return False
        
#         print("\n🔄 Processing data...")
        
#         # Convert engine size from cm3 to liters
#         print("  ⚙️  Converting engine sizes (cm3 → liters)...")
#         df['engine_size'] = df['engine_size_cm3'].apply(convert_engine_size)
        
#         # Convert fuel consumption from L/100km to MPG
#         print("  ⛽ Converting fuel consumption (L/100km → MPG)...")
#         df['combined_mpg'] = df['fuel_consumption_l100km'].apply(convert_fuel_consumption)
        
#         # Estimate city and highway MPG (typical split)
#         df['city_mpg'] = df['combined_mpg'] * 0.9  # City is typically 10% worse
#         df['highway_mpg'] = df['combined_mpg'] * 1.08  # Highway is typically 8% better
        
#         # Estimate cylinders
#         print("  🔢 Estimating cylinders...")
#         df['cylinders'] = df['engine_size'].apply(estimate_cylinders)
        
#         # Determine fuel type
#         print("  ⚡ Determining fuel types...")
#         df['fuel_type'] = df['combined_mpg'].apply(determine_fuel_type)
        
#         # Add default transmission (most common)
#         df['transmission'] = 'Automatic'
        
#         # Calculate CO2 emissions from fuel consumption
#         # Formula: CO2 (g/km) = L/100km * 23.2 (approximate for gasoline)
#         df['co2_emissions'] = df['fuel_consumption_l100km'] * 23.2
#         df.loc[df['fuel_type'] == 'Electric', 'co2_emissions'] = 0
        
#         # Clean year data
#         df['year'] = pd.to_numeric(df['year'], errors='coerce')
        
#         # Remove rows with missing critical data
#         print("\n🧹 Cleaning data...")
#         initial_count = len(df)
        
#         df = df.dropna(subset=['make', 'model', 'year', 'engine_size', 'combined_mpg'])
#         df = df[df['year'] >= 1990]  # Only recent vehicles
#         df = df[df['year'] <= 2030]
#         df = df[df['engine_size'] >= 0]
#         df = df[df['combined_mpg'] > 0]
#         df = df[df['combined_mpg'] < 200]  # Remove unrealistic values
        
#         # Convert to appropriate types
#         df['year'] = df['year'].astype(int)
#         df['cylinders'] = df['cylinders'].astype(int)
        
#         cleaned_count = len(df)
#         removed_count = initial_count - cleaned_count
        
#         print(f"  ✓ Removed {removed_count} invalid rows")
#         print(f"  ✓ {cleaned_count} valid rows remaining")
        
#         # Select final columns
#         final_columns = ['make', 'model', 'year', 'engine_size', 'cylinders', 
#                         'transmission', 'fuel_type', 'city_mpg', 'highway_mpg', 
#                         'combined_mpg', 'co2_emissions']
        
#         df = df[final_columns]
        
#         print(f"\n📊 Final dataset statistics:")
#         print(f"  • Total records: {len(df)}")
#         print(f"  • Makes: {df['make'].nunique()}")
#         print(f"  • Models: {df['model'].nunique()}")
#         print(f"  • Year range: {int(df['year'].min())} - {int(df['year'].max())}")
#         print(f"  • MPG range: {df['combined_mpg'].min():.1f} - {df['combined_mpg'].max():.1f}")
        
#         # Create tables and load data
#         with data_app.app_context():
#             print("\n🗄️  Creating database tables...")
#             data_db.create_all()
#             print("✅ Database tables created")
            
#             # Clear existing data
#             print("\n🗑️  Clearing existing data...")
#             Vehicle.query.delete()
#             data_db.session.commit()
#             print("✅ Existing data cleared")
            
#             # Insert data in batches
#             print("\n💾 Inserting data into database...")
#             records_added = 0
#             batch_size = 100
#             errors = 0
            
#             for i in range(0, len(df), batch_size):
#                 batch = df.iloc[i:i+batch_size]
                
#                 for _, row in batch.iterrows():
#                     try:
#                         vehicle = Vehicle(
#                             make=str(row['make'])[:100],
#                             model=str(row['model'])[:100],
#                             year=int(row['year']),
#                             engine_size=float(row['engine_size']),
#                             cylinders=int(row['cylinders']),
#                             transmission=str(row['transmission']),
#                             fuel_type=str(row['fuel_type']),
#                             city_mpg=float(row['city_mpg']),
#                             highway_mpg=float(row['highway_mpg']),
#                             combined_mpg=float(row['combined_mpg']),
#                             co2_emissions=float(row['co2_emissions'])
#                         )
#                         data_db.session.add(vehicle)
#                         records_added += 1
#                     except Exception as e:
#                         errors += 1
#                         if errors <= 5:  # Show first 5 errors only
#                             print(f"    ⚠️  Error adding row: {e}")
                
#                 # Commit batch
#                 try:
#                     data_db.session.commit()
#                     progress = min(i + batch_size, len(df))
#                     if progress % 1000 == 0 or progress == len(df):
#                         print(f"  ✓ Processed {progress}/{len(df)} rows ({records_added} added, {errors} errors)")
#                 except Exception as e:
#                     print(f"    ❌ Batch commit error: {e}")
#                     data_db.session.rollback()
            
#             print("\n" + "="*60)
#             print(f"✅ DATA IMPORT COMPLETED!")
#             print("="*60)
#             print(f"📊 Summary:")
#             print(f"  • Total records imported: {records_added}")
#             print(f"  • Errors encountered: {errors}")
#             print(f"  • Success rate: {(records_added/len(df)*100):.1f}%")
            
#             # Show sample data
#             print(f"\n📋 Sample vehicles in database:")
#             sample_vehicles = Vehicle.query.limit(10).all()
#             for v in sample_vehicles:
#                 print(f"  • {v.year} {v.make} {v.model} - {v.combined_mpg:.1f} MPG ({v.fuel_type})")
            
#             # Show statistics by fuel type
#             print(f"\n📈 Fuel type distribution:")
#             for fuel_type in ['Gasoline', 'Hybrid', 'Electric']:
#                 count = Vehicle.query.filter_by(fuel_type=fuel_type).count()
#                 if count > 0:
#                     print(f"  • {fuel_type}: {count} vehicles")
            
#             return True
            
#     except Exception as e:
#         print(f"\n❌ Error loading CSV: {str(e)}")
#         import traceback
#         traceback.print_exc()
#         return False

# if __name__ == '__main__':
#     print("\n" + "="*60)
#     print("🚗 VEHICLE DATA PROCESSOR")
#     print("="*60)
    
#     # Automatically use vehicles.csv
#     csv_file = 'vehicles.csv'
    
#     print(f"\n🔍 Looking for file: {csv_file}")
#     print(f"📁 Current directory: {os.getcwd()}\n")
    
#     success = load_csv_to_database(csv_file)
    
#     if success:
#         print("\n" + "="*60)
#         print("✅ DATA PROCESSING COMPLETED SUCCESSFULLY!")
#         print("="*60)
#         print("\n🎯 Next steps:")
#         print("  1. Run: python app.py")
#         print("  2. Test API: http://localhost:5000/api/health")
#         print("  3. Train models: python train_models.py")
#         print("\n")
#     else:
#         print("\n" + "="*60)
#         print("❌ DATA PROCESSING FAILED!")
#         print("="*60)
#         print("\n📝 Please check:")
#         print("  • CSV file exists and is readable")
#         print("  • Column names match expected format")
#         print("  • Data quality and formatting")
#         print("\n")
#         sys.exit(1)


import pandas as pd
import os
import sys
import re
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the backend directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import after path is set
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

# Create a separate Flask app instance for data loading
data_app = Flask(__name__)

# Database configuration - Force SQLite (no PostgreSQL fallback)
data_app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///fuel_efficiency.db'
data_app.config['DEBUG'] = os.environ.get('FLASK_ENV', 'development') != 'production'
data_app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
data_app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
}

print("✅ Using SQLite database: fuel_efficiency.db")

# Initialize database
data_db = SQLAlchemy(data_app)

# Define Vehicle model (using your four factors: make, model, year, engine_size)
class Vehicle(data_db.Model):
    __tablename__ = 'vehicle'
    
    id = data_db.Column(data_db.Integer, primary_key=True)
    make = data_db.Column(data_db.String(100), nullable=False)
    model = data_db.Column(data_db.String(100), nullable=False)
    year = data_db.Column(data_db.Integer, nullable=False)
    engine_size = data_db.Column(data_db.Float, nullable=False)
    # Optional fields for completeness (from processing)
    cylinders = data_db.Column(data_db.Integer, nullable=False)
    transmission = data_db.Column(data_db.String(50), nullable=False)
    fuel_type = data_db.Column(data_db.String(50), nullable=False)
    city_mpg = data_db.Column(data_db.Float, nullable=False)
    highway_mpg = data_db.Column(data_db.Float, nullable=False)
    combined_mpg = data_db.Column(data_db.Float, nullable=False)
    co2_emissions = data_db.Column(data_db.Float, nullable=False)
    created_at = data_db.Column(data_db.DateTime, default=datetime.utcnow)

def clean_column_names(df):
    """Clean and standardize column names"""
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('-', '_')
    return df

def convert_engine_size(value):
    """Convert engine size from cm3 to liters"""
    try:
        if pd.isna(value):
            return None
        # Remove any non-numeric characters except decimal point
        cleaned = re.sub(r'[^\d.]', '', str(value))
        if not cleaned:
            return None
        # Convert from cm3 to liters (divide by 1000)
        cm3 = float(cleaned)
        liters = cm3 / 1000.0
        return round(liters, 1)
    except:
        return None

def convert_fuel_consumption(value):
    """Convert L/100km to MPG"""
    try:
        if pd.isna(value) or value == 0:
            return None
        l_per_100km = float(value)
        # Formula: MPG = 235.215 / (L/100km)
        mpg = 235.215 / l_per_100km
        return round(mpg, 1)
    except:
        return None

def estimate_cylinders(engine_size):
    """Estimate number of cylinders based on engine size"""
    if engine_size is None or engine_size == 0:
        return 0  # Electric
    elif engine_size < 1.0:
        return 3
    elif engine_size < 2.0:
        return 4
    elif engine_size < 3.5:
        return 6
    else:
        return 8

def determine_fuel_type(fuel_consumption):
    """Determine fuel type based on consumption"""
    if fuel_consumption is None or fuel_consumption > 100:
        return 'Electric'
    elif fuel_consumption > 45:
        return 'Hybrid'
    else:
        return 'Gasoline'

def load_csv_to_database(csv_file='vehicles.csv'):
    """
    Load data from CSV file into the database.
    Handles your specific CSV format with columns:
    - Make
    - Modle (Model)
    - Year_from
    - capacity_cm3
    - mixed_fuel_consumption_per_100_km_l
    """
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"❌ Error: File '{csv_file}' not found!")
        print(f"📁 Current directory: {os.getcwd()}")
        print(f"📂 Files in directory: {os.listdir('.')}")
        return False
    
    try:
        print("="*60)
        print("📊 VEHICLE DATA PROCESSOR")
        print("="*60)
        print(f"\n📥 Reading CSV file: {csv_file}")
        
        # Read CSV
        df = pd.read_csv(csv_file, low_memory=False)
        print(f"✅ Successfully read {len(df)} rows from CSV")
        
        # Clean column names
        df = clean_column_names(df)
        print(f"\n📋 Columns found: {list(df.columns)}")
        
        # Create column mapping for your CSV format
        column_mapping = {
            'make': 'make',
            'modle': 'model',  # Note: Your CSV has 'Modle' (typo)
            'model': 'model',
            'year_from': 'year',
            'capacity_cm3': 'engine_size_cm3',
            'mixed_fuel_consumption_per_100_km_l': 'fuel_consumption_l100km'
        }
        
        # Rename columns
        rename_dict = {}
        for csv_col in df.columns:
            for std_col, target_col in column_mapping.items():
                if std_col in csv_col.lower():
                    rename_dict[csv_col] = target_col
                    break
        
        df = df.rename(columns=rename_dict)
        print(f"\n📝 Mapped columns: {rename_dict}")
        
        # Check required columns
        required_base_columns = ['make', 'model', 'year', 'engine_size_cm3', 'fuel_consumption_l100km']
        missing = [col for col in required_base_columns if col not in df.columns]
        
        if missing:
            print(f"\n❌ Missing required columns: {missing}")
            print(f"📋 Available columns: {list(df.columns)}")
            return False
        
        print("\n🔄 Processing data...")
        
        # Convert engine size from cm3 to liters
        print("  ⚙️  Converting engine sizes (cm3 → liters)...")
        df['engine_size'] = df['engine_size_cm3'].apply(convert_engine_size)
        
        # Convert fuel consumption from L/100km to MPG
        print("  ⛽ Converting fuel consumption (L/100km → MPG)...")
        df['combined_mpg'] = df['fuel_consumption_l100km'].apply(convert_fuel_consumption)
        
        # Estimate city and highway MPG (typical split)
        df['city_mpg'] = df['combined_mpg'] * 0.9  # City is typically 10% worse
        df['highway_mpg'] = df['combined_mpg'] * 1.08  # Highway is typically 8% better
        
        # Estimate cylinders
        print("  🔢 Estimating cylinders...")
        df['cylinders'] = df['engine_size'].apply(estimate_cylinders)
        
        # Determine fuel type
        print("  ⚡ Determining fuel types...")
        df['fuel_type'] = df['combined_mpg'].apply(determine_fuel_type)
        
        # Add default transmission (most common)
        df['transmission'] = 'Automatic'
        
        # Calculate CO2 emissions from fuel consumption
        # Formula: CO2 (g/km) = L/100km * 23.2 (approximate for gasoline)
        df['co2_emissions'] = df['fuel_consumption_l100km'] * 23.2
        df.loc[df['fuel_type'] == 'Electric', 'co2_emissions'] = 0
        
        # Clean year data
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        
        # Remove rows with missing critical data
        print("\n🧹 Cleaning data...")
        initial_count = len(df)
        
        df = df.dropna(subset=['make', 'model', 'year', 'engine_size', 'combined_mpg'])
        df = df[df['year'] >= 1990]  # Only recent vehicles
        df = df[df['year'] <= 2030]
        df = df[df['engine_size'] >= 0]
        df = df[df['combined_mpg'] > 0]
        df = df[df['combined_mpg'] < 200]  # Remove unrealistic values
        
        # Convert to appropriate types
        df['year'].astype(int)
        df['cylinders'] = df['cylinders'].astype(int)
        
        cleaned_count = len(df)
        removed_count = initial_count - cleaned_count
        
        print(f"  ✓ Removed {removed_count} invalid rows")
        print(f"  ✓ {cleaned_count} valid rows remaining")
        
        # Select final columns
        final_columns = ['make', 'model', 'year', 'engine_size', 'cylinders', 
                        'transmission', 'fuel_type', 'city_mpg', 'highway_mpg', 
                        'combined_mpg', 'co2_emissions']
        
        df = df[final_columns]
        
        print(f"\n📊 Final dataset statistics:")
        print(f"  • Total records: {len(df)}")
        print(f"  • Makes: {df['make'].nunique()}")
        print(f"  • Models: {df['model'].nunique()}")
        print(f"  • Year range: {int(df['year'].min())} - {int(df['year'].max())}")
        print(f"  • MPG range: {df['combined_mpg'].min():.1f} - {df['combined_mpg'].max():.1f}")
        
        # Create tables and load data
        with data_app.app_context():
            print("\n🗄️  Creating database tables...")
            data_db.create_all()
            print("✅ Database tables created")
            
            # Clear existing data
            print("\n🗑️  Clearing existing data...")
            Vehicle.query.delete()
            data_db.session.commit()
            print("✅ Existing data cleared")
            
            # Insert data in batches
            print("\n💾 Inserting data into database...")
            records_added = 0
            batch_size = 100
            errors = 0
            
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                
                for _, row in batch.iterrows():
                    try:
                        vehicle = Vehicle(
                            make=str(row['make'])[:100],
                            model=str(row['model'])[:100],
                            year=int(row['year']),
                            engine_size=float(row['engine_size']),
                            cylinders=int(row['cylinders']),
                            transmission=str(row['transmission']),
                            fuel_type=str(row['fuel_type']),
                            city_mpg=float(row['city_mpg']),
                            highway_mpg=float(row['highway_mpg']),
                            combined_mpg=float(row['combined_mpg']),
                            co2_emissions=float(row['co2_emissions'])
                        )
                        data_db.session.add(vehicle)
                        records_added += 1
                    except Exception as e:
                        errors += 1
                        if errors <= 5:  # Show first 5 errors only
                            print(f"    ⚠️  Error adding row: {e}")
                
                # Commit batch
                try:
                    data_db.session.commit()
                    progress = min(i + batch_size, len(df))
                    if progress % 1000 == 0 or progress == len(df):
                        print(f"  ✓ Processed {progress}/{len(df)} rows ({records_added} added, {errors} errors)")
                except Exception as e:
                    print(f"    ❌ Batch commit error: {e}")
                    data_db.session.rollback()
            
            print("\n" + "="*60)
            print(f"✅ DATA IMPORT COMPLETED!")
            print("="*60)
            print(f"📊 Summary:")
            print(f"  • Total records imported: {records_added}")
            print(f"  • Errors encountered: {errors}")
            print(f"  • Success rate: {(records_added/len(df)*100):.1f}%")
            
            # Show sample data
            print(f"\n📋 Sample vehicles in database:")
            sample_vehicles = Vehicle.query.limit(10).all()
            for v in sample_vehicles:
                print(f"  • {v.year} {v.make} {v.model} - {v.combined_mpg:.1f} MPG ({v.fuel_type})")
            
            # Show statistics by fuel type
            print(f"\n📈 Fuel type distribution:")
            for fuel_type in ['Gasoline', 'Hybrid', 'Electric']:
                count = Vehicle.query.filter_by(fuel_type=fuel_type).count()
                if count > 0:
                    print(f"  • {fuel_type}: {count} vehicles")
            
            return True
            
    except Exception as e:
        print(f"\n❌ Error loading CSV: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚗 VEHICLE DATA PROCESSOR")
    print("="*60)
    
    # Automatically use vehicles.csv
    csv_file = 'vehicles.csv'
    
    print(f"\n🔍 Looking for file: {csv_file}")
    print(f"📁 Current directory: {os.getcwd()}\n")
    
    success = load_csv_to_database(csv_file)
    
    if success:
        print("\n" + "="*60)
        print("✅ DATA PROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n🎯 Next steps:")
        print("  1. Run: python app.py")
        print("  2. Test API: http://localhost:5000/api/health")
        print("  3. Train models: python train_models.py")
        print("\n")
    else:
        print("\n" + "="*60)
        print("❌ DATA PROCESSING FAILED!")
        print("="*60)
        print("\n📝 Please check:")
        print("  • CSV file exists and is readable")
        print("  • Column names match expected format")
        print("  • Data quality and formatting")
        print("\n")
        sys.exit(1)

