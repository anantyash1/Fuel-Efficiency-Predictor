# scheduled_updates.py
import schedule
import time
from api_manager import APIManager

def update_vehicle_data():
    print("🔄 Running scheduled vehicle data update...")
    manager = APIManager()
    new_vehicles = manager.fetch_all_available_data(max_vehicles=50)  # Smaller batch
    
    if new_vehicles:
        manager.save_to_database(new_vehicles)
        print(f"✅ Updated {len(new_vehicles)} vehicles")
    else:
        print("ℹ️ No new vehicles to update")

# Schedule updates
schedule.every().week.do(update_vehicle_data)  # Weekly updates
schedule.every().day.at("02:00").do(update_vehicle_data)  # Daily at 2 AM

print("⏰ Scheduler started. Vehicle data will update weekly.")
while True:
    schedule.run_pending()
    time.sleep(3600)  # Check every hour