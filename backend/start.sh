#!/usr/bin/env bash
set -e

echo "🚀 Starting application..."

# Create database tables
python -c "from app import initialize_database; initialize_database()"

# Load CSV data if exists
if [ -f "vehicles.csv" ]; then
    echo "📊 Loading vehicle data..."
    python csv_data_processor.py || echo "⚠️ CSV load completed"
fi

# Start application
exec gunicorn --bind 0.0.0.0:$PORT --workers 2 --timeout 120 app:app