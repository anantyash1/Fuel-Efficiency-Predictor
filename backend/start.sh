#!/usr/bin/env bash
set -e

echo "============================================"
echo "🚀 STARTING FUEL EFFICIENCY PREDICTOR"
echo "============================================"

# Check DATABASE_URL
if [ -z "$DATABASE_URL" ]; then
    echo "❌ ERROR: DATABASE_URL not set!"
    echo "Please configure DATABASE_URL in Render Dashboard"
fi

# Initialize database
echo "📊 Initializing database..."
python -c "
from app import app, db
with app.app_context():
    db.create_all()
    print('✅ Database tables created')
" || echo "⚠️ Database initialization completed with warnings"

# Load data from CSV if exists
if [ -f "vehicles.csv" ]; then
    echo "📥 Loading vehicle data from CSV..."
    timeout 180 python csv_data_processor.py || echo "⚠️ CSV processing timeout or error"
else
    echo "⚠️ vehicles.csv not found - using sample data"
fi

echo "============================================"
echo "✅ Starting Gunicorn server..."
echo "============================================"

# Start with appropriate timeout for large datasets
exec gunicorn --bind 0.0.0.0:$PORT \
    --workers 2 \
    --timeout 300 \
    --access-logfile - \
    --error-logfile - \
    app:app