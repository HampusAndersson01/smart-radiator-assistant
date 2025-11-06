#!/bin/bash
# Reset database and restart training from scratch

echo "⚠️  WARNING: This will delete ALL training data, models, and history!"
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

echo "🗑️  Dropping and recreating database..."

# Connect to PostgreSQL and reset
docker compose exec -T postgres psql -U postgres <<EOF
-- Drop the database
DROP DATABASE IF EXISTS smart_radiator_ai;

-- Recreate the database
CREATE DATABASE smart_radiator_ai;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE smart_radiator_ai TO postgres;

EOF

echo "✅ Database reset complete!"
echo ""
echo "🔄 Restarting AI service to initialize fresh schema..."
docker compose restart ai_service

echo ""
echo "✅ Done! The system is now starting fresh with no training data."
echo "   All models will learn from scratch based on new data."
echo ""
echo "📊 The AI will now use:"
echo "   - Both 3h and 10h weather forecasts"
echo "   - Night mode (8h optimization in evenings)"
echo "   - Prediction validation and self-training"
echo "   - Improved feature set (forecast_3h_temp + forecast_10h_temp)"
