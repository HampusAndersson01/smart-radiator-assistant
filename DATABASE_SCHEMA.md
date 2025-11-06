# Database Schema Documentation

## Overview

The Smart Radiator AI now uses PostgreSQL as the primary storage for all data, models, and statistics. This ensures data persistence across restarts and provides comprehensive historical data for analysis and graphing.

## Database Tables

### 1. `room_states`
Stores all temperature readings and radiator levels over time.

```sql
CREATE TABLE room_states (
    id SERIAL PRIMARY KEY,
    room VARCHAR(50) NOT NULL,
    current_temp FLOAT NOT NULL,
    target_temp FLOAT NOT NULL,
    radiator_level FLOAT NOT NULL,
    outdoor_temp FLOAT,
    forecast_temp FLOAT,
    timestamp TIMESTAMP DEFAULT NOW(),
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Purpose:** Historical tracking of all room conditions  
**Used for:** Graphing temperature trends, analyzing patterns  
**Indexes:** room, timestamp

### 2. `ai_metrics`
Stores AI model performance metrics per room.

```sql
CREATE TABLE ai_metrics (
    id SERIAL PRIMARY KEY,
    room VARCHAR(50) UNIQUE NOT NULL,
    mae FLOAT DEFAULT 0,
    rmse FLOAT DEFAULT 0,
    r2_score FLOAT DEFAULT 0,
    training_samples INTEGER DEFAULT 0,
    predictions_made INTEGER DEFAULT 0,
    adjustments_made INTEGER DEFAULT 0,
    total_error FLOAT DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    last_updated TIMESTAMP DEFAULT NOW()
);
```

**Purpose:** Track model performance and learning progress  
**Updated:** Every train/predict operation  
**Key Metrics:** MAE, RMSE, R², training count

### 3. `training_history`
Detailed log of each training event.

```sql
CREATE TABLE training_history (
    id SERIAL PRIMARY KEY,
    room VARCHAR(50) NOT NULL,
    current_temp FLOAT NOT NULL,
    target_temp FLOAT NOT NULL,
    radiator_level FLOAT NOT NULL,
    outdoor_temp FLOAT,
    forecast_temp FLOAT,
    temperature_delta FLOAT NOT NULL,
    predicted_delta FLOAT,
    hour_of_day INTEGER,
    timestamp TIMESTAMP DEFAULT NOW()
);
```

**Purpose:** Audit trail of all training operations  
**Used for:** Analyzing learning patterns, debugging  
**Indexes:** room, timestamp

### 4. `predictions`
Log of all predictions and recommendations.

```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    room VARCHAR(50) NOT NULL,
    current_temp FLOAT NOT NULL,
    target_temp FLOAT NOT NULL,
    current_radiator_level FLOAT NOT NULL,
    recommended_level FLOAT NOT NULL,
    predicted_error FLOAT NOT NULL,
    outdoor_temp FLOAT,
    forecast_temp FLOAT,
    adjustment_made BOOLEAN DEFAULT FALSE,
    timestamp TIMESTAMP DEFAULT NOW()
);
```

**Purpose:** Track all AI predictions and adjustments  
**Used for:** Performance analysis, efficiency metrics  
**Indexes:** room, timestamp

### 5. `ml_models`
Stores serialized machine learning models.

```sql
CREATE TABLE ml_models (
    room VARCHAR(50) PRIMARY KEY,
    model_data BYTEA NOT NULL,
    version INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

**Purpose:** Persist trained models across restarts  
**Benefits:** No need to retrain after container restart  
**Storage:** Binary (BYTEA) serialized with joblib

### 6. `radiators`
Current radiator level settings.

```sql
CREATE TABLE radiators (
    id SERIAL PRIMARY KEY,
    room VARCHAR(50) UNIQUE NOT NULL,
    level FLOAT DEFAULT 0,
    updated_at TIMESTAMP DEFAULT NOW()
);
```

**Purpose:** Store current radiator settings  
**Updated:** When radiator level changes

### 7. `ai_logs` (Legacy)
General purpose activity log.

```sql
CREATE TABLE ai_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT NOW(),
    action VARCHAR(50),
    room VARCHAR(50),
    data JSONB
);
```

**Purpose:** Backward compatibility, general logging  
**Note:** Specific tables (training_history, predictions) are preferred

## Data Flow

### Training Flow
```
1. POST /train
2. Load model from ml_models table
3. Get previous temp from room_states
4. Train model with new data
5. Save model to ml_models table
6. Update ai_metrics table
7. Insert into training_history
8. Insert into room_states
```

### Prediction Flow
```
1. POST /predict
2. Load model from ml_models table
3. Predict optimal radiator level
4. Update ai_metrics table
5. Insert into predictions table
6. Insert into room_states
```

## New API Endpoints

### Historical Data Export

#### GET /history/{room}?hours=24
Get historical data for a specific room.

**Response:**
```json
{
  "room": "Sovrum",
  "hours": 24,
  "data_points": 48,
  "data": [
    {
      "room": "Sovrum",
      "current_temp": 19.5,
      "target_temp": 20.0,
      "radiator_level": 3.5,
      "outdoor_temp": -2.0,
      "forecast_temp": -1.5,
      "timestamp": "2025-11-06T10:00:00"
    },
    ...
  ]
}
```

#### GET /history?hours=24
Get historical data for all rooms.

**Response:**
```json
{
  "hours": 24,
  "total_data_points": 192,
  "rooms": {
    "Sovrum": [...],
    "Kontor": [...],
    "Vardagsrum": [...],
    "Badrum": [...]
  }
}
```

#### GET /export/csv/{room}?hours=168
Export room data as CSV file (default 1 week).

**Returns:** CSV file download  
**Use case:** Import into Excel, Google Sheets, matplotlib, etc.

#### GET /training/history/{room}
Get training statistics for a room.

**Response:**
```json
{
  "room": "Sovrum",
  "stats": {
    "total_training_events": 150,
    "avg_temp_delta": 0.234,
    "avg_prediction_error": 0.156,
    "first_training": "2025-11-01T10:00:00",
    "last_training": "2025-11-06T14:30:00"
  }
}
```

## Benefits of Database Storage

### ✅ Data Persistence
- **Models survive restarts** - No need to retrain
- **Statistics preserved** - Historical performance tracking
- **Complete audit trail** - Every action logged

### ✅ Historical Analysis
- **Time-series data** - Graph temperature trends
- **Performance tracking** - See model improvement over time
- **Pattern detection** - Identify heating patterns

### ✅ Scalability
- **Indexed queries** - Fast data retrieval
- **Concurrent access** - Multiple services can query
- **Data integrity** - ACID transactions

### ✅ Assignment Benefits
- **Graph generation** - Export CSV for analysis
- **Statistical evidence** - Prove AI learning
- **Complete documentation** - Full data trail

## Usage Examples

### Python - Get historical data for graphing
```python
import requests
import matplotlib.pyplot as plt
from datetime import datetime

# Get 24 hours of data
response = requests.get('http://localhost:8000/history/Sovrum?hours=24')
data = response.json()['data']

# Extract data for plotting
times = [datetime.fromisoformat(d['timestamp']) for d in data]
temps = [d['current_temp'] for d in data]
targets = [d['target_temp'] for d in data]
levels = [d['radiator_level'] for d in data]

# Plot temperature over time
plt.figure(figsize=(12, 6))
plt.plot(times, temps, label='Current Temp')
plt.plot(times, targets, label='Target Temp', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Temperature (°C)')
plt.title('Bedroom Temperature Over 24 Hours')
plt.legend()
plt.grid(True)
plt.savefig('temp_graph.png')
```

### SQL - Direct database queries
```sql
-- Get average temperature by hour of day
SELECT 
    EXTRACT(HOUR FROM timestamp) as hour,
    room,
    AVG(current_temp) as avg_temp,
    AVG(radiator_level) as avg_level
FROM room_states
WHERE timestamp > NOW() - INTERVAL '7 days'
GROUP BY hour, room
ORDER BY room, hour;

-- See model improvement over time
SELECT 
    room,
    DATE(timestamp) as date,
    COUNT(*) as training_count,
    AVG(ABS(temperature_delta - predicted_delta)) as avg_error
FROM training_history
WHERE predicted_delta IS NOT NULL
GROUP BY room, DATE(timestamp)
ORDER BY room, date;

-- Efficiency metrics
SELECT 
    room,
    COUNT(*) as total_predictions,
    SUM(CASE WHEN adjustment_made THEN 1 ELSE 0 END) as adjustments,
    ROUND(100.0 * SUM(CASE WHEN adjustment_made THEN 1 ELSE 0 END) / COUNT(*), 2) as efficiency_pct
FROM predictions
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY room;
```

## Migration Notes

### Removed Files
- ❌ `last_temps.json` - Replaced by `room_states` table
- ❌ `ai_stats.json` - Replaced by `ai_metrics` table
- ❌ `models/*.pkl` - Replaced by `ml_models` table

### Backward Compatibility
- Old endpoints still work
- Data automatically migrates to database
- No manual migration needed

### First Startup
Database tables are created automatically on first run via `db.init_database()`.

## Backup and Recovery

### Backup Database
```bash
docker-compose exec postgres pg_dump -U postgres radiators > backup.sql
```

### Restore Database
```bash
cat backup.sql | docker-compose exec -T postgres psql -U postgres radiators
```

### Export All Data
```bash
# Export all room data as CSV
for room in Sovrum Kontor Vardagsrum Badrum; do
    curl "http://localhost:8000/export/csv/$room?hours=720" > "${room}_data.csv"
done
```

## Performance Considerations

- **Indexes:** room and timestamp columns are indexed for fast queries
- **Cleanup:** Consider archiving old data (>6 months) for performance
- **Query limits:** Historical queries default to 24 hours, adjust as needed

## For Assignment Submission

### Generate Graphs
1. Collect data: `GET /history/{room}?hours=168` (1 week)
2. Export CSV: `GET /export/csv/{room}?hours=168`
3. Import to Python/Excel and create graphs showing:
   - Temperature vs Time
   - Radiator Level vs Time
   - Prediction Accuracy over Time
   - Model MAE improvement

### Show Learning Progress
```sql
-- Query to show AI learning improvement
SELECT 
    DATE(timestamp) as training_date,
    COUNT(*) as samples,
    AVG(ABS(temperature_delta - predicted_delta)) as prediction_error
FROM training_history
WHERE room = 'Sovrum' AND predicted_delta IS NOT NULL
GROUP BY DATE(timestamp)
ORDER BY training_date;
```

This demonstrates that prediction error decreases as the model trains!
