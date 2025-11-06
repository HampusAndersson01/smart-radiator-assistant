# Database module for Smart Radiator AI
import psycopg2
from psycopg2.extras import RealDictCursor
import json
from datetime import datetime
from typing import Optional, Dict, List
import os

DATABASE_URL = os.getenv("DATABASE_URL")

def get_connection():
    """Get a database connection"""
    if not DATABASE_URL:
        raise Exception("DATABASE_URL not configured")
    return psycopg2.connect(DATABASE_URL)

def init_database():
    """Initialize all required database tables"""
    if not DATABASE_URL:
        print("⚠️  DATABASE_URL not configured - skipping database initialization")
        return
        
    conn = get_connection()
    cursor = conn.cursor()
    
    # Room states table - stores current and historical temperature readings
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS room_states (
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
        CREATE INDEX IF NOT EXISTS idx_room_states_room ON room_states(room);
        CREATE INDEX IF NOT EXISTS idx_room_states_timestamp ON room_states(timestamp);
    """)
    
    # AI metrics table - stores model performance metrics per room
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ai_metrics (
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
    """)
    
    # Training history table - detailed log of each training event
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_history (
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
        CREATE INDEX IF NOT EXISTS idx_training_history_room ON training_history(room);
        CREATE INDEX IF NOT EXISTS idx_training_history_timestamp ON training_history(timestamp);
    """)
    
    # Predictions table - log of all predictions and recommendations
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
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
        CREATE INDEX IF NOT EXISTS idx_predictions_room ON predictions(room);
        CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
    """)
    
    # ML models table - store serialized models
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ml_models (
            room VARCHAR(50) PRIMARY KEY,
            model_data BYTEA NOT NULL,
            version INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW()
        );
    """)
    
    # Keep the existing ai_logs table for backward compatibility
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ai_logs (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT NOW(),
            action VARCHAR(50),
            room VARCHAR(50),
            data JSONB
        );
    """)
    
    # Radiators table (if not exists)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS radiators (
            id SERIAL PRIMARY KEY,
            room VARCHAR(50) UNIQUE NOT NULL,
            level FLOAT DEFAULT 0,
            updated_at TIMESTAMP DEFAULT NOW()
        );
    """)
    
    # Radiator history table - track all level changes
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS radiator_history (
            id SERIAL PRIMARY KEY,
            room VARCHAR(50) NOT NULL,
            level FLOAT NOT NULL,
            previous_level FLOAT,
            changed_by VARCHAR(50),
            timestamp TIMESTAMP DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_radiator_history_room ON radiator_history(room);
        CREATE INDEX IF NOT EXISTS idx_radiator_history_timestamp ON radiator_history(timestamp);
    """)
    
    conn.commit()
    cursor.close()
    conn.close()
    print("✅ Database schema initialized successfully")

def save_room_state(room: str, current_temp: float, target_temp: float, 
                     radiator_level: float, outdoor_temp: float = None, 
                     forecast_temp: float = None):
    """Save a room state reading to the database"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO room_states 
        (room, current_temp, target_temp, radiator_level, outdoor_temp, forecast_temp)
        VALUES (%s, %s, %s, %s, %s, %s)
    """, (room, current_temp, target_temp, radiator_level, outdoor_temp, forecast_temp))
    
    conn.commit()
    cursor.close()
    conn.close()

def get_latest_temp(room: str) -> Optional[float]:
    """Get the latest temperature reading for a room"""
    if not DATABASE_URL:
        return None
        
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT current_temp FROM room_states 
            WHERE room = %s 
            ORDER BY timestamp DESC 
            LIMIT 1
        """, (room,))
        
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        
        return result[0] if result else None
    except Exception as e:
        print(f"Error getting latest temp for {room}: {e}")
        return None

def save_training_event(room: str, current_temp: float, target_temp: float,
                        radiator_level: float, temperature_delta: float,
                        outdoor_temp: float = None, forecast_temp: float = None,
                        predicted_delta: float = None, hour_of_day: int = None):
    """Save a training event to the database"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO training_history 
        (room, current_temp, target_temp, radiator_level, temperature_delta,
         outdoor_temp, forecast_temp, predicted_delta, hour_of_day)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (room, current_temp, target_temp, radiator_level, temperature_delta,
          outdoor_temp, forecast_temp, predicted_delta, hour_of_day))
    
    conn.commit()
    cursor.close()
    conn.close()

def save_prediction(room: str, current_temp: float, target_temp: float,
                    current_radiator_level: float, recommended_level: float,
                    predicted_error: float, adjustment_made: bool,
                    outdoor_temp: float = None, forecast_temp: float = None):
    """Save a prediction to the database"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO predictions 
        (room, current_temp, target_temp, current_radiator_level, 
         recommended_level, predicted_error, adjustment_made,
         outdoor_temp, forecast_temp)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (room, current_temp, target_temp, current_radiator_level,
          recommended_level, predicted_error, adjustment_made,
          outdoor_temp, forecast_temp))
    
    conn.commit()
    cursor.close()
    conn.close()

def update_ai_metrics(room: str, mae: float = None, rmse: float = None,
                      r2_score: float = None, training_samples: int = None,
                      predictions_made: int = None, adjustments_made: int = None,
                      total_error: float = None):
    """Update AI metrics for a room"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if metrics exist for this room
    cursor.execute("SELECT id FROM ai_metrics WHERE room = %s", (room,))
    exists = cursor.fetchone()
    
    if exists:
        # Update existing metrics
        updates = []
        values = []
        
        if mae is not None:
            updates.append("mae = %s")
            values.append(mae)
        if rmse is not None:
            updates.append("rmse = %s")
            values.append(rmse)
        if r2_score is not None:
            updates.append("r2_score = %s")
            values.append(r2_score)
        if training_samples is not None:
            updates.append("training_samples = %s")
            values.append(training_samples)
        if predictions_made is not None:
            updates.append("predictions_made = %s")
            values.append(predictions_made)
        if adjustments_made is not None:
            updates.append("adjustments_made = %s")
            values.append(adjustments_made)
        if total_error is not None:
            updates.append("total_error = %s")
            values.append(total_error)
        
        updates.append("last_updated = NOW()")
        values.append(room)
        
        cursor.execute(f"""
            UPDATE ai_metrics 
            SET {', '.join(updates)}
            WHERE room = %s
        """, values)
    else:
        # Insert new metrics
        cursor.execute("""
            INSERT INTO ai_metrics 
            (room, mae, rmse, r2_score, training_samples, predictions_made, 
             adjustments_made, total_error)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (room, mae or 0, rmse or 0, r2_score or 0, training_samples or 0,
              predictions_made or 0, adjustments_made or 0, total_error or 0))
    
    conn.commit()
    cursor.close()
    conn.close()

def get_ai_metrics(room: str = None) -> Dict:
    """Get AI metrics for a specific room or all rooms"""
    if not DATABASE_URL:
        return {} if room else {}
    
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        if room:
            cursor.execute("SELECT * FROM ai_metrics WHERE room = %s", (room,))
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            return dict(result) if result else {}
        else:
            cursor.execute("SELECT * FROM ai_metrics")
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            return {row['room']: dict(row) for row in results}
    except Exception as e:
        print(f"Error getting AI metrics: {e}")
        return {} if room else {}

def save_model(room: str, model_bytes: bytes):
    """Save a serialized model to the database"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("""
        INSERT INTO ml_models (room, model_data, updated_at)
        VALUES (%s, %s, NOW())
        ON CONFLICT (room) 
        DO UPDATE SET model_data = %s, version = ml_models.version + 1, updated_at = NOW()
    """, (room, model_bytes, model_bytes))
    
    conn.commit()
    cursor.close()
    conn.close()

def load_model(room: str) -> Optional[bytes]:
    """Load a serialized model from the database"""
    if not DATABASE_URL:
        return None
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT model_data FROM ml_models WHERE room = %s", (room,))
        result = cursor.fetchone()
        
        cursor.close()
        conn.close()
        
        return result[0] if result else None
    except Exception as e:
        print(f"Error loading model for {room}: {e}")
        return None

def get_historical_data(room: str = None, hours: int = 24) -> List[Dict]:
    """Get historical temperature and radiator level data for graphing"""
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    if room:
        cursor.execute("""
            SELECT room, current_temp, target_temp, radiator_level, 
                   outdoor_temp, forecast_temp, timestamp
            FROM room_states
            WHERE room = %s AND timestamp > NOW() - INTERVAL '%s hours'
            ORDER BY timestamp ASC
        """, (room, hours))
    else:
        cursor.execute("""
            SELECT room, current_temp, target_temp, radiator_level, 
                   outdoor_temp, forecast_temp, timestamp
            FROM room_states
            WHERE timestamp > NOW() - INTERVAL '%s hours'
            ORDER BY room, timestamp ASC
        """, (hours,))
    
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    
    return [dict(row) for row in results]

def get_training_stats(room: str = None) -> Dict:
    """Get training statistics for analysis"""
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    if room:
        cursor.execute("""
            SELECT 
                COUNT(*) as total_training_events,
                AVG(ABS(temperature_delta)) as avg_temp_delta,
                AVG(ABS(temperature_delta - COALESCE(predicted_delta, temperature_delta))) as avg_prediction_error,
                MIN(timestamp) as first_training,
                MAX(timestamp) as last_training
            FROM training_history
            WHERE room = %s
        """, (room,))
        result = cursor.fetchone()
        cursor.close()
        conn.close()
        return dict(result) if result else {}
    else:
        cursor.execute("""
            SELECT 
                room,
                COUNT(*) as total_training_events,
                AVG(ABS(temperature_delta)) as avg_temp_delta,
                MIN(timestamp) as first_training,
                MAX(timestamp) as last_training
            FROM training_history
            GROUP BY room
        """)
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return {row['room']: dict(row) for row in results}

def get_prediction_stats(room: str = None, hours: int = 24) -> Dict:
    """Get prediction statistics"""
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    if room:
        cursor.execute("""
            SELECT 
                COUNT(*) as total_predictions,
                SUM(CASE WHEN adjustment_made THEN 1 ELSE 0 END) as adjustments_made,
                AVG(predicted_error) as avg_predicted_error,
                AVG(ABS(recommended_level - current_radiator_level)) as avg_level_change
            FROM predictions
            WHERE room = %s AND timestamp > NOW() - INTERVAL '%s hours'
        """, (room, hours))
    else:
        cursor.execute("""
            SELECT 
                room,
                COUNT(*) as total_predictions,
                SUM(CASE WHEN adjustment_made THEN 1 ELSE 0 END) as adjustments_made,
                AVG(predicted_error) as avg_predicted_error
            FROM predictions
            WHERE timestamp > NOW() - INTERVAL '%s hours'
            GROUP BY room
        """, (hours,))
    
    result = cursor.fetchone() if room else cursor.fetchall()
    cursor.close()
    conn.close()
    
    if room:
        return dict(result) if result else {}
    else:
        return {row['room']: dict(row) for row in result} if result else {}

def update_radiator_level(room: str, level: float, changed_by: str = "AI"):
    """Update radiator level and log the change to history"""
    if not DATABASE_URL:
        return
    
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Get current level
        cursor.execute("SELECT level FROM radiators WHERE room = %s", (room,))
        result = cursor.fetchone()
        previous_level = result[0] if result else None
        
        # Update current level
        cursor.execute("""
            INSERT INTO radiators (room, level, updated_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (room) 
            DO UPDATE SET level = %s, updated_at = NOW()
        """, (room, level, level))
        
        # Log to history
        cursor.execute("""
            INSERT INTO radiator_history (room, level, previous_level, changed_by)
            VALUES (%s, %s, %s, %s)
        """, (room, level, previous_level, changed_by))
        
        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ Updated {room} radiator: {previous_level} → {level} (by {changed_by})")
    except Exception as e:
        print(f"Error updating radiator level: {e}")

def get_radiator_history(room: str = None, hours: int = 24) -> List[Dict]:
    """Get radiator level change history"""
    if not DATABASE_URL:
        return []
    
    try:
        conn = get_connection()
        cursor = conn.cursor(cursor_factory=RealDictCursor)
        
        if room:
            cursor.execute("""
                SELECT room, level, previous_level, changed_by, timestamp
                FROM radiator_history
                WHERE room = %s AND timestamp > NOW() - INTERVAL '%s hours'
                ORDER BY timestamp DESC
            """, (room, hours))
        else:
            cursor.execute("""
                SELECT room, level, previous_level, changed_by, timestamp
                FROM radiator_history
                WHERE timestamp > NOW() - INTERVAL '%s hours'
                ORDER BY timestamp DESC
            """, (hours,))
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return [dict(row) for row in results]
    except Exception as e:
        print(f"Error getting radiator history: {e}")
        return []
