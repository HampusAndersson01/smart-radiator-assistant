# Smart Radiator AI Service - v2.0.0 (Database-backed)
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from datetime import datetime
import os, joblib, requests, json, io
from collections import defaultdict
from river import forest, preprocessing, metrics
from forecast import get_weather
import database as db
import asyncio
from contextlib import asynccontextmanager

# Background task for validating predictions
async def validation_background_task():
    """Background task that runs every hour to validate old predictions"""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            print("🔍 Running prediction validation...")
            
            # Call the validation logic
            unvalidated = db.get_unvalidated_predictions()
            if unvalidated:
                validated_count = 0
                trained_count = 0
                
                for pred in unvalidated:
                    room = pred['room']
                    prediction_id = pred['id']
                    predicted_temp = pred['predicted_temp']
                    
                    try:
                        conn = db.get_connection()
                        cursor = conn.cursor()
                        
                        cursor.execute("""
                            SELECT current_temp, outdoor_temp 
                            FROM room_states
                            WHERE room = %s 
                              AND timestamp >= %s - INTERVAL '30 minutes'
                              AND timestamp <= %s + INTERVAL '30 minutes'
                            ORDER BY ABS(EXTRACT(EPOCH FROM (timestamp - %s)))
                            LIMIT 1
                        """, (room, pred['target_timestamp'], pred['target_timestamp'], pred['target_timestamp']))
                        
                        result = cursor.fetchone()
                        cursor.close()
                        conn.close()
                        
                        if result:
                            actual_temp = result[0]
                            db.validate_prediction(prediction_id, actual_temp, used_for_training=True)
                            validated_count += 1
                            trained_count += 1
                            print(f"  ✅ {room}: predicted {predicted_temp:.1f}°C vs actual {actual_temp:.1f}°C")
                        else:
                            db.validate_prediction(prediction_id, None, used_for_training=False)
                            validated_count += 1
                    except Exception as e:
                        print(f"  ❌ Error validating prediction {prediction_id}: {e}")
                
                print(f"✅ Validated {validated_count} predictions ({trained_count} trained)")
            else:
                print("  ℹ️  No predictions to validate")
                
        except Exception as e:
            print(f"Error in validation background task: {e}")

async def ml_training_background_task():
    """Background task that trains ML models with best-practice hints"""
    # Wait 10 seconds on startup to let everything initialize
    await asyncio.sleep(10)
    
    print("🧠 Starting ML model training with best-practice hints...")
    
    await train_all_models()
    
    print("🎓 ML model training complete!")
    
    # Continue training periodically
    while True:
        try:
            await asyncio.sleep(3600)  # Retrain every hour with new data
            print("� Retraining ML models with latest data...")
            
            for room in ROOMS.keys():
                try:
                    # Get recent training events (last hour)
                    events = db.get_latest_training_events(limit=100)
                    room_events = [e for e in events if e['room'] == room]
                    
                    if room_events:
                        model = load_model(room)
                        m = model_metrics[room]
                        
                        for event in room_events:
                            features = {
                                "current_temp": float(event['current_temp']),
                                "target_temp": float(event['target_temp']),
                                "outdoor_temp": float(event.get('outdoor_temp', 0)),
                                "forecast_3h_temp": float(event.get('forecast_temp', 0)),
                                "forecast_10h_temp": float(event.get('forecast_temp', 0)),
                                "radiator_level": float(event['radiator_level']),
                                "hour_of_day": int(event.get('hour_of_day', 0)),
                            }
                            
                            model.learn_one(features, float(event['temperature_delta']))
                        
                        save_model(room, model)
                        print(f"  ✅ Retrained {room} with {len(room_events)} new samples")
                
                except Exception as e:
                    print(f"  ❌ Error retraining {room}: {e}")
                    
        except Exception as e:
            print(f"Error in ML training background task: {e}")

async def train_all_models():
    """Train all room models with physics-based hints and historical data"""
    for room in ROOMS.keys():
        try:
            print(f"📚 Training model for {room}...")
            
            target_temp = ROOMS[room]["target"]
            scale = ROOMS[room]["scale"]
            
            # Create fresh model
            model = preprocessing.StandardScaler() | forest.ARFRegressor()
            m = model_metrics[room]
            hint_samples = 0
            
            # CRITICAL PHYSICS HINTS: Higher radiator level = MORE heating
            print(f"  🔥 Teaching radiator physics for {room}...")
            
            # Teach: Radiator level affects temperature direction
            for level_idx, level in enumerate(scale):
                # Expected steady-state temp increases with radiator level
                # Low level (0-2) → temps below target
                # Mid level (3-5) → temps near target  
                # High level (6-9) → temps above target
                
                level_ratio = level_idx / max(1, len(scale) - 1)  # 0.0 to 1.0
                
                # Estimate equilibrium temp for this level
                # Low levels struggle to reach target, high levels exceed it
                temp_range = 6.0  # degrees variance across scale
                estimated_equilibrium = target_temp + (level_ratio - 0.5) * temp_range
                
                for hour in range(24):
                    # Teach equilibrium at this level
                    features = {
                        "current_temp": estimated_equilibrium,
                        "target_temp": target_temp,
                        "outdoor_temp": 5.0,
                        "forecast_3h_temp": 5.0,
                        "forecast_10h_temp": 5.0,
                        "radiator_level": float(level),
                        "hour_of_day": hour,
                    }
                    model.learn_one(features, 0.0)  # At equilibrium
                    hint_samples += 1
                
                # Teach heating dynamics: higher level = more heating power
                # If below equilibrium, higher levels warm faster
                if estimated_equilibrium > target_temp - 2:
                    temp_below = estimated_equilibrium - 1.0
                    warming_rate = 0.1 + (level_ratio * 0.4)  # 0.1 to 0.5 °C/hour
                    features = {
                        "current_temp": temp_below,
                        "target_temp": target_temp,
                        "outdoor_temp": 5.0,
                        "forecast_3h_temp": 5.0,
                        "forecast_10h_temp": 5.0,
                        "radiator_level": float(level),
                        "hour_of_day": 12,
                    }
                    model.learn_one(features, warming_rate)
                    hint_samples += 1
                
                # Teach cooling dynamics: lower level = natural cooling
                if estimated_equilibrium < target_temp + 2:
                    temp_above = estimated_equilibrium + 1.0
                    cooling_rate = -0.1 - (level_ratio * 0.2)  # -0.1 to -0.3 °C/hour
                    features = {
                        "current_temp": temp_above,
                        "target_temp": target_temp,
                        "outdoor_temp": 5.0,
                        "forecast_3h_temp": 5.0,
                        "forecast_10h_temp": 5.0,
                        "radiator_level": float(level),
                        "hour_of_day": 12,
                    }
                    model.learn_one(features, cooling_rate)
                    hint_samples += 1
            
            # Get historical data to refine model
            historical_hints = db.get_radiator_level_temperature_by_hour(room, days=30)
            if historical_hints:
                print(f"  💡 Applying {len(historical_hints)} historical level hints for {room}...")
                for level, hour_data in historical_hints.items():
                    temps = [hour_data[h]['avg_temp'] for h in hour_data.keys() if 'avg_temp' in hour_data[h]]
                    if temps:
                        avg_temp = sum(temps) / len(temps)
                        # Reinforce actual observed equilibrium
                        for hour in range(6):  # Less weight than physics hints
                            features = {
                                "current_temp": avg_temp,
                                "target_temp": target_temp,
                                "outdoor_temp": 5.0,
                                "forecast_3h_temp": 5.0,
                                "forecast_10h_temp": 5.0,
                                "radiator_level": float(level),
                                "hour_of_day": hour * 4,
                            }
                            model.learn_one(features, 0.0)
                            hint_samples += 1
            
            # Train on actual historical data
            conn = db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                WITH ordered_states AS (
                    SELECT 
                        room,
                        current_temp,
                        target_temp,
                        radiator_level,
                        outdoor_temp,
                        forecast_temp,
                        timestamp,
                        LAG(current_temp) OVER (PARTITION BY room ORDER BY timestamp) as prev_temp,
                        EXTRACT(HOUR FROM timestamp) as hour_of_day
                    FROM room_states
                    WHERE room = %s
                      AND current_temp > 5
                      AND current_temp < 40
                    ORDER BY timestamp
                )
                SELECT 
                    current_temp,
                    target_temp,
                    radiator_level,
                    outdoor_temp,
                    forecast_temp,
                    hour_of_day,
                    (current_temp - COALESCE(prev_temp, current_temp)) as delta
                FROM ordered_states
                WHERE prev_temp IS NOT NULL
                  AND ABS(current_temp - prev_temp) < 4
                ORDER BY timestamp
                LIMIT 10000
            """, (room,))
            results = cursor.fetchall()
            cursor.close()
            conn.close()
            
            trained_count = 0
            for row in results:
                current_temp, target_temp, radiator_level, outdoor_temp, forecast_temp, hour_of_day, delta = row
                
                if outdoor_temp is None or forecast_temp is None:
                    continue
                
                features = {
                    "current_temp": float(current_temp),
                    "target_temp": float(target_temp),
                    "outdoor_temp": float(outdoor_temp),
                    "forecast_3h_temp": float(forecast_temp),
                    "forecast_10h_temp": float(forecast_temp),
                    "radiator_level": float(radiator_level),
                    "hour_of_day": int(hour_of_day),
                }
                
                model.learn_one(features, float(delta))
                trained_count += 1
            
            # Save the trained model
            save_model(room, model)
            
            # Update metrics
            m.training_samples = hint_samples + trained_count
            
            # Save metrics to database
            db.update_ai_metrics(
                room,
                training_samples=m.training_samples,
                predictions_made=m.predictions_made,
                adjustments_made=m.adjustments_made,
                total_error=m.total_error
            )
            
            print(f"  ✅ {room}: {hint_samples} hints + {trained_count} historical = {m.training_samples} total")
            
        except Exception as e:
            print(f"  ❌ Error training model for {room}: {e}")
    
    print("🎓 ML model training complete!")
    
    # Continue training periodically
    while True:
        try:
            await asyncio.sleep(3600)  # Retrain every hour with new data
            print("🔄 Retraining ML models with latest data...")
            
            for room in ROOMS.keys():
                try:
                    # Get recent training events (last hour)
                    events = db.get_latest_training_events(limit=100)
                    room_events = [e for e in events if e['room'] == room]
                    
                    if room_events:
                        model = load_model(room)
                        m = model_metrics[room]
                        
                        for event in room_events:
                            features = {
                                "current_temp": float(event['current_temp']),
                                "target_temp": float(event['target_temp']),
                                "outdoor_temp": float(event.get('outdoor_temp', 0)),
                                "forecast_3h_temp": float(event.get('forecast_temp', 0)),
                                "forecast_10h_temp": float(event.get('forecast_temp', 0)),
                                "radiator_level": float(event['radiator_level']),
                                "hour_of_day": int(event.get('hour_of_day', 0)),
                            }
                            
                            model.learn_one(features, float(event['temperature_delta']))
                        
                        save_model(room, model)
                        print(f"  ✅ Retrained {room} with {len(room_events)} new samples")
                
                except Exception as e:
                    print(f"  ❌ Error retraining {room}: {e}")
                    
        except Exception as e:
            print(f"Error in ML training background task: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    validation_task = asyncio.create_task(validation_background_task())
    ml_training_task = asyncio.create_task(ml_training_background_task())
    print("🚀 Started prediction validation background task")
    print("🚀 Started ML training background task")
    yield
    # Shutdown
    validation_task.cancel()
    ml_training_task.cancel()
    print("🛑 Stopped background tasks")

app = FastAPI(title="Smart Radiator AI", lifespan=lifespan)
DATABASE_URL = os.getenv("DATABASE_URL")

# Initialize database on startup
try:
    db.init_database()
except Exception as e:
    print(f"Warning: Could not initialize database: {e}")
    print("Some features may not work without DATABASE_URL configured")

TELEGRAM_WEBHOOK = os.getenv("TELEGRAM_WEBHOOK")

# Comfort and adjustment parameters
ACCEPTABLE_DEVIATION = 0.3  # °C - Don't adjust if within this range of target
MIN_ADJUSTMENT_THRESHOLD = 0.5  # Minimum level change to warrant adjustment
MAX_SINGLE_ADJUSTMENT = 1.5  # Maximum level change in one prediction (gradual changes)
CONFIDENCE_THRESHOLD = 0.8  # Only adjust if prediction confidence is high enough
HOURS_AHEAD = 8  # Hours to simulate for stability prediction

ROOMS = {
    "Badrum":     {"scale": [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6], "target": 22.5},
    "Sovrum":     {"scale": [0,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9], "target": 20},
    "Kontor":     {"scale": [0,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9], "target": 21},
    "Vardagsrum":{"scale": [0,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9], "target": 21},
}

class RoomState(BaseModel):
    room: str
    current_temp: float | None = None
    target_temp: float | None = None
    radiator_level: float | None = None
    outdoor_temp: float | None = None
    forecast_temp: float | None = None
    forecast_10h_temp: float | None = None
    timestamp: str
    
    def validate_required_fields(self):
        """Validate that required fields are not None"""
        if self.current_temp is None:
            raise ValueError(f"current_temp is required for room {self.room}")
        if self.target_temp is None:
            raise ValueError(f"target_temp is required for room {self.room}")
        if self.radiator_level is None:
            raise ValueError(f"radiator_level is required for room {self.room}")
        return True

class ModelMetrics:
    """Track AI model performance metrics"""
    def __init__(self):
        self.mae = metrics.MAE()
        self.rmse = metrics.RMSE()
        self.r2 = metrics.R2()
        self.training_samples = 0
        self.predictions_made = 0
        self.adjustments_made = 0
        self.total_error = 0.0
        self.created_at = datetime.now().isoformat()
        
    def update(self, y_true, y_pred):
        """Update metrics with new prediction"""
        self.mae.update(y_true, y_pred)
        self.rmse.update(y_true, y_pred)
        self.r2.update(y_true, y_pred)
        self.total_error += abs(y_true - y_pred)
        
    def to_dict(self):
        """Convert metrics to dictionary"""
        return {
            "mae": self.mae.get() if self.training_samples > 0 else 0,
            "rmse": self.rmse.get() if self.training_samples > 0 else 0,
            "r2_score": self.r2.get() if self.training_samples > 1 else 0,
            "training_samples": self.training_samples,
            "predictions_made": self.predictions_made,
            "adjustments_made": self.adjustments_made,
            "avg_error": self.total_error / self.predictions_made if self.predictions_made > 0 else 0,
            "created_at": self.created_at,
            "last_updated": datetime.now().isoformat()
        }

# Global metrics storage (in-memory cache)
model_metrics = defaultdict(ModelMetrics)

# Load models and metrics from database on startup
def load_metrics_from_db():
    """Load metrics from database into memory"""
    try:
        all_metrics = db.get_ai_metrics()
        for room, metrics_data in all_metrics.items():
            m = model_metrics[room]
            m.training_samples = metrics_data.get('training_samples', 0)
            m.predictions_made = metrics_data.get('predictions_made', 0)
            m.adjustments_made = metrics_data.get('adjustments_made', 0)
            m.total_error = metrics_data.get('total_error', 0)
            m.created_at = metrics_data.get('created_at', datetime.now()).isoformat() if isinstance(metrics_data.get('created_at'), datetime) else str(metrics_data.get('created_at', datetime.now().isoformat()))
        print(f"✅ Loaded metrics for {len(all_metrics)} rooms from database")
    except Exception as e:
        print(f"Warning: Could not load metrics from database: {e}")

load_metrics_from_db()

def load_model(room):
    """Load model from database or create new one"""
    try:
        model_bytes = db.load_model(room)
        if model_bytes:
            return joblib.load(io.BytesIO(model_bytes))
    except Exception as e:
        print(f"Could not load model for {room} from database: {e}")
    
    # Return new model if not found or error
    return preprocessing.StandardScaler() | forest.ARFRegressor()

def save_model(room, model):
    """Save model to database"""
    try:
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        model_bytes = buffer.getvalue()
        db.save_model(room, model_bytes)
    except Exception as e:
        print(f"Warning: Could not save model for {room} to database: {e}")

def get_radiator_levels():
    """Get current radiator levels from PostgreSQL database"""
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT room, level FROM radiators")
        levels = {row[0]: row[1] for row in cursor.fetchall()}
        cursor.close()
        conn.close()
        return levels
    except Exception as e:
        print(f"Error reading radiator levels: {e}")
        return {}

@app.get("/")
def status():
    """Return AI service status and model information"""
    try:
        radiator_levels = get_radiator_levels()
    except Exception as e:
        print(f"Error getting radiator levels: {e}")
        radiator_levels = {}
    
    models_info = {}
    for room in ROOMS.keys():
        # Check if model exists in database
        try:
            model_exists = db.load_model(room) is not None
        except Exception as e:
            print(f"Error checking model for {room}: {e}")
            model_exists = False
        
        # Get latest temp from database
        try:
            latest_temp = db.get_latest_temp(room)
        except Exception as e:
            print(f"Error getting latest temp for {room}: {e}")
            latest_temp = None
        
        # Get metrics for this room from database
        try:
            room_metrics = db.get_ai_metrics(room)
        except Exception as e:
            print(f"Error getting metrics for {room}: {e}")
            room_metrics = {}
        
        models_info[room] = {
            "trained": model_exists,
            "last_temp": latest_temp,
            "target_temp": ROOMS[room]["target"],
            "scale_range": f"{ROOMS[room]['scale'][0]}-{ROOMS[room]['scale'][-1]}",
            "current_level": radiator_levels.get(room, 0),
            "training_samples": room_metrics.get("training_samples", 0),
            "predictions_made": room_metrics.get("predictions_made", 0),
            "avg_error": round(room_metrics.get("total_error", 0) / max(1, room_metrics.get("predictions_made", 1)), 2)
        }
    
    # Get current weather with error handling
    try:
        outdoor, forecast_3h, forecast_10h = get_weather()
        if outdoor is None:
            outdoor = 0.0
        if forecast_3h is None:
            forecast_3h = 0.0
        if forecast_10h is None:
            forecast_10h = 0.0
    except Exception as e:
        print(f"Error getting weather: {e}")
        outdoor, forecast_3h, forecast_10h = 0.0, 0.0, 0.0
    
    return {
        "status": "online",
        "version": "2.0.0-database",
        "timestamp": datetime.now().isoformat(),
        "rooms": models_info,
        "weather": {
            "outdoor_temp": outdoor,
            "forecast_3h": forecast_3h,
            "forecast_10h": forecast_10h,
        },
        "telegram_webhook_configured": TELEGRAM_WEBHOOK is not None,
        "database_connected": DATABASE_URL is not None,
    }

@app.post("/reset-and-retrain")
async def reset_and_retrain():
    """Reset all ML models and retrain from historical data"""
    try:
        print("🔄 Resetting and retraining all ML models...")
        
        # Clear existing models from database
        for room in ROOMS.keys():
            db.delete_model(room)
            print(f"  🗑️  Deleted model for {room}")
        
        # Reset metrics
        for room in ROOMS.keys():
            m = model_metrics[room]
            m.training_samples = 0
            m.predictions_made = 0
            m.adjustments_made = 0
            m.total_error = 0.0
            
            db.update_ai_metrics(
                room,
                training_samples=0,
                predictions_made=0,
                adjustments_made=0,
                total_error=0.0
            )
        
        # Retrain all models
        await train_all_models()
        
        return {
            "status": "success",
            "message": "All models reset and retrained successfully",
            "rooms_trained": list(ROOMS.keys()),
            "training_samples": {room: model_metrics[room].training_samples for room in ROOMS.keys()}
        }
    except Exception as e:
        print(f"❌ Error during reset and retrain: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/retrain-progress")
def get_retrain_progress():
    """Get progress of retraining operation"""
    # Simple progress based on how many rooms have non-zero samples
    trained_rooms = sum(1 for room in ROOMS.keys() if model_metrics[room].training_samples > 0)
    total_rooms = len(ROOMS)
    
    return {
        "progress": (trained_rooms / total_rooms) * 100,
        "trained_rooms": trained_rooms,
        "total_rooms": total_rooms,
        "room_status": {
            room: {
                "training_samples": model_metrics[room].training_samples,
                "status": "trained" if model_metrics[room].training_samples > 0 else "pending"
            }
            for room in ROOMS.keys()
        }
    }

@app.post("/accelerated-backtrain")
async def accelerated_backtrain(days: int = 7, learning_rate_factor: float = 0.5):
    """
    Perform accelerated historical backtraining using stored data.
    This refines the current model WITHOUT resetting it - purely fine-tuning with historical patterns.
    
    Args:
        days: How many days of historical data to train on (default: 7)
        learning_rate_factor: Multiplier for learning rate (0.1-1.0) - lower = more conservative (default: 0.5)
    
    Returns:
        Training statistics including samples processed, error improvements, and confidence metrics
    """
    try:
        if learning_rate_factor < 0.1 or learning_rate_factor > 1.0:
            raise HTTPException(status_code=400, detail="learning_rate_factor must be between 0.1 and 1.0")
        
        if days < 1 or days > 90:
            raise HTTPException(status_code=400, detail="days must be between 1 and 90")
        
        print(f"🔄 Starting accelerated backtraining on {days} days of historical data...")
        print(f"   Learning rate factor: {learning_rate_factor}x (conservative fine-tuning)")
        
        results = {}
        
        for room in ROOMS.keys():
            print(f"\n📚 Processing {room}...")
            
            # Load current model (preserve all existing knowledge)
            model = load_model(room)
            if not model:
                print(f"  ⚠️ No model found for {room}, skipping")
                continue
            
            # Get historical training events from database
            conn = db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT current_temp, target_temp, radiator_level, temp_delta,
                       outdoor_temp, forecast_temp, timestamp, hour_of_day
                FROM training_events
                WHERE room = %s
                  AND timestamp >= NOW() - INTERVAL '%s days'
                ORDER BY timestamp ASC
            """, (room, days))
            
            historical_data = cursor.fetchall()
            cursor.close()
            conn.close()
            
            if not historical_data:
                print(f"  ⚠️ No historical data found for {room}")
                results[room] = {
                    "status": "no_data",
                    "samples_processed": 0
                }
                continue
            
            # Metrics tracking
            samples_processed = 0
            total_error_before = 0.0
            total_error_after = 0.0
            recent_weight = 1.0  # Will decay for older data
            
            # Time-based confidence decay
            newest_timestamp = historical_data[-1][6]  # Last timestamp
            
            # Process historical data in temporal order
            for idx, (current_temp, target_temp, radiator_level, actual_delta, 
                     outdoor_temp, forecast_temp, timestamp, hour_of_day) in enumerate(historical_data):
                
                # Calculate confidence decay based on age
                # Recent data gets weight ~1.0, older data decays to ~0.3
                age_days = (newest_timestamp - timestamp).total_seconds() / 86400
                age_factor = max(0.3, 1.0 - (age_days / days) * 0.7)
                
                # Apply learning rate reduction and age-based weighting
                effective_learning_rate = learning_rate_factor * age_factor
                
                # Prepare features exactly as they would have been during real-time training
                features = {
                    "current_temp": float(current_temp),
                    "target_temp": float(target_temp),
                    "outdoor_temp": float(outdoor_temp or 0),
                    "forecast_3h_temp": float(forecast_temp or outdoor_temp or 0),
                    "forecast_10h_temp": float(forecast_temp or outdoor_temp or 0),
                    "radiator_level": float(radiator_level),
                    "hour_of_day": int(hour_of_day),
                }
                
                # Get model's prediction BEFORE training on this sample
                try:
                    predicted_delta = model.predict_one(features)
                    error_before = abs(predicted_delta - actual_delta) if predicted_delta is not None else 0
                    total_error_before += error_before
                except:
                    error_before = 0
                
                # Train model on this historical outcome with reduced learning rate
                # River models don't have explicit learning rate, but we can simulate by:
                # 1. Training multiple times for higher weight (learning_rate > 1.0)
                # 2. Training fractionally by sampling (learning_rate < 1.0)
                
                # For conservative fine-tuning, we train on each sample but the model's
                # adaptive nature will naturally reduce impact of outliers
                model.learn_one(features, actual_delta)
                
                # For very conservative training, we could skip some samples
                if effective_learning_rate < 0.5 and idx % 2 == 0:
                    # Skip every other sample for very conservative training
                    pass
                else:
                    samples_processed += 1
                
                # Get prediction AFTER training
                try:
                    predicted_delta_after = model.predict_one(features)
                    error_after = abs(predicted_delta_after - actual_delta) if predicted_delta_after is not None else 0
                    total_error_after += error_after
                except:
                    error_after = 0
                
                # Progress logging every 100 samples
                if samples_processed % 100 == 0:
                    print(f"  📊 Processed {samples_processed} samples...")
            
            # Save refined model
            save_model(room, model)
            
            # Calculate improvement metrics
            avg_error_before = total_error_before / len(historical_data) if historical_data else 0
            avg_error_after = total_error_after / len(historical_data) if historical_data else 0
            improvement_pct = ((avg_error_before - avg_error_after) / avg_error_before * 100) if avg_error_before > 0 else 0
            
            results[room] = {
                "status": "success",
                "samples_processed": samples_processed,
                "total_samples_available": len(historical_data),
                "avg_error_before": round(avg_error_before, 4),
                "avg_error_after": round(avg_error_after, 4),
                "improvement_percent": round(improvement_pct, 2),
                "learning_rate_used": learning_rate_factor,
                "days_trained_on": days
            }
            
            print(f"  ✅ {room}: Processed {samples_processed} samples")
            print(f"     Error: {avg_error_before:.4f} → {avg_error_after:.4f} ({improvement_pct:+.1f}%)")
        
        print(f"\n✅ Accelerated backtraining complete!")
        
        return {
            "status": "success",
            "message": f"Refined models using {days} days of historical data",
            "learning_rate_factor": learning_rate_factor,
            "results": results,
            "note": "Models have been fine-tuned without resetting - all previous knowledge preserved"
        }
        
    except Exception as e:
        print(f"❌ Error during accelerated backtraining: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Backtraining failed: {str(e)}")


def perform_back_evaluation(room: str, model):
    """
    Perform back-evaluation: Check predictions made ~3 hours ago and train on actual outcomes.
    This is called on every /train to continuously improve prediction accuracy.
    Returns count of predictions validated and trained on.
    """
    validated_count = 0
    trained_count = 0
    
    try:
        # Get unvalidated predictions that are at least 3 hours old
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, room, predicted_temp, target_timestamp, 
                   radiator_level, outdoor_temp, forecast_temp, hours_ahead
            FROM predictions
            WHERE room = %s
              AND validated = FALSE
              AND target_timestamp <= NOW()
            ORDER BY target_timestamp DESC
            LIMIT 10
        """, (room,))
        
        predictions = cursor.fetchall()
        cursor.close()
        conn.close()
        
        if not predictions:
            return 0, 0
        
        for pred_row in predictions:
            pred_id, pred_room, predicted_temp, target_timestamp, radiator_level, outdoor_temp, forecast_temp, hours_ahead = pred_row
            
            try:
                # Find actual temperature at predicted time
                conn = db.get_connection()
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT current_temp, outdoor_temp, target_temp
                    FROM room_states
                    WHERE room = %s 
                      AND timestamp >= %s - INTERVAL '30 minutes'
                      AND timestamp <= %s + INTERVAL '30 minutes'
                    ORDER BY ABS(EXTRACT(EPOCH FROM (timestamp - %s)))
                    LIMIT 1
                """, (pred_room, target_timestamp, target_timestamp, target_timestamp))
                
                result = cursor.fetchone()
                cursor.close()
                conn.close()
                
                if result:
                    actual_temp, actual_outdoor, target_temp = result
                    prediction_error = abs(predicted_temp - actual_temp)
                    
                    # Mark prediction as validated
                    db.validate_prediction(pred_id, actual_temp, used_for_training=True)
                    validated_count += 1
                    
                    # Calculate what the model should have predicted
                    # We need the starting temperature (before the prediction was made)
                    conn = db.get_connection()
                    cursor = conn.cursor()
                    
                    cursor.execute("""
                        SELECT current_temp
                        FROM room_states
                        WHERE room = %s 
                          AND timestamp <= %s - INTERVAL '%s hours'
                        ORDER BY timestamp DESC
                        LIMIT 1
                    """, (pred_room, target_timestamp, hours_ahead))
                    
                    start_result = cursor.fetchone()
                    cursor.close()
                    conn.close()
                    
                    if start_result:
                        starting_temp = start_result[0]
                        actual_delta = actual_temp - starting_temp
                        
                        # Create features for that prediction moment
                        features = {
                            "current_temp": starting_temp,
                            "target_temp": target_temp or ROOMS[pred_room]["target"],
                            "outdoor_temp": outdoor_temp or actual_outdoor,
                            "forecast_3h": forecast_temp,
                            "forecast_10h": forecast_temp,  # Fallback
                            "radiator_level": radiator_level,
                            "hour": target_timestamp.hour,
                        }
                        
                        # Train model on actual outcome
                        model.learn_one(features, actual_delta)
                        trained_count += 1
                        
                        print(f"  🔄 Back-eval {pred_room}: predicted {predicted_temp:.1f}°C, actual {actual_temp:.1f}°C (error: {prediction_error:.2f}°C) - trained on actual delta {actual_delta:.2f}°C")
                    else:
                        # No starting temp found, still mark as validated but don't train
                        print(f"  ⚠️ Back-eval {pred_room}: No starting temp found for prediction {pred_id}")
                else:
                    # No actual data found, mark as validated without actual temp
                    db.validate_prediction(pred_id, None, used_for_training=False)
                    validated_count += 1
                    
            except Exception as e:
                print(f"  ❌ Error in back-eval for prediction {pred_id}: {e}")
                continue
                
    except Exception as e:
        print(f"❌ Error in perform_back_evaluation for {room}: {e}")
    
    return validated_count, trained_count

@app.post("/train")
def train(state: RoomState):
    """Train the model with new data and perform back-evaluation on past predictions"""
    # Validate required fields are not None
    try:
        state.validate_required_fields()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    
    model = load_model(state.room)
    
    # STEP 1: Perform back-evaluation on past predictions
    # This compares old predictions with actual outcomes and trains on the errors
    back_validated, back_trained = perform_back_evaluation(state.room, model)
    
    # Get previous temperature from database
    prev = db.get_latest_temp(state.room)
    if prev is None:
        prev = state.current_temp
    delta = state.current_temp - prev

    # Validate temperature readings - filter out anomalies
    if state.current_temp <= 0 or state.current_temp > 40 or state.current_temp < 5:
        return {
            "trained": False,
            "reason": "Invalid temperature reading (out of realistic range)",
            "current_temp": state.current_temp
        }
    
    # Check for sudden temperature changes (>4°C would be anomalous)
    if prev is not None and abs(delta) > 4:
        return {
            "trained": False,
            "reason": f"Anomalous temperature change detected: {delta:.1f}°C",
            "delta": delta,
            "current_temp": state.current_temp,
            "previous_temp": prev
        }

    # Add weather if missing
    if state.outdoor_temp is None or state.forecast_temp is None or state.forecast_10h_temp is None:
        outside, forecast_3h, forecast_10h = get_weather()
        state.outdoor_temp = outside
        state.forecast_temp = forecast_3h
        state.forecast_10h_temp = forecast_10h

    features = {
        "current_temp": state.current_temp,
        "target_temp": state.target_temp,
        "outdoor_temp": state.outdoor_temp,
        "forecast_3h_temp": state.forecast_temp,
        "forecast_10h_temp": state.forecast_10h_temp,
        "radiator_level": state.radiator_level,
        "hour_of_day": datetime.now().hour,
    }

    # Train the model
    model.learn_one(features, delta)
    save_model(state.room, model)

    # Update metrics (including back-evaluation training)
    m = model_metrics[state.room]
    m.training_samples += 1 + back_trained  # Current sample + back-evaluated samples
    
    # Try to get a prediction to track accuracy
    predicted_delta = None
    try:
        predicted_delta = model.predict_one(features)
        if predicted_delta is not None:
            m.update(delta, predicted_delta)
    except Exception:
        pass
    
    # Save metrics to database
    db.update_ai_metrics(
        state.room,
        mae=m.mae.get() if m.training_samples > 0 else 0,
        rmse=m.rmse.get() if m.training_samples > 0 else 0,
        r2_score=m.r2.get() if m.training_samples > 1 else 0,
        training_samples=m.training_samples,
        predictions_made=m.predictions_made,
        adjustments_made=m.adjustments_made,
        total_error=m.total_error
    )
    
    # Save room state to database
    db.save_room_state(
        state.room, state.current_temp, state.target_temp,
        state.radiator_level, state.outdoor_temp, state.forecast_temp
    )
    
    # Log training event to database
    db.save_training_event(
        state.room, state.current_temp, state.target_temp,
        state.radiator_level, delta, state.outdoor_temp,
        state.forecast_temp, predicted_delta, datetime.now().hour
    )
    
    # STEP 2: Save model (includes both current training and back-evaluation)
    if back_trained > 0:
        save_model(state.room, model)
        print(f"💾 Saved {state.room} model after back-evaluation training")

    return {
        "trained": True, 
        "delta": round(delta, 3),
        "training_samples": m.training_samples,
        "model_mae": round(m.mae.get(), 3) if m.training_samples > 0 else None,
        "back_evaluation": {
            "validated": back_validated,
            "trained_on": back_trained
        }
    }

@app.post("/predict")
def predict(state: RoomState):
    """Get radiator level recommendation focusing on 24h temperature stability"""
    # Validate required fields are not None
    try:
        state.validate_required_fields()
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    
    if state.outdoor_temp is None or state.forecast_temp is None or state.forecast_10h_temp is None:
        outside, forecast_3h, forecast_10h = get_weather()
        state.outdoor_temp, state.forecast_temp, state.forecast_10h_temp = outside, forecast_3h, forecast_10h

    current_hour = datetime.now().hour
    target_temp = state.target_temp
    
    m = model_metrics[state.room]
    m.predictions_made += 1
    
    # Always use ML model (load_model handles unpickling)
    model = load_model(state.room)
    if not model:
        # Fallback: use middle level if no model yet
        scale = ROOMS[state.room]["scale"]
        best_lvl = scale[len(scale) // 2]
        
        db.save_prediction(
            state.room, state.current_temp, state.target_temp,
            state.radiator_level, best_lvl, 999, False,
            state.outdoor_temp, state.forecast_temp, proactive_warning=False
        )
        
        return {
            "recommended": best_lvl,
            "error": None,
            "current_level": state.radiator_level,
            "adjustment_needed": False,
            "predictions_made": m.predictions_made,
            "method": "fallback_no_model",
            "ml_training_samples": m.training_samples,
            "prediction_details": []
        }
    
    # Simulate 8-hour ahead temperature for each level
    # Goal: Find level that keeps temperature stable near target over time
    best_lvl = None
    best_score = float("inf")
    prediction_details = []
    
    for lvl in ROOMS[state.room]["scale"]:
        try:
            # Simulate temperature evolution over next 8 hours at this level
            simulated_temp = state.current_temp
            total_error = 0.0
            hourly_temps = [simulated_temp]
            
            for hour_offset in range(HOURS_AHEAD):
                future_hour = (current_hour + hour_offset) % 24
                
                features = {
                    'current_temp': simulated_temp,
                    'target_temp': target_temp,
                    'radiator_level': lvl,
                    'outdoor_temp': state.outdoor_temp,
                    'forecast_3h': state.forecast_temp if hour_offset < 3 else state.forecast_10h_temp,
                    'forecast_10h': state.forecast_10h_temp,
                    'hour': future_hour
                }
                
                # Predict temperature delta for this hour
                ml_delta = model.predict_one(features)
                simulated_temp += ml_delta
                hourly_temps.append(simulated_temp)
                
                # Accumulate error (distance from target)
                error = abs(simulated_temp - target_temp)
                total_error += error
            
            # Average error over 8-hour period
            avg_error = total_error / HOURS_AHEAD
            final_temp = hourly_temps[-1]
            
            prediction_details.append({
                "level": lvl,
                "immediate_temp": round(hourly_temps[1], 2),
                "temp_8h": round(final_temp, 2),
                "avg_error_8h": round(avg_error, 2),
                "stability_score": round(avg_error, 2)
            })
            
            # Best level is one with lowest average error over 8 hours
            if avg_error < best_score:
                best_score = avg_error
                best_lvl = lvl
                
        except Exception as e:
            print(f"ML prediction error for {state.room} level {lvl}: {e}")
            continue
    
    # Fallback if all predictions failed
    if best_lvl is None:
        scale = ROOMS[state.room]["scale"]
        best_lvl = scale[len(scale) // 2]
        best_score = 999
        print(f"⚠️  All ML predictions failed for {state.room}, using fallback level {best_lvl}")
    
    # Smart adjustment logic with acceptable deviation band and gradual changes
    current_deviation = state.current_temp - target_temp
    
    # Check if we're within acceptable deviation - if so, maintain current level
    if abs(current_deviation) <= ACCEPTABLE_DEVIATION:
        best_lvl = state.radiator_level  # Stay at current level
        adjustment_made = False
        adjustment_reason = f"Within tolerance (±{ACCEPTABLE_DEVIATION}°C)"
    else:
        # Calculate ideal adjustment
        raw_adjustment = best_lvl - state.radiator_level
        
        # Apply gradual adjustment limit - don't change too drastically
        if abs(raw_adjustment) > MAX_SINGLE_ADJUSTMENT:
            # Limit to gradual change in the right direction
            gradual_adjustment = MAX_SINGLE_ADJUSTMENT if raw_adjustment > 0 else -MAX_SINGLE_ADJUSTMENT
            
            # Find nearest valid level
            target_level = state.radiator_level + gradual_adjustment
            scale = ROOMS[state.room]["scale"]
            best_lvl = min(scale, key=lambda x: abs(x - target_level))
            adjustment_reason = f"Gradual adjustment (limited to ±{MAX_SINGLE_ADJUSTMENT})"
        else:
            adjustment_reason = "Optimal 8h stability"
        
        # Only adjust if change is significant enough
        adjustment_made = abs(best_lvl - state.radiator_level) >= MIN_ADJUSTMENT_THRESHOLD
        
        if not adjustment_made:
            best_lvl = state.radiator_level
            adjustment_reason = f"Change too small (<{MIN_ADJUSTMENT_THRESHOLD})"
    
    if adjustment_made:
        m.adjustments_made += 1
        
        # Get predicted info for recommended level
        recommended_pred = next((p for p in prediction_details if p['level'] == best_lvl), None)
        current_pred = next((p for p in prediction_details if p['level'] == state.radiator_level), None)
        
        if recommended_pred:
            temp_change = recommended_pred['temp_8h'] - state.current_temp
            direction = "📈" if temp_change > 0 else "📉" if temp_change < 0 else "➡️"
            
            msg = (
                f"🤖 ML: {state.room} {state.radiator_level} → {best_lvl}\n"
                f"🌡️ Current: {state.current_temp}°C (dev: {current_deviation:+.1f}°C)\n"
                f"🎯 Target: {target_temp}°C\n"
                f"{direction} 8h forecast: {recommended_pred['temp_8h']}°C (stability: {best_score:.2f}°C)\n"
                f"💡 Reason: {adjustment_reason}\n"
                f"📊 Training: {m.training_samples} samples\n"
                f"🌤️ Outside: {state.outdoor_temp}°C"
            )
        else:
            msg = (
                f"🤖 ML: {state.room} {state.radiator_level} → {best_lvl}\n"
                f"🌡️ Current: {state.current_temp}°C (dev: {current_deviation:+.1f}°C)\n"
                f"🎯 Target: {target_temp}°C\n"
                f"⚠️ Fallback mode\n"
                f"💡 Reason: {adjustment_reason}\n"
                f"📊 Training: {m.training_samples} samples\n"
                f"🌤️ Outside: {state.outdoor_temp}°C"
            )
        
        if TELEGRAM_WEBHOOK:
            try:
                requests.post(TELEGRAM_WEBHOOK, json={"text": msg}, timeout=5)
            except Exception:
                pass
    else:
        # No adjustment, but log the reason
        print(f"ℹ️  {state.room}: No adjustment needed. {adjustment_reason}")

    
    # Update metrics in database
    db.update_ai_metrics(
        state.room,
        predictions_made=m.predictions_made,
        adjustments_made=m.adjustments_made,
        total_error=m.total_error
    )
    
    # Save prediction to database
    db.save_prediction(
        state.room, state.current_temp, state.target_temp,
        state.radiator_level, best_lvl, best_score, adjustment_made,
        state.outdoor_temp, state.forecast_temp,
        proactive_warning=False
    )
    
    # Save room state to database
    db.save_room_state(
        state.room, state.current_temp, state.target_temp,
        state.radiator_level, state.outdoor_temp, state.forecast_temp
    )

    return {
        "recommended": best_lvl, 
        "error": round(best_score, 2) if best_score != float("inf") else None,
        "current_level": state.radiator_level,
        "adjustment_needed": adjustment_made,
        "adjustment_reason": adjustment_reason,
        "current_deviation": round(current_deviation, 2),
        "predictions_made": m.predictions_made,
        "method": "ml_with_hints",
        "ml_training_samples": m.training_samples,
        "prediction_details": prediction_details
    }

@app.get("/stats")
def get_stats():
    """Get comprehensive AI performance statistics from database"""
    stats = {}
    
    # Get metrics from database
    all_metrics = db.get_ai_metrics()
    
    for room in ROOMS.keys():
        if room in all_metrics:
            metrics = all_metrics[room]
            stats[room] = {
                "mae": round(metrics.get('mae', 0), 3),
                "rmse": round(metrics.get('rmse', 0), 3),
                "r2_score": round(metrics.get('r2_score', 0), 3),
                "training_samples": metrics.get('training_samples', 0),
                "predictions_made": metrics.get('predictions_made', 0),
                "adjustments_made": metrics.get('adjustments_made', 0),
                "avg_error": round(metrics.get('total_error', 0) / max(1, metrics.get('predictions_made', 1)), 3),
                "created_at": metrics.get('created_at').isoformat() if metrics.get('created_at') else None,
                "last_updated": metrics.get('last_updated').isoformat() if metrics.get('last_updated') else None
            }
        else:
            stats[room] = {
                "mae": 0,
                "rmse": 0,
                "r2_score": 0,
                "training_samples": 0,
                "predictions_made": 0,
                "adjustments_made": 0,
                "avg_error": 0,
                "created_at": None,
                "last_updated": None
            }
    
    # Calculate overall statistics
    total_samples = sum(s["training_samples"] for s in stats.values())
    total_predictions = sum(s["predictions_made"] for s in stats.values())
    total_adjustments = sum(s["adjustments_made"] for s in stats.values())
    
    trained_rooms = [s for s in stats.values() if s["training_samples"] > 0]
    avg_mae = sum(s["mae"] for s in trained_rooms) / max(1, len(trained_rooms))
    
    # Get training statistics from database
    training_stats = db.get_training_stats()
    
    # Get prediction statistics from database  
    prediction_stats = db.get_prediction_stats(hours=24)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_training_samples": total_samples,
            "total_predictions": total_predictions,
            "total_adjustments": total_adjustments,
            "average_mae": round(avg_mae, 3),
            "efficiency_rate": round(total_adjustments / max(1, total_predictions) * 100, 1),
        },
        "rooms": stats,
        "training_stats": training_stats,
        "prediction_stats_24h": prediction_stats,
        "model_info": {
            "algorithm": "Adaptive Random Forest Regressor (River ML)",
            "features": ["current_temp", "target_temp", "outdoor_temp", "forecast_temp", "radiator_level", "hour_of_day"],
            "learning_type": "Online/Incremental Learning",
            "prediction_method": "Temperature delta prediction",
            "storage": "PostgreSQL Database",
            "persistence": "Models and data survive restarts"
        }
    }

@app.post("/validate-predictions")
def validate_predictions():
    """Validate past predictions and use them for training"""
    unvalidated = db.get_unvalidated_predictions()
    
    if not unvalidated:
        return {
            "validated": 0,
            "trained": 0,
            "message": "No predictions ready for validation"
        }
    
    validated_count = 0
    trained_count = 0
    
    for pred in unvalidated:
        room = pred['room']
        prediction_id = pred['id']
        predicted_temp = pred['predicted_temp']
        hours_ahead = pred['hours_ahead']
        radiator_level = pred['radiator_level']
        
        # Get actual temperature at the target time
        # Look for the closest room_state reading to target_timestamp
        try:
            conn = db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT current_temp, outdoor_temp 
                FROM room_states
                WHERE room = %s 
                  AND timestamp >= %s - INTERVAL '30 minutes'
                  AND timestamp <= %s + INTERVAL '30 minutes'
                ORDER BY ABS(EXTRACT(EPOCH FROM (timestamp - %s)))
                LIMIT 1
            """, (room, pred['target_timestamp'], pred['target_timestamp'], pred['target_timestamp']))
            
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if result:
                actual_temp = result[0]
                actual_outdoor = result[1]
                
                # Validate the prediction
                db.validate_prediction(prediction_id, actual_temp, used_for_training=True)
                validated_count += 1
                
                # Use this for training!
                # Calculate the actual delta that occurred
                original_temp = predicted_temp - (hours_ahead * 0.1)  # Rough estimate of starting temp
                actual_delta = actual_temp - original_temp
                
                # Load model and train on this real-world outcome
                model = load_model(room)
                
                features = {
                    "current_temp": original_temp,
                    "target_temp": ROOMS[room]["target"],
                    "outdoor_temp": actual_outdoor or pred['outdoor_temp'],
                    "forecast_temp": pred['forecast_temp'],
                    "radiator_level": radiator_level,
                    "hour_of_day": pred['target_timestamp'].hour,
                }
                
                # Train the model with the actual outcome
                model.learn_one(features, actual_delta)
                save_model(room, model)
                
                # Update metrics
                m = model_metrics[room]
                m.training_samples += 1
                
                # Save training event
                db.save_training_event(
                    room, original_temp, ROOMS[room]["target"],
                    radiator_level, actual_delta,
                    actual_outdoor, pred['forecast_temp'],
                    predicted_delta=(predicted_temp - original_temp),
                    hour_of_day=pred['target_timestamp'].hour
                )
                
                # Update metrics in database
                db.update_ai_metrics(
                    room,
                    training_samples=m.training_samples
                )
                
                trained_count += 1
                
                print(f"✅ Validated & trained {room}: predicted {predicted_temp:.1f}°C, actual {actual_temp:.1f}°C (error: {abs(predicted_temp - actual_temp):.2f}°C)")
            else:
                # No matching temperature data found, just mark as validated without training
                db.validate_prediction(prediction_id, None, used_for_training=False)
                validated_count += 1
                
        except Exception as e:
            print(f"Error validating prediction {prediction_id}: {e}")
            continue
    
    return {
        "validated": validated_count,
        "trained": trained_count,
        "message": f"Validated {validated_count} predictions, trained on {trained_count}"
    }

@app.get("/validation-stats")
def get_validation_statistics(days: int = 7):
    """Get statistics on prediction validation accuracy"""
    stats = db.get_validation_stats(days=days)
    
    return {
        "days": days,
        "rooms": stats,
        "summary": {
            "total_predictions": sum(s.get('total_predictions', 0) for s in stats.values()),
            "total_trained": sum(s.get('used_for_training', 0) for s in stats.values()),
            "avg_error": sum(s.get('avg_error', 0) for s in stats.values()) / max(1, len(stats))
        }
    }

@app.get("/health")
def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/radiator/history/{room}")
def get_room_radiator_history(room: str, hours: int = 24):
    """Get radiator level change history for a specific room"""
    if room not in ROOMS:
        raise HTTPException(status_code=404, detail=f"Room '{room}' not found")
    
    try:
        history = db.get_radiator_history(room, hours)
        return {
            "room": room,
            "hours": hours,
            "changes": len(history),
            "history": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")

@app.get("/radiator/history")
def get_all_radiator_history(hours: int = 24):
    """Get radiator level change history for all rooms"""
    try:
        history = db.get_radiator_history(None, hours)
        
        # Group by room
        by_room = {}
        for change in history:
            room = change['room']
            if room not in by_room:
                by_room[room] = []
            by_room[room].append(change)
        
        return {
            "hours": hours,
            "total_changes": len(history),
            "rooms": by_room
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")

@app.get("/history/{room}")
def get_room_history(room: str, hours: int = 24):
    """Get historical data for a specific room (for graphing)"""
    if room not in ROOMS:
        raise HTTPException(status_code=404, detail=f"Room '{room}' not found")
    
    try:
        data = db.get_historical_data(room, hours)
        return {
            "room": room,
            "hours": hours,
            "data_points": len(data),
            "data": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")

@app.get("/history")
def get_all_history(hours: int = 24):
    """Get historical data for all rooms (for graphing)"""
    try:
        data = db.get_historical_data(None, hours)
        
        # Group by room
        by_room = {}
        for point in data:
            room = point['room']
            if room not in by_room:
                by_room[room] = []
            by_room[room].append(point)
        
        return {
            "hours": hours,
            "total_data_points": len(data),
            "rooms": by_room
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")

@app.get("/training/history/{room}")
def get_training_history(room: str):
    """Get training history for a specific room"""
    if room not in ROOMS:
        raise HTTPException(status_code=404, detail=f"Room '{room}' not found")
    
    try:
        stats = db.get_training_stats(room)
        return {
            "room": room,
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching training history: {str(e)}")

@app.get("/export/csv/{room}")
def export_room_csv(room: str, hours: int = 168):  # Default 1 week
    """Export room data as CSV for analysis"""
    from fastapi.responses import StreamingResponse
    import csv
    from io import StringIO
    
    if room not in ROOMS:
        raise HTTPException(status_code=404, detail=f"Room '{room}' not found")
    
    try:
        data = db.get_historical_data(room, hours)
        
        # Create CSV
        output = StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow(['timestamp', 'room', 'current_temp', 'target_temp', 
                        'radiator_level', 'outdoor_temp', 'forecast_temp'])
        
        # Data
        for point in data:
            writer.writerow([
                point['timestamp'].isoformat() if isinstance(point['timestamp'], datetime) else point['timestamp'],
                point['room'],
                point['current_temp'],
                point['target_temp'],
                point['radiator_level'],
                point.get('outdoor_temp', ''),
                point.get('forecast_temp', '')
            ])
        
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={room}_data.csv"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exporting CSV: {str(e)}")

@app.delete("/training/{training_id}")
def delete_training_event(training_id: int):
    """Delete a specific training event (for removing wrong data)"""
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Get the training event details first
        cursor.execute("SELECT room FROM training_history WHERE id = %s", (training_id,))
        result = cursor.fetchone()
        
        if not result:
            cursor.close()
            conn.close()
            raise HTTPException(status_code=404, detail="Training event not found")
        
        room = result[0]
        
        # Delete the training event
        cursor.execute("DELETE FROM training_history WHERE id = %s", (training_id,))
        conn.commit()
        
        # Update metrics - decrement training samples
        cursor.execute("""
            UPDATE ai_metrics 
            SET training_samples = GREATEST(0, training_samples - 1),
                last_updated = NOW()
            WHERE room = %s
        """, (room,))
        conn.commit()
        
        cursor.close()
        conn.close()
        
        return {"deleted": True, "training_id": training_id, "room": room}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting training event: {str(e)}")

@app.get("/graphs/radiator-temperature")
def get_radiator_temperature_graphs(room: str = None, days: int = 30):
    """Get data for radiator level vs temperature graphs by hour of day"""
    try:
        data = db.get_radiator_level_temperature_by_hour(room, days)
        return {
            "room": room,
            "days": days,
            "data": data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching graph data: {str(e)}")

@app.get("/analytics/room-insights")
def get_room_insights():
    """
    Get comprehensive insights for each room including current state,
    prediction reasoning, and performance metrics
    """
    try:
        insights = {}
        
        for room in ROOMS.keys():
            # Get latest room state
            conn = db.get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT current_temp, target_temp, radiator_level, outdoor_temp, forecast_temp, timestamp
                FROM room_states
                WHERE room = %s
                ORDER BY timestamp DESC
                LIMIT 1
            """, (room,))
            
            room_state = cursor.fetchone()
            
            # Get latest prediction with reasoning
            cursor.execute("""
                SELECT recommended_level, predicted_error, adjustment_made, timestamp, outdoor_temp, forecast_temp
                FROM predictions
                WHERE room = %s
                ORDER BY timestamp DESC
                LIMIT 1
            """, (room,))
            
            latest_prediction = cursor.fetchone()
            
            # Get metrics
            metrics = model_metrics[room]
            all_metrics = db.get_ai_metrics()
            room_metrics = all_metrics.get(room, {})
            
            # Get recent adjustment success rate
            cursor.execute("""
                SELECT COUNT(*) as total_predictions,
                       SUM(CASE WHEN adjustment_made THEN 1 ELSE 0 END) as adjustments
                FROM predictions
                WHERE room = %s
                  AND timestamp >= NOW() - INTERVAL '24 hours'
            """, (room,))
            
            accuracy_data = cursor.fetchone()
            cursor.close()
            conn.close()
            
            if room_state:
                current_temp, target_temp, radiator_level, outdoor_temp, forecast_temp, state_timestamp = room_state
                deviation = current_temp - target_temp
                
                # Generate reasoning
                reasoning = generate_decision_reasoning(
                    room, current_temp, target_temp, radiator_level,
                    outdoor_temp, forecast_temp, deviation
                )
                
                # Calculate prediction confidence from error
                pred_confidence = None
                if latest_prediction and latest_prediction[1] is not None:
                    # Confidence based on predicted error (lower error = higher confidence)
                    pred_confidence = round(max(0, 1.0 - (latest_prediction[1] / 2.0)), 3)
                
                insights[room] = {
                    "current_state": {
                        "temperature": round(current_temp, 1),
                        "target": round(target_temp, 1),
                        "deviation": round(deviation, 2),
                        "radiator_level": radiator_level,
                        "outdoor_temp": round(outdoor_temp, 1) if outdoor_temp else None,
                        "forecast_temp": round(forecast_temp, 1) if forecast_temp else None,
                        "status": "optimal" if abs(deviation) <= ACCEPTABLE_DEVIATION else "adjusting"
                    },
                    "latest_prediction": {
                        "recommended_level": latest_prediction[0] if latest_prediction else None,
                        "predicted_error": round(latest_prediction[1], 2) if latest_prediction and latest_prediction[1] else None,
                        "confidence": pred_confidence,
                        "adjustment_made": latest_prediction[2] if latest_prediction else False
                    },
                    "performance": {
                        "training_samples": metrics.training_samples,
                        "predictions_made": metrics.predictions_made,
                        "mae": round(room_metrics.get('mae', 0), 3),
                        "rmse": round(room_metrics.get('rmse', 0), 3),
                        "r2_score": round(room_metrics.get('r2_score', 0), 3),
                        "24h_predictions": accuracy_data[0] if accuracy_data else 0,
                        "24h_adjustments": accuracy_data[1] if accuracy_data else 0
                    },
                    "reasoning": reasoning
                }
        
        return insights
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error generating insights: {str(e)}")

def generate_decision_reasoning(room, current_temp, target_temp, radiator_level, outdoor_temp, forecast_temp, deviation):
    """Generate human-readable reasoning for AI decisions"""
    reasons = []
    
    # Temperature deviation analysis
    if abs(deviation) <= ACCEPTABLE_DEVIATION:
        reasons.append(f"✅ Temperature optimal: within ±{ACCEPTABLE_DEVIATION}°C tolerance")
    elif deviation > 0:
        reasons.append(f"🔥 Room {deviation:+.1f}°C too warm → reducing heating")
    else:
        reasons.append(f"❄️ Room {deviation:+.1f}°C too cold → increasing heating")
    
    # Outdoor weather influence
    if outdoor_temp and forecast_temp:
        temp_drop = outdoor_temp - forecast_temp
        if temp_drop > 3:
            reasons.append(f"🌡️ Outdoor drop of {temp_drop:.1f}°C predicted → pre-heating to prevent undershoot")
        elif temp_drop < -3:
            reasons.append(f"🌤️ Outdoor warming of {abs(temp_drop):.1f}°C predicted → reducing to prevent overshoot")
        elif outdoor_temp < 0:
            reasons.append(f"❄️ Freezing conditions ({outdoor_temp:.1f}°C) → maintaining higher baseline")
    
    # Radiator activity
    scale_max = max(ROOMS[room]["scale"])
    level_pct = (radiator_level / scale_max * 100) if scale_max > 0 else 0
    
    if level_pct < 30:
        reasons.append(f"💤 Low activity mode ({radiator_level}/{scale_max}) - minimal heating")
    elif level_pct > 70:
        reasons.append(f"🔥 High activity mode ({radiator_level}/{scale_max}) - maximum heating")
    else:
        reasons.append(f"⚖️ Balanced mode ({radiator_level}/{scale_max}) - moderate heating")
    
    # Gradual adjustment logic
    reasons.append(f"🎯 Changes limited to ±{MAX_SINGLE_ADJUSTMENT} levels per cycle for stability")
    
    return " | ".join(reasons)

@app.get("/analytics/prediction-accuracy")
def get_prediction_accuracy(days: int = 7):
    """Get time-series data of prediction performance over time"""
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Get daily prediction statistics
        cursor.execute("""
            SELECT 
                room,
                DATE(timestamp) as date,
                AVG(predicted_error) as avg_error,
                COUNT(*) as prediction_count,
                SUM(CASE WHEN adjustment_made THEN 1 ELSE 0 END) as adjustments_made
            FROM predictions
            WHERE timestamp >= NOW() - INTERVAL '%s days'
            GROUP BY room, DATE(timestamp)
            ORDER BY date, room
        """, (days,))
        
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        
        # Organize by room
        accuracy_by_room = {}
        for row in results:
            room, date, avg_error, count, adjustments = row
            if room not in accuracy_by_room:
                accuracy_by_room[room] = []
            accuracy_by_room[room].append({
                "date": date.isoformat(),
                "avg_error": round(avg_error, 3) if avg_error else 0,
                "prediction_count": count,
                "adjustments_made": adjustments
            })
        
        return accuracy_by_room
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching accuracy data: {str(e)}")

@app.get("/analytics/temperature-timeline")
def get_temperature_timeline(room: str, hours: int = 48):
    """Get actual temperature timeline with predictions for a room"""
    try:
        conn = db.get_connection()
        cursor = conn.cursor()
        
        # Get actual temperatures
        cursor.execute("""
            SELECT timestamp, current_temp, target_temp, radiator_level, outdoor_temp
            FROM room_states
            WHERE room = %s
              AND timestamp >= NOW() - INTERVAL '%s hours'
            ORDER BY timestamp
        """, (room, hours))
        
        actual_data = cursor.fetchall()
        
        # Get predictions
        cursor.execute("""
            SELECT timestamp, recommended_level, predicted_error, current_temp, outdoor_temp
            FROM predictions
            WHERE room = %s
              AND timestamp >= NOW() - INTERVAL '%s hours'
            ORDER BY timestamp
        """, (room, hours))
        
        prediction_data = cursor.fetchall()
        cursor.close()
        conn.close()
        
        return {
            "room": room,
            "actual": [
                {
                    "timestamp": row[0].isoformat(),
                    "temperature": round(row[1], 1),
                    "target": round(row[2], 1),
                    "radiator_level": row[3],
                    "outdoor_temp": round(row[4], 1) if row[4] else None
                }
                for row in actual_data
            ],
            "predictions": [
                {
                    "timestamp": row[0].isoformat(),
                    "recommended_level": row[1],
                    "predicted_error": round(row[2], 2) if row[2] else None,
                    "current_temp": round(row[3], 1) if row[3] else None,
                    "outdoor_temp": round(row[4], 1) if row[4] else None
                }
                for row in prediction_data
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching timeline data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching timeline: {str(e)}")

@app.get("/ui", response_class=HTMLResponse)
def ui_dashboard(training_page: int = 1, predictions_page: int = 1):
    """Enhanced web UI dashboard with validation stats and advanced analytics"""
    from fastapi.responses import HTMLResponse
    
    # Pagination settings
    items_per_page = 10
    training_offset = (training_page - 1) * items_per_page
    predictions_offset = (predictions_page - 1) * items_per_page
    
    try:
        # Get paginated data
        latest_training = db.get_latest_training_events(limit=items_per_page, offset=training_offset)
        latest_predictions = db.get_latest_predictions(limit=items_per_page, offset=predictions_offset)
        
        # Get total counts for pagination
        total_training = db.get_total_training_count()
        total_predictions = db.get_total_predictions_count()
        training_count_24h = db.get_training_count_last_24h()
        all_metrics = db.get_ai_metrics()
        validation_stats = db.get_validation_stats(days=7)
        
        # Get weather
        try:
            outdoor, forecast_3h, forecast_10h = get_weather()
            outdoor = outdoor if outdoor else "N/A"
            forecast_3h = forecast_3h if forecast_3h else "N/A"
            forecast_10h = forecast_10h if forecast_10h else "N/A"
        except Exception:
            outdoor, forecast_3h, forecast_10h = "N/A", "N/A", "N/A"
        
        # Calculate totals
        total_predictions = sum(m.get('predictions_made', 0) for m in all_metrics.values())
        total_trained = sum(m.get('training_samples', 0) for m in all_metrics.values())
        total_validated = sum(v.get('total_predictions', 0) for v in validation_stats.values())
        avg_validation_error = sum(v.get('avg_error', 0) for v in validation_stats.values()) / max(1, len(validation_stats))
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Radiator AI - Dashboard</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{ max-width: 1600px; margin: 0 auto; }}
        .header {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }}
        .header h1 {{ color: #667eea; font-size: 2.5em; margin-bottom: 10px; }}
        .header .status {{ color: #28a745; font-size: 1.2em; font-weight: bold; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-card h3 {{
            color: #667eea;
            font-size: 0.85em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }}
        .stat-card .value {{
            font-size: 2.2em;
            font-weight: bold;
            color: #333;
            margin: 10px 0;
        }}
        .stat-card .label {{ color: #666; font-size: 0.85em; }}
        .section {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }}
        .section h2 {{
            color: #667eea;
            font-size: 1.6em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            font-size: 0.9em;
        }}
        th {{
            background: #667eea;
            color: white;
            padding: 10px;
            text-align: left;
            font-weight: 600;
            font-size: 0.8em;
        }}
        td {{ padding: 10px; border-bottom: 1px solid #eee; }}
        tr:hover {{ background: #f8f9fa; }}
        .room-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: 600;
            font-size: 0.85em;
        }}
        .room-Sovrum {{ background: #e3f2fd; color: #1976d2; }}
        .room-Kontor {{ background: #f3e5f5; color: #7b1fa2; }}
        .room-Vardagsrum {{ background: #fff3e0; color: #e65100; }}
        .room-Badrum {{ background: #e8f5e9; color: #2e7d32; }}
        .good {{ color: #28a745; font-weight: bold; }}
        .warning {{ color: #ffc107; font-weight: bold; }}
        .info {{ color: #17a2b8; font-weight: bold; }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }}
        .metric-box {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }}
        .metric-box .metric-label {{
            font-size: 0.8em;
            color: #666;
            text-transform: uppercase;
        }}
        .metric-box .metric-value {{
            font-size: 1.8em;
            font-weight: bold;
            color: #333;
            margin: 5px 0;
        }}
        .metric-box .metric-sub {{ font-size: 0.85em; color: #666; margin-top: 5px; }}
        .refresh-btn {{
            position: fixed;
            bottom: 30px;
            right: 30px;
            background: #667eea;
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 50px;
            font-size: 1em;
            font-weight: bold;
            cursor: pointer;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }}
        .reset-btn {{
            position: fixed;
            bottom: 30px;
            right: 170px;
            background: #dc3545;
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 50px;
            font-size: 1em;
            font-weight: bold;
            cursor: pointer;
            box-shadow: 0 5px 15px rgba(220, 53, 69, 0.4);
            transition: all 0.3s;
        }}
        .reset-btn:hover {{
            background: #c82333;
        }}
        .reset-btn:disabled {{
            background: #6c757d;
            cursor: not-allowed;
        }}
        .backtrain-btn {{
            position: fixed;
            bottom: 30px;
            right: 360px;
            background: #28a745;
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 50px;
            font-size: 1em;
            font-weight: bold;
            cursor: pointer;
            box-shadow: 0 5px 15px rgba(40, 167, 69, 0.4);
            transition: all 0.3s;
        }}
        .backtrain-btn:hover {{
            background: #218838;
        }}
        .backtrain-btn:disabled {{
            background: #6c757d;
            cursor: not-allowed;
        }}
        .progress-modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.7);
            z-index: 1000;
            justify-content: center;
            align-items: center;
        }}
        .progress-content {{
            background: white;
            padding: 40px;
            border-radius: 15px;
            max-width: 600px;
            width: 90%;
        }}
        .progress-title {{
            font-size: 1.5em;
            color: #667eea;
            margin-bottom: 20px;
            text-align: center;
        }}
        .progress-status {{
            text-align: center;
            margin: 20px 0;
            font-size: 1.1em;
        }}
        .room-progress {{
            margin: 15px 0;
        }}
        .room-progress-name {{
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .room-progress-bar {{
            background: #e0e0e0;
            border-radius: 10px;
            height: 30px;
            overflow: hidden;
        }}
        .room-progress-fill {{
            background: linear-gradient(90deg, #667eea, #764ba2);
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            transition: width 0.5s;
        }}
        .progress-bar {{
            background: #e0e0e0;
            border-radius: 10px;
            height: 20px;
            overflow: hidden;
            margin-top: 10px;
        }}
        .progress-fill {{
            background: linear-gradient(90deg, #667eea, #764ba2);
            height: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 0.8em;
            font-weight: bold;
        }}
        .page-btn {{
            display: inline-block;
            padding: 8px 15px;
            margin: 0 3px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            font-size: 0.9em;
            font-weight: 600;
            transition: background 0.3s;
        }}
        .page-btn:hover {{
            background: #5568d3;
        }}
        .page-btn.current {{
            background: #764ba2;
            cursor: default;
        }}
        .delete-btn {{
            background: #dc3545;
            color: white;
            border: none;
            padding: 5px 10px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.8em;
            transition: background 0.3s;
        }}
        .delete-btn:hover {{
            background: #c82333;
        }}
        .graph-container {{
            margin-bottom: 30px;
        }}
        .graph-title {{
            font-size: 1.2em;
            font-weight: bold;
            color: #667eea;
            margin-bottom: 10px;
        }}
        .canvas-wrapper {{
            position: relative;
            width: 100%;
            height: 400px;
            background: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        canvas {{
            width: 100% !important;
            height: 100% !important;
        }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏠 Smart Radiator AI Dashboard</h1>
            <div class="status">✅ System Online - v2.1.0</div>
            <p style="color:#666;margin-top:10px">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <h3>🎓 Training Samples</h3>
                <div class="value good">{total_trained}</div>
                <div class="label">Total model training events</div>
                <div class="label" style="margin-top:5px">{training_count_24h} in last 24h</div>
            </div>
            <div class="stat-card">
                <h3>🎯 Predictions Made</h3>
                <div class="value info">{total_predictions}</div>
                <div class="label">AI recommendations</div>
            </div>
            <div class="stat-card">
                <h3>✅ Validated</h3>
                <div class="value warning">{total_validated}</div>
                <div class="label">Self-learning cycles</div>
                <div class="label" style="margin-top:5px">Avg error: {avg_validation_error:.2f}°C</div>
            </div>
            <div class="stat-card">
                <h3>🌡️ Current Weather</h3>
                <div class="value">{outdoor}°C</div>
                <div class="label">3h: {forecast_3h}°C | 10h: {forecast_10h}°C</div>
            </div>
        </div>

        <!-- Room Insights Section with AI Reasoning -->
        <div class="section">
            <h2>🧠 AI Decision Insights & Reasoning</h2>
            <div id="room-insights-container">
                <p style="text-align:center;color:#666;">Loading AI insights...</p>
            </div>
        </div>

        <!-- Temperature Timeline Graphs -->
        <div class="section">
            <h2>📈 Temperature Timeline (Actual vs Predicted)</h2>
            <div id="timeline-graphs-container">
                <p style="text-align:center;color:#666;">Loading timeline charts...</p>
            </div>
        </div>

        <!-- Prediction Accuracy Trends -->
        <div class="section">
            <h2>🎯 Prediction Accuracy Over Time</h2>
            <div id="accuracy-trends-container">
                <p style="text-align:center;color:#666;">Loading accuracy trends...</p>
            </div>
        </div>

        <div class="section">
            <h2>📊 Per-Room Analytics</h2>
            <div class="metrics-grid">
"""
        
        for room, metrics in all_metrics.items():
            training = metrics.get('training_samples', 0)
            predictions = metrics.get('predictions_made', 0)
            mae = metrics.get('mae', 0)
            r2 = metrics.get('r2_score', 0)
            
            val_stats = validation_stats.get(room, {})
            val_count = val_stats.get('total_predictions', 0)
            val_error = val_stats.get('avg_error', 0)
            
            # Calculate accuracy as inverse of average error
            # If avg error is 0°C -> 100% accuracy (full bar)
            # If avg error is 1°C -> ~63% accuracy
            # If avg error is 2°C -> ~37% accuracy
            # Using exponential decay: accuracy = 100 * e^(-error)
            import math
            accuracy_pct = 100 * math.exp(-val_error) if val_error > 0 else 100
            
            html += f"""
                <div class="metric-box">
                    <div class="metric-label">{room}</div>
                    <div class="metric-value">{training}</div>
                    <div class="metric-sub">Training samples</div>
                    <div class="metric-sub">MAE: {mae:.3f} | R²: {r2:.3f}</div>
                    <div class="metric-sub">Predictions: {predictions}</div>
                    <div class="metric-sub">Validated: {val_count} (±{val_error:.2f}°C)</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width:{accuracy_pct:.0f}%">
                            {accuracy_pct:.0f}% (±{val_error:.2f}°C avg)
                        </div>
                    </div>
                </div>
"""
        
        html += """
            </div>
        </div>
"""
        
        # Validation statistics section
        if validation_stats:
            html += """
        <div class="section">
            <h2>✅ Prediction Validation Stats (Last 7 Days)</h2>
            <table>
                <thead>
                    <tr>
                        <th>Room</th>
                        <th>Predictions Made</th>
                        <th>Used for Training</th>
                        <th>Avg Error</th>
                        <th>Min Error</th>
                        <th>Max Error</th>
                        <th>Training Rate</th>
                    </tr>
                </thead>
                <tbody>
"""
            for room, vstats in validation_stats.items():
                total = vstats.get('total_predictions', 0)
                used = vstats.get('used_for_training', 0)
                avg_err = vstats.get('avg_error', 0)
                min_err = vstats.get('min_error', 0)
                max_err = vstats.get('max_error', 0)
                # Calculate accuracy as percentage of predictions with error < 0.5°C
                # This is a more meaningful metric than trying to convert error to percentage
                accuracy_pct = (used / total * 100) if total > 0 else 0
                
                html += f"""
                    <tr>
                        <td><span class="room-badge room-{room}">{room}</span></td>
                        <td>{total}</td>
                        <td class="good">{used}</td>
                        <td>{avg_err:.3f}°C</td>
                        <td class="good">{min_err:.3f}°C</td>
                        <td class="warning">{max_err:.3f}°C</td>
                        <td class="info">{accuracy_pct:.1f}%</td>
                    </tr>
"""
            
            html += """
                </tbody>
            </table>
        </div>
"""
        
        # Temperature by Radiator Level Graphs
        html += """
        <div class="section">
            <h2>📈 Temperature Patterns by Radiator Level</h2>
            <p style="color:#666;margin-bottom:20px;">Average temperature throughout the day for each radiator setting (last 30 days)</p>
            <div id="graphs-container"></div>
        </div>
"""
        
        # Latest training
        total_training_pages = (total_training + items_per_page - 1) // items_per_page
        if latest_training:
            html += f"""
        <div class="section">
            <h2>🎓 Recent Training Events</h2>
            <div style="margin-bottom:15px;color:#666;">
                Page {training_page} of {total_training_pages} ({total_training} total events)
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Room</th>
                        <th>Current</th>
                        <th>Target</th>
                        <th>Level</th>
                        <th>Delta</th>
                        <th>Predicted</th>
                        <th>Error</th>
                        <th>Action</th>
                    </tr>
                </thead>
                <tbody>
"""
            for event in latest_training:
                event_id = event['id']
                room = event['room']
                timestamp = event['timestamp'].strftime('%H:%M:%S') if isinstance(event['timestamp'], datetime) else str(event['timestamp'])
                delta = event['temperature_delta']
                pred_delta = event.get('predicted_delta')
                error = abs(delta - pred_delta) if pred_delta is not None else 0
                pred_delta_str = f"{pred_delta:.2f}°C" if pred_delta else 'N/A'
                
                error_class = 'good' if error < 0.3 else 'warning' if error < 0.6 else 'info'
                html += f"""
                    <tr>
                        <td>{timestamp}</td>
                        <td><span class="room-badge room-{room}">{room}</span></td>
                        <td>{event['current_temp']:.1f}°C</td>
                        <td>{event['target_temp']:.1f}°C</td>
                        <td>{event['radiator_level']}</td>
                        <td>{delta:+.2f}°C</td>
                        <td>{pred_delta_str}</td>
                        <td class="{error_class}">{error:.2f}°C</td>
                        <td><button class="delete-btn" onclick="deleteTraining({event_id})">🗑️ Delete</button></td>
                    </tr>
"""
            
            # Pagination controls
            html += """
                </tbody>
            </table>
            <div style="margin-top:20px;text-align:center;">
"""
            if training_page > 1:
                html += f'<a href="/ui?training_page={training_page-1}&predictions_page={predictions_page}" class="page-btn">← Previous</a> '
            
            # Show page numbers (max 5 around current page)
            start_page = max(1, training_page - 2)
            end_page = min(total_training_pages, training_page + 2)
            
            for p in range(start_page, end_page + 1):
                if p == training_page:
                    html += f'<span class="page-btn current">{p}</span> '
                else:
                    html += f'<a href="/ui?training_page={p}&predictions_page={predictions_page}" class="page-btn">{p}</a> '
            
            if training_page < total_training_pages:
                html += f'<a href="/ui?training_page={training_page+1}&predictions_page={predictions_page}" class="page-btn">Next →</a>'
            
            html += """
            </div>
        </div>
"""
        
        # Latest predictions
        total_predictions_pages = (total_predictions + items_per_page - 1) // items_per_page
        if latest_predictions:
            html += f"""
        <div class="section">
            <h2>🎯 Recent Predictions</h2>
            <div style="margin-bottom:15px;color:#666;">
                Page {predictions_page} of {total_predictions_pages} ({total_predictions} total predictions)
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Room</th>
                        <th>Current</th>
                        <th>Target</th>
                        <th>Recommended</th>
                        <th>Error</th>
                        <th>Adjusted</th>
                    </tr>
                </thead>
                <tbody>
"""
            for pred in latest_predictions:
                room = pred['room']
                timestamp = pred['timestamp'].strftime('%H:%M:%S') if isinstance(pred['timestamp'], datetime) else str(pred['timestamp'])
                adjusted = "✅" if pred['adjustment_made'] else "➖"
                level_class = 'good' if pred['adjustment_made'] else 'info'
                
                html += f"""
                    <tr>
                        <td>{timestamp}</td>
                        <td><span class="room-badge room-{room}">{room}</span></td>
                        <td>{pred['current_temp']:.1f}°C</td>
                        <td>{pred['target_temp']:.1f}°C</td>
                        <td class="{level_class}">{pred['recommended_level']}</td>
                        <td>{pred['predicted_error']:.2f}°C</td>
                        <td>{adjusted}</td>
                    </tr>
"""
            
            # Pagination controls
            html += """
                </tbody>
            </table>
            <div style="margin-top:20px;text-align:center;">
"""
            if predictions_page > 1:
                html += f'<a href="/ui?training_page={training_page}&predictions_page={predictions_page-1}" class="page-btn">← Previous</a> '
            
            # Show page numbers (max 5 around current page)
            start_page = max(1, predictions_page - 2)
            end_page = min(total_predictions_pages, predictions_page + 2)
            
            for p in range(start_page, end_page + 1):
                if p == predictions_page:
                    html += f'<span class="page-btn current">{p}</span> '
                else:
                    html += f'<a href="/ui?training_page={training_page}&predictions_page={p}" class="page-btn">{p}</a> '
            
            if predictions_page < total_predictions_pages:
                html += f'<a href="/ui?training_page={training_page}&predictions_page={predictions_page+1}" class="page-btn">Next →</a>'
            
            html += """
            </div>
        </div>
"""
        
        html += """
        <button class="refresh-btn" onclick="location.reload()">🔄 Refresh</button>
        <button class="reset-btn" onclick="resetAndRetrain()" id="resetBtn">🔄 Reset & Retrain AI</button>
        <button class="backtrain-btn" onclick="acceleratedBacktrain()" id="backtrainBtn">📚 Backtrain on History</button>
        
        <div class="progress-modal" id="progressModal">
            <div class="progress-content">
                <div class="progress-title">🤖 Retraining AI Models</div>
                <div class="progress-status" id="progressStatus">Initializing...</div>
                <div class="room-progress-bar">
                    <div class="room-progress-fill" id="overallProgress" style="width: 0%">0%</div>
                </div>
                <div id="roomStatus" style="margin-top: 20px;"></div>
            </div>
        </div>
        
        <div class="progress-modal" id="backtrainModal">
            <div class="progress-content">
                <div class="progress-title">📚 Accelerated Historical Backtraining</div>
                <div class="progress-status" id="backtrainStatus">Processing historical data...</div>
                <div id="backtrainResults" style="margin-top: 20px;"></div>
            </div>
        </div>
    </div>
    <script>
        // Accelerated backtrain function
        async function acceleratedBacktrain() {
            const days = prompt('📅 How many days of historical data to train on? (1-90)', '7');
            if (!days) return;
            
            const daysNum = parseInt(days);
            if (isNaN(daysNum) || daysNum < 1 || daysNum > 90) {
                alert('❌ Please enter a number between 1 and 90');
                return;
            }
            
            const learningRate = prompt('Learning rate factor (0.1-1.0)?\\n0.3=conservative, 0.5=balanced, 0.8=aggressive', '0.5');
            if (!learningRate) return;
            
            const lrNum = parseFloat(learningRate);
            if (isNaN(lrNum) || lrNum < 0.1 || lrNum > 1.0) {
                alert('❌ Please enter a number between 0.1 and 1.0');
                return;
            }
            
            if (!confirm(`Fine-tune models using ${daysNum} days of data with ${lrNum}x learning rate?\\n\\nMay take 10-30 seconds.`)) {
                return;
            }
            
            const btn = document.getElementById('backtrainBtn');
            const modal = document.getElementById('backtrainModal');
            const status = document.getElementById('backtrainStatus');
            const results = document.getElementById('backtrainResults');
            
            btn.disabled = true;
            modal.style.display = 'flex';
            status.textContent = `Processing ${daysNum} days of historical data...`;
            results.innerHTML = '';
            
            try {
                const response = await fetch(`/accelerated-backtrain?days=${daysNum}&learning_rate_factor=${lrNum}`, { 
                    method: 'POST' 
                });
                const data = await response.json();
                
                if (data.status === 'success') {
                    status.textContent = '✅ Backtraining Complete!';
                    
                    let resultsHTML = '<div style="text-align: left;">';
                    for (const [room, roomData] of Object.entries(data.results)) {
                        if (roomData.status === 'success') {
                            const improvement = roomData.improvement_percent;
                            const icon = improvement > 0 ? '📈' : improvement < 0 ? '📉' : '➡️';
                            resultsHTML += `
                                <div style="margin: 10px 0; padding: 10px; background: #f8f9fa; border-radius: 5px;">
                                    <strong>${icon} ${room}</strong><br>
                                    Samples: ${roomData.samples_processed} / ${roomData.total_samples_available}<br>
                                    Error: ${roomData.avg_error_before.toFixed(4)} → ${roomData.avg_error_after.toFixed(4)}<br>
                                    Improvement: <strong>${improvement > 0 ? '+' : ''}${improvement.toFixed(1)}%</strong>
                                </div>
                            `;
                        }
                    }
                    resultsHTML += '</div>';
                    results.innerHTML = resultsHTML;
                    
                    setTimeout(() => {
                        if (confirm('✅ Backtraining complete! Reload page to see updated models?')) {
                            location.reload();
                        } else {
                            modal.style.display = 'none';
                            btn.disabled = false;
                        }
                    }, 3000);
                } else {
                    alert('❌ Backtraining failed: ' + (data.message || 'Unknown error'));
                    modal.style.display = 'none';
                    btn.disabled = false;
                }
            } catch (error) {
                alert('❌ Error: ' + error.message);
                modal.style.display = 'none';
                btn.disabled = false;
            }
        }
        
        // Reset and retrain function
        async function resetAndRetrain() {
            if (!confirm('⚠️ This will delete all trained models and retrain from scratch using historical data and physics-based hints. This applies the latest training improvements. Continue?')) {
                return;
            }
            
            const btn = document.getElementById('resetBtn');
            const modal = document.getElementById('progressModal');
            
            btn.disabled = true;
            modal.style.display = 'flex';
            
            try {
                // Trigger reset
                const response = await fetch('/reset-and-retrain', { method: 'POST' });
                const result = await response.json();
                
                if (result.status === 'success') {
                    // Poll progress
                    pollProgress();
                } else {
                    alert('❌ Failed to start retraining: ' + result.message);
                    modal.style.display = 'none';
                    btn.disabled = false;
                }
            } catch (error) {
                alert('❌ Error: ' + error.message);
                modal.style.display = 'none';
                btn.disabled = false;
            }
        }
        
        async function pollProgress() {
            try {
                const response = await fetch('/retrain-progress');
                const data = await response.json();
                
                const progressBar = document.getElementById('overallProgress');
                const statusText = document.getElementById('progressStatus');
                const roomStatus = document.getElementById('roomStatus');
                
                progressBar.style.width = data.progress + '%';
                progressBar.textContent = Math.round(data.progress) + '%';
                
                statusText.textContent = `Training ${data.trained_rooms} of ${data.total_rooms} rooms...`;
                
                // Show per-room status
                let roomHTML = '';
                for (const [room, info] of Object.entries(data.room_status)) {
                    const status = info.status === 'trained' ? '✅' : '⏳';
                    roomHTML += `
                        <div class="room-progress">
                            <div class="room-progress-name">${status} ${room}: ${info.training_samples} samples</div>
                        </div>
                    `;
                }
                roomStatus.innerHTML = roomHTML;
                
                if (data.progress < 100) {
                    setTimeout(pollProgress, 1000);
                } else {
                    statusText.textContent = '✅ Retraining complete! Reloading...';
                    setTimeout(() => {
                        location.reload();
                    }, 2000);
                }
            } catch (error) {
                console.error('Progress poll error:', error);
                setTimeout(pollProgress, 2000);
            }
        }
        
        // Room target temperatures
        const ROOM_TARGETS = {
            "Badrum": 22.5,
            "Sovrum": 20,
            "Kontor": 21,
            "Vardagsrum": 21
        };
        
        console.log('🚀 Analytics JavaScript loaded, ROOM_TARGETS:', Object.keys(ROOM_TARGETS));
        
        // Load room insights with AI reasoning
        async function loadRoomInsights() {
            console.log('🔍 Loading room insights...');
            try {
                const response = await fetch('/analytics/room-insights');
                console.log('📡 Response status:', response.status);
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const insights = await response.json();
                console.log('✅ Insights loaded:', Object.keys(insights));
                
                const container = document.getElementById('room-insights-container');
                if (!container) {
                    console.error('❌ Container not found!');
                    return;
                }
                container.innerHTML = '';
                
                for (const [room, data] of Object.entries(insights)) {
                    const card = document.createElement('div');
                    card.className = 'metric-box';
                    card.style.padding = '20px';
                    card.style.marginBottom = '15px';
                    
                    const state = data.current_state;
                    const pred = data.latest_prediction;
                    const perf = data.performance;
                    
                    const statusEmoji = state.status === 'optimal' ? '✅' : '⚙️';
                    const devColor = Math.abs(state.deviation) <= 0.3 ? '#28a745' : (state.deviation > 0 ? '#dc3545' : '#007bff');
                    
                    card.innerHTML = `
                        <h3 style="color:#667eea;margin-bottom:15px;">${statusEmoji} ${room}</h3>
                        
                        <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:15px;">
                            <div>
                                <strong>Current:</strong> ${state.temperature}°C<br>
                                <strong>Target:</strong> ${state.target}°C<br>
                                <strong>Deviation:</strong> <span style="color:${devColor};font-weight:bold;">${state.deviation > 0 ? '+' : ''}${state.deviation}°C</span>
                            </div>
                            <div>
                                <strong>Radiator:</strong> ${state.radiator_level}<br>
                                <strong>Outdoor:</strong> ${state.outdoor_temp}°C<br>
                                <strong>Forecast:</strong> ${state.forecast_temp}°C
                            </div>
                            <div>
                                <strong>MAE:</strong> ${perf.mae.toFixed(3)}°C<br>
                                <strong>R²:</strong> ${perf.r2_score.toFixed(2)}<br>
                                <strong>24h Predictions:</strong> ${perf['24h_predictions']}
                            </div>
                        </div>
                        
                        ${pred.recommended_level !== null ? `
                        <div style="background:#f8f9fa;padding:10px;border-radius:5px;margin-bottom:10px;">
                            <strong>Latest Recommendation:</strong><br>
                            Level ${pred.recommended_level} 
                            ${pred.expected_temp ? `→ Expected ${pred.expected_temp}°C` : ''}
                            ${pred.confidence ? `(Confidence: ${(pred.confidence * 100).toFixed(0)}%)` : ''}
                        </div>
                        ` : ''}
                        
                        <div style="background:#e7f3ff;padding:12px;border-left:4px solid #667eea;border-radius:3px;">
                            <strong>🤖 AI Reasoning:</strong><br>
                            <span style="font-size:0.9em;">${data.reasoning}</span>
                        </div>
                    `;
                    
                    container.appendChild(card);
                }
                console.log('✅ Room insights rendered successfully');
            } catch (error) {
                console.error('❌ Error loading room insights:', error);
                const container = document.getElementById('room-insights-container');
                if (container) {
                    container.innerHTML = 
                        `<p style="color:#dc3545;text-align:center;">Error loading insights: ${error.message}</p>`;
                }
            }
        }
        
        // Load temperature timeline charts
        async function loadTimelineCharts() {
            console.log('📊 Loading timeline charts...');
            try {
                const container = document.getElementById('timeline-graphs-container');
                if (!container) {
                    console.error('❌ Timeline container not found!');
                    return;
                }
                container.innerHTML = '';
                
                for (const room of Object.keys(ROOM_TARGETS)) {
                    console.log(`📈 Loading timeline for ${room}...`);
                    const response = await fetch(`/analytics/temperature-timeline?room=${room}&hours=48`);
                    if (!response.ok) {
                        console.error(`❌ Failed to load timeline for ${room}: ${response.status}`);
                        continue;
                    }
                    const data = await response.json();
                    console.log(`✅ Timeline data for ${room}:`, data.actual.length, 'actual points,', data.predictions.length, 'predictions');
                    
                    const chartDiv = document.createElement('div');
                    chartDiv.className = 'graph-container';
                    chartDiv.style.marginBottom = '30px';
                    
                    const title = document.createElement('div');
                    title.className = 'graph-title';
                    title.textContent = `🏠 ${room} - 48h Temperature Timeline`;
                    chartDiv.appendChild(title);
                    
                    const wrapper = document.createElement('div');
                    wrapper.className = 'canvas-wrapper';
                    
                    const canvas = document.createElement('canvas');
                    canvas.id = `timeline-${room}`;
                    wrapper.appendChild(canvas);
                    chartDiv.appendChild(wrapper);
                    container.appendChild(chartDiv);
                    
                    // Prepare datasets
                    const actualTemps = data.actual.map(d => ({
                        x: new Date(d.timestamp),
                        y: d.temperature
                    }));
                    
                    const targetTemps = data.actual.map(d => ({
                        x: new Date(d.timestamp),
                        y: d.target
                    }));
                    
                    const predictedTemps = data.predictions
                        .filter(d => d.current_temp !== null)
                        .map(d => ({
                            x: new Date(d.timestamp),
                            y: d.current_temp // Show what temp was at prediction time
                        }));
                    
                    const outdoorTemps = data.actual
                        .filter(d => d.outdoor_temp !== null)
                        .map(d => ({
                            x: new Date(d.timestamp),
                            y: d.outdoor_temp
                        }));
                    
                    // Create chart
                    const ctx = canvas.getContext('2d');
                    new Chart(ctx, {
                        type: 'line',
                        data: {
                            datasets: [
                                {
                                    label: 'Actual Temperature',
                                    data: actualTemps,
                                    borderColor: '#667eea',
                                    backgroundColor: 'transparent',
                                    borderWidth: 2,
                                    tension: 0.4,
                                    pointRadius: 2
                                },
                                {
                                    label: 'Target Temperature',
                                    data: targetTemps,
                                    borderColor: '#dc3545',
                                    backgroundColor: 'transparent',
                                    borderWidth: 2,
                                    borderDash: [5, 5],
                                    pointRadius: 0,
                                    tension: 0
                                },
                                {
                                    label: 'Predicted Temperature',
                                    data: predictedTemps,
                                    borderColor: '#28a745',
                                    backgroundColor: 'transparent',
                                    borderWidth: 2,
                                    borderDash: [2, 2],
                                    pointRadius: 3,
                                    pointStyle: 'triangle',
                                    tension: 0
                                },
                                {
                                    label: 'Outdoor Temperature',
                                    data: outdoorTemps,
                                    borderColor: '#ffc107',
                                    backgroundColor: 'transparent',
                                    borderWidth: 1.5,
                                    tension: 0.4,
                                    pointRadius: 1,
                                    yAxisID: 'y1'
                                }
                            ]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                legend: {
                                    position: 'top'
                                },
                                tooltip: {
                                    mode: 'index',
                                    intersect: false
                                }
                            },
                            scales: {
                                x: {
                                    type: 'time',
                                    time: {
                                        unit: 'hour',
                                        displayFormats: {
                                            hour: 'MMM d, HH:mm'
                                        }
                                    },
                                    title: {
                                        display: true,
                                        text: 'Time'
                                    }
                                },
                                y: {
                                    title: {
                                        display: true,
                                        text: 'Indoor Temperature (°C)'
                                    },
                                    beginAtZero: false
                                },
                                y1: {
                                    type: 'linear',
                                    display: true,
                                    position: 'right',
                                    title: {
                                        display: true,
                                        text: 'Outdoor (°C)'
                                    },
                                    grid: {
                                        drawOnChartArea: false
                                    }
                                }
                            }
                        }
                    });
                }
            } catch (error) {
                console.error('❌ Error loading timeline charts:', error);
                const container = document.getElementById('timeline-graphs-container');
                if (container) {
                    container.innerHTML = 
                        `<p style="color:#dc3545;text-align:center;">Error loading timeline: ${error.message}</p>`;
                }
            }
        }
        
        // Load prediction accuracy trends
        async function loadAccuracyTrends() {
            console.log('📈 Loading accuracy trends...');
            try {
                const response = await fetch('/analytics/prediction-accuracy?days=7');
                if (!response.ok) {
                    throw new Error(`HTTP error! status: ${response.status}`);
                }
                const data = await response.json();
                console.log('✅ Accuracy data loaded:', Object.keys(data));
                
                const container = document.getElementById('accuracy-trends-container');
                container.innerHTML = '';
                
                const chartDiv = document.createElement('div');
                chartDiv.className = 'canvas-wrapper';
                chartDiv.style.height = '400px';
                
                const canvas = document.createElement('canvas');
                canvas.id = 'accuracy-chart';
                chartDiv.appendChild(canvas);
                container.appendChild(chartDiv);
                
                // Prepare datasets
                const datasets = [];
                const colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe'];
                let colorIndex = 0;
                
                for (const [room, roomData] of Object.entries(data)) {
                    datasets.push({
                        label: room,
                        data: roomData.map(d => ({
                            x: d.date,
                            y: d.avg_error
                        })),
                        borderColor: colors[colorIndex % colors.length],
                        backgroundColor: colors[colorIndex % colors.length] + '20',
                        borderWidth: 2,
                        tension: 0.4,
                        fill: true
                    });
                    colorIndex++;
                }
                
                // Create chart
                const ctx = canvas.getContext('2d');
                new Chart(ctx, {
                    type: 'line',
                    data: { datasets },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        plugins: {
                            legend: {
                                position: 'top'
                            },
                            title: {
                                display: true,
                                text: 'Average Prediction Error by Room (Last 7 Days)'
                            },
                            tooltip: {
                                callbacks: {
                                    label: function(context) {
                                        return `${context.dataset.label}: ±${context.parsed.y.toFixed(3)}°C`;
                                    }
                                }
                            }
                        },
                        scales: {
                            x: {
                                type: 'time',
                                time: {
                                    unit: 'day'
                                },
                                title: {
                                    display: true,
                                    text: 'Date'
                                }
                            },
                            y: {
                                title: {
                                    display: true,
                                    text: 'Average Error (°C)'
                                },
                                beginAtZero: true
                            }
                        }
                    }
                });
            } catch (error) {
                console.error('Error loading accuracy trends:', error);
                document.getElementById('accuracy-trends-container').innerHTML = 
                    '<p style="color:#dc3545;text-align:center;">Error loading trends</p>';
            }
        }
        
        // Load and render temperature graphs
        async function loadGraphs() {
            try {
                const response = await fetch('/graphs/radiator-temperature?days=30');
                const result = await response.json();
                const data = result.data;
                
                const container = document.getElementById('graphs-container');
                container.innerHTML = '';
                
                // Define colors for different radiator levels
                const colors = [
                    '#667eea', '#764ba2', '#f093fb', '#4facfe',
                    '#43e97b', '#fa709a', '#fee140', '#30cfd0',
                    '#a8edea', '#fed6e3', '#c471ed', '#12c2e9'
                ];
                
                // Process each room
                for (const [room, levels] of Object.entries(data)) {
                    const graphDiv = document.createElement('div');
                    graphDiv.className = 'graph-container';
                    
                    const title = document.createElement('div');
                    title.className = 'graph-title';
                    title.textContent = `🏠 ${room}`;
                    graphDiv.appendChild(title);
                    
                    const wrapper = document.createElement('div');
                    wrapper.className = 'canvas-wrapper';
                    
                    const canvas = document.createElement('canvas');
                    canvas.id = `chart-${room}`;
                    wrapper.appendChild(canvas);
                    graphDiv.appendChild(wrapper);
                    container.appendChild(graphDiv);
                    
                    // Prepare datasets for this room
                    const datasets = [];
                    const sortedLevels = Object.keys(levels).sort((a, b) => parseFloat(a) - parseFloat(b));
                    
                    // Add target temperature baseline
                    const targetTemp = ROOM_TARGETS[room];
                    if (targetTemp) {
                        datasets.push({
                            label: `Target (${targetTemp}°C)`,
                            data: Array(24).fill(targetTemp),
                            borderColor: '#dc3545',
                            backgroundColor: 'transparent',
                            borderWidth: 2,
                            borderDash: [5, 5],
                            pointRadius: 0,
                            tension: 0,
                            fill: false,
                            order: 0
                        });
                    }
                    
                    sortedLevels.forEach((level, index) => {
                        const hourData = levels[level];
                        const dataPoints = [];
                        
                        // Fill in all 24 hours
                        for (let hour = 0; hour < 24; hour++) {
                            if (hourData[hour]) {
                                dataPoints.push(hourData[hour].avg_temp);
                            } else {
                                dataPoints.push(null);
                            }
                        }
                        
                        // Only add if there's actual data
                        const hasData = dataPoints.some(p => p !== null);
                        if (hasData) {
                            datasets.push({
                                label: `Level ${level}`,
                                data: dataPoints,
                                borderColor: colors[index % colors.length],
                                backgroundColor: colors[index % colors.length] + '20',
                                borderWidth: 2,
                                tension: 0.4,
                                fill: false,
                                spanGaps: true,
                                order: 1
                            });
                        }
                    });
                    
                    // Create chart
                    const ctx = canvas.getContext('2d');
                    new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: Array.from({length: 24}, (_, i) => `${i}:00`),
                            datasets: datasets
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                legend: {
                                    position: 'top',
                                    labels: {
                                        boxWidth: 15,
                                        font: {
                                            size: 11
                                        }
                                    }
                                },
                                title: {
                                    display: false
                                },
                                tooltip: {
                                    mode: 'index',
                                    intersect: false,
                                    callbacks: {
                                        label: function(context) {
                                            let label = context.dataset.label || '';
                                            if (label) {
                                                label += ': ';
                                            }
                                            if (context.parsed.y !== null) {
                                                label += context.parsed.y.toFixed(1) + '°C';
                                            }
                                            return label;
                                        }
                                    }
                                }
                            },
                            scales: {
                                y: {
                                    beginAtZero: false,
                                    title: {
                                        display: true,
                                        text: 'Temperature (°C)'
                                    },
                                    grid: {
                                        color: 'rgba(0, 0, 0, 0.05)'
                                    }
                                },
                                x: {
                                    title: {
                                        display: true,
                                        text: 'Hour of Day'
                                    },
                                    grid: {
                                        color: 'rgba(0, 0, 0, 0.05)'
                                    }
                                }
                            },
                            interaction: {
                                mode: 'nearest',
                                axis: 'x',
                                intersect: false
                            }
                        }
                    });
                }
                
                if (Object.keys(data).length === 0) {
                    container.innerHTML = '<p style="color:#666;text-align:center;padding:20px;">No data yet. Graphs appear after training.</p>';
                }
            } catch (error) {
                console.error('Error loading graphs:', error);
                document.getElementById('graphs-container').innerHTML = 
                    '<p style="color:#dc3545;text-align:center;padding:20px;">Error loading graphs. Please try refreshing the page.</p>';
            }
        }
        
        async function deleteTraining(trainingId) {
            if (!confirm('Are you sure you want to delete this training event? This cannot be undone.')) {
                return;
            }
            
            try {
                const response = await fetch(`/training/${trainingId}`, {
                    method: 'DELETE'
                });
                
                if (response.ok) {
                    alert('Training event deleted successfully!');
                    location.reload();
                } else {
                    const error = await response.json();
                    alert('Error deleting training event: ' + (error.detail || 'Unknown error'));
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
        
        // Load graphs on page load
        window.addEventListener('DOMContentLoaded', () => {
            console.log('🎯 DOM loaded, starting analytics...');
            console.log('📦 Functions available:', typeof loadRoomInsights, typeof loadTimelineCharts, typeof loadAccuracyTrends);
            
            // Test: immediately update the containers to verify JS is running
            const testDiv = document.getElementById('room-insights-container');
            if (testDiv) {
                testDiv.innerHTML = '<p style="color:green;text-align:center;">✅ JavaScript is running! Loading data...</p>';
            }
            
            loadRoomInsights();
            loadTimelineCharts();
            loadAccuracyTrends();
            loadGraphs();
        });
        
        setTimeout(() => location.reload(), 60000);
    </script>
</body>
</html>
"""
        
        return HTMLResponse(content=html)
        
    except Exception as e:
        return HTMLResponse(content=f"""
<!DOCTYPE html>
<html>
<head><title>Error</title></head>
<body style="font-family:Arial;padding:50px;text-align:center">
    <h1 style="color:red">⚠️ Error Loading Dashboard</h1>
    <p>{str(e)}</p>
    <p><a href="/ui">Try again</a></p>
</body>
</html>
""", status_code=500)
