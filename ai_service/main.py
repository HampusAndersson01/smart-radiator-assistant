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

app = FastAPI(title="Smart Radiator AI")
DATABASE_URL = os.getenv("DATABASE_URL")

# Initialize database on startup
try:
    db.init_database()
except Exception as e:
    print(f"Warning: Could not initialize database: {e}")
    print("Some features may not work without DATABASE_URL configured")

TELEGRAM_WEBHOOK = os.getenv("TELEGRAM_WEBHOOK")

ROOMS = {
    "Badrum":     {"scale": [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6], "target": 22.5},
    "Sovrum":     {"scale": [0,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9], "target": 20},
    "Kontor":     {"scale": [0,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9], "target": 21},
    "Vardagsrum":{"scale": [0,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9], "target": 21},
}

class RoomState(BaseModel):
    room: str
    current_temp: float
    target_temp: float
    radiator_level: int
    outdoor_temp: float | None = None
    forecast_temp: float | None = None
    timestamp: str

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
        outdoor, forecast = get_weather()
        if outdoor is None:
            outdoor = 0.0
        if forecast is None:
            forecast = 0.0
    except Exception as e:
        print(f"Error getting weather: {e}")
        outdoor, forecast = 0.0, 0.0
    
    return {
        "status": "online",
        "version": "2.0.0-database",
        "timestamp": datetime.now().isoformat(),
        "rooms": models_info,
        "weather": {
            "outdoor_temp": outdoor,
            "forecast_3h": forecast,
        },
        "telegram_webhook_configured": TELEGRAM_WEBHOOK is not None,
        "database_connected": DATABASE_URL is not None,
    }

@app.post("/train")
def train(state: RoomState):
    """Train the model with new data"""
    model = load_model(state.room)
    
    # Get previous temperature from database
    prev = db.get_latest_temp(state.room)
    if prev is None:
        prev = state.current_temp
    delta = state.current_temp - prev

    # Add weather if missing
    if state.outdoor_temp is None or state.forecast_temp is None:
        outside, forecast = get_weather()
        state.outdoor_temp = outside
        state.forecast_temp = forecast

    features = {
        "current_temp": state.current_temp,
        "target_temp": state.target_temp,
        "outdoor_temp": state.outdoor_temp,
        "forecast_temp": state.forecast_temp,
        "radiator_level": state.radiator_level,
        "hour_of_day": datetime.now().hour,
    }

    # Train the model
    model.learn_one(features, delta)
    save_model(state.room, model)

    # Update metrics
    m = model_metrics[state.room]
    m.training_samples += 1
    
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

    return {
        "trained": True, 
        "delta": round(delta, 3),
        "training_samples": m.training_samples,
        "model_mae": round(m.mae.get(), 3) if m.training_samples > 0 else None
    }

@app.post("/predict")
def predict(state: RoomState):
    """Get radiator level recommendation"""
    model = load_model(state.room)
    if state.outdoor_temp is None or state.forecast_temp is None:
        outside, forecast = get_weather()
        state.outdoor_temp, state.forecast_temp = outside, forecast

    best_lvl = None
    best_err = float("inf")
    prediction_details = []

    for lvl in ROOMS[state.room]["scale"]:
        feat = {
            "current_temp": state.current_temp,
            "target_temp": state.target_temp,
            "outdoor_temp": state.outdoor_temp,
            "forecast_temp": state.forecast_temp,
            "radiator_level": lvl,
            "hour_of_day": datetime.now().hour,
        }
        try:
            delta = model.predict_one(feat) or 0
        except Exception:
            delta = 0
        predicted_temp = state.current_temp + delta
        error = abs(predicted_temp - state.target_temp)
        
        prediction_details.append({
            "level": lvl,
            "predicted_temp": round(predicted_temp, 2),
            "error": round(error, 2)
        })
        
        if error < best_err:
            best_err, best_lvl = error, lvl

    # Update metrics
    m = model_metrics[state.room]
    m.predictions_made += 1
    
    adjustment_made = False
    if best_lvl is None:
        # Update metrics in database
        db.update_ai_metrics(
            state.room,
            predictions_made=m.predictions_made
        )
        return {"recommended": None, "error": None}

    if abs(best_lvl - state.radiator_level) >= 1:
        m.adjustments_made += 1
        adjustment_made = True
        
        msg = (
            f"🏠 {state.room}: set radiator to {best_lvl}\n"
            f"🌡️ now {state.current_temp}°C → target {state.target_temp}°C\n"
            f"🌤️ outside {state.outdoor_temp}°C, forecast {state.forecast_temp}°C"
        )
        if TELEGRAM_WEBHOOK:
            try:
                requests.post(TELEGRAM_WEBHOOK, json={"text": msg}, timeout=5)
            except Exception:
                pass
    
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
        state.radiator_level, best_lvl, best_err, adjustment_made,
        state.outdoor_temp, state.forecast_temp
    )
    
    # Save room state to database
    db.save_room_state(
        state.room, state.current_temp, state.target_temp,
        state.radiator_level, state.outdoor_temp, state.forecast_temp
    )

    return {
        "recommended": best_lvl, 
        "error": round(best_err, 2),
        "current_level": state.radiator_level,
        "adjustment_needed": adjustment_made,
        "predictions_made": m.predictions_made,
        "prediction_details": prediction_details[:5]  # Top 5 options
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

@app.get("/ui", response_class=HTMLResponse)
def ui_dashboard():
    """Web UI dashboard showing AI service status, training, and predictions"""
    from fastapi.responses import HTMLResponse
    
    try:
        # Get latest training events (last 10)
        latest_training = db.get_latest_training_events(limit=10)
        
        # Get latest predictions (last 10)
        latest_predictions = db.get_latest_predictions(limit=10)
        
        # Get training count in last 24 hours
        training_count_24h = db.get_training_count_last_24h()
        
        # Get overall metrics
        all_metrics = db.get_ai_metrics()
        
        # Get current weather
        try:
            outdoor, forecast = get_weather()
            if outdoor is None:
                outdoor = "N/A"
            if forecast is None:
                forecast = "N/A"
        except Exception:
            outdoor, forecast = "N/A", "N/A"
        
        # Build HTML
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Smart Radiator AI - Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
            min-height: 100vh;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        .header {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
            text-align: center;
        }}
        .header h1 {{
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}
        .header .status {{
            color: #28a745;
            font-size: 1.2em;
            font-weight: bold;
        }}
        .header .timestamp {{
            color: #666;
            font-size: 0.9em;
            margin-top: 10px;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
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
            font-size: 0.9em;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }}
        .stat-card .value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #333;
            margin: 10px 0;
        }}
        .stat-card .label {{
            color: #666;
            font-size: 0.85em;
        }}
        .section {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }}
        .section h2 {{
            color: #667eea;
            font-size: 1.8em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th {{
            background: #667eea;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
            text-transform: uppercase;
            font-size: 0.85em;
            letter-spacing: 0.5px;
        }}
        td {{
            padding: 12px;
            border-bottom: 1px solid #eee;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
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
        .temp {{
            font-weight: 600;
            color: #ff6b6b;
        }}
        .target {{
            font-weight: 600;
            color: #4ecdc4;
        }}
        .good {{
            color: #28a745;
            font-weight: bold;
        }}
        .warning {{
            color: #ffc107;
            font-weight: bold;
        }}
        .info {{
            color: #17a2b8;
            font-weight: bold;
        }}
        .no-data {{
            text-align: center;
            padding: 40px;
            color: #999;
            font-style: italic;
        }}
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
            transition: all 0.3s ease;
        }}
        .refresh-btn:hover {{
            background: #764ba2;
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.6);
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
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
            letter-spacing: 0.5px;
        }}
        .metric-box .metric-value {{
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏠 Smart Radiator AI Dashboard</h1>
            <div class="status">✅ System Online</div>
            <div class="timestamp">Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <h3>Training Events (24h)</h3>
                <div class="value good">{training_count_24h}</div>
                <div class="label">Model updates in last 24 hours</div>
            </div>
            <div class="stat-card">
                <h3>Total Rooms</h3>
                <div class="value info">{len(ROOMS)}</div>
                <div class="label">Active monitoring zones</div>
            </div>
            <div class="stat-card">
                <h3>Outdoor Temperature</h3>
                <div class="value temp">{outdoor}°C</div>
                <div class="label">Current weather</div>
            </div>
            <div class="stat-card">
                <h3>Forecast (3h)</h3>
                <div class="value target">{forecast}°C</div>
                <div class="label">Predicted temperature</div>
            </div>
        </div>

        <div class="section">
            <h2>📊 Room Performance Metrics</h2>
            <div class="metrics-grid">
"""
        
        # Add room metrics
        for room, metrics in all_metrics.items():
            training_samples = metrics.get('training_samples', 0)
            predictions_made = metrics.get('predictions_made', 0)
            mae = metrics.get('mae', 0)
            
            html_content += f"""
                <div class="metric-box">
                    <div class="metric-label">{room}</div>
                    <div class="metric-value">{training_samples}</div>
                    <div class="label">Training samples</div>
                    <div class="label" style="margin-top: 5px;">MAE: {mae:.3f}</div>
                    <div class="label">Predictions: {predictions_made}</div>
                </div>
"""
        
        html_content += """
            </div>
        </div>

        <div class="section">
            <h2>🎓 Latest Training Events</h2>
"""
        
        if latest_training:
            html_content += """
            <table>
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Room</th>
                        <th>Current Temp</th>
                        <th>Target Temp</th>
                        <th>Radiator Level</th>
                        <th>Temp Delta</th>
                        <th>Predicted Delta</th>
                        <th>Outdoor Temp</th>
                    </tr>
                </thead>
                <tbody>
"""
            for event in latest_training:
                room = event['room']
                timestamp = event['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(event['timestamp'], datetime) else str(event['timestamp'])
                current_temp = event['current_temp']
                target_temp = event['target_temp']
                radiator_level = event['radiator_level']
                temp_delta = event['temperature_delta']
                predicted_delta = event.get('predicted_delta')
                outdoor_temp = event.get('outdoor_temp', 'N/A')
                
                predicted_str = f"{predicted_delta:.3f}" if predicted_delta is not None else "N/A"
                
                html_content += f"""
                    <tr>
                        <td>{timestamp}</td>
                        <td><span class="room-badge room-{room}">{room}</span></td>
                        <td class="temp">{current_temp:.1f}°C</td>
                        <td class="target">{target_temp:.1f}°C</td>
                        <td>{radiator_level}</td>
                        <td>{temp_delta:+.3f}°C</td>
                        <td>{predicted_str}</td>
                        <td>{outdoor_temp if outdoor_temp != 'N/A' else 'N/A'}</td>
                    </tr>
"""
            
            html_content += """
                </tbody>
            </table>
"""
        else:
            html_content += '<div class="no-data">No training events found</div>'
        
        html_content += """
        </div>

        <div class="section">
            <h2>🎯 Latest Predictions</h2>
"""
        
        if latest_predictions:
            html_content += """
            <table>
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Room</th>
                        <th>Current Temp</th>
                        <th>Target Temp</th>
                        <th>Current Level</th>
                        <th>Recommended Level</th>
                        <th>Predicted Error</th>
                        <th>Adjustment Made</th>
                    </tr>
                </thead>
                <tbody>
"""
            for pred in latest_predictions:
                room = pred['room']
                timestamp = pred['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if isinstance(pred['timestamp'], datetime) else str(pred['timestamp'])
                current_temp = pred['current_temp']
                target_temp = pred['target_temp']
                current_level = pred['current_radiator_level']
                recommended_level = pred['recommended_level']
                predicted_error = pred['predicted_error']
                adjustment_made = pred['adjustment_made']
                
                adjustment_icon = "✅" if adjustment_made else "➖"
                adjustment_class = "good" if adjustment_made else "info"
                
                html_content += f"""
                    <tr>
                        <td>{timestamp}</td>
                        <td><span class="room-badge room-{room}">{room}</span></td>
                        <td class="temp">{current_temp:.1f}°C</td>
                        <td class="target">{target_temp:.1f}°C</td>
                        <td>{current_level}</td>
                        <td class="{adjustment_class}">{recommended_level}</td>
                        <td>{predicted_error:.2f}°C</td>
                        <td class="{adjustment_class}">{adjustment_icon}</td>
                    </tr>
"""
            
            html_content += """
                </tbody>
            </table>
"""
        else:
            html_content += '<div class="no-data">No predictions found</div>'
        
        html_content += """
        </div>

        <button class="refresh-btn" onclick="location.reload()">🔄 Refresh</button>
    </div>

    <script>
        // Auto-refresh every 30 seconds
        setTimeout(function() {
            location.reload();
        }, 30000);
    </script>
</body>
</html>
"""
        
        return HTMLResponse(content=html_content)
        
    except Exception as e:
        error_html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Error - Smart Radiator AI</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            background: #f44336;
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100vh;
            margin: 0;
        }}
        .error-container {{
            background: white;
            color: #333;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            text-align: center;
        }}
        h1 {{ color: #f44336; }}
    </style>
</head>
<body>
    <div class="error-container">
        <h1>⚠️ Error</h1>
        <p>Could not load dashboard: {str(e)}</p>
        <p><a href="/ui">Try again</a></p>
    </div>
</body>
</html>
"""
        return HTMLResponse(content=error_html, status_code=500)

