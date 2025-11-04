# Smart Radiator AI Service - v1.0.1
from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import os, joblib, requests, json
from river import forest, preprocessing
from forecast import get_weather

app = FastAPI(title="Smart Radiator AI")
MODELS_DIR = "models"
STATE_FILE = "last_temps.json"
os.makedirs(MODELS_DIR, exist_ok=True)
if not os.path.exists(STATE_FILE):
    open(STATE_FILE, "w").write("{}")

TELEGRAM_WEBHOOK = os.getenv("TELEGRAM_WEBHOOK")

ROOMS = {
    "Badrum":     {"scale": [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6], "target": 22.5},
    "Sovrum":     {"scale": [0,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9], "target": 20},
    "Kontor":     {"scale": [0,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9], "target": 20},
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

def load_model(room):
    path = f"{MODELS_DIR}/{room}.pkl"
    if os.path.exists(path):
        return joblib.load(path)
    return preprocessing.StandardScaler() | forest.ARFRegressor()

def save_model(room, model):
    joblib.dump(model, f"{MODELS_DIR}/{room}.pkl")

def load_state():
    with open(STATE_FILE, "r") as f:
        return json.load(f)

def save_state(data):
    with open(STATE_FILE, "w") as f:
        json.dump(data, f)

@app.get("/")
def status():
    """Return AI service status and model information"""
    state = load_state()
    
    models_info = {}
    for room in ROOMS.keys():
        model_path = f"{MODELS_DIR}/{room}.pkl"
        model_exists = os.path.exists(model_path)
        models_info[room] = {
            "trained": model_exists,
            "last_temp": state.get(room),
            "target_temp": ROOMS[room]["target"],
            "scale_range": f"{ROOMS[room]['scale'][0]}-{ROOMS[room]['scale'][-1]}",
        }
    
    # Get current weather
    outdoor, forecast = get_weather()
    
    return {
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "rooms": models_info,
        "weather": {
            "outdoor_temp": outdoor,
            "forecast_3h": forecast,
        },
        "telegram_webhook_configured": TELEGRAM_WEBHOOK is not None,
    }

@app.post("/train")
def train(state: RoomState):
    model = load_model(state.room)
    db = load_state()
    prev = db.get(state.room, state.current_temp)
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

    model.learn_one(features, delta)
    save_model(state.room, model)

    db[state.room] = state.current_temp
    save_state(db)

    return {"trained": True, "delta": delta}

@app.post("/predict")
def predict(state: RoomState):
    model = load_model(state.room)
    if state.outdoor_temp is None or state.forecast_temp is None:
        outside, forecast = get_weather()
        state.outdoor_temp, state.forecast_temp = outside, forecast

    best_lvl = None
    best_err = float("inf")

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
        if error < best_err:
            best_err, best_lvl = error, lvl

    if best_lvl is None:
        return {"recommended": None, "error": None}

    if abs(best_lvl - state.radiator_level) >= 1:
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

    return {"recommended": best_lvl, "error": best_err}
