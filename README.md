<div align="center">

# 🏠 Radiator AI Assistant WIP

**Intelligent home heating automation with AI-powered predictions**

[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://hub.docker.com/u/namelesshampus)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*Automate your radiator settings with machine learning and smart home integration*

[Features](#-features) • [Quick Start](#-quick-start) • [Documentation](#-documentation) • [API](#-api-endpoints)

</div>

---

## 📋 Overview

Smart Radiator Assistant is a complete home automation solution that uses **online machine learning** to optimize your radiator settings. It learns from your preferences and environmental data to maintain perfect room temperature while minimizing energy consumption.

### �️ Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  Home Assistant │      │      n8n         │      │   Telegram Bot  │
│   (Sensors)     │────▶│   (Automation)    │◀──▶│  (Manual Ctrl)  │
└─────────────────┘      └──────────────────┘      └─────────────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │   AI Service     │
                         │   FastAPI + ML   │
                         └──────────────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │   PostgreSQL     │
                         │   (Model State)  │
                         └──────────────────┘
```

## ✨ Features

<table>
<tr>
<td width="50%">

### 🤖 AI-Powered
- **Online Machine Learning** using River (ARFRegressor)
- **8-hour stability prediction** - optimizes for long-term comfort, not instant reactions
- **Dual weather forecasts** (3h + 10h ahead) for better predictions
- **Automatic back-evaluation** - learns from past prediction errors every training cycle
- **Smart adjustment logic** - gradual changes with ±0.3°C tolerance band
- **Physics-based training** - understands radiator dynamics (higher level = more heat)
- Adapts to your heating preferences in real-time

</td>
<td width="50%">

### 🔧 Easy Integration
- **Docker-based deployment** - up and running in minutes
- **Telegram bot** for manual control
- **n8n workflows** for Home Assistant integration
- **RESTful API** with interactive docs
- **Web dashboard** - visual monitoring and analytics

</td>
</tr>
<tr>
<td width="50%">

### 📊 Advanced Analytics
- Real-time AI statistics (MAE, RMSE, R²)
- **Automatic back-evaluation** - compares predictions vs actual outcomes
- **Accelerated historical backtraining** - refine models without reset
- Training history and prediction logs
- **Database-backed persistence** - survives restarts
- Export data to CSV for analysis

</td>
<td width="50%">

### 🚀 Production Ready
- Pre-built Docker images on Docker Hub
- Automated build & deployment scripts
- PostgreSQL for data persistence
- **Automatic back-evaluation** - learns from past predictions every hour
- **Gradual adjustment logic** - prevents drastic temperature changes
- Scalable microservice architecture

</td>
</tr>
</table>

### �️ 8-Hour Stability Prediction
The AI simulates temperature evolution **8 hours ahead** for each radiator level:
- Predicts how each setting will affect temperature over time
- Uses 3h and 10h weather forecasts for outdoor conditions
- Selects level with **lowest average error** over 8 hours
- Optimizes for long-term comfort, not instant reactions

### 🎯 Self-Improving AI
The system continuously learns from its own predictions:
1. **Makes predictions** (e.g., "temperature will be 19.5°C in 3 hours")
2. **Stores predictions** with timestamp in database
3. **After 3 hours**, compares actual temperature to prediction
4. **Automatically trains** on the difference (back-evaluation)
5. **Gets better over time** without manual intervention

### 🔧 Smart Adjustment Logic
Prevents overreacting and ensures comfort:
- **Acceptable Deviation**: ±0.3°C tolerance (no adjustment if close to target)
- **Gradual Changes**: Maximum ±1.5 level change per cycle
- **Minimum Threshold**: Only adjusts if change ≥0.5 levels
- **Example**: Instead of jumping 5 → 0, gradually adjusts 5 → 3.5

### 📚 Accelerated Historical Backtraining
Refine models using past data **without resetting**:
- Processes days/weeks of historical training data
- Preserves all current learned knowledge
- Time-based confidence decay (recent data weighted higher)
- Configurable learning rate (conservative to aggressive)
- Shows before/after error improvement metrics

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- PostgreSQL database
- Telegram Bot Token ([get one from @BotFather](https://t.me/botfather))
- (Optional) n8n instance for automation

### 1️⃣ Clone & Configure

```bash
git clone https://github.com/HampusAndersson01/smart-radiator-assistant.git
cd smart-radiator-assistant
cp .env.example .env
```

Edit `.env` and add your credentials:
```bash
BOT_TOKEN=your_telegram_bot_token
POSTGRES_PASSWORD=your_db_password
POSTGRES_HOST=your_db_host
```

### 2️⃣ Deploy with Docker Compose

```bash
# Pull latest images and start services
docker compose pull
docker compose up -d

# Check status
docker compose ps
docker compose logs -f
```

### 3️⃣ Verify Installation

```bash
# Test AI service
curl http://localhost:8000/docs

# Check AI statistics
curl http://localhost:8000/stats

# Send test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "room": "Living Room",
    "current_temp": 20.5,
    "target_temp": 22.0,
    "radiator_level": 3,
    "timestamp": "now"
  }'
```

## 📦 Services

### AI Service
**Port:** `8000`  
**Image:** `namelesshampus/radiator-ai-service:latest`

FastAPI microservice providing ML-powered radiator predictions and training endpoints.

### Telegram Bot
**Image:** `namelesshampus/radiator-bot:latest`

Interactive Telegram bot for manual radiator control and status queries.

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `BOT_TOKEN` | Telegram bot authentication token | ✅ | - |
| `DATABASE_URL` | PostgreSQL connection string | ✅ | - |
| `POSTGRES_USER` | Database username | ❌ | `postgres` |
| `POSTGRES_PASSWORD` | Database password | ✅ | - |
| `POSTGRES_HOST` | Database host | ❌ | `host.docker.internal` |
| `TELEGRAM_WEBHOOK` | Webhook URL for notifications | ❌ | - |
| `ALLOWED_CHAT_IDS` | Comma-separated list of allowed Telegram chat IDs | ❌ | - |

## 🌐 API Endpoints

### Core Endpoints

#### Training
```http
POST /train
```
Train the model with new sensor data. **Automatically performs back-evaluation** on past predictions.

**Body:**
```json
{
  "room": "Bedroom",
  "current_temp": 21.5,
  "target_temp": 22.0,
  "radiator_level": 4.0,
  "outdoor_temp": 2.0,
  "forecast_temp": -1.0,
  "forecast_10h_temp": -5.0,
  "timestamp": "2025-11-06T18:00:00"
}
```

**Note:** `radiator_level` accepts both integers and floats (e.g., 4, 4.0, 4.5) for precise control.

**Response:**
```json
{
  "trained": true,
  "delta": 0.234,
  "training_samples": 732,
  "model_mae": 0.156,
  "back_evaluation": {
    "validated": 3,
    "trained_on": 2
  }
}
```

#### Prediction
```http
POST /predict
```
Get AI-recommended radiator level with 8-hour stability prediction.

**Response:**
```json
{
  "recommended": 4.5,
  "error": 0.3,
  "current_level": 4.0,
  "adjustment_needed": true,
  "adjustment_reason": "Gradual adjustment (limited to ±1.5)",
  "current_deviation": 0.4,
  "method": "ml_with_hints",
  "prediction_details": [
    {
      "level": 4.5,
      "immediate_temp": 21.8,
      "temp_8h": 22.1,
      "avg_error_8h": 0.3,
      "stability_score": 0.3
    }
  ]
}
```

#### Reset and Retrain
```http
POST /reset-and-retrain
```
Delete all models and retrain from scratch with physics-based hints.

**Response:**
```json
{
  "status": "success",
  "message": "All models reset and retrained successfully",
  "rooms_trained": ["Badrum", "Sovrum", "Kontor", "Vardagsrum"],
  "training_samples": {
    "Badrum": 610,
    "Sovrum": 735
  }
}
```

#### Accelerated Historical Backtraining
```http
POST /accelerated-backtrain?days=7&learning_rate_factor=0.5
```
Refine models using historical data **without resetting**. Preserves all current knowledge.

**Parameters:**
- `days` (1-90): Days of historical data to train on
- `learning_rate_factor` (0.1-1.0): Training conservativeness
  - 0.3 = very conservative
  - 0.5 = balanced (recommended)
  - 0.8 = aggressive

**Response:**
```json
{
  "status": "success",
  "results": {
    "Kontor": {
      "samples_processed": 482,
      "avg_error_before": 0.3421,
      "avg_error_after": 0.2834,
      "improvement_percent": 17.2
    }
  }
}
```

#### Retrain Progress
```http
GET /retrain-progress
```
Get progress of reset/retrain operation.

**Response:**
```json
{
  "progress": 100.0,
  "trained_rooms": 4,
  "total_rooms": 4,
  "room_status": {
    "Kontor": {
      "training_samples": 732,
      "status": "trained"
    }
  }
}
```

### Analytics & Monitoring

#### Statistics
```http
GET /stats
```
Get comprehensive AI performance metrics including validation accuracy.

#### Validation Stats
```http
GET /validation-stats?days=7
```
View prediction accuracy over time.

**Response:**
```json
{
  "summary": {
    "total_predictions": 150,
    "total_trained": 142,
    "avg_error": 0.34
  },
  "rooms": {
    "Bedroom": {
      "total_predictions": 50,
      "avg_error": 0.28,
      "used_for_training": 48
    }
  }
}
```

#### Validate Predictions
```http
POST /validate-predictions
```
Manually trigger validation of past predictions (also runs hourly automatically).

#### Room Insights
```http
GET /analytics/room-insights
```
Comprehensive analytics with AI reasoning for all rooms.

**Response:**
```json
{
  "Bedroom": {
    "current_state": {
      "temperature": 21.5,
      "target": 22.0,
      "deviation": -0.5,
      "status": "adjusting"
    },
    "latest_prediction": {
      "recommended_level": 5.5,
      "predicted_error": 0.4,
      "confidence": 0.85
    },
    "performance": {
      "training_samples": 732,
      "mae": 0.156,
      "24h_predictions": 48
    },
    "reasoning": "❄️ Room -0.5°C too cold → increasing heating | ⚖️ Balanced mode..."
  }
}
```

#### Prediction Accuracy Trends
```http
GET /analytics/prediction-accuracy?days=7
```
Time-series accuracy data showing model improvement over time.

**Response:**
```json
{
  "Bedroom": [
    {
      "date": "2025-11-07",
      "avg_error": 0.52,
      "prediction_count": 24,
      "adjustments_made": 8
    }
  ]
}
```

#### Temperature Timeline
```http
GET /analytics/temperature-timeline?room=Bedroom&hours=48
```
Historical temperature data with predictions for charting.

**Response:**
```json
{
  "actual": [
    {
      "timestamp": "2025-11-09T10:00:00",
      "temperature": 21.5,
      "target": 22.0,
      "outdoor_temp": 5.0
    }
  ],
  "predictions": [
    {
      "timestamp": "2025-11-09T10:00:00",
      "recommended_level": 5.0,
      "predicted_error": 0.3,
      "current_temp": 21.5
    }
  ]
}
```

### Data Export

#### Historical Data
```http
GET /history/{room}?hours=24
```
Get temperature and radiator level history for graphing.

#### CSV Export
```http
GET /export/csv/{room}?hours=168
```
Download CSV file for analysis in Excel/Python.

### Web Dashboard
```http
GET /ui
```
Comprehensive visual dashboard with real-time analytics:

**Core Metrics:**
- Real-time training events with pagination
- Latest predictions per room
- Per-room performance metrics (MAE, RMSE, R²)
- Training samples and prediction counts
- Weather conditions (current, 3h, 10h forecast)

**Advanced Analytics (New):**
- 🧠 **AI Decision Insights & Reasoning** - Real-time explanations of why the AI recommends each radiator setting
- 📈 **Temperature Timeline Charts** - 48-hour actual vs predicted temperature graphs with outdoor correlation
- 🎯 **Prediction Accuracy Trends** - 7-day time-series showing model improvement over time
- Auto-refresh every 60 seconds

#### Dashboard Controls
The UI includes three action buttons:

1. **🔄 Refresh** (Blue) - Reload page to see latest data
2. **🔄 Reset & Retrain AI** (Red) - Destructive operation
   - Deletes all trained models
   - Retrains from scratch with physics hints + historical data
   - Shows progress modal with per-room status
   - Use when: Algorithm improvements, feature changes
   
3. **📚 Backtrain on History** (Green) - Non-destructive refinement
   - Refines existing models using historical data
   - Preserves all current knowledge
   - Interactive prompts for days (1-90) and learning rate (0.1-1.0)
   - Shows before/after error improvement per room
   - Use when: Seasonal changes, poor predictions, want to leverage past data

### Interactive Documentation
Visit `http://localhost:8000/docs` for full Swagger UI documentation.

## 🔄 n8n Integration

### Import Workflow

1. Open your n8n instance
2. Go to **Workflows** → **Import from File**
3. Select `n8n/Smart Radiator Training.json`
4. Update HTTP Request nodes with your AI service URL:
   - If using Docker network: `http://ai_service:8000`
   - If using host: `http://<your-server-ip>:8000`

### Network Setup

**Option A: Shared Docker Network (Recommended)**
```bash
docker network create radiator-network
```
Add to both n8n and radiator stacks' compose files:
```yaml
networks:
  radiator-network:
    external: true
```

**Option B: Host Network**
Use `http://<server-ip>:8000` in n8n HTTP Request nodes.

## 🛠️ Development

### Build Images Locally

```bash
# Build and push to Docker Hub
./build-and-push.sh

# Build specific version
./build-and-push.sh v1.0.0
```

### Manual Build

```bash
# Build AI service
docker build -t namelesshampus/radiator-ai-service:latest ./ai_service

# Build bot
docker build -t namelesshampus/radiator-bot:latest ./bot
```

### Update Deployment

```bash
# Pull latest images and restart
docker compose pull && docker compose up -d
```

## 📊 Monitoring & Logs

```bash
# View all logs
docker compose logs -f

# View specific service
docker compose logs -f ai_service
docker compose logs -f bot

# Check service status
docker compose ps

# Restart services
docker compose restart
```

## 🐛 Troubleshooting

<details>
<summary><b>Bot not responding</b></summary>

- Verify `BOT_TOKEN` is correct in `.env`
- Check bot logs: `docker compose logs bot`
- Ensure bot is started in Telegram (send `/start`)
</details>

<details>
<summary><b>AI service unreachable</b></summary>

- Verify service is running: `docker compose ps`
- Check port 8000 is not in use: `netstat -tuln | grep 8000`
- Review logs: `docker compose logs ai_service`
</details>

<details>
<summary><b>Database connection errors</b></summary>

- Verify PostgreSQL is running and accessible
- Check `DATABASE_URL` format: `postgresql://user:password@host:5432/smart_radiator_ai`
- Ensure database `smart_radiator_ai` exists
</details>

<details>
<summary><b>n8n can't reach AI service</b></summary>

- Ensure both are on the same Docker network
- Use service name `ai_service` not `localhost`
- Check firewall rules if using host networking
</details>

<details>
<summary><b>Predictions seem inaccurate</b></summary>

**Try these solutions in order:**

1. **Accelerated Backtrain** (Recommended, non-destructive)
   - Use UI button "📚 Backtrain on History"
   - Or: `curl -X POST http://localhost:8000/accelerated-backtrain?days=7&learning_rate_factor=0.5`
   - Refines model using past data without losing current knowledge

2. **Check Training Data**
   - Visit `/ui` dashboard
   - Verify training samples > 500 per room
   - Check MAE < 0.5°C

3. **Reset & Retrain** (Last resort, destructive)
   - Use UI button "🔄 Reset & Retrain AI"
   - Only if backtraining doesn't help
   - Starts completely fresh
</details>

<details>
<summary><b>Model making drastic changes</b></summary>

This should be fixed with the gradual adjustment logic:
- Maximum ±1.5 level change per cycle
- ±0.3°C tolerance band (no adjustment if close)
- Check `/predict` response for `adjustment_reason`
- Review Telegram notifications for adjustment explanations
</details>

## 📚 Documentation

- **[Database Schema](DATABASE_SCHEMA.md)** - Complete database structure
- **[API Documentation](http://localhost:8000/docs)** - Interactive Swagger UI (when running)
- **[Web Dashboard](http://localhost:8000/ui)** - Visual monitoring interface
- **[n8n Workflows](n8n/)** - Automation templates

## 🎓 AI Training Strategies

### Automatic Back-Evaluation (Always Active)
- ✅ Enabled on every `/train` call
- Validates predictions made ~3 hours ago
- Trains on prediction errors automatically
- No manual intervention needed

### When to Use Each Training Method

| Method | Type | Preserves Knowledge | Best For |
|--------|------|---------------------|----------|
| **Back-Evaluation** | Automatic | ✅ Yes | Daily operation |
| **Accelerated Backtrain** | Manual | ✅ Yes | Seasonal changes, refinement |
| **Reset & Retrain** | Manual | ❌ No | Algorithm updates, feature changes |

### Accelerated Backtraining Guide

#### Conservative (Recommended for first time)
```bash
POST /accelerated-backtrain?days=7&learning_rate_factor=0.5
```
- Last week's data
- Balanced learning
- Safe for production

#### Aggressive (After major changes)
```bash
POST /accelerated-backtrain?days=30&learning_rate_factor=0.8
```
- Last month's data
- Rapid adaptation
- Use after holidays or weather pattern changes

#### Gentle Refinement
```bash
POST /accelerated-backtrain?days=14&learning_rate_factor=0.3
```
- Two weeks of data
- Very conservative
- Minimal disruption to current behavior

### Recommended Schedule
```
Hourly:    Automatic back-evaluation (built-in)
Daily:     Monitor prediction accuracy via /ui
Weekly:    Accelerated backtrain (7 days, 0.5 rate)
Monthly:   Review metrics and adjust strategy
Seasonal:  Accelerated backtrain (30 days, 0.3 rate)
Never:     Automatic reset (manual control only)
```

## 🔄 Model Management

### Reset Database (Start Fresh)
If you want to clear all training data and start learning from scratch:

```bash
./reset_database.sh
```

This will:
- ❌ Delete all training history
- ❌ Remove all ML models
- ❌ Clear prediction logs
- ✅ Recreate fresh database schema
- ✅ Restart AI service

**Use this when:**
- Updating database schema
- Major system architecture changes
- Switching room configurations
- Testing different learning approaches

### Backup Database
```bash
docker compose exec postgres pg_dump -U postgres radiators > backup.sql
```

### Restore Database
```bash
cat backup.sql | docker compose exec -T postgres psql -U postgres radiators
```

## 🎓 Key Technical Features

This project implements advanced thermal control AI with:

### Smart Prediction System
- **8-Hour Lookahead**: Simulates temperature evolution for stability optimization
- **Physics-Based Learning**: Understands radiator dynamics (level → heat output)
- **Dual Weather Integration**: SMHI 3h and 10h forecasts for outdoor conditions
- **Per-Room Models**: Independent learning for each room's unique characteristics

### Intelligent Adjustment Logic
- **Gradual Changes**: Maximum ±1.5 level change prevents thermal shock
- **Tolerance Band**: ±0.3°C acceptable deviation reduces unnecessary adjustments
- **Minimum Threshold**: Only acts on changes ≥0.5 levels
- **Reasoning Output**: Every adjustment includes human-readable explanation

### Self-Improving Training
- **Automatic Back-Evaluation**: Validates predictions every training cycle
- **Historical Backtraining**: Refine models using past data without reset
- **Confidence Decay**: Time-based weighting (recent data > old data)
- **Adaptive Learning**: River's online ML adjusts to changing conditions

### Production-Ready Architecture
- **Database Persistence**: Models survive restarts (PostgreSQL)
- **Manual Reset Control**: Never auto-resets, only via UI/API
- **Comprehensive Metrics**: MAE, RMSE, R², training samples, validation accuracy
- **Complete Audit Trail**: All training events, predictions, and validations logged
- **Export Capabilities**: CSV export for analysis (Excel, Python, R)

## 🐛 Troubleshooting

### n8n Integration Issues

#### "Input should be a valid number" Error
**Problem:** n8n sends `radiator_level` with null values or decimal points that were previously rejected.

**Solution:** The API now accepts both integers and floats for `radiator_level` (e.g., 4, 4.0, 4.5). Ensure your n8n workflow:
1. Filters out items with null values before sending to `/train` or `/predict`
2. Uses proper data mapping from Home Assistant sensors
3. Provides default values for missing data: `{{ $json.current_temp ?? 0 }}`

**Add a Filter Node:**
```
Conditions: 
  current_temp is not empty AND 
  target_temp is not empty AND 
  radiator_level is not empty
```

#### Dashboard Analytics Not Loading
**Problem:** "Loading AI insights..." stays indefinitely.

**Solution:** This was caused by JavaScript syntax errors. The issue has been fixed in the latest version. If you still see this:
1. Hard refresh your browser (`Ctrl+F5` or `Cmd+Shift+R`)
2. Check browser console (F12) for any JavaScript errors
3. Verify CDN access to Chart.js and date-fns libraries
4. Ensure analytics endpoints return data: `curl http://localhost:8000/analytics/room-insights`

### Database Issues

#### Connection Errors
**Problem:** `connection refused` or `authentication failed`

**Solution:**
```bash
# Check if PostgreSQL is running
docker compose ps

# Restart services
docker compose restart

# Check logs
docker compose logs postgres
```

#### Reset After Schema Changes
If database schema is updated:
```bash
./reset_database.sh  # WARNING: Deletes all data
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[River](https://riverml.xyz/)** - Online machine learning framework
- **[FastAPI](https://fastapi.tiangolo.com/)** - Modern web framework
- **[aiogram](https://aiogram.dev/)** - Telegram bot framework
- **[n8n](https://n8n.io/)** - Workflow automation

---

<div align="center">

[⬆ Back to Top](#-smart-radiator-assistant)

</div>
