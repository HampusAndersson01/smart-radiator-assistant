<div align="center">

# 🏠 Smart Radiator Assistant

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
│   Home Assistant │      │      n8n         │      │   Telegram Bot  │
│   (Sensors)      │─────▶│   (Automation)   │◀────▶│  (Manual Ctrl)  │
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
- Adapts to your heating preferences in real-time
- Weather-aware predictions via Open-Meteo API
- Comprehensive performance metrics

</td>
<td width="50%">

### 🔧 Easy Integration
- **Docker-based deployment** - up and running in minutes
- **Telegram bot** for manual control
- **n8n workflows** for Home Assistant integration
- **RESTful API** with interactive docs

</td>
</tr>
<tr>
<td width="50%">

### 📊 Monitoring
- Real-time AI statistics (MAE, RMSE, R²)
- Training history and prediction logs
- Database-backed persistence
- FastAPI interactive documentation

</td>
<td width="50%">

### 🚀 Production Ready
- Pre-built Docker images on Docker Hub
- Automated build & deployment scripts
- PostgreSQL for data persistence
- Scalable microservice architecture

</td>
</tr>
</table>

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

### Training
```http
POST /train
```
Train the model with new sensor data.

**Body:**
```json
{
  "room": "Bedroom",
  "current_temp": 21.5,
  "target_temp": 22.0,
  "radiator_level": 4,
  "timestamp": "2025-11-06T18:00:00"
}
```

### Prediction
```http
POST /predict
```
Get AI-recommended radiator level.

**Body:** Same as `/train`

**Response:**
```json
{
  "predicted_level": 3.8,
  "room": "Bedroom",
  "outdoor_temp": 5.2,
  "timestamp": "2025-11-06T18:00:00"
}
```

### Statistics
```http
GET /stats
```
Get comprehensive AI performance metrics.

**Response:**
```json
{
  "total_predictions": 1250,
  "total_training_samples": 850,
  "mae": 0.42,
  "rmse": 0.58,
  "r2_score": 0.87
}
```

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
- Check `DATABASE_URL` format: `postgresql://user:password@host:5432/radiator`
- Ensure database `radiator` exists
</details>

<details>
<summary><b>n8n can't reach AI service</b></summary>

- Ensure both are on the same Docker network
- Use service name `ai_service` not `localhost`
- Check firewall rules if using host networking
</details>

## 📚 Documentation

- **[Database Schema](DATABASE_SCHEMA.md)** - Complete database structure
- **[API Documentation](http://localhost:8000/docs)** - Interactive Swagger UI (when running)
- **[n8n Workflows](n8n/)** - Automation templates

## 🎓 Academic Features

This project includes comprehensive metrics for academic evaluation:

- **Performance Metrics**: MAE, RMSE, R² scoring
- **Training Analytics**: Sample counts, model convergence tracking
- **Prediction Logging**: Full audit trail of AI decisions
- **Database Persistence**: PostgreSQL-backed model state

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

**Made with ❤️ for smart home automation**

[⬆ Back to Top](#-smart-radiator-assistant)

</div>
