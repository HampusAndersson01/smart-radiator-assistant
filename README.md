# Smart Radiator Assistant — Local Portainer Setup

This repository provides a local, modular Smart Radiator Assistant consisting of:

- `ai_service` — FastAPI microservice that trains/predicts radiator level using River (online ML) **with comprehensive AI statistics and performance tracking**.
- `bot` — Telegram bot (aiogram) for manual radiator setting (`/set`).
- `n8n` — n8n workflow to collect sensor data and call the AI service.
- `docker-compose.yml` — stack suitable for deploying in Portainer (paste into a Stack).

## 🎓 Assignment Documentation

**NEW:** This project includes comprehensive AI statistics and metrics for academic evaluation:

- **📊 `/stats` endpoint** - Detailed AI performance metrics (MAE, RMSE, R², training samples, predictions, etc.)
- **📈 Performance tracking** - Continuous monitoring of model accuracy and efficiency
- **💾 Database logging** - PostgreSQL logging of all AI decisions and training events
- **📝 Full documentation** - See `AI_ASSIGNMENT_DOCUMENTATION.md` for complete writeup
- **🧪 Testing suite** - Run `python3 test_ai_stats.py` to demonstrate AI capabilities

**Quick start for testing:** See `QUICK_REFERENCE.md`

## Quick files added

- `ai_service/` — FastAPI app, forecast util, Dockerfile, requirements.
- `bot/` — Telegram bot, config, Dockerfile, requirements.
- `n8n/radiator_flow.json` — n8n flow you can import.
- `.env.example` — example env file for credentials.

## Goals

1. Run everything locally on your Ubuntu server.
2. Deploy via Portainer Stacks (docker-compose paste).
3. Keep data local (models saved to `ai_service/models`).

## Steps to deploy with Portainer (recommended)

1. Copy `.env.example` to `.env` and fill in your `BOT_TOKEN` (and optionally `TELEGRAM_WEBHOOK`).

2. Deploy with Portainer (Stack):

   - Open Portainer > Stacks > Add stack.
   - Give it a name (e.g., `smart-radiator-assistant`).
   - Paste the contents of `docker-compose.yml` into the Web editor.
   - Under "Environment variables" provide `BOT_TOKEN` and `TELEGRAM_WEBHOOK` or add an `.env` file accessible to Portainer.
   - Deploy the stack. Portainer will build the stack from the provided `build:` contexts.

3. **Connect to your existing n8n instance:**

   Since n8n is already running in a separate Portainer stack, you'll need to ensure network connectivity:
   
   - **Option A (Recommended)**: Add both stacks to the same Docker network:
     - Create a shared network: `docker network create radiator-network`
     - In Portainer, edit both stacks to use this network (add `networks:` section)
     - In n8n flows, use `http://ai_service:8000` to reach the AI service
   
   - **Option B**: Use host networking or publish the AI service port (already exposed on 8000) and access via `http://<server-ip>:8000` from n8n workflows.

Notes on Portainer building: Portainer will attempt to build images if it has access to the stack files; if it cannot build from local Dockerfiles, either build locally and push to a registry (then change compose to use `image: <repo/name:tag>`) or use Portainer's Git integration to point to a repo URL.

## Import the n8n flow (into your existing n8n instance)

1. Open your existing n8n instance (e.g., `http://<server>:5678`).
2. In n8n: Click the 3-dot menu > Import > paste the JSON from `n8n/radiator_flow.json`.
3. **Important**: Update the HTTP Request node URLs to match how n8n will reach the AI service:
   - If using shared Docker network: `http://ai_service:8000/train` and `http://ai_service:8000/predict`
   - If using host networking: `http://<server-ip>:8000/train` and `http://<server-ip>:8000/predict`
4. Edit the HTTP Request nodes: set the Home Assistant URL (if different), map sensor payloads to the AI `/train` and `/predict` inputs.

## File/Deployment notes

- Models and state are stored in `ai_service/models` and `ai_service/last_temps.json`. These are mounted as volumes in `docker-compose.yml` so data persists.
- The bot uses long polling (aiogram) and requires `BOT_TOKEN` to be set. Webhook support can be added later.
- `TELEGRAM_WEBHOOK` in `.env` is used by `ai_service` to send simple alert messages (optional).

### GPU Support

The `docker-compose.yml` now includes GPU runtime configuration for the AI service. **Important notes:**

1. **River (current ML library) doesn't use GPU** — River is designed for online/incremental learning and runs efficiently on CPU. The current model (ARFRegressor) won't benefit from GPU acceleration.

2. **GPU prerequisites** (if you want GPU-ready for future models):
   - NVIDIA GPU on the host
   - NVIDIA Container Toolkit installed: [Install Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
   - Test with: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`

3. **Remove GPU config if not needed**: If you don't have a GPU or don't plan to use GPU-accelerated models, remove the `deploy.resources` section from `ai_service` in `docker-compose.yml` to avoid deployment errors.

4. **For GPU-accelerated ML** (future enhancement): Replace River with PyTorch/TensorFlow and update requirements. For this radiator use case, CPU-based River is perfectly adequate and more efficient.

## Quick validation (after deploy)

- AI service health: `curl http://<server-ip>:8000/docs` to open FastAPI docs.
- Predict endpoint example:

  ```bash
  curl -X POST http://<server-ip>:8000/predict -H "Content-Type: application/json" \
    -d '{"room":"Bathroom","current_temp":20.1,"target_temp":22.5,"radiator_level":3,"timestamp":"now"}'
  ```

- Bot: Start the bot's Telegram chat and run `/set`.

## Troubleshooting tips

- If Portainer fails to build: run `docker compose build` on the server manually to see build logs.
- If ai_service can't reach Open-Meteo, check outgoing network/firewall.
- If bot doesn't start, verify `BOT_TOKEN` is correct and container logs.

## Next steps / enhancements

- Wire real sensor values from Home Assistant in the n8n flow and map them to the AI endpoints.
- Add authentication to the AI endpoints (e.g., simple token) for security.
- Convert the bot to use webhooks (requires HTTPS + public URL or a reverse proxy).

---

If you want, I can now:

1. Build the docker-compose locally and run a quick smoke test (if Docker is available here).
2. Create a small example n8n node mapping for Home Assistant sensor payloads.

Tell me which one to do next.
