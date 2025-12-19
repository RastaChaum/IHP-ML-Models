# Development Guide - IHP ML Models Add-on

This guide explains how to set up a local development environment for testing the IHP ML Models add-on.

## 📋 Prerequisites

- Docker and Docker Compose installed
- Git
- Python 3.12+ with Poetry (for local development without Docker)
- Basic understanding of Home Assistant add-ons

## 🚀 Quick Start

### Option 1: Docker (Recommended for testing)

```bash
# Make scripts executable
chmod +x scripts/*.sh

# Start development environment
./scripts/develop.sh
```

This will:
- Create a test Home Assistant instance at `http://localhost:8123`
- Start the IHP ML addon at `http://localhost:5000`
- Create sample configuration files
- Set up test data directories

### Option 2: Local Python (Recommended for debugging)

```bash
# 1. Install dependencies
poetry install

# 2. Configure environment variables
cp .env.example .env
# Edit .env with your local settings

# 3. Run the server directly
poetry run python ihp_ml_addon/rootfs/app/infrastructure/api/server.py
```

**Avantages du mode local :**
- Debugging avec VSCode (breakpoints, step-by-step)
- Rechargement automatique du code
- Pas besoin de rebuild Docker à chaque modification
- Les variables d'environnement sont chargées automatiquement depuis `.env`

### 2. Test the addon

```bash
# Run automated tests
./scripts/test-addon.sh
```

### 3. Access services

- **Home Assistant**: http://localhost:8123
- **IHP ML Addon API**: http://localhost:5000

## ⚙️ Configuration des variables d'environnement

Le projet utilise `python-dotenv` pour charger automatiquement les variables depuis un fichier `.env`.

### Fichier `.env` (développement local)

```bash
# Configuration locale pour VSCode
LOG_LEVEL=DEBUG
LOG_DIR=./test-data/logs
MODEL_PERSISTENCE_PATH=./test-data/models
SUPERVISOR_URL=http://192.168.1.100:8123
SUPERVISOR_TOKEN=ton_token_longue_duree
```

### Docker Compose

Le fichier `.env` est aussi utilisé par docker-compose. Pas besoin de dupliquer la configuration !

**Priorité des variables :**
1. Variables système (export dans shell)
2. Variables docker-compose
3. Variables `.env`

📖 Voir [docs/DOTENV.md](docs/DOTENV.md) pour plus de détails.

## 📁 Project Structure

```
IHP-ML-Models/
├── ihp_ml_addon/           # Add-on source code
│   ├── config.yaml         # Add-on configuration
│   ├── Dockerfile          # Container definition
│   ├── requirements.txt    # Python dependencies
│   └── rootfs/
│       └── app/            # Application code
├── test-config/            # Home Assistant test configuration (auto-generated)
├── test-data/              # Test data and models (auto-generated)
├── .env                    # Variables d'environnement (local, non versionné)
├── .env.example            # Template de configuration
├── docker compose.yml      # Local development setup
├── pyproject.toml          # Poetry dependencies
└── scripts/
    ├── develop.sh          # Start development environment
    ├── test-addon.sh       # Run API tests
    └── clean.sh            # Clean up test environment
```

## 🔧 Development Workflow

### Starting Development (Docker)

```bash
./scripts/develop.sh
```

### Starting Development (Local Python + VSCode)

1. **Configurer `.env`** :
   ```bash
   cp .env.example .env
   # Éditer .env avec tes valeurs
   ```

2. **Installer les dépendances** :
   ```bash
   poetry install
   ```

3. **Lancer avec le debugger VSCode** :
   - Ouvrir `server.py` dans VSCode
   - Appuyer sur F5 ou "Run > Start Debugging"
   - Le fichier `.env` sera chargé automatiquement

### Viewing Logs

#### Docker

```bash
# All logs
docker compose logs -f

# Just addon logs
docker compose logs -f ihp-ml-addon

# Just Home Assistant logs
docker compose logs -f homeassistant
```

#### Local Python

Les logs sont écrits dans les fichiers configurés via `LOG_DIR` (défaut: `./test-data/logs/`) :

```bash
# Logs principaux (niveau LOG_LEVEL)
tail -f ./test-data/logs/ihp_ml.log

# Logs debug complets
tail -f ./test-data/logs/ihp_ml_debug.log

# Analyser les logs
./scripts/analyze-logs.sh
```

📖 Voir [docs/LOGGING.md](docs/LOGGING.md) pour plus de détails sur la configuration des logs.

### Making Code Changes

The addon code is mounted as a volume, so changes to Python files in `ihp_ml_addon/rootfs/app/` require a restart:

```bash
docker compose restart ihp-ml-addon
```

For Dockerfile or requirements changes, rebuild:

```bash
docker compose build --no-cache ihp-ml-addon
docker compose up -d
```

### Testing with Real Home Assistant Data

1. Access Home Assistant at http://localhost:8123
2. Complete the onboarding process
3. Go to your profile → "Long-Lived Access Tokens" → "Create Token"
4. Copy the token and update `.env`:
   ```bash
   SUPERVISOR_TOKEN=your_long_lived_token_here
   ```
5. Restart the addon:
   ```bash
   docker compose restart ihp-ml-addon
   ```

Now you can test the `/api/v1/train/device` endpoint with real sensor data!

## 🧪 Manual API Testing

### Health Check
```bash
curl http://localhost:5000/health
```

### Get Status
```bash
curl http://localhost:5000/api/v1/status | jq
```

### Train with Fake Data
```bash
curl -X POST http://localhost:5000/api/v1/train/fake \
  -H "Content-Type: application/json" \
  -d '{"num_samples": 100}' | jq
```

### Make a Prediction
```bash
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "outdoor_temp": 5.0,
    "indoor_temp": 18.0,
    "target_temp": 21.0,
    "humidity": 65.0,
    "hour_of_day": 7,
    "day_of_day": 1
  }' | jq
```

### Train with Device Config (requires HA token)
```bash
curl -X POST http://localhost:5000/api/v1/train/device \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "test_device",
    "indoor_temp_entity_id": "sensor.test_indoor_temperature",
    "outdoor_temp_entity_id": "sensor.test_outdoor_temperature",
    "target_temp_entity_id": "climate.demo_thermostat",
    "heating_state_entity_id": "climate.demo_thermostat",
    "humidity_entity_id": "sensor.test_humidity",
    "history_days": 7
  }' | jq
```

### List Models
```bash
curl http://localhost:5000/api/v1/models | jq
```

## 🐛 Debugging

### Check if containers are running
```bash
docker compose ps
```

### Inspect addon container
```bash
docker exec -it ihp-ml-addon-dev /bin/bash
```

### Check environment variables
```bash
docker exec ihp-ml-addon-dev env | grep -E '(SUPERVISOR|LOG_LEVEL|MODEL)'
```

### View real-time logs with filtering
```bash
docker compose logs -f ihp-ml-addon | grep -E "(ERROR|WARNING|is_available)"
```

## 🧹 Cleanup

Remove all test data and containers:

```bash
./scripts/clean.sh
```

This removes:
- Docker containers and volumes
- `test-config/` directory
- `test-data/` directory
- `.env` file

## 🔄 Typical Development Cycle

1. **Make code changes** in `ihp_ml_addon/rootfs/app/`
2. **Restart addon**: `docker compose restart ihp-ml-addon`
3. **View logs**: `docker compose logs -f ihp-ml-addon`
4. **Test API**: `./scripts/test-addon.sh` or manual `curl` commands
5. **Run unit tests**: `poetry run pytest -v`
6. **Commit changes** when tests pass

## 📝 Testing Checklist

Before pushing changes:

- [ ] Unit tests pass: `poetry run pytest`
- [ ] Addon builds successfully: `docker compose build ihp-ml-addon`
- [ ] All API endpoints respond: `./scripts/test-addon.sh`
- [ ] Check logs for errors: `docker compose logs ihp-ml-addon | grep ERROR`
- [ ] Test with fake data training works
- [ ] Test predictions return valid results
- [ ] Code follows DDD architecture (domain/application/infrastructure separation)

## 🆘 Common Issues

### Port 8123 already in use
If you have Home Assistant running elsewhere:
```bash
# Edit docker compose.yml and change:
# ports: "8123:8123" → "7123:8123"
# Then access HA at http://localhost:7123
```

### Addon can't connect to Home Assistant
1. Check Home Assistant is running: `docker compose ps homeassistant`
2. Verify SUPERVISOR_TOKEN in `.env`
3. Check network connectivity: `docker exec ihp-ml-addon-dev curl http://homeassistant:8123/api/`

### Changes not reflected
Make sure you restart after changes:
```bash
docker compose restart ihp-ml-addon
```

Or for Dockerfile/requirements changes:
```bash
docker compose build --no-cache ihp-ml-addon && docker compose up -d
```

## 📚 Additional Resources

- [Home Assistant Add-on Development](https://developers.home-assistant.io/docs/add-ons)
- [Home Assistant REST API](https://developers.home-assistant.io/docs/api/rest/)
- [Project Architecture](./ARCHITECTURE.md)
- [Contributing Guide](./CONTRIBUTING.md)
