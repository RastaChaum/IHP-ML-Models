# Quick Start - Local Development

Get started testing the IHP ML Models addon in under 5 minutes.

## Prerequisites

✅ Docker installed  
✅ Docker Compose installed  
✅ Git repository cloned

## Steps

### 1️⃣ Start the environment

```bash
./scripts/develop.sh
```

This command will:
- Build the addon Docker image
- Start Home Assistant test instance
- Create sample configuration
- Display service URLs

### 2️⃣ Wait for services to start

Give it 30-60 seconds for Home Assistant to initialize.

### 3️⃣ Test the addon

```bash
./scripts/test-addon.sh
```

You should see:
```
Testing Health Check... ✓ PASSED
Testing Status... ✓ PASSED
Testing Train with Fake Data... ✓ PASSED
Testing Predict... ✓ PASSED
Testing List Models... ✓ PASSED

All tests passed!
```

### 4️⃣ Access the services

- **Home Assistant**: http://localhost:8123
- **Addon API**: http://localhost:5000

### 5️⃣ View logs

```bash
# All logs
docker compose logs -f

# Just addon
docker compose logs -f ihp-ml-addon
```

### 6️⃣ Make changes and test

1. Edit code in `ihp_ml_addon/rootfs/app/`
2. Restart addon: `docker compose restart ihp-ml-addon`
3. Test: `./scripts/test-addon.sh`

### 7️⃣ Clean up when done

```bash
./scripts/clean.sh
```

## Common Commands

```bash
# Restart addon after code changes
docker compose restart ihp-ml-addon

# Rebuild after Dockerfile/requirements changes
docker compose build --no-cache ihp-ml-addon && docker compose up -d

# View only errors
docker compose logs ihp-ml-addon | grep ERROR

# Stop everything
docker compose down
```

## Manual API Testing

```bash
# Health check
curl http://localhost:5000/health

# Get status
curl http://localhost:5000/api/v1/status | jq

# Train with fake data
curl -X POST http://localhost:5000/api/v1/train/fake \
  -H "Content-Type: application/json" \
  -d '{"num_samples": 100}' | jq

# Make prediction
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "outdoor_temp": 5.0,
    "indoor_temp": 18.0,
    "target_temp": 21.0,
    "humidity": 65.0,
    "hour_of_day": 7,
    "day_of_week": 1
  }' | jq
```

## Troubleshooting

**Port 8123 already in use?**
```bash
# Edit docker compose.yml, change port to 7123:8123
# Access HA at http://localhost:7123
```

**Addon not responding?**
```bash
docker compose ps  # Check if running
docker compose logs ihp-ml-addon  # Check logs
```

**Changes not reflected?**
```bash
docker compose restart ihp-ml-addon
```

## Next Steps

📖 Read [DEVELOPMENT.md](./DEVELOPMENT.md) for detailed documentation  
🧪 Run unit tests: `poetry run pytest -v`  
🏗️ Learn architecture: [ARCHITECTURE.md](./ARCHITECTURE.md)

## Need Help?

- Check logs: `docker compose logs -f ihp-ml-addon`
- See [DEVELOPMENT.md](./DEVELOPMENT.md#-common-issues)
- Open an issue on GitHub
