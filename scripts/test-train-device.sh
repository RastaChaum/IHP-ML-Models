#!/bin/bash
# Script de test pour l'endpoint /api/v1/train/device

set -e

# Configuration
API_URL="http://localhost:5000"
DEVICE_CONFIG_FILE="${1:-test_device_config.json}"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Test de l'entraînement avec données HA"
echo "=========================================="
echo ""

# 1. Vérifier le statut
echo "1️⃣  Vérification du statut de l'addon..."
STATUS=$(curl -s "${API_URL}/api/v1/status")
HA_AVAILABLE=$(echo "$STATUS" | jq -r '.ha_integration_available')

if [ "$HA_AVAILABLE" = "true" ]; then
    echo -e "${GREEN}✅ Home Assistant intégration disponible${NC}"
else
    echo -e "${RED}❌ Home Assistant intégration non disponible${NC}"
    echo "Status complet:"
    echo "$STATUS" | jq
    exit 1
fi
echo ""

# 2. Configuration du device
echo "2️⃣  Configuration du device à tester..."
if [ ! -f "$DEVICE_CONFIG_FILE" ]; then
    echo -e "${YELLOW}⚠️  Fichier de config non trouvé: $DEVICE_CONFIG_FILE${NC}"
    echo "Création d'un exemple de configuration..."
    cat > test_device_config.json <<EOF
{
  "device_id": "thermostat_salle",
  "indoor_temp_entity_id": "climate.thermostat_salle",
  "outdoor_temp_entity_id": "climate.thermostat_salle",
  "target_temp_entity_id": "climate.thermostat_salle",
  "heating_state_entity_id": "climate.thermostat_salle",
  "humidity_entity_id": null,
  "history_days": 7
}
EOF
    DEVICE_CONFIG_FILE="test_device_config.json"
    echo -e "${GREEN}✅ Fichier créé: $DEVICE_CONFIG_FILE${NC}"
fi

echo "Configuration à utiliser:"
cat "$DEVICE_CONFIG_FILE" | jq
echo ""

# 3. Lancer l'entraînement
echo "3️⃣  Lancement de l'entraînement..."
echo "⏳ Ceci peut prendre quelques secondes..."
echo ""

RESPONSE=$(curl -s -X POST "${API_URL}/api/v1/train/device" \
  -H "Content-Type: application/json" \
  -d @"${DEVICE_CONFIG_FILE}")

# Vérifier le résultat
SUCCESS=$(echo "$RESPONSE" | jq -r '.success // false')

if [ "$SUCCESS" = "true" ]; then
    echo -e "${GREEN}✅ Entraînement réussi !${NC}"
    echo ""
    echo "📊 Résultats:"
    echo "$RESPONSE" | jq '{
      device_id,
      model_id,
      training_samples,
      metrics: {
        r2,
        rmse,
        training_samples: .metrics.training_samples,
        validation_samples: .metrics.validation_samples
      }
    }'
else
    echo -e "${RED}❌ Entraînement échoué${NC}"
    echo ""
    echo "Erreur:"
    echo "$RESPONSE" | jq
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Test terminé avec succès"
echo "=========================================="
