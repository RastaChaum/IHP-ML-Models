#!/bin/bash
# Script de test de l'entraînement avec différentes périodes d'historique

API_URL="http://localhost:5000"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Test d'entraînement avec pagination"
echo "=========================================="
echo ""

# Demander le nombre de jours
if [ -z "$1" ]; then
    echo -e "${YELLOW}Usage: $0 <nombre_de_jours>${NC}"
    echo "Exemple: $0 30"
    echo ""
    echo "Utilisation de 30 jours par défaut..."
    DAYS=30
else
    DAYS=$1
fi

echo -e "${BLUE}📅 Période d'historique: $DAYS jours${NC}"
echo ""

# Configuration du device
CONFIG=$(cat <<EOF
{
  "device_id": "thermostat_salle",
  "indoor_temp_entity_id": "sensor.capteur_tdeg_hdeg_salle_temperature",
  "outdoor_temp_entity_id": "sensor.openweathermap_temperature",
  "target_temp_entity_id": "climate.thermostat_salle",
  "heating_state_entity_id": "climate.thermostat_salle",
  "humidity_entity_id": "sensor.capteur_tdeg_hdeg_salle_humidity",
  "history_days": $DAYS
}
EOF
)

echo "Configuration:"
echo "$CONFIG" | jq
echo ""

echo "⏳ Lancement de l'entraînement (peut prendre du temps avec beaucoup de données)..."
echo ""

START_TIME=$(date +%s)

RESPONSE=$(curl -s -X POST "${API_URL}/api/v1/train/device" \
  -H "Content-Type: application/json" \
  -d "$CONFIG")

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Vérifier le résultat
SUCCESS=$(echo "$RESPONSE" | jq -r '.success // false')

if [ "$SUCCESS" = "true" ]; then
    echo -e "${GREEN}✅ Entraînement réussi en ${DURATION}s !${NC}"
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
    
    # Afficher le nombre de samples par rapport aux jours
    SAMPLES=$(echo "$RESPONSE" | jq -r '.training_samples')
    SAMPLES_PER_DAY=$(echo "scale=2; $SAMPLES / $DAYS" | bc)
    echo ""
    echo -e "${BLUE}📈 Statistiques:${NC}"
    echo "  • $SAMPLES cycles détectés sur $DAYS jours"
    echo "  • ~$SAMPLES_PER_DAY cycles par jour"
else
    echo -e "${RED}❌ Entraînement échoué${NC}"
    echo ""
    echo "Erreur:"
    echo "$RESPONSE" | jq
fi

echo ""
echo "=========================================="
