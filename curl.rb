curl -X POST http://localhost:5000/api/v1/train/device \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "sensor.ihp_salle_anticipated_start_time_hms",
    "indoor_temp_entity_id": "sensor.capteur_tdeg_hdeg_salle_temperature",
    "outdoor_temp_entity_id": "sensor.openweathermap_temperature",
    "target_temp_entity_id": "climate.thermostat_salle",
    "heating_state_entity_id": "climate.thermostat_salle",
    "humidity_entity_id": "sensor.capteur_tdeg_hdeg_salle_humidity",
    "history_days": 60,
    "cycle_split_duration_minutes": 50
  }'


curl -X POST http://localhost:5000/api/v1/train/device \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "sensor.ihp_ambre_anticipated_start_time_hms",
    "indoor_temp_entity_id": "sensor.capteur_tdeg_hdeg_ambre_temperature",
    "outdoor_temp_entity_id": "sensor.openweathermap_temperature",
    "target_temp_entity_id": "climate.thermostat_ambre",
    "heating_state_entity_id": "climate.thermostat_ambre",
    "humidity_entity_id": "sensor.capteur_tdeg_hdeg_ambre_humidity",
    "history_days": 60,
    "cycle_split_duration_minutes": 15
  }'


curl -X POST http://localhost:5000/api/v1/train/device \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "sensor.ihp_nora_anticipated_start_time_hms",
    "indoor_temp_entity_id": "sensor.capteur_tdeg_hdeg_nora_temperature",
    "outdoor_temp_entity_id": "sensor.openweathermap_temperature",
    "target_temp_entity_id": "climate.thermostat_nora",
    "heating_state_entity_id": "climate.thermostat_nora",
    "humidity_entity_id": "sensor.capteur_tdeg_hdeg_nora_humidity",
    "history_days": 60
  }'

curl -X POST http://localhost:5000/api/v1/train/device \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "sensor.ihp_maxence_anticipated_start_time_hms",
    "indoor_temp_entity_id": "sensor.capteur_tdeg_hdeg_maxence_temperature",
    "outdoor_temp_entity_id": "sensor.openweathermap_temperature",
    "target_temp_entity_id": "climate.thermostat_maxence",
    "heating_state_entity_id": "climate.thermostat_maxence",
    "humidity_entity_id": "sensor.capteur_tdeg_hdeg_maxence_humidity",
    "history_days": 60,
    "cycle_split_duration_minutes": 15
  }'

curl -X POST http://localhost:5000/api/v1/train/device \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "sensor.ihp_mezzanine_anticipated_start_time_hms",
    "indoor_temp_entity_id": "sensor.capteur_tdeg_hdeg_mezzanine_temperature",
    "outdoor_temp_entity_id": "sensor.openweathermap_temperature",
    "target_temp_entity_id": "climate.thermostat_mezzanine",
    "heating_state_entity_id": "climate.thermostat_mezzanine",
    "humidity_entity_id": "sensor.capteur_tdeg_hdeg_mezzanine_humidity",
    "history_days": 60,
    "cycle_split_duration_minutes": 15
  }'

curl -X POST http://localhost:5000/api/v1/train/device \
  -H "Content-Type: application/json" \
  -d '{
    "device_id": "sensor.ihp_chambre_anticipated_start_time_hms",
    "indoor_temp_entity_id": "sensor.capteur_tdeg_hdeg_chambre_temperature",
    "outdoor_temp_entity_id": "sensor.openweathermap_temperature",
    "target_temp_entity_id": "climate.thermostat_chambre",
    "heating_state_entity_id": "climate.thermostat_chambre",
    "humidity_entity_id": "sensor.capteur_tdeg_hdeg_chambre_humidity",
    "history_days": 60,
    "cycle_split_duration_minutes": 15
  }'  