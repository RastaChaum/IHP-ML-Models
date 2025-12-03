# Tests d'Intégration API - IHP ML Models

Ce répertoire contient les tests d'intégration pour l'API Flask du service ML.

## Vue d'ensemble

Les tests d'intégration valident le bon fonctionnement de tous les endpoints API en simulant les réponses de Home Assistant. Contrairement aux tests unitaires qui testent des composants isolés, ces tests vérifient le cycle complet requête/réponse.

## Structure

```
tests/integration/
├── __init__.py
├── conftest.py              # Fixtures pour les tests d'intégration
├── test_api_endpoints.py    # Tests de tous les endpoints
└── README.md                # Ce fichier
```

## Fixtures Disponibles

### Données de Test

- **`mock_ha_history_data`** : Données historiques simulées de Home Assistant
- **`sample_training_data`** : Données d'entraînement pour les tests
- **`sample_device_config`** : Configuration d'appareil pour les tests
- **`sample_prediction_request`** : Requête de prédiction exemple

### Services Mock

- **`mock_ha_client`** : Client HTTP simulé pour Home Assistant
- **`mock_ha_history_reader`** : Lecteur d'historique HA avec réponses mockées
- **`ml_service`** : Service ML avec toutes les dépendances mockées
- **`flask_app`** : Application Flask configurée pour les tests
- **`client`** : Client de test Flask

## Exécution des Tests

### Tous les tests d'intégration

```bash
# Depuis la racine du projet
pytest tests/integration/ -v

# Avec couverture
pytest tests/integration/ --cov=ihp_ml_addon/rootfs/app --cov-report=html
```

### Tests spécifiques

```bash
# Tester uniquement l'endpoint de santé
pytest tests/integration/test_api_endpoints.py::TestHealthEndpoint -v

# Tester l'entraînement
pytest tests/integration/test_api_endpoints.py::TestTrainEndpoint -v

# Tester les prédictions
pytest tests/integration/test_api_endpoints.py::TestPredictEndpoint -v

# Tester la gestion des modèles
pytest tests/integration/test_api_endpoints.py::TestModelsEndpoints -v
```

### Avec mode verbeux et détails

```bash
pytest tests/integration/ -vv -s
```

## Endpoints Testés

### Health & Status

- ✅ `GET /health` - Vérification de santé
- ✅ `GET /api/v1/status` - Statut du service

### Entraînement

- ✅ `POST /api/v1/train` - Entraînement avec données fournies
- ✅ `POST /api/v1/train/fake` - Entraînement avec données synthétiques
- ✅ `POST /api/v1/train/device` - Entraînement depuis l'historique HA

### Prédiction

- ✅ `POST /api/v1/predict` - Prédiction de durée de chauffe

### Gestion des Modèles

- ✅ `GET /api/v1/models` - Liste tous les modèles
- ✅ `GET /api/v1/models/device/<device_id>` - Modèles par appareil
- ✅ `GET /api/v1/models/<model_id>` - Info d'un modèle spécifique
- ✅ `DELETE /api/v1/models/<model_id>` - Suppression d'un modèle

## Scénarios Testés

### Tests de Succès

- Entraînement avec données valides
- Entraînement avec `minutes_since_last_cycle` (optionnel)
- Prédiction après entraînement
- Prédiction avec `minutes_since_last_cycle` (optionnel)
- Prédiction sans `minutes_since_last_cycle` (champ optionnel)
- Gestion de multiples modèles
- Modèles spécifiques par appareil

### Tests d'Erreur

- Données manquantes ou invalides
- Prédiction sans modèle entraîné
- Modèle inexistant
- JSON invalide
- Mauvaise méthode HTTP

### Tests Edge Cases

- Entraînement avec peu de données
- Multiples modèles pour différents appareils
- Paramètres optionnels
- Valeurs limites (min/max)

## Simulation de Home Assistant

Les tests utilisent des mocks pour simuler les réponses de Home Assistant :

```python
# Exemple de données historiques simulées
mock_ha_history_data = [
    {
        "entity_id": "sensor.indoor_temp",
        "state": "18.5",
        "last_changed": "2024-12-03T08:00:00",
    },
    # ... plus de données
]
```

Le `HomeAssistantHistoryReader` est mocké pour retourner ces données sans réellement contacter Home Assistant.

## Ajout de Nouveaux Tests

Pour ajouter de nouveaux tests :

1. Créer une nouvelle classe de test dans `test_api_endpoints.py`
2. Utiliser les fixtures disponibles dans `conftest.py`
3. Suivre le pattern AAA (Arrange, Act, Assert)

Exemple :

```python
class TestNewEndpoint:
    """Tests for /api/v1/new-endpoint."""
    
    def test_successful_request(self, client: Any) -> None:
        """Test should describe what it validates."""
        # ARRANGE: Prepare test data
        request_data = {"key": "value"}
        
        # ACT: Make the request
        response = client.post(
            "/api/v1/new-endpoint",
            json=request_data,
            content_type="application/json"
        )
        
        # ASSERT: Verify the response
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data["success"] is True
```

## Dépendances Requises

Les tests nécessitent :

```
pytest>=7.0.0
flask>=2.0.0
xgboost>=1.7.0
numpy>=1.24.0
scikit-learn>=1.3.0
```

Ces dépendances sont déjà définies dans `pyproject.toml`.

## Résolution de Problèmes

### Tests échouent avec "Module not found"

Assurez-vous que le `PYTHONPATH` inclut le répertoire de l'application :

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/ihp_ml_addon/rootfs/app"
pytest tests/integration/
```

### Tests lents

Les tests d'intégration peuvent être plus lents que les tests unitaires car ils testent le cycle complet. Pour exécuter uniquement les tests rapides :

```bash
pytest tests/integration/ -m "not slow"
```

### Erreurs de fixture

Si vous obtenez des erreurs de fixture, vérifiez que `conftest.py` est bien présent et que les imports sont corrects.

## Intégration Continue

Ces tests sont conçus pour être exécutés dans un pipeline CI/CD. Ils ne nécessitent pas de Home Assistant réel ni de connexion réseau.

```yaml
# Exemple GitHub Actions
- name: Run integration tests
  run: |
    pytest tests/integration/ -v --cov
```

## Métriques de Couverture

Objectif de couverture : **>80%** pour les endpoints API.

Pour vérifier la couverture :

```bash
pytest tests/integration/ --cov=ihp_ml_addon/rootfs/app/infrastructure/api --cov-report=term-missing
```
