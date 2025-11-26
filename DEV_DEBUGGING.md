# Guide de Développement et Debugging

## 🚀 Configuration de l'environnement de développement

### 1. Volumes montés

Le docker-compose monte maintenant les volumes suivants en mode **éditable** :
```yaml
volumes:
  - ./ihp_ml_addon/rootfs/app:/app          # Code source (éditable en direct)
  - ./test-data:/data                        # Données de test
  - ./tests:/tests                           # Tests unitaires
```

**Avantage** : Vous pouvez modifier le code localement et les changements sont immédiatement visibles dans le conteneur (pas besoin de rebuild pour les changements Python).

### 2. Debugging avec VSCode

#### Option A : Mode debug avec attente du debugger

1. **Activer le mode debug** dans `.env` :
   ```bash
   DEBUG_MODE=true
   ```

2. **Redémarrer l'addon** :
   ```bash
   docker compose restart ihp-ml-addon
   ```

3. **Vérifier les logs** - vous devriez voir :
   ```
   🔍 Debugpy listening on port 5678 - waiting for debugger to attach...
   💡 In VSCode: Run 'Python: Remote Attach (Docker)' debug configuration
   ```

4. **Dans VSCode** :
   - Ouvrir la vue Debug (`Ctrl+Shift+D`)
   - Sélectionner **"Python: Remote Attach (Docker)"**
   - Cliquer sur le bouton ▶️ (ou `F5`)

5. **Confirmer l'attachement** - les logs montreront :
   ```
   ✅ Debugger attached!
   ```

#### Option B : Mode debug sans attente (développement rapide)

1. **Désactiver le mode debug** dans `.env` :
   ```bash
   DEBUG_MODE=false
   ```

2. **L'addon démarre normalement**

3. **Attacher le debugger quand nécessaire** :
   - Dans VSCode : `Ctrl+Shift+D` → **"Python: Remote Attach (Docker)"** → `F5`
   - Vous pouvez attacher/détacher à tout moment

### 3. Configurations de debugging disponibles

#### 🐳 Python: Remote Attach (Docker)
- **Usage** : Debugging du code dans le conteneur Docker
- **Port** : 5678
- **Mapping** : `./ihp_ml_addon/rootfs/app` ↔ `/app`
- **Cas d'usage** : Debugging de l'API Flask, des adapters, des services

#### 📄 Python: Current File
- **Usage** : Exécuter et debugger un fichier Python local
- **Cas d'usage** : Tester rapidement un module isolé

#### 🧪 Python: Run Tests
- **Usage** : Exécuter les tests avec le debugger
- **Cas d'usage** : Debugging des tests unitaires

## 🛠️ Workflow de développement

### Développement sans rebuild (modifications Python uniquement)

1. **Modifier le code** dans `ihp_ml_addon/rootfs/app/`
2. **Redémarrer l'addon** (pas de rebuild nécessaire) :
   ```bash
   docker compose restart ihp-ml-addon
   ```
3. **Tester** : `curl http://localhost:5000/api/v1/health`

### Développement avec rebuild (changements de dépendances)

Si vous modifiez `requirements.txt` ou le `Dockerfile` :
```bash
docker compose build --no-cache ihp-ml-addon
docker compose up -d ihp-ml-addon
```

### Hot reload (pour développement intensif)

Pour éviter de redémarrer à chaque changement, vous pouvez utiliser Flask en mode debug :
```python
# Dans server.py, ligne "app.run(...)", changez:
app.run(host=host, port=port, debug=True)  # Hot reload activé
```

⚠️ **Attention** : En mode `debug=True`, Flask redémarre automatiquement, mais le debugpy ne se reconnecte pas automatiquement.

## 🔍 Debugging Tips

### Mettre des breakpoints

1. **Dans VSCode** : Cliquez dans la marge gauche du code (point rouge)
2. **Dans le code** : Ajoutez `breakpoint()` (Python 3.7+)

### Logs de debug

```python
import logging
_LOGGER = logging.getLogger(__name__)

_LOGGER.debug("Variable value: %s", my_var)
_LOGGER.info("Important info")
_LOGGER.warning("Warning message")
_LOGGER.error("Error occurred", exc_info=True)
```

### Inspecter les variables dans le conteneur

```bash
# Ouvrir un shell dans le conteneur
docker exec -it ihp-ml-addon-dev bash

# Tester un import
/opt/venv/bin/python -c "from domain.value_objects import TrainingData; print(TrainingData)"

# Vérifier les variables d'environnement
env | grep SUPERVISOR
```

### Tester les endpoints

```bash
# Status
curl -s http://localhost:5000/api/v1/status | jq

# Entraînement avec fake data
curl -s -X POST http://localhost:5000/api/v1/train/fake \
  -H "Content-Type: application/json" \
  -d '{"num_samples": 50}' | jq

# Prédiction
curl -s -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "outdoor_temp": 5.0,
    "indoor_temp": 18.0,
    "target_temp": 21.0,
    "humidity": 60.0,
    "hour_of_day": 14,
    "day_of_week": 2
  }' | jq
```

## 📝 Exemples de debug

### Exemple 1 : Debugger une erreur dans l'entraînement

1. **Mettre un breakpoint** dans `ml_application_service.py`, méthode `train_with_device_config()`
2. **Activer DEBUG_MODE=true** et redémarrer
3. **Attacher le debugger** VSCode
4. **Faire une requête** :
   ```bash
   curl -X POST http://localhost:5000/api/v1/train/device \
     -H "Content-Type: application/json" \
     -d @test_device_config.json
   ```
5. **Le debugger s'arrête** au breakpoint → inspecter les variables

### Exemple 2 : Debugger is_available()

1. **Breakpoint** dans `ha_history_reader.py`, méthode `is_available()`
2. **Requête de statut** : `curl http://localhost:5000/api/v1/status`
3. **Inspecter** :
   - La construction de l'URL
   - Les headers d'authentification
   - La réponse HTTP

## 🚨 Troubleshooting

### Le debugger ne se connecte pas

**Vérifier** :
```bash
# Port 5678 est bien exposé
docker ps | grep ihp-ml-addon

# Debugpy écoute bien
docker logs ihp-ml-addon-dev | grep debugpy
```

**Solution** :
- S'assurer que `DEBUG_MODE=true` dans `.env`
- Reconstruire : `docker compose build ihp-ml-addon`

### Les changements de code ne sont pas pris en compte

**Vérifier** :
```bash
# Le volume est bien monté
docker inspect ihp-ml-addon-dev | grep -A 5 Mounts
```

**Solution** :
- Redémarrer : `docker compose restart ihp-ml-addon`
- Vérifier que vous éditez le bon fichier (pas une copie dans le conteneur)

### ImportError ou ModuleNotFoundError

**Vérifier** :
```bash
# Python trouve bien les modules
docker exec ihp-ml-addon-dev /opt/venv/bin/python -c "import sys; print('\n'.join(sys.path))"
```

**Solution** :
- Vérifier le `PYTHONPATH` dans VSCode `launch.json`
- S'assurer que les `__init__.py` existent

## 📚 Ressources

- [Debugpy documentation](https://github.com/microsoft/debugpy)
- [VSCode Python debugging](https://code.visualstudio.com/docs/python/debugging)
- [Flask debugging](https://flask.palletsprojects.com/en/3.0.x/debugging/)
