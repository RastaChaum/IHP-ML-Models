# Configuration avec python-dotenv

## Vue d'ensemble

Le serveur Flask charge automatiquement les variables d'environnement depuis un fichier `.env` au démarrage grâce à `python-dotenv`.

## Utilisation

### Développement local avec VSCode

1. **Copier le fichier d'exemple** (déjà fait dans ton cas) :
   ```bash
   cp .env.example .env
   ```

2. **Éditer `.env`** avec tes valeurs locales :
   ```bash
   # Ajuster pour ton instance HA locale
   SUPERVISOR_URL=http://192.168.1.100:8123
   SUPERVISOR_TOKEN=ton_token_longue_duree
   
   # Logs locaux
   LOG_LEVEL=DEBUG
   LOG_DIR=./test-data/logs
   
   # Models locaux
   MODEL_PERSISTENCE_PATH=./test-data/models
   ```

3. **Lancer le serveur** directement avec VSCode Debugger :
   - Le fichier `.env` sera chargé automatiquement
   - Pas besoin d'exporter manuellement les variables

### Développement avec Docker

Le fichier `.env` est **aussi** utilisé par docker-compose :

```bash
# Docker charge automatiquement .env
docker compose up ihp-ml-addon
```

## Priorité des variables

`python-dotenv` **ne remplace pas** les variables déjà définies :

1. **Variables système** (exportées dans le shell) : priorité maximale
2. **Variables docker-compose** : priorité moyenne
3. **Variables .env** : priorité minimale

### Exemple

```bash
# Dans .env
LOG_LEVEL=DEBUG

# Si tu exportes dans le shell :
export LOG_LEVEL=INFO

# Résultat : INFO (la variable système gagne)
```

## Fichiers .env multiples (optionnel)

Pour différents environnements :

```bash
.env                  # Par défaut (développement local)
.env.production       # Production (pas chargé automatiquement)
.env.test            # Tests (pas chargé automatiquement)
```

Pour charger un fichier spécifique, modifier `server.py` :

```python
# Dans server.py
from dotenv import load_dotenv

# Charger un fichier spécifique
load_dotenv('.env.production')
```

## Sécurité

⚠️ **Important** : Le fichier `.env` est déjà dans `.gitignore`

- **Ne jamais commiter `.env`** avec des tokens réels
- Utiliser `.env.example` comme template (sans valeurs sensibles)
- Pour production HA, utiliser les secrets Home Assistant

## Variables disponibles

Voir [.env.example](.env.example) pour la liste complète des variables configurables.

## Dépannage

### Le .env n'est pas chargé

**Vérifier que le fichier existe** :
```bash
ls -la .env
```

**Vérifier que python-dotenv est installé** :
```bash
poetry show python-dotenv
```

**Vérifier les logs au démarrage** :
```
2025-12-14 20:30:00 - INFO - Logging Configuration
2025-12-14 20:30:00 - INFO -   Log directory: ./test-data/logs
```

### Les variables ne sont pas prises en compte

**Vérifier l'ordre de chargement** dans `server.py` :

```python
# load_dotenv() DOIT être appelé AVANT d'importer les modules
# qui utilisent os.getenv()

from dotenv import load_dotenv
load_dotenv()  # ← En premier

# Puis les autres imports
from infrastructure.logging_config import setup_logging
```

## Commandes utiles

```bash
# Afficher les variables actuellement définies
printenv | grep -E "(LOG_|SUPERVISOR_|MODEL_)"

# Tester le chargement du .env
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.getenv('LOG_LEVEL'))"

# Recharger l'environnement Poetry après modification
poetry install
```
