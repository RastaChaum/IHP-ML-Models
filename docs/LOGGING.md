# Logging Configuration - IHP ML Models

## Overview

Le système de logging utilise Python's `logging` avec rotation automatique des fichiers pour éviter qu'ils ne deviennent trop volumineux.

## Fichiers de logs

Les logs sont écrits dans `/data/logs/` (ou selon `LOG_DIR`) :

- **`ihp_ml.log`** : Logs principaux (niveau configurable via `LOG_LEVEL`)
- **`ihp_ml_debug.log`** : Logs complets en mode DEBUG (toujours)

Chaque fichier :
- Taille maximale : 10 MB
- Rotation automatique : 5 backups (50 MB total par fichier)
- Format : `ihp_ml.log.1`, `ihp_ml.log.2`, etc.

## Configuration via variables d'environnement

### `LOG_LEVEL`

Contrôle le niveau de verbosité dans le terminal et le fichier principal.

```bash
# Options : DEBUG, INFO, WARNING, ERROR, CRITICAL
export LOG_LEVEL=DEBUG    # Pour développement/troubleshooting
export LOG_LEVEL=INFO     # Par défaut - recommandé
export LOG_LEVEL=WARNING  # Production - erreurs uniquement
```

### `LOG_DIR`

Change le répertoire de destination des logs.

```bash
export LOG_DIR=/data/logs              # Par défaut (addon HA)
export LOG_DIR=/tmp/ihp_logs           # Pour tests locaux
```

## Exemple de configuration pour tests

Pour des logs très détaillés lors de vos tests :

```bash
# Dans docker-compose.yml ou .env
LOG_LEVEL=DEBUG
LOG_DIR=/data/logs
```

## Lecture des logs

### En temps réel (terminal)

```bash
# Suivre les logs principaux
tail -f /data/logs/ihp_ml.log

# Suivre les logs debug
tail -f /data/logs/ihp_ml_debug.log

# Filtrer par module
tail -f /data/logs/ihp_ml_debug.log | grep "gymnasium_heating_env"
```

### Analyser un épisode spécifique

```bash
# Voir tous les rewards d'un épisode
grep -A 30 "Episode reset" /data/logs/ihp_ml_debug.log | grep "reward="

# Statistiques d'entraînement
grep "Episode.*finished" /data/logs/ihp_ml.log
```

### Compter les épisodes

```bash
# Nombre d'épisodes entraînés
grep -c "Episode.*finished" /data/logs/ihp_ml.log

# Longueur moyenne des épisodes
grep "Episode.*finished" /data/logs/ihp_ml.log | \
  awk -F'length=' '{print $2}' | awk '{print $1}' | \
  awk '{sum+=$1; count++} END {print "Avg:", sum/count}'
```

## Format des logs

### Terminal / ihp_ml.log
```
2025-12-14 18:29:38 - infrastructure.adapters.sb3_rl_trainer - INFO - Episode 1 finished: reward=1.00, length=30
```

### ihp_ml_debug.log (avec fonction et ligne)
```
2025-12-14 18:29:38 - infrastructure.adapters.gymnasium_heating_env - DEBUG - step:152 - Step 30: action=0, reward=0.100, done=True
```

## Modules principaux à surveiller

- `infrastructure.adapters.ha_history_reader` : Récupération des données HA
- `infrastructure.adapters.gymnasium_heating_env` : Environnement RL
- `infrastructure.adapters.sb3_rl_trainer` : Entraînement PPO
- `domain.services.heating_reward_calculator` : Calcul des rewards
- `domain.services.rl_episode_service` : Détection des fins d'épisodes

## Désactiver la rotation (pour debug temporaire)

Si vous voulez un seul gros fichier sans rotation pendant un test :

```python
# Dans logging_config.py, ligne 67 et 78, mettre backup_count=0
```

## Nettoyage manuel

```bash
# Supprimer tous les anciens logs
rm -f /data/logs/ihp_ml*.log*

# Garder uniquement les 2 dernières rotations
find /data/logs -name "ihp_ml*.log.*" -type f | sort -r | tail -n +3 | xargs rm -f
```
