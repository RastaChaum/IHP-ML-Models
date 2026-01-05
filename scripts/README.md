# Development Scripts

Ce dossier contient des scripts pour le développement local, les tests et la gestion du workflow GitHub.

## 🚀 Scripts de Workflow GitHub (NOUVEAUX)

### `setup-workflow.sh` ⭐
**Configuration initiale automatique du workflow Git Flow.**

```bash
./scripts/setup-workflow.sh
```

Ce script configure automatiquement :
- ✅ Création de la branche `integration`
- ✅ Synchronisation avec `main`
- ✅ Push vers origin
- ✅ Vérification finale

**À exécuter une seule fois lors de la mise en place du workflow.**

### `workflow-helper.sh` ⭐
**Assistant pour toutes les opérations de développement quotidiennes.**

```bash
# Créer une nouvelle feature
./scripts/workflow-helper.sh feature

# Créer un fix
./scripts/workflow-helper.sh fix

# Préparer une release
./scripts/workflow-helper.sh prepare-release

# Vérifier le CHANGELOG
./scripts/workflow-helper.sh check-changelog

# Afficher le guide complet
./scripts/workflow-helper.sh workflow
```

Ce script gère :
- ✅ Création de branches feature-* et fix-*
- ✅ Incrémentation de version automatique
- ✅ Validation du CHANGELOG
- ✅ Workflow complet expliqué

**Utilisez-le pour TOUTES vos opérations de développement.**

### `check-workflow-setup.sh`
**Vérification complète de la configuration du workflow.**

```bash
./scripts/check-workflow-setup.sh
```

Vérifie :
- ✅ Installation GitHub CLI (gh)
- ✅ Authentification GitHub
- ✅ Existence des branches (main, integration)
- ✅ Présence des workflows GitHub Actions
- ✅ Documentation complète
- ⚠️ Protection des branches (à configurer sur GitHub)
- 📊 Status des releases

**Exécutez régulièrement pour vérifier que tout est configuré correctement.**

## 🧪 Scripts de Développement & Tests

### `develop.sh`
Démarrer l'environnement de développement local avec Home Assistant et l'addon.

```bash
./scripts/develop.sh
```

Ce script :
- Crée les répertoires de configuration de test
- Initialise une instance Home Assistant de test
- Build et démarre le conteneur de l'addon
- Affiche informations utiles et commandes

### `test-addon.sh`
Exécuter les tests API automatisés contre l'addon en cours d'exécution.

```bash
./scripts/test-addon.sh
```

Tests inclus :
- Health check endpoint
- Status endpoint
- Training avec données fake
- Endpoint de prédiction
- Liste des modèles

### `run-tests.sh`
Exécuter les tests unitaires avec Poetry.

```bash
./scripts/run-tests.sh
```

### `run-integration-tests.sh`
Exécuter les tests d'intégration.

```bash
./scripts/run-integration-tests.sh
```

### `run-e2e-tests.sh`
Exécuter les tests end-to-end.

```bash
./scripts/run-e2e-tests.sh
```

### `clean.sh`
Nettoyer toutes les données de test et conteneurs.

```bash
./scripts/clean.sh
```

Supprime :
- Conteneurs et volumes Docker
- Répertoire de configuration de test
- Répertoire de données de test
- Fichier d'environnement

## 📋 Workflow recommandé

### 1️⃣ Configuration initiale (une seule fois)

```bash
# 1. Configurer le workflow Git Flow
./scripts/setup-workflow.sh

# 2. Vérifier la configuration
./scripts/check-workflow-setup.sh

# 3. Configurer les protections de branches sur GitHub
# Suivre : .github/BRANCH_PROTECTION_SETUP.md
```

### 2️⃣ Développement quotidien

```bash
# Nouvelle fonctionnalité
./scripts/workflow-helper.sh feature
# → Développer avec TDD
# → Mettre à jour CHANGELOG.md
# → git push + créer PR sur GitHub

# Correction de bug
./scripts/workflow-helper.sh fix
# → Corriger avec test de régression
# → Mettre à jour CHANGELOG.md
# → git push + créer PR sur GitHub

# Tests locaux
./scripts/run-tests.sh
./scripts/test-addon.sh
```

### 3️⃣ Préparation release

```bash
# 1. Tester la pre-release (créée auto sur integration)
./scripts/develop.sh  # Tester localement

# 2. Incrémenter version et préparer release
git checkout integration
./scripts/workflow-helper.sh prepare-release

# 3. Créer PR integration → main sur GitHub
# → Release automatique après merge
```

## 🔍 Diagnostic et vérification

```bash
# Vérifier la configuration complète
./scripts/check-workflow-setup.sh

# Afficher le workflow complet
./scripts/workflow-helper.sh workflow

# Vérifier que le CHANGELOG est à jour
./scripts/workflow-helper.sh check-changelog
```

## 📚 Documentation

Pour plus de détails sur le workflow GitHub :

- **📖 [.github/README.md](../.github/README.md)** - Point d'entrée de la documentation
- **📖 [.github/WORKFLOW_GUIDE.md](../.github/WORKFLOW_GUIDE.md)** - Guide complet du workflow
- **🔧 [.github/BRANCH_PROTECTION_SETUP.md](../.github/BRANCH_PROTECTION_SETUP.md)** - Configuration GitHub
- **🚀 [.github/FUTURE_IMPROVEMENTS.md](../.github/FUTURE_IMPROVEMENTS.md)** - Améliorations futures

## ⚙️ Configuration requise

- **Git** - Gestion de version
- **GitHub CLI (gh)** - Pour interactions avec GitHub API
  ```bash
  # Installation
  sudo apt install gh
  # Ou : https://cli.github.com/
  
  # Authentification
  gh auth login
  ```
- **Docker & Docker Compose** - Pour développement local
- **Poetry** - Gestion des dépendances Python

## 🆘 Aide

En cas de problème :

1. Vérifier la configuration : `./scripts/check-workflow-setup.sh`
2. Consulter la documentation : `.github/README.md`
3. Voir le workflow détaillé : `./scripts/workflow-helper.sh workflow`
4. Vérifier les logs GitHub Actions sur le repository
3. **View logs**: `docker-compose logs -f ihp-ml-addon`
4. **Clean up**: `./scripts/clean.sh`

See [DEVELOPMENT.md](../DEVELOPMENT.md) for detailed documentation.
