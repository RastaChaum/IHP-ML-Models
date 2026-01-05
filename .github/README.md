# GitHub Configuration & Workflows

Ce dossier contient toute la configuration GitHub Actions et la documentation du workflow de développement.

## 📁 Structure

```
.github/
├── workflows/              # GitHub Actions workflows
│   ├── pre-release.yml            # Auto pre-release sur push integration
│   ├── release.yml                # Auto release sur merge integration→main
│   ├── feature-fix-pr.yml         # Validation PR feature/fix→integration
│   ├── integration-pr.yml         # Validation PR integration→main
│   ├── cleanup-branches.yml       # Suppression auto branches mergées
│   └── e2e-tests.yml.example      # Template tests E2E
├── agents/                 # Configuration agents GitHub Copilot
├── pull_request_template.md      # Template pour toutes les PR
├── copilot-instructions.md       # Instructions pour GitHub Copilot
├── BRANCH_PROTECTION_SETUP.md    # 🔧 Setup initial GitHub (À FAIRE EN PREMIER)
├── WORKFLOW_GUIDE.md             # 📖 Guide complet du workflow
└── FUTURE_IMPROVEMENTS.md        # 🚀 Améliorations futures possibles
```

## 🚀 Démarrage rapide

### 1. Configuration initiale (une seule fois)

Suivez **[BRANCH_PROTECTION_SETUP.md](BRANCH_PROTECTION_SETUP.md)** pour :
- Créer la branche `integration`
- Configurer les protections de branches sur GitHub
- Tester le workflow avec une branche de test

**⚠️ À faire avant toute chose !**

### 2. Utilisation quotidienne

Consultez **[WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md)** pour :
- Créer une nouvelle feature
- Corriger un bug
- Préparer une release
- Publier en production

**Script helper disponible :**
```bash
./scripts/workflow-helper.sh workflow  # Voir le workflow complet
./scripts/workflow-helper.sh feature   # Créer une branche feature
./scripts/workflow-helper.sh fix       # Créer une branche fix
./scripts/workflow-helper.sh prepare-release  # Préparer une release
```

## 🔄 Workflows automatisés

### 1. Feature/Fix → Integration

**Déclenché par :** PR vers `integration` depuis branche `feature-*` ou `fix-*`

**Actions automatiques :**
- ✅ Validation du nom de branche
- ✅ Vérification CHANGELOG mis à jour
- ✅ Tests unitaires
- ✅ Vérification documentation (features uniquement)
- 💬 Commentaires automatiques avec suggestions

**Fichier :** [workflows/feature-fix-pr.yml](workflows/feature-fix-pr.yml)

### 2. Push sur Integration

**Déclenché par :** Push ou merge sur branche `integration`

**Actions automatiques :**
- 🏷️ Création/mise à jour pre-release GitHub
- 📝 Extraction CHANGELOG pour release notes
- ⚠️ Warning si docs non mises à jour

**Fichier :** [workflows/pre-release.yml](workflows/pre-release.yml)

### 3. Integration → Main (PR)

**Déclenché par :** PR vers `main` depuis branche `integration`

**Actions automatiques :**
- ✅ Vérification version incrémentée
- ✅ Vérification CHANGELOG contient changements
- ✅ Tests complets (unit + integration)
- 📋 Checklist de release dans commentaire
- ✅ Vérification existence pre-release

**Fichier :** [workflows/integration-pr.yml](workflows/integration-pr.yml)

### 4. Integration → Main (Merge)

**Déclenché par :** Merge de PR `integration` → `main`

**Actions automatiques :**
- 🚀 Création release GitHub officielle
- 📝 Mise à jour CHANGELOG avec date
- 🗑️ Suppression pre-release beta
- 🏷️ Création tag Git
- 💬 Commentaire confirmation sur PR

**Fichier :** [workflows/release.yml](workflows/release.yml)

### 5. Nettoyage des branches

**Déclenché par :** Fermeture de PR (merged)

**Actions automatiques :**
- 🗑️ Suppression automatique de la branche mergée
- 🛡️ Protection branches `main` et `integration`

**Fichier :** [workflows/cleanup-branches.yml](workflows/cleanup-branches.yml)

## 📋 Modèle de développement

```
main (production)
  └─ Seule integration peut merger
  └─ Releases officielles uniquement

integration (pre-release)
  ├─ feature-* (nouvelles fonctionnalités)
  ├─ fix-* (corrections de bugs)
  └─ Préparation release + tests

feature-*/fix-*
  └─ Branches de développement
  └─ Merge via PR vers integration
```

## 🎯 Processus complet

### Développement d'une feature

```bash
# 1. Créer branche
./scripts/workflow-helper.sh feature
# → Entrer nom: awesome-feature

# 2. Développer avec TDD
# - Tests d'abord
# - Implémentation
# - Mise à jour CHANGELOG.md

# 3. Push et PR
git push origin feature-awesome-feature
# → Créer PR sur GitHub vers integration

# 4. Review automatique
# → Workflows vérifient tout
# → Commentaires automatiques si problèmes

# 5. Merge PR
# → Pre-release créée/mise à jour automatiquement
```

### Publication d'une release

```bash
# 1. Tester la pre-release
# → Installer sur instance HA réelle
# → Vérifier fonctionnalités
# → Corriger si nécessaire (nouvelles PR)

# 2. Préparer release
git checkout integration
./scripts/workflow-helper.sh prepare-release
# → Choisir version (patch/minor/major)
# → Version incrémentée automatiquement

# 3. PR vers main
# → Créer PR: integration → main
# → Vérifications automatiques
# → Checklist affichée

# 4. Merge
# → Release créée automatiquement
# → CHANGELOG mis à jour
# → Tag Git créé
```

## 📚 Documentation

### Pour les contributeurs

1. **[WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md)** - Guide complet du processus de développement
   - Diagrammes du workflow
   - Cas d'usage détaillés
   - Commandes utiles
   - Best practices

2. **[pull_request_template.md](pull_request_template.md)** - Template de PR
   - Checklist complète
   - Format standardisé
   - Rappel des bonnes pratiques

### Pour les mainteneurs

1. **[BRANCH_PROTECTION_SETUP.md](BRANCH_PROTECTION_SETUP.md)** - Configuration GitHub
   - Protections de branches
   - Rulesets
   - Tests de validation

2. **[FUTURE_IMPROVEMENTS.md](FUTURE_IMPROVEMENTS.md)** - Améliorations futures
   - Liste priorisée
   - Pros/cons de chaque amélioration
   - Recommandations d'implémentation

### Pour GitHub Copilot

**[copilot-instructions.md](copilot-instructions.md)** - Instructions pour l'IA
- Architecture DDD obligatoire
- Standards TDD
- Conventions de code
- Anti-patterns à éviter

## 🔧 Configuration requise

### Secrets GitHub

Aucun secret supplémentaire requis ! Les workflows utilisent `GITHUB_TOKEN` qui est automatiquement fourni.

### Permissions

Les workflows nécessitent ces permissions (déjà configurées) :
- `contents: write` - Pour créer releases et tags
- `pull-requests: write` - Pour commenter sur les PR

### Branch Protection Rules

À configurer dans GitHub Settings → Branches :

**main :**
- Require PR before merging
- Restrict who can push: `integration` only
- Do NOT require linear history (on veut les merge commits)

**integration :**
- Require PR before merging
- Require status checks: `validate-pr`
- Allow force pushes (pour corrections)

Voir détails dans [BRANCH_PROTECTION_SETUP.md](BRANCH_PROTECTION_SETUP.md)

## ❓ FAQ

### Pourquoi deux branches (main + integration) ?

- `main` = code en production, stable, releases officielles
- `integration` = laboratoire de préparation, pre-releases, tests
- Permet corrections itératives sans affecter production

### Pourquoi pas de squash commits ?

Pour `integration → main`, on veut garder l'historique complet :
- Traçabilité des features/fixes individuels
- Facilite les git bisect
- Historique plus riche

### Peut-on forcer-push sur integration ?

Oui, mais seulement en cas de nécessité (correction d'erreur critique).
La protection permet force-push pour cette raison.

### Comment annuler une pre-release ?

```bash
gh release delete v0.2.0-beta --yes
git push origin :refs/tags/v0.2.0-beta
```

La prochaine fois qu'on push sur integration, elle sera recréée.

### Comment faire un hotfix ?

Toujours passer par `integration` :
1. Créer fix depuis integration
2. PR → integration
3. Tester pre-release
4. Bump version (patch)
5. PR integration → main

**Jamais** créer de branche depuis `main` directement.

## 🆘 Support

En cas de problème avec les workflows :

1. Vérifier que les protections de branches sont bien configurées
2. Consulter les logs dans l'onglet "Actions" de GitHub
3. Vérifier que `GITHUB_TOKEN` a les bonnes permissions
4. Relire [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md)

## 🎉 Remerciements

Ce workflow s'inspire de :
- GitFlow (branches séparées dev/production)
- GitHub Flow (simplification via PR)
- Trunk-Based Development (intégration continue)
- Semantic Versioning (versions explicites)

Adapté spécifiquement pour le développement d'addons Home Assistant avec pre-releases obligatoires.
