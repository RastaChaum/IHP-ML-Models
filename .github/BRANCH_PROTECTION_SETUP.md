# Configuration GitHub - Protection des Branches

## ⚙️ Configuration à effectuer dans GitHub

### 1. Créer la branche `integration`

```bash
# Localement
git checkout main
git checkout -b integration
git push -u origin integration
```

### 2. Configurer les Branch Protection Rules

Allez dans **Settings** → **Branches** → **Add branch protection rule**

#### Pour la branche `main` :

**Branch name pattern:** `main`

✅ Activer :
- **Require a pull request before merging**
  - Required approvals: 0 (ou 1 si vous voulez une validation)
  - ⚠️ IMPORTANT: **NE PAS** activer "Require linear history" (car on veut un merge commit)
- **Require status checks to pass before merging**
  - Status checks: (laisser vide, les workflows s'exécuteront après merge)
- **Do not allow bypassing the above settings** (recommandé)
- **Restrict who can push to matching branches**
  - Ajouter: `integration` (seule cette branche peut merger)

✅ Désactiver :
- Require linear history (on veut garder l'historique des merges)
- Require signed commits (sauf si vous utilisez GPG)

#### Pour la branche `integration` :

**Branch name pattern:** `integration`

✅ Activer :
- **Require a pull request before merging**
  - Required approvals: 0
- **Require status checks to pass before merging**
  - Status checks à sélectionner après le premier workflow:
    - `validate-pr` (de feature-fix-pr.yml)
- **Allow force pushes** (utile pour corrections en pre-release)

### 3. Configurer les Branch Rulesets (optionnel mais recommandé)

Allez dans **Settings** → **Rules** → **Rulesets** → **New ruleset**

**Nom:** `feature-fix-branches`

**Target branches:**
- Include by pattern: `feature-*`, `fix-*`

**Rules:**
- Branch naming conventions:
  - Pattern: `^(feature|fix)-[a-z0-9-]+$`
  - Message: "Les branches doivent commencer par 'feature-' ou 'fix-' suivi d'un nom en minuscules"

### 4. Configurer les secrets (si nécessaire)

Les workflows utilisent `GITHUB_TOKEN` qui est automatiquement fourni.

Aucun secret supplémentaire n'est nécessaire pour les workflows de base.

## 📋 Checklist de vérification

Après configuration, vérifiez :

- [ ] La branche `integration` existe
- [ ] Les règles de protection sur `main` empêchent les push directs
- [ ] Les règles de protection sur `integration` requièrent des PR
- [ ] Le template de PR apparaît lors de la création d'une nouvelle PR
- [ ] Les workflows GitHub Actions sont visibles dans l'onglet "Actions"

## 🔄 Workflow initial

1. **Assurez-vous que `main` et `integration` sont synchronisées** :
   ```bash
   git checkout main
   git pull origin main
   git checkout integration
   git merge main
   git push origin integration
   ```

2. **Testez le workflow** :
   ```bash
   ./scripts/workflow-helper.sh feature
   # Entrez: test-workflow
   
   # Faites un petit changement de test
   echo "# Test" >> test.md
   git add test.md
   
   # Mettez à jour le CHANGELOG
   # Ajoutez dans [Unreleased] / ### Added:
   # - Test du nouveau workflow
   
   git commit -m "feat: test workflow automation"
   git push origin feature-test-workflow
   ```

3. **Créez une PR sur GitHub** :
   - Allez sur GitHub
   - Créez une PR de `feature-test-workflow` vers `integration`
   - Vérifiez que le workflow `Feature/Fix PR Checks` s'exécute
   - Vérifiez les commentaires automatiques

4. **Mergez et testez la pre-release** :
   - Mergez la PR
   - Vérifiez qu'une pre-release est créée automatiquement
   - Allez dans "Releases" pour voir la pre-release

## 🚨 Points d'attention

### Gestion des corrections en integration

Quand vous êtes sur `integration` et que vous devez corriger quelque chose avant la release :

**Option 1 - Correction directe (simple):**
```bash
git checkout integration
# Faire la correction
git add .
git commit -m "fix: correction pre-release"
git push origin integration
# → Pre-release mise à jour automatiquement
```

**Option 2 - Via branche fix (recommandé):**
```bash
./scripts/workflow-helper.sh fix
# Nommer: prerelease-correction
# Faire la correction
# Créer PR vers integration
# Merger
```

### Incrémentation de version

La version dans `pyproject.toml` doit être incrémentée **avant** la PR integration → main :

```bash
git checkout integration
./scripts/workflow-helper.sh prepare-release
# Choisir: patch/minor/major
```

### Format du CHANGELOG

Toujours utiliser ce format :

```markdown
## [Unreleased]

### Added
- Nouvelles fonctionnalités visibles par l'utilisateur

### Changed
- Modifications de comportement existant

### Fixed
- Corrections de bugs

### Removed
- Fonctionnalités supprimées
```

**Bonnes pratiques :**
- ✅ "Amélioration de la précision des prédictions de température"
- ✅ "Correction du calcul de durée de chauffage pour les pièces >25m²"
- ❌ "Refactorisation de XGBoostAdapter.train_model()"
- ❌ "Ajout de type hints dans domain/entities"

## 🎯 Résumé du workflow automatisé

```
feature-*/fix-* → integration:
  ✅ Vérification du nom de branche
  ✅ Vérification CHANGELOG mis à jour
  ✅ Suggestion de template CHANGELOG
  ✅ Vérification docs (pour features)
  ✅ Tests unitaires
  ⚡ Création/MAJ pre-release automatique

integration → main:
  ✅ Vérification version incrémentée
  ✅ Vérification CHANGELOG non vide
  ✅ Tests complets
  ✅ Checklist dans commentaire PR
  ⚡ Après merge:
    - Création release GitHub
    - MAJ CHANGELOG avec date
    - Suppression pre-release
```
