#!/bin/bash
# Script helper pour gérer le workflow de développement

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

function print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

function print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

function print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

function print_error() {
    echo -e "${RED}❌ $1${NC}"
}

function create_feature_branch() {
    print_header "Création d'une branche feature"
    
    read -p "Nom de la feature (sans 'feature-'): " feature_name
    
    # Assurez-vous d'être sur integration
    git checkout integration
    git pull origin integration
    
    # Créez la branche
    git checkout -b "feature-$feature_name"
    
    print_success "Branche feature-$feature_name créée à partir de integration"
    print_warning "N'oubliez pas de mettre à jour le CHANGELOG.md !"
}

function create_fix_branch() {
    print_header "Création d'une branche fix"
    
    read -p "Nom du fix (sans 'fix-'): " fix_name
    
    # Assurez-vous d'être sur integration
    git checkout integration
    git pull origin integration
    
    # Créez la branche
    git checkout -b "fix-$fix_name"
    
    print_success "Branche fix-$fix_name créée à partir de integration"
    print_warning "N'oubliez pas de mettre à jour le CHANGELOG.md !"
}

function prepare_release() {
    print_header "Préparation d'une release"
    
    # Vérifier qu'on est sur integration
    CURRENT_BRANCH=$(git branch --show-current)
    if [ "$CURRENT_BRANCH" != "integration" ]; then
        print_error "Vous devez être sur la branche integration"
        exit 1
    fi
    
    # Lire la version actuelle
    CURRENT_VERSION=$(grep -m 1 'version = ' pyproject.toml | cut -d'"' -f2)
    print_header "Version actuelle: $CURRENT_VERSION"
    
    echo "Choisissez le type de release:"
    echo "1) Patch (bug fixes)       - $CURRENT_VERSION → $(bump_version $CURRENT_VERSION patch)"
    echo "2) Minor (new features)    - $CURRENT_VERSION → $(bump_version $CURRENT_VERSION minor)"
    echo "3) Major (breaking changes) - $CURRENT_VERSION → $(bump_version $CURRENT_VERSION major)"
    read -p "Votre choix (1/2/3): " choice
    
    case $choice in
        1) NEW_VERSION=$(bump_version $CURRENT_VERSION patch) ;;
        2) NEW_VERSION=$(bump_version $CURRENT_VERSION minor) ;;
        3) NEW_VERSION=$(bump_version $CURRENT_VERSION major) ;;
        *) print_error "Choix invalide"; exit 1 ;;
    esac
    
    print_header "Mise à jour de la version vers $NEW_VERSION"
    
    # Mise à jour de pyproject.toml
    sed -i "s/version = \"$CURRENT_VERSION\"/version = \"$NEW_VERSION\"/" pyproject.toml
    
    git add pyproject.toml
    git commit -m "chore: bump version to $NEW_VERSION"
    git push origin integration
    
    print_success "Version mise à jour vers $NEW_VERSION"
    print_warning "Une pre-release sera créée automatiquement sur GitHub"
    echo ""
    print_header "Prochaines étapes:"
    echo "1. Testez la pre-release sur votre instance"
    echo "2. Si des corrections sont nécessaires, créez des branches fix-*"
    echo "3. Quand tout est prêt, créez une PR: integration → main"
}

function bump_version() {
    local version=$1
    local part=$2
    
    IFS='.' read -r -a parts <<< "$version"
    major=${parts[0]}
    minor=${parts[1]}
    patch=${parts[2]}
    
    case $part in
        patch) echo "$major.$minor.$((patch + 1))" ;;
        minor) echo "$major.$((minor + 1)).0" ;;
        major) echo "$((major + 1)).0.0" ;;
    esac
}

function check_changelog() {
    print_header "Vérification du CHANGELOG"
    
    if ! git diff --cached --name-only | grep -q "CHANGELOG.md"; then
        print_error "CHANGELOG.md n'a pas été modifié"
        echo ""
        echo "Ajoutez une entrée dans la section [Unreleased]:"
        echo ""
        echo "### Added (pour features)"
        echo "- [Description orientée utilisateur]"
        echo ""
        echo "### Fixed (pour fixes)"
        echo "- [Description du problème corrigé]"
        exit 1
    fi
    
    print_success "CHANGELOG.md a été modifié"
}

function show_workflow() {
    cat << 'EOF'

📋 WORKFLOW DE DÉVELOPPEMENT
════════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────┐
│  1. NOUVELLE FONCTIONNALITÉ                                 │
└─────────────────────────────────────────────────────────────┘
   
   $ ./scripts/workflow-helper.sh feature
   
   → Créez votre branche feature-* depuis integration
   → Développez avec TDD (tests d'abord!)
   → Mettez à jour CHANGELOG.md (section [Unreleased])
   → Créez une PR vers integration
   → Les checks automatiques vérifient:
     - Nommage de branche
     - CHANGELOG mis à jour
     - Tests passent
     - Documentation (si feature)

┌─────────────────────────────────────────────────────────────┐
│  2. CORRECTION DE BUG                                       │
└─────────────────────────────────────────────────────────────┘
   
   $ ./scripts/workflow-helper.sh fix
   
   → Créez votre branche fix-* depuis integration
   → Corrigez avec TDD (tests de régression!)
   → Mettez à jour CHANGELOG.md (section [Unreleased])
   → Créez une PR vers integration
   → Mêmes checks automatiques

┌─────────────────────────────────────────────────────────────┐
│  3. PRÉPARER UNE RELEASE                                    │
└─────────────────────────────────────────────────────────────┘
   
   $ git checkout integration
   $ ./scripts/workflow-helper.sh prepare-release
   
   → Incrémente la version dans pyproject.toml
   → Push vers integration
   → Une PRE-RELEASE est créée automatiquement sur GitHub
   → Testez sur votre instance Home Assistant

┌─────────────────────────────────────────────────────────────┐
│  4. PUBLIER LA RELEASE                                      │
└─────────────────────────────────────────────────────────────┘
   
   → Créez une PR: integration → main
   → Les checks automatiques vérifient:
     - Version incrémentée
     - CHANGELOG contient des changements
     - Tests complets passent
   → Mergez la PR (avec un commit de merge)
   → La RELEASE est créée automatiquement sur GitHub
   → Le CHANGELOG est mis à jour avec la date

┌─────────────────────────────────────────────────────────────┐
│  5. CORRECTIONS POST-RELEASE (HOTFIX)                       │
└─────────────────────────────────────────────────────────────┘
   
   Si bug critique en production:
   
   $ git checkout main
   $ git checkout -b fix-critical-issue
   → Corrigez le bug
   → PR vers integration (tests)
   → Puis PR integration → main (nouvelle release)

EOF
}

# Menu principal
case "${1:-}" in
    feature)
        create_feature_branch
        ;;
    fix)
        create_fix_branch
        ;;
    prepare-release)
        prepare_release
        ;;
    check-changelog)
        check_changelog
        ;;
    workflow|help|--help|-h)
        show_workflow
        ;;
    *)
        echo "IHP ML Models - Workflow Helper"
        echo ""
        echo "Usage: $0 [command]"
        echo ""
        echo "Commands:"
        echo "  feature          Créer une nouvelle branche feature"
        echo "  fix              Créer une nouvelle branche fix"
        echo "  prepare-release  Préparer une nouvelle release"
        echo "  check-changelog  Vérifier que le CHANGELOG est à jour"
        echo "  workflow         Afficher le workflow complet"
        echo ""
        echo "Pour plus d'aide: $0 workflow"
        ;;
esac
