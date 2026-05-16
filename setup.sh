#!/bin/bash

set -e

echo "===================================================="
echo "INITIALISATION DU PROJET - TP Génération Paroles"
echo "===================================================="
echo ""

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "📁 Répertoire du projet: $PROJECT_DIR"
cd "$PROJECT_DIR"

echo ""
echo "--- 1. Vérification des dépendances Python ---"
echo ""

if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 n'est pas installé"
    echo "   Sur Ubuntu/Debian: sudo apt-get install python3 python3-pip"
    echo "   Sur macOS: brew install python3"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
echo "✓ Python3 $PYTHON_VERSION trouvé"

if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 n'est pas installé"
    exit 1
fi

echo "✓ pip3 trouvé"

echo ""
echo "--- 2. Installation des dépendances ---"
echo ""

pip3 install -q -r requirements.txt
echo "✓ Dépendances installées"

echo ""
echo "--- 3. Vérification des fichiers ---"
echo ""

FILES_REQUIRED=(
    "TP_Paroles_Code_Complet.py"
    "infer_lyrics.py"
    "spotify_songs.csv"
    "requirements.txt"
)

MISSING_FILES=0
for file in "${FILES_REQUIRED[@]}"; do
    if [ -f "$file" ]; then
        SIZE=$(du -h "$file" | cut -f1)
        echo "✓ $file ($SIZE)"
    else
        echo "❌ $file (MANQUANT)"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
done

if [ $MISSING_FILES -gt 0 ]; then
    echo ""
    echo "⚠️  Fichiers manquants. Assurez-vous que tous les fichiers"
    echo "   sont dans le même répertoire que ce script."
    exit 1
fi

echo ""
echo "--- 4. Test d'import des dépendances ---"
echo ""

python3 << 'EOF'
import sys

modules = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'sklearn': 'scikit-learn',
    'matplotlib': 'matplotlib',
}

missing = []
for module, package in modules.items():
    try:
        __import__(module)
        print(f"✓ {package}")
    except ImportError:
        print(f"❌ {package}")
        missing.append(package)

if missing:
    print("\n❌ Dépendances manquantes:", ', '.join(missing))
    print("Relancez: pip3 install -r requirements.txt")
    sys.exit(1)

print("\n✓ Tous les imports réussissent")
EOF

if [ $? -ne 0 ]; then
    exit 1
fi

echo ""
echo "===================================================="
echo "✅ INITIALISATION TERMINÉE"
echo "===================================================="
echo ""
echo "Pour entraîner le modèle:"
echo "  python3 TP_Paroles_Code_Complet.py"
echo ""
echo "Pour générer des paroles:"
echo "  python3 infer_lyrics.py --genre rock --samples 3"
echo "  python3 infer_lyrics.py --list-genres"
echo ""
