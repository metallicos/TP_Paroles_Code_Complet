#!/bin/bash

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                                                                    ║"
echo "║  🎵 TP: GÉNÉRATION AUTOMATIQUE DE PAROLES DE CHANSON 🎵           ║"
echo "║                                                                    ║"
echo "║  Utilise les concepts du cours: Régression, Classification,       ║"
echo "║  Réseaux de Neurones, Gradient Descent, Cross-Entropy Loss        ║"
echo "║                                                                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"

echo ""
echo "📂 Répertoire du TP: /home/abdou/Public/sites/cours/"
echo ""

# Vérifier les prérequis
echo "✓ Vérification des prérequis..."

# Vérifier le CSV
if [ ! -f "spotify_songs.csv" ]; then
    echo "❌ ERREUR: spotify_songs.csv non trouvé"
    exit 1
fi
echo "  ✓ spotify_songs.csv trouvé ($(du -h spotify_songs.csv | cut -f1))"

# Vérifier Python
if ! command -v python &> /dev/null; then
    echo "❌ ERREUR: Python non trouvé"
    exit 1
fi
echo "  ✓ Python: $(python --version)"

# Vérifier les librairies
echo "  ✓ Vérification des librairies..."
python << 'PYEOF'
try:
    import numpy as np
    import pandas as pd
    from sklearn import __version__ as sklearn_ver
    import matplotlib.pyplot as plt
    print(f"    • numpy: {np.__version__}")
    print(f"    • pandas: {pd.__version__}")
    print(f"    • sklearn: {sklearn_ver}")
except ImportError as e:
    print(f"    ❌ ERREUR: Librairie manquante - {e}")
    exit(1)
PYEOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Installation des librairies nécessaires:"
    echo "   pip install numpy pandas scikit-learn matplotlib"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "                     FICHIERS DISPONIBLES"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "1. TP_Paroles_Code_Complet.py (Recommandé) ⭐"
echo "   → Script Python autonome complet"
echo "   → Exécution: python TP_Paroles_Code_Complet.py"
echo "   → Durée: ~5-10 minutes"
echo ""
echo "2. TP_Generation_Paroles_Chanson.ipynb"
echo "   → Notebook Jupyter pour VS Code"
echo "   → Exécution interactive avec explications"
echo ""
echo "3. infer_lyrics.py"
echo "   → Génération de paroles après entraînement"
echo "   → Usage: python infer_lyrics.py --genre rock --samples 3"
echo ""
echo "4. README_TP_Paroles.md"
echo "   → Documentation complète du TP"
echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "                   CHOIX D'EXÉCUTION"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "Sélectionnez une option:"
echo "  1) Exécuter le TP complet (Python script)"
echo "  2) Ouvrir le Notebook Jupyter"
echo "  3) Générer des paroles (après entraînement)"
echo "  4) Lire la documentation"
echo "  5) Quitter"
echo ""

read -p "Veuillez entrer votre choix (1-5): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Lancement du TP complet..."
        echo "════════════════════════════════════════════════════════════════════"
        echo ""
        python TP_Paroles_Code_Complet.py
        ;;
    2)
        echo ""
        echo "📓 Ouverture du Notebook Jupyter..."
        echo ""
        if command -v code &> /dev/null; then
            code TP_Generation_Paroles_Chanson.ipynb
        else
            echo "VS Code non trouvé. Fichier: TP_Generation_Paroles_Chanson.ipynb"
        fi
        ;;
    3)
        echo ""
        echo "🎵 Générer des paroles..."
        echo ""
        read -p "Genre (ex: rock, pop): " genre
        read -p "Nombre d'exemples (défaut: 1): " samples
        samples=${samples:-1}
        
        if [ ! -f "lyrics_model.pkl" ]; then
            echo "❌ ERREUR: lyrics_model.pkl non trouvé"
            echo "   Veuillez d'abord entraîner le modèle avec l'option 1"
            exit 1
        fi
        
        python infer_lyrics.py --genre "$genre" --samples "$samples"
        ;;
    4)
        echo ""
        echo "📚 Lecture de la documentation..."
        echo ""
        if command -v less &> /dev/null; then
            less README_TP_Paroles.md
        else
            cat README_TP_Paroles.md
        fi
        ;;
    5)
        echo "Au revoir! 👋"
        exit 0
        ;;
    *)
        echo "❌ Choix invalide"
        exit 1
        ;;
esac

echo ""
echo "════════════════════════════════════════════════════════════════════"
echo "✓ Opération complétée!"
echo "════════════════════════════════════════════════════════════════════"
