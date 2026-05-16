#!/usr/bin/env python3

import os
import sys
import pandas as pd


def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))


def test_imports():
    print("=" * 60)
    print("TEST 1: Vérification des imports")
    print("=" * 60)
    
    required_modules = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        're': 're',
        'pickle': 'pickle',
    }
    
    failed = []
    for module, name in required_modules.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            failed.append(name)
    
    if failed:
        print(f"\n❌ Imports échoués: {', '.join(failed)}")
        return False
    
    print("✓ Tous les imports réussissent\n")
    return True


def test_dataset():
    print("=" * 60)
    print("TEST 2: Vérification du Dataset")
    print("=" * 60)
    
    project_root = get_project_root()
    csv_path = os.path.join(project_root, 'spotify_songs.csv')
    
    if not os.path.exists(csv_path):
        print(f"❌ Dataset introuvable: {csv_path}")
        return False
    
    try:
        df = pd.read_csv(csv_path)
        print(f"✓ Dataset chargé: {len(df):,} lignes")
        
        required_cols = ['lyrics', 'playlist_genre']
        for col in required_cols:
            if col in df.columns:
                print(f"✓ Colonne '{col}' trouvée")
            else:
                print(f"❌ Colonne '{col}' manquante")
                return False
        
        df_clean = df[df['lyrics'].notna()].copy()
        print(f"✓ Paroles valides: {len(df_clean):,}")
        
        genres = df['playlist_genre'].nunique()
        print(f"✓ Genres distincts: {genres}")
    except Exception as e:
        print(f"❌ Erreur lors de la lecture: {e}")
        return False
    
    print("✓ Dataset valide\n")
    return True


def test_model_existance():
    print("=" * 60)
    print("TEST 3: Vérification du Modèle")
    print("=" * 60)
    
    project_root = get_project_root()
    model_path = os.path.join(project_root, 'outputs', 'lyrics_model.pkl')
    legacy_model_path = os.path.join(project_root, 'lyrics_model.pkl')
    
    if not os.path.exists(model_path) and os.path.exists(legacy_model_path):
        model_path = legacy_model_path
    
    if os.path.exists(model_path):
        size_kb = os.path.getsize(model_path) / 1024
        print(f"✓ Modèle trouvé: {size_kb:.1f} KB")
        
        try:
            import pickle
            with open(model_path, 'rb') as f:
                pkg = pickle.load(f)
                print(f"✓ Modèle valide")
                print(f"  Vocabulaire: {len(pkg['vocab']['word2idx']):,} mots")
                print(f"  Genres: {pkg['config']['num_genres']}")
        except Exception as e:
            print(f"❌ Erreur de lecture: {e}")
            return False
    else:
        print(f"⚠️  Modèle non encore entraîné")
        print(f"   Lancez d'abord: python3 TP_Paroles_Code_Complet.py")
    
    print()
    return True


def test_scripts():
    print("=" * 60)
    print("TEST 4: Vérification des Scripts")
    print("=" * 60)
    
    project_root = get_project_root()
    scripts = [
        ('TP_Paroles_Code_Complet.py', 'Entraînement'),
        ('infer_lyrics.py', 'Inférence'),
    ]
    
    for script, desc in scripts:
        path = os.path.join(project_root, script)
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"✓ {script} ({size_kb:.1f} KB) - {desc}")
        else:
            print(f"❌ {script} manquant")
            return False
    
    print("✓ Tous les scripts présents\n")
    return True


def main():
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " TESTS DE VÉRIFICATION DU PROJET ".center(58) + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    tests = [
        test_imports,
        test_dataset,
        test_scripts,
        test_model_existance,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Erreur: {e}\n")
            results.append(False)
    
    print("=" * 60)
    print("RÉSUMÉ")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✓ Réussis: {passed}/{total}")
    
    if passed == total:
        print("\n✅ TOUS LES TESTS RÉUSSISSENT!")
        print("\nPour entraîner le modèle:")
        print("  python3 TP_Paroles_Code_Complet.py")
        return 0
    else:
        print("\n❌ CERTAINS TESTS ONT ÉCHOUÉ")
        print("Vérifiez l'installation et les fichiers requis.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
