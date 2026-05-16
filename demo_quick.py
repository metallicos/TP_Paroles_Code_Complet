#!/usr/bin/env python3

"""
Script de test rapide - Génère des paroles si le modèle existe
Utile pour les démos sans avoir à réentraîner
"""

import os
import sys


def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))


def test_quick_demo():
    project_root = get_project_root()
    model_path = os.path.join(project_root, 'outputs', 'lyrics_model.pkl')
    
    print("\n" + "="*60)
    print("DEMO RAPIDE - Génération de Paroles")
    print("="*60)
    
    if not os.path.exists(model_path):
        print("\n❌ Le modèle n'existe pas encore!")
        print(f"    Chemin attendu: {model_path}")
        print("\nPour entraîner le modèle:")
        print("    python3 TP_Paroles_Code_Complet.py")
        return False
    
    print(f"\n✓ Modèle trouvé: {model_path}")
    print("✓ Chargement...")
    
    try:
        from infer_lyrics import LyricsGenerator
        
        generator = LyricsGenerator(model_path)
        
        print("\n" + "-"*60)
        genres_list = generator.list_genres()
        
        for idx, genre in enumerate(genres_list[:3], 1):
            print(f"\n🎵 Exemple {idx}: {genre.upper()}")
            print("-" * 60)
            
            lyrics = generator.generate(
                genre,
                max_length=40,
                temperature=0.8
            )
            
            print(lyrics[:200] + "..." if len(lyrics) > 200 else lyrics)
        
        print("\n" + "="*60)
        print("✅ DÉMO RÉUSSIE!")
        print("="*60)
        print("\nPour générer d'autres paroles:")
        print("    python3 infer_lyrics.py --genre GENRE")
        print("    python3 infer_lyrics.py --list-genres")
        print()
        
        return True
        
    except ImportError as e:
        print(f"\n❌ Erreur d'import: {e}")
        print("\nInstallez les dépendances:")
        print("    pip3 install -r requirements.txt")
        return False
    
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        return False


if __name__ == '__main__':
    success = test_quick_demo()
    sys.exit(0 if success else 1)
