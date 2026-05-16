# TP: Génération Automatique de Paroles de Chanson 🎵

Projet de machine learning permettant de générer des paroles de chanson basées sur le genre musical, en utilisant les techniques du cours: régression, classification et réseaux de neurones.

## Installation Rapide

### Pour Linux/macOS:

```bash
# 1. Rendre le script executable
chmod +x setup.sh

# 2. Lancer l'installation
./setup.sh

# 3. Vérifier que tout fonctionne
python3 test_project.py
```

### Pour Windows:

```bash
# 1. Installer Python et pip si nécessaire

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. Vérifier installation
python test_project.py
```

## Structure du Projet

```
.
├── TP_Paroles_Code_Complet.py    # Script d'entraînement (main)
├── infer_lyrics.py                # Script d'inférence (génération)
├── spotify_songs.csv              # Dataset (18,454 chansons)
├── lyrics_model.pkl               # Modèle entraîné (créé après étape 1)
├── requirements.txt               # Dépendances Python
├── setup.sh                        # Script d'initialisation (Linux/macOS)
├── test_project.py                # Tests de vérification
└── README.md                       # Ce fichier
```

## Utilisation

### Étape 1: Entraîner le Modèle

```bash
python3 TP_Paroles_Code_Complet.py
```

Cela va:
- Charger le dataset Spotify (18,454 chansons)
- Prétraiter les paroles (tokenization, nettoyage)
- Créer un vocabulaire de ~15k mots
- Entraîner un réseau de neurones pendant 10 epochs
- Sauvegarder le modèle dans `lyrics_model.pkl`

**Durée estimée:** 5-15 minutes selon votre machine

**Output:** Un modèle d'~2MB sauvegardé automatiquement

### Étape 2: Générer des Paroles

```bash
# Générer une chanson rock
python3 infer_lyrics.py --genre rock

# Générer 3 exemples pop
python3 infer_lyrics.py --genre pop --samples 3

# Lister tous les genres disponibles
python3 infer_lyrics.py --list-genres

# Options avancées:
# --length N      : Nombre de mots max (défaut: 50)
# --temperature X : Contrôle la créativité (< 1 = déterministe, > 1 = aléatoire)
```

## Architecture du Modèle

### Concepts Appliqués du Cours

| Concept | Application |
|---------|------------|
| **Régression** | Prédiction du prochain mot (tâche de classification multilabel) |
| **Classification** | Encodage du genre musical (5 classes) |
| **Réseaux Neuronaux** | Architecture 2 couches avec embeddings |
| **Gradient Descent** | Mise à jour des poids pendant l'entraînement |
| **Cross-Entropy** | Fonction de coût pour la classification |
| **Backpropagation** | Calcul des gradients via chaîne de dérivation |

### Structure Technique

```
Input: Séquence de tokens (10 mots max)
  ↓
Word Embeddings (vocabulaire → vecteurs 16D)
  ↓
Genre Embeddings (genre → vecteur 16D)
  ↓
Concatenation: [word_embed, genre_embed]
  ↓
Dense Layer 1: 176 → 32 (ReLU activation)
  ↓
Dense Layer 2: 32 → vocab_size (Softmax activation)
  ↓
Output: Probabilités du prochain mot
```

### Hyperparamètres

- **Embedding Dimension:** 16
- **Hidden Layer Size:** 32
- **Sequence Length:** 10 tokens
- **Epochs:** 10
- **Batch Size:** 128
- **Learning Rate:** 0.001
- **Vocabulary Size:** ~15,000 mots

## Dépendances

- `pandas >= 1.1.0` - Manipulation de données
- `numpy >= 1.19.0` - Calcul numérique
- `scikit-learn >= 0.23.0` - Prétraitement données
- `matplotlib >= 3.3.0` - Visualisation

Toutes les dépendances sont listées dans `requirements.txt`

## Vérification de l'Installation

```bash
python3 test_project.py
```

Vérifiera:
- ✓ Tous les imports Python
- ✓ Présence du dataset
- ✓ Validité des fichiers
- ✓ État du modèle

## Fichiers Explicités

### TP_Paroles_Code_Complet.py (Principal)
Script d'entraînement complet divisé en 10 sections:
1. **Section 1**: Chargement du dataset
2. **Section 2**: Exploration des genres
3. **Section 3**: Prétraitement du texte
4. **Section 4**: Construction du vocabulaire
5. **Section 5**: Encodage des données
6. **Section 6**: Préparation train/validation
7. **Section 7**: Modèle neural network
8. **Section 8**: Entraînement avec gradient descent
9. **Section 9**: Génération et tests
10. **Section 10**: Sauvegarde du modèle

**Chemin du modèle:** Automatiquement déterminé (même répertoire)
**Chemin du dataset:** Automatiquement déterminé (même répertoire)

### infer_lyrics.py (Inférence)
Script d'inférence pour générer des paroles avec le modèle entraîné.

**Features:**
- Interface CLI avec argparse
- Support multi-échantillons
- Contrôle température (créativité)
- Lister les genres disponibles

**Commandes:**
```bash
python3 infer_lyrics.py --genre GENRE [--samples N] [--length L] [--temperature T]
```

### requirements.txt
Spécifie toutes les dépendances avec versions minimales.
Utilisé par pip pour l'installation.

### setup.sh (Linux/macOS)
Script d'automatisation qui:
- Vérifie Python3 et pip3
- Installe les dépendances
- Vérifie les fichiers requis
- Teste les imports

### test_project.py
Suite de tests pour vérifier:
- Importabilité des modules
- Chargeabilité du dataset
- Validité des fichiers
- État du modèle

## Dépannage

### Python/pip non trouvé
```bash
# Ubuntu/Debian
sudo apt-get install python3 python3-pip

# macOS (avec brew)
brew install python3

# Windows
# Télécharger depuis python.org
```

### Dépendances manquantes
```bash
pip3 install -r requirements.txt
```

### Dataset non trouvé
Assurez-vous que `spotify_songs.csv` est dans le même répertoire que les scripts.

### L'entraînement est trop lent
- C'est normal sur CPU (5-15 min)
- Réduisez le nombre d'epochs (ligne 300): `NUM_EPOCHS = 5`
- Réduisez la taille du batch: `BATCH_SIZE = 64`

### Pas assez de mémoire
- Réduisez le batch size: `BATCH_SIZE = 32`
- Réduisez le vocabulaire minimum: `MIN_FREQ = 5`

## Structure des Chemins Dynamiques

Les scripts utilisent des chemins **dynamiques et relatifs**:
```python
def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))

def get_data_path(filename):
    return os.path.join(get_project_root(), filename)
```

Cela signifie:
- ✓ Fonctionne sur n'importe quel ordinateur
- ✓ Fonctionne depuis n'importe quel répertoire
- ✓ Le modèle est sauvegardé dans le même dossier que les scripts

## Améliorations Possibles

1. **Augmentation de données:** Ajouter des transformations textuelles
2. **Meilleur modèle:** LSTM ou Transformers
3. **Fine-tuning:** En fonction d'un genre spécifique
4. **Visualisations:** Courbes de loss, embeddings t-SNE
5. **API Web:** Flask ou FastAPI pour servir le modèle

## Concepts Mathématiques Appliqués

### Math utilisée:
- **Embeddings:** $\text{embedding}(w) \in \mathbb{R}^d$
- **Forward Pass:** $z = Wx + b$, $a = \text{ReLU}(z)$
- **Softmax:** $\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$
- **Cross-Entropy:** $L = -\sum_i y_i \log(\hat{y}_i)$
- **Backpropagation:** $\frac{\partial L}{\partial W} = \frac{\partial L}{\partial a} \cdot \frac{\partial a}{\partial W}$
- **Gradient Descent:** $W \leftarrow W - \alpha \nabla L$

## Résultats Attendus

Après l'entraînement, vous devriez voir:
- **Train Loss:** ~1.2 → ~0.6
- **Validation Loss:** ~1.3 → ~0.7
- **Génération:** Paroles lisibles mais simples (modèle basique)

Exemple:
```
ROCK:
  the day when you want to be with me i know...

POP:  
  love is in the air tonight feel the magic...
```

## Auteur & Licences

Ce TP utilise:
- Dataset Spotify (18,454 chansons avec paroles)
- Techniques du cours de Machine Learning

## Support

Pour toute question, vérifiez:
1. Les fichiers sont dans le bon répertoire
2. Les dépendances sont installées
3. Le dataset existe et n'est pas corrompu
4. La version de Python est >= 3.7

---

**Bon codage! 🎤🎵**
