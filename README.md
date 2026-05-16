# 🎵 TP — Génération Automatique de Paroles de Chanson

> Projet de Machine Learning : génération de paroles par genre musical
> Techniques appliquées : embeddings, réseaux de neurones, backpropagation, gradient descent

---

## Table des Matières

- [Installation](#installation)
- [Utilisation](#utilisation)
- [Paramètres & Configuration](#paramètres--configuration)
- [Architecture du Modèle](#architecture-du-modèle)
- [Métriques & Suivi](#métriques--suivi)
- [Support GPU](#support-gpu)
- [Structure des Fichiers](#structure-des-fichiers)
- [Dépannage](#dépannage)
- [Concepts Mathématiques](#concepts-mathématiques)

---

## Installation

### Linux / macOS

```bash
# 1. Rendre le script exécutable
chmod +x setup.sh

# 2. Installer les dépendances
./setup.sh

# 3. Vérifier l'installation
python3 test_project.py
```

### Windows

```bash
pip install -r requirements.txt
python test_project.py
```

### GPU — optionnel (A100 / CUDA)

```bash
# Vérifier votre version CUDA
nvcc --version

# Installer CuPy pour votre version CUDA (ex: CUDA 12.x)
pip install cupy-cuda12x
```

---

## Utilisation

### 1. Entraîner le modèle

```bash
# Entraînement standard (CPU, dataset complet)
python3 TP_Paroles_Code_Complet.py

# Entraînement rapide sur un sous-ensemble (debug)
MAX_SAMPLES=2000 python3 TP_Paroles_Code_Complet.py

# Entraînement sur GPU (A100)
USE_GPU=1 python3 TP_Paroles_Code_Complet.py

# GPU + dataset réduit
USE_GPU=1 MAX_SAMPLES=5000 python3 TP_Paroles_Code_Complet.py
```

### 2. Générer des paroles

```bash
# Générer une chanson rock
python3 infer_lyrics.py --genre rock

# 3 exemples pop avec plus de créativité
python3 infer_lyrics.py --genre pop --samples 3 --temperature 1.2

# Lister les genres disponibles
python3 infer_lyrics.py --list-genres

# Utiliser un checkpoint spécifique
python3 infer_lyrics.py --model outputs/checkpoint_epoch5.pkl --genre rock
```

| Option | Défaut | Description |
|--------|--------|-------------|
| `--genre` | — | Genre musical (ex: `rock`, `pop`, `rap`) |
| `--samples` | `1` | Nombre de générations |
| `--length` | `50` | Nombre maximum de mots |
| `--temperature` | `0.8` | Créativité (`< 1` = déterministe, `> 1` = aléatoire) |
| `--list-genres` | — | Afficher les genres disponibles |
| `--model` | `outputs/lyrics_model.pkl` | Chemin vers un modèle personnalisé |

---

## Paramètres & Configuration

Tous les hyperparamètres sont dans `TP_Paroles_Code_Complet.py` et peuvent être surchargés via variables d'environnement.

### Taille du dataset

```python
MAX_SAMPLES = None   # None = tout le dataset (~18 000 chansons)
                     # 2000 = debug rapide (~2 min)
                     # 5000 = bon compromis (~5 min)
                     # 10000 = haute qualité (~10 min)
```

Sans modifier le code :

```bash
MAX_SAMPLES=3000 python3 TP_Paroles_Code_Complet.py
```

### Hyperparamètres d'entraînement

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `MAX_SAMPLES` | `None` | Entrées du dataset à utiliser (`None` = tout) |
| `TOP_GENRES` | `5` | Nombre de genres à retenir |
| `MAX_VOCAB_SIZE` | `20 000` | Taille maximale du vocabulaire |
| `MIN_FREQ` | `2` | Fréquence minimale d'un mot pour être inclus |
| `SEQ_LEN` | `10` | Longueur de la séquence d'entrée (tokens) |
| `NUM_EPOCHS` | `10` | Nombre d'époques |
| `BATCH_SIZE` | `128` | Taille du batch |
| `LEARNING_RATE` | `0.001` | Taux d'apprentissage |
| `EARLY_STOP_PATIENCE` | `3` | Arrêt anticipé si val_loss stagne |
| `CHECKPOINT_EVERY` | `1` | Sauvegarde checkpoint toutes les N époques |
| `GRAD_CLIP` | `5.0` | Seuil de gradient clipping |

### Variables d'environnement

| Variable | Valeurs possibles | Défaut |
|----------|-------------------|--------|
| `USE_GPU` | `1`, `true`, `auto`, `0` | `auto` |
| `MAX_SAMPLES` | entier positif | non défini (= tout) |

---

## Architecture du Modèle

### Pipeline complet

```
Dataset Spotify (18 454 chansons)
        │
        ▼
  Nettoyage & Tokenisation
  (lowercase, regex, <NEW_LINE>)
        │
        ▼
  Vocabulaire (cap: 20 000 tokens, freq ≥ 2)
  Tokens spéciaux : <PAD>  <UNK>  <BOS>  <EOS>
        │
        ▼
  Paires (séquence → mot cible) + encodage genre
        │
        ▼
  ┌─────────────────────────────────┐
  │          MODÈLE NN              │
  │                                 │
  │  Word Embeddings  (SEQ_LEN×16D) │
  │  Genre Embedding  (1×16D)       │
  │         ↓ concatenation         │
  │  Dense 1 : 176 → 32  (ReLU)    │
  │  Dense 2 : 32 → vocab (Softmax) │
  └─────────────────────────────────┘
        │
        ▼
  Probabilités du prochain mot
```

### Concepts du cours appliqués

| Concept | Application dans le projet |
|---------|---------------------------|
| **Embeddings** | Représentation vectorielle des mots et genres |
| **Classification** | Prédiction du mot suivant parmi ~20k classes |
| **Cross-Entropy** | Fonction de coût |
| **Backpropagation** | Calcul des gradients par dérivation en chaîne |
| **Gradient Descent** | Mise à jour des poids |
| **Gradient Clipping** | Stabilisation (`GRAD_CLIP = 5.0`) |
| **Early Stopping** | Arrêt si val_loss stagne (`patience = 3`) |

---

## Métriques & Suivi

Affichage à chaque batch (500 itérations) :

```
Batch 500/3210 | Loss: 6.4321 | Acc: 0.0521 | Elapsed: 14.2s | ETA: 77s
```

Affichage à chaque époque :

```
Époque | Train Loss | Train Acc | Val Loss | Val Acc | Train PPL | Val PPL |   Time
     1 |   6.432100 |    0.0521 | 6.721000 |  0.0498 |  621.4000 | 827.200 |  92.3s ✓ best
     2 |   5.981200 |    0.0634 | 6.432000 |  0.0551 |  394.8000 | 621.400 |  88.7s ✓ best
     3 |   5.760400 |    0.0710 | 6.432100 |  0.0548 |  317.8000 | 621.500 |  89.1s (patience 1/3)
```

### Métriques disponibles

| Métrique | Formule | Interprétation |
|----------|---------|----------------|
| **Loss** | Cross-Entropy | Plus bas = meilleur |
| **Accuracy** | % de mots corrects | Plus haut = meilleur |
| **Perplexity** | exp(loss) | Plus bas = meilleur |

### Graphiques produits (`outputs/training_stats.png`)

1. **Loss** train vs validation par époque
2. **Perplexity** train vs validation par époque
3. **Distribution des genres** (bar chart annoté)
4. **Distribution des longueurs de séquences** (histogramme avec moyenne/médiane)

### Checkpoints automatiques

```
outputs/checkpoint_epoch1.pkl
outputs/checkpoint_epoch2.pkl
...
outputs/lyrics_model.pkl   ← modèle final
```

Si l'entraînement plante à l'époque 8, les 7 checkpoints précédents sont préservés.

---

## Support GPU

Le script détecte automatiquement un GPU CUDA via CuPy :

```bash
# Forcer GPU
USE_GPU=1 python3 TP_Paroles_Code_Complet.py
# → ✓ Backend: GPU (CuPy)

# Forcer CPU
USE_GPU=0 python3 TP_Paroles_Code_Complet.py
# → ✓ Backend: CPU (NumPy)

# Auto-détection (défaut)
python3 TP_Paroles_Code_Complet.py
# → ✓ Backend: GPU (CuPy)   si GPU disponible
# → ✓ Backend: CPU (NumPy)  sinon (silencieux)
```

Si CuPy n'est pas installé ou si aucun GPU n'est détecté, le script continue sur CPU sans interruption.

---

## Structure des Fichiers

```
.
├── TP_Paroles_Code_Complet.py    # Script principal (entraînement, 11 sections)
├── infer_lyrics.py               # Génération de paroles (CLI)
├── spotify_songs.csv             # Dataset (~18 454 chansons, ~42 MB)
├── outputs/                      # Créé automatiquement
│   ├── lyrics_model.pkl          # Modèle final
│   ├── training_stats.png        # Graphiques d'entraînement
│   ├── checkpoint_epoch1.pkl     # Checkpoints
│   └── checkpoint_epochN.pkl
├── requirements.txt              # Dépendances Python
├── setup.sh                      # Installation automatique (Linux/macOS)
├── test_project.py               # Tests de vérification
└── README.md                     # Ce fichier
```

### Sections du script principal

| # | Section | Description |
|---|---------|-------------|
| 1 | Chargement | Lecture CSV, nettoyage, filtrage, `MAX_SAMPLES` |
| 2 | Genres | Sélection des `TOP_GENRES` genres dominants |
| 3 | Prétraitement | Tokenisation, lowercase, regex |
| 4 | Vocabulaire | Construction avec fréquence et cap |
| 5 | Encodage | Séquences numériques + encodage genre |
| 6 | Train/Val split | Génération des paires + progression + monitoring RAM |
| 7 | Modèle | Définition de `LyricsGenerationModel` |
| 8 | Entraînement | NaN guard, grad clipping, early stopping, checkpoints |
| 9 | Génération | Tests de génération par genre |
| 10 | Visualisations | 4 graphiques matplotlib sauvegardés |
| 11 | Sauvegarde | Export `outputs/lyrics_model.pkl` avec fallback |

---

## Dépannage

### Entraînement trop lent

```bash
MAX_SAMPLES=2000 python3 TP_Paroles_Code_Complet.py   # dataset réduit
USE_GPU=1 python3 TP_Paroles_Code_Complet.py           # GPU
```

Ou dans le script :
```python
NUM_EPOCHS = 3
BATCH_SIZE = 256
```

### Manque de mémoire RAM

```bash
MAX_SAMPLES=3000 python3 TP_Paroles_Code_Complet.py
```

Ou :
```python
MAX_VOCAB_SIZE = 10000
BATCH_SIZE = 64
MIN_FREQ = 5
```

### Loss NaN / entraînement instable

Le script skippe automatiquement les batchs NaN (jusqu'à 5 consécutifs). Si le problème persiste :

```python
LEARNING_RATE = 0.0001
GRAD_CLIP = 1.0
```

### Reprendre depuis un checkpoint

```bash
python3 infer_lyrics.py --model outputs/checkpoint_epoch5.pkl --genre rock
```

### Dataset non trouvé

`spotify_songs.csv` doit être dans le même répertoire que les scripts. Les chemins sont calculés dynamiquement :

```python
def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))
```

Le script fonctionne depuis n'importe quel répertoire courant.

### plt.show() plante (serveur sans affichage)

Le script capture automatiquement cette erreur et continue. Les graphiques sont toujours sauvegardés dans `outputs/training_stats.png`.

---

## Concepts Mathématiques

### Embeddings

$$\text{embed}(w) \in \mathbb{R}^{16}, \quad \text{embed}(g) \in \mathbb{R}^{16}$$

### Forward Pass

$$z_1 = X W_1 + b_1, \quad a_1 = \text{ReLU}(z_1) = \max(0,\, z_1)$$
$$z_2 = a_1 W_2 + b_2$$

### Softmax

$$\hat{y}_i = \frac{e^{z_i}}{\displaystyle\sum_j e^{z_j}}$$

### Fonction de coût (Cross-Entropy)

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log \hat{y}_{i,\, t_i}$$

### Perplexité

$$\text{PPL} = e^{\mathcal{L}}$$

### Backpropagation & Gradient Descent

$$\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial a} \cdot \frac{\partial a}{\partial W}, \qquad W \leftarrow W - \alpha\, \nabla_W \mathcal{L}$$

### Gradient Clipping

$$\text{si} \;\|W\| > \delta :\quad W \leftarrow W \cdot \frac{\delta}{\|W\| + \varepsilon}$$

---

## Dépendances

| Package | Version min | Usage |
|---------|-------------|-------|
| `numpy` | ≥ 1.19.0 | Calcul numérique (backend CPU) |
| `pandas` | ≥ 1.1.0 | Chargement et manipulation du dataset |
| `scikit-learn` | ≥ 0.23.0 | `train_test_split`, `LabelEncoder` |
| `matplotlib` | ≥ 3.3.0 | Graphiques et visualisations |
| `cupy` | optionnel | Backend GPU (CUDA) |
| `psutil` | optionnel | Monitoring mémoire RAM |

```bash
pip install -r requirements.txt

# Optionnel
pip install psutil cupy-cuda12x
```

---

## Résultats Attendus

| MAX_SAMPLES | Époques | Durée CPU | Train Loss | Val Perplexity |
|-------------|---------|-----------|------------|----------------|
| 2 000 | 10 | ~2 min | ~5.5 | ~250 |
| 5 000 | 10 | ~5 min | ~5.0 | ~150 |
| 18 000 (complet) | 10 | ~15 min | ~4.5 | ~90 |

Exemple de génération :

```
ROCK:
  the day when you want to be with me i know the way
  we used to fall together in the fire and the rain

POP:
  love is in the air tonight feel the magic all around
  baby you and i were made to dance under the lights
```

---

**Bon entraînement ! 🎤🎵**
