# 🎵 Génération de Paroles de Chanson par Apprentissage Automatique
## Présentation du Travail Pratique — Deep Learning

**Réalisé par :** Abdellah Saaid & Chihab Eddine Ethamry  
**Date :** Mai 2026  

---

## 📋 Table des Matières

1. [Introduction & Objectifs](#1-introduction--objectifs)
2. [Dataset Spotify](#2-dataset-spotify)
3. [Architecture 1 — Modèle Feed-Forward Custom](#3-architecture-1--modèle-feed-forward-custom)
4. [Pipeline d'Entraînement](#4-pipeline-dentraînement)
5. [Résultats & Métriques](#5-résultats--métriques)
6. [Architecture 2 — GPT-2 Fine-tuné (HuggingFace)](#6-architecture-2--gpt-2-fine-tuné-huggingface)
7. [Comparaison des Deux Approches](#7-comparaison-des-deux-approches)
8. [Exemples de Paroles Générées](#8-exemples-de-paroles-générées)
9. [Difficultés Rencontrées & Solutions Apportées](#9-difficultés-rencontrées--solutions-apportées)
10. [Conclusion & Perspectives](#10-conclusion--perspectives)

---

## 1. Introduction & Objectifs

### Problématique
> Comment entraîner un modèle de langage capable de **générer des paroles de chansons** cohérentes, différenciées par genre musical, à partir de données brutes Spotify ?

### Objectifs du TP
- Implémenter **from scratch** un réseau de neurones pour la génération de texte (sans frameworks haut niveau)
- Comprendre les mécanismes fondamentaux : embeddings, forward pass, backpropagation, gradient descent
- Comparer notre implémentation custom avec un modèle pré-entraîné de l'état de l'art (**GPT-2**)
- Mettre en œuvre les bonnes pratiques : séparation train/val, early stopping, configuration centralisée

### Technologies Utilisées
| Composant | Technologie |
|---|---|
| Langage | Python 3.10+ |
| Backend calcul | NumPy / CuPy (GPU auto-détecté) |
| Modèle pré-entraîné | HuggingFace Transformers — GPT-2 |
| Dataset | Spotify Songs CSV (~18 000 chansons) |
| Gestion config | JSON + variables d'environnement |

---

## 2. Dataset Spotify

### Description
- **Source :** Kaggle — Spotify Song Attributes + Lyrics
- **Taille initiale :** ~18 000 entrées
- **Après nettoyage :** ~8 000 chansons en anglais avec paroles valides
- **Utilisé pour l'entraînement :** 2 845 chansons (filtre langue + genres)

### Genres Retenus (Top 5)
| Genre | Nb. Chansons | Caractéristiques |
|---|---|---|
| EDM | ~600 | Répétitif, énergie haute, peu de narration |
| Pop | ~700 | Structures couplet/refrain, thèmes d'amour |
| R&B | ~550 | Riches en émotions, rythme soul |
| Rap | ~500 | Flux rapide, argot, rimes complexes |
| Rock | ~495 | Intensité, rébellion, narration |

### Prétraitement du Texte
```
Texte brut  →  lowercase  →  suppression caractères spéciaux
           →  tokenisation (split par espaces)
           →  encodage en indices vocabulaire
           →  séquences de contexte (fenêtre glissante)
```

### Construction du Vocabulaire
- Fréquence minimale : `MIN_FREQ = 3` (mots rares exclus)
- Taille maximale : `MAX_VOCAB_SIZE = 12 000`
- Tokens spéciaux : `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`
- **Vocabulaire final : 12 004 tokens**

### Création des Paires (X, y)
Pour chaque chanson, on génère des paires de type **next-word prediction** :

```
Chanson : [<BOS>, "i", "love", "you", <EOS>]

Paires générées :
  X = [<BOS>]              → y = "i"
  X = [<BOS>, "i"]         → y = "love"
  X = [<BOS>, "i", "love"] → y = "you"
```

**Total : 1 087 570 paires** (856 744 train / 230 826 val)

---

## 3. Architecture 1 — Modèle Feed-Forward Custom

### Vue d'Ensemble
```
Entrée X [batch, SEQ_LEN=20]
        │
        ▼
┌──────────────────────────────┐
│  Word Embedding              │   vocab_size × EMBED_DIM (64)
│  [batch, 20, 64]             │
└──────────┬───────────────────┘
           │ flatten → [batch, 1280]
           │
┌──────────▼──────────────┐
│  Genre Embedding         │   num_genres × EMBED_DIM (64)
│  [batch, 64]             │
└──────────┬───────────────┘
           │
           ▼ concat → [batch, 1344]
┌──────────────────────────┐
│  Couche Cachée (W1, b1)  │   1344 → 256 neurones
│  + ReLU + Dropout (0.15) │
└──────────┬───────────────┘
           │ [batch, 256]
           ▼
┌──────────────────────────┐
│  Couche Sortie (W2, b2)  │   256 → vocab_size (12 004)
│  + Softmax               │
└──────────────────────────┘
           │
           ▼
    Distribution P(mot suivant | contexte, genre)
```

### Paramètres du Modèle
| Composant | Dimensions | Nb. Paramètres |
|---|---|---|
| `word_embedding` | 12 004 × 64 | 768 256 |
| `genre_embedding` | 5 × 64 | 320 |
| `W1` | 1 344 × 256 | 344 064 |
| `b1` | 1 × 256 | 256 |
| `W2` | 256 × 12 004 | 3 073 024 |
| `b2` | 1 × 12 004 | 12 004 |
| **Total** | | **~4,2 M paramètres** |

### Initialisation des Poids
Initialisation critique pour la convergence :

```python
# He initialization — adapté aux couches ReLU
W1 = randn(input_dim, hidden_dim) * sqrt(2 / input_dim)

# Xavier initialization — adapté à la sortie linéaire
W2 = randn(hidden_dim, vocab_size) * sqrt(1 / hidden_dim)

# Embeddings — variance modérée
word_embedding = randn(vocab_size, embed_dim) * 0.1
```

> ⚠️ **Bug critique détecté et corrigé :** l'initialisation initiale `* 0.01` sur tous les poids créait des gradients de l'ordre de 10⁻¹¹ — le modèle n'apprenait rien.

---

## 4. Pipeline d'Entraînement

### Hyperparamètres (Configuration Finale)
```json
{
  "NUM_EPOCHS":      15,
  "BATCH_SIZE":      128,
  "LEARNING_RATE":   0.01,
  "LR_DECAY":        0.95,
  "DROPOUT_RATE":    0.15,
  "LABEL_SMOOTHING": 0.05,
  "EMBEDDING_DIM":   64,
  "HIDDEN_DIM":      256,
  "SEQ_LEN":         20,
  "MIN_FREQ":        3,
  "MAX_VOCAB_SIZE":  12000
}
```

### Fonction de Perte — Cross-Entropie avec Label Smoothing
$$\mathcal{L} = -\sum_{c=1}^{V} \tilde{y}_c \cdot \log(\hat{p}_c)$$

Avec label smoothing ($\varepsilon = 0.05$) :
$$\tilde{y}_c = \begin{cases} 1 - \varepsilon & \text{si } c = \text{cible} \\ \frac{\varepsilon}{V-1} & \text{sinon} \end{cases}$$

> Le label smoothing évite la sur-confiance et améliore la généralisation.

### Backpropagation (Implémentée à la Main)
```
dL/dW2 = a1ᵀ · d_logits
dL/dW1 = combined ᵀ · (d_a1 ⊙ relu_mask)
dL/d_embedding = scatter_add(d_word_embed → word_embedding)
```

> ⚠️ **Bug critique corrigé :** les embeddings ne recevaient jamais de gradient — seules les couches denses W1/W2 apprenaient.

### Gradient Clipping
Pour éviter l'explosion des gradients :
$$\text{grad} \leftarrow \text{grad} \cdot \min\left(1, \frac{5.0}{\|\text{grad}\|}\right)$$

### Séparation Train / Validation (Anti-fuite)
Séparation **au niveau chanson** (song-level split) avant la création des paires :
```
songs → train_songs (80%) | val_songs (20%)
         ↓                        ↓
    paires train             paires val
```

> Sans cette précaution, des séquences du même morceau pourraient apparaître dans les deux ensembles → fuite de données → surestimation des performances.

### Early Stopping
Arrêt automatique si la val_loss n'améliore pas pendant `patience = 3` époques consécutives.

### Décroissance du Learning Rate
$$\text{lr}_{\text{epoch}} = \max(\text{lr}_0 \times 0.95^{\text{epoch}}, \text{lr}_{\min})$$

---

## 5. Résultats & Métriques

### Évolution de l'Entraînement

| Époque | Train Loss | Val Loss | Val PPL | Meilleur |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 8.050 | 6.852 | 946 | ✓ |
| 2 | 6.920 | 6.605 | 739 | ✓ |
| 3 | 6.778 | 6.513 | 674 | ✓ |
| 5 | 6.58 | 6.35 | 574 | ✓ |
| 10 | 6.43 | 6.22 | 500 | ✓ |
| **15** | **6.255** | **6.071** | **433** | **✓** |

> **Perplexité** = e^(val_loss) : nombre moyen de mots entre lesquels le modèle hésite.  
> Une PPL de 433 signifie que le modèle choisit parmi ~433 mots plausibles à chaque étape — contre ~10 665 en début d'entraînement (**amélioration ×23**).

### Progrès Global
| Phase | Val PPL | Gain |
|---|---|---|
| Avant corrections (modèle cassé) | 10 665 | — |
| Après fix initialisation + backprop | 946 (époque 1) | ×11 |
| Fin entraînement (époque 15) | **433** | **×24** |

### Métriques de Qualité de Génération
| Genre | Tokens | Ratio Unique | Rép. Bigrams | Rép. Trigrams |
|---|:---:|:---:|:---:|:---:|
| EDM | 40 | 0.75 | 0.00 | 0.00 |
| Pop | 40 | 0.72 | 0.05 | 0.00 |
| R&B | 40 | 0.65 | 0.03 | 0.00 |
| Rap | 40 | 0.68 | 0.00 | 0.00 |
| Rock | 25 | 0.72 | 0.00 | 0.00 |

> Ratio unique > 0.65 = bonne diversité lexicale. Répétition de bigrams < 0.05 = pas de boucles.

---

## 6. Architecture 2 — GPT-2 Fine-tuné (HuggingFace)

### Qu'est-ce que GPT-2 ?
GPT-2 (Generative Pre-trained Transformer 2) est un modèle de langage développé par **OpenAI** en 2019. Il repose sur l'architecture **Transformer** avec mécanisme d'**attention multi-têtes**.

```
Architecture GPT-2 (version small) :
  ┌─────────────────────────────────┐
  │  Token Embedding (50 257 vocab) │ dim = 768
  │  + Position Embedding           │
  └────────────────┬────────────────┘
                   │
          ┌────────┴────────┐
          │  × 12 couches   │
          │  ┌────────────┐ │
          │  │ Self-Attn  │ │  12 têtes, dim_head=64
          │  │ (Masked)   │ │
          │  ├────────────┤ │
          │  │ Layer Norm │ │
          │  ├────────────┤ │
          │  │ Feed-Fwd   │ │  768 → 3072 → 768
          │  │ (GELU)     │ │
          │  └────────────┘ │
          └────────┬────────┘
                   │
          ┌────────▼────────┐
          │  LM Head        │  768 → 50 257 (vocab)
          └─────────────────┘
```

### Différences Clés vs Notre Modèle
| Aspect | Feed-Forward Custom | GPT-2 Fine-tuné |
|---|---|---|
| Paramètres | ~4.2 M | **117 M** |
| Architecture | FC + ReLU | Transformer + Self-Attention |
| Contexte (mémoire) | 20 tokens (fenêtre fixe) | **1 024 tokens** |
| Pré-entraînement | Aucun | 40 GB de texte web (WebText) |
| Temps fine-tuning | N/A | ~5–10 min (3 époques, GPU) |
| Vocabulaire | 12 004 (construit) | 50 257 (BPE pré-défini) |
| Mécanisme clé | Embedding + Dense | **Self-Attention** |

### Pourquoi le Self-Attention est Supérieur
GPT-2 peut relier **n'importe quels mots dans le contexte**, pas juste les voisins immédiats :

```
"I never gonna give you up, never gonna let you [?]"
       ↑___________________________________|
         GPT-2 voit la relation "give up / let down"
         Notre modèle ne voit que les 20 derniers tokens
```

### Fine-tuning avec HuggingFace
```python
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments

model = GPT2LMHeadModel.from_pretrained('gpt2')

training_args = TrainingArguments(
    num_train_epochs=3,
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    fp16=True,               # mixed precision
    weight_decay=0.01,
    warmup_steps=50,
)

trainer = Trainer(model=model, args=training_args, ...)
trainer.train()
```

**3 lignes suffisent** là où notre implémentation custom nécessite >400 lignes de code.

---

## 7. Comparaison des Deux Approches

### Tableau Comparatif Complet

| Critère | Feed-Forward Custom | GPT-2 Fine-tuné |
|---|:---:|:---:|
| **Val Perplexity** | 433 | ~50–80 (estimé) |
| **Paramètres** | 4.2 M | 117 M |
| **Temps d'entraînement** | 7.6 min (15 epochs) | ~5–10 min (3 epochs) |
| **Cohérence grammaticale** | Faible | **Bonne** |
| **Diversité lexicale** | Moyenne | **Élevée** |
| **Mémoire du contexte** | 20 tokens | 1 024 tokens |
| **Contrôle du genre** | ✓ (embedding dédié) | Partiel (prompt) |
| **Compréhension du code** | ✓ Pédagogique | ✗ Boîte noire |
| **Dépendances** | NumPy / CuPy | PyTorch + HuggingFace |
| **Installation** | Aucune | `pip install transformers` |

### Courbe d'Apprentissage

```
Val Loss
  9.3 │●  ← Modèle cassé (avant fixes)
      │
  8.0 │   ●  ← Epoch 1 (après fixes)
      │
  7.0 │      ●  ← Epoch 2
      │
  6.5 │         ●  ← Epoch 5
      │
  6.1 │                    ●  ← Epoch 15 (final)
      │
  ?   │ ════════════════════════ GPT-2 (estimé ~4.5)
      └──────────────────────────────── Epochs
```

### Pourquoi Utiliser l'Approche Custom ?

1. **Valeur pédagogique** : implémentation complète de la backpropagation, des embeddings, du gradient clipping
2. **Debugging profond** : chaque bug trouvé (PAD masking, initialisation, gradient embedding) a renforcé la compréhension
3. **Contrôle total** : chaque composant est transparent et modifiable
4. **Sans dépendances lourdes** : fonctionne avec NumPy uniquement

### Pourquoi GPT-2 Gagne en Qualité ?

1. **Attention multi-têtes** : capture les relations à longue distance
2. **Pré-entraînement massif** : déjà "au courant" de l'anglais
3. **BPE Tokenizer** : gère les mots rares, les suffixes, les contractions
4. **117 M paramètres** : capacité de représentation sans commune mesure

---

## 8. Exemples de Paroles Générées

### Modèle Feed-Forward Custom — Genre ROCK

**Exemple 1 :**
> *it like a and do the you down all get take is never me no got i'm of it's your life ain't be so more to love but if i know you, for this baby what with my time oh i think too say that when we are away don't tell me from the eyes i'll need on me, let*

**Exemple 2 :**
> *i all yeah get to your the in of been and on my you but a it be what there's you, is our love it's just at me not for this so long can i'm like to keep if oh, from they go right up and you're through it look with me when we will are an 'cause i want*

### Analyse de la Qualité
| Aspect | Observation |
|---|---|
| ✅ Mots reconnaissables | Vocabulaire anglais courant |
| ✅ Pas de boucles | Pénalités logit-space efficaces |
| ✅ Diversité lexicale | Ratio unique 0.68–0.75 |
| ⚠️ Grammaire | Faible — pas de mémoire syntaxique |
| ⚠️ Cohérence narrative | Limitée — fenêtre de 20 tokens |
| ⚠️ Tokens `<UNK>` | Quelques mots hors vocabulaire |

### Inférence — Stratégie d'Échantillonnage

```python
# Pipeline de décodage (logit space)
logits[PAD] = logits[BOS] = logits[UNK] = -∞    # bloquer tokens spéciaux
logits[tok] -= log(repeat_penalty)                # pénalité contexte récent
logits[tok] -= presence_penalty                   # pénalité présence globale
logits[tok] -= frequency_penalty × count          # pénalité fréquence
```

Puis → **Top-K (K=60)** → **Top-P nucleus (p=0.92)** → **sampling**

---

## 9. Difficultés Rencontrées & Solutions Apportées

### Bug 1 — Initialisation des Poids (`* 0.01`)
| | Détail |
|---|---|
| **Symptôme** | Loss stuck à 9.29, acc=0.000 après 2 époques |
| **Cause** | `W1 * 0.01 × W2 * 0.01` → gradients ~10⁻¹¹ |
| **Solution** | He init pour W1, Xavier pour W2 |
| **Résultat** | Loss tombe à 6.85 dès l'époque 1 |

### Bug 2 — Embeddings Sans Gradient
| | Détail |
|---|---|
| **Symptôme** | Seules les couches denses apprenaient |
| **Cause** | `word_embedding` et `genre_embedding` non mis à jour dans `backward()` |
| **Solution** | `np.add.at(grad_word_embedding, token_idx, token_grads)` |

### Bug 3 — PAD Tokens dans les Gradients
| | Détail |
|---|---|
| **Symptôme** | Loss oscillait, direction de gradient corrompue |
| **Cause** | `d_logits /= batch_size` incluait les lignes PAD |
| **Solution** | `d_logits[~valid_mask] = 0.0` ; division par `valid_count` uniquement |

### Bug 4 — Pénalités d'Inférence Inefficaces
| | Détail |
|---|---|
| **Symptôme** | "you / i / the / a" dominaient toutes les générations |
| **Cause** | Pénalités appliquées en **espace probabilité** (`p /= 1.5`) → impact minimal |
| **Solution** | Pénalités en **espace logit** (`logit -= log(penalty)`) → impact exponentiel |

### Bug 5 — Fuite Train/Val
| | Détail |
|---|---|
| **Symptôme** | Métriques val trop optimistes |
| **Cause** | Split fait sur les paires, pas sur les chansons |
| **Solution** | Song-level split avant `build_pairs()` |

---

## 10. Conclusion & Perspectives

### Ce que Nous Avons Appris
- **Backpropagation from scratch** : chain rule, masking PAD, scatter-add pour embeddings
- **Importance de l'initialisation** : He/Xavier vs naïf
- **Espace logit vs probabilité** : pourquoi les pénalités doivent être appliquées avant softmax
- **Pipelines ML robustes** : split song-level, checkpoints, early stopping, config centralisée
- **Comparaison architectures** : Feed-Forward vs Transformer

### Limites du Modèle Custom
- Pas de mémoire à long terme (fenêtre de 20 tokens seulement)
- Architecture trop simple pour la syntaxe complexe
- Val PPL = 433 → le modèle hésite encore entre ~433 mots

### Perspectives d'Amélioration
| Amélioration | Gain Estimé | Complexité |
|---|---|---|
| Remplacer Feed-Forward par GRU/LSTM | PPL ~200 | Moyenne |
| Augmenter HIDDEN_DIM à 512 | PPL ~380 | Faible |
| GPT-2 fine-tuné complet | PPL ~50–80 | Faible (HuggingFace) |
| Entraîner sur dataset complet (18k songs) | PPL ~300 | Faible |

### GPT-2 — La Comparaison Qui Illustre le Fossé

Notre modèle custom avec 4.2M paramètres et une architecture simple montre les **limites fondamentales** d'une approche sans attention. GPT-2, avec ses 117M paramètres et 12 couches Transformer, produit des textes beaucoup plus fluides grâce à sa capacité à **modéliser les dépendances à longue distance**.

> Cette comparaison illustre parfaitement pourquoi les Transformers ont révolutionné le NLP depuis 2017 (Vaswani et al., "Attention is All You Need").

---

## 📁 Structure du Projet

```
TP_Paroles_Code_Complet/
├── TP_Paroles_Code_Complet.py   ← Modèle custom (entraînement)
├── infer_lyrics.py              ← Inférence avancée (CLI)
├── gpt2_finetune.py             ← Fine-tuning GPT-2 (comparaison)
├── config/
│   └── train_config.json        ← Hyperparamètres centralisés
├── outputs/
│   ├── lyrics_model_best.pkl    ← Meilleur modèle custom
│   ├── run_metadata.json        ← Métriques et configuration du run
│   ├── training_stats.png       ← Courbes loss/perplexity
│   └── gpt2_finetune/           ← Modèle GPT-2 fine-tuné
├── spotify_songs.csv            ← Dataset
└── PRESENTATION_TP.md           ← Ce document
```

## 🚀 Commandes de Démonstration

```bash
# Entraînement du modèle custom
USE_GPU=1 python3 TP_Paroles_Code_Complet.py

# Génération avec le modèle custom
python3 infer_lyrics.py \
  --model outputs/lyrics_model_best.pkl \
  --genre rock --samples 3 \
  --temperature 1.0 --top-k 60 --top-p 0.92 \
  --repeat-penalty 2.5 --presence-penalty 2.0 \
  --frequency-penalty 0.5 --no-repeat-ngram 3 --length 60

# Fine-tuning GPT-2 + génération
python3 gpt2_finetune.py --genre pop --samples 3 --epochs 3

# Génération GPT-2 seule (après fine-tuning)
python3 gpt2_finetune.py --only-gen --genre pop
```

---

*Présentation réalisée dans le cadre du module Deep Learning / NLP*  
*Abdellah Saaid — Chihab Eddine Ethamry — Mai 2026*
