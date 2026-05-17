# 🎵 Génération de Paroles de Chanson par Apprentissage Automatique
## Présentation du Travail Pratique — Deep Learning

**Réalisé par :** Abdellah Saaid & Chihab Eddine Ethamry  
**Date :** Mai 2026  

---

## 📋 Table des Matières

1. [Introduction & Objectifs](#1-introduction--objectifs)
2. [Pourquoi ces Choix Architecturaux ?](#2-pourquoi-ces-choix-architecturaux-)
3. [Dataset Spotify](#3-dataset-spotify)
4. [Architecture 1 — Modèle Feed-Forward Custom](#4-architecture-1--modèle-feed-forward-custom)
5. [Pipeline d'Entraînement](#5-pipeline-dentraînement)
6. [Résultats & Métriques](#6-résultats--métriques)
7. [Architecture 2 — GPT-2 Fine-tuné (HuggingFace)](#7-architecture-2--gpt-2-fine-tuné-huggingface)
8. [Comparaison des Deux Approches](#8-comparaison-des-deux-approches)
9. [Exemples de Paroles Générées](#9-exemples-de-paroles-générées)
10. [Difficultés Rencontrées & Solutions Apportées](#10-difficultés-rencontrées--solutions-apportées)
11. [Conclusion & Perspectives](#11-conclusion--perspectives)

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

## 2. Pourquoi ces Choix Architecturaux ?

### Pourquoi un Modèle Feed-Forward Custom ?

Notre premier modèle est un réseau de neurones entièrement implémenté **from scratch** avec NumPy — sans PyTorch, sans TensorFlow, sans aucun framework de haut niveau.

**Raisons pédagogiques :**

| Raison | Explication |
|---|---|
| 🔬 Comprendre la backprop | Implémenter `dL/dW2`, `dL/dW1`, `scatter_add` à la main force une compréhension profonde de la chaîne de dérivation |
| 🐛 Debugging profond | Chaque bug trouvé (initialisation, PAD masking, gradient embedding) a renforcé notre compréhension des mécanismes internes |
| 🎛️ Contrôle total | Chaque composant est transparent, modifiable, instrospectable — aucune "magie" cachée |
| ⚡ Sans dépendances lourdes | Fonctionne avec NumPy uniquement (~1 MB), portable et reproductible |
| 🧠 Architecture explicite | La structure Feed-Forward avec embeddings de genre représente la **brique de base** du NLP moderne |

> Ce modèle ne vise pas à battre GPT-2. Il vise à **comprendre** comment un modèle de langage fonctionne réellement, de l'embedding jusqu'au softmax.

---

### Pourquoi GPT-2 comme Référence ?

GPT-2 (OpenAI, 2019) est le choix naturel pour une **comparaison de référence** dans ce contexte.

**Raisons du choix :**

| Raison | Explication |
|---|---|
| 📏 Standard de l'industrie | GPT-2 est l'un des modèles de langage les plus étudiés et documentés — idéal comme baseline |
| 🤗 Accessible via HuggingFace | `GPT2LMHeadModel.from_pretrained('gpt2')` — 3 lignes de code pour charger 124.4 M de paramètres pré-entraînés |
| ⚡ Fine-tuning ultra-rapide | 58.3 secondes seulement pour 3 époques sur GPU — démontre la puissance du transfer learning |
| 🏗️ Architecture Transformer | Mécanisme d'attention multi-têtes — représente l'évolution fondamentale par rapport au Feed-Forward |
| 📚 Pré-entraîné sur 40 GB | Déjà capable de produire de l'anglais grammaticalement correct avant même le fine-tuning |
| 🔍 Contraste clair | La différence FFN → Transformer illustre parfaitement pourquoi "Attention is All You Need" (Vaswani et al., 2017) a révolutionné le NLP |

---

### Pourquoi Comparer les Deux ?

La valeur de ce TP réside précisément dans cette **dualité** :

```
  Modèle Custom           GPT-2
  ─────────────           ──────
  4.2 M params       vs   124.4 M params
  from scratch       vs   pre-trained
  NumPy uniquement   vs   PyTorch + HuggingFace
  PPL = 220.81       vs   PPL = 12.52  (×17.6 meilleur)
  39.2 min           vs   58.3 secondes
  pédagogique ✓      vs   performant ✓
```

> **Conclusion :** Le modèle custom nous apprend *comment* ça marche. GPT-2 nous montre *à quel point* ça peut être meilleur avec les bonnes architectures. Les deux sont indispensables pour ce TP.

---

## 3. Dataset Spotify

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

![Distribution des genres dans le dataset](graphs+logs/dataset_distribution.png)

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

**Total : 5 547 617 paires** (4 449 959 train / 1 097 658 val)

![Fréquence des mots dans le vocabulaire](graphs+logs/vocab_frequency.png)

---

## 4. Architecture 1 — Modèle Feed-Forward Custom

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

### Pourquoi avons-nous choisi cette architecture ?

| Critère | Justification |
|---|---|
| **Pédagogie avant tout** | Un réseau Feed-Forward est la structure minimale qui contient *tous* les concepts fondamentaux : embedding, multiplication matricielle, activation non-linéaire, softmax, backprop. Aucune abstraction cachée. |
| **Implémentation from scratch possible** | Contrairement aux RNN/LSTM ou aux Transformers, le FFN ne nécessite pas de boucle temporelle ni de mécanisme d'attention complexe — on peut écrire chaque dérivée à la main. |
| **Contrôle du genre par embedding dédié** | L'ajout d'un `genre_embedding` concaténé à l'entrée est simple et transparent : le modèle *apprend* la représentation de chaque genre sans supervision explicite. |
| **Référence de base (baseline)** | Un modèle simple offre une baseline honnête : toute amélioration future (LSTM, Transformer) est mesurable par rapport à ce point de départ. |
| **Rapidité de débogage** | Avec ~4.2 M paramètres et NumPy uniquement, chaque passe prend quelques ms — idéal pour itérer rapidement sur les bugs. |

> **Limite assumée :** le FFN n'a pas de mémoire séquentielle. Il traite les 20 tokens de contexte comme un sac de mots ordonnés, sans comprendre les dépendances à longue distance. C'est exactement ce que GPT-2 corrige.

---

## 5. Pipeline d'Entraînement

### Hyperparamètres (Configuration Finale) — Explication Détaillée

| Paramètre | Valeur | Rôle | Pourquoi cette valeur |
|---|:---:|---|---|
| `NUM_EPOCHS` | 15 | Nombre de passes complètes sur les données | Early stopping déclenché à 15 ; au-delà, la val_loss stagne. Augmenter risquerait le sur-apprentissage. |
| `BATCH_SIZE` | 128 | Nombre d'exemples par mise à jour des poids | Compromis entre stabilité du gradient (grand batch) et vitesse de convergence (petit batch). 128 est standard pour les modèles de langage légers. |
| `LEARNING_RATE` | 0.01 | Taille du pas de descente de gradient | Valeur initiale élevée pour une convergence rapide ; atténuée par `LR_DECAY` à chaque époque. |
| `LR_DECAY` | 0.95 | Facteur multiplicatif de décroissance du LR | $\text{lr}_{e} = 0.01 \times 0.95^{e}$. Permet un apprentissage rapide au début et une convergence fine en fin d'entraînement. |
| `DROPOUT_RATE` | 0.15 | Proportion de neurones désactivés aléatoirement | Régularisation légère (15 %) — suffisante pour éviter le sur-apprentissage sans trop degrader les performances. |
| `LABEL_SMOOTHING` | 0.05 | Redistribution de la masse de probabilité | Évite que le modèle attribue 100 % de confiance à un seul mot — améliore la généralisation et réduit la sur-confiance. |
| `EMBEDDING_DIM` | 64 | Dimension des vecteurs de représentation | Assez grand pour capturer des relations sémantiques, assez petit pour rester calculable sur CPU/GPU modeste. |
| `HIDDEN_DIM` | 256 | Nombre de neurones dans la couche cachée | Capacité de représentation de la couche intermédiaire. 4× l'embedding dim — ratio classique. |
| `SEQ_LEN` | 20 | Longueur de la fenêtre de contexte (tokens) | Contexte suffisant pour capturer des syntagmes (2-4 mots) ; fenêtre plus courte = plus de paires, plus vite. |
| `MIN_FREQ` | 3 | Fréquence minimale pour inclure un mot | Élimine les fautes de frappe et hapax qui n'apporteraient que du bruit. |
| `MAX_VOCAB_SIZE` | 12 000 | Taille maximale du vocabulaire | Vocabulaire suffisamment riche pour les 5 genres sans exploser la taille de la couche de sortie (W2). |

### Fonction de Perte — Pourquoi la Cross-Entropie ?

Nous utilisons la **cross-entropie catégorielle** car il s'agit d'un problème de **classification multi-classes** : à chaque étape, le modèle doit choisir un mot parmi les 12 004 du vocabulaire.

$$\mathcal{L}_{CE} = -\log(\hat{p}_{\text{cible}})$$

**Pourquoi pas d'autres fonctions de perte ?**

| Fonction | Pourquoi inadaptée |
|---|---|
| MSE (erreur quadratique) | Conçue pour des sorties continues — pénalise les erreurs de manière quadratique, non adaptée aux probabilités discrètes |
| Hinge loss (SVM) | Maximise une marge binaire — ne produit pas une distribution de probabilité sur le vocabulaire |
| KL-Divergence | Équivalente à la cross-entropie quand la cible est one-hot, mais moins intuitive à implémenter manuellement |
| **Cross-Entropie** ✓ | Mesure directement à quel point la distribution prédite s'éloigne de la vraie cible — se couple naturellement avec softmax |

**Lien avec la Perplexité :**
$$\text{PPL} = e^{\mathcal{L}_{CE}} \implies \text{minimiser } \mathcal{L} \equiv \text{minimiser PPL}$$

### Pourquoi le Label Smoothing ($\varepsilon = 0.05$) ?

Sans label smoothing, la cible est un vecteur **one-hot** (1 pour le bon mot, 0 pour tous les autres). Cela pousse le modèle à être sûr à 100 %, ce qui cause :
- **Sur-confiance** : le modèle ignore les nuances sémantiques entre mots proches
- **Gradients saturés** : le softmax sature, les mises à jour deviennent très faibles

Avec label smoothing ($\varepsilon = 0.05$) :
$$\tilde{y}_c = \begin{cases} 1 - \varepsilon = 0.95 & \text{si } c = \text{mot cible} \\ \frac{\varepsilon}{V-1} \approx 4 \times 10^{-6} & \text{pour les autres mots} \end{cases}$$

$$\mathcal{L}_{LS} = -\sum_{c=1}^{V} \tilde{y}_c \cdot \log(\hat{p}_c)$$

> 5 % de la masse de probabilité est redistribuée uniformément → le modèle reste ouvert à plusieurs mots plausibles, ce qui améliore la qualité de génération.

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

## 6. Résultats & Métriques

### Évolution de l'Entraînement

| Époque | Train Loss | Val Loss | Val PPL | Meilleur |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 6.853 | 6.225 | 505 | ✓ |
| 2 | 6.364 | 6.014 | 409 | ✓ |
| 3 | 6.202 | 5.875 | 356 | ✓ |
| 5 | 6.006 | 5.706 | 301 | ✓ |
| 10 | 5.772 | 5.499 | 244 | ✓ |
| **15** | **5.654** | **5.397** | **221** | **✓** |

![Courbes d'entraînement — Loss, Perplexité, Accuracy, Écart Val-Train](graphs+logs/training_curves.png)

> **Perplexité** = e^(val_loss) : nombre moyen de mots entre lesquels le modèle hésite.  
> Une PPL de 220.81 signifie que le modèle choisit parmi ~221 mots plausibles à chaque étape — contre ~12 004 (distribution uniforme initiale) (**amélioration ×54**).

### Progrès Global
| Phase | Val PPL | Gain |
|---|---|---|
| Avant corrections (modèle cassé) | 12 004 (uniforme) | — |
| Après fix initialisation + backprop | 505 (époque 1) | ×23.8 |
| Fin entraînement (époque 15) | **220.81** | **×54** |

### Métriques de Qualité de Génération
| Genre | Tokens | Ratio Unique | Rép. Bigrams | Rép. Trigrams |
|---|:---:|:---:|:---:|:---:|
| EDM | 40 | 0.60 | 0.15 | 0.08 |
| Pop | 40 | 0.60 | 0.05 | 0.00 |
| R&B | 40 | **0.33** | **0.41** | **0.18** |
| Rap | 40 | 0.47 | 0.28 | 0.16 |
| Rock | 40 | 0.53 | 0.08 | 0.00 |

> R&B montre plus de répétitions — ce genre musical utilise naturellement beaucoup de refrains répétés.

![Qualité de génération par genre](graphs+logs/generation_quality.png)

![Écart Validation − Train (surapprentissage)](graphs+logs/overfitting_gap.png)

---

## 7. Architecture 2 — GPT-2 Fine-tuné (HuggingFace)

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
| Paramètres | ~4.2 M | **124.4 M** |
| Architecture | FC + ReLU | Transformer + Self-Attention |
| Contexte (mémoire) | 20 tokens (fenêtre fixe) | **1 024 tokens** |
| Pré-entraînement | Aucun | 40 GB de texte web (WebText) |
| Temps fine-tuning | N/A | **58.3 secondes (3 époques, GPU)** |
| Vocabulaire | 12 004 (construit) | 50 257 (BPE pré-défini) |
| Mécanisme clé | Embedding + Dense | **Self-Attention** |

### Pourquoi avons-nous choisi GPT-2 ?

GPT-2 n'a pas été choisi au hasard. Chaque critère de sélection répond à un besoin précis du TP :

| Critère | Détail |
|---|---|
| **Comparaison architecturale claire** | GPT-2 représente l'étape suivante naturelle après un FFN : il ajoute exactement ce qui manque à notre modèle (attention, mémoire longue). Le contraste est pédagogiquement parfait. |
| **Disponibilité et reproductibilité** | `GPT2LMHeadModel.from_pretrained('gpt2')` — le modèle est téléchargeable en 1 commande, les résultats sont reproductibles par n'importe qui. |
| **Fine-tuning rapide (58.3 s)** | Grâce au transfer learning, 3 époques suffisent pour adapter GPT-2 aux paroles de chansons. Cela permet de comparer sur le *même dataset* sans semaines d'entraînement. |
| **Standard de l'état de l'art 2019–2022** | GPT-2 est le point d'entrée canonique des LLM modernes. Comprendre son architecture, c'est comprendre la base de GPT-4, LLaMA, Mistral. |
| **Vocabulaire BPE (50 257 tokens)** | Le Byte-Pair Encoding gère les mots rares, les contractions (`'cause`, `ain't`), et les fautes — problème que notre vocabulaire de fréquence minimale ne résout pas. |
| **Pré-entraînement sur WebText (40 GB)** | GPT-2 arrive déjà avec une connaissance de l'anglais, des rimes, des structures de phrases. Notre modèle part de zéro sur ~14 500 chansons. |

> **Résultat attendu et confirmé :** GPT-2 atteint PPL = 12.52 vs 220.81 pour notre modèle — soit **×17.6 plus précis** en ×40 moins de temps d'entraînement. Ce ratio illustre exactement la puissance du transfer learning et de l'attention multi-têtes.

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

### Courbes d'Entraînement GPT-2

![GPT-2 Fine-tuning — Loss, PPL, Learning Rate, Grad Norm](graphs+logs/gpt2_training_curve.png)

> Les 4 courbes confirment une convergence saine : la loss val descend régulièrement (2.562 → 2.527), la PPL atteint **12.52** en 3 époques, le learning rate suit le schedule linéaire HuggingFace, et la norme des gradients se stabilise autour de 4.5–5 après l'époque 1.

---

## 8. Comparaison des Deux Approches

### Tableau Comparatif Complet

| Critère | Feed-Forward Custom | GPT-2 Fine-tuné |
|---|:---:|:---:|
| **Val Perplexity** | 220.81 | **12.52** |
| **Val Loss** | 5.397 | **2.527** |
| **Paramètres** | 4.2 M | 124.4 M |
| **Temps d'entraînement** | 39.2 min (15 epochs) | **58.3 sec (3 epochs)** |
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
  9.3 │● (poids ×0.01 — cassé, avant fix)
      │
  6.9 │   ●  ← Epoch 1 (après corrections)
      │
  6.0 │      ●  ← Epoch 2 (5.397 val_loss final)
      │
  5.7 │         ●  ← Epoch 5
      │
  5.4 │                    ●  ← Epoch 15 (val_loss=5.397, PPL=220.81)
      │
  2.5 │ ════════════════════════ GPT-2 fine-tuné (val_loss=2.527, PPL=12.52)
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
4. **124.4 M paramètres** : capacité de représentation sans commune mesure

![Comparaison des deux modèles — PPL, Loss, vitesse, paramètres](graphs+logs/model_comparison.png)

---

## 9. Exemples de Paroles Générées

### Modèle Feed-Forward Custom — Genre ROCK

**Exemple 1 :**
> *it like a and do the you down all get take is never me no got i'm of it's your life ain't be so more to love but if i know you, for this baby what with my time oh i think too say that when we are away don't tell me from the eyes i'll need on me, let*

**Exemple 2 :**
> *i all yeah get to your the in of been and on my you but a it be what there's you, is our love it's just at me not for this so long can i'm like to keep if oh, from they go right up and you're through it look with me when we will are an 'cause i want*

### GPT-2 Fine-tuné — Genre POP

**Exemple 1 :**
> *no, i don't like it popout the lights down 'cause you gotta have a drink now so that we can get out there sing along all night (don't know what to say) oh baby when will ya take my breath away? doin' shit on your head and body for free! yeah if they're gonna come over tonight or tomorrow cause one of us got high just tryna*

**Exemple 2 :**
> *hey, let's go grab some water and run 'round to the beach or just ride on our old horses now i see you've got a bottle of wine in your hand tell me if this time is different we'll do it right with love again don't be shy about saying what matters when things are good at heart baby talk like an animal without any emotions so listen up (oh-oo)*

**Exemple 3 :**
> *to your heart, i think it's gone now you're a lost cause oh-oh uh... and no need for my love the only thing that matters is not getting older just one more weekend with another drink 'cause we've been through hell together but sometimes (yeah) when things get tough there are always lessons left undone these days if nothing else can make us stronger at times even worse than they*

### Analyse Comparative
| Aspect | Feed-Forward Custom | GPT-2 Fine-tuné |
|---|---|---|
| ✅ Mots reconnaissables | Vocabulaire anglais courant | ✅ Naturel, contractions (`'cause`, `ain't`) |
| ✅ Pas de boucles | Pénalités logit-space | ✅ `no_repeat_ngram_size=3` |
| ✅ Diversité lexicale | Ratio unique 0.68–0.75 | ✅ Très élevée |
| ⚠️ Grammaire | Faible — pas de mémoire syntaxique | ✅ **Bonne** — Transformer 12 couches |
| ⚠️ Cohérence narrative | Limitée — fenêtre de 20 tokens | ✅ **Bonne** — 1024 tokens de contexte |
| ⚠️ Émotions / images | Absentes | ✅ Présentes (`"you're a lost cause"`) |

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

## 10. Difficultés Rencontrées & Solutions Apportées

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

## 11. Conclusion & Perspectives

### Ce que Nous Avons Appris
- **Backpropagation from scratch** : chain rule, masking PAD, scatter-add pour embeddings
- **Importance de l'initialisation** : He/Xavier vs naïf
- **Espace logit vs probabilité** : pourquoi les pénalités doivent être appliquées avant softmax
- **Pipelines ML robustes** : split song-level, checkpoints, early stopping, config centralisée
- **Comparaison architectures** : Feed-Forward vs Transformer

### Limites du Modèle Custom
- Pas de mémoire à long terme (fenêtre de 20 tokens seulement)
- Architecture trop simple pour la syntaxe complexe
- Val PPL = 220.81 → le modèle hésite encore entre ~221 mots
  (vs 12.52 pour GPT-2, soit ×17.6 plus précis)

### Perspectives d'Amélioration
| Amélioration | Gain Estimé | Complexité |
|---|---|---|
| Remplacer Feed-Forward par GRU/LSTM | PPL ~200 | Moyenne |
| Augmenter HIDDEN_DIM à 512 | PPL ~380 | Faible |
| GPT-2 fine-tuné complet | PPL ~50–80 | Faible (HuggingFace) |
| Entraîner sur dataset complet (18k songs) | PPL ~300 | Faible |

### GPT-2 — La Comparaison Qui Illustre le Fossé

Notre modèle custom avec 4.2M paramètres et une architecture simple montre les **limites fondamentales** d'une approche sans attention. GPT-2, avec ses 124.4M paramètres et 12 couches Transformer, atteint une **PPL de 12.52 contre 220.81** — soit **×17.6 plus précis** — en seulement 58.3 secondes de fine-tuning.

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
