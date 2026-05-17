# TP: Génération Automatique de Paroles de Chanson 🎵

## Objectif
Développer un modèle de **Machine Learning** pour générer des paroles originales basées sur un **genre musicale** spécifique, en appliquant les techniques du cours.

## Dataset
- **Source**: `spotify_songs.csv` 
- **Contenu**: 18,454 chansons avec paroles et métadonnées
- **Après nettoyage**: ~17,000 chansons exploitables

## Concepts du Cours Appliqués

### 1. **Régression Linéaire** 
- Utilisée pour prédire la longueur des paroles
- Modèle: `y = ax + b` (contexte → mot suivant)

### 2. **Classification**
- Classification multi-classe: prédire le genre basé sur les paroles
- Softmax pour obtenir les probabilités par mot

### 3. **Réseaux de Neurones (Perceptron Multicouche)**
- **Couche d'embedding**: Convertit les indices de mots en vecteurs denses
- **Couche cachée**: Transformation non-linéaire avec ReLU
- **Couche de sortie**: Logits pour chaque mot du vocabulaire
- **Activation**: ReLU dans la couche cachée, Softmax en sortie

### 4. **Fonction de Coût**
- **Cross-Entropy Loss**: $ J = -\frac{1}{m} \sum_{i=1}^{m} y_i \log(p_i) $
- Pour chaque exemple d'entraînement

### 5. **Algorithme d'Apprentissage**
- **Gradient Descent**: $ w_{t+1} = w_t - \alpha \frac{\partial E}{\partial w_t} $
- Learning rate: $\alpha = 0.001$
- Rétro-propagation pour calculer les gradients

### 6. **Représentation Matricielle**
- Données: Matrices NumPy
- Poids: Matrices alignées pour multiplication matricielle
- Batch processing: Calcul vectorisé

## Architecture du Modèle

```
Input (SEQ_LEN tokens) 
    ↓
[Word Embeddings (SEQ_LEN × embedding_dim)]
[Genre Embedding (embedding_dim)]
    ↓
Concatenation
    ↓
[Fully Connected Layer: hidden_dim neurons with ReLU]
    ↓
[Output Layer: vocab_size neurons]
    ↓
Softmax → Probabilités
    ↓
Sample → Prochain token
```

## Étapes du TP

### 1. Chargement et Exploration
```python
# Charger le CSV
df = pd.read_csv('spotify_songs.csv')
# Nettoyer les paroles invalides
df = df[df['lyrics'].notna() & df['lyrics'].str.len() > 50]
```

### 2. Prétraitement du Texte
```python
# Minuscule
text = text.lower()
# Supprimer les caractères spéciaux
text = re.sub(r'[^a-z0-9\s]', '', text)
# Tokenization
tokens = text.split()
```

### 3. Création du Vocabulaire
```python
# Encoder les tokens en indices
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}
```

### 4. Préparation des Données
- Créer des paires (contexte, mot_suivant)
- Padding/Troncature à taille fixe
- Encoder le genre avec LabelEncoder
- Split train/validation (80/20)

### 5. Construction du Modèle
```python
class LyricsGenerationModel:
    def forward(self, X_batch, genres_batch):
        # Forward pass
        ...
    
    def compute_loss(self, logits, y_batch):
        # Cross-entropy loss
        ...
    
    def backward(self, gradients, learning_rate):
        # Gradient descent update
        ...
```

### 6. Entraînement
```python
for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch(model, X_train, y_train, genres_train)
    val_loss = evaluate(model, X_val, y_val, genres_val)
    
    # Early stopping si pas d'amélioration
```

### 7. Génération Auto-Régressive
```python
def generate_lyrics(model, genre_idx, max_length=50, temperature=0.8):
    tokens = [BOS_IDX]
    
    for i in range(max_length):
        # Prédire le prochain token
        logits = model.forward(tokens)
        probs = softmax(logits / temperature)
        
        # Sampler le token suivant
        next_token = np.random.choice(vocab_size, p=probs)
        tokens.append(next_token)
        
        if next_token == EOS_IDX:
            break
    
    return tokens
```

### 8. Post-traitement
- Supprimer les tokens spéciaux
- Limiter les répétitions
- Vérifier la longueur minimale

### 9. Sauvegarde
```python
# Sauvegarder les poids, vocabulaire et config
pickle.dump(inference_package, open('lyrics_model.pkl', 'wb'))
```

## Fichiers du TP

1. **TP_Paroles_Code_Complet.py** - Script Python complet (peut être exécuté directement)
2. **TP_Generation_Paroles_Chanson.ipynb** - Notebook Jupyter (à exécuter dans VS Code)
3. **spotify_songs.csv** - Dataset
4. **lyrics_model.pkl** - Modèle sauvegardé après entraînement

## Utilisation

### Option 1: Exécuter le script complet
```bash
cd <chemin_du_projet>
python3 TP_Paroles_Code_Complet.py
```

### Option 2: Utiliser le Notebook Jupyter
```bash
# Ouvrir dans VS Code
# Exécuter les cellules dans l'ordre
```

### Option 3: Utiliser le modèle sauvegardé
```python
import pickle

class LyricsGenerator:
    def __init__(self, model_path):
        with open(model_path, 'rb') as f:
            self.pkg = pickle.load(f)
    
    def generate(self, genre, max_length=50):
        # Charger le modèle et générer
        ...

# Utiliser
gen = LyricsGenerator('lyrics_model.pkl')
lyrics = gen.generate('rock', max_length=50)
print(lyrics)
```

## Résultats Attendus

### Métriques
- **Vocabulaire**: ~3,000-5,000 mots
- **Loss initial**: ~8-10
- **Loss final**: ~2-3
- **Temps d'entraînement**: ~5-10 minutes (10 epochs)

### Exemples de Génération
```
Genre: ROCK
"i want to rock your world tonight
the feeling is so strong and bright
let's dance until the morning light"

Genre: POP
"baby love me like you do
my heart is beating true
forever you and me together"

Genre: HIP-HOP
"yo check the rhyme i spit so tight
making moves day and night
the beat is flowing real right"
```

## Paramètres Importants

| Paramètre | Valeur | Description |
|-----------|--------|-------------|
| `embedding_dim` | 16 | Dimension des embeddings |
| `hidden_dim` | 32 | Neurons in hidden layer |
| `SEQ_LEN` | 10 | Context window size |
| `MAX_SEQ_LEN` | ~200 | Max song length |
| `BATCH_SIZE` | 128 | Training batch size |
| `LEARNING_RATE` | 0.001 | Gradient descent step |
| `NUM_EPOCHS` | 10-15 | Training epochs |
| `temperature` | 0.7-1.2 | Sampling diversity |

## Améliorations Possibles

1. **Architecture**
   - Ajouter un LSTM/GRU pour meilleure capture de contexte
   - Augmenter hidden_dim et embedding_dim
   - Ajouter des couches supplémentaires

2. **Données**
   - Filtrer les genres avec plus de textes
   - Augmente la taille du vocabulaire
   - Nettoyer davantage les paroles

3. **Entraînement**
   - Learning rate scheduling
   - Batch normalization
   - Dropout regularization

4. **Génération**
   - Beam search au lieu de sampling
   - Top-k et top-p sampling
   - Seed tokens pour meilleur contrôle

## Ressources Utilisées

- **NumPy**: Opérations matricielles
- **Pandas**: Manipulation du dataset
- **Scikit-learn**: Encodeurs et split train/test
- **Matplotlib**: Visualisation
- **Pickle**: Sérialisation du modèle

## Conclusion

Ce TP applique les concepts fondamentaux du Machine Learning:
- ✅ Représentation matricielle des données
- ✅ Fonction de coût et optimisation
- ✅ Gradient descent et rétro-propagation
- ✅ Réseaux de neurones multi-couches
- ✅ Classification conditionnée (par genre)
- ✅ Généralisation et validation

C'est une excellente introduction aux modèles génératifs et aux réseaux neuronaux pour le traitement du langage naturel!

---

**Auteur**: TP Machine Learning  
**Date**: Mai 2026  
**Niveau**: Intermédiaire  
**Durée estimée**: 2-3 heures
