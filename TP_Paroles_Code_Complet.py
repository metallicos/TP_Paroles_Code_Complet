import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import re
import pickle


def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))


def get_data_path(filename):
    return os.path.join(get_project_root(), filename)


print("="*60)
print("SECTION 1: Chargement du Dataset Spotify")
print("="*60)

csv_path = get_data_path('spotify_songs.csv')
if not os.path.exists(csv_path):
    print(f"❌ Dataset non trouvé: {csv_path}")
    sys.exit(1)

df = pd.read_csv(csv_path)

print(f"✓ Dataset chargé: {len(df):,} chansons")

df = df[df['lyrics'].notna()].copy()
df = df[df['lyrics'].str.len() > 50].copy()
df = df[df['lyrics'] != 'NA'].copy()

print(f"✓ Après nettoyage: {len(df):,} chansons avec paroles valides")

print("\n" + "="*60)
print("SECTION 2: Exploration des Genres")
print("="*60)

TOP_GENRES = 5
top_genres = df['playlist_genre'].value_counts().head(TOP_GENRES).index.tolist()
print(f"✓ Top {TOP_GENRES} genres:")
for i, g in enumerate(top_genres, 1):
    count = len(df[df['playlist_genre'] == g])
    print(f"  {i}. {g}: {count} chansons")

print("\n" + "="*60)
print("SECTION 3: Prétraitement du Texte")
print("="*60)

def preprocess_text(text):
    text = text.lower()
    text = text.replace('\n', ' <NEW_LINE> ')
    text = re.sub(r'[^a-z0-9\s\'\-,.:!?()]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize(text):
    return text.split()

df['lyrics_clean'] = df['lyrics'].apply(preprocess_text)
df['tokens'] = df['lyrics_clean'].apply(tokenize)
print(f"✓ Prétraitement: taille moyenne {df['tokens'].str.len().mean():.0f} tokens")

print("\n" + "="*60)
print("SECTION 4: Construction du Vocabulaire")
print("="*60)

all_tokens = []
for tokens_list in df['tokens']:
    all_tokens.extend(tokens_list)

word_freq = Counter(all_tokens)
MIN_FREQ = 2
vocab = {word: freq for word, freq in word_freq.items() if freq >= MIN_FREQ}

SPECIAL_TOKENS = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
word2idx = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
idx = len(SPECIAL_TOKENS)

for word in sorted(vocab.keys()):
    word2idx[word] = idx
    idx += 1

idx2word = {idx: word for word, idx in word2idx.items()}
PAD_IDX = word2idx['<PAD>']
UNK_IDX = word2idx['<UNK>']
BOS_IDX = word2idx['<BOS>']
EOS_IDX = word2idx['<EOS>']

print(f"✓ Vocabulaire: {len(word2idx)} tokens")

print("\n" + "="*60)
print("SECTION 5: Encodage des Données")
print("="*60)

def encode_tokens(tokens, word2idx, max_len=None):
    encoded = [word2idx.get(token, UNK_IDX) for token in tokens]
    if max_len:
        if len(encoded) > max_len:
            encoded = encoded[:max_len]
        else:
            encoded += [PAD_IDX] * (max_len - len(encoded))
    return encoded

MAX_SEQ_LEN = int(np.percentile(df['tokens'].str.len(), 90))
SEQ_LEN = 10

df['encoded_tokens'] = df['tokens'].apply(lambda x: encode_tokens(x, word2idx, MAX_SEQ_LEN))
df_filtered = df[df['playlist_genre'].isin(top_genres)].copy()

genre_encoder = LabelEncoder()
df_filtered['genre_encoded'] = genre_encoder.fit_transform(df_filtered['playlist_genre'])
NUM_GENRES = len(genre_encoder.classes_)

print(f"✓ Données filtrées: {len(df_filtered)} chansons, {NUM_GENRES} genres")

print("\n" + "="*60)
print("SECTION 6: Préparation Train/Validation")
print("="*60)

X, y, genres_list = [], [], []

for idx, row in df_filtered.iterrows():
    tokens = row['encoded_tokens']
    genre = row['genre_encoded']
    for i in range(len(tokens) - 1):
        X.append(tokens[:i+1])
        y.append(tokens[i+1])
        genres_list.append(genre)

def pad_sequence(seq, max_len):
    if len(seq) == max_len:
        return seq
    elif len(seq) < max_len:
        return seq + [PAD_IDX] * (max_len - len(seq))
    else:
        return seq[-max_len:]

X_padded = np.array([pad_sequence(seq, SEQ_LEN) for seq in X])
y_array = np.array(y)
genres_array = np.array(genres_list)

X_train, X_val, y_train, y_val, genres_train, genres_val = train_test_split(
    X_padded, y_array, genres_array, test_size=0.2, random_state=42
)

print(f"✓ Train: {len(X_train):,} | Val: {len(X_val):,}")

print("\n" + "="*60)
print("SECTION 7: Modèle Neural Network")
print("="*60)

class LyricsGenerationModel:
    def __init__(self, vocab_size, embedding_dim=16, hidden_dim=32, num_genres=5):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        np.random.seed(42)
        self.word_embedding = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.genre_embedding = np.random.randn(num_genres, embedding_dim) * 0.01
        self.W1 = np.random.randn(SEQ_LEN * embedding_dim + embedding_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, vocab_size) * 0.01
        self.b2 = np.zeros((1, vocab_size))
        
        self.loss_history = []
        self.val_loss_history = []
    
    def forward(self, X_batch, genres_batch):
        batch_size = X_batch.shape[0]
        word_embeds = self.word_embedding[X_batch]
        word_embeds_flat = word_embeds.reshape(batch_size, -1)
        genre_embeds = self.genre_embedding[genres_batch]
        combined = np.concatenate([word_embeds_flat, genre_embeds], axis=1)
        
        z1 = np.dot(combined, self.W1) + self.b1
        a1 = np.maximum(z1, 0)
        z2 = np.dot(a1, self.W2) + self.b2
        
        return z2, a1, combined
    
    def compute_loss(self, logits, y_batch):
        batch_size = logits.shape[0]
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        correct_log_probs = -np.log(probs[np.arange(batch_size), y_batch])
        loss = np.mean(correct_log_probs)
        return loss, probs
    
    def backward(self, X_batch, genres_batch, y_batch, logits, a1, combined, probs, learning_rate=0.01):
        batch_size = X_batch.shape[0]
        d_logits = probs.copy()
        d_logits[np.arange(batch_size), y_batch] -= 1
        d_logits /= batch_size
        
        dW2 = np.dot(a1.T, d_logits)
        db2 = np.sum(d_logits, axis=0, keepdims=True)
        
        d_a1 = np.dot(d_logits, self.W2.T)
        d_z1 = d_a1 * (a1 > 0)
        
        dW1 = np.dot(combined.T, d_z1)
        db1 = np.sum(d_z1, axis=0, keepdims=True)
        
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

model = LyricsGenerationModel(
    vocab_size=len(word2idx),
    embedding_dim=16,
    hidden_dim=32,
    num_genres=NUM_GENRES
)
print(f"✓ Modèle créé")

print("\n" + "="*60)
print("SECTION 8: Entraînement du Modèle")
print("="*60)

def train_epoch(model, X_train, y_train, genres_train, batch_size=128, lr=0.01):
    total_loss = 0
    num_batches = len(X_train) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(X_train))
        
        X_batch = X_train[start_idx:end_idx]
        y_batch = y_train[start_idx:end_idx]
        genres_batch = genres_train[start_idx:end_idx]
        
        logits, a1, combined = model.forward(X_batch, genres_batch)
        loss, probs = model.compute_loss(logits, y_batch)
        model.backward(X_batch, genres_batch, y_batch, logits, a1, combined, probs, lr)
        total_loss += loss
    
    return total_loss / num_batches

def evaluate(model, X_test, y_test, genres_test, batch_size=128):
    total_loss = 0
    num_batches = len(X_test) // batch_size
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(X_test))
        
        X_batch = X_test[start_idx:end_idx]
        y_batch = y_test[start_idx:end_idx]
        genres_batch = genres_test[start_idx:end_idx]
        
        logits, _, _ = model.forward(X_batch, genres_batch)
        loss, _ = model.compute_loss(logits, y_batch)
        total_loss += loss
    
    return total_loss / num_batches

NUM_EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001

print(f"Époque | Train Loss | Val Loss")
print("-" * 35)

for epoch in range(NUM_EPOCHS):
    train_loss = train_epoch(model, X_train, y_train, genres_train, BATCH_SIZE, LEARNING_RATE)
    val_loss = evaluate(model, X_val, y_val, genres_val, BATCH_SIZE)
    
    model.loss_history.append(train_loss)
    model.val_loss_history.append(val_loss)
    
    print(f"{epoch+1:5d} | {train_loss:.6f}  | {val_loss:.6f}")

print(f"\n✓ Entraînement terminé!")

print("\n" + "="*60)
print("SECTION 9: Génération de Paroles")
print("="*60)

def generate_lyrics(model, genre_idx, max_length=50, temperature=1.0):
    generated_tokens = [BOS_IDX]
    context_buffer = generated_tokens[-SEQ_LEN:] if len(generated_tokens) >= 1 else generated_tokens
    
    for _ in range(max_length):
        X_input = np.array([pad_sequence(context_buffer, SEQ_LEN)])
        genres_input = np.array([genre_idx])
        
        logits, _, _ = model.forward(X_input, genres_input)
        logits = logits[0] / temperature
        
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)
        next_token = np.random.choice(len(probs), p=probs)
        
        if next_token == EOS_IDX:
            break
        
        generated_tokens.append(next_token)
        context_buffer.append(next_token)
        if len(context_buffer) > SEQ_LEN:
            context_buffer = context_buffer[-SEQ_LEN:]
    
    return generated_tokens

def tokens_to_lyrics(token_indices):
    words = []
    for token_idx in token_indices:
        if token_idx == EOS_IDX:
            break
        word = idx2word.get(token_idx, '<UNK>')
        if word not in ['<PAD>', '<BOS>', '<EOS>']:
            words.append(word)
    return ' '.join(words)

print("\nGénération par genre:")
for genre_idx, genre_name in enumerate(genre_encoder.classes_):
    tokens = generate_lyrics(model, genre_idx, max_length=40, temperature=0.8)
    lyrics = tokens_to_lyrics(tokens)
    print(f"\n{genre_name.upper()}:")
    print(f"  {lyrics[:150]}...")

print("\n" + "="*60)
print("SECTION 10: Sauvegarde du Modèle")
print("="*60)

inference_package = {
    'model_weights': {
        'word_embedding': model.word_embedding,
        'genre_embedding': model.genre_embedding,
        'W1': model.W1,
        'b1': model.b1,
        'W2': model.W2,
        'b2': model.b2,
    },
    'vocab': {
        'word2idx': word2idx,
        'idx2word': idx2word,
    },
    'config': {
        'embedding_dim': 16,
        'hidden_dim': 32,
        'seq_len': SEQ_LEN,
        'num_genres': NUM_GENRES,
        'genres': list(genre_encoder.classes_),
    },
    'constants': {
        'PAD_IDX': PAD_IDX,
        'UNK_IDX': UNK_IDX,
        'BOS_IDX': BOS_IDX,
        'EOS_IDX': EOS_IDX,
    }
}

model_path = get_data_path('lyrics_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(inference_package, f)

print(f"✓ Modèle sauvegardé: {model_path}")
print(f"✓ Taille: {os.path.getsize(model_path) / 1024:.1f} KB")

print("\n" + "="*60)
print("RÉSUMÉ DU TP")
print("="*60)
print(f"✓ Dataset: {len(df_filtered)} chansons")
print(f"✓ Vocabulaire: {len(word2idx):,} tokens")
print(f"✓ Genres: {NUM_GENRES}")
print(f"✓ Exemples: {len(X_train):,} train / {len(X_val):,} val")
print(f"✓ Loss final: {model.val_loss_history[-1]:.4f}")
print("="*60)
