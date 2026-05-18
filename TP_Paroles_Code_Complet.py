"""
Pipeline principal du TP — modèle custom de génération de paroles.

Ce script couvre toute la chaîne :
1) chargement + nettoyage du dataset,
2) création vocabulaire / encodage,
3) entraînement d'un modèle feed-forward conditionné par genre,
4) sauvegarde checkpoints + métriques,
5) génération et évaluation de base.

Note: les hyperparamètres peuvent être définis via `config/train_config.json`
et/ou variables d'environnement.
"""

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
import time
import traceback
import glob
import json
import subprocess

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def log_memory():
    """Print current RAM usage if psutil is available."""
    if HAS_PSUTIL:
        mem = psutil.virtual_memory()
        used_gb = (mem.total - mem.available) / 1024**3
        total_gb = mem.total / 1024**3
        pct = mem.percent
        print(f"  [MEM] RAM: {used_gb:.1f}/{total_gb:.1f} GB ({pct:.1f}% used)")
        if pct > 85:
            print(f"  ⚠️  RAM usage high ({pct:.1f}%) — risk of crash!")


def get_array_backend():
    use_gpu = os.getenv("USE_GPU", "auto").lower()
    if use_gpu in ("1", "true", "yes", "auto"):
        try:
            import cupy as cp
            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count > 0:
                return cp, True
        except Exception:
            pass
    return np, False


xp, USING_GPU = get_array_backend()


def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))


def get_data_path(filename):
    return os.path.join(get_project_root(), filename)


def get_outputs_dir():
    output_dir = os.path.join(get_project_root(), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_output_path(filename):
    return os.path.join(get_outputs_dir(), filename)


def load_config_defaults():
    """Load optional config/train_config.json and apply values as env defaults.
    Existing environment variables keep priority for easy overrides.
    """
    cfg_path = os.path.join(get_project_root(), 'config', 'train_config.json')
    if not os.path.exists(cfg_path):
        return {}

    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        for key, value in cfg.items():
            if os.getenv(key) is None:
                os.environ[key] = str(value)
        print(f"✓ Config chargée: {cfg_path}")
        return cfg
    except Exception as e:
        print(f"⚠️  Config ignorée ({e})")
        return {}


CONFIG_DEFAULTS = load_config_defaults()


print("="*60)
print("SECTION 1: Chargement du Dataset Spotify")
print("="*60)

if USING_GPU:
    print("✓ Backend: GPU (CuPy)")
else:
    print("✓ Backend: CPU (NumPy)")

csv_path = get_data_path('spotify_songs.csv')
if not os.path.exists(csv_path):
    print(f"❌ Dataset non trouvé: {csv_path}")
    sys.exit(1)

df = pd.read_csv(csv_path)

print(f"✓ Dataset chargé: {len(df):,} chansons")

df = df[df['lyrics'].notna()].copy()
df = df[df['lyrics'].str.len() > 50].copy()
df = df[df['lyrics'] != 'NA'].copy()

# ──────────────────────────────────────────────────────────────
# FILTRE LANGUE: English uniquement (colonne 'language' == 'en')
# Évite le mélange espagnol/français/hollandais/italien dans le modèle
# Désactiver: mettre LANG_FILTER = None
# ──────────────────────────────────────────────────────────────
LANG_FILTER = os.getenv("LANG_FILTER", "en").strip() or None
if LANG_FILTER and 'language' in df.columns:
    before = len(df)
    df = df[df['language'] == LANG_FILTER].copy()
    print(f"✓ Filtre langue '{LANG_FILTER}': {len(df):,} chansons (sur {before:,})")
else:
    print(f"ℹ️  Pas de filtre langue appliqué")

print(f"✓ Après nettoyage: {len(df):,} chansons avec paroles valides")

# ──────────────────────────────────────────────────────────────
# PARAMÈTRE: Nombre maximum d'entrées à utiliser
#   None  → utiliser tout le dataset
#   2000  → rapide pour tester / debug
#   5000  → équilibre vitesse/qualité
#   None  → entraînement complet (~18 000 chansons)
# Peut aussi être surchargé via la variable d'environnement MAX_SAMPLES
# Ex: MAX_SAMPLES=3000 python3 TP_Paroles_Code_Complet.py
# ──────────────────────────────────────────────────────────────
_env_max = os.getenv("MAX_SAMPLES", "").strip()
MAX_SAMPLES = int(_env_max) if _env_max.isdigit() else None  # None = tout le dataset

if MAX_SAMPLES is not None:
    if MAX_SAMPLES >= len(df):
        print(f"ℹ️  MAX_SAMPLES={MAX_SAMPLES} >= dataset ({len(df):,}) — utilisation complète")
    else:
        df = df.sample(n=MAX_SAMPLES, random_state=42).reset_index(drop=True)
        print(f"✂️  Dataset réduit à {MAX_SAMPLES:,} entrées (sur {len(df) + MAX_SAMPLES - MAX_SAMPLES:,} originales)")
else:
    print(f"ℹ️  MAX_SAMPLES=None — utilisation du dataset complet ({len(df):,} entrées)")

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

try:
    df['lyrics_clean'] = df['lyrics'].apply(preprocess_text)
    df['tokens'] = df['lyrics_clean'].apply(tokenize)
    print(f"✓ Prétraitement: taille moyenne {df['tokens'].str.len().mean():.0f} tokens")
except Exception as e:
    print(f"❌ Erreur prétraitement: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("SECTION 4: Construction du Vocabulaire")
print("="*60)

all_tokens = []
for tokens_list in df['tokens']:
    all_tokens.extend(tokens_list)

word_freq = Counter(all_tokens)
MIN_FREQ = int(os.getenv("MIN_FREQ", "5"))
MAX_VOCAB_SIZE = int(os.getenv("MAX_VOCAB_SIZE", "12000"))
filtered_tokens = [(word, freq) for word, freq in word_freq.items() if freq >= MIN_FREQ]
filtered_tokens.sort(key=lambda item: item[1], reverse=True)
filtered_tokens = filtered_tokens[:MAX_VOCAB_SIZE]
vocab = {word: freq for word, freq in filtered_tokens}

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

print(f"✓ Vocabulaire: {len(word2idx)} tokens (cap: {MAX_VOCAB_SIZE})")

print("\n" + "="*60)
print("SECTION 5: Encodage des Données")
print("="*60)

def encode_tokens(tokens, word2idx, max_len=None):
    encoded = [BOS_IDX] + [word2idx.get(token, UNK_IDX) for token in tokens] + [EOS_IDX]
    if max_len:
        if len(encoded) > max_len:
            encoded = encoded[:max_len]
            encoded[-1] = EOS_IDX
    return encoded

MAX_SEQ_LEN = int(np.percentile(df['tokens'].str.len(), 90))
SEQ_LEN = int(os.getenv("SEQ_LEN", "20"))

df['encoded_tokens'] = df['tokens'].apply(lambda x: encode_tokens(x, word2idx, MAX_SEQ_LEN))
df_filtered = df[df['playlist_genre'].isin(top_genres)].copy()

genre_encoder = LabelEncoder()
df_filtered['genre_encoded'] = genre_encoder.fit_transform(df_filtered['playlist_genre'])
NUM_GENRES = len(genre_encoder.classes_)

print(f"✓ Données filtrées: {len(df_filtered)} chansons, {NUM_GENRES} genres")

print("\n" + "="*60)
print("SECTION 6: Préparation Train/Validation")
print("="*60)

def build_pairs(df_subset):
    X_local, y_local, genres_local = [], [], []
    total_rows = len(df_subset)
    pair_start = time.time()

    for idx, (_, row) in enumerate(df_subset.iterrows()):
        if (idx + 1) % 500 == 0:
            elapsed = time.time() - pair_start
            remaining = elapsed / (idx + 1) * max(total_rows - idx - 1, 0)
            print(f"  Progression: {idx + 1}/{total_rows} | ETA: {remaining:.0f}s | Paires: {len(X_local):,}")
            log_memory()

        tokens = row['encoded_tokens']
        genre = row['genre_encoded']
        for i in range(len(tokens) - 1):
            X_local.append(tokens[:i+1])
            y_local.append(tokens[i+1])
            genres_local.append(genre)

    return X_local, y_local, genres_local, time.time() - pair_start


print("Séparation au niveau chanson (anti-fuite train/val)...")
song_train_df, song_val_df = train_test_split(df_filtered, test_size=0.2, random_state=42)
print(f"✓ Chansons train: {len(song_train_df):,} | val: {len(song_val_df):,}")

train_song_ids = set(song_train_df.index.tolist())
val_song_ids = set(song_val_df.index.tolist())
overlap_ids = train_song_ids.intersection(val_song_ids)
if overlap_ids:
    print(f"⚠️  Chevauchement train/val détecté: {len(overlap_ids)}")
else:
    print("✓ Aucun chevauchement chanson entre train et val")

print("Création des paires train...")
log_memory()
try:
    X_train_raw, y_train_raw, genres_train_raw, train_pair_time = build_pairs(song_train_df)
    print(f"  ✓ {len(X_train_raw):,} paires train en {train_pair_time:.1f}s")

    print("Création des paires val...")
    X_val_raw, y_val_raw, genres_val_raw, val_pair_time = build_pairs(song_val_df)
    print(f"  ✓ {len(X_val_raw):,} paires val en {val_pair_time:.1f}s")
except MemoryError:
    print("❌ MemoryError lors de la création des paires")
    print("  💡 Conseil: Réduire MAX_VOCAB_SIZE, SEQ_LEN ou MAX_SAMPLES")
    sys.exit(1)
except Exception as e:
    print(f"❌ Erreur: {e}")
    traceback.print_exc()
    sys.exit(1)

def pad_sequence(seq, max_len):
    if len(seq) == max_len:
        return seq
    elif len(seq) < max_len:
        return seq + [PAD_IDX] * (max_len - len(seq))
    else:
        return seq[-max_len:]

X_train = np.array([pad_sequence(seq, SEQ_LEN) for seq in X_train_raw])
X_val = np.array([pad_sequence(seq, SEQ_LEN) for seq in X_val_raw])
y_train = np.array(y_train_raw)
y_val = np.array(y_val_raw)
genres_train = np.array(genres_train_raw)
genres_val = np.array(genres_val_raw)

print(f"✓ Train: {len(X_train):,} | Val: {len(X_val):,}")

print("\n" + "="*60)
print("SECTION 7: Modèle Neural Network")
print("="*60)

class LyricsGenerationModel:
    def __init__(self, vocab_size, embedding_dim=32, hidden_dim=128, num_genres=5):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        xp.random.seed(42)
        # He initialization for hidden layer, Xavier for output, 0.1 for embeddings
        input_dim = SEQ_LEN * embedding_dim + embedding_dim
        self.word_embedding = xp.random.randn(vocab_size, embedding_dim) * 0.1
        self.genre_embedding = xp.random.randn(num_genres, embedding_dim) * 0.1
        self.W1 = xp.random.randn(input_dim, hidden_dim) * float(np.sqrt(2.0 / input_dim))
        self.b1 = xp.zeros((1, hidden_dim))
        self.W2 = xp.random.randn(hidden_dim, vocab_size) * float(np.sqrt(1.0 / hidden_dim))
        self.b2 = xp.zeros((1, vocab_size))
        
        self.loss_history = []
        self.val_loss_history = []
    
    def forward(self, X_batch, genres_batch, training=False, dropout_rate=0.0):
        batch_size = X_batch.shape[0]
        word_embeds = self.word_embedding[X_batch]
        word_embeds_flat = word_embeds.reshape(batch_size, -1)
        genre_embeds = self.genre_embedding[genres_batch]
        combined = xp.concatenate([word_embeds_flat, genre_embeds], axis=1)
        
        z1 = xp.dot(combined, self.W1) + self.b1
        relu_mask = (z1 > 0)
        a1 = xp.maximum(z1, 0)
        dropout_mask = None
        if training and dropout_rate > 0.0:
            keep_prob = 1.0 - dropout_rate
            dropout_mask = (xp.random.rand(*a1.shape) < keep_prob).astype(a1.dtype) / max(keep_prob, 1e-8)
            a1 = a1 * dropout_mask
        z2 = xp.dot(a1, self.W2) + self.b2
        
        return z2, a1, combined, relu_mask, dropout_mask
    
    def compute_loss(self, logits, y_batch, label_smoothing=0.0):
        batch_size = logits.shape[0]
        exp_logits = xp.exp(logits - xp.max(logits, axis=1, keepdims=True))
        probs = exp_logits / xp.sum(exp_logits, axis=1, keepdims=True)
        probs = xp.clip(probs, 1e-12, 1.0)

        valid_mask = (y_batch != PAD_IDX)
        valid_count = int(valid_mask.sum().item()) if hasattr(valid_mask.sum(), 'item') else int(valid_mask.sum())
        if valid_count == 0:
            return 0.0, probs

        if label_smoothing and label_smoothing > 0.0:
            smooth = float(label_smoothing)
            target = xp.full_like(probs, smooth / max(self.vocab_size - 1, 1))
            target[xp.arange(batch_size), y_batch] = 1.0 - smooth
            per_sample_loss = -xp.sum(target * xp.log(probs), axis=1)
            loss = float(per_sample_loss[valid_mask].mean().item())
        else:
            correct_log_probs = -xp.log(probs[xp.arange(batch_size), y_batch])
            loss = float(correct_log_probs[valid_mask].mean().item())
        return loss, probs
    
    def backward(self, X_batch, genres_batch, y_batch, logits, a1, combined, relu_mask, dropout_mask, probs,
                 learning_rate=0.01, label_smoothing=0.0, grad_clip=None):
        batch_size = X_batch.shape[0]
        # Mask out PAD targets — they must not contribute gradients
        valid_mask = (y_batch != PAD_IDX)
        valid_count = int(valid_mask.sum().item()) if hasattr(valid_mask.sum(), 'item') else int(valid_mask.sum())
        if valid_count == 0:
            return

        if label_smoothing and label_smoothing > 0.0:
            smooth = float(label_smoothing)
            target = xp.full_like(probs, smooth / max(self.vocab_size - 1, 1))
            target[xp.arange(batch_size), y_batch] = 1.0 - smooth
            d_logits = probs - target
        else:
            d_logits = probs.copy()
            d_logits[xp.arange(batch_size), y_batch] -= 1
        # Zero out gradients for PAD-target rows and divide by valid_count only
        d_logits[~valid_mask] = 0.0
        d_logits /= max(valid_count, 1)
        
        dW2 = xp.dot(a1.T, d_logits)
        db2 = xp.sum(d_logits, axis=0, keepdims=True)
        
        d_a1 = xp.dot(d_logits, self.W2.T)
        if dropout_mask is not None:
            d_a1 = d_a1 * dropout_mask
        d_z1 = d_a1 * relu_mask
        
        dW1 = xp.dot(combined.T, d_z1)
        db1 = xp.sum(d_z1, axis=0, keepdims=True)

        d_combined = xp.dot(d_z1, self.W1.T)
        d_word_flat = d_combined[:, :SEQ_LEN * self.embedding_dim]
        d_genre_embed = d_combined[:, SEQ_LEN * self.embedding_dim:]
        d_word_embed = d_word_flat.reshape(X_batch.shape[0], SEQ_LEN, self.embedding_dim)

        if grad_clip is not None and grad_clip > 0:
            for grad in [dW1, db1, dW2, db2]:
                grad_norm = float(xp.linalg.norm(grad).item())
                if grad_norm > grad_clip:
                    grad *= grad_clip / (grad_norm + 1e-8)
        
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1

        # Update embeddings (token + genre) via scatter-add style accumulation
        grad_word_embedding = xp.zeros_like(self.word_embedding)
        flat_tokens = X_batch.reshape(-1)
        flat_grads = d_word_embed.reshape(-1, self.embedding_dim)
        valid_mask = (flat_tokens != PAD_IDX)
        token_idx = flat_tokens[valid_mask]
        token_grads = flat_grads[valid_mask]

        if USING_GPU:
            import cupyx
            cupyx.scatter_add(grad_word_embedding, token_idx, token_grads)
        else:
            np.add.at(grad_word_embedding, token_idx, token_grads)

        grad_genre_embedding = xp.zeros_like(self.genre_embedding)
        if USING_GPU:
            import cupyx
            cupyx.scatter_add(grad_genre_embedding, genres_batch, d_genre_embed)
        else:
            np.add.at(grad_genre_embedding, genres_batch, d_genre_embed)

        if grad_clip is not None and grad_clip > 0:
            for grad in [grad_word_embedding, grad_genre_embedding]:
                grad_norm = float(xp.linalg.norm(grad).item())
                if grad_norm > grad_clip:
                    grad *= grad_clip / (grad_norm + 1e-8)

        self.word_embedding -= learning_rate * grad_word_embedding
        self.genre_embedding -= learning_rate * grad_genre_embedding

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "32"))
HIDDEN_DIM = int(os.getenv("HIDDEN_DIM", "128"))

model = LyricsGenerationModel(
    vocab_size=len(word2idx),
    embedding_dim=EMBEDDING_DIM,
    hidden_dim=HIDDEN_DIM,
    num_genres=NUM_GENRES
)
print(f"✓ Modèle créé")

print("\n" + "="*60)
print("SECTION 8: Entraînement du Modèle")
print("="*60)

def compute_accuracy(logits, y_batch):
    predictions = xp.argmax(logits, axis=1)
    valid_mask = (y_batch != PAD_IDX)
    valid_count = int(valid_mask.sum().item()) if hasattr(valid_mask.sum(), 'item') else int(valid_mask.sum())
    if valid_count == 0:
        return 0.0
    accuracy = float((predictions[valid_mask] == y_batch[valid_mask]).mean().item())
    return accuracy

def compute_perplexity(loss):
    return np.exp(loss)


X_train = np.asarray(X_train, dtype=np.int32)
X_val = np.asarray(X_val, dtype=np.int32)
y_train = np.asarray(y_train, dtype=np.int32)
y_val = np.asarray(y_val, dtype=np.int32)
genres_train = np.asarray(genres_train, dtype=np.int32)
genres_val = np.asarray(genres_val, dtype=np.int32)

if USING_GPU:
    print("✓ Transfert des tenseurs vers GPU...")
    try:
        X_train = xp.asarray(X_train)
        X_val = xp.asarray(X_val)
        y_train = xp.asarray(y_train)
        y_val = xp.asarray(y_val)
        genres_train = xp.asarray(genres_train)
        genres_val = xp.asarray(genres_val)
        print("  ✓ Transfert GPU réussi")
    except Exception as e:
        print(f"  ⚠️  Transfert GPU échoué ({e}), repli sur CPU")
        xp = np
        USING_GPU = False

GRAD_CLIP = 5.0  # max gradient norm before clipping


def train_epoch(model, X_train, y_train, genres_train, batch_size=128, lr=0.01):
    total_loss = 0
    total_accuracy = 0
    num_batches = int(np.ceil(len(X_train) / batch_size))
    epoch_start = time.time()
    batch_times = []
    nan_batches = 0

    for batch_idx in range(num_batches):
        batch_t0 = time.time()
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(X_train))

        X_batch = X_train[start_idx:end_idx]
        y_batch = y_train[start_idx:end_idx]
        genres_batch = genres_train[start_idx:end_idx]

        try:
            logits, a1, combined, relu_mask, dropout_mask = model.forward(
                X_batch,
                genres_batch,
                training=True,
                dropout_rate=DROPOUT_RATE,
            )
            loss, probs = model.compute_loss(logits, y_batch, label_smoothing=LABEL_SMOOTHING)

            # NaN / Inf guard — skip bad batch
            if np.isnan(loss) or np.isinf(loss):
                nan_batches += 1
                print(f"    ⚠️  NaN/Inf loss at batch {batch_idx + 1} — skipping")
                if nan_batches >= 5:
                    print("    ❌ Too many NaN batches — stopping epoch early")
                    break
                continue

            accuracy = compute_accuracy(logits, y_batch)
            model.backward(
                X_batch,
                genres_batch,
                y_batch,
                logits,
                a1,
                combined,
                relu_mask,
                dropout_mask,
                probs,
                lr,
                label_smoothing=LABEL_SMOOTHING,
                grad_clip=GRAD_CLIP,
            )

            total_loss += loss
            total_accuracy += accuracy
        except Exception as e:
            print(f"    ⚠️  Erreur batch {batch_idx + 1}: {e}")
            continue

        batch_times.append(time.time() - batch_t0)
        if (batch_idx + 1) % 500 == 0 or (batch_idx + 1) == num_batches:
            elapsed = time.time() - epoch_start
            avg_bt = np.mean(batch_times[-100:]) if batch_times else 0
            eta = (num_batches - batch_idx - 1) * avg_bt
            print(f"    Batch {batch_idx + 1}/{num_batches} | Loss: {loss:.4f} | Acc: {accuracy:.4f} | Elapsed: {elapsed:.1f}s | ETA: {eta:.0f}s")

    valid_batches = max(num_batches - nan_batches, 1)
    return total_loss / valid_batches, total_accuracy / valid_batches

def evaluate(model, X_test, y_test, genres_test, batch_size=128):
    total_loss = 0
    total_accuracy = 0
    num_batches = int(np.ceil(len(X_test) / batch_size))
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(X_test))
        
        X_batch = X_test[start_idx:end_idx]
        y_batch = y_test[start_idx:end_idx]
        genres_batch = genres_test[start_idx:end_idx]
        
        logits, _, _, _, _ = model.forward(X_batch, genres_batch, training=False, dropout_rate=0.0)
        loss, _ = model.compute_loss(logits, y_batch, label_smoothing=0.0)
        accuracy = compute_accuracy(logits, y_batch)
        
        total_loss += loss
        total_accuracy += accuracy
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    return avg_loss, avg_accuracy

NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "15"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "128"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.01"))
LR_DECAY = float(os.getenv("LR_DECAY", "0.95"))
MIN_LEARNING_RATE = float(os.getenv("MIN_LEARNING_RATE", "0.0001"))
DROPOUT_RATE = float(os.getenv("DROPOUT_RATE", "0.15"))
LABEL_SMOOTHING = float(os.getenv("LABEL_SMOOTHING", "0.05"))
EARLY_STOP_PATIENCE = 3  # stop if val_loss doesn't improve for N epochs
CHECKPOINT_EVERY = 1     # save checkpoint every N epochs

resume_from = os.getenv("RESUME_FROM", "").strip()
start_epoch = 0

print(f"Config: epochs={NUM_EPOCHS}, batch={BATCH_SIZE}, lr={LEARNING_RATE}, decay={LR_DECAY}, dropout={DROPOUT_RATE}, label_smoothing={LABEL_SMOOTHING}, patience={EARLY_STOP_PATIENCE}")
log_memory()
print(f"\n{'Époque':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>7} | {'Train PPL':>9} | {'Val PPL':>7} | {'Time':>6}")
print("-" * 90)

best_val_loss = float('inf')
best_epoch = 0
patience_counter = 0
training_start = time.time()

if resume_from:
    if resume_from.lower() == "latest":
        ckpts = sorted(glob.glob(get_output_path('checkpoint_epoch*.pkl')))
        resume_path = ckpts[-1] if ckpts else ""
    elif os.path.isabs(resume_from):
        resume_path = resume_from
    else:
        resume_path = get_output_path(resume_from)

    if resume_path and os.path.exists(resume_path):
        try:
            with open(resume_path, 'rb') as f:
                ckpt = pickle.load(f)

            weights = ckpt.get('model_weights', {})
            for key in ['word_embedding', 'genre_embedding', 'W1', 'b1', 'W2', 'b2']:
                if key in weights:
                    setattr(model, key, xp.asarray(weights[key]))

            model.loss_history = ckpt.get('loss_history', [])
            model.val_loss_history = ckpt.get('val_loss_history', [])

            start_epoch = int(ckpt.get('epoch', 0))
            if model.val_loss_history:
                best_val_loss = min(model.val_loss_history)
                best_epoch = int(np.argmin(model.val_loss_history)) + 1

            print(f"✓ Reprise depuis: {resume_path} (epoch {start_epoch})")
        except Exception as e:
            print(f"⚠️  Reprise échouée ({e}) — démarrage depuis epoch 1")
            start_epoch = 0
    else:
        print(f"⚠️  Checkpoint introuvable: {resume_from} — démarrage depuis epoch 1")

for epoch in range(start_epoch, NUM_EPOCHS):
    epoch_start = time.time()
    effective_lr = max(LEARNING_RATE * (LR_DECAY ** epoch), MIN_LEARNING_RATE)
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS} | lr={effective_lr:.6f}")
    log_memory()
    try:
        train_loss, train_acc = train_epoch(model, X_train, y_train, genres_train, BATCH_SIZE, effective_lr)
        val_loss, val_acc = evaluate(model, X_val, y_val, genres_val, BATCH_SIZE)
    except Exception as e:
        print(f"  ❌ Erreur epoch {epoch + 1}: {e}")
        traceback.print_exc()
        print("  ⚠️  Tentative de continuer avec l'époque suivante...")
        continue

    if np.isnan(train_loss) or np.isnan(val_loss):
        print(f"  ❌ NaN détecté epoch {epoch + 1} — arrêt de l'entraînement")
        break

    train_ppl = compute_perplexity(train_loss)
    val_ppl = compute_perplexity(val_loss)
    epoch_time = time.time() - epoch_start

    model.loss_history.append(train_loss)
    model.val_loss_history.append(val_loss)

    improved = val_loss < best_val_loss
    if improved:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        patience_counter = 0
        tag = " ✓ best"
        # Save best model with full inference package
        try:
            best_pkg = {
                'model_weights': {
                    'word_embedding': model.word_embedding,
                    'genre_embedding': model.genre_embedding,
                    'W1': model.W1, 'b1': model.b1,
                    'W2': model.W2, 'b2': model.b2,
                },
                'vocab': {'word2idx': word2idx, 'idx2word': idx2word},
                'config': {
                    'embedding_dim': EMBEDDING_DIM, 'hidden_dim': HIDDEN_DIM,
                    'seq_len': SEQ_LEN, 'num_genres': NUM_GENRES,
                    'genres': list(genre_encoder.classes_),
                },
                'constants': {
                    'PAD_IDX': PAD_IDX, 'UNK_IDX': UNK_IDX,
                    'BOS_IDX': BOS_IDX, 'EOS_IDX': EOS_IDX,
                },
                'epoch': epoch + 1,
            }
            best_path = get_output_path('lyrics_model_best.pkl')
            with open(best_path, 'wb') as f:
                pickle.dump(best_pkg, f)
        except Exception as e:
            print(f"  ⚠️  Best model save échoué: {e}")
    else:
        patience_counter += 1
        tag = f" (patience {patience_counter}/{EARLY_STOP_PATIENCE})"

    print(f"{epoch+1:6d} | {train_loss:10.6f} | {train_acc:9.4f} | {val_loss:8.6f} | {val_acc:7.4f} | {train_ppl:9.4f} | {val_ppl:7.4f} | {epoch_time:5.1f}s{tag}")

    # Checkpoint save
    if (epoch + 1) % CHECKPOINT_EVERY == 0:
        ckpt_path = get_output_path(f'checkpoint_epoch{epoch+1}.pkl')
        try:
            ckpt_data = {
                'epoch': epoch + 1,
                'model_weights': {
                    'word_embedding': model.word_embedding,
                    'genre_embedding': model.genre_embedding,
                    'W1': model.W1, 'b1': model.b1,
                    'W2': model.W2, 'b2': model.b2,
                },
                'vocab': {'word2idx': word2idx, 'idx2word': idx2word},
                'config': {
                    'embedding_dim': EMBEDDING_DIM, 'hidden_dim': HIDDEN_DIM,
                    'seq_len': SEQ_LEN, 'num_genres': NUM_GENRES,
                    'genres': list(genre_encoder.classes_),
                },
                'constants': {
                    'PAD_IDX': PAD_IDX, 'UNK_IDX': UNK_IDX,
                    'BOS_IDX': BOS_IDX, 'EOS_IDX': EOS_IDX,
                },
                'loss_history': model.loss_history,
                'val_loss_history': model.val_loss_history,
            }
            with open(ckpt_path, 'wb') as f:
                pickle.dump(ckpt_data, f)
            print(f"  💾 Checkpoint sauvegardé: {ckpt_path}")
        except Exception as e:
            print(f"  ⚠️  Checkpoint échoué: {e}")

    # Early stopping
    if patience_counter >= EARLY_STOP_PATIENCE:
        print(f"\n🛑 Early stopping: val_loss n'a pas amélioré depuis {EARLY_STOP_PATIENCE} époques")
        print(f"   Meilleure val_loss: {best_val_loss:.6f} à l'époque {best_epoch}")
        break

total_time = time.time() - training_start
print(f"\n✓ Entraînement terminé en {total_time:.1f}s ({total_time/60:.1f} min)")
print(f"✓ Meilleure val_loss: {best_val_loss:.6f} (époque {best_epoch})")

print("\n" + "="*60)
print("SECTION 9: Génération de Paroles")
print("="*60)

def generate_lyrics(model, genre_idx, max_length=50, temperature=1.0):
    generated_tokens = [BOS_IDX]
    context_buffer = generated_tokens[-SEQ_LEN:] if len(generated_tokens) >= 1 else generated_tokens
    
    for _ in range(max_length):
        X_input = xp.array([pad_sequence(context_buffer, SEQ_LEN)], dtype=xp.int32)
        genres_input = xp.array([genre_idx], dtype=xp.int32)
        
        logits, _, _, _, _ = model.forward(X_input, genres_input, training=False, dropout_rate=0.0)
        logits = logits[0] / temperature
        
        exp_logits = xp.exp(logits - xp.max(logits))
        probs = exp_logits / xp.sum(exp_logits)
        sampled = xp.random.choice(len(probs), size=1, p=probs)
        token_value = sampled[0]
        next_token = int(token_value.item() if hasattr(token_value, 'item') else token_value)
        
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


def generation_quality_metrics(text):
    tokens = text.split()
    if not tokens:
        return {
            'token_count': 0,
            'unique_ratio': 0.0,
            'repeat_bigram_rate': 0.0,
            'repeat_trigram_rate': 0.0,
        }

    token_count = len(tokens)
    unique_ratio = len(set(tokens)) / token_count

    bigrams = list(zip(tokens, tokens[1:]))
    trigrams = list(zip(tokens, tokens[1:], tokens[2:]))
    repeat_bigram_rate = 0.0 if not bigrams else 1.0 - (len(set(bigrams)) / len(bigrams))
    repeat_trigram_rate = 0.0 if not trigrams else 1.0 - (len(set(trigrams)) / len(trigrams))

    return {
        'token_count': token_count,
        'unique_ratio': unique_ratio,
        'repeat_bigram_rate': repeat_bigram_rate,
        'repeat_trigram_rate': repeat_trigram_rate,
    }

print("\nGénération par genre:")
generation_metrics = {}
for genre_idx, genre_name in enumerate(genre_encoder.classes_):
    tokens = generate_lyrics(model, genre_idx, max_length=40, temperature=0.8)
    lyrics = tokens_to_lyrics(tokens)
    generation_metrics[genre_name] = generation_quality_metrics(lyrics)
    print(f"\n{genre_name.upper()}:")
    print(f"  {lyrics[:150]}...")
    gm = generation_metrics[genre_name]
    print(f"  [Qualité] tokens={gm['token_count']}, unique={gm['unique_ratio']:.2f}, rep2={gm['repeat_bigram_rate']:.2f}, rep3={gm['repeat_trigram_rate']:.2f}")

print("\n" + "="*60)
print("SECTION 10: Visualisations et Statistiques")
print("="*60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Statistiques du Modèle de Génération de Paroles', fontsize=16, fontweight='bold')

epochs_range = range(1, len(model.loss_history) + 1)

ax1 = axes[0, 0]
ax1.plot(epochs_range, model.loss_history, 'b-o', label='Train', linewidth=2, markersize=6)
ax1.plot(epochs_range, model.val_loss_history, 'r-s', label='Validation', linewidth=2, markersize=6)
ax1.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax1.set_ylabel('Loss (Cross-Entropy)', fontsize=11, fontweight='bold')
ax1.set_title('Loss: Train vs Validation', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

train_ppl = [compute_perplexity(loss) for loss in model.loss_history]
val_ppl = [compute_perplexity(loss) for loss in model.val_loss_history]

ax2 = axes[0, 1]
ax2.plot(epochs_range, train_ppl, 'g-o', label='Train', linewidth=2, markersize=6)
ax2.plot(epochs_range, val_ppl, 'orange', marker='s', label='Validation', linewidth=2, markersize=6)
ax2.set_xlabel('Epoch', fontsize=11, fontweight='bold')
ax2.set_ylabel('Perplexity', fontsize=11, fontweight='bold')
ax2.set_title('Perplexity: Train vs Validation', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

genre_counts = df_filtered['playlist_genre'].value_counts()
ax3 = axes[1, 0]
colors = plt.cm.Set3(range(len(genre_counts)))
bars = ax3.bar(range(len(genre_counts)), genre_counts.values, color=colors, edgecolor='black', linewidth=1.5)
ax3.set_xticks(range(len(genre_counts)))
ax3.set_xticklabels(genre_counts.index, rotation=45, ha='right')
ax3.set_ylabel('Nombre de Chansons', fontsize=11, fontweight='bold')
ax3.set_title('Distribution des Genres', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height):,}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

seq_lengths = df_filtered['tokens'].str.len()
ax4 = axes[1, 1]
ax4.hist(seq_lengths, bins=50, color='purple', edgecolor='black', alpha=0.7)
ax4.axvline(seq_lengths.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {seq_lengths.mean():.1f}')
ax4.axvline(seq_lengths.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {seq_lengths.median():.1f}')
ax4.set_xlabel('Longueur de Séquence (tokens)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Fréquence', fontsize=11, fontweight='bold')
ax4.set_title(f'Distribution des Longueurs (Total: {len(X_train):,} paires)', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot_path = get_output_path('training_stats.png')
try:
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Graphiques sauvegardés: {plot_path}")
except Exception as e:
    print(f"  ⚠️  Impossible de sauvegarder les graphiques: {e}")
try:
    plt.show()
except Exception:
    print("  ℹ️  plt.show() ignoré (environnement sans affichage graphique)")

print("\n📊 STATISTIQUES FINALES:")
print(f"  • Meilleur Train Loss: {min(model.loss_history):.4f}")
print(f"  • Meilleur Val Loss: {min(model.val_loss_history):.4f}")
print(f"  • Train Perplexity (epoch 1): {compute_perplexity(model.loss_history[0]):.4f}")
print(f"  • Val Perplexity (epoch 1): {compute_perplexity(model.val_loss_history[0]):.4f}")
print(f"  • Train Perplexity (final): {compute_perplexity(model.loss_history[-1]):.4f}")
print(f"  • Val Perplexity (final): {compute_perplexity(model.val_loss_history[-1]):.4f}")
print(f"  • Longueur moyenne paroles: {seq_lengths.mean():.1f} tokens")
print(f"  • Longueur max paroles: {seq_lengths.max():.0f} tokens")
print(f"  • Total paires (input, target): {len(X_train) + len(X_val):,}")

print("\n" + "="*60)
print("SECTION 11: Sauvegarde du Modèle")
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
        'embedding_dim': EMBEDDING_DIM,
        'hidden_dim': HIDDEN_DIM,
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

model_path = get_output_path('lyrics_model.pkl')
try:
    with open(model_path, 'wb') as f:
        pickle.dump(inference_package, f)
    print(f"✓ Modèle sauvegardé: {model_path}")
    print(f"✓ Taille: {os.path.getsize(model_path) / 1024:.1f} KB")
except Exception as e:
    print(f"❌ Sauvegarde échouée: {e}")
    fallback_path = os.path.join(get_project_root(), 'lyrics_model_backup.pkl')
    try:
        with open(fallback_path, 'wb') as f:
            pickle.dump(inference_package, f)
        print(f"  💾 Sauvegarde de secours: {fallback_path}")
    except Exception as e2:
        print(f"  ❌ Sauvegarde de secours aussi échouée: {e2}")

run_metadata = {
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'git_commit': '',
    'config': {
        'LANG_FILTER': LANG_FILTER,
        'MAX_SAMPLES': MAX_SAMPLES,
        'MIN_FREQ': MIN_FREQ,
        'MAX_VOCAB_SIZE': MAX_VOCAB_SIZE,
        'SEQ_LEN': SEQ_LEN,
        'EMBEDDING_DIM': EMBEDDING_DIM,
        'HIDDEN_DIM': HIDDEN_DIM,
        'NUM_EPOCHS': NUM_EPOCHS,
        'BATCH_SIZE': BATCH_SIZE,
        'LEARNING_RATE': LEARNING_RATE,
        'LR_DECAY': LR_DECAY,
        'MIN_LEARNING_RATE': MIN_LEARNING_RATE,
        'DROPOUT_RATE': DROPOUT_RATE,
        'LABEL_SMOOTHING': LABEL_SMOOTHING,
        'CHECKPOINT_EVERY': CHECKPOINT_EVERY,
        'EARLY_STOP_PATIENCE': EARLY_STOP_PATIENCE,
    },
    'dataset': {
        'songs_total': len(df),
        'songs_filtered': len(df_filtered),
        'songs_train': len(song_train_df),
        'songs_val': len(song_val_df),
        'vocab_size': len(word2idx),
        'genres': list(genre_encoder.classes_),
    },
    'metrics': {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'train_loss_final': model.loss_history[-1] if model.loss_history else None,
        'val_loss_final': model.val_loss_history[-1] if model.val_loss_history else None,
        'train_ppl_final': compute_perplexity(model.loss_history[-1]) if model.loss_history else None,
        'val_ppl_final': compute_perplexity(model.val_loss_history[-1]) if model.val_loss_history else None,
        'generation_quality_by_genre': generation_metrics,
    },
    'artifacts': {
        'model_path': model_path,
        'plot_path': plot_path,
    }
}

try:
    run_metadata['git_commit'] = subprocess.check_output(
        ['git', 'rev-parse', '--short', 'HEAD'],
        cwd=get_project_root(),
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()
except Exception:
    run_metadata['git_commit'] = ''

metadata_path = get_output_path('run_metadata.json')
try:
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(run_metadata, f, indent=2, ensure_ascii=False)
    print(f"✓ Métadonnées run sauvegardées: {metadata_path}")
except Exception as e:
    print(f"⚠️  Sauvegarde métadonnées échouée: {e}")

print("\n" + "="*60)
print("RÉSUMÉ DU TP")
print("="*60)
print(f"✓ Dataset: {len(df_filtered)} chansons")
print(f"✓ Vocabulaire: {len(word2idx):,} tokens")
print(f"✓ Genres: {NUM_GENRES}")
print(f"✓ Exemples: {len(X_train):,} train / {len(X_val):,} val")
print(f"\n📊 MÉTRIQUES FINALES:")
print(f"✓ Train Loss: {model.loss_history[-1]:.4f}")
print(f"✓ Val Loss: {model.val_loss_history[-1]:.4f}")
print(f"✓ Train Perplexity: {compute_perplexity(model.loss_history[-1]):.4f}")
print(f"✓ Val Perplexity: {compute_perplexity(model.val_loss_history[-1]):.4f}")
print(f"\n📈 Graphiques sauvegardés: {plot_path}")
print(f"💾 Modèle sauvegardé: {model_path}")
print("="*60)
