import pickle
import numpy as np
import argparse
import sys
import os
import re
from collections import Counter


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


def get_output_path(filename):
    return os.path.join(get_project_root(), 'outputs', filename)


def rebuild_vocab_from_csv(csv_path, max_samples=None):
    """Reconstruct vocab/config/constants from the original CSV.
    Must use the same hyperparameters as the training script."""
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    MIN_FREQ = 2
    MAX_VOCAB_SIZE = 20000
    TOP_GENRES = 5
    SEQ_LEN = 10
    EMBED_DIM = 16
    HIDDEN_DIM = 32
    SPECIAL_TOKENS = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']

    print(f"  → Reconstruction du vocabulaire depuis: {os.path.basename(csv_path)}")
    df = pd.read_csv(csv_path)
    df = df[df['lyrics'].notna()].copy()
    df = df[df['lyrics'].str.len() > 50].copy()
    df = df[df['lyrics'] != 'NA'].copy()

    # Apply same language filter as training
    lang_filter = os.getenv("LANG_FILTER", "en").strip() or None
    if lang_filter and 'language' in df.columns:
        df = df[df['language'] == lang_filter].copy()

    if max_samples and max_samples < len(df):
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
        print(f"  → MAX_SAMPLES={max_samples} appliqué")

    def preprocess_text(text):
        text = text.lower()
        text = text.replace('\n', ' <NEW_LINE> ')
        text = re.sub(r"[^a-z0-9\s\'\-,.:!?()]", '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['tokens'] = df['lyrics'].apply(lambda t: preprocess_text(t).split())

    all_tokens = []
    for tl in df['tokens']:
        all_tokens.extend(tl)
    word_freq = Counter(all_tokens)
    filtered = [(w, f) for w, f in word_freq.items() if f >= MIN_FREQ]
    filtered.sort(key=lambda x: x[1], reverse=True)
    filtered = filtered[:MAX_VOCAB_SIZE]
    vocab_words = {w for w, _ in filtered}

    word2idx = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for idx_w, word in enumerate(sorted(vocab_words), start=len(SPECIAL_TOKENS)):
        word2idx[word] = idx_w
    idx2word = {i: w for w, i in word2idx.items()}

    top_genres = df['playlist_genre'].value_counts().head(TOP_GENRES).index.tolist()
    df_filtered = df[df['playlist_genre'].isin(top_genres)].copy()
    genre_encoder = LabelEncoder()
    genre_encoder.fit(df_filtered['playlist_genre'])
    num_genres = len(genre_encoder.classes_)

    PAD_IDX = word2idx['<PAD>']
    UNK_IDX = word2idx['<UNK>']
    BOS_IDX = word2idx['<BOS>']
    EOS_IDX = word2idx['<EOS>']

    print(f"  → Vocabulaire: {len(word2idx):,} tokens | Genres: {list(genre_encoder.classes_)}")
    return {
        'vocab': {'word2idx': word2idx, 'idx2word': idx2word},
        'config': {
            'embedding_dim': EMBED_DIM, 'hidden_dim': HIDDEN_DIM,
            'seq_len': SEQ_LEN, 'num_genres': num_genres,
            'genres': list(genre_encoder.classes_),
        },
        'constants': {
            'PAD_IDX': PAD_IDX, 'UNK_IDX': UNK_IDX,
            'BOS_IDX': BOS_IDX, 'EOS_IDX': EOS_IDX,
        },
    }


class LyricsGenerationModel:
    def __init__(self, vocab_size, embedding_dim=16, hidden_dim=32, num_genres=5):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
    def forward(self, X_batch, genres_batch, weights):
        batch_size = X_batch.shape[0]
        word_embeds = weights['word_embedding'][X_batch]
        word_embeds_flat = word_embeds.reshape(batch_size, -1)
        genre_embeds = weights['genre_embedding'][genres_batch]
        combined = xp.concatenate([word_embeds_flat, genre_embeds], axis=1)
        
        z1 = xp.dot(combined, weights['W1']) + weights['b1']
        a1 = xp.maximum(z1, 0)
        z2 = xp.dot(a1, weights['W2']) + weights['b2']
        
        return z2


class LyricsGenerator:
    def __init__(self, model_path, csv_path=None, max_samples=None):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
        
        with open(model_path, 'rb') as f:
            pkg = pickle.load(f)
        
        if 'vocab' not in pkg:
            if csv_path:
                print(f"  ⚠️  Ancien checkpoint détecté — reconstruction du vocabulaire depuis le CSV...")
                rebuilt = rebuild_vocab_from_csv(csv_path, max_samples)
                pkg.update(rebuilt)
            else:
                raise KeyError(
                    f"Le fichier '{os.path.basename(model_path)}' est un ancien checkpoint sans vocabulaire.\n"
                    "  → Option 1: ajoutez --csv spotify_songs.csv pour reconstruire le vocabulaire automatiquement.\n"
                    "  → Option 2: utilisez outputs/lyrics_model_best.pkl\n"
                    "  → Option 3: relancez l'entraînement (les nouveaux checkpoints incluent le vocabulaire)."
                )

        self.weights = pkg['model_weights']
        self.word2idx = pkg['vocab']['word2idx']
        self.idx2word = pkg['vocab']['idx2word']
        self.config = pkg['config']
        self.constants = pkg['constants']
        
        self.PAD_IDX = self.constants['PAD_IDX']
        self.UNK_IDX = self.constants['UNK_IDX']
        self.BOS_IDX = self.constants['BOS_IDX']
        self.EOS_IDX = self.constants['EOS_IDX']
        self.SEQ_LEN = self.config['seq_len']
        
        self.model = LyricsGenerationModel(
            vocab_size=len(self.word2idx),
            embedding_dim=self.config['embedding_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_genres=self.config['num_genres']
        )

        if USING_GPU:
            self.weights = {k: xp.asarray(v) for k, v in self.weights.items()}
        
        print("✓ Modèle chargé avec succès!")
        print(f"  Vocabulaire: {len(self.word2idx):,} mots")
        print(f"  Genres: {self.config['num_genres']}")
        print(f"  Genres disponibles: {', '.join(self.config['genres'])}")
    
    def pad_sequence(self, seq, max_len):
        if len(seq) == max_len:
            return seq
        elif len(seq) < max_len:
            return seq + [self.PAD_IDX] * (max_len - len(seq))
        else:
            return seq[-max_len:]
    
    def _topk_probs(self, logits, temperature, top_k):
        """Apply temperature + top-k filtering, return normalized probabilities."""
        logits = logits / max(temperature, 1e-8)
        if top_k and top_k > 0:
            # Zero out everything except the top-k logits
            top_k = min(top_k, len(logits))
            threshold = xp.sort(logits)[-top_k]
            logits = xp.where(logits >= threshold, logits, xp.full_like(logits, -1e9))
        exp_logits = xp.exp(logits - xp.max(logits))
        return exp_logits / xp.sum(exp_logits)

    def generate(self, genre, max_length=50, temperature=1.0, seed_tokens=None, top_k=40):
        genre_idx = None
        genre_lower = genre.lower()
        
        for idx, g in enumerate(self.config['genres']):
            if g.lower() == genre_lower:
                genre_idx = idx
                break
        
        if genre_idx is None:
            return f"❌ Genre '{genre}' non trouvé.\nGenres disponibles: {', '.join(self.config['genres'])}"
        
        generated_tokens = [self.BOS_IDX]
        context_buffer = list(generated_tokens)
        
        for step in range(max_length):
            X_input = xp.array([self.pad_sequence(context_buffer, self.SEQ_LEN)], dtype=xp.int32)
            genres_input = xp.array([genre_idx], dtype=xp.int32)
            
            logits = self.model.forward(X_input, genres_input, self.weights)[0]
            probs = self._topk_probs(logits, temperature, top_k)
            probs_np = np.array(probs, dtype=np.float64)
            probs_np = probs_np / probs_np.sum()  # ensure sums to 1 (float precision)
            next_token = int(np.random.choice(len(probs_np), p=probs_np))
            
            if next_token == self.EOS_IDX:
                break
            
            generated_tokens.append(next_token)
            context_buffer.append(next_token)
            
            if len(context_buffer) > self.SEQ_LEN:
                context_buffer = context_buffer[-self.SEQ_LEN:]
        
        return self._tokens_to_lyrics(generated_tokens)
    
    def _tokens_to_lyrics(self, token_indices):
        words = []
        for token_idx in token_indices:
            if token_idx == self.EOS_IDX:
                break
            word = self.idx2word.get(token_idx, '<UNK>')
            if word not in ['<PAD>', '<BOS>', '<EOS>', '<UNK>']:
                words.append(word)
        return ' '.join(words) if words else "(Paroles vides - essayez avec une autre température)"
    
    def generate_multiple(self, genre, num_samples=3, max_length=50, temperature=0.8, top_k=40):
        print(f"\n🎵 Génération de {num_samples} paroles pour le genre: {genre.upper()}")
        print("-" * 60)
        
        for i in range(num_samples):
            print(f"\n📝 Exemple {i+1}:")
            lyrics = self.generate(genre, max_length, temperature, top_k=top_k)
            print(lyrics)
            print()
    
    def list_genres(self):
        return self.config['genres']


def main():
    parser = argparse.ArgumentParser(
        description="Générer des paroles de chanson avec le modèle ML entraîné"
    )
    
    parser.add_argument('--model', type=str, 
                       default=None,
                       help='Chemin du modèle (par défaut: outputs/lyrics_model.pkl)')
    
    parser.add_argument('--genre', type=str,
                       required=True,
                       help='Genre musical (rock, pop, hip-hop, etc.)')
    
    parser.add_argument('--length', type=int,
                       default=50,
                       help='Longueur max des paroles (en tokens)')
    
    parser.add_argument('--temperature', type=float,
                       default=0.8,
                       help='Temperature (< 1 = déterministe, > 1 = aléatoire)')
    
    parser.add_argument('--samples', type=int,
                       default=1,
                       help='Nombre de paroles à générer')
    
    parser.add_argument('--list-genres', action='store_true',
                       help='Afficher les genres disponibles')

    parser.add_argument('--top-k', type=int, default=40,
                       help='Top-k sampling: limiter aux k tokens les plus probables (défaut: 40, 0=désactivé)')

    parser.add_argument('--csv', type=str, default=None,
                       help='Chemin vers spotify_songs.csv pour reconstruire le vocabulaire '
                            "d'un ancien checkpoint (ex: --csv spotify_songs.csv)")

    parser.add_argument('--max-samples', type=int, default=None,
                       help='MAX_SAMPLES utilisé pendant l\'entraînement (pour reproduire le même vocabulaire)')

    args = parser.parse_args()

    if USING_GPU:
        print("✓ Backend: GPU (CuPy)")
    else:
        print("✓ Backend: CPU (NumPy)")
    
    if args.model:
        model_path = args.model
    else:
        # Try final model first, then best checkpoint, then any checkpoint
        candidates = [
            get_output_path('lyrics_model.pkl'),
            get_output_path('lyrics_model_best.pkl'),
        ]
        # Also look for the latest checkpoint
        import glob
        ckpts = sorted(glob.glob(get_output_path('checkpoint_epoch*.pkl')))
        if ckpts:
            candidates.append(ckpts[-1])
        model_path = next((p for p in candidates if os.path.exists(p)), candidates[0])
        if os.path.exists(model_path):
            print(f"ℹ️  Modèle trouvé: {os.path.basename(model_path)}")

    csv_path = args.csv
    if csv_path and not os.path.isabs(csv_path):
        # Resolve relative to project root
        csv_path = os.path.join(get_project_root(), csv_path)

    try:
        generator = LyricsGenerator(model_path, csv_path=csv_path, max_samples=args.max_samples)
    except FileNotFoundError:
        print(f"❌ Erreur: Aucun modèle trouvé.")
        print(f"   Cherché: lyrics_model.pkl, lyrics_model_best.pkl, checkpoint_epoch*.pkl")
        print(f"   → Entraînez d'abord: python3 TP_Paroles_Code_Complet.py")
        sys.exit(1)
    except KeyError as e:
        print(f"❌ {e}")
        sys.exit(1)
    
    if args.list_genres:
        print("\n📋 Genres disponibles:")
        for genre in generator.list_genres():
            print(f"  - {genre}")
        print()
        return
    
    if args.samples > 1:
        generator.generate_multiple(args.genre, args.samples, args.length, args.temperature, top_k=args.top_k)
    else:
        print(f"\n🎵 Génération de paroles pour le genre: {args.genre.upper()}")
        print(f"   (Température: {args.temperature}, Max tokens: {args.length}, Top-k: {args.top_k})")
        print("-" * 60)
        lyrics = generator.generate(args.genre, args.length, args.temperature, top_k=args.top_k)
        print(lyrics)
        print()


if __name__ == '__main__':
    main()
