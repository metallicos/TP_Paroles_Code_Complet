import pickle
import numpy as np
import argparse
import sys
import os


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
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
        
        with open(model_path, 'rb') as f:
            pkg = pickle.load(f)
        
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
    
    def generate(self, genre, max_length=50, temperature=1.0, seed_tokens=None):
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
            
            logits = self.model.forward(X_input, genres_input, self.weights)
            logits = logits[0] / temperature
            
            exp_logits = xp.exp(logits - xp.max(logits))
            probs = exp_logits / xp.sum(exp_logits)
            next_token = int(xp.random.choice(len(probs), p=probs).item())
            
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
    
    def generate_multiple(self, genre, num_samples=3, max_length=50, temperature=0.8):
        print(f"\n🎵 Génération de {num_samples} paroles pour le genre: {genre.upper()}")
        print("-" * 60)
        
        for i in range(num_samples):
            print(f"\n📝 Exemple {i+1}:")
            lyrics = self.generate(genre, max_length, temperature)
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
    
    args = parser.parse_args()

    if USING_GPU:
        print("✓ Backend: GPU (CuPy)")
    else:
        print("✓ Backend: CPU (NumPy)")
    
    model_path = args.model if args.model else get_output_path('lyrics_model.pkl')
    
    try:
        generator = LyricsGenerator(model_path)
    except FileNotFoundError:
        print(f"❌ Erreur: Le modèle n'existe pas à '{model_path}'")
        print("   Entraînez d'abord le modèle avec TP_Paroles_Code_Complet.py")
        sys.exit(1)
    
    if args.list_genres:
        print("\n📋 Genres disponibles:")
        for genre in generator.list_genres():
            print(f"  - {genre}")
        print()
        return
    
    if args.samples > 1:
        generator.generate_multiple(args.genre, args.samples, args.length, args.temperature)
    else:
        print(f"\n🎵 Génération de paroles pour le genre: {args.genre.upper()}")
        print(f"   (Température: {args.temperature}, Max tokens: {args.length})")
        print("-" * 60)
        lyrics = generator.generate(args.genre, args.length, args.temperature)
        print(lyrics)
        print()


if __name__ == '__main__':
    main()
