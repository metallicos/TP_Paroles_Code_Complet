"""
GPT-2 Fine-tuning pour la Génération de Paroles de Chanson
============================================================
Comparaison avec le modèle Feed-Forward custom (TP_Paroles_Code_Complet.py)

Prérequis:
    pip install transformers datasets accelerate torch

Usage:
    python3 gpt2_finetune.py              # entraîner + générer
    python3 gpt2_finetune.py --only-gen   # générer depuis checkpoint existant
"""

import os
import sys
import argparse
import time
import json
import pickle
import re
import numpy as np
import pandas as pd
from pathlib import Path

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--only-gen', action='store_true', help='Skip training, run generation only')
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--batch', type=int, default=4)
parser.add_argument('--max-samples', type=int, default=2000, help='Max songs to use (None=all)')
parser.add_argument('--genre', type=str, default='pop')
parser.add_argument('--samples', type=int, default=3)
parser.add_argument('--length', type=int, default=80)
parser.add_argument('--temperature', type=float, default=0.9)
parser.add_argument('--top-p', type=float, default=0.92)
parser.add_argument('--top-k', type=int, default=50)
args = parser.parse_args()

# ── imports HuggingFace ───────────────────────────────────────────────────────
try:
    from transformers import (
        GPT2LMHeadModel,
        GPT2Tokenizer,
        Trainer,
        TrainingArguments,
        DataCollatorForLanguageModeling,
        set_seed,
    )
    from datasets import Dataset
    import torch
except ImportError:
    print("❌ Dépendances manquantes. Installer avec:")
    print("   pip install transformers datasets accelerate torch")
    sys.exit(1)

set_seed(42)

PROJECT_ROOT = Path(__file__).parent
OUTPUT_DIR   = PROJECT_ROOT / 'outputs' / 'gpt2_finetune'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_ID = 'gpt2'  # 117M params, téléchargeable sans compte


# ══════════════════════════════════════════════════════════════════════════════
# 1. CHARGEMENT & PRÉTRAITEMENT DES DONNÉES
# ══════════════════════════════════════════════════════════════════════════════
print("="*60)
print("SECTION 1: Chargement du Dataset")
print("="*60)

csv_path = PROJECT_ROOT / 'spotify_songs.csv'
if not csv_path.exists():
    print(f"❌ Dataset non trouvé: {csv_path}")
    sys.exit(1)

df = pd.read_csv(csv_path)
df = df[df['lyrics'].notna()].copy()
df = df[df['lyrics'].str.len() > 50].copy()
df = df[df['lyrics'] != 'NA'].copy()

# Filtre langue anglaise
if 'language' in df.columns:
    df = df[df['language'] == 'en'].copy()

# Filtre genre
genre_filter = args.genre.lower()
top_genres   = df['playlist_genre'].value_counts().head(5).index.tolist()
df_genre     = df[df['playlist_genre'].str.lower().isin([genre_filter])]
if len(df_genre) == 0:
    print(f"⚠️  Genre '{genre_filter}' introuvable — utilisation de tous les genres")
    df_genre = df[df['playlist_genre'].isin(top_genres)]

if args.max_samples and args.max_samples < len(df_genre):
    df_genre = df_genre.sample(n=args.max_samples, random_state=42)

print(f"✓ {len(df_genre)} chansons ({genre_filter}) chargées")


def clean_lyric(text):
    text = text.lower()
    text = text.replace('\n', ' \n ')
    text = re.sub(r'[^a-z0-9\s\'\-,.:!?()\n]', '', text)
    text = re.sub(r' +', ' ', text).strip()
    return text


lyrics_list = [clean_lyric(l) for l in df_genre['lyrics'].tolist()]

print(f"✓ Exemple: {lyrics_list[0][:120]}...")


# ══════════════════════════════════════════════════════════════════════════════
# 2. TOKENIZER & DATASET
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 2: Tokenizer GPT-2")
print("="*60)

tokenizer = GPT2Tokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 n'a pas de pad token natif

MAX_LEN = 256  # tokens par exemple

def tokenize_fn(examples):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=MAX_LEN,
        padding='max_length',
    )

hf_dataset = Dataset.from_dict({'text': lyrics_list})
hf_dataset = hf_dataset.train_test_split(test_size=0.1, seed=42)
tokenized   = hf_dataset.map(tokenize_fn, batched=True, remove_columns=['text'])

print(f"✓ Train: {len(tokenized['train'])} | Val: {len(tokenized['test'])}")

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # CLM (Causal Language Modeling), pas MLM
)


# ══════════════════════════════════════════════════════════════════════════════
# 3. MODÈLE GPT-2
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 3: Chargement du Modèle GPT-2")
print("="*60)

model = GPT2LMHeadModel.from_pretrained(MODEL_ID)

params_total    = sum(p.numel() for p in model.parameters())
params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✓ GPT-2 chargé: {params_total/1e6:.1f}M paramètres totaux, {params_trainable/1e6:.1f}M entraînables")
print(f"  Couches Transformer: 12 | Têtes d'attention: 12 | Embedding dim: 768")


# ══════════════════════════════════════════════════════════════════════════════
# 4. FINE-TUNING
# ══════════════════════════════════════════════════════════════════════════════
if not args.only_gen:
    print("\n" + "="*60)
    print("SECTION 4: Fine-tuning")
    print("="*60)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR / 'checkpoints'),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=50,
        logging_dir=str(OUTPUT_DIR / 'logs'),
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        report_to='none',
        dataloader_drop_last=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized['train'],
        eval_dataset=tokenized['test'],
        data_collator=data_collator,
    )

    t0 = time.time()
    print(f"Démarrage fine-tuning: {args.epochs} époques, batch={args.batch}")
    train_result = trainer.train()
    elapsed = time.time() - t0

    print(f"\n✓ Fine-tuning terminé en {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Train Loss final: {train_result.training_loss:.4f}")

    # Sauvegarde
    model.save_pretrained(str(OUTPUT_DIR / 'model'))
    tokenizer.save_pretrained(str(OUTPUT_DIR / 'model'))
    print(f"✓ Modèle sauvegardé: {OUTPUT_DIR / 'model'}")

    # Métriques de comparaison
    eval_result = trainer.evaluate()
    gpt2_val_loss = eval_result['eval_loss']
    gpt2_val_ppl  = float(np.exp(gpt2_val_loss))
    print(f"\n📊 Métriques GPT-2 fine-tuné:")
    print(f"  Val Loss: {gpt2_val_loss:.4f}")
    print(f"  Val PPL:  {gpt2_val_ppl:.2f}")

    # Comparaison avec notre modèle custom
    meta_path = PROJECT_ROOT / 'outputs' / 'run_metadata.json'
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        custom_ppl = meta.get('metrics', {}).get('val_ppl_final', None)
        print(f"\n📊 COMPARAISON:")
        print(f"  {'Modèle':<30} {'Val PPL':>10}")
        print(f"  {'-'*42}")
        print(f"  {'Feed-Forward custom (TP)':<30} {custom_ppl:>10.2f}")
        print(f"  {'GPT-2 fine-tuné (HuggingFace)':<30} {gpt2_val_ppl:>10.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. GÉNÉRATION DE PAROLES
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SECTION 5: Génération de Paroles")
print("="*60)

# Charger depuis checkpoint si --only-gen
gen_model_path = OUTPUT_DIR / 'model'
if args.only_gen:
    if not gen_model_path.exists():
        print(f"❌ Aucun checkpoint trouvé: {gen_model_path}")
        print("   Lancer d'abord sans --only-gen pour fine-tuner.")
        sys.exit(1)
    model     = GPT2LMHeadModel.from_pretrained(str(gen_model_path))
    tokenizer = GPT2Tokenizer.from_pretrained(str(gen_model_path))
    tokenizer.pad_token = tokenizer.eos_token

model.eval()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

seed_text = f"[{genre_filter}]"  # prompt de départ

print(f"\n🎵 Génération de {args.samples} paroles pour le genre: {genre_filter.upper()}")
print("-"*60)

for i in range(args.samples):
    inputs = tokenizer.encode(seed_text, return_tensors='pt').to(device)
    with torch.no_grad():
        output = model.generate(
            inputs,
            max_new_tokens=args.length,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=True,
            repetition_penalty=1.4,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    # Supprimer le prompt du début
    generated = generated[len(seed_text):].strip()
    print(f"\n📝 Exemple {i+1}:")
    print(f"{generated}")

print("\n" + "="*60)
print("✓ Génération GPT-2 terminée")
print("="*60)
