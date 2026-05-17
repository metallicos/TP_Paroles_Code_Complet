"""
run_all.py — Script de lancement complet du TP
================================================
Lance dans l'ordre :
  1. TP_Paroles_Code_Complet.py  (modèle Feed-Forward custom)
  2. gpt2_finetune.py            (GPT-2 fine-tuné, 3 genres x 3 exemples)
  3. make_graphs.py              (6 figures PNG pour la présentation)

Tout est sauvegardé dans :
  run_YYYYMMDD_HHMMSS/
  ├── logs/
  │   ├── custom_training.log
  │   ├── gpt2_finetune.log
  │   └── graphs.log
  ├── graphs/          (copie des PNG produits par make_graphs.py)
  ├── outputs/         (copie du dossier outputs/ : modèles, metadata)
  └── run_summary.txt  (résumé rapide avec métriques clés)

Usage:
    python3 run_all.py                   # tout lancer
    python3 run_all.py --skip-custom     # sauter le training custom (déjà fait)
    python3 run_all.py --skip-gpt2       # sauter GPT-2
    python3 run_all.py --skip-graphs     # sauter la génération de graphiques
    python3 run_all.py --gpu             # forcer USE_GPU=1
"""

import os
import sys
import time
import shutil
import argparse
import subprocess
import re
from datetime import datetime
from pathlib import Path

# ─── Args ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--skip-custom', action='store_true', help='Ne pas lancer TP_Paroles_Code_Complet.py')
parser.add_argument('--skip-gpt2',   action='store_true', help='Ne pas lancer gpt2_finetune.py')
parser.add_argument('--skip-graphs', action='store_true', help='Ne pas générer les graphiques')
parser.add_argument('--gpu',         action='store_true', help='Forcer USE_GPU=1')
parser.add_argument('--gpt2-epochs', type=int, default=3)
parser.add_argument('--gpt2-samples', type=int, default=3)
parser.add_argument('--gpt2-genre',   type=str, default='pop')
args = parser.parse_args()

# ─── Paths ────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent
TS      = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_DIR = ROOT / f'run_{TS}'
LOGS    = RUN_DIR / 'logs'
GRAPHS  = RUN_DIR / 'graphs'
OUTPUTS = RUN_DIR / 'outputs'

for d in [LOGS, GRAPHS, OUTPUTS]:
    d.mkdir(parents=True, exist_ok=True)

PYTHON = sys.executable

# ─── Helpers ──────────────────────────────────────────────────────────────────
SEP = '═' * 70

def header(title):
    print(f'\n{SEP}')
    print(f'  {title}')
    print(f'{SEP}')

def extract_metric(log_text, pattern, group=1, default='N/A'):
    m = re.search(pattern, log_text)
    return m.group(group) if m else default

def run_step(label, cmd, log_path, env=None):
    """Run a subprocess, tee output to console + log file. Returns (returncode, log_text)."""
    header(label)
    print(f'  Commande : {" ".join(str(c) for c in cmd)}')
    print(f'  Log      : {log_path}\n')

    merged_env = {**os.environ}
    if env:
        merged_env.update(env)

    t0 = time.time()
    log_lines = []

    with open(log_path, 'w', encoding='utf-8') as log_f:
        proc = subprocess.Popen(
            [str(c) for c in cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=merged_env,
            cwd=str(ROOT),
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
            log_lines.append(line)
        proc.wait()

    elapsed = time.time() - t0
    status  = '✅ OK' if proc.returncode == 0 else f'❌ ERREUR (code {proc.returncode})'
    print(f'\n  {status} — {elapsed:.1f}s ({elapsed/60:.1f} min)')

    return proc.returncode, ''.join(log_lines)


# ─── Results accumulator ──────────────────────────────────────────────────────
results = {
    'timestamp': TS,
    'run_dir':   str(RUN_DIR),
    'steps':     {},
}

t_total = time.time()

# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 1 : Modèle Feed-Forward Custom
# ══════════════════════════════════════════════════════════════════════════════
if not args.skip_custom:
    env_custom = {}
    if args.gpu:
        env_custom['USE_GPU'] = '1'

    rc, log = run_step(
        label    = 'ÉTAPE 1 — Entraînement Feed-Forward Custom',
        cmd      = [PYTHON, ROOT / 'TP_Paroles_Code_Complet.py'],
        log_path = LOGS / 'custom_training.log',
        env      = env_custom,
    )

    # Parse key metrics from the log
    results['steps']['custom'] = {
        'returncode': rc,
        'val_loss':   extract_metric(log, r'Meilleure val_loss:\s*([\d.]+)'),
        'best_epoch': extract_metric(log, r'val_loss.*?(\d+)\)$', 1),
        'vocab_size': extract_metric(log, r'Vocabulaire:\s*([\d,]+) tokens'),
        'train_pairs':extract_metric(log, r'Train:\s*([\d,]+) \|'),
        'val_ppl':    extract_metric(log, r'Val Perplexity.*?:\s*([\d.]+)'),
    }
else:
    print('\n⏭  Étape 1 ignorée (--skip-custom)')
    results['steps']['custom'] = {'returncode': 'skipped'}

# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 2 : GPT-2 Fine-tuning
# ══════════════════════════════════════════════════════════════════════════════
if not args.skip_gpt2:
    gpt2_cmd = [
        PYTHON, ROOT / 'gpt2_finetune.py',
        '--genre',   args.gpt2_genre,
        '--epochs',  str(args.gpt2_epochs),
        '--samples', str(args.gpt2_samples),
    ]

    rc, log = run_step(
        label    = 'ÉTAPE 2 — Fine-tuning GPT-2 (HuggingFace)',
        cmd      = gpt2_cmd,
        log_path = LOGS / 'gpt2_finetune.log',
    )

    results['steps']['gpt2'] = {
        'returncode': rc,
        'train_loss': extract_metric(log, r'Train Loss final:\s*([\d.]+)'),
        'val_loss':   extract_metric(log, r'Val Loss:\s*([\d.]+)'),
        'val_ppl':    extract_metric(log, r'Val PPL:\s*([\d.]+)'),
    }
else:
    print('\n⏭  Étape 2 ignorée (--skip-gpt2)')
    results['steps']['gpt2'] = {'returncode': 'skipped'}

# ══════════════════════════════════════════════════════════════════════════════
# ÉTAPE 3 : Génération des graphiques
# ══════════════════════════════════════════════════════════════════════════════
if not args.skip_graphs:
    rc, log = run_step(
        label    = 'ÉTAPE 3 — Génération des Graphiques',
        cmd      = [PYTHON, ROOT / 'make_graphs.py'],
        log_path = LOGS / 'graphs.log',
    )
    results['steps']['graphs'] = {'returncode': rc}

    # Copy generated PNGs into run folder
    src_graphs = ROOT / 'outputs' / 'graphs'
    if src_graphs.exists():
        for png in src_graphs.glob('*.png'):
            shutil.copy2(png, GRAPHS / png.name)
        print(f'\n  📁 {len(list(GRAPHS.glob("*.png")))} graphiques copiés → {GRAPHS}')

    # Also copy the old training_stats.png if present
    legacy_png = ROOT / 'outputs' / 'training_stats.png'
    if legacy_png.exists():
        shutil.copy2(legacy_png, GRAPHS / 'training_stats.png')
else:
    print('\n⏭  Étape 3 ignorée (--skip-graphs)')
    results['steps']['graphs'] = {'returncode': 'skipped'}

# ══════════════════════════════════════════════════════════════════════════════
# COPIE DES OUTPUTS (modèles, metadata, etc.)
# ══════════════════════════════════════════════════════════════════════════════
header('Copie des Artéfacts (outputs/)')

src_outputs = ROOT / 'outputs'
collected   = []

for pattern in ['*.json', '*.pkl', '*.png', '*.txt']:
    for f in src_outputs.glob(pattern):
        dest = OUTPUTS / f.name
        shutil.copy2(f, dest)
        collected.append(f.name)

# Copy gpt2 model folder if exists
gpt2_model_dir = src_outputs / 'gpt2_finetune' / 'model'
if gpt2_model_dir.exists():
    dest_gpt2 = OUTPUTS / 'gpt2_model'
    if dest_gpt2.exists():
        shutil.rmtree(dest_gpt2)
    shutil.copytree(gpt2_model_dir, dest_gpt2)
    collected.append('gpt2_model/')

print(f'  ✅ {len(collected)} artéfacts copiés')
for f in collected[:12]:
    print(f'     • {f}')
if len(collected) > 12:
    print(f'     … et {len(collected)-12} autres')

# ══════════════════════════════════════════════════════════════════════════════
# RÉSUMÉ
# ══════════════════════════════════════════════════════════════════════════════
total_time = time.time() - t_total

summary_lines = [
    '=' * 60,
    f'  RÉSUMÉ DU RUN — {TS}',
    '=' * 60,
    '',
    f'  Durée totale : {total_time:.0f}s ({total_time/60:.1f} min)',
    f'  Dossier      : {RUN_DIR}',
    '',
    '  ┌─────────────────────────────────────────────────┐',
    '  │      MÉTRIQUES CLÉS                             │',
    '  ├─────────────────────────────────────────────────┤',
]

c = results['steps'].get('custom', {})
if c.get('returncode') != 'skipped':
    summary_lines += [
        f'  │  Feed-Forward Custom                            │',
        f'  │    Val Loss      : {c.get("val_loss","N/A"):<30}│',
        f'  │    Val PPL       : {c.get("val_ppl","N/A"):<30}│',
        f'  │    Vocab size    : {c.get("vocab_size","N/A"):<30}│',
        f'  ├─────────────────────────────────────────────────┤',
    ]

g = results['steps'].get('gpt2', {})
if g.get('returncode') != 'skipped':
    summary_lines += [
        f'  │  GPT-2 Fine-tuné                                │',
        f'  │    Val Loss      : {g.get("val_loss","N/A"):<30}│',
        f'  │    Val PPL       : {g.get("val_ppl","N/A"):<30}│',
        f'  │    Train Loss    : {g.get("train_loss","N/A"):<30}│',
        f'  ├─────────────────────────────────────────────────┤',
    ]

if c.get('returncode') != 'skipped' and g.get('returncode') != 'skipped':
    try:
        ratio = float(c.get('val_ppl','0').replace(',','')) / float(g.get('val_ppl','1').replace(',',''))
        summary_lines.append(f'  │  GPT-2 ×{ratio:.1f} meilleur PPL vs Custom             │')
        summary_lines.append(f'  ├─────────────────────────────────────────────────┤')
    except Exception:
        pass

summary_lines += [
    f'  │  Graphiques : {len(list(GRAPHS.glob("*.png"))):<3} PNG dans run_{TS}/graphs/   │',
    '  └─────────────────────────────────────────────────┘',
    '',
    '  Fichiers utiles pour la présentation :',
    f'    logs/custom_training.log',
    f'    logs/gpt2_finetune.log',
    f'    graphs/*.png',
    f'    outputs/run_metadata.json',
    '',
    '=' * 60,
]

summary_text = '\n'.join(summary_lines)
print('\n' + summary_text)

with open(RUN_DIR / 'run_summary.txt', 'w', encoding='utf-8') as f:
    f.write(summary_text)

print(f'\n📁 Tout le run est dans : {RUN_DIR}\n')
print('  Envoyez ce dossier (zip) pour enrichir la présentation.')
