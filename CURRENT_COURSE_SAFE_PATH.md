# Current Course-Safe Path (No External APIs)

## Scope
This plan stays **fully course-safe**:
- No external LLM APIs
- No paid inference services
- Train/infer locally with Python + NumPy/CuPy
- Improve code quality, reproducibility, and lyric quality using classical ML/NLP engineering

---

## 1) Audit Summary (Hardcoded & Static Issues)

### A. Hardcoded / portability issues found
1. **`LANCER_TP.sh`**
   - Prints fixed path: `/home/abdou/Public/sites/cours/`
   - Checks `lyrics_model.pkl` at project root, while project now saves in `outputs/lyrics_model.pkl`
   - Uses `python` instead of consistent `python3`

2. **`generate_notebook.py`**
   - Hardcoded absolute save path: `/home/abdou/Public/sites/cours/lyrics_model.pkl`
   - Hardcoded print path for model size

3. **Docs drift / stale defaults** (`README.md`, helper docs)
   - Some sections still mention older defaults (`SEQ_LEN=10`, dims 16/32, etc.) while code evolved
   - Some examples reference legacy model path assumptions

### B. Architecture & evaluation quality issues
1. **Potential data leakage risk**
   - Training pairs are created from each song, then split at pair-level.
   - This can put segments from the same song in both train and val.
   - Result: optimistic validation metrics, less real-world generation quality.

2. **No checkpoint-aware generation in training section**
   - Main script does sectioned generation but does not score generation quality metrics (diversity/repetition)

3. **Config spread across env vars only**
   - Many env vars are good, but no single source of truth config file (`yaml/json`) for reproducible experiments.

4. **Limited test coverage for quality regressions**
   - Existing tests are mostly sanity checks; no regression tests for decoding behavior.

---

## 2) High-Impact Improvements (Course-Safe)

## Priority P0 (Do first)

### P0.1 Remove hardcoded paths/scripts
- `LANCER_TP.sh`
  - Use `SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"`
  - Use `outputs/lyrics_model.pkl` as canonical model path
  - Use `python3` consistently
- `generate_notebook.py`
  - Replace absolute save path with project-root relative helper

**Acceptance:** project runs from any directory/machine without editing scripts.

### P0.2 Fix train/validation split strategy (major quality gain)
- Split **songs first**, then generate training pairs separately for train and val.
- Prevent same-song leakage between train and validation.

**Acceptance:** no song ID appears in both train and validation; validation curve is trustworthy.

### P0.3 Stabilize decode defaults
- Keep current anti-repetition stack:
  - `top-k`, `top-p`, `min-length`, `repeat-penalty`, `no-repeat-window`, `no-repeat-ngram`, `presence/frequency penalty`
- Provide one validated “quality preset” command in README.

**Acceptance:** no frequent crash/NaN; reduced looping in generated text.

---

## Priority P1 (Next)

### P1.1 Centralized config file (non-static, high-quality code)
Add `config/train_config.json` (or `.yaml`) with:
- data filters (`LANG_FILTER`, min length)
- model dims (`SEQ_LEN`, `EMBEDDING_DIM`, `HIDDEN_DIM`)
- optimizer schedule (`LR`, `LR_DECAY`, `MIN_LR`)
- regularization (`DROPOUT`, `LABEL_SMOOTHING`)

CLI/env overrides remain possible.

**Acceptance:** one command reproduces same run from config file.

### P1.2 Experiment tracking file
Save run metadata in `outputs/run_metadata.json`:
- git commit hash
- all effective hyperparameters
- dataset size / vocab size / genre list
- best epoch and metrics

**Acceptance:** every model artifact is traceable.

### P1.3 Add quality metrics for generation
Add automatic post-generation metrics:
- unique token ratio
- repeated n-gram rate
- average line length

**Acceptance:** objective quality comparison between runs.

---

## Priority P2 (Model-quality upgrades still course-safe)

### P2.1 Upgrade architecture to recurrent sequence model
Move from current feed-forward context window to **GRU/LSTM** (still local, no APIs).
- Better long-range structure in lyrics
- Still fully aligned with course-safe ML path

### P2.2 Subword tokenization (BPE/WordPiece-like)
- Reduce OOV / weird rare words
- Better handling of contractions and morphology

### P2.3 Genre-conditioned prompts in training data
- Structured sequence format: `[GENRE] [BOS] ... [EOS]`
- Helps style control without external models

---

## 3) Immediate Commands (Recommended Run)

```bash
# Training (course-safe, local)
USE_GPU=1 \
LANG_FILTER=en \
NUM_EPOCHS=25 \
SEQ_LEN=24 \
EMBEDDING_DIM=48 \
HIDDEN_DIM=192 \
MIN_FREQ=4 \
MAX_VOCAB_SIZE=15000 \
LEARNING_RATE=0.0008 \
LR_DECAY=0.98 \
MIN_LEARNING_RATE=0.00005 \
DROPOUT_RATE=0.15 \
LABEL_SMOOTHING=0.05 \
python3 TP_Paroles_Code_Complet.py
```

```bash
# Inference quality preset (course-safe)
python3 infer_lyrics.py \
  --genre rock --samples 3 --length 140 \
  --temperature 1.05 --top-k 70 --top-p 0.92 \
  --min-length 40 \
  --repeat-penalty 1.4 --presence-penalty 1.25 --frequency-penalty 0.15 \
  --no-repeat-window 24 --no-repeat-ngram 3
```

---

## 4) Definition of “High Quality, Not Static” for this project
A run is considered high quality when:
1. No absolute machine paths in code/scripts
2. Config is centralized and reproducible
3. No train/val leakage by song
4. Artifacts include run metadata
5. Decoding is robust (no NaN, no repeated loops)
6. Docs match actual defaults and commands

---

## 5) Proposed Execution Order
1. Fix hardcoded paths (`LANCER_TP.sh`, `generate_notebook.py`)
2. Implement song-level split before pair generation
3. Add config file + run metadata
4. Update README with quality preset and true defaults
5. Add generation quality metrics
6. (Optional) migrate model to GRU/LSTM for major lyric quality jump

---

## Final Note
If strict course compliance is required, this path is the right one: **all local, no external APIs, better engineering and better output quality**.
