# Plan de migration vers une approche LLM (qualité de paroles)

## 1) Objectif
Remplacer le générateur actuel (réseau feed-forward mot-à-mot) par une approche **LLM** pour obtenir des paroles nettement plus cohérentes, moins répétitives, et stylistiquement meilleures.

## 2) Pourquoi le modèle actuel plafonne
- Contexte court (`SEQ_LEN`) et architecture simple → perte de structure longue.
- Génération token locale → répétitions fréquentes (`i / you / the`).
- Même avec bon `val_loss`, la qualité textuelle peut rester faible (métrique ≠ qualité créative).

## 3) Choix possibles (du plus rapide au plus avancé)

### Option A — API LLM (recommandé pour résultat immédiat)
- Utiliser un modèle LLM via API (OpenAI/Azure/OpenRouter, etc.).
- Prompting par genre + contraintes (thème, schéma couplet/refrain, longueur).
- Avantages: meilleure qualité immédiatement, peu d’implémentation.
- Inconvénients: coût API, dépendance externe.

### Option B — LLM local instruction (sans API)
- Exécuter un LLM local (ex: via Ollama/vLLM/Transformers).
- Avantages: pas de coût par requête, contrôle local.
- Inconvénients: besoin GPU VRAM, qualité selon modèle.

### Option C — Fine-tuning (LoRA) sur dataset lyrics (meilleur long terme)
- Fine-tuner un LLM de base avec vos données (`spotify_songs.csv`, filtrage `language=en`).
- Avantages: style adapté au domaine.
- Inconvénients: plus de temps, pipeline MLOps, coûts GPU.

## 4) Recommandation pragmatique
**Phase 1 (rapide): Option A** pour prouver la qualité.
**Phase 2 (coût/perf): Option B** si vous voulez local.
**Phase 3 (personnalisation): Option C (LoRA)** quand le format final est validé.

---

## 5) Plan d’implémentation détaillé

### Étape 1 — Nouveau module de génération LLM
Créer `infer_lyrics_llm.py` avec:
- Entrées: `--genre`, `--theme`, `--mood`, `--language`, `--bars`, `--style`.
- Prompt template structuré:
  - Rôle: lyricist expert
  - Contrainte: éviter répétitions abusives
  - Structure: intro / couplet / refrain / pont
  - Vocabulaire adapté au genre
- Sortie: texte final + variante (n-best).

### Étape 2 — Intégration provider (API ou local)
- Ajouter abstraction `provider`:
  - `provider=openai|azure|ollama`
- Variables d’environnement:
  - `LLM_PROVIDER`
  - `LLM_MODEL`
  - `LLM_API_KEY` (si API)
  - `LLM_BASE_URL` (si self-host)

### Étape 3 — Contrôle qualité automatique
Ajouter un post-processing:
- Déduplication de lignes répétées.
- Règles simples anti-boucles (n-gram repeat limit).
- Score heuristique interne:
  - diversité lexicale,
  - taux de répétition,
  - longueur utile.

### Étape 4 — Fallback intelligent
- Si LLM indisponible: fallback vers `infer_lyrics.py` actuel.
- Log explicite: provider down / quota / timeout.

### Étape 5 — Évaluation comparative (A/B)
Comparer ancien vs LLM sur 20 prompts:
- Cohérence thématique
- Répétition
- Créativité
- Lisibilité

Sortie attendue: tableau de score + exemples.

---

## 6) Plan de fichiers
- **Nouveau**: `infer_lyrics_llm.py`
- **Nouveau**: `llm_prompt_templates.py`
- **Nouveau**: `llm_providers.py`
- **Nouveau**: `scripts/eval_lyrics_ab.py`
- **Maj**: `README.md` (section LLM + config env)
- **Maj**: `requirements.txt` (si SDK provider)

## 7) Critères d’acceptation
- Générer 3 sorties lisibles d’un genre donné sans boucles majeures.
- Réduction d’au moins 50% des répétitions vs pipeline actuel.
- Temps de génération < 10s/sortie (API) ou < 20s (local) selon infra.
- Pipeline actuel conservé comme fallback.

## 8) Risques & mitigation
- **Coût API** → cache des prompts, batch, limites.
- **Hallucination/qualité variable** → prompt strict + post-check.
- **Latence** → streaming + timeout + retry.
- **Dépendance provider** → abstraction multi-provider.

## 9) Roadmap suggérée
- J1: Module LLM minimal + 1 provider
- J2: Prompt tuning + post-processing
- J3: A/B evaluation + README
- J4+: (optionnel) LoRA fine-tuning

---

## 10) Commandes de départ (proposées)

```bash
# Exemple API
LLM_PROVIDER=openai \
LLM_MODEL=gpt-4o-mini \
LLM_API_KEY=... \
python3 infer_lyrics_llm.py --genre rock --theme heartbreak --bars 24
```

```bash
# Exemple local (Ollama)
LLM_PROVIDER=ollama \
LLM_MODEL=llama3.1:8b \
LLM_BASE_URL=http://localhost:11434 \
python3 infer_lyrics_llm.py --genre rap --theme ambition --bars 32
```

## 11) Décision recommandée
Commencer tout de suite par **Option A (API)** pour obtenir des résultats convaincants rapidement, puis basculer vers **Option B/C** selon contraintes coût/confidentialité.
