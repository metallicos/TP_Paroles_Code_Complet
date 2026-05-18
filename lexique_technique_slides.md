# Lexique technique + logique des slides

Ce document explique les mots techniques utilisés dans la présentation, avec une formulation simple et orientée soutenance.

---

## 1) Logique globale de la présentation

La présentation suit une chaîne logique complète:

1. **Problème**: générer le prochain mot d’une chanson en gardant cohérence et style.
2. **Données**: nettoyer et structurer le corpus pour créer un signal d’apprentissage fiable.
3. **Modèle custom**: comprendre les mécanismes d’un réseau neuronal explicable.
4. **Apprentissage**: optimiser le modèle (loss, gradients, mise à jour des poids).
5. **Inférence**: contrôler la génération (diversité vs cohérence).
6. **Limites**: identifier ce que le modèle custom ne capture pas bien.
7. **GPT-2**: ajouter une architecture plus puissante pour les dépendances longues.
8. **Comparaison**: expliquer le compromis explicabilité/coût/qualité.

---

## 2) Lexique des mots techniques (version soutenance)

## A. Données et prétraitement

- **Corpus**: ensemble des textes utilisés pour entraîner le modèle.
- **Nettoyage (cleaning)**: suppression du bruit (lignes vides, caractères inutiles, etc.).
- **Tokenisation**: découpage du texte en unités (mots/tokens).
- **Token**: unité manipulée par le modèle (souvent un mot ici).
- **Vocabulaire (vocab)**: liste des tokens connus du modèle.
- **MIN_FREQ**: fréquence minimale pour garder un mot dans le vocabulaire.
- **MAX_VOCAB_SIZE**: taille maximale du vocabulaire.
- **PAD**: token de remplissage pour aligner les séquences à la même longueur.
- **UNK**: token inconnu utilisé quand un mot n’est pas dans le vocabulaire.
- **BOS/EOS**: début/fin de séquence.
- **Fenêtre glissante**: méthode pour créer beaucoup de paires d’apprentissage à partir d’un texte.
- **Paire (X, y)**: `X` = contexte, `y` = prochain token à prédire.
- **Train/Validation split**: séparation des données pour entraîner puis évaluer.
- **Split au niveau chanson**: séparation par chanson complète, pas par lignes, pour éviter la fuite de données.

## B. Modèle et architecture

- **Réseau neuronal (NN)**: modèle composé de couches de calcul avec paramètres appris.
- **Feed-forward**: architecture sans mémoire récurrente explicite; l’information avance couche par couche.
- **Conditionnement par genre**: ajout du genre musical comme signal d’entrée.
- **Embedding**: vecteur dense qui représente un token ou un genre.
- **Word embedding**: représentation vectorielle d’un mot.
- **Genre embedding**: représentation vectorielle du genre musical.
- **Concaténation**: fusion de plusieurs vecteurs en un seul.
- **Dense layer (couche dense)**: couche linéaire `z = Wx + b`.
- **ReLU**: fonction d’activation qui garde les valeurs positives.
- **Logits**: scores bruts avant softmax.
- **Softmax**: conversion des logits en probabilités sur le vocabulaire.

## C. Objectif d’apprentissage et optimisation

- **Cross-Entropy (CE)**: fonction de perte pour mesurer l’erreur de prédiction.
- **Loss**: erreur moyenne que l’entraînement cherche à minimiser.
- **Label smoothing**: adoucit les cibles pour éviter la sur-confiance.
- **Perplexité (PPL)**: indicateur d’incertitude du modèle (`PPL = exp(loss)`).
- **Gradient**: direction de variation de la loss par rapport aux paramètres.
- **Backpropagation**: calcul des gradients de la sortie vers les couches précédentes.
- **Descente de gradient**: mise à jour des paramètres avec la règle
  `θ ← θ - η∇θL`.
- **Learning rate (η)**: taille du pas de mise à jour.
- **Gradient clipping**: limite la norme du gradient pour stabiliser l’entraînement.
- **Mini-batch**: sous-ensemble de données traité à chaque itération.
- **Epoch**: passage complet sur le dataset d’entraînement.
- **Checkpoint**: sauvegarde de l’état du modèle pendant l’entraînement.
- **Early stopping**: arrêt anticipé si la validation n’améliore plus.

## D. Inférence (génération)

- **Inférence**: phase de génération après entraînement.
- **Argmax / Greedy decoding**: choisit toujours le token le plus probable.
- **Temperature**: contrôle l’aléa (plus haut = plus créatif, plus bas = plus conservateur).
- **Top-k**: garde seulement les `k` tokens les plus probables.
- **Top-p (nucleus sampling)**: garde le plus petit ensemble de tokens couvrant une masse de proba `p`.
- **Pénalité de répétition**: baisse le score des tokens déjà utilisés.
- **No-repeat n-gram**: interdit certains motifs répétés.
- **Compromis diversité/cohérence**: équilibre entre texte varié et texte stable.

## E. GPT-2 et Transformer

- **Transformer**: architecture basée sur l’attention, efficace pour les dépendances longues.
- **GPT-2**: modèle Transformer auto-régressif pré-entraîné sur de grands corpus.
- **Self-attention causale**: chaque token regarde les tokens passés, pas les futurs.
- **Fine-tuning**: adaptation d’un modèle pré-entraîné à un domaine spécifique (ici: lyrics).
- **Adaptation de domaine**: spécialisation d’un modèle général vers un corpus ciblé.
- **Coût calculatoire**: ressources GPU/temps nécessaires; souvent plus élevé pour GPT-2.

## F. Évaluation et interprétation

- **Courbes train/val**: évolution des métriques sur entraînement et validation.
- **Underfitting**: modèle trop simple, performances faibles partout.
- **Overfitting**: modèle trop adapté au train, généralise mal.
- **Gap train/val**: différence de performance entre entraînement et validation.
- **Convergence**: stabilisation de l’optimisation vers une meilleure solution.

---

## 3) Logique technique slide par slide (résumé court)

- **Slides 1–2**: définir le problème, les objectifs, la méthode.
- **Slide 3**: montrer l’architecture globale du pipeline.
- **Slides 4–6**: transformer des textes bruts en données supervisées exploitables.
- **Slide 7**: expliquer le modèle custom (structure + tensors).
- **Slides 8–9**: expliquer comment le modèle apprend (loss + gradients + update).
- **Slides 10–11**: analyser la dynamique d’entraînement et la généralisation.
- **Slides 12–13**: expliquer la génération et les limites structurelles du custom.
- **Slides 14–15**: introduire GPT-2 et son intégration dans le workflow.
- **Slide 16**: comparer custom vs GPT-2 selon objectif, qualité et coût.
- **Slide 17**: conclure et ouvrir sur les améliorations futures.

---

## 4) Formules à retenir (version simple)

- **Softmax**: transforme les scores en probabilités.
- **Cross-entropy**: mesure l’erreur entre proba prédite et vraie cible.
- **Perplexité**: `PPL = exp(L)` où `L` est la loss.
- **Update des paramètres**: `θ ← θ - η∇θL`.

Ces 4 éléments suffisent pour répondre à la plupart des questions techniques de base à l’oral.
