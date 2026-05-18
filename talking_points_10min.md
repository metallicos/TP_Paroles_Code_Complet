

## Slide 1 — Introduction
Nous présentons un projet de génération automatique de parole .L’idée centrale est de prédire le prochain mot en fonction du contexte et du genre musical. L’objectif n’est pas uniquement d’obtenir des mots plausibles localement, mais de produire un texte qui reste cohérent sur plusieurs lignes.

 Côté algorithmes, nous utilisons deux approches complémentaires: un modèle neuronal feed-forward conditionné par le genre , puis un  Distilled GPT-2 pour améliorer la cohérence globale. Côté libraries, le socle repose sur Python avec NumPy/CuPy pour le calcul, pandas pour les données, matplotlib pour les graphes, et Hugging Face Transformers/Datasets pour Distilled GPT-2. Les features principales sont la génération conditionnée par genre, le décodage contrôlé (temperature, top-k, top-p, pénalités de répétition), les checkpoints d’entraînement, et l’analyse comparative entre modèle custom et Distilled GPT-2.

## Slide 2 — Cadre de soutenance
La logique de la soutenance est de relier théorie et implémentation. Nous allons montrer comment les choix algorithmiques se traduisent dans le code: données, architecture, apprentissage, inférence, puis amélioration avec Distilled GPT-2. Le but est de défendre une démarche méthodique et reproductible.

## Slide 3 — Architecture système
Le pipeline est découpé en blocs: ingestion, prétraitement, entraînement, checkpoints, inférence et évaluation. Cette séparation aide à diagnostiquer les problèmes, car on sait exactement où intervenir si la qualité se dégrade. La branche Distilled GPT-2 est une extension du pipeline initial pour améliorer la cohérence longue portée.

## Slide 4 — Ingénierie de données
La qualité du modèle dépend directement de la qualité statistique du corpus. On nettoie les textes, on filtre la langue et on garde les genres dominants pour réduire le bruit et la variance inter-domaines. Le point méthodologique important est le split au niveau chanson, qui évite les fuites entre entraînement et validation.

## Slide 5 — Prétraitement
Le prétraitement est volontairement déterministe: normalisation, nettoyage des caractères, tokenisation simple. Ce choix rend l’expérience traçable et facile à expliquer, même s’il est moins riche qu’une tokenisation subword. Les tokens spéciaux comme BOS/EOS/UNK structurent la séquence et sécurisent l’encodage.

## Slide 6 — Vocabulaire et séquences
Le vocabulaire est contrôlé par fréquence minimale et taille maximale pour éviter une sortie trop sparse. Ensuite, chaque chanson est transformée en paires supervisées `(contexte, prochain token)` avec fenêtre glissante. Techniquement, c’est ce qui fournit un très grand volume d’exemples pour le next-token learning.

## Slide 7 — Architecture du modèle custom
Le modèle custom est un réseau neuronal feed-forward conditionné par le genre. On concatène embeddings de mots et embedding de genre, puis on passe par des couches denses jusqu’aux logits sur le vocabulaire. Son intérêt principal est l’explicabilité: on suit facilement le chemin des tenseurs et des gradients.

## Slide 8 — Fonction objectif
Ici, nous disons simplement: on mesure l’erreur entre le mot prédit et le vrai mot suivant. La cross-entropy est la loss principale, et le label smoothing évite une confiance excessive trop tôt. La perplexité sert d’indicateur global: quand elle baisse, l’incertitude du modèle diminue.

## Slide 9 — Backpropagation
La backpropagation corrige les paramètres à partir de l’erreur de sortie. En pratique: on calcule l’erreur, on la propage vers l’arrière, puis on met à jour les poids avec la règle $\theta \leftarrow \theta - \eta \nabla_\theta L$. Le clipping de gradient stabilise l’apprentissage en empêchant des mises à jour trop brutales.

## Slide 10 — Boucle d’entraînement
La boucle d’entraînement combine apprentissage par mini-batch, validation régulière, sauvegarde de checkpoints et early stopping. Le learning rate est ajusté progressivement pour éviter les oscillations. Cette partie garantit une convergence exploitable et évite de surentraîner inutilement.

## Slide 11 — Dynamique d’apprentissage
Cette slide interprète les courbes train/val du modèle custom. On cherche une baisse de loss, une perplexité en recul, et un écart train/val raisonnable. Si la loss baisse mais que la qualité textuelle reste limitée, cela indique souvent une limite de capacité du modèle plutôt qu’un simple problème d’optimiseur.

## Slide 12 — Inférence
En génération, on n’utilise pas seulement l’argmax car cela donne des textes répétitifs. On combine température, top-k, top-p et pénalités de répétition en espace logit. Le but est de contrôler le compromis entre diversité et cohérence.

## Slide 13 — Limites du modèle custom
Le modèle custom a une limite structurelle: il ne gère pas bien les dépendances longues comparé aux architectures à attention. On observe donc parfois dérive sémantique et répétitions sur des sorties longues. C’est précisément cette limite qui justifie l’intégration de Distilled GPT-2.

## Slide 14 — Fondements Distilled GPT-2
Distilled GPT-2 repose sur l’auto-attention causale, ce qui permet de mieux exploiter le contexte distant. Le fine-tuning adapte un modèle pré-entraîné au domaine des lyrics avec un learning rate faible. Résultat attendu: meilleure cohérence globale et meilleure fluidité linguistique.

## Slide 15 — Intégration GPT dans le workflow
Cette slide montre comment Distilled GPT-2 s’insère dans le pipeline existant: corpus nettoyé, fine-tuning, génération guidée, validation. L’idée n’est pas d’abandonner le modèle custom, mais de compléter ses limites. On garde donc une logique hybride orientée performance et interprétabilité.

## Slide 16 — Comparaison custom vs Distilled GPT-2
Ici on lit le graphe de comparaison: le modèle custom reste excellent pour l’analyse pédagogique et la traçabilité des mécanismes, alors que Distilled GPT-2 produit généralement une meilleure qualité textuelle. Le coût calculatoire est plus élevé côté Distilled GPT-2. La conclusion pratique est un compromis: custom pour comprendre, Distilled GPT-2 pour générer mieux.

## Slide 17 — Conclusion
Le projet couvre toute la chaîne: préparation des données, apprentissage supervisé, stratégie d’inférence et extension Transformer. La valeur du travail est double: explicabilité du modèle custom et gain qualitatif avec Distilled GPT-2. En perspective, le prochain niveau est d’améliorer l’évaluation qualitative humaine et la robustesse MLOps.

## Phrase de clôture
En résumé, on construit d’abord une base explicable, puis on l’augmente avec un modèle plus puissant pour gagner en qualité sans perdre la rigueur technique.

---

## Questions probables du professeur (avec réponses)

### Slide 1 — Introduction
**Q:** Pourquoi avoir choisi un sujet de génération de paroles plutôt qu’un cas plus classique de classification ?  
**R:** Parce que la génération oblige à maîtriser toute la chaîne: préparation séquentielle, apprentissage probabiliste, décodage, et évaluation qualitative. C’est plus exigeant et plus démonstratif techniquement.

### Slide 2 — Cadre de soutenance
**Q:** Quelle est votre contribution principale par rapport à un simple usage d’un modèle pré-entraîné ?  
**R:** Nous avons construit un pipeline complet avec un modèle custom explicable, puis ajouté Distilled GPT-2 comme extension. La contribution est la comparaison méthodique entre explicabilité et qualité de génération.

### Slide 3 — Architecture système
**Q:** Pourquoi découper le système en modules séparés ?  
**R:** Pour isoler les responsabilités et faciliter le diagnostic. Si un problème apparaît, on identifie rapidement s’il vient des données, du modèle, ou du décodage.

### Slide 4 — Données
**Q:** Pourquoi le split au niveau chanson est-il important ?  
**R:** Il évite la fuite d’information entre train et validation. Si on split par séquences, des fragments d’une même chanson peuvent se retrouver dans les deux ensembles, ce qui biaise l’évaluation.

### Slide 5 — Prétraitement
**Q:** Pourquoi ne pas utiliser directement une tokenisation subword moderne ?  
**R:** Nous avons privilégié une tokenisation simple pour la traçabilité et l’explicabilité pédagogique. C’est un choix volontaire pour comprendre finement le comportement du pipeline.

### Slide 6 — Vocabulaire et paires
**Q:** Quel est l’impact du `MIN_FREQ` et de la taille de vocabulaire ?  
**R:** Ils contrôlent le compromis couverture/stabilité. Un vocabulaire trop grand augmente la sparsité et la difficulté d’optimisation; un vocabulaire trop petit perd de l’information lexicale.

### Slide 7 — Modèle custom
**Q:** Pourquoi conditionner le modèle par le genre musical ?  
**R:** Le genre apporte un signal contextuel global qui influence le style lexical et rythmique. Sans ce conditionnement, la génération devient plus générique et moins cohérente stylistiquement.

### Slide 8 — Fonction objectif
**Q:** Pourquoi utiliser cross-entropy + label smoothing ?  
**R:** La cross-entropy est adaptée au next-token multiclasses. Le label smoothing réduit la sur-confiance, stabilise les gradients et améliore souvent la généralisation.

### Slide 9 — Backpropagation
**Q:** Que représente exactement la formule $\theta \leftarrow \theta - \eta \nabla_\theta L$ ?  
**R:** C’est la règle de descente de gradient: on met à jour chaque paramètre dans la direction opposée au gradient pour réduire la loss à l’itération suivante.

### Slide 10 — Boucle d’entraînement
**Q:** Pourquoi utiliser `early stopping` et `checkpoints` ensemble ?  
**R:** L’early stopping limite le surapprentissage, et les checkpoints permettent de sauvegarder les meilleurs états du modèle. Ensemble, ils rendent l’entraînement plus robuste et reproductible.

### Slide 11 — Dynamique d’apprentissage
**Q:** Comment savoir si le modèle underfit ou overfit ?  
**R:** On observe l’écart train/val et l’évolution des courbes. Un grand écart avec val qui se dégrade indique plutôt overfit; des performances faibles sur les deux indiquent underfit.

### Slide 12 — Inférence
**Q:** Pourquoi ne pas faire uniquement un décodage greedy (argmax) ?  
**R:** Le greedy réduit fortement la diversité et favorise la répétition. Les stratégies temperature/top-k/top-p gardent de la variété tout en conservant le contrôle sur la qualité.

### Slide 13 — Limites du modèle custom
**Q:** Quelle est la limite la plus critique de votre modèle custom ?  
**R:** La modélisation des dépendances longues. Avec un contexte borné et sans mécanisme d’attention globale, la cohérence peut se dégrader sur les générations longues.

### Slide 14 — Distilled GPT-2
**Q:** Pourquoi Distilled GPT-2 améliore-t-il la cohérence globale ?  
**R:** Grâce à l’auto-attention causale, Distilled GPT-2 capture mieux les relations à longue distance entre tokens, ce qui améliore la continuité sémantique et syntaxique.

### Slide 15 — Intégration workflow
**Q:** Pourquoi garder deux approches au lieu de ne conserver que Distilled GPT-2 ?  
**R:** Le modèle custom apporte une forte explicabilité pédagogique et un coût plus léger, tandis que Distilled GPT-2 apporte la qualité textuelle. Le duo est plus pertinent qu’un choix unique.

### Slide 16 — Comparaison
**Q:** Comment justifier le choix final entre custom et Distilled GPT-2 ?  
**R:** Selon l’objectif: pour expliquer et expérimenter finement, custom est idéal; pour la qualité finale de génération, Distilled GPT-2 est meilleur. En pratique, nous retenons une stratégie hybride.

### Slide 17 — Conclusion
**Q:** Quelle serait votre prochaine amélioration prioritaire ?  
**R:** Mettre en place une évaluation humaine structurée (cohérence, créativité, style) et renforcer le suivi expérimental MLOps pour comparer les versions de façon plus rigoureuse.
