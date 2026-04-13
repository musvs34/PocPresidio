# Moteur Article 9 RGPD

## Objectif
Ce moteur est un dispositif de pre-controle et de scoring. Il ne tranche pas seul la conformite juridique. Il detecte des indices potentiels de categories particulieres de donnees au sens de l'article 9 du RGPD dans des documents de souscription assurance-vie, en particulier dans les zones de saisie libre.

## Architecture multicouche
1. Normalisation
   - minuscules
   - neutralisation des accents
   - uniformisation des apostrophes, tirets, ponctuation et espaces
   - preparation d'une forme comparable pour les lexiques, le fuzzy et les ancres semantiques
2. Analyse linguistique francaise
   - segmentation en phrases
   - lemmatisation via spaCy si disponible
   - repli simple par expressions regulieres sinon
   - calcul de racines heuristiques pour rapprocher des variantes comme `croyance` et `croyante`
3. Detection lexicale
   - recherche exacte dans des lexiques metier par categorie sensible
   - prise en compte des synonymes et des mots de contexte
4. Detection approximative
   - fuzzy matching via RapidFuzz
   - seuil configurable au niveau global
   - utile pour fautes d'orthographe, accents manquants ou variantes de saisie
5. Detection linguistique
   - comparaison sur lemmes et racines
   - prise en compte du contexte local et des labels de champs comme `conseil` ou `beneficiaire`
6. Detection semantique
   - similarite entre la phrase et des ancres semantiques de categorie
   - basee sur Sentence Transformers si disponible
   - en scan, chargement local uniquement
   - desactivee proprement si le modele n'est pas installe localement
7. Fusion des signaux
   - score final par categorie
   - ponderation par methode
   - seuils de `clear`, `review`, `alert`
8. Explicabilite et audit
   - evidences par categorie
   - methode contributrice
   - terme ou ancre declenchante
   - extrait de phrase
   - score et justification textuelle

## Arborescence
```text
configs/
  article9_categories.yml
data/
  raw/
  processed/
docs/
  article9_engine_architecture.md
examples/
  article9_sample_sentences.json
src/
  run_article9_examples.py
  run_article9_prepare_resources.py
  run_article9_scan.py
  article9_engine/
    __init__.py
    config.py
    detectors.py
    documents.py
    engine.py
    linguistics.py
    models.py
    normalization.py
    reporting.py
    scoring.py
tests/
  test_article9_engine.py
```

## Choix de bibliotheques
- `spaCy`
  Pour la segmentation et la lemmatisation francaise. Le moteur fonctionne sans, mais perd en qualite linguistique.
- `RapidFuzz`
  Pour le fuzzy matching des fautes et variantes.
- `Sentence Transformers`
  Pour une couche semantique explicable et parametree par ancres. Le telechargement des modeles doit etre realise dans une etape de preparation distincte du scan.
- `Presidio`
  Reste pertinent comme brique complementaire ou pour l'orchestration ulterieure, mais la couche metier article 9 reste ici pilotee par une configuration specifique assurance-vie.

## Seuils et calibrage
- `alert_threshold`
  Niveau a partir duquel une categorie est remontee comme alerte forte.
- `review_threshold`
  Niveau a partir duquel une revue humaine est recommandee.
- `method_weights`
  Importance relative des signaux. Recommandation initiale :
  - lexical : fort pour les correspondances exactes
  - fuzzy : utile mais moins determinant
  - linguistic : intermediaire
  - semantic : utile mais a surveiller de pres au debut

### Strategie de calibrage
1. Constituer un jeu de documents annotes.
2. Mesurer precision et rappel par categorie.
3. Augmenter d'abord la precision sur `alert`.
4. Laisser plus de rappel sur `review`.
5. Ajuster les poids avant d'ajuster les seuils.

### Valeurs initiales recommandees
- `review` autour de `0.45`
- `alert` autour de `0.75`
- seuil semantique autour de `0.60` a `0.65`
- fuzzy autour de `88`

## Journalisation et audit
Le moteur renvoie :
- un fichier detail des evidences
- un recapitulatif par document
- un JSON detaille contenant les justifications

Le `decision_log` documente explicitement l'etat de la couche semantique :
- `semantic:network_disabled`
- `semantic:local_loaded`
- `semantic:disabled_missing_local_model`

## Separation bootstrap / scan
- `run_article9_prepare_resources.py`
  Etape autorisee a telecharger les ressources externes.
- `run_article9_scan.py`
  Etape offline-by-default. Elle n'est pas autorisee a telecharger de modele.

Chaque evidence inclut :
- categorie
- score
- methode
- element declencheur
- contexte
- extrait
- explication
