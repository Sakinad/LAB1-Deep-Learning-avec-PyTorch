# TP1 : Deep Learning avec PyTorch

## 1. Présentation du TP

Ce TP a pour objectif de mettre en pratique les concepts fondamentaux du Deep Learning à travers la construction, l'entraînement et l'évaluation de réseaux de neurones de type **Multilayer Perceptron (MLP)**.

Deux problèmes sont étudiés :

- **Régression** : prédire le prix de clôture d'une action Apple (AAPL) à partir de données boursières.
- **Classification** : prédire le type de panne d'une machine à partir de ses caractéristiques de fonctionnement.

## 2. Environnement et outils

- Python
- PyTorch 2.11.0
- Pandas et NumPy
- Scikit-learn
- Matplotlib et Seaborn
- Imbalanced-learn
- Skorch
- Google Colab

Les expériences ont été exécutées sur CPU, car aucun GPU n'était disponible.

---

## 3. Partie 1 : Régression sur les données boursières

### 3.1. Objectif et données

Le dataset NYSE contient **851 264 observations et 7 colonnes**. L'exploration initiale n'a révélé aucune valeur manquante ni aucune ligne dupliquée.

Pour cette expérience, seule l'action Apple (AAPL) est utilisée, avec **1 762 observations**.

Les variables d'entrée sont :

- `open` : prix d'ouverture ;
- `high` : prix le plus élevé ;
- `low` : prix le plus bas ;
- `volume` : volume des transactions.

La variable cible est `close`, c'est-à-dire le prix de clôture.

Les données sont séparées chronologiquement en trois ensembles :

- Entraînement : 1 233 observations (70 %).
- Validation : 264 observations (15 %).
- Test : 265 observations (15 %).

Les variables d'entrée sont standardisées à partir de l'ensemble d'entraînement.

### 3.2. Architecture et entraînement

Un MLP est construit avec deux couches cachées contenant respectivement **64 et 32 neurones**, avec la fonction d'activation ReLU.

Le modèle est entraîné pendant 100 époques avec :

- Optimiseur : Adam.
- Taux d'apprentissage : 0,001.
- Fonction de perte : Mean Squared Error (MSE).
- Taille du lot : 64.

La loss d'entraînement diminue de 661,96 à l'époque 10 à 0,67 à l'époque 100. La loss de validation diminue également, passant de 1 851,84 à 2,82. Cette évolution indique que le modèle apprend progressivement à ajuster ses prédictions aux données.

### 3.3. Évaluation du modèle de base

Les résultats sur l'ensemble de test sont les suivants :

| Métrique | Résultat |
| -------- | -------: |
| MAE      |   0,7255 |
| RMSE     |   0,9236 |
| R²       |   0,9848 |

Le MAE indique une erreur absolue moyenne d'environ **0,73 unité de prix**. Le RMSE, d'environ 0,92, donne davantage de poids aux erreurs importantes.

Le coefficient R² de 0,9848 indique que le modèle explique une grande partie de la variabilité des prix de clôture sur cet ensemble de test. Ces résultats sont encourageants, mais ils ne garantissent pas que le modèle puisse prévoir avec la même précision des prix futurs dans d'autres conditions de marché.

### 3.4. Optimisation des hyperparamètres

Une recherche GridSearchCV avec validation croisée chronologique (`TimeSeriesSplit`) a été réalisée pour comparer plusieurs configurations.

Les meilleurs hyperparamètres obtenus sont :

- Nombre de neurones dans la première couche cachée : 64.
- Taux d'apprentissage : 0,001.
- Nombre d'époques : 100.
- Taille du lot : 32.
- Optimiseur : Adam.

Le MSE moyen de validation associé à cette configuration est de **580,6861**.

Sur l'ensemble de test, les résultats du meilleur estimateur sont :

| Métrique | Modèle de base | Modèle optimisé |
| -------- | -------------: | --------------: |
| MAE      |         0,7255 |          0,6395 |
| RMSE     |         0,9236 |          0,8845 |
| R²       |         0,9848 |          0,9861 |

La recherche d'hyperparamètres améliore les trois métriques sur le test : les erreurs diminuent et le coefficient R² augmente légèrement.

### 3.5. Comparaison des techniques de régularisation

Trois configurations ont été comparées : le modèle de base, le modèle avec Dropout (0,2) et le modèle avec Weight Decay (L2, (10^{-4})).

| Modèle       |    MAE |   RMSE |     R² |
| ------------ | -----: | -----: | -----: |
| Base         | 0,6110 | 0,8029 | 0,9885 |
| Dropout      | 1,8533 | 2,0024 | 0,9286 |
| Weight Decay | 0,6277 | 0,8260 | 0,9879 |

Dans cette expérience, le modèle de base obtient les meilleurs résultats sur le test. Le Weight Decay produit des résultats proches, tandis que le Dropout dégrade nettement les performances.

Cela montre que la régularisation n'améliore pas automatiquement un modèle : son efficacité dépend notamment de l'architecture, des données et des paramètres choisis.

---

## 4. Partie 2 : Classification des pannes de machines

### 4.1. Objectif et exploration des données

Cette partie utilise le dataset *Machine Predictive Maintenance Classification*, composé de **10 000 observations**.

L'objectif est de prédire le type de panne à partir des caractéristiques de fonctionnement de la machine, telles que la température de l'air, la température du processus, la vitesse de rotation, le couple et l'usure de l'outil.

Six classes sont considérées :

1. Heat Dissipation Failure
2. No Failure
3. Overstrain Failure
4. Power Failure
5. Random Failures
6. Tool Wear Failure

La distribution des classes est fortement déséquilibrée : la classe `No Failure` contient 9 652 observations, alors que `Random Failures` n'en contient que 18.

Les données sont réparties en trois ensembles :

- Entraînement : 7 000 observations.
- Validation : 1 500 observations.
- Test : 1 500 observations.

Les variables d'entrée sont standardisées et la variable catégorielle `Type` est encodée. Un suréchantillonnage aléatoire (`RandomOverSampler`) est appliqué uniquement aux données d'entraînement. Après cette étape, chaque classe contient 6 756 observations dans l'ensemble d'entraînement équilibré.

### 4.2. Architecture et entraînement

Le modèle de classification est un MLP comportant :

- Une première couche cachée de 64 neurones.
- Une deuxième couche cachée de 32 neurones.
- Une couche de sortie de 6 neurones, correspondant aux six classes.
- Des activations ReLU et une fonction de perte Cross Entropy.

L'entraînement utilise l'optimiseur Adam et un mécanisme d'**Early Stopping** pour arrêter l'apprentissage lorsque la loss de validation ne s'améliore plus suffisamment.

Une recherche GridSearchCV avec validation croisée stratifiée (`StratifiedKFold`) sélectionne les hyperparamètres. La meilleure configuration trouvée utilise 64 neurones dans la première couche cachée, un taux d'apprentissage de 0,001, 20 époques et une taille de lot de 64.

Le meilleur F1-score macro obtenu pendant la validation croisée est de **0,7299**.

### 4.3. Évaluation sur l'ensemble de test

Les résultats du modèle sélectionné sur l'ensemble de test sont :

| Métrique        |         Résultat |
| --------------- | ---------------: |
| Accuracy        | 0,9367 (93,67 %) |
| Precision macro |           0,4967 |
| Recall macro    |           0,6590 |
| F1-score macro  |           0,5519 |

L'accuracy de 93,67 % paraît élevée, mais elle doit être interprétée avec prudence en raison du déséquilibre des classes.

Le F1-score macro de 0,5519 montre que les performances restent inégales entre les six classes. Le rappel macro de 0,6590 indique également que la capacité à détecter les différentes classes varie selon le type de panne.

Le rapport de classification donne plus de détails :

| Type de panne            | Precision | Recall | F1-score |
| ------------------------ | --------: | -----: | -------: |
| Heat Dissipation Failure |      0,44 |   0,94 |     0,60 |
| No Failure               |      0,99 |   0,94 |     0,97 |
| Overstrain Failure       |      0,75 |   1,00 |     0,86 |
| Power Failure            |      0,76 |   0,93 |     0,84 |
| Random Failures          |      0,00 |   0,00 |     0,00 |
| Tool Wear Failure        |      0,03 |   0,14 |     0,04 |

Le modèle reconnaît bien la classe majoritaire `No Failure` et obtient de bons résultats sur plusieurs types de panne. En revanche, il ne détecte aucune observation de la classe `Random Failures` dans le test et obtient un très faible F1-score pour `Tool Wear Failure`.

Ces résultats soulignent l'importance d'examiner le rappel, le F1-score et la matrice de confusion, plutôt que de se limiter à l'accuracy. Les classes rares restent difficiles à apprendre et à prédire.

### 4.4. Comparaison des techniques de régularisation

Quatre configurations ont été comparées sur l'ensemble de test :

| Modèle                 | Accuracy | Recall macro | F1-score macro |
| ---------------------- | -------: | -----------: | -------------: |
| Base                   |   0,9293 |       0,7051 |         0,5007 |
| Dropout                |   0,8713 |       0,7069 |         0,4292 |
| Weight Decay           |   0,9400 |       0,6596 |         0,5329 |
| Dropout + Weight Decay |   0,8787 |       0,6608 |         0,4543 |

Le modèle **Weight Decay** obtient la meilleure accuracy (94,00 %) et le meilleur F1-score macro (0,5329) parmi ces quatre configurations.

Le modèle de base présente cependant le meilleur rappel macro (0,7051). Le Dropout réduit l'accuracy et le F1-score macro dans cette expérience. La combinaison Dropout et Weight Decay ne donne pas de meilleurs résultats que le Weight Decay seul.

Le choix du modèle dépend donc de la métrique privilégiée. Pour un problème de maintenance prédictive, il est particulièrement important d'étudier les performances par type de panne, car manquer une panne rare peut avoir des conséquences importantes.

---

## 5. Synthèse — Ce que j'ai appris

Ce TP m'a permis de comprendre les principales étapes d'un projet de Deep Learning, de l'exploration des données jusqu'à l'évaluation et à la comparaison des modèles.

J'ai appris à construire et entraîner des réseaux de neurones MLP avec PyTorch, à utiliser les fonctions d'activation, les fonctions de perte et les optimiseurs, ainsi qu'à séparer les données en ensembles d'entraînement, de validation et de test.

La partie régression m'a permis d'utiliser les métriques MAE, RMSE et R², d'étudier l'évolution de la loss et de comparer différentes configurations d'hyperparamètres.

La partie classification m'a permis de travailler avec plusieurs classes, de traiter un fort déséquilibre des données par suréchantillonnage, d'utiliser GridSearchCV et l'Early Stopping, et d'évaluer les résultats avec la précision, le rappel, le F1-score et la matrice de confusion.

Enfin, les expériences de régularisation m'ont montré que le Dropout et le Weight Decay n'améliorent pas systématiquement les performances. Le choix d'un modèle doit tenir compte de la nature des données, de la généralisation et de la métrique adaptée au problème étudié.

