# 🧠 2048 AI Solver

machine learning 2048 solver - student project

```pip install nibabel numpy ```
```python main_game.py```
```python main_game.py --bench 100000 ```
```python interface_jeu_pygame.py ```
```python interface_jeu_pygame.py --fps 120 ```

---

## 🎯 Objectif

Ce projet vise à créer une intelligence artificielle capable de résoudre le jeu **2048**, un jeu de puzzle populaire basé sur la fusion de tuiles. Vous pouvez tester le jeu ici : [https://play2048.co](https://play2048.co)

Le défi principal du jeu réside dans son caractère **aléatoire** : à chaque mouvement, une nouvelle tuile (2 ou 4) apparaît dans une case vide. Cela rend impossible d’avoir une stratégie gagnante à 100 %.

Deux approches sont explorées dans le projet :

- **Apprentissage par renforcement (RL)** : laisser l’IA apprendre en jouant (non encore implémenté)
- **Recherche arborescente (Expectimax)** : simuler les possibilités à l’aide d’un arbre décisionnel

---

## 🧱 Structure du projet

```bash
2048-AI-Solver/
├── board.py                  # Représente l’état du plateau et les opérations sur la grille
├── game.py                   # Logique du jeu (exécute les tours, mouvements, etc.)
├── interface_jeu_pygame.py   # Interface graphique (optionnelle, basée sur Pygame)
├── main.py                   # Script de lancement principal
├── main_game.py              # Variante avec interface utilisateur
├── random_play.py            # Génération aléatoire de parties (utile pour créer des données d'entraînement)
├── simulate.py               # Permet de simuler des parties avec différentes IA
├── README.md                 # Fichier d’explication du projet
├── eval/
│   └── heuristics.py         # Fonctions d’évaluation de la grille (heuristiques ou IA Victor)
├── search/
│   └── expectimax.py         # Implémentation de l’algorithme Expectimax
├── tests/
│   └── test_expectimax.py    # Tests unitaires pour la logique de recherche
```

---

## 🤖 Algorithme Expectimax

L'algorithme **Expectimax** est une version modifiée de Minimax qui gère l’aléatoire :
- **Nœuds MAX** : les décisions du joueur (haut, bas, gauche, droite)
- **Nœuds CHANCE** : l’apparition aléatoire de 2 ou 4 dans une case vide

Il explore un **arbre de décisions** jusqu’à une certaine profondeur, en utilisant :
- **Élagage (pruning)** pour éviter des branches peu prometteuses
- Une **fonction d’évaluation** personnalisable (ex : IA Victor)
- Une **table de transposition** pour éviter de recalculer des états déjà vus

Exemple d’appel :
```python
move = best_move(board, depth=3, time_limit_ms=1000, eval_fn=DummyVictor())
```

---

## 🧠 Évaluation : heuristique ou IA Victor

Deux façons d’évaluer une grille :

- `basic_eval` : basée sur le nombre de cases vides + valeur max
- `Victor` (à développer) : IA entraînée à prédire la qualité d'une grille (ex. combien de coups restants)

Un exemple de fonction d’évaluation simple (DummyVictor) est utilisée dans les tests pour simuler ce comportement.

---

## ✅ Tests unitaires

Les tests se trouvent dans `tests/test_expectimax.py` et couvrent :

- Le bon choix de mouvement dans une situation déterministe
- L’intégration d’une fonction d’évaluation personnalisée

Lancer les tests :

```bash
python -m unittest discover -s tests
```

---

## 🚀 Prochaines étapes

- 🔬 Implémenter Victor (avec scikit-learn ou PyTorch)
- 📊 Générer des données avec `random_play.py` pour entraîner Victor
- ⚡ Ajouter un cache global pour les grilles fréquentes
- 🕹️ Finaliser l’interface Pygame et permettre de jouer contre l’IA

---

## 📚 Bibliothèques utilisées

- `pygame` *(facultatif)* : interface graphique
- `pandas` : gestion de données
- `sklearn` / `torch` *(optionnel)* : pour entraîner Victor

---

## ✨ Auteurs

Projet réalisé dans le cadre d’un projet IA par [votre équipe].
