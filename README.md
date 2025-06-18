# 🧠 2048 AI Solver

> **Machine‑Learning 2048** – cours & playground *(projet terminé)*

---

## ⚡ Installation ultra‑rapide

```bash
# dépendances minimales (CPU uniquement)
pip install numpy pandas pyarrow scikit_learn joblib pygame   # pygame facultatif

# (option) accélération JIT / roll‑out
pip install numba
```

| Fichier / script                      | Rôle principal                                 | Exemple de lancement                                         |
| ------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------ |
| **main\_game.py**                     | Console 2048 + IA                              | `python main_game.py`                                        |
| **interface\_jeu\_pygame.py**         | UI graphique, IA temps réel, dataset, bench    | `python interface_jeu_pygame.py --auto`                      |
| **interface\_jeu\_pygame.py --bench** | Benchmark multi‑proc (sans UI)                 | `python interface_jeu_pygame.py --preset turbo --bench 1000` |
| **train\_hgb\_rg.py**                 | Pré‑traitement + entraînement MoveNet (HistGB) | `python train_hgb_rg.py`                                     |
| **model.joblib**                      | Classifieur HistGradientBoosting pré‑entraîné  | Chargé automatiquement par l’UI                              |

---

## 🎯 Deux IA complémentaires

| Moteur               | Idée                                                                                                                           |  Forces                                                   | Faiblesses                                                  |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------- | ----------------------------------------------------------- |
| **Expectimax BEPP**  | Explore un arbre de coups jusqu’à *depth d*, coupe les branches peu probables (θ) et garde un faisceau des *k* meilleurs coups | Score très élevé, déterministe                            | Gourmand en calcul (≈ 10⁵ états analysés / coup)            |
| **MoveNet (HistGB)** | Classifieur supervisé qui prédit directement *le prochain coup* à partir de `empty_cnt` + 16 cases                             | 100× plus rapide, parallélisable, \~ 90 % du score BEPP‑4 | Peut se tromper si la grille sort du domaine d’entraînement |

L’interface lance **MoveNet** par défaut. Ajoutez `--auto bepp` pour basculer sur BEPP.

---

## 🤖 Recherche : Expectimax BEPP

```python
from search.expectimax import best_move
move = best_move(board, depth=3, time_limit_ms=60)
```

### Paramètres principaux (exposés en CLI)

| Flag      | Signification           | Défaut |
| --------- | ----------------------- | ------ |
| `--depth` | profondeur MAX          | 3      |
| `--time`  | budget (ms) / coup      | 60 ms  |
| `--beam`  | largeur du faisceau *k* | 2      |
| `--prob`  | coupure proba θ         | 0.04   |

#### Presets prêts à l’emploi

| preset    | depth | time (ms) | beam | prob | moteur                   |
| --------- | ----- | --------- | ---- | ---- | ------------------------ |
| `default` | 3     | 60        | 2    | 0.04 | BEPP                     |
| `turbo`   | 2     | 40        | 1    | 0.10 | BEPP                     |
| `rollout` | 3     | –         | –    | –    | `fast_best_move` (Numba) |

---

## 📊 MoveNet : du dataset au modèle

### 1. Génération du dataset

```bash
python interface_jeu_pygame.py \
       --preset turbo \
       --bg inf --workers 8 \
       --save data.csv --headless
```

Une ligne de log `[BG] 100,200 parties …` s’affiche toutes 100 parties.

**Contenu du CSV**

| Colonne      | Description                                      |
| ------------ | ------------------------------------------------ |
| `empty_cnt`  | nombre de cases vides                            |
| `c0…c15`     | valeurs (0 → vide, 2, 4, 8, …) rangées ligne→col |
| `bepp2_move` | coup choisi par BEPP‑2 (label de classe)         |
| `bepp2_val`  | évaluation bornée \[0‑1] de la grille            |

### 2. Nettoyage & Parquet

`train_hgb_rg.py` détecte et supprime les lignes corrompues (UUID, NaN, décimales irrégulières), cast les entiers en **uint16**, puis écrit `train_clean.parquet` (≈ 4× plus compact que le CSV).

### 3. Entraînement incrémental HistGradientBoosting

* **Streaming row‑group** : à peine \~800 Mo RAM pour 31 M de lignes.
* `warm_start=True` permet de continuer l’apprentissage entre les folds.
* CV 5 folds puis fit final **80 / 10** split (train/test).

| Étape               | Durée (s) |
| ------------------- | --------: |
| Lecture CSV         |       112 |
| Pré‑traitement      |       156 |
| Sauvegarde Parquet  |        39 |
| Split 90/10         |        17 |
| Cross‑validation 5× |      1038 |
| Entraînement final  |       200 |
| Sauvegarde modèle   |      0.03 |
| Évaluation          |       9.3 |

### 4. Scores obtenus

| Jeu (échantillon) |          Exactitude | F1‑macro |  Taille modèle |
| ----------------- | ------------------: | -------: | -------------: |
| CV (5 folds)      | **0.8916 ± 0.0002** |    0.923 |         9.3 Mo |
| Test 10 %         |          **0.8916** |    0.889 |         9.3 Mo |

#### Matrice de confusion sur le test (bruts)

| true \ pred |        up |    down |  left |   right |
| ----------: | --------: | ------: | ----: | ------: |
|      **up** | 1 736 664 |  83 775 |     0 |       2 |
|    **down** |   304 320 |   8 779 |     0 |       0 |
|    **left** |    14 474 | 745 951 |     0 | 232 095 |
|   **right** |         0 |       0 | 2 137 |       0 |

*(lignes → réponses attendues, colonnes → prédictions)*

> **Lecture rapide** : MoveNet se trompe surtout entre *left* et *down*. Les coups *right* sont quasi sûrs (jeu “accumulate‑à‑droite”).

---

## 🕹️ Arguments majeurs (UI & batch)

| Catégorie     | Flags                                          | Description                   |
| ------------- | ---------------------------------------------- | ----------------------------- |
| **Moteur IA** | `--auto` (MoveNet), `--auto bepp` (Expectimax) | Lance la partie en IA directe |
| **Recherche** | `--depth` `--time` `--beam` `--prob`           | BEPP uniquement               |
| **Affichage** | `--fps` `--speed`                              | Rendu Pygame                  |
| **Dataset**   | `--save` `--bg` `--workers`                    | Parties hors‑écran multiproc  |
| **Benchmark** | `--bench N`                                    | Simule N parties CPU‑only     |
| **Headless**  | `--headless`                                   | Sans fenêtre (serveur / SSH)  |

---

## ✅ Tests unitaires

```bash
python -m unittest discover -s tests
```

Valide : bit‑board, heuristiques, Expectimax, MoveNet wrapper.

---

## 📚 Références rapides

* **Expectimax for 2048** – Young et al., 2014
* **Learning n‑tuple networks for 2048** – Szubert & Jaśkowski, 2014
* **MuZero** – Schrittwieser et al., *Nature* 2020 (adapté à 2048)

*(Dernière mise à jour : 2025‑06)*
