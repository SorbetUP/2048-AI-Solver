# 🧠 2048 AI Solver

> **Machine‑Learning 2048** – cours & playground

---

## ⚡ Installation rapide

```bash
# dépendances minimales
pip install numpy  pandas  pyarrow  scikit_learn pygame (fac.)

# (option) accélération JIT
pip install numba
```

| fichier                               | rôle                                        | lancer                                                       |
| ------------------------------------- | ------------------------------------------- | ------------------------------------------------------------ |
| **main\_game.py**                     | console 2048 + IA                           | `python main_game.py`                                        |
| **interface\_jeu\_pygame.py**         | interface graphique, IA, génération dataset | `python interface_jeu_pygame.py --auto`                      |
| **interface\_jeu\_pygame.py --bench** | bench IA (multiproc)                        | `python interface_jeu_pygame.py --preset turbo --bench 1000` |
| **train\_hgb\_rg.py**                 | entraînement MoveNet (HistGB, 5‑fold)       | `python train_hgb_rg.py`                                     |

---

## 🎯 Objectif

Créer (et comparer) plusieurs intelligences artificielles pour [**2048**](https://play2048.co). Deux familles :

1. **Recherche Expectimax BEPP** (déterministe, "fort" mais lent)
2. **MoveNet** (classeur de coups appris sur dataset → ultra‑rapide)

> BEPP = *Bounded Expectation & Probability Pruning* : beam search + coupures proba.

---

## 🧱 Arborescence

```
2048-AI-Solver/
├─ board.py               # bit‑board 64 bits + kernels numba
├─ game.py                # logique de jeu
├─ interface_jeu_pygame.py# UI + dataset + bench (multiprocessing)
├─ train_hgb_rg.py        # entraînement MoveNet (row‑group streaming)
├─ search/
│   ├─ expectimax.py      # BEPP classique
│   └─ fast_expectimax.py # roll‑out numba (≈ 20 k states/s)
├─ eval/
│   └─ heuristics.py      # basic_eval, bounded_eval, Victor (à venir)
└─ tests/                 # unittest
```

---

## 🤖 IA Recherche – Expectimax BEPP

```python
from search.expectimax import best_move
mv = best_move(board, depth=4, time_limit_ms=120)  # BEPP 4‑ply
```

- `--beam k` pour la largeur du faisceau
- `--prob θ`  coupure de branches à faible proba
- Table de transposition + approfondissement itératif

Préréglages CLI :

| preset      | depth | budget ms | beam | prob | moteur                   |
| ----------- | ----- | --------- | ---- | ---- | ------------------------ |
| default     |  3    |  60       | 2    | 0.04 | BEPP                     |
| **turbo**   |  2    | 40        | 1    | 0.10 | BEPP                     |
| **rollout** | 3     | –         | –    | –    | `fast_best_move` (numba) |

---

## 📊 Génération de dataset

```bash
# boucle infinie en tâche de fond (8 workers) + enregistrement CSV
python interface_jeu_pygame.py \
       --preset turbo \
       --bg inf \
       --workers 8 \
       --save data.csv \
       --auto             # UI cachée si pas de focus
```

Chaque grille enregistrée contient :

- état 16 cases (`c0…c15`)
- score, max tile, cases vides
- `bepp2_move`  + `bepp2_val`  *(label pour MoveNet)*

Progrès affiché toutes 100 parties.

---

## 🕹️ Exécution complète – **tous les arguments**

Pour voir toutes les options de `interface_jeu_pygame.py` en action :

```bash
python interface_jeu_pygame.py \
    --preset turbo \            # réglages rapides (écrase depth/time/beam/prob)
    --fps 60 \                   # fréquence d'affichage
    --speed 1.0 \                # facteur de délai IA (UI)
    --depth 2 --time 40 \        # profondeur + budget ms par coup
    --beam 1 --prob 0.10 \       # paramètres BEPP
    --save data.csv \            # CSV de sortie pour le dataset
    --bg 10000 \                 # nombre de parties IA en arrière‑plan ("inf" pour infini)
    --workers 8 \                # processus parallèles pour bg / bench
    --bench 1000 \               # benchmark hors‑écran (1000 parties)
    --auto                       # démarre la fenêtre en mode IA
```

> 🎛️ Utilise **seulement** les switches utiles : `--bench` *ignore* `--fps/--auto`, `--bg` tourne en tâche de fond même si la fenêtre IA est fermée, etc.

---

## 🏋️ Entraînement MoveNet (HistGradientBoosting)

 Entraînement MoveNet (HistGradientBoosting)

```bash
python train_hgb_rg.py          # lit le Parquet en streaming
```

Le script :

1. Convertit **data.csv → train\_clean.parquet** (uint16, compact)
2. 5‑fold CV par row‑group (stream, warm\_start)
3. Split 80/20, modèle final enregistré → `hgb_2048.joblib`
4. Rapport métriques JSON

Inference :

```python
import joblib, numpy as np
model = joblib.load("hgb_2048.joblib")
move_id = model.predict(np.array([grid_vector]))[0]
move = ["up","down","left","right"][move_id]
```

> **Gain** : 100× plus rapide que Expectimax‑4, \~90 % du score moyen.

---

## ✅ Tests

```bash
python -m unittest discover -s tests
```

*Test OK ⇒ logique Expectimax et heuristique validées.*

---

## 🚀 Roadmap

- 🔬 Entraîner **Victor** (réseau régressant les coups restants)
- 🤝 Fusion MoveNet + Victor → "depth 1.5" (policy + value)
- ♻️ Self‑play RL (DQN ou MuZero‑light)
- 🌐 Web demo (PWA + WASM)

---

## 📚 Références rapides

- Papier original : **Young et al., Expectimax for 2048, 2014**
- "Learning n‑tuple networks" — Szubert & Jaśkowski (2014)
- MuZero — Schrittwieser et al., Nature 2020 (adapté 2048)

---

## ✨ Auteurs

Projet pédagogique – n’hésitez pas à *fork* et proposer vos PR !

