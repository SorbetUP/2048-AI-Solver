# main.py – interface interactive + benchmark aléatoire haute-perf
import argparse, time
from game import Game
from random_play import random_benchmark        # <-- nouvelle import

# ─────────────────────────────────────────────
def _clear_screen():
    import os, sys
    if sys.stdout.isatty():
        os.system("cls" if os.name == "nt" else "clear")

def _print_board(g: Game):
    _clear_screen()
    print(g.board)
    print(f"Score : {g.score}")

# ─────────────────────────────────────────────
def interactive():
    g = Game()
    mapping = {"z": "up", "s": "down", "q": "left", "d": "right"}
    while not g.is_over():
        _print_board(g)
        mv = input("Direction (z/q/s/d) ou x pour quitter : ").lower()
        if mv == "x":
            break
        if mv not in mapping:
            continue
        g.move(mapping[mv])
    _print_board(g)
    print("Gagné !" if g.is_won() else "Perdu…")

# ─────────────────────────────────────────────
def bench(n_games: int):
    print(f"Lancement benchmark aléatoire : {n_games:_} parties…")
    t0     = time.perf_counter()
    scores = random_benchmark(n_games)          # <-- appel au moteur JIT
    dt     = time.perf_counter() - t0
    rate   = n_games / dt
    print(f"{rate:,.0f} parties/s – durée {dt:.3f}s")
    print(f"Score moyen      : {scores.mean():.1f}")
    print(f"Score médian     : {float(__import__('numpy').median(scores)):.1f}")
    print(f"Max / Min scores : {scores.max()} / {scores.min()}")

# ─────────────────────────────────────────────
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bench", type=int, help="Nombre de parties à simuler pour le benchmark aléatoire")
    args = p.parse_args()

    if args.bench:
        bench(args.bench)
    else:
        interactive()
