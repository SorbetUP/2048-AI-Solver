"""
interface_jeu_pygame.py
───────────────────────────────────────────────────────────────
• --auto [ia|bepp]   → lance la fenêtre directement en IA
                      (MoveNet par défaut)
• --movenet <path>   → chemin modèle .joblib alternatif
• --headless         → aucun rendu graphique (BG/bench only)
• Presets turbo / rollout (voir README)
"""

from __future__ import annotations
import argparse, csv, uuid, threading, time, multiprocessing as mp, os, sys
from typing import Optional, Literal

from game import Game
from search import expectimax
from search.expectimax import best_move as bepp_best_move
from search.fast_expectimax import fast_best_move
from eval.heuristics import bounded_eval

# ─────────────────────────── MoveNet loader ────────────────────────────
def load_movenet(path: str | None):
    """
    Charge algo/movenet.py (HistGradientBoostingClassifier wrapper).
    À l’échec, renvoie None → fallback BEPP.
    """
    try:
        from algo.movenet import MoveNet
        return MoveNet(path or "model/hgb_2048.joblib")
    except Exception as e:
        print(f"[WARN] MoveNet non disponible : {e}", file=sys.stderr)
        return None

# ─────────────────────────── CSV logger ────────────────────────────────
class DataLogger:
    COLS = ["game_id","move_idx","score","max_tile","empty_cnt",
            "bepp2_move","bepp2_val"] + [f"c{i}" for i in range(16)]

    def __init__(self, path: Optional[str]):
        self.path = path
        self.lock = threading.Lock()
        if path:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "a", newline="") as f:
                if not f.tell():
                    csv.writer(f).writerow(self.COLS)

    def record(self, *, gid:str, idx:int, raw:int, score:int,
               bepp2_move:str, bepp2_val:float):
        empties, cells = 0, []
        for p in range(16):
            exp = (raw >> (p*4)) & 0xF
            empties += (exp == 0)
            cells.append(0 if exp == 0 else 1 << exp)
        with self.lock, open(self.path,"a",newline="") as f:
            csv.writer(f).writerow(
                [gid, idx, score, max(cells), empties,
                 bepp2_move, round(bepp2_val,5)] + cells
            )

# ─────────────────────── IA « headless » (BG / bench) ─────────────────────
def _play_game(depth:int, ms:int, engine, csv_path:Optional[str]):
    logger = DataLogger(csv_path) if csv_path else None
    g = Game(); gid = str(uuid.uuid4()); idx = 0
    while not g.is_over():
        mv = engine(g.board, depth, ms) if engine is bepp_best_move \
             else engine(g.board)
        bepp2_mv  = bepp_best_move(g.board, depth=2, time_limit_ms=10)
        bepp2_val = bounded_eval(g.board)
        if logger:
            snap = g.board.clone(); snap.move(mv, add_random=False)
            logger.record(gid=gid, idx=idx, raw=snap.raw, score=g.score,
                          bepp2_move=bepp2_mv, bepp2_val=bepp2_val)
        idx += 1
        g.move(mv)
    return g.score

def _bench_mp(n, depth, ms, workers, engine):
    import numpy as np
    workers = max(1, min(workers, n))
    t0 = time.perf_counter()
    with mp.Pool(workers) as pool:
        scores = pool.starmap(_play_game, [(depth, ms, engine, None)]*n)
    dt = time.perf_counter() - t0
    scores = np.asarray(scores)
    print(f"{n/dt:,.1f} parties/s ({workers} proc) – durée {dt:.3f}s")
    print(f"Moy {scores.mean():.1f}  Méd {float(np.median(scores)):.1f}  "
          f"Max {scores.max()}  Min {scores.min()}")

def _bg_worker_mp(n_games, depth, ms, csv_path, workers, engine):
    todo = float("inf") if n_games == "inf" else int(n_games)
    done = 0
    with mp.Pool(workers) as pool:
        while done < todo:
            chunk = 64 if todo == float("inf") else min(64, todo-done)
            pool.starmap(_play_game, [(depth, ms, engine, csv_path)]*chunk)
            done += chunk
            if done % 100 == 0:
                print(f"[BG] {done:,} parties enregistrées → {csv_path}", flush=True)

# ───────────────────────────── Pygame UI ─────────────────────────────
class Pygame2048UI:
    def __init__(self, *, fps:int, speed:float, depth:int, ms:int,
                 logger_path:Optional[str], engine, start_ai:bool):
        import pygame
        self.pg = pygame
        self.speed = speed
        self.engine, self.depth, self.ms = engine, depth, ms
        self.logger = DataLogger(logger_path) if logger_path else None

        # --- UI -----------------------------------------------------------
        self.T, self.M = 120, 20
        self.W, self.H = 4*(self.T+self.M)+self.M, 4*(self.T+self.M)+self.M+150
        pygame.init()
        self.screen = pygame.display.set_mode((self.W, self.H))
        pygame.display.set_caption("2048 — A=IA  ·  R=reset")
        self.clock = pygame.time.Clock()

        self.F_BIG = pygame.font.SysFont("Segoe UI", 36, True)
        self.F_SCO = pygame.font.SysFont("Segoe UI", 28, True)
        self.F_MSG = pygame.font.SysFont("Segoe UI", 28)
        self.F_BTN = pygame.font.SysFont("Segoe UI", 26, True)

        self.colors = {0:(205,193,180),2:(238,228,218),4:(237,224,200),
            8:(242,177,121),16:(245,149,99),32:(246,124,95),64:(246,94,59),
            128:(237,207,114),256:(237,204,97),512:(237,200,80),
            1024:(237,197,63),2048:(237,194,46),4096:(60,58,50)}

        self.fps = fps
        self.ai_delay = int(150 / max(self.speed, .1))

        self.game = Game()
        self.gid = str(uuid.uuid4())
        self.move_idx = 0
        self.ai_on = start_ai
        self.show_pop = False
        self.last_ai = pygame.time.get_ticks()
        self.buttons = {}

        self._render()

    # ---------- Rendu ----------------------------------------------------
    def _col(self, v): return (119,110,101) if v <= 4 else (255,255,255)

    def _draw_board(self):
        pg,s = self.pg, self.screen
        s.fill((250,248,239))
        txt = self.F_SCO.render(f"Score : {self.game.score}", True, (0,0,0))
        r = txt.get_rect(topright=(self.W-self.M, self.M))
        pg.draw.rect(s, (237,224,200), r.inflate(30,20), border_radius=8)
        s.blit(txt, r)

        off = r.bottom + self.M
        tiles = [((self.game.board.raw>>(i*4)) & 0xF) for i in range(16)]
        for i in range(4):
            for j in range(4):
                e = tiles[i*4+j]; v = 0 if e == 0 else 1 << e
                rect = pg.Rect(self.M + j*(self.T+self.M),
                               off + i*(self.T+self.M),
                               self.T, self.T)
                pg.draw.rect(s, self.colors.get(v,(60,58,50)),
                             rect, border_radius=8)
                if v:
                    t = self.F_BIG.render(str(v), True, self._col(v))
                    s.blit(t, t.get_rect(center=rect.center))

        if self.ai_on and not self.show_pop:
            tag = self.F_SCO.render("AUTO-IA", True, (255,0,0))
            s.blit(tag, tag.get_rect(topleft=(self.M, self.M)))

    def _draw_popup(self, msg:str):
        pg,s = self.pg, self.screen
        ov = pg.Surface((self.W,self.H), pg.SRCALPHA)
        ov.fill((0,0,0,180)); s.blit(ov,(0,0))
        w,h = 540,280; x=(self.W-w)//2; y=(self.H-h)//2
        box = pg.Rect(x,y,w,h)
        pg.draw.rect(s,(250,248,239),box,border_radius=15)
        pg.draw.rect(s,(100,100,100),box,3,border_radius=15)
        s.blit(self.F_MSG.render(msg,True,(50,50,50)),
               self.F_MSG.render(msg,True,(50,50,50))
               .get_rect(center=(self.W//2,y+70)))
        bw,bh,esp = 220,70,40
        bx = x+(w-(bw*2+esp))//2; by=y+h-bh-50
        self.buttons = {"restart":pg.Rect(bx,by,bw,bh),
                        "quit":   pg.Rect(bx+bw+esp,by,bw,bh)}
        pg.draw.rect(s,(100,200,100), self.buttons["restart"], border_radius=10)
        pg.draw.rect(s,(200,100,100), self.buttons["quit"],    border_radius=10)
        for k,lbl in (("restart","Recommencer"),("quit","Quitter")):
            t=self.F_BTN.render(lbl,True,(255,255,255))
            s.blit(t,t.get_rect(center=self.buttons[k].center))

    def _render(self):
        self._draw_board()
        if self.show_pop:
            self._draw_popup("🎉 Bravo !" if self.game.is_won() else "💀 Partie terminée.")
        self.pg.display.flip()

    # ---------- Moteur ----------------------------------------------------
    def _play_engine(self):
        return (self.engine(self.game.board, self.depth, self.ms)
                if self.engine is bepp_best_move else
                self.engine(self.game.board))

    def _log_current(self, mv:str):
        if not self.logger: return
        bepp2_mv  = bepp_best_move(self.game.board, depth=2, time_limit_ms=10)
        bepp2_val = bounded_eval(self.game.board)
        snap = self.game.board.clone(); snap.move(mv, add_random=False)
        self.logger.record(gid=self.gid, idx=self.move_idx, raw=snap.raw,
                           score=self.game.score,
                           bepp2_move=bepp2_mv, bepp2_val=bepp2_val)

    # ----------- IA / manuel / restart -----------------------------------
    def _ai_step(self):
        if not self.ai_on or self.show_pop: return
        if self.pg.time.get_ticks() - self.last_ai < self.ai_delay: return
        mv = self._play_engine()
        self._log_current(mv)
        self.move_idx += 1
        self.game.move(mv)
        if self.game.is_over(): self.show_pop = True
        self.last_ai = self.pg.time.get_ticks()
        self._render()

    def _manual(self, direction:str):
        if self.show_pop: return
        self._log_current(direction)
        self.move_idx += 1
        moved,_ = self.game.move(direction)
        if moved and self.game.is_over(): self.show_pop = True
        if moved: self._render()

    def _restart(self):
        """Redémarre une partie en conservant *speed* et paramètres actuels."""
        self.__init__(fps=self.fps, speed=self.speed,
                      depth=self.depth, ms=self.ms,
                      logger_path=self.logger.path if self.logger else None,
                      engine=self.engine, start_ai=False)

    def tick(self):
        self.clock.tick(self.fps)
        self._ai_step()
        for e in self.pg.event.get():
            if e.type == self.pg.QUIT:
                self.pg.quit(); sys.exit()
            if e.type == self.pg.KEYDOWN and not self.show_pop:
                km = {self.pg.K_UP:"up", self.pg.K_DOWN:"down",
                      self.pg.K_LEFT:"left", self.pg.K_RIGHT:"right"}
                if e.key in km:
                    self.ai_on=False; self._manual(km[e.key])
                if e.key == self.pg.K_a:
                    self.ai_on = not self.ai_on; self._render()
                if e.key == self.pg.K_r:
                    self._restart()
            if e.type == self.pg.MOUSEBUTTONDOWN and self.show_pop:
                pos = self.pg.mouse.get_pos()
                if self.buttons["restart"].collidepoint(pos):
                    self._restart()
                elif self.buttons["quit"].collidepoint(pos):
                    self.pg.quit(); sys.exit()

    def run(self):
        while True: self.tick()

# ───────────────────────────── CLI & main ─────────────────────────────
if __name__ == "__main__":
    mp.freeze_support()

    pa = argparse.ArgumentParser()
    pa.add_argument("--preset", choices=["default","turbo","rollout"], default="default")
    pa.add_argument("--depth", type=int,   default=3)
    pa.add_argument("--time",  type=int,   default=60)
    pa.add_argument("--beam",  type=int,   default=2)
    pa.add_argument("--prob",  type=float, default=0.04)
    pa.add_argument("--fps",   type=int,   default=30)
    pa.add_argument("--speed", type=float, default=1.0)
    pa.add_argument("--save")                # CSV dataset
    pa.add_argument("--bg")                  # parties en arrière-plan
    pa.add_argument("--bench", type=int)     # benchmark
    pa.add_argument("--workers", type=int, default=max(1, mp.cpu_count()//2))
    pa.add_argument("--headless", action="store_true")
    pa.add_argument("--movenet",  help="chemin modèle MoveNet .joblib")
    pa.add_argument("--auto", nargs="?", const="ia",
                    choices=["ia","bepp"],
                    help="démarre l’UI en mode IA (MoveNet ou BEPP)")
    args = pa.parse_args()

    # --- presets ---------------------------------------------------------
    if args.preset == "turbo":
        args.depth, args.time, args.beam, args.prob = 2, 40, 1, 0.10
        default_engine = bepp_best_move
    elif args.preset == "rollout":
        args.depth = 3
        default_engine = fast_best_move
    else:
        default_engine = bepp_best_move

    expectimax.set_bepp_params(prob_cutoff=args.prob, beam_k=args.beam)

    # --- moteur MoveNet (si dispo) ---------------------------------------
    movenet_engine = load_movenet(args.movenet)
    ia_engine = movenet_engine or default_engine

    # ---------- bench only ----------------------------------------------
    if args.bench:
        _bench_mp(args.bench, args.depth, args.time,
                  args.workers, ia_engine)
        sys.exit()

    # ---------- BG dataset ----------------------------------------------
    csv_path = args.save or "dataset.csv"
    if args.bg:
        n = "inf" if args.bg.lower() == "inf" else int(args.bg)
        threading.Thread(target=_bg_worker_mp,
                         args=(n, args.depth, args.time,
                               csv_path, args.workers, ia_engine),
                         daemon=True).start()

    # ---------- headless -------------------------------------------------
    if args.headless:
        try:
            while True: time.sleep(3600)
        except KeyboardInterrupt:
            print("Arrêt demandé.")
        sys.exit()

    # ---------- sélection moteur pour l’UI ------------------------------
    if args.auto == "bepp":
        ui_engine, start_ai = bepp_best_move, True
    elif args.auto == "ia":
        ui_engine, start_ai = ia_engine, True
    else:                               # auto non fourni
        ui_engine, start_ai = ia_engine, False

    # ---------- lance la fenêtre ----------------------------------------
    import pygame
    Pygame2048UI(fps=args.fps, speed=args.speed,
                 depth=args.depth, ms=args.time,
                 logger_path=csv_path if args.save else None,
                 engine=ui_engine, start_ai=start_ai).run()
