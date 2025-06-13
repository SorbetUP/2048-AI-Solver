from __future__ import annotations

import argparse
import pygame
from typing import Literal
from game import Game

# ─────────────────────────────────────────────────────────────────────────────
class Pygame2048UI:
    # --- constantes visuelles ------------------------------------------------
    TILE_COLORS = {
        0: (205, 193, 180), 2: (238, 228, 218), 4: (237, 224, 200),
        8: (242, 177, 121), 16: (245, 149, 99), 32: (246, 124, 95),
        64: (246, 94, 59), 128: (237, 207, 114), 256: (237, 204, 97),
        512:(237, 200, 80), 1024:(237,197,63), 2048:(237,194,46),
        4096:(60,  58,  50)
    }

    def __init__(self, tile_size: int = 120, margin: int = 20,max_fps :int = 30):
        # géométrie -----------------------------------------------------------
        self.TAILLE  = tile_size
        self.MARGE   = margin
        self.N       = 4
        self.WIDTH   = self.N * (self.TAILLE + self.MARGE) + self.MARGE
        self.HEIGHT  = self.N * (self.TAILLE + self.MARGE) + self.MARGE + 150

        # pygame init ---------------------------------------------------------
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("2048 (classe UI)")
        self.clock  = pygame.time.Clock()

        # polices -------------------------------------------------------------
        self.FONT       = pygame.font.SysFont("Segoe UI", 36,  bold=True)
        self.FONT_SCORE = pygame.font.SysFont("Segoe UI", 28,  bold=True)
        self.FONT_MSG   = pygame.font.SysFont("Segoe UI", 28)
        self.FONT_BTN   = pygame.font.SysFont("Segoe UI", 26,  bold=True)

        # état du jeu ---------------------------------------------------------
        self.game:      Game = Game()
        self.show_popup: bool = False
        self.buttons:    dict[str, pygame.Rect] = {}
        self.running:    bool = False

        # fps ----------------------------------------------------------------
        self.max_fps = max_fps

        # rendu initial -------------------------------------------------------
        self._render()

    # ───────────────────────── utilitaires visuels ──────────────────────────
    def _text_color(self, val: int):
        return (119, 110, 101) if val <= 4 else (255, 255, 255)

    def _draw_board(self):
        s = self.screen
        s.fill((250, 248, 239))
        # score
        txt = self.FONT_SCORE.render(f"Score : {self.game.score}", True, (0,0,0))
        r   = txt.get_rect(topright=(self.WIDTH - self.MARGE, self.MARGE))
        cadre = r.inflate(30,20)
        pygame.draw.rect(s, (237,224,200), cadre, border_radius=8)
        s.blit(txt, r)
        # grille
        off_y = cadre.bottom + self.MARGE
        tiles = [((self.game.board.raw >> (i*4)) & 0xF) for i in range(16)]
        for i in range(self.N):
            for j in range(self.N):
                p   = tiles[i*4+j]
                val = 0 if p==0 else 1<<p
                rect = pygame.Rect(
                    self.MARGE + j*(self.TAILLE+self.MARGE),
                    off_y + i*(self.TAILLE+self.MARGE),
                    self.TAILLE, self.TAILLE)
                pygame.draw.rect(s, self.TILE_COLORS.get(val,(60,58,50)), rect, border_radius=8)
                if val:
                    t = self.FONT.render(str(val), True, self._text_color(val))
                    s.blit(t, t.get_rect(center=rect.center))

    # -------------------------- pop‑up --------------------------------------
    def _draw_popup(self, msg: str):
        s = self.screen
        overlay = pygame.Surface((self.WIDTH,self.HEIGHT), pygame.SRCALPHA)
        overlay.fill((0,0,0,180))
        s.blit(overlay,(0,0))
        # fenêtre
        w,h=540,280
        x=(self.WIDTH-w)//2; y=(self.HEIGHT-h)//2
        cadre=pygame.Rect(x,y,w,h)
        pygame.draw.rect(s,(250,248,239),cadre,border_radius=15)
        pygame.draw.rect(s,(100,100,100),cadre,3,border_radius=15)
        # message
        t=self.FONT_MSG.render(msg,True,(50,50,50))
        s.blit(t,t.get_rect(center=(self.WIDTH//2,y+70)))
        # boutons
        bw,bh,esp=220,70,40
        total=bw*2+esp
        start_x=x+(w-total)//2
        by=y+h-bh-50
        self.buttons={
            "restart":pygame.Rect(start_x,by,bw,bh),
            "quit"   :pygame.Rect(start_x+bw+esp,by,bw,bh)}
        pygame.draw.rect(s,(100,200,100),self.buttons["restart"],border_radius=10)
        pygame.draw.rect(s,(200,100,100),self.buttons["quit"],border_radius=10)
        s.blit(self.FONT_BTN.render("Recommencer",True,(255,255,255)),
                self.FONT_BTN.render("Recommencer",True,(255,255,255)).get_rect(center=self.buttons["restart"].center))
        s.blit(self.FONT_BTN.render("Quitter",True,(255,255,255)),
                self.FONT_BTN.render("Quitter",True,(255,255,255)).get_rect(center=self.buttons["quit"].center))

    # -------------------------- rendu global --------------------------------
    def _render(self):
        self._draw_board()
        if self.show_popup:
            self._draw_popup("🎉 Bravo !" if self.game.is_won() else "💀 Partie terminée.")
        pygame.display.flip()

    # ─────────────────────────── API publique ───────────────────────────────
    def ai_move(self, direction: Literal["up","down","left","right"]) -> bool:
        if self.show_popup:
            return False
        moved,_ = self.game.move(direction)
        if moved and self.game.is_over():
            self.show_popup = True
        if moved:
            self._render()
        return moved

    def tick(self):
        self.clock.tick(self.max_fps)
        for e in pygame.event.get():
            if e.type==pygame.QUIT:
                pygame.quit(); raise SystemExit
            if e.type==pygame.KEYDOWN and not self.show_popup:
                key_map={pygame.K_UP:"up",pygame.K_DOWN:"down",pygame.K_LEFT:"left",pygame.K_RIGHT:"right"}
                if e.key in key_map:
                    self.ai_move(key_map[e.key])
                if e.key==pygame.K_r:
                    self.game=Game(); self.show_popup=False; self._render()
            if e.type==pygame.MOUSEBUTTONDOWN and self.show_popup:
                pos=pygame.mouse.get_pos()
                if self.buttons.get("restart") and self.buttons["restart"].collidepoint(pos):
                    self.game=Game(); self.show_popup=False; self._render()
                elif self.buttons.get("quit") and self.buttons["quit"].collidepoint(pos):
                    pygame.quit(); raise SystemExit

    def run(self):
        """Boucle principale bloquante ; gère clavier & souris.``ai_move`` reste utilisable."""
        self.running=True
        while self.running:
            self.tick()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fps", type=int, help="Nombre d'image par seconde")
    args = p.parse_args()
    if args.fps:
        ui = Pygame2048UI(max_fps = args.fps)
    else :
        ui = Pygame2048UI()
    ui.run()