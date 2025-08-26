#!/usr/bin/env python3
"""
Cognitive Task (Pygame) — Multiplication Scroller (Ubuntu 20)

Implements exactly the "Cognitive Task" described in the provided PDF:
- Equations (multiplications) scroll from the top to the bottom of the screen within 5 seconds.
- Participant types the result using the keyboard.
- If correct before reaching the bottom → immediately replaced by a new one at the top.
- If incorrect (on Enter) or time runs out → counts as error and a new one appears at the top.
- Two difficulty levels:
    * easy: both factors uniformly sampled from 0–9
    * difficult: first factor 10–19, second factor 0–9

No adaptation, no pupil input — just the task.

Dependencies (Ubuntu 20):
  pip install pygame

Usage examples:
  # Easy for 3 minutes
  python cognitive_task.py --level easy --duration 180

  # Difficult for 3 minutes
  python cognitive_task.py --level difficult --duration 180

  # Fullscreen (Esc to quit)
  python cognitive_task.py --level easy --fullscreen

  # Sequenced blocks with breaks
  python cognitive_task.py --sequence easy:180 difficult:180 easy:60 --intermission 15

Outputs:
  - CSV log per run in ./logs with per-problem correctness and response time.
"""
import argparse
import csv
import os
import sys
import time
import random

import pygame

# ------------------------------ Config ------------------------------
FPS = 60
SCREEN_W, SCREEN_H = 1280, 720
BG_GRAY = (150, 150, 150)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 200)

EASY_A = (0, 9)
EASY_B = (0, 9)
DIFF_A = (10, 19)
DIFF_B = (0, 9)

# ------------------------------ Helpers ------------------------------
class Problem:
    __slots__ = ("a","b","text","answer")
    def __init__(self, a:int, b:int):
        self.a, self.b = a, b
        self.text = f"{a} × {b}"
        self.answer = a*b

def gen_problem(level: str) -> Problem:
    if level == 'easy':
        a = random.randint(*EASY_A)
        b = random.randint(*EASY_B)
    elif level == 'difficult':
        a = random.randint(*DIFF_A)
        b = random.randint(*DIFF_B)
    else:
        raise ValueError('Unknown level')
    return Problem(a, b)

# ------------------------------ Core Task ------------------------------
class MultiplicationScroller:
    def __init__(self, args):
        pygame.init()
        flags = pygame.FULLSCREEN if args.fullscreen else 0
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H), flags)
        pygame.display.set_caption("Cognitive Task — Multiplication Scroller")
        self.clock = pygame.time.Clock()
        self.font_big = pygame.font.SysFont(None, 160)
        self.font_mid = pygame.font.SysFont(None, 36)
        self.font_small = pygame.font.SysFont(None, 24)

        self.level = args.level
        self.duration = float(args.duration)
        self.time_limit = 5.0  # exactly as specified

        # Sequencer: tokens like "easy:180"
        self.sequence = []
        if args.sequence:
            for tok in args.sequence:
                try:
                    lvl, dur = tok.split(":")
                    lvl = lvl.strip().lower()
                    if lvl not in ("easy","difficult"):
                        raise ValueError
                    dur = float(dur)
                    self.sequence.append((lvl, dur))
                except Exception:
                    raise SystemExit(f"Invalid --sequence token '{tok}'. Use LEVEL:DURATION (e.g., easy:180)")
        self.intermission = float(args.intermission)

        # logging
        os.makedirs(args.output_dir, exist_ok=True)
        ts = time.strftime('%Y%m%d_%H%M%S')
        self.log_path = os.path.join(args.output_dir, f"run_seq_{ts}.csv" if self.sequence else f"run_{self.level}_{ts}.csv")
        self.logf = open(self.log_path, 'w', newline='')
        self.writer = csv.writer(self.logf)
        self.writer.writerow(['timestamp','block_index','level','equation','correct','typed','response_time_s'])

    def _intermission(self, seconds: float, label: str):
        if seconds <= 0:
            return
        start = time.time()
        while True:
            t = time.time()
            if t - start >= seconds:
                break
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit(0)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit(0)
            self.screen.fill(BLACK)
            msg = self.font_mid.render(f"Break — next: {label}", True, WHITE)
            rem = max(0, int(seconds - (t - start)))
            sub = self.font_small.render(f"Resuming in {rem}s  (Esc to quit)", True, WHITE)
            self.screen.blit(msg, msg.get_rect(center=(SCREEN_W//2, SCREEN_H//2 - 20)))
            self.screen.blit(sub, sub.get_rect(center=(SCREEN_W//2, SCREEN_H//2 + 20)))
            pygame.display.flip()
            self.clock.tick(30)

    def run_block(self, level: str, duration: float, block_index: int):
        start_t = time.time()
        t = start_t
        typed = ''
        problem = gen_problem(level)
        problem_start_t = t

        while True:
            self.clock.tick(FPS)
            t = time.time()
            elapsed = t - start_t
            if elapsed >= duration:
                break

            # Input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit(0)
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit(); sys.exit(0)
                    elif event.key == pygame.K_BACKSPACE:
                        typed = typed[:-1]
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        # check answer
                        try:
                            ans = int(typed) if typed else None
                        except ValueError:
                            ans = None
                        rt = t - problem_start_t
                        is_correct = (ans == problem.answer)
                        self.writer.writerow([f"{t:.6f}", block_index, level, problem.text, int(is_correct), typed, f"{rt:.3f}"])
                        # spawn next problem immediately
                        typed = ''
                        problem = gen_problem(level)
                        problem_start_t = t
                    else:
                        ch = event.unicode
                        if ch.isdigit():
                            typed += ch

            # Progress of falling equation
            progress = (t - problem_start_t) / self.time_limit
            if progress >= 1.0:
                # timeout → counts as error (correct=0) and new problem
                self.writer.writerow([f"{t:.6f}", block_index, level, problem.text, 0, typed, f"{self.time_limit:.3f}"])
                typed = ''
                problem = gen_problem(level)
                problem_start_t = t
                progress = 0.0

            # ---- Render ----
            self.screen.fill(BG_GRAY)
            # Equation position (centered horizontally, moving top→bottom over 5s)
            eq_surf = self.font_big.render(problem.text, True, BLACK)
            eq_rect = eq_surf.get_rect(center=(SCREEN_W//2, int(progress*SCREEN_H)))
            self.screen.blit(eq_surf, eq_rect)

            # Typed input just below the equation
            typed_surf = self.font_mid.render(typed or ' ', True, BLUE)
            typed_rect = typed_surf.get_rect(center=(SCREEN_W//2, min(eq_rect.bottom + 60, SCREEN_H - 40)))
            self.screen.blit(typed_surf, typed_rect)

            # HUD
            rem_block = int(max(0, duration - elapsed))
            hud_lines = [
                f"Block {block_index+1}",
                f"Level: {level}",
                f"Time left in block: {rem_block} s",
                "Press ESC to quit",
            ]
            for i, line in enumerate(hud_lines):
                surf = self.font_small.render(line, True, BLACK)
                self.screen.blit(surf, (20, 20 + i*22))

            # Progress bar for current equation (remaining time to bottom)
            rem = max(0.0, self.time_limit - (t - problem_start_t))
            pct = rem / self.time_limit
            bar_w = int(pct * (SCREEN_W * 0.6))
            pygame.draw.rect(self.screen, (60,60,60), pygame.Rect(SCREEN_W*0.2, SCREEN_H-40, int(SCREEN_W*0.6), 10))
            pygame.draw.rect(self.screen, (0,120,0), pygame.Rect(SCREEN_W*0.2, SCREEN_H-40, bar_w, 10))

            pygame.display.flip()

    def run(self):
        try:
            if self.sequence:
                for i, (lvl, dur) in enumerate(self.sequence):
                    if i > 0:
                        self._intermission(self.intermission, f"Block {i+1}: {lvl} ({int(dur)}s)")
                    self.run_block(lvl, dur, i)
            else:
                self.run_block(self.level, self.duration, 0)
        finally:
            self.logf.close()
            pygame.quit()
            print(f"[DONE] Log saved to: {self.log_path}")

# ------------------------------ Main ------------------------------
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Cognitive Task — Multiplication Scroller (easy/difficult)')
    ap.add_argument('--level', choices=['easy','difficult'], default='easy', help='Difficulty level (ignored if --sequence is used)')
    ap.add_argument('--duration', type=float, default=180, help='Duration in seconds (ignored if --sequence is used)')
    ap.add_argument('--sequence', nargs='*', help='Run multiple blocks, e.g., easy:180 difficult:180 easy:60')
    ap.add_argument('--intermission', type=float, default=0, help='Seconds of break screen between blocks when using --sequence')
    ap.add_argument('--fullscreen', action='store_true', help='Fullscreen mode')
    ap.add_argument('--output-dir', default='./logs', help='Directory for CSV logs')
    args = ap.parse_args()

    app = MultiplicationScroller(args)
    app.run()
