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
    * difficult: first factor 10–19 (excluding 11), second factor 2–9 (no ×0/×1, no 11×single-digit)

No adaptation, no pupil input — just the task.

Dependencies (Ubuntu 20):
  pip install pygame

Usage examples:
  # Single block (easy) for 3 minutes
  python cognitive_task.py --level easy --duration 180

  # Sequenced blocks with a 15 s break and larger UI
  python cognitive_task.py --sequence easy:180 difficult:180 --intermission 15 --size 1920x1080 --scale 1.2

  # Fullscreen
  python cognitive_task.py --fullscreen

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
SCREEN_W, SCREEN_H = 1920, 1080  # big by default; can override with --size
BG_GRAY = (150, 150, 150)
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 200)
GREEN = (0, 150, 0)
RED = (200, 0, 0)

EASY_A = (0, 9)
EASY_B = (0, 9)
DIFF_A = (10, 19)   # a ∈ [10..19], but we exclude 11 below
DIFF_B = (3, 9)     # b ∈ [2..9] — no ×0 or ×1
DIFF_EXCLUDE_A = {11}

# ------------------------------ Helpers ------------------------------
class Problem:
    __slots__ = ("a","b","text","answer")
    def __init__(self, a: int, b: int):
        self.a, self.b = a, b
        self.text = f"{a} × {b}"
        self.answer = a * b

def gen_problem(level: str) -> Problem:
    if level == 'easy':
        a = random.randint(*EASY_A)
        b = random.randint(*EASY_B)
        return Problem(a, b)
    elif level == 'difficult':
        # Constraints:
        # - a in 10..19 except 11
        # - b in 2..9 (no ×0 or ×1)
        while True:
            a = random.randint(*DIFF_A)
            if a in DIFF_EXCLUDE_A:
                continue
            b = random.randint(*DIFF_B)
            return Problem(a, b)
    else:
        raise ValueError('Unknown level: ' + str(level))

# ------------------------------ Core Task ------------------------------
class MultiplicationScroller:
    def __init__(self, args):
        pygame.init()

        # Size & scaling
        self.scale = float(args.scale)
        if args.size:
            try:
                w, h = args.size.lower().split('x')
                sw, sh = int(w), int(h)
            except Exception:
                raise SystemExit("--size must be like 1920x1080")
        else:
            sw, sh = SCREEN_W, SCREEN_H

        flags = pygame.FULLSCREEN if args.fullscreen else 0
        self.screen = pygame.display.set_mode((sw, sh), flags)
        pygame.display.set_caption("Cognitive Task — Multiplication Scroller")
        self.clock = pygame.time.Clock()

        # Fonts scale with window height
        h = self.screen.get_height()
        big_px = max(48, int(0.16 * h * self.scale))      # equation
        mid_px = max(24, int(0.06 * h * self.scale))      # HUD and feedback
        small_px = max(18, int(0.03 * h * self.scale))    # small HUD
        input_px = max(36, int(0.11 * h * self.scale))    # typed answer (large)
        self.font_big = pygame.font.SysFont(None, big_px)
        self.font_mid = pygame.font.SysFont(None, mid_px)
        self.font_small = pygame.font.SysFont(None, small_px)
        self.font_input = pygame.font.SysFont(None, input_px)

        # Parameters
        self.level = args.level
        self.duration = float(args.duration)
        self.time_limit = 5.0  # seconds, exact spec

        # Sequencer: tokens like "easy:180"
        self.sequence = []
        if args.sequence:
            for tok in args.sequence:
                try:
                    lvl, dur = tok.split(":")
                    lvl = lvl.strip().lower()
                    if lvl not in ("easy", "difficult"):
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

        # Visual feedback flash
        self.flash_kind = None   # 'ok' or 'err'
        self.flash_until = 0.0

    # -------------------- UX Helpers --------------------
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
            self.screen.blit(msg, msg.get_rect(center=(self.screen.get_width()//2, self.screen.get_height()//2 - 20)))
            self.screen.blit(sub, sub.get_rect(center=(self.screen.get_width()//2, self.screen.get_height()//2 + 20)))
            pygame.display.flip()
            self.clock.tick(30)

    # -------------------- Core Block Loop --------------------
    def run_block(self, level: str, duration: float, block_index: int):
        start_t = time.time()
        t = start_t
        typed = ''
        problem = gen_problem(level)
        problem_start_t = t
        self.flash_kind = None
        self.flash_until = 0.0

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
                        # flash feedback
                        self.flash_kind = 'ok' if is_correct else 'err'
                        self.flash_until = t + 0.8
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
                self.flash_kind = 'err'
                self.flash_until = t + 0.8

            # ---- Render ----
            self.screen.fill(BG_GRAY)

            # Flash overlay for feedback
            if self.flash_kind and t < self.flash_until:
                border_color = GREEN if self.flash_kind == 'ok' else RED
                pygame.draw.rect(self.screen, border_color, pygame.Rect(10, 10, self.screen.get_width()-20, self.screen.get_height()-20), width=10)
                msg_txt = "Correct" if self.flash_kind == 'ok' else "Wrong"
                msg_col = border_color
                msg = self.font_mid.render(msg_txt, True, msg_col)
                self.screen.blit(msg, msg.get_rect(center=(self.screen.get_width()//2, int(self.screen.get_height()*0.12))))
            else:
                self.flash_kind = None

            # Equation position (centered horizontally, moving top→bottom over 5s)
            eq_surf = self.font_big.render(problem.text, True, BLACK)
            eq_rect = eq_surf.get_rect(center=(self.screen.get_width()//2, int(progress*self.screen.get_height())))
            self.screen.blit(eq_surf, eq_rect)

            # Typed input just below the equation — larger font
            typed_surf = self.font_input.render(typed or ' ', True, BLUE)
            typed_rect = typed_surf.get_rect(center=(self.screen.get_width()//2, min(eq_rect.bottom + int(0.06*self.screen.get_height()), self.screen.get_height() - 60)))
            self.screen.blit(typed_surf, typed_rect)

            # HUD
            rem_block = int(max(0, duration - (t - start_t)))
            hud_lines = [
                f"Block {block_index+1}",
                f"Level: {level}",
                f"Time left in block: {rem_block} s",
                "Press ESC to quit",
            ]
            for i, line in enumerate(hud_lines):
                surf = self.font_small.render(line, True, BLACK)
                self.screen.blit(surf, (20, 20 + i*int(24*self.scale)))

            # Progress bar for current equation (remaining time to bottom)
            rem = max(0.0, self.time_limit - (t - problem_start_t))
            pct = rem / self.time_limit
            bar_w = int(pct * (self.screen.get_width() * 0.6))
            pygame.draw.rect(self.screen, (60,60,60), pygame.Rect(self.screen.get_width()*0.2, self.screen.get_height()-60, int(self.screen.get_width()*0.6), 16))
            pygame.draw.rect(self.screen, (0,120,0), pygame.Rect(self.screen.get_width()*0.2, self.screen.get_height()-60, bar_w, 16))

            pygame.display.flip()

    # -------------------- Orchestration --------------------
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
    ap.add_argument('--size', help='Window size, e.g., 1920x1080')
    ap.add_argument('--scale', type=float, default=1.0, help='Global UI scale multiplier (fonts, HUD)')
    ap.add_argument('--fullscreen', action='store_true', help='Fullscreen mode')
    ap.add_argument('--output-dir', default='./logs', help='Directory for CSV logs')
    args = ap.parse_args()

    app = MultiplicationScroller(args)
    app.run()
