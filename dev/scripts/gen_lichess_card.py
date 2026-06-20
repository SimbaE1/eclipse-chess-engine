#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render EclipseBOT's latest Lichess game as an animated SVG card for the README.

A no-Vercel take on the Lichess-Game-Readme serverless app: instead of a server
generating the image live per request, the GitHub Action in
.github/workflows/lichess-readme.yml runs this on a schedule and commits
assets/lichess_latest_game.svg, which GitHub serves statically.

The card replays the whole game move-by-move at 1s/move on an infinite loop. It's
a single self-contained SVG: the piece shapes (<defs>) and the board squares are
emitted once, then one <g class="fr"> per half-move holds just that position's
pieces + last-move highlight. A CSS keyframe animation shows one frame per second
(each <g> staggered by its own animation-delay). CSS animation inside an SVG
embedded via <img> runs on GitHub (same technique as the README "snake" cards),
and because it's vector text it stays small and diffs well in git — unlike a GIF.

Ratings are handled separately by live shields.io badges in the README.

Only dependency beyond the stdlib is python-chess (`pip install chess`).
"""
from __future__ import annotations

import datetime
import io
import json
import re
import sys
import urllib.request
import xml.dom.minidom as minidom

import chess
import chess.pgn
import chess.svg

USER = "EclipseBOT"
OUT = "assets/lichess_latest_game.svg"
BOARDPX = 360         # python-chess board edge (internal viewBox is 0 0 360 360)
FOOTER = 48
H = BOARDPX + FOOTER
BG = "#312e2b"
FG = "#e8e6e3"
SUB = "#b0aca8"

_DEFS_RE = re.compile(r"<defs>.*?</defs>", re.S)
_SQUARE_RE = re.compile(r'<rect[^>]*class="square[^"]*"[^>]*/>')
_LASTMOVE_RE = re.compile(r"<rect[^>]*lastmove[^>]*/>")
_USE_RE = re.compile(r"<use [^>]*/>")


def _fetch(url: str, accept: str = "application/json") -> str:
    req = urllib.request.Request(
        url, headers={"Accept": accept, "User-Agent": f"{USER}-readme-card"}
    )
    with urllib.request.urlopen(req, timeout=25) as r:
        return r.read().decode("utf-8")


def latest_game() -> dict:
    raw = _fetch(
        f"https://lichess.org/api/games/user/{USER}"
        "?max=1&pgnInJson=true&clocks=false&evals=false&opening=true",
        accept="application/x-ndjson",
    ).strip()
    if not raw:
        raise SystemExit("no games returned for user")
    return json.loads(raw.splitlines()[0])


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _render(board: chess.Board, lastmove, orient: bool) -> str:
    return chess.svg.board(
        board, lastmove=lastmove, orientation=orient, size=BOARDPX, coordinates=False
    )


def build_svg(g: dict) -> str:
    game = chess.pgn.read_game(io.StringIO(g["pgn"]))

    white = (g["players"]["white"].get("user") or {}).get("name") or "?"
    black = (g["players"]["black"].get("user") or {}).get("name") or "?"
    ecl_white = white.lower() == USER.lower()
    opp = black if ecl_white else white
    opp_rating = g["players"]["black" if ecl_white else "white"].get("rating", "")
    orient = chess.WHITE if ecl_white else chess.BLACK

    winner = g.get("winner")
    if winner is None:
        result, color = "½–½ draw", SUB
    elif (winner == "white") == ecl_white:
        result, color = "1–0 win", "#7fa650"
    else:
        result, color = "0–1 loss", "#c75c5c"
    sub = _esc(f"{g.get('speed','')} · by {g.get('status','')} · "
               + datetime.datetime.fromtimestamp(
                   g.get("createdAt", 0) / 1000, datetime.timezone.utc
               ).strftime("%b %-d, %Y"))

    # Build every position: the start, then after each half-move (with the move
    # that produced it, for the highlight).
    board = game.board()
    states = [(board.copy(), None)]
    for mv in game.mainline_moves():
        board.push(mv)
        states.append((board.copy(), mv))
    n = len(states)

    # Shared defs + static squares: emit once from a plain (no-highlight) render.
    base = _render(game.board(), None, orient)
    defs = _DEFS_RE.search(base).group(0)
    squares = "".join(_SQUARE_RE.findall(base))

    # One <g> per frame holding only that position's highlight + pieces.
    frames = []
    for i, (bd, last) in enumerate(states):
        fr = _render(bd, last, orient)
        hi = "".join(_LASTMOVE_RE.findall(fr))
        uses = "".join(_USE_RE.findall(fr))
        frames.append(f'<g class="fr" style="animation-delay:{i}s">{hi}{uses}</g>')

    # Each frame is opacity:0 by default and shown for exactly 1s of an n-second
    # loop; step-end makes the toggle instant, animation-delay staggers them.
    vis_pct = 100.0 / n
    css = (
        ".fr{opacity:0;animation:play %ds step-end infinite}"
        "@keyframes play{0%%{opacity:1}%.4f%%{opacity:0}100%%{opacity:0}}"
        % (n, vis_pct)
    )

    opp_label = _esc(opp) + (f" ({opp_rating})" if opp_rating else "")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{BOARDPX}" height="{H}" viewBox="0 0 {BOARDPX} {H}" '
        f'font-family="Segoe UI, Helvetica, Arial, sans-serif">\n'
        f"<style>{css}</style>\n"
        f'<rect width="{BOARDPX}" height="{H}" fill="{BG}"/>\n'
        f"{defs}\n{squares}\n"
        f"{''.join(frames)}\n"
        f'<text x="6" y="{BOARDPX + 22}" fill="{FG}" font-size="15" '
        f'font-weight="600">vs {opp_label}</text>\n'
        f'<text x="{BOARDPX - 6}" y="{BOARDPX + 22}" fill="{color}" font-size="15" '
        f'font-weight="700" text-anchor="end">{result}</text>\n'
        f'<text x="6" y="{BOARDPX + 40}" fill="{SUB}" font-size="12">{sub}</text>\n'
        f"</svg>\n"
    )


def main() -> None:
    g = latest_game()
    svg = build_svg(g)
    minidom.parseString(svg)  # fail loudly on malformed XML before committing
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"wrote {OUT} ({len(svg)} bytes) — game {g.get('id')}", file=sys.stderr)


if __name__ == "__main__":
    main()
