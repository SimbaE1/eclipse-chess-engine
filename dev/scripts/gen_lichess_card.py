#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Render EclipseBOT's latest Lichess game as a static SVG card for the README.

A no-Vercel alternative to the Lichess-Readme / Lichess-Game-Readme serverless
projects: instead of a server generating the SVG live per request, the GitHub
Action in .github/workflows/lichess-readme.yml runs this on a schedule and
commits assets/lichess_latest_game.svg, which GitHub then serves statically.

The board shows the final position (last move highlighted, oriented to Eclipse's
side) plus a caption (opponent, result, speed, date) baked into the SVG so the
README block itself never has to change. Ratings are handled separately by live
shields.io badges in the README, so they need no commits at all.

Only dependency beyond the stdlib is python-chess (`pip install chess`).
"""
from __future__ import annotations

import datetime
import io
import json
import sys
import urllib.request
import xml.dom.minidom as minidom

import chess
import chess.pgn
import chess.svg

USER = "EclipseBOT"
OUT = "assets/lichess_latest_game.svg"
BOARD = 336          # board edge in px
PAD = 12
W = BOARD + 2 * PAD  # 360
FOOTER = 50
H = PAD + BOARD + FOOTER
BG = "#312e2b"
FG = "#e8e6e3"
SUB = "#b0aca8"


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


def build_svg(g: dict) -> str:
    game = chess.pgn.read_game(io.StringIO(g["pgn"]))
    board = game.board()
    last = None
    for mv in game.mainline_moves():
        board.push(mv)
        last = mv

    white = (g["players"]["white"].get("user") or {}).get("name") or "?"
    black = (g["players"]["black"].get("user") or {}).get("name") or "?"
    ecl_white = white.lower() == USER.lower()
    opp = black if ecl_white else white
    opp_rating = g["players"]["black" if ecl_white else "white"].get("rating", "")
    orientation = chess.WHITE if ecl_white else chess.BLACK

    winner = g.get("winner")
    if winner is None:
        result, color = "½–½ draw", "#b0aca8"
    elif (winner == "white") == ecl_white:
        result, color = "1–0 win", "#7fa650"
    else:
        result, color = "0–1 loss", "#c75c5c"

    speed = g.get("speed", "")
    status = g.get("status", "")
    when = datetime.datetime.fromtimestamp(
        g.get("createdAt", 0) / 1000, datetime.timezone.utc
    ).strftime("%b %-d, %Y")

    board_svg = chess.svg.board(
        board, lastmove=last, orientation=orientation, size=BOARD, coordinates=True
    )
    # Strip the XML prolog so the board <svg> can nest inside the outer card.
    board_svg = board_svg.split("?>", 1)[-1].lstrip() if "?>" in board_svg else board_svg

    opp_label = f"{_esc(opp)}" + (f" ({opp_rating})" if opp_rating else "")
    sub = _esc(f"{speed} · by {status} · {when}")

    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
        f'font-family="Segoe UI, Helvetica, Arial, sans-serif">\n'
        f'  <rect width="{W}" height="{H}" rx="10" fill="{BG}"/>\n'
        f'  <g transform="translate({PAD},{PAD})">{board_svg}</g>\n'
        f'  <text x="{PAD}" y="{PAD + BOARD + 22}" fill="{FG}" font-size="15" '
        f'font-weight="600">vs {opp_label}</text>\n'
        f'  <text x="{W - PAD}" y="{PAD + BOARD + 22}" fill="{color}" '
        f'font-size="15" font-weight="700" text-anchor="end">{result}</text>\n'
        f'  <text x="{PAD}" y="{PAD + BOARD + 40}" fill="{SUB}" '
        f'font-size="12">{sub}</text>\n'
        f"</svg>\n"
    )


def main() -> None:
    g = latest_game()
    svg = build_svg(g)
    # Validate it parses as XML before writing (a malformed card breaks the README).
    minidom.parseString(svg)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(svg)
    print(f"wrote {OUT} ({len(svg)} bytes) — game {g.get('id')}", file=sys.stderr)


if __name__ == "__main__":
    main()
