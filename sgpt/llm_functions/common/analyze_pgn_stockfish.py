"""
Function: analyze_pgn_stockfish
--------------------------------
Analyse a PGN string at an optional ply using Stockfish and return eval & PV.

Dependencies:
    pip install chess stockfish

Set env var STOCKFISH_BIN if the `stockfish` executable is not in $PATH.
"""

from __future__ import annotations

import io
import os
import shutil
import textwrap
from typing import Optional

from instructor import OpenAISchema
from pydantic import Field


class Function(OpenAISchema):
    """Analyse a chess move using PGN format with Stockfish chess engine and return evaluation & PV."""

    pgn: str = Field(..., description="Full PGN text")
    ply: Optional[int] = Field(
        default=None,
        ge=0,
        description="Half-move number to analyse (0 for final position).",
    )
    depth: int = Field(
        default=14,
        ge=1,
        le=99,
        description="Search depth used when calling Stockfish.",
    )

    class Config:
        title = "analyze_pgn_stockfish"

    @classmethod
    def execute(cls, *, pgn: str, ply: Optional[int] = None, depth: int = 18) -> str:  # type: ignore[override]
        try:
            import chess
            import chess.pgn
            import chess.engine
        except ModuleNotFoundError:
            return "❌ python-chess missing – install with `pip install chess`."

        # Parse first game
        game = chess.pgn.read_game(io.StringIO(pgn))
        if game is None:
            return "❌ Could not parse PGN (no games found)."

        node = game
        if ply and ply > 0:
            for _ in range(ply):
                if node.is_end():
                    break
                node = node.variation(0)

        board = node.board()

        engine_path = os.environ.get("STOCKFISH_BIN", "stockfish")
        if not shutil.which(engine_path):
            return f"❌ Stockfish binary not found (looked for '{engine_path}')."

        try:
            with chess.engine.SimpleEngine.popen_uci(engine_path) as eng:
                info = eng.analyse(board, chess.engine.Limit(depth=depth))
        except Exception as exc:
            return f"❌ Engine error: {exc}"

        score = info.get("score")
        if score:
            pov = score.pov(board.turn)
            if pov.is_mate():
                eval_str = f"Mate in {pov.mate()}"
            else:
                eval_str = f"{pov.score() / 100:.2f} cp"
        else:
            eval_str = "?"

        pv = board.variation_san(info.get("pv", []))

        return textwrap.dedent(
            f"""
            🧠 Stockfish analysis
            Depth : {depth}
            Ply   : {ply if ply is not None else 'final'}
            Eval  : {eval_str}
            PV    : {pv}
            FEN   : {board.fen()}
            """
        ).strip()
