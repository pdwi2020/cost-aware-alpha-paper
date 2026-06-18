"""Universe-suffix path helper for re-instantiating the CAVAL pipeline on a
different market segment (V13: Russell 2000).

Set the environment variable ``CAVAL_UNIVERSE=r2000`` to make every stage
script read the ``*_r2000`` inputs (``features_all_r2000.parquet``,
``daily_ohlcv_r2000.parquet``) and write ``*_r2000`` outputs.  When the
variable is unset the suffix is empty, so the original S&P-500 behaviour is
exactly preserved (no functional change to the published pipeline).
"""

import os
from pathlib import Path


def universe_suffix() -> str:
    """Return e.g. ``"_r2000"`` when CAVAL_UNIVERSE=r2000, else ``""``."""
    u = os.environ.get("CAVAL_UNIVERSE", "").strip()
    return f"_{u}" if u else ""


def proc(root: Path, filename: str) -> Path:
    """``data/processed/<filename>`` with the universe suffix before the ext."""
    base = Path(root) / "data" / "processed"
    suf = universe_suffix()
    if not suf:
        return base / filename
    dot = filename.rfind(".")
    return base / f"{filename[:dot]}{suf}{filename[dot:]}"
