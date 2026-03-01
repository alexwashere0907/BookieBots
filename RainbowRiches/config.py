import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


def _optional_float(key: str) -> Optional[float]:
    """Return float if the env var is set and non-empty, else None."""
    val = os.getenv(key, "").strip()
    return float(val) if val else None


# Stake & price rules
@dataclass
class BookieConfig:
    target_profit: float       = float(os.getenv("RR_TARGET_PROFIT",   "15.00"))
    min_value_pct: float       = float(os.getenv("RR_MIN_VALUE_PCT",   "104.0"))
    max_odds: Optional[float]  = _optional_float("RR_MAX_ODDS")           # None = no cap
    odds_tolerance_abs: float  = float(os.getenv("RR_ODDS_TOL_ABS",   "0.02"))
    odds_tolerance_pct: float  = float(os.getenv("RR_ODDS_TOL_PCT",   "0.0"))


CFG = BookieConfig()

