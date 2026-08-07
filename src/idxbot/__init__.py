"""idxbot - IDX broker-flow accumulation engine.

Reconstructs what each exchange member is doing in a stock from broker summary,
segments that flow into campaigns, profiles how each desk enters and takes
profit, screens a universe for accumulation, and turns a signal into an
executable trading plan with IDX-correct tick, lot and auto-rejection handling.
"""

__version__ = "0.1.0"

from .config import Config, load_config  # noqa: F401
from .engine import Engine, TickerAnalysis  # noqa: F401

__all__ = ["Config", "Engine", "TickerAnalysis", "load_config", "__version__"]
