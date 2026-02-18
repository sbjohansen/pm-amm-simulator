"""Polymarket-portable pm-AMM utilities.

This package adds an off-chain implementation layer intended for CLOB execution
experiments (e.g. Polymarket). It keeps pm-AMM state as a *virtual pricing
model* and simulates buy-only YES/NO routing against external bid/ask quotes.
"""

from .pmamm_math import (
    phi,
    Phi,
    Phi_inv,
    effective_liquidity,
    price_yes_from_reserves,
    reserves_from_price,
    apply_buy_no,
    apply_buy_yes,
)
from .clob_simulator import (
    StrategyParams,
    SimulationParams,
    generate_synthetic_path,
    simulate_single_run,
    run_parameter_iterations,
)

__all__ = [
    "phi",
    "Phi",
    "Phi_inv",
    "effective_liquidity",
    "price_yes_from_reserves",
    "reserves_from_price",
    "apply_buy_no",
    "apply_buy_yes",
    "StrategyParams",
    "SimulationParams",
    "generate_synthetic_path",
    "simulate_single_run",
    "run_parameter_iterations",
]
