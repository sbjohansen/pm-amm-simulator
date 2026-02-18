from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .pmamm_math import (
    EPS,
    apply_buy_no,
    apply_buy_yes,
    price_yes_from_reserves,
    reserves_from_price,
)


@dataclass
class StrategyParams:
    """Parameters for pm-AMM-guided Polymarket buy-only routing simulation."""

    L: float = 300.0
    dynamic: bool = True
    market_duration_days: float = 14.0
    initial_price_yes: float = 0.50

    # Entry gating
    edge_threshold_bps: float = 35.0
    alpha_weight: float = 1.0  # multiply external alpha_bps before applying

    # Inventory handling
    inventory_skew_bps: float = 20.0
    max_position_per_side: float = 200.0
    trade_size: float = 10.0


@dataclass
class SimulationParams:
    """Synthetic tape generation parameters."""

    n_steps: int = 180
    step_minutes: int = 5
    sigma_mid_bps: float = 25.0
    spread_mean_cents: float = 1.0
    spread_std_cents: float = 0.2
    alpha_noise_bps: float = 15.0
    alpha_signal_beta: float = 0.8  # alpha_t ~= beta * next_ret_bps + noise


def _clip_prob(x: float) -> float:
    return float(np.clip(x, 0.001, 0.999))


def _quotes_from_mid(mid_yes: float, spread_cents: float) -> Tuple[float, float]:
    half = max(spread_cents, 0.1) / 200.0
    bid = _clip_prob(mid_yes - half)
    ask = _clip_prob(mid_yes + half)
    if bid >= ask:
        ask = _clip_prob(bid + 0.001)
    return bid, ask


def _yes_quotes_to_no_quotes(yes_bid: float, yes_ask: float) -> Tuple[float, float]:
    # Binary complement mapping with simple consistency bounds
    no_bid = _clip_prob(1.0 - yes_ask)
    no_ask = _clip_prob(1.0 - yes_bid)
    if no_bid >= no_ask:
        no_ask = _clip_prob(no_bid + 0.001)
    return no_bid, no_ask


def generate_synthetic_path(
    sim: SimulationParams,
    initial_price_yes: float,
    market_duration_days: float,
    seed: int,
) -> pd.DataFrame:
    """Generate a synthetic Polymarket-like bid/ask + alpha tape."""
    rng = np.random.default_rng(seed)

    # Next-step returns in bps drive the latent process
    ret_bps = rng.normal(0.0, sim.sigma_mid_bps, size=sim.n_steps + 1)

    mids = [_clip_prob(initial_price_yes)]
    for i in range(sim.n_steps):
        mids.append(_clip_prob(mids[-1] + ret_bps[i] / 10_000.0))

    rows: List[Dict[str, float]] = []
    elapsed_days_per_step = sim.step_minutes / (24.0 * 60.0)

    for i in range(sim.n_steps):
        mid = mids[i]
        spread_cents = max(0.1, rng.normal(sim.spread_mean_cents, sim.spread_std_cents))
        yes_bid, yes_ask = _quotes_from_mid(mid, spread_cents)

        next_ret_bps = ret_bps[i + 1]
        alpha_bps = (
            sim.alpha_signal_beta * next_ret_bps
            + rng.normal(0.0, sim.alpha_noise_bps)
        )

        tte_days = max(0.001, market_duration_days - i * elapsed_days_per_step)

        rows.append(
            {
                "step": i,
                "tte_days": tte_days,
                "yes_mid": mid,
                "yes_bid": yes_bid,
                "yes_ask": yes_ask,
                "alpha_bps": alpha_bps,
                "next_ret_bps": next_ret_bps,
            }
        )

    return pd.DataFrame(rows)


def simulate_single_run(
    tape: pd.DataFrame,
    params: StrategyParams,
    outcome_yes: int,
    return_trades: bool = False,
) -> Tuple[Dict[str, float], Optional[pd.DataFrame]]:
    """Run one buy-only YES/NO simulation on synthetic or real tape.

    Trading rule:
    - Compute pm-AMM fair YES from virtual reserves.
    - Shift fair by alpha and inventory skew.
    - BUY YES if ask_yes is sufficiently below shifted fair.
    - BUY NO  if ask_no is sufficiently below shifted fair_no.
    - If both fire, take larger edge.
    """
    if tape.empty:
        raise ValueError("tape is empty")

    first_mid = float(tape.iloc[0]["yes_mid"])
    initial_tte = float(tape.iloc[0]["tte_days"])
    x, y = reserves_from_price(first_mid, params.L, initial_tte, params.dynamic)

    pos_yes = 0.0
    pos_no = 0.0
    cash_spent = 0.0

    trades: List[Dict[str, float]] = []

    for _, row in tape.iterrows():
        tte = float(row["tte_days"])
        yes_bid = float(row["yes_bid"])
        yes_ask = float(row["yes_ask"])
        alpha_bps = float(row["alpha_bps"])

        no_bid, no_ask = _yes_quotes_to_no_quotes(yes_bid, yes_ask)

        fair_yes = price_yes_from_reserves(x, y, params.L, tte, params.dynamic)

        inv_imbalance = (pos_yes - pos_no) / max(params.max_position_per_side, EPS)
        inv_penalty_bps = params.inventory_skew_bps * inv_imbalance

        fair_yes_shifted = _clip_prob(
            fair_yes + (params.alpha_weight * alpha_bps - inv_penalty_bps) / 10_000.0
        )
        fair_no_shifted = 1.0 - fair_yes_shifted

        edge_yes_bps = (fair_yes_shifted - yes_ask) * 10_000.0
        edge_no_bps = (fair_no_shifted - no_ask) * 10_000.0

        do_yes = edge_yes_bps >= params.edge_threshold_bps and pos_yes < params.max_position_per_side
        do_no = edge_no_bps >= params.edge_threshold_bps and pos_no < params.max_position_per_side

        side = None
        if do_yes and do_no:
            side = "YES" if edge_yes_bps >= edge_no_bps else "NO"
        elif do_yes:
            side = "YES"
        elif do_no:
            side = "NO"

        if side is None:
            continue

        if side == "YES":
            size = min(params.trade_size, params.max_position_per_side - pos_yes)
            if size <= 0:
                continue
            x, y, model_avg, fair_after = apply_buy_yes(x, y, size, params.L, tte, params.dynamic)
            fill_px = yes_ask
            pos_yes += size
        else:
            size = min(params.trade_size, params.max_position_per_side - pos_no)
            if size <= 0:
                continue
            x, y, model_avg, fair_after = apply_buy_no(x, y, size, params.L, tte, params.dynamic)
            fill_px = no_ask
            pos_no += size

        cash_spent += fill_px * size

        if return_trades:
            trades.append(
                {
                    "step": float(row["step"]),
                    "side": side,
                    "size": size,
                    "fill_px": fill_px,
                    "model_avg_px": model_avg,
                    "fair_before_yes": fair_yes,
                    "fair_after_yes": fair_after,
                    "edge_yes_bps": edge_yes_bps,
                    "edge_no_bps": edge_no_bps,
                    "alpha_bps": alpha_bps,
                    "pos_yes": pos_yes,
                    "pos_no": pos_no,
                    "cash_spent": cash_spent,
                }
            )

    final_mid = float(tape.iloc[-1]["yes_mid"])
    mark_value = pos_yes * final_mid + pos_no * (1.0 - final_mid)
    payout = pos_yes if int(outcome_yes) == 1 else pos_no

    pnl_mark = mark_value - cash_spent
    pnl_settle = payout - cash_spent
    total_trades = len(trades) if return_trades else int((pos_yes + pos_no) / max(params.trade_size, 1e-9))

    metrics = {
        "outcome_yes": int(outcome_yes),
        "cash_spent": cash_spent,
        "pos_yes": pos_yes,
        "pos_no": pos_no,
        "final_mid_yes": final_mid,
        "mark_value": mark_value,
        "settlement_value": payout,
        "pnl_mark": pnl_mark,
        "pnl_settle": pnl_settle,
        "n_trades": float(total_trades),
        "edge_threshold_bps": params.edge_threshold_bps,
        "L": params.L,
        "inventory_skew_bps": params.inventory_skew_bps,
        "alpha_weight": params.alpha_weight,
        "trade_size": params.trade_size,
        "max_position_per_side": params.max_position_per_side,
        "dynamic": float(1 if params.dynamic else 0),
    }

    trade_df = pd.DataFrame(trades) if return_trades else None
    return metrics, trade_df


def run_parameter_iterations(
    strategy_grid: Sequence[StrategyParams],
    sim: SimulationParams,
    trials_per_iteration: int,
    seed: int = 7,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run iterative simulation over parameter candidates.

    Returns:
        summary_df: one row per iteration candidate
        trials_df: per-trial details for deeper inspection
    """
    if trials_per_iteration <= 0:
        raise ValueError("trials_per_iteration must be > 0")

    summary_rows: List[Dict[str, float]] = []
    trial_rows: List[Dict[str, float]] = []
    rng = np.random.default_rng(seed)

    for i, params in enumerate(strategy_grid, start=1):
        pnls: List[float] = []
        marks: List[float] = []
        trades: List[float] = []

        for t in range(trials_per_iteration):
            run_seed = int(seed + i * 100_000 + t)
            tape = generate_synthetic_path(
                sim,
                initial_price_yes=params.initial_price_yes,
                market_duration_days=params.market_duration_days,
                seed=run_seed,
            )

            final_p = float(tape.iloc[-1]["yes_mid"])
            outcome_yes = int(rng.random() < final_p)

            metrics, _ = simulate_single_run(tape, params, outcome_yes, return_trades=False)
            metrics.update({"iteration": float(i), "trial": float(t + 1)})
            trial_rows.append(metrics)

            pnls.append(float(metrics["pnl_settle"]))
            marks.append(float(metrics["pnl_mark"]))
            trades.append(float(metrics["n_trades"]))

        summary_rows.append(
            {
                "iteration": float(i),
                "trials": float(trials_per_iteration),
                "mean_pnl_settle": float(np.mean(pnls)),
                "std_pnl_settle": float(np.std(pnls)),
                "win_rate_settle": float(np.mean(np.array(pnls) > 0.0)),
                "mean_pnl_mark": float(np.mean(marks)),
                "mean_n_trades": float(np.mean(trades)),
                **asdict(params),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["mean_pnl_settle", "win_rate_settle"], ascending=False
    )
    trials_df = pd.DataFrame(trial_rows)
    return summary_df, trials_df
