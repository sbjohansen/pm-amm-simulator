#!/usr/bin/env python3
"""Run iterative pm-AMM -> Polymarket portability simulations.

Example:
    python scripts/run_polymarket_port_iterations.py \
        --iterations 8 \
        --trials-per-iteration 200 \
        --out-dir outputs/polymarket_port
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Allow running this script directly from repo root without installing package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from polymarket_port.clob_simulator import (
    SimulationParams,
    StrategyParams,
    run_parameter_iterations,
)


def build_strategy_grid(iterations: int, seed: int) -> list[StrategyParams]:
    rng = np.random.default_rng(seed)
    grid: list[StrategyParams] = []

    for _ in range(iterations):
        grid.append(
            StrategyParams(
                L=float(rng.uniform(120.0, 550.0)),
                dynamic=bool(rng.random() > 0.15),
                market_duration_days=float(rng.choice([3.0, 7.0, 14.0])),
                initial_price_yes=float(rng.uniform(0.35, 0.65)),
                edge_threshold_bps=float(rng.uniform(15.0, 75.0)),
                alpha_weight=float(rng.uniform(0.6, 1.6)),
                inventory_skew_bps=float(rng.uniform(5.0, 60.0)),
                max_position_per_side=float(rng.choice([50.0, 100.0, 150.0, 200.0, 250.0])),
                trade_size=float(rng.choice([5.0, 10.0, 15.0, 20.0])),
            )
        )

    return grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterative pm-AMM Polymarket-port simulation")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--trials-per-iteration", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--step-minutes", type=int, default=5)
    parser.add_argument("--sigma-mid-bps", type=float, default=25.0)
    parser.add_argument("--spread-mean-cents", type=float, default=1.0)
    parser.add_argument("--spread-std-cents", type=float, default=0.2)
    parser.add_argument("--alpha-noise-bps", type=float, default=15.0)
    parser.add_argument("--alpha-signal-beta", type=float, default=0.8)
    parser.add_argument("--out-dir", default="outputs/polymarket_port")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sim = SimulationParams(
        n_steps=args.steps,
        step_minutes=args.step_minutes,
        sigma_mid_bps=args.sigma_mid_bps,
        spread_mean_cents=args.spread_mean_cents,
        spread_std_cents=args.spread_std_cents,
        alpha_noise_bps=args.alpha_noise_bps,
        alpha_signal_beta=args.alpha_signal_beta,
    )

    strategy_grid = build_strategy_grid(args.iterations, args.seed)
    summary_df, trials_df = run_parameter_iterations(
        strategy_grid=strategy_grid,
        sim=sim,
        trials_per_iteration=args.trials_per_iteration,
        seed=args.seed,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    summary_path = os.path.join(args.out_dir, f"iteration_summary_{ts}.csv")
    trials_path = os.path.join(args.out_dir, f"iteration_trials_{ts}.csv")
    best_path = os.path.join(args.out_dir, f"best_iteration_{ts}.json")

    summary_df.to_csv(summary_path, index=False)
    trials_df.to_csv(trials_path, index=False)

    best = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
    payload = {
        "generated_at": ts,
        "simulation": asdict(sim),
        "iterations": args.iterations,
        "trials_per_iteration": args.trials_per_iteration,
        "best": best,
    }
    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("Saved:")
    print(f"  - {summary_path}")
    print(f"  - {trials_path}")
    print(f"  - {best_path}")

    print("\nTop 5 iterations:")
    if not summary_df.empty:
        cols = [
            "iteration",
            "mean_pnl_settle",
            "win_rate_settle",
            "mean_n_trades",
            "L",
            "edge_threshold_bps",
            "alpha_weight",
            "inventory_skew_bps",
            "trade_size",
            "max_position_per_side",
            "dynamic",
        ]
        print(summary_df[cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
