#!/usr/bin/env python3
"""Run iterative paper-buy simulations on real Polymarket trade data.

This script:
1) resolves a market (slug or auto-discovery from latest global trades)
2) builds a real-data tape from trade prints (token A/B)
3) runs parameter iterations using the buy-only YES/NO simulator
4) saves CSV/JSON artifacts for analysis

Example:
  .venv/bin/python scripts/run_polymarket_realdata_iterations.py \
      --slug btc-updown-15m-1771409700 \
      --iterations 10 \
      --trials-per-iteration 150 \
      --window-steps 60 \
      --out-dir outputs/polymarket_port_real
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Allow running directly from repo root without package installation.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from polymarket_port.clob_simulator import StrategyParams, simulate_single_run
from polymarket_port.real_data import (
    RealDataError,
    build_market_trade_tape,
    discover_hot_slug,
    fetch_recent_global_trades,
    get_market_by_slug,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Real Polymarket paper-buy iteration runner")
    p.add_argument("--slug", default=None, help="Exact market slug. If omitted, auto-discover from recent trades")
    p.add_argument("--query", default="btc-updown-15m", help="Auto-discovery substring query when --slug omitted")

    p.add_argument("--iterations", type=int, default=10)
    p.add_argument("--trials-per-iteration", type=int, default=120)
    p.add_argument("--window-steps", type=int, default=80, help="Bars per trial window")

    p.add_argument("--bucket-seconds", type=int, default=60)
    p.add_argument("--alpha-lookback-buckets", type=int, default=3)
    p.add_argument("--spread-cents", type=float, default=None, help="Optional fixed spread proxy; default uses live book estimate")
    p.add_argument("--min-buckets", type=int, default=4)
    p.add_argument("--fallback-history-hours", type=int, default=24)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-offset", type=int, default=3000, help="Max data-api offset for global trades crawl")
    p.add_argument("--limit", type=int, default=500)

    p.add_argument("--out-dir", default="outputs/polymarket_port_real")
    return p.parse_args()


def build_strategy_grid(
    iterations: int,
    seed: int,
    initial_price: float,
    market_duration_days: float,
) -> List[StrategyParams]:
    rng = np.random.default_rng(seed)
    grid: List[StrategyParams] = []

    for _ in range(iterations):
        grid.append(
            StrategyParams(
                L=float(rng.uniform(80.0, 450.0)),
                dynamic=bool(rng.random() > 0.20),
                market_duration_days=float(np.clip(market_duration_days, 0.25, 30.0)),
                initial_price_yes=float(np.clip(initial_price, 0.05, 0.95)),
                edge_threshold_bps=float(rng.uniform(10.0, 70.0)),
                alpha_weight=float(rng.uniform(0.5, 1.8)),
                inventory_skew_bps=float(rng.uniform(5.0, 70.0)),
                max_position_per_side=float(rng.choice([40.0, 80.0, 120.0, 160.0, 220.0])),
                trade_size=float(rng.choice([2.0, 5.0, 8.0, 10.0, 15.0])),
            )
        )

    return grid


def _draw_window(tape: pd.DataFrame, window_steps: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(tape) <= window_steps:
        w = tape.copy().reset_index(drop=True)
        w["step"] = np.arange(len(w))
        return w

    start_max = len(tape) - window_steps
    start = int(rng.integers(0, start_max + 1))
    w = tape.iloc[start : start + window_steps].copy().reset_index(drop=True)
    w["step"] = np.arange(len(w))
    return w


def run_iterations_on_real_tape(
    tape: pd.DataFrame,
    strategy_grid: List[StrategyParams],
    trials_per_iteration: int,
    window_steps: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    trial_rows: List[Dict[str, float]] = []
    summary_rows: List[Dict[str, float]] = []

    for i, params in enumerate(strategy_grid, start=1):
        pnls_settle: List[float] = []
        pnls_mark: List[float] = []
        n_trades: List[float] = []

        for trial in range(1, trials_per_iteration + 1):
            window = _draw_window(tape, window_steps=window_steps, rng=rng)
            final_mid = float(window.iloc[-1]["yes_mid"])

            # Simulate uncertain settlement outcome from final implied probability.
            outcome_yes = int(rng.random() < final_mid)

            metrics, _ = simulate_single_run(window, params, outcome_yes=outcome_yes, return_trades=False)
            metrics.update(
                {
                    "iteration": float(i),
                    "trial": float(trial),
                    "window_start_ts": float(window.iloc[0]["bucket_ts"]),
                    "window_end_ts": float(window.iloc[-1]["bucket_ts"]),
                    "window_start_yes_mid": float(window.iloc[0]["yes_mid"]),
                    "window_end_yes_mid": final_mid,
                }
            )

            trial_rows.append(metrics)
            pnls_settle.append(float(metrics["pnl_settle"]))
            pnls_mark.append(float(metrics["pnl_mark"]))
            n_trades.append(float(metrics["n_trades"]))

        summary_rows.append(
            {
                "iteration": float(i),
                "trials": float(trials_per_iteration),
                "mean_pnl_settle": float(np.mean(pnls_settle)),
                "std_pnl_settle": float(np.std(pnls_settle)),
                "win_rate_settle": float(np.mean(np.array(pnls_settle) > 0.0)),
                "mean_pnl_mark": float(np.mean(pnls_mark)),
                "mean_n_trades": float(np.mean(n_trades)),
                **asdict(params),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["mean_pnl_settle", "win_rate_settle", "mean_n_trades"],
        ascending=False,
    )
    trials_df = pd.DataFrame(trial_rows)
    return summary_df, trials_df


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    print("Fetching recent global trades...")
    all_trades = fetch_recent_global_trades(limit=args.limit, max_offset=args.max_offset)
    if not all_trades:
        raise RealDataError("No trades returned from data-api")

    slug = args.slug
    slug_counts: Dict[str, int] = {}
    if not slug:
        slug, slug_counts = discover_hot_slug(all_trades, query=args.query, min_trades=20)
        print(f"Auto-selected slug: {slug} (count={slug_counts.get(slug, 0)})")

    market = get_market_by_slug(slug)
    print(f"Market: {market.question}")
    print(f"Slug: {market.slug}")
    print(f"Outcomes: A={market.outcome_a_name} B={market.outcome_b_name}")

    tape, raw_trades, spread_cents_used = build_market_trade_tape(
        market=market,
        all_trades=all_trades,
        bucket_seconds=args.bucket_seconds,
        alpha_lookback_buckets=args.alpha_lookback_buckets,
        spread_cents_assumed=args.spread_cents,
        min_buckets=args.min_buckets,
        fallback_history_hours=args.fallback_history_hours,
    )

    if tape.empty:
        raise RealDataError("Tape ended up empty")

    initial_price = float(tape.iloc[0]["yes_mid"])
    max_tte = float(np.max(tape["tte_days"]))
    strategy_grid = build_strategy_grid(
        iterations=args.iterations,
        seed=args.seed,
        initial_price=initial_price,
        market_duration_days=max_tte,
    )

    summary_df, trials_df = run_iterations_on_real_tape(
        tape=tape,
        strategy_grid=strategy_grid,
        trials_per_iteration=args.trials_per_iteration,
        window_steps=args.window_steps,
        seed=args.seed,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    tape_path = os.path.join(args.out_dir, f"real_tape_{market.slug}_{ts}.csv")
    raw_path = os.path.join(args.out_dir, f"raw_trades_{market.slug}_{ts}.csv")
    summary_path = os.path.join(args.out_dir, f"iteration_summary_{market.slug}_{ts}.csv")
    trials_path = os.path.join(args.out_dir, f"iteration_trials_{market.slug}_{ts}.csv")
    best_path = os.path.join(args.out_dir, f"best_iteration_{market.slug}_{ts}.json")

    tape.to_csv(tape_path, index=False)
    raw_trades.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    trials_df.to_csv(trials_path, index=False)

    best = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
    metadata = {
        "generated_at": ts,
        "market": asdict(market),
        "settings": {
            "iterations": args.iterations,
            "trials_per_iteration": args.trials_per_iteration,
            "window_steps": args.window_steps,
            "bucket_seconds": args.bucket_seconds,
            "alpha_lookback_buckets": args.alpha_lookback_buckets,
            "spread_cents_used": spread_cents_used,
            "min_buckets": args.min_buckets,
            "fallback_history_hours": args.fallback_history_hours,
            "seed": args.seed,
            "max_offset": args.max_offset,
            "limit": args.limit,
        },
        "tape_stats": {
            "rows": int(len(tape)),
            "raw_trades_rows": int(len(raw_trades)),
            "yes_mid_min": float(tape["yes_mid"].min()),
            "yes_mid_max": float(tape["yes_mid"].max()),
            "alpha_bps_std": float(tape["alpha_bps"].std()),
        },
        "best": best,
        "top5": summary_df.head(5).to_dict(orient="records"),
    }

    with open(best_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Saved:")
    print(f"  - {tape_path}")
    print(f"  - {raw_path}")
    print(f"  - {summary_path}")
    print(f"  - {trials_path}")
    print(f"  - {best_path}")

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
        print("\nTop 5 iterations:")
        print(summary_df[cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
