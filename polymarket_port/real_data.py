from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

from .pmamm_math import EPS

GAMMA_API = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"


@dataclass
class MarketRef:
    market_id: str
    slug: str
    question: str
    condition_id: str
    outcome_a_name: str
    outcome_b_name: str
    token_a: str
    token_b: str
    end_ts: Optional[int]


class RealDataError(RuntimeError):
    pass


def _get(url: str, params: Optional[dict] = None, timeout: int = 30):
    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _parse_list_field(value) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass
    return []


def _parse_ts(ts: Optional[str]) -> Optional[int]:
    if not ts:
        return None
    try:
        if ts.endswith("Z"):
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        else:
            dt = datetime.fromisoformat(ts)
        return int(dt.replace(tzinfo=timezone.utc).timestamp()) if dt.tzinfo is None else int(dt.timestamp())
    except Exception:
        return None


def get_market_by_slug(slug: str) -> MarketRef:
    rows = _get(f"{GAMMA_API}/markets", {"slug": slug, "limit": 10})
    if not rows:
        raise RealDataError(f"No market found for slug={slug}")

    # Prefer exact slug if present.
    row = next((r for r in rows if str(r.get("slug", "")) == slug), rows[0])

    outcomes = _parse_list_field(row.get("outcomes"))
    token_ids = _parse_list_field(row.get("clobTokenIds"))

    if len(outcomes) < 2 or len(token_ids) < 2:
        raise RealDataError(
            f"Market {slug} does not look binary or missing token ids (outcomes={outcomes}, token_ids={token_ids})"
        )

    return MarketRef(
        market_id=str(row.get("id", "")),
        slug=str(row.get("slug", slug)),
        question=str(row.get("question", "")),
        condition_id=str(row.get("conditionId", "")),
        outcome_a_name=outcomes[0],
        outcome_b_name=outcomes[1],
        token_a=token_ids[0],
        token_b=token_ids[1],
        end_ts=_parse_ts(row.get("endDate") or row.get("endDateIso")),
    )


def fetch_recent_global_trades(limit: int = 500, max_offset: int = 3000) -> List[dict]:
    rows_all: List[dict] = []
    offset = 0
    while offset <= max_offset:
        rows = _get(f"{DATA_API}/trades", {"limit": limit, "offset": offset})
        if not rows:
            break
        rows_all.extend(rows)
        if len(rows) < limit:
            break
        offset += limit
    return rows_all


def discover_hot_slug(
    trades: Sequence[dict],
    query: Optional[str] = None,
    min_trades: int = 20,
) -> Tuple[str, Dict[str, int]]:
    counts: Dict[str, int] = {}
    q = (query or "").strip().lower()

    for t in trades:
        slug = str(t.get("slug") or "").strip()
        if not slug:
            continue
        if q and q not in slug.lower() and q not in str(t.get("title", "")).lower():
            continue
        counts[slug] = counts.get(slug, 0) + 1

    if not counts:
        hint = f" matching query='{query}'" if query else ""
        raise RealDataError(f"No candidate market slug found in recent trades{hint}")

    best_slug = max(counts.items(), key=lambda kv: kv[1])[0]
    if counts[best_slug] < min_trades:
        raise RealDataError(
            f"Best candidate slug '{best_slug}' has only {counts[best_slug]} trades (<{min_trades})"
        )
    return best_slug, counts


def estimate_live_spread_cents(token_id: str) -> float:
    try:
        book = _get(f"{CLOB_API}/book", {"token_id": token_id})
    except Exception:
        return 1.0

    bids = book.get("bids") or []
    asks = book.get("asks") or []

    if not bids or not asks:
        return 1.0

    try:
        best_bid = float(bids[0]["price"])
        best_ask = float(asks[0]["price"])
    except Exception:
        return 1.0

    spread = max(best_ask - best_bid, 0.0)
    spread_cents = spread * 100.0
    # Keep realistic bounds for paper execution proxy.
    return float(np.clip(spread_cents, 0.2, 8.0))


def _clip_prob(x: float) -> float:
    return float(np.clip(float(x), 0.001, 0.999))


def _fetch_prices_history(token_id: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    rows = _get(
        f"{CLOB_API}/prices-history",
        {"market": token_id, "startTs": int(start_ts), "endTs": int(end_ts)},
    ).get("history", [])
    if not rows:
        return pd.DataFrame(columns=["ts", "price"])

    df = pd.DataFrame(rows)
    if "t" not in df.columns or "p" not in df.columns:
        return pd.DataFrame(columns=["ts", "price"])

    out = pd.DataFrame(
        {
            "ts": pd.to_numeric(df["t"], errors="coerce"),
            "price": pd.to_numeric(df["p"], errors="coerce"),
        }
    ).dropna()
    out["price"] = out["price"].clip(0.001, 0.999)
    out = out.sort_values("ts").reset_index(drop=True)
    return out


def _build_points_from_price_histories(
    market: MarketRef,
    start_ts: int,
    end_ts: int,
) -> pd.DataFrame:
    a = _fetch_prices_history(market.token_a, start_ts=start_ts, end_ts=end_ts)
    b = _fetch_prices_history(market.token_b, start_ts=start_ts, end_ts=end_ts)

    if a.empty and b.empty:
        return pd.DataFrame(columns=["ts", "yes_mid", "price_a_last", "price_b_last"])

    if a.empty:
        pts = b.rename(columns={"price": "price_b_last"}).copy()
        pts["price_a_last"] = np.nan
    elif b.empty:
        pts = a.rename(columns={"price": "price_a_last"}).copy()
        pts["price_b_last"] = np.nan
    else:
        a2 = a.rename(columns={"price": "price_a_last"})
        b2 = b.rename(columns={"price": "price_b_last"})
        pts = pd.merge_asof(
            a2.sort_values("ts"),
            b2.sort_values("ts"),
            on="ts",
            direction="nearest",
            tolerance=60,
        )

    pts["yes_mid"] = np.nan
    if "price_a_last" in pts.columns:
        pts.loc[pts["price_a_last"].notna(), "yes_mid"] = pts.loc[
            pts["price_a_last"].notna(), "price_a_last"
        ]
    if "price_b_last" in pts.columns:
        b_comp = 1.0 - pts["price_b_last"]
        if "yes_mid" not in pts.columns:
            pts["yes_mid"] = b_comp
        else:
            pts["yes_mid"] = np.where(
                pts["yes_mid"].notna(),
                (pts["yes_mid"] + b_comp) / 2.0,
                b_comp,
            )

    pts = pts.dropna(subset=["yes_mid"]).copy()
    pts["yes_mid"] = pts["yes_mid"].clip(0.001, 0.999)
    return pts[["ts", "yes_mid", "price_a_last", "price_b_last"]].sort_values("ts")


def build_market_trade_tape(
    market: MarketRef,
    all_trades: Sequence[dict],
    bucket_seconds: int = 60,
    alpha_lookback_buckets: int = 3,
    spread_cents_assumed: Optional[float] = None,
    min_buckets: int = 4,
    fallback_history_hours: int = 24,
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """Build model tape from real Polymarket trade prints.

    Returns:
      tape_df: bucketed tape with yes_mid/bid/ask + alpha + next returns
      raw_df:  raw filtered trade rows for auditability
      spread_cents_used: spread assumption used to derive bid/ask from mids
    """
    if bucket_seconds <= 0:
        raise ValueError("bucket_seconds must be > 0")
    if min_buckets < 3:
        raise ValueError("min_buckets must be >= 3")

    token_set = {market.token_a, market.token_b}
    filt = [t for t in all_trades if str(t.get("asset")) in token_set and str(t.get("slug", "")) == market.slug]

    if not filt:
        # Fallback: some APIs are inconsistent on slug filtering; keep token match only.
        filt = [t for t in all_trades if str(t.get("asset")) in token_set]

    if not filt:
        raise RealDataError(
            f"No recent trades found for market={market.slug} token_a={market.token_a} token_b={market.token_b}. "
            "Try another slug with more recent activity."
        )

    raw_df = pd.DataFrame(filt)
    raw_df["timestamp"] = pd.to_numeric(raw_df.get("timestamp"), errors="coerce")
    raw_df["price"] = pd.to_numeric(raw_df.get("price"), errors="coerce")
    raw_df = raw_df.dropna(subset=["timestamp", "price"]).copy()
    raw_df = raw_df.sort_values("timestamp")

    last_price: Dict[str, float] = {}
    points: List[Dict[str, float]] = []

    for _, r in raw_df.iterrows():
        asset = str(r.get("asset"))
        px = _clip_prob(float(r.get("price")))
        ts = int(float(r.get("timestamp")))
        last_price[asset] = px

        pa = last_price.get(market.token_a)
        pb = last_price.get(market.token_b)

        yes_candidates = []
        if pa is not None:
            yes_candidates.append(pa)
        if pb is not None:
            yes_candidates.append(1.0 - pb)

        if not yes_candidates:
            continue

        yes_mid = _clip_prob(float(np.mean(yes_candidates)))
        points.append(
            {
                "ts": ts,
                "yes_mid": yes_mid,
                "price_a_last": pa if pa is not None else np.nan,
                "price_b_last": pb if pb is not None else np.nan,
            }
        )

    pts = pd.DataFrame(points)

    # If recent global tape is too short for this market, augment with CLOB prices-history.
    if pts.empty or len(pts) < min_buckets:
        now_ts = int(datetime.now(timezone.utc).timestamp())
        end_ts = int(raw_df["timestamp"].max()) if not raw_df.empty else now_ts
        start_ts = end_ts - max(int(fallback_history_hours), 1) * 3600

        hist_pts = _build_points_from_price_histories(market, start_ts=start_ts, end_ts=end_ts)
        if not hist_pts.empty:
            if pts.empty:
                pts = hist_pts
            else:
                pts = pd.concat([pts, hist_pts], ignore_index=True).sort_values("ts")
                pts = pts.drop_duplicates(subset=["ts"], keep="last")

    if pts.empty:
        raise RealDataError("Could not construct midpoint points from trade tape or prices-history")

    pts["bucket_ts"] = (pd.to_numeric(pts["ts"], errors="coerce") // bucket_seconds) * bucket_seconds
    pts = pts.dropna(subset=["bucket_ts", "yes_mid"]).copy()
    tape = pts.groupby("bucket_ts", as_index=False).last().sort_values("bucket_ts")

    if len(tape) < min_buckets:
        raise RealDataError(
            f"Too few buckets ({len(tape)}) for market={market.slug}; need >={min_buckets}. "
            "Try smaller bucket_seconds or a more active slug."
        )

    tape["ret_bps"] = (tape["yes_mid"].pct_change().fillna(0.0)) * 10_000.0
    lookback = max(int(alpha_lookback_buckets), 1)
    tape["alpha_bps"] = (
        (tape["yes_mid"] - tape["yes_mid"].shift(lookback))
        / tape["yes_mid"].shift(lookback).clip(lower=EPS)
        * 10_000.0
    ).fillna(0.0)
    tape["next_ret_bps"] = (
        (tape["yes_mid"].shift(-1) - tape["yes_mid"]) / tape["yes_mid"].clip(lower=EPS) * 10_000.0
    ).fillna(0.0)

    spread_cents = (
        float(spread_cents_assumed)
        if spread_cents_assumed is not None
        else estimate_live_spread_cents(market.token_a)
    )
    half = spread_cents / 200.0
    tape["yes_bid"] = (tape["yes_mid"] - half).clip(0.001, 0.999)
    tape["yes_ask"] = (tape["yes_mid"] + half).clip(0.001, 0.999)

    if market.end_ts is not None:
        tape["tte_days"] = ((market.end_ts - tape["bucket_ts"]) / 86400.0).clip(lower=0.001)
    else:
        # Fallback pseudo time-to-expiry if market metadata lacks end date.
        steps_left = np.arange(len(tape), 0, -1)
        tape["tte_days"] = np.maximum(steps_left * (bucket_seconds / 86400.0), 0.001)

    tape = tape.reset_index(drop=True)
    tape["step"] = np.arange(len(tape))

    cols = [
        "step",
        "bucket_ts",
        "tte_days",
        "yes_mid",
        "yes_bid",
        "yes_ask",
        "alpha_bps",
        "next_ret_bps",
        "ret_bps",
        "price_a_last",
        "price_b_last",
    ]
    tape = tape[cols]

    return tape, raw_df, spread_cents
