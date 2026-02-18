from __future__ import annotations

import math
from typing import Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

EPS = 1e-10


def phi(z: float) -> float:
    """Standard normal PDF."""
    return float(norm.pdf(z))


def Phi(z: float) -> float:
    """Standard normal CDF."""
    return float(norm.cdf(z))


def Phi_inv(p: float) -> float:
    """Inverse normal CDF with safe clamping."""
    p = float(np.clip(p, EPS, 1.0 - EPS))
    return float(norm.ppf(p))


def effective_liquidity(L: float, tte_days: float, dynamic: bool) -> float:
    """Effective liquidity (L_eff) for static or dynamic pm-AMM."""
    if dynamic:
        return float(L) * math.sqrt(max(float(tte_days), EPS))
    return float(L)


def invariant_value(x: float, y: float, L: float, tte_days: float, dynamic: bool) -> float:
    """Invariant value f(x, y) that should be ~0 on-curve."""
    l_eff = effective_liquidity(L, tte_days, dynamic)
    z = (y - x) / max(l_eff, EPS)
    return (y - x) * Phi(z) + l_eff * phi(z) - y


def price_yes_from_reserves(x: float, y: float, L: float, tte_days: float, dynamic: bool) -> float:
    """YES fair probability from virtual reserves."""
    l_eff = effective_liquidity(L, tte_days, dynamic)
    if l_eff <= 0:
        return 0.5
    z = (y - x) / l_eff
    return float(np.clip(Phi(z), EPS, 1.0 - EPS))


def reserves_from_price(price_yes: float, L: float, tte_days: float, dynamic: bool) -> Tuple[float, float]:
    """Closed-form reserve pair from price + liquidity for pm-AMM."""
    l_eff = effective_liquidity(L, tte_days, dynamic)
    z = Phi_inv(price_yes)
    diff = l_eff * z
    y = diff * price_yes + l_eff * phi(z)
    x = y - diff
    return float(x), float(y)


def _solve_bracket(fn, lo: float, hi: float, fallback: float) -> float:
    if lo >= hi:
        return fallback
    try:
        return float(brentq(fn, lo, hi))
    except Exception:
        return fallback


def solve_y_given_x(
    x_new: float,
    L: float,
    tte_days: float,
    dynamic: bool,
    y_hint: float,
) -> float:
    """Solve y for fixed x on pm-AMM invariant."""
    l_eff = effective_liquidity(L, tte_days, dynamic)

    def eq(yv: float) -> float:
        return invariant_value(x_new, yv, L, tte_days, dynamic)

    candidates = [10.0, 20.0, 50.0, 100.0]
    fallback = max(y_hint, EPS)
    for width in candidates:
        lo = max(EPS, x_new - width * l_eff)
        hi = x_new + width * l_eff
        try:
            f_lo, f_hi = eq(lo), eq(hi)
            if np.sign(f_lo) == np.sign(f_hi):
                continue
            return float(brentq(eq, lo, hi))
        except Exception:
            continue
    return _solve_bracket(eq, max(EPS, x_new - 100.0 * l_eff), x_new + 100.0 * l_eff, fallback)


def solve_x_given_y(
    y_new: float,
    L: float,
    tte_days: float,
    dynamic: bool,
    x_hint: float,
) -> float:
    """Solve x for fixed y on pm-AMM invariant."""
    l_eff = effective_liquidity(L, tte_days, dynamic)

    def eq(xv: float) -> float:
        return invariant_value(xv, y_new, L, tte_days, dynamic)

    candidates = [10.0, 20.0, 50.0, 100.0]
    fallback = max(x_hint, EPS)
    for width in candidates:
        lo = max(EPS, y_new - width * l_eff)
        hi = y_new + width * l_eff
        try:
            f_lo, f_hi = eq(lo), eq(hi)
            if np.sign(f_lo) == np.sign(f_hi):
                continue
            return float(brentq(eq, lo, hi))
        except Exception:
            continue
    return _solve_bracket(eq, max(EPS, y_new - 100.0 * l_eff), y_new + 100.0 * l_eff, fallback)


def apply_buy_yes(
    x: float,
    y: float,
    shares: float,
    L: float,
    tte_days: float,
    dynamic: bool,
) -> Tuple[float, float, float, float]:
    """Execute virtual BUY YES on pm-AMM state.

    Returns: (x_new, y_new, avg_price_yes, p_after_yes)
    """
    shares = max(float(shares), 0.0)
    p_before = price_yes_from_reserves(x, y, L, tte_days, dynamic)
    x_new = x + shares
    y_new = solve_y_given_x(x_new, L, tte_days, dynamic, y_hint=y)
    p_after = price_yes_from_reserves(x_new, y_new, L, tte_days, dynamic)
    avg_price = 0.5 * (p_before + p_after)
    return float(x_new), float(y_new), float(avg_price), float(p_after)


def apply_buy_no(
    x: float,
    y: float,
    shares: float,
    L: float,
    tte_days: float,
    dynamic: bool,
) -> Tuple[float, float, float, float]:
    """Execute virtual BUY NO on pm-AMM state.

    Returns: (x_new, y_new, avg_price_no, p_after_yes)
    """
    shares = max(float(shares), 0.0)
    p_before = price_yes_from_reserves(x, y, L, tte_days, dynamic)
    y_new = y + shares
    x_new = solve_x_given_y(y_new, L, tte_days, dynamic, x_hint=x)
    p_after = price_yes_from_reserves(x_new, y_new, L, tte_days, dynamic)
    avg_price_no = 0.5 * ((1.0 - p_before) + (1.0 - p_after))
    return float(x_new), float(y_new), float(avg_price_no), float(p_after)
