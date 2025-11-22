# bs_pricing.py
import math
import numpy as np

# ---------------- Normal PDF / CDF ---------------- #

def _norm_pdf(x: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """
    Fast, fully vectorised approximation to standard normal CDF N(x).

    Abramowitz–Stegun 7.1.26 style:
        N(x) ≈ 1 - φ(x) * P(t),  t = 1 / (1 + p |x|)
    with max error ~1e-7 in practice.
    """
    x = np.asarray(x, dtype=float)

    # coefficients
    p  = 0.2316419
    b1 = 0.319381530
    b2 = -0.356563782
    b3 = 1.781477937
    b4 = -1.821255978
    b5 = 1.330274429

    abs_x = np.abs(x)
    t = 1.0 / (1.0 + p * abs_x)

    # φ(x) = standard normal PDF
    pdf = 0.3989422804014327 * np.exp(-0.5 * abs_x * abs_x)

    poly = (((((b5 * t + b4) * t) + b3) * t + b2) * t + b1) * t
    cdf_approx = 1.0 - pdf * poly

    # reflect for x < 0
    return np.where(x >= 0.0, cdf_approx, 1.0 - cdf_approx)
# ---------------- Black–Scholes core ---------------- #

def bs_price(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    is_call: np.ndarray,
    q: float = 0.0,
) -> np.ndarray:
    """
    Vectorised Black–Scholes price with continuous compounding.
    q is dividend yield (we will pass q=0 in your project).
    All inputs are numpy arrays, broadcastable to a common shape.
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)

    # Safe masks
    valid = (S > 0.0) & (K > 0.0) & (T > 0.0) & (sigma > 0.0)
    out = np.full_like(S, np.nan, dtype=float)
    if not np.any(valid):
        return out

    S_v = S[valid]
    K_v = K[valid]
    T_v = T[valid]
    r_v = r[valid]
    sigma_v = sigma[valid]
    is_call_v = is_call[valid]

    sqrtT = np.sqrt(T_v)
    d1 = (np.log(S_v / K_v) + (r_v - q + 0.5 * sigma_v * sigma_v) * T_v) / (sigma_v * sqrtT)
    d2 = d1 - sigma_v * sqrtT

    Nd1 = _norm_cdf(d1)
    Nd2 = _norm_cdf(d2)
    Nmd1 = _norm_cdf(-d1)
    Nmd2 = _norm_cdf(-d2)

    disc_r = np.exp(-r_v * T_v)
    disc_q = np.exp(-q * T_v)

    call_prices = disc_q * S_v * Nd1 - disc_r * K_v * Nd2
    put_prices  = disc_r * K_v * Nmd2 - disc_q * S_v * Nmd1

    prices = np.where(is_call_v, call_prices, put_prices)
    out[valid] = prices
    return out


def bs_greeks(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    is_call: np.ndarray,
    q: float = 0.0,
):
    """
    Vectorised BS greeks: delta, gamma, vega, theta.
    Vega here is dPrice/dSigma (per 1.0 absolute sigma unit, not per 1%).
    """
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)

    valid = (S > 0.0) & (K > 0.0) & (T > 0.0) & (sigma > 0.0)
    delta = np.full_like(S, np.nan, dtype=float)
    gamma = np.full_like(S, np.nan, dtype=float)
    vega = np.full_like(S, np.nan, dtype=float)
    theta = np.full_like(S, np.nan, dtype=float)

    if not np.any(valid):
        return delta, gamma, vega, theta

    S_v = S[valid]
    K_v = K[valid]
    T_v = T[valid]
    r_v = r[valid]
    sigma_v = sigma[valid]
    is_call_v = is_call[valid]

    sqrtT = np.sqrt(T_v)
    d1 = (np.log(S_v / K_v) + (r_v - q + 0.5 * sigma_v * sigma_v) * T_v) / (sigma_v * sqrtT)
    d2 = d1 - sigma_v * sqrtT

    Nd1 = _norm_cdf(d1)
    Nmd1 = _norm_cdf(-d1)
    pdf_d1 = _norm_pdf(d1)

    disc_r = np.exp(-r_v * T_v)
    disc_q = np.exp(-q * T_v)

    # Delta
    delta_v = np.where(is_call_v, disc_q * Nd1, -disc_q * Nmd1)

    # Gamma
    gamma_v = disc_q * pdf_d1 / (S_v * sigma_v * sqrtT)

    # Vega (per 1.0 volatility)
    vega_v = disc_q * S_v * pdf_d1 * sqrtT

    # Theta (calendar-day basis; divide by 365)
    call_theta = (
        -disc_q * S_v * pdf_d1 * sigma_v / (2.0 * sqrtT)
        - r_v * disc_r * K_v * _norm_cdf(d2)
        + q * disc_q * S_v * Nd1
    )
    put_theta = (
        -disc_q * S_v * pdf_d1 * sigma_v / (2.0 * sqrtT)
        + r_v * disc_r * K_v * _norm_cdf(-d2)
        - q * disc_q * S_v * Nmd1
    )
    theta_v = np.where(is_call_v, call_theta, put_theta) / 365.0

    delta[valid] = delta_v
    gamma[valid] = gamma_v
    vega[valid] = vega_v
    theta[valid] = theta_v

    return delta, gamma, vega, theta


# ---------------- Implied volatility (vectorised Newton) ---------------- #

def implied_vol_newton(
    price: np.ndarray,
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    r: np.ndarray,
    is_call: np.ndarray,
    q: float = 0.0,
    sigma_init: float = 0.3,
    tol: float = 1e-6,
    max_iter: int = 40,
):
    """
    Vectorised Newton–Raphson IV solver.
    - Returns sigma array (float) with NaN where invalid/unconverged.
    - Handles near-intrinsic options by clamping to small vol.
    """
    price = np.asarray(price, dtype=float)
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    is_call = np.asarray(is_call, dtype=bool)

    sigma = np.full_like(price, np.nan, dtype=float)

    valid = (price > 0.0) & (S > 0.0) & (K > 0.0) & (T > 0.0)
    if not np.any(valid):
        return sigma

    # work arrays
    p_v = price[valid]
    S_v = S[valid]
    K_v = K[valid]
    T_v = T[valid]
    r_v = r[valid]
    is_call_v = is_call[valid]

    disc_r = np.exp(-r_v * T_v)

    call_intrinsic = np.maximum(S_v - K_v * disc_r, 0.0)
    put_intrinsic  = np.maximum(K_v * disc_r - S_v, 0.0)
    intrinsic = np.where(is_call_v, call_intrinsic, put_intrinsic)

    # if market price is at or below intrinsic: treat as tiny vol
    near_intrinsic = p_v <= intrinsic + 1e-10
    sigma_v = np.full_like(p_v, np.nan, dtype=float)
    sigma_v[near_intrinsic] = 1e-4

    work_mask = ~near_intrinsic
    if np.any(work_mask):
        sig = np.full(np.sum(work_mask), sigma_init, dtype=float)
        sig = np.clip(sig, 1e-4, 5.0)

        p_w = p_v[work_mask]
        S_w = S_v[work_mask]
        K_w = K_v[work_mask]
        T_w = T_v[work_mask]
        r_w = r_v[work_mask]
        is_call_w = is_call_v[work_mask]

        for _ in range(max_iter):
            # compute price and vega for current sig
            px = bs_price(S_w, K_w, T_w, r_w, sig, is_call_w, q=q)
            _, _, vega_w, _ = bs_greeks(S_w, K_w, T_w, r_w, sig, is_call_w, q=q)

            diff = px - p_w
            done = np.abs(diff) < tol
            if np.all(done | (vega_w <= 1e-8)):
                break

            # avoid divide by small vega
            upd_mask = ~done & (vega_w > 1e-8)
            sig_new = sig.copy()
            sig_new[upd_mask] = sig[upd_mask] - diff[upd_mask] / vega_w[upd_mask]
            sig = np.clip(sig_new, 1e-4, 5.0)

        sigma_v[work_mask] = sig

    sigma[valid] = sigma_v
    return sigma
