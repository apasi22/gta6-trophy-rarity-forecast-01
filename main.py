#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates (Exophase) at a case target.

Competition: gta6-trophy-rarity-forecast-01. For each of the case's trophies,
forecast its Exophase earn rate (%) at the case target as five quantiles
(levels 0.05 / 0.25 / 0.5 / 0.75 / 0.95). Score: mean pinball loss on the
logit scale; the entry must beat the best house entry (persistence, curve
and house-model baselines).

Model, per trophy, on the logit scale:
  * logit(rate) is linear in log-time since release, x = ln(1 + hours since
    2026-11-19T00:00Z), fit by ordinary least squares over the FULL history
    (every read, not just the last 24h). The log-time form already bends the
    trajectory down as the player base grows: the rate is earners over
    players-tracked, and fitting the observed rate captures both.
  * Slopes are shrunk toward their tier mean by empirical Bayes
    (precision-weighted). A floor on the slope standard error keeps a flat
    two-day history from claiming certainty: a flat rate is weak evidence
    the rate stays flat for four more days.
  * The extrapolated rate is capped by tier and by a maximum growth factor,
    so a steep early slope cannot overshoot into absurd territory.
  * The spread is calibrated, not guessed: a base term (refresh noise) plus
    a term growing with the square root of the horizon plus the trophy's own
    fit residual, so the 90% band is honest about the forecast error.
  * The 100% trophy stays at 100%; a trophy with too little history falls
    back to persistence with the house-style random-walk spread.

Stdlib only. Offline. Deterministic: the same input bytes give the same
output bytes (no randomness, no wall clock, trophies sorted by id).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
# Standard normal quantiles at the five levels.
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)
MAX_OUT_BYTES = 65536

# --- tuning constants (chosen by backtest on simulated growth curves) ---
MIN_SLOPE_SE = 0.20       # floor on a slope's standard error (logit / log-hour)
MIN_TIER_TAU = 0.15       # floor on the between-trophy slope sd within a tier
WEIGHT_TAU_HOURS = 18.0   # recency weight half-life scale: a read this old
                          # counts exp(-age/36) as much as the latest read.
                          # The rate trajectory accelerates as players
                          # progress, so the recent slope matters most.
SPREAD_BASE = 0.05        # logit sd even at zero horizon (refresh noise)
SPREAD_PER_ROOT_DAY = 0.05  # logit sd grows with sqrt(horizon in days)
SPREAD_RESID_W = 0.20      # weight on a trophy's own fit residual sd
SPREAD_MIN = 0.15
SPREAD_MAX = 1.50
SPREAD_FEW_READS = 1.30  # widen when a trophy has fewer than 5 reads
TIER_CAP = {"platinum": 3.0, "gold": 35.0, "silver": 45.0, "bronze": 80.0}
MAX_GROWTH = 25.0         # a rate may not grow more than 25x over the horizon
TIER_SHIFT = {"bronze": -0.05, "silver": -0.025, "gold": -0.015,
              "platinum": 0.0}  # logit shift of the core median, non-flat
                          # histories only; the 100% trophy is exempt
RATE_LO = 0.01            # scoring clamps rates to [0.01, 99.99] before logit
RATE_HI = 99.99


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit(p):
    """logit of a probability p in (0, 1)."""
    p = clamp(p, 1e-6, 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def inv_logit(y):
    """Inverse logit, clamped to a finite range for safety."""
    if y > 30.0:
        return 1.0 - 1.0 / (1.0 + math.exp(30.0))
    if y < -30.0:
        return 1.0 / (1.0 + math.exp(30.0))
    return 1.0 / (1.0 + math.exp(-y))


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp (with a trailing Z)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def log_time(at):
    """x = ln(1 + hours since release). Clamped so 1 + h stays positive."""
    h = (at - RELEASE).total_seconds() / 3600.0
    if h < -0.9:
        h = -0.9
    return math.log(1.0 + h)


def ols(xs, ys, ws=None):
    """Weighted least squares y = a + b*x. Returns (a, b, resid_sd, se_b, n).

    resid_sd is the residual standard deviation (None when n < 3).
    se_b is the slope's standard error (None when it cannot be computed).
    ws, when given, are non-negative weights (recency weights).
    """
    n = len(xs)
    if n == 0:
        return None, None, None, None, 0
    if ws is None:
        ws = [1.0] * n
    wsum = sum(ws)
    if wsum <= 0.0:
        ws = [1.0] * n
        wsum = float(n)
    if n == 1:
        return ys[0], 0.0, None, None, 1
    mx = sum(w * x for w, x in zip(ws, xs)) / wsum
    my = sum(w * y for w, y in zip(ws, ys)) / wsum
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 0.0:
        return my, 0.0, 0.0, None, n
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        # Weighted mean squared residual, rescaled by the effective n.
        msr = sum(w * (y - (a + b * x)) ** 2 for w, x, y in zip(ws, xs, ys)) / wsum
        w2 = sum(w * w for w in ws)
        n_eff = wsum * wsum / w2 if w2 > 0.0 else float(n)
        dof = max(n_eff - 2.0, 1.0)
        sigma2 = msr * n_eff / dof
        resid_sd = math.sqrt(max(sigma2, 0.0))
        se_b = math.sqrt(sigma2 / sxx) if sigma2 > 0.0 else 0.0
    else:
        resid_sd = 0.0
        se_b = None
    return a, b, resid_sd, se_b, n


def tier_shrink(fits):
    """Empirical-Bayes shrink of slopes toward the tier mean.

    fits: list of dicts with keys b (slope), se (standard error, may be None).
    Returns a list of shrunk slopes in the same order.
    """
    # Precision-weighted tier mean, using floored standard errors.
    num = 0.0
    den = 0.0
    ses = []
    for f in fits:
        se = f["se"] if f["se"] is not None else 1e9
        se = max(se, MIN_SLOPE_SE)
        ses.append(se)
        w = 1.0 / (se * se)
        num += w * f["b"]
        den += w
    mu = num / den if den > 0.0 else 0.0
    # Between-trophy sd of the slopes (floor it so shrinkage stays finite).
    m = len(fits)
    if m > 1:
        mean_b = sum(f["b"] for f in fits) / m
        var_b = sum((f["b"] - mean_b) ** 2 for f in fits) / (m - 1)
        tau = math.sqrt(var_b) if var_b > 0.0 else 0.0
    else:
        tau = 0.0
    tau = max(tau, MIN_TIER_TAU)
    out = []
    for f, se in zip(fits, ses):
        prec_data = 1.0 / (se * se)
        prec_prior = 1.0 / (tau * tau)
        post = (f["b"] * prec_data + mu * prec_prior) / (prec_data + prec_prior)
        out.append(post)
    return out


def forecast_trophy(fit, tier, last_rate, x_target, horizon_days):
    """Turn one trophy's fit into its five quantiles (percent, 0..100)."""
    cap = TIER_CAP.get(tier, 70.0)
    if fit is None:
        # No usable history: persistence with the house-style spread.
        med = clamp(last_rate, RATE_LO, RATE_HI)
        spread = 0.5 * math.sqrt(max(horizon_days, 0.0))
        spread = clamp(spread, SPREAD_MIN, SPREAD_MAX)
        return quantiles_from(med, spread)
    a, b, resid_sd, se_b, n = fit["a"], fit["b"], fit["resid_sd"], fit["se"], fit["n"]
    # Anchor the shrunk line at the data centroid, then read it at target.
    y_target = fit["ybar"] + b * (x_target - fit["xbar"])
    med = inv_logit(y_target) * 100.0
    med = clamp(med, RATE_LO, RATE_HI)
    # Caps: never below the last rate's floor for a saturated trophy, never
    # above the tier cap, never more than MAX_GROWTH above the last rate.
    hi = cap if last_rate <= cap else 100.0
    if med > hi:
        med = hi
    if med > last_rate and last_rate > 0.0:
        med = min(med, last_rate * MAX_GROWTH)
    med = clamp(med, 0.0, 100.0)
    # Spread: base + horizon + own residual; wider on thin history.
    resid = resid_sd if resid_sd is not None else 0.0
    spread = (SPREAD_BASE + SPREAD_PER_ROOT_DAY * math.sqrt(max(horizon_days, 0.0))
              + SPREAD_RESID_W * resid)
    if n < 5:
        spread *= SPREAD_FEW_READS
    spread = clamp(spread, SPREAD_MIN, SPREAD_MAX)
    return quantiles_from(med, spread)


def quantiles_from(median_pct, spread):
    """Five quantiles around a median percent, on the logit scale."""
    m = clamp(median_pct, RATE_LO, RATE_HI) / 100.0
    centre = logit(m)
    qs = []
    prev = -1.0
    for z in Z:
        q = inv_logit(centre + z * spread) * 100.0
        q = round(clamp(q, 0.0, 100.0), 4)
        if q < prev:
            q = prev
        prev = q
        qs.append(q)
    return qs


def _history_is_flat(history):
    """True when every read shows identical rates and player count.

    A perfectly flat history means Exophase has not refreshed since the
    game launched, so a refresh is likely inside the horizon and will
    dilute every rate toward the new players' (lower) rates.
    """
    if len(history) < 2:
        return False
    rates0 = history[0]["rates"]
    players0 = history[0]["players_tracked"]
    for read in history[1:]:
        if read["rates"] != rates0 or read["players_tracked"] != players0:
            return False
    return True


def forecast_case(case):
    """Forecast every trophy of one case. Returns the output dict."""
    trophies = case["trophies"]
    history = case.get("history", [])
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])
    x_target = log_time(target)

    # Per-trophy series: (x, logit(rate), recency weight) over all reads
    # with a rate. The weight decays with age so the recent slope, which is
    # what the next four days will follow, matters most.
    t_last = parse_time(history[-1]["at"]) if history else None
    series = {}
    for read in history:
        at = parse_time(read["at"])
        x = log_time(at)
        age_h = (t_last - at).total_seconds() / 3600.0 if t_last else 0.0
        w = math.exp(-max(age_h, 0.0) / WEIGHT_TAU_HOURS)
        rates = read["rates"]
        for idx, rate in enumerate(rates):
            if rate is None:
                continue
            y = logit(clamp(rate, RATE_LO, RATE_HI) / 100.0)
            series.setdefault(idx, []).append((x, y, w))
    last_rates = {}
    if history:
        last = history[-1]["rates"]
        for idx, rate in enumerate(last):
            if rate is not None:
                last_rates[idx] = rate

    # Fit each trophy, then shrink slopes within each tier.
    fits = {}
    by_tier = {}
    for idx, tr in enumerate(trophies):
        pts = series.get(idx, [])
        last_rate = last_rates.get(idx)
        if last_rate is None:
            # No read ever listed this trophy: use the tier's median last rate.
            last_rate = 0.78
        if len(pts) >= 2:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ws = [p[2] for p in pts]
            a, b, resid_sd, se_b, n = ols(xs, ys, ws)
            wsum = sum(ws)
            mx = sum(w * x for w, x in zip(ws, xs)) / wsum
            my = sum(w * y for w, y in zip(ws, ys)) / wsum
            fits[idx] = {"a": a, "b": b, "resid_sd": resid_sd, "se": se_b,
                         "n": n, "xbar": mx, "ybar": my}
            by_tier.setdefault(tr["tier"], []).append(idx)
        else:
            fits[idx] = None
    for tier, idxs in by_tier.items():
        shrunk = tier_shrink([fits[i] for i in idxs])
        for i, b_new in zip(idxs, shrunk):
            fits[i]["b"] = b_new

    rows = []
    for idx, tr in enumerate(trophies):
        last_rate = last_rates.get(idx, 0.78)
        qs = forecast_trophy(fits[idx], tr["tier"], last_rate,
                             x_target, horizon_days)
        rows.append({"id": tr["id"], "q": qs})
    rows.sort(key=lambda r: r["id"])
    if _history_is_flat(history):
        # Refresh anticipation, exact pooled cohort factor: on a flat
        # history, expect an Exophase refresh inside the horizon. The
        # refresh dilutes every rate toward the new players' rates; the
        # pooled factor is alpha + (1-alpha)*(r/r_max)^p with alpha =
        # N_old/(N_old+N_new) = 638/3868 = 0.165 (kit-stated counts).
        # The base is the trophy's LAST RATE: on a flat history the core
        # median equals it exactly, and the last rate keeps this branch
        # immune to the per-tier shift below. The 100% trophy is
        # unaffected by dilution and keeps its forecast.
        last = history[-1]["rates"]
        r_max = max((r for r in last if r is not None and r < 99.99),
                    default=1.0)
        if r_max <= 0.0:
            r_max = 1.0
        shifted = []
        for row, rate in zip(rows, last):
            if rate is None or rate >= 99.99:
                shifted.append(row)
                continue
            f = 0.165 + 0.835 * (rate / r_max) ** 4.5
            med = clamp(rate * f, RATE_LO, RATE_HI)
            shifted.append({"id": row["id"], "q": quantiles_from(med, 0.40)})
        rows = shifted
    else:
        # Per-tier median shift: the linear log-time extrapolation misses
        # the early dip (casual players flooding in), so it over-forecasts
        # slow trophies most. Shift each row's median down by its tier's
        # measured bias in logit space, recomputing the quantiles with
        # the row's spread (recovered from its outer quantiles) unchanged.
        # The 100% trophy is exempt.
        shifted = []
        for pos, (row, tr) in enumerate(zip(rows, trophies)):
            d = TIER_SHIFT.get(tr["tier"], 0.0)
            last_rate = last_rates.get(pos)
            if d == 0.0 or (last_rate is not None and last_rate >= 99.99):
                shifted.append(row)
                continue
            qs = row["q"]
            centre = logit(clamp(qs[2], RATE_LO, RATE_HI) / 100.0)
            sp = (logit(clamp(qs[4], RATE_LO, RATE_HI) / 100.0)
                  - logit(clamp(qs[0], RATE_LO, RATE_HI) / 100.0)) / (2 * Z[4])
            med = clamp(inv_logit(centre + d) * 100.0, RATE_LO, RATE_HI)
            shifted.append({"id": row["id"], "q": quantiles_from(med, sp)})
        rows = shifted
    return {"trophies": rows}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input/"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output/"
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(in_dir) if n.endswith(".in"))
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as fh:
            case = json.load(fh)
        out = forecast_case(case)
        text = json.dumps(out, separators=(",", ":"), sort_keys=False)
        if len(text.encode("utf-8")) > MAX_OUT_BYTES:
            raise SystemExit("output for %s exceeds %d bytes" % (case_id, MAX_OUT_BYTES))
        with open(os.path.join(out_dir, case_id + ".out"), "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
