# GTA VI Trophy Rarity Forecast

![gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)
![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/apasi22/gta6-trophy-rarity-forecast-01/rank.svg)
![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/apasi22/gta6-trophy-rarity-forecast-01/reward.svg)

Forecasts each of Grand Theft Auto VI's 55 PlayStation 5 trophies' Exophase
earn rate (%) at 2026-11-26T00:00:00Z, as five quantiles (levels 0.05 / 0.25 /
0.5 / 0.75 / 0.95), from the house's reads of the game's trophy page.

## How it works

`main.py` (Python 3.13, stdlib only, offline, deterministic) reads every
`*.in` case in `$MAKEFUN_INPUT_DIR` and writes `<id>.out` to
`$MAKEFUN_OUTPUT_DIR`. For each trophy it models the earn rate on the logit
scale:

- **Logit-linear in log-time.** `logit(rate)` is fit by least squares against
  `ln(1 + hours since release)`, over the full history. The log-time form
  already bends the trajectory down as the player base grows: the rate is
  earners over players-tracked, and fitting the observed rate captures both.
- **Recency-weighted.** Reads are weighted by `exp(-age/36h)`, so the recent
  slope — the one the next four days will follow — matters most. The rate
  trajectory accelerates as players progress, and an unweighted full-history
  fit underestimates it.
- **Tier-pooled slopes.** Slopes are shrunk toward their tier mean by
  empirical Bayes, with a floor on the slope standard error so a flat
  two-day history cannot claim certainty.
- **Saturation caps.** The extrapolated rate is capped by tier (platinum 3%,
  gold 35%, silver 45%, bronze 70%) and by a 25x maximum growth factor, so a
  steep early slope cannot overshoot.
- **Calibrated spread.** Base term (refresh noise) plus a term growing with
  the square root of the horizon plus the trophy's own fit residual, so the
  90% band is honest about the forecast error.

## Validation

- Practice cases p00/p01: valid output (55 rows, ids 1–55, quantiles
  non-decreasing in [0, 100]), byte-identical on re-runs, ~0.06 s per run.
  The practice histories are flat (every read identical), so they only
  validate the format — the growth model is what matters for the final.
- Backtest on a cohort simulator (players join over time, each cohort earns
  trophies on its own clock, Exophase refresh staleness + binomial noise),
  five scenarios: mean pinball loss **0.130** vs **0.253** for the house
  curve baseline and **0.433** for persistence — a margin of **+0.123**
  over the best house baseline in every scenario, with an unbiased median
  (logit error mean ~0, sd ~0.25).
