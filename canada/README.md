# Canada 2026 Grand Prix — Model Documentation

**File:** `canada/canada_model.py`
**Run:** `uv run python canada/canada_model.py` (from repo root)

---

## Circuit profile

Circuit Gilles Villeneuve, Montreal. A semi-street circuit with one of the
longest uninterrupted straights in F1 (~1.1 km), heavy braking zones at the
hairpin (T10) and chicane complex (T13-14), and limited overtaking opportunities
outside of DRS activation zones.

```python
CANADA_CIRCUIT_PROFILE = {
    "high_speed_ratio":      0.45,
    "overtaking_difficulty": 0.30,   # DRS effective; wall proximity limits divebombs
    "brake_intensity":       0.75,   # T10 hairpin, T13-14 chicane — heavy flat-spots
    "pu_sensitivity":        0.80,   # long main straight and DRS zones strongly reward PU
    "street_circuit":        1,
    "traction_zones":        0.55,
}
```

---

## Weekend format

Sprint weekend: `FP1 -> SQ -> Sprint -> Q -> Race`

**Critical implication:** FP2 and FP3, which together account for approximately
42% of feature importance in the standard-weekend version of this model, are
absent. The model at inference time leans primarily on:

- FP1 baseline pace and tyre data
- Sprint Qualifying sector times and lap deltas
- Sprint race result (positions gained, finishing order)
- Saturday Q grid position
- PU score and team features

The sprint adjustment layer is therefore the most load-bearing correction for
this model, directly incorporating the Sprint race result after prediction.

---

## Training data

| Source | Races | Weight | Rationale |
|---|---|---|---|
| 2026 season | Australia, China, Japan, Miami (dynamic) | 1.00 | Same season, same regs |
| Canada 2025 | Canada | 0.22 | Same circuit, previous regs |
| Canada 2024 | Canada | 0.05 | Same circuit, two seasons back |
| Full 2024 season | ~24 races | 0.22 | Regulation context; gated by `INCLUDE_FULL_HISTORICAL` |
| Full 2023 season | ~22 races | 0.05 | Weak historical prior; gated by `INCLUDE_FULL_HISTORICAL` |

Total at prediction time: approximately 607 rows x 102 columns
(88 rows from 2026, 519 historical).

`INCLUDE_FULL_HISTORICAL = True` by default. Set to `False` for faster
iteration during development.

`discover_full_season(year, weight)` queries the FastF1 calendar for all
completed rounds in a given historical year.

---

## Configuration constants

### Upgrade scores (Canada 2026)

Sourced from F1.com technical reports and specialist press (June 2026).

| Team | Score | Technical summary |
|---|---|---|
| McLaren | 3 | Major package — floor, chassis, wings, bodywork |
| Mercedes | 2 | Significant — floor / front wing |
| Racing Bulls | 2 | Significant — floor / front wing |
| Alpine | 2 | Significant — floor / front wing |
| Williams | 1 | Minor — cooling / aero tweaks |
| Haas | 1 | Minor — cooling / aero tweaks |
| Cadillac | 1 | Minor — cooling / aero tweaks |
| Red Bull Racing | 0 | No performance upgrade |
| Ferrari | 0 | No performance upgrade |
| Aston Martin | 0 | No performance upgrade |
| Audi | 0 | No performance upgrade |

```
Scale: 0 = none  1 = minor/cooling  2 = significant floor/wing  3 = major package
```

### Sigma constants

```python
UPGRADE_SIGMA_PER_POINT = 0.18   # initial default; overwritten by calibrate_sigma()
SPRINT_BOOST_SIGMA = 0.20        # initial default; overwritten by calibrate_sigma()
```

Both are calibrated by `calibrate_sigma()` (Phase 1.5) before the final
prediction runs.

---

## Pipeline phases

### Phase 0 — Race discovery

`discover_2026_races_before_target("Canada")` queries the 2026 FastF1 calendar.
Falls back to `[Australia, China, Japan, Miami]` if the API is unavailable.

Historical Canada entries (2024, 2025) and full historical seasons are appended
from `HISTORICAL_CANADA` and `FULL_HISTORICAL_YEARS`.

### Phase 1 — Training set assembly (3 passes)

**Pass 1:** For each race in the combined list, extract features and labels and
push into `race_cache`.

**Pass 2:** Rolling 2026 season form injection. For each 2026 race in
chronological order, `recent_form_avg`, `recent_form_wins`, and
`team_2026_pace_rank` are computed from all prior 2026 races only. This
prevents future-race leakage into training rows.

**Pass 3:** Concatenate all cached frames into `df_train`.

### Phase 1.5 — Sigma calibration

`calibrate_sigma(race_cache, feature_cols)` grid-searches 25 combinations of
`(upgrade_sigma, sprint_sigma)` on completed sprint weekends with known race
results, maximising mean Spearman rho in a LOO loop.

```
upgrade_sigmas = [0.10, 0.15, 0.18, 0.20, 0.25]
sprint_sigmas  = [0.10, 0.15, 0.20, 0.25, 0.30]
```

Sprint weekend pool used for calibration:

| Season | Weekends |
|---|---|
| 2026 | China, Miami (and any subsequent sprint rounds completed before Canada) |
| 2024 | China, Miami, Austria, Azerbaijan, USA, Sao Paulo, Qatar |
| 2023 | Austria, Belgium, Brazil, Qatar |

If fewer than 2 sprint weekends are available with known results, defaults are
used (`upgrade_sigma=0.18`, `sprint_sigma=0.20`).

### Phase 2 — LOO-CV

Leave-one-out cross-validation over all training races. Reports per-fold NDCG
and Spearman rho, plus the mean and std of both metrics vs the `grid_position`
baseline. Canada 2026 reference results (4 folds):

```
Model    NDCG: 0.950 +/- 0.042   delta vs baseline: -0.006
Baseline NDCG: 0.956 +/- 0.040
Spearman rho:  0.716 +/- 0.157
```

### Phase 2.5 — Ablation study

Measures the incremental Spearman rho contribution of each adjustment layer on
completed 2026 races. Canada 2026 reference:

| Variant | Mean rho | vs Base |
|---|---|---|
| Base only | +0.553 | — |
| + Upgrade | +0.563 | +0.010 |
| + Sprint | +0.651 | +0.099 |
| All combined | +0.701 | +0.148 |

### Phase 3 — Target feature matrix

`build_race_features(2026, "Canada")` extracts inference features. For a sprint
weekend the FP2/FP3 path is skipped and the SQ/S path is used instead. All
FP2/FP3 columns are `NaN` at inference.

### Phase 4 — Train final model

`train_model(df_train, feature_cols)` trains on all data. One weight per race
group is computed from the `sample_weight` column in `df_train`.

### Phase 5 — Predict and adjust

1. `predict_top()` — raw XGBRanker scores -> ranked list.
2. Upgrade adjustment with calibrated `best_upgrade_sigma`.
3. Sprint adjustment with calibrated `best_sprint_sigma`.

---

## Feature coverage at inference

| Session | Status | Notes |
|---|---|---|
| FP1 | Available | Baseline pace and tyre data |
| FP2 | Not applicable | Sprint weekend |
| FP3 | Not applicable | Sprint weekend |
| SQ | Available | Sector times, lap delta, speed trap |
| Sprint race | Available after Saturday | `s_finish_position`, `s_positions_gained` |
| Q | Available after Saturday | Grid position, Q3 delta to pole |

The FP2/FP3 absence is the largest structural risk. Feature importance analysis
shows FP2 and FP3 combined contribute ~42% in the standard-weekend training rows.
At inference those columns are `NaN`, so XGBoost's Sparsity-Aware Split Finding
routes those observations through the missing-value branch in each tree.

---

## Known limitations

- No weather data. Montreal has historically unpredictable weather and rain
  conditions substantially change race outcomes. Weather could be added as
  manual override features (similar to upgrade scores) once the race-day
  forecast is known.
- FP2/FP3 absence significantly reduces the model's ability to differentiate
  long-run race pace. The sprint adjustment compensates but cannot fully
  replicate 90 minutes of race simulation data.
