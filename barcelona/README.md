# Barcelona 2026 Grand Prix — Model Documentation

**File:** `barcelona/barcelona_model.py`
**Run:** `uv run python barcelona/barcelona_model.py` (from repo root)

---

## Circuit disambiguation

There are two Spanish rounds in the 2026 calendar:

| Round | Circuit | GP name to use |
|---|---|---|
| Round 9 (June) | Circuit de Barcelona-Catalunya | `gp="Barcelona"` |
| Round 16 | TBD, Madrid | `gp="Spain"` or Madrid-specific |

`TARGET_RACE = {"year": 2026, "gp": "Barcelona"}`. Historical priors use
`gp="Spain"` because FastF1 labels the Spanish GP at Barcelona-Catalunya as
"Spain" in pre-2026 seasons.

---

## Circuit profile

Circuit de Barcelona-Catalunya, Montmelo. A permanent circuit that has served
as the primary aerodynamic development benchmark in F1 for decades. High-speed
sweepers in the opening sector, medium-speed technical sections in sectors 2
and 3, and strong PU demand on the pit straight.

```python
BARCELONA_CIRCUIT_PROFILE = {
    "high_speed_ratio":      0.60,   # Turn 3, Turn 9 complex — significant high-speed load
    "overtaking_difficulty": 0.65,   # DRS helps but tyre deg often produces strategic overtakes
    "brake_intensity":       0.55,   # moderate braking — no extreme hairpins
    "pu_sensitivity":        0.70,   # long main straight and DRS zone
    "street_circuit":        0,
    "traction_zones":        0.65,
}
```

---

## Grid-anchor blend

Barcelona has moderate overtaking difficulty (0.65). Following the Monaco
model pattern, predictions are blended with the grid-position score, but with
a lower anchor weight than Monaco:

```python
GRID_ANCHOR_WEIGHT = 0.30   # initial; overwritten by calibrate_params()
```

```python
grid_score = (n_drivers + 1) - grid_position
final_score = alpha * grid_score + (1 - alpha) * adjusted_model_score
```

The lower alpha (0.30 vs Monaco's 0.50) reflects that on-track overtaking and
strategy plays are more frequent at Barcelona than at Monaco, so the model
score carries more weight.

---

## Weekend format

Standard weekend: `FP1 -> FP2 -> FP3 -> Q -> Race`

Full free practice coverage. FP2 race simulation data is the primary long-run
signal. FP3 provides final setup tuning and qualifying simulation context.

---

## Training data

| Source | Races | Weight | Rationale |
|---|---|---|---|
| 2026 season | Prior 2026 rounds (dynamic) | 1.00 | Same regs, same season |
| Spain 2025 (Barcelona) | Spain | 0.22 | Same circuit, previous regs |
| Spain 2024 (Barcelona) | Spain | 0.05 | Same circuit, two seasons back |
| Full 2024 season | ~24 races | 0.22 | Regulation context; gated by `INCLUDE_FULL_HISTORICAL` |
| Full 2023 season | ~22 races | 0.05 | Weak historical prior; gated by `INCLUDE_FULL_HISTORICAL` |

`INCLUDE_FULL_HISTORICAL = True` by default. Set to `False` for faster
iteration.

`discover_2026_races_before_target("Barcelona")` queries the 2026 calendar and
returns all rounds preceding the Barcelona round number.

---

## Configuration constants

### Upgrade scores (Barcelona 2026)

Sourced from F1.com technical reports and specialist press (June 2026).
Ferrari's score is kept deliberately conservative relative to the volume of
reported components — experience from previous models showed over-scoring
Ferrari upgrades at aero-sensitive circuits introduced noise.

| Team | Score | Technical summary |
|---|---|---|
| Ferrari | 2 | Significant — floor / front wing |
| Red Bull Racing | 2 | Significant — floor / front wing |
| Racing Bulls | 2 | Significant — floor / front wing |
| McLaren | 1 | Minor — cooling / aero tweaks |
| Mercedes | 1 | Minor — cooling / aero tweaks |
| Williams | 1 | Minor — cooling / aero tweaks |
| Cadillac | 1 | Minor — cooling / aero tweaks |
| Haas | 1 | Minor — cooling / aero tweaks |
| Aston Martin | 0 | No performance upgrade |
| Alpine | 0 | No performance upgrade |
| Audi | 0 | No performance upgrade |

```
Scale: 0 = none  1 = minor/cooling  2 = significant floor/wing  3 = major package
```

### Calibrated constants

```python
GRID_ANCHOR_WEIGHT = 0.30        # initial; overwritten by calibrate_params()
UPGRADE_SIGMA_PER_POINT = 0.18   # initial; overwritten by calibrate_params()
```

Both are calibrated jointly by `calibrate_params()` in Phase 1.5.

---

## Pipeline phases

### Phase 0 — Race discovery

`discover_2026_races_before_target("Barcelona")` queries the 2026 FastF1
calendar. Historical Barcelona/Spain entries and full historical seasons are
appended from `HISTORICAL_BARCELONA` and `FULL_HISTORICAL_YEARS`.

### Phase 1 — Training set assembly (3 passes)

**Pass 1:** Feature and label extraction for every race in the combined list,
stored in `race_cache`.

**Pass 2:** Rolling 2026 season form injection (`recent_form_avg`,
`recent_form_wins`, `team_2026_pace_rank`) computed from prior 2026 races only.
Prevents future-race leakage into any training row.

**Pass 3:** Concatenate all cached frames into `df_train`.

### Phase 1.5 — Parameter calibration

`calibrate_params(race_cache, feature_cols)` grid-searches combinations of
`(upgrade_sigma, grid_anchor_weight)` on completed standard-weekend races in
the training set, maximising mean Spearman rho in a LOO loop.

No sprint calibration is run — Barcelona is a standard weekend with no sprint
adjustment layer.

### Phase 2 — LOO-CV

Leave-one-out cross-validation over all training races. Reports NDCG and
Spearman rho vs `grid_position` baseline.

### Phase 2.5 — Ablation study

Measures the incremental Spearman rho contribution of each layer:
- Base XGBRanker output
- Base + upgrade adjustment
- Base + grid anchor
- All combined

### Phase 3 — Target feature matrix

`build_race_features(2026, "Barcelona")` extracts inference features from
`FP1 + FP2 + FP3 + Q`. All sessions are available.

### Phase 4 — Train final model

`train_model(df_train, feature_cols)` trains on all available data.

### Phase 5 — Predict and adjust

1. `predict_top()` — raw XGBRanker scores.
2. Upgrade adjustment with calibrated `best_upgrade_sigma`.
3. Grid-anchor blend: final score = `best_alpha * grid_score + (1 - best_alpha) * adjusted_score`.

No sprint adjustment. The sprint layer is present only in sprint-weekend models
(Canada, Miami).

---

## Feature coverage at inference

| Session | Status | Notes |
|---|---|---|
| FP1 | Available | Baseline pace, tyre data |
| FP2 | Available | Long-run race simulation (primary pace signal) |
| FP3 | Available | Final setup tune, qualy-sim delta, track evolution |
| SQ | Not applicable | Standard weekend |
| Sprint race | Not applicable | Standard weekend |
| Q | Available | Grid position, Q3 delta to pole |

Barcelona is the most feature-complete model in the pipeline — all session types
are available and the ERS/PU deployment features from FP2 (`fp2_ers_efficiency_proxy`,
`fp2_pu_asymmetry_delta`, `fp2_speed_trap_std_kmh`, `fp2_avg_lift_coast_time_s`)
are meaningful given Active Aero is fully enabled and the main straight is long
enough to create measurable deployment differences.

---

## Differences from the Canada model (template)

| Aspect | Canada | Barcelona |
|---|---|---|
| Weekend format | Sprint | Standard |
| Grid anchor | No | Yes (alpha=0.30, calibrated) |
| Sprint adjustment | Yes (calibrated sigma) | No |
| FP2/FP3 at inference | Not available (NaN) | Fully available |
| Ferrari upgrade score | 0 | 2 (conservative) |
| SPRINT_BOOST_SIGMA | 0.20 | Not defined |

Barcelona is structurally identical to the Monaco model but with lower
`GRID_ANCHOR_WEIGHT` (0.30 vs 0.50), reflecting the higher overtaking
probability and greater role of race pace relative to starting position.

---

## Known limitations

- No weather data. Afternoon races in June at Barcelona can produce unpredictable
  tyre degradation in high ambient temperatures, which the model does not model
  explicitly.
- Ferrari upgrade scores are intentionally conservative. If Ferrari brings a
  significantly larger package than reported, the upgrade adjustment will
  underestimate the performance gain.
- Barcelona has served as the development test circuit for decades, meaning teams
  optimise specifically for this track. The `fp3_vs_fp2_improvement` and
  `fp3_s{1,2,3}_delta_vs_fp2` features should capture overnight setup work that
  is particularly pronounced here.
