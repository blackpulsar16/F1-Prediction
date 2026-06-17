# F1 Race Prediction Pipeline — 2026 Season

Machine learning pipeline for predicting Formula 1 race finishing order.
Telemetry features are extracted via [FastF1](https://theoehrly.github.io/Fast-F1/)
and fed into an `XGBRanker` pairwise ranking model trained per race weekend.

---

## Repository layout

```
F1-Prediction/
├── f1-api/
│   └── f1_fe.py              # Shared feature engineering library
├── miami/
│   ├── miami_model.py        # Miami 2026 GP model
│   └── README.md             # Miami-specific documentation
├── canada/
│   ├── canada_model.py       # Canada 2026 GP model
│   └── README.md
├── monaco/
│   ├── monaco_model.py       # Monaco 2026 GP model
│   └── README.md
├── barcelona/
│   ├── barcelona_model.py    # Barcelona 2026 GP model
│   └── README.md
├── team_mappings.json        # PU score and works-team flags per constructor
├── f1_cache/                 # FastF1 disk cache (auto-populated on first run)
├── pyproject.toml            # Project metadata and dependency spec (PEP 621)
├── uv.lock                   # Locked dependency graph
└── .python-version           # Pinned interpreter: Python 3.13
```

---

## Setup & Usage

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Create virtual environment and sync dependencies

```bash
uv sync
```

Creates `.venv/` with Python 3.13 and installs all locked dependencies.

### Run a prediction model

```bash
uv run python miami/miami_model.py
uv run python canada/canada_model.py
uv run python monaco/monaco_model.py
uv run python barcelona/barcelona_model.py
```

> **FastF1 cache:** session data is fetched from the API on first run and stored
> in `f1_cache/`. Subsequent runs load from disk — typically 5-10x faster. The
> first run with `INCLUDE_FULL_HISTORICAL = True` downloads ~50 race weekends
> and may hit the Ergast/Jolpi rate limit (500 calls/h); the library falls back
> to cached responses automatically.

### Add or update dependencies

```bash
uv add <package>
uv lock --upgrade-package <package>
```

---

## Feature engineering library — `f1-api/f1_fe.py`

Shared library of pure functions, each loading one FastF1 session and returning
a `pd.DataFrame` keyed by `Driver`. No global state is written.

### Universal quality filters

Applied in every function before computation:

| Filter | Condition |
|---|---|
| Green flag only | `TrackStatus == '1'` (excludes yellow, red, SC, VSC) |
| Clean lap | `PitOutTime` and `PitInTime` are both `NaN` |
| Valid time | `LapTime` is not `NaN` and `> Timedelta(0)` |
| 107% gate | `LapTime <= 1.07 * session_best` (circuit baseline functions) |
| Per-stint median gate | Long-run functions use `_keep_representative_laps()`, which filters within `1.07 * stint_median` instead of session-best. Prevents high-fuel laps from being wrongly discarded as outliers. |

### Column prefix convention

Every output column is prefixed with the lowercase session type so DataFrames
from different sessions merge without collision:

```
fp1_best_lap_delta           (FP1)
fp2_longrun_medium_avg_pace  (FP2)
sq_best_lap_delta            (Sprint Qualifying)
```

---

### Circuit baseline

| Function | Key output columns |
|---|---|
| `get_best_lap_delta(year, gp, session_type)` | `{p}_driver_best_laptime_s`, `{p}_field_best_laptime_s`, `{p}_best_lap_delta` |
| `get_clean_laps_count(year, gp, session_type)` | `{p}_clean_laps_count` |
| `get_compound_avg(year, gp, session_type, compound)` | `{p}_compound_{c}_avg`, `{p}_compound_{c}_laps`, `{p}_compound_{c}_best` |
| `get_sector_deltas(year, gp, session_type)` | `{p}_sector{1,2,3}_delta` |
| `get_max_speed_trap(year, gp, session_type)` | `{p}_max_speed_trap`, `{p}_speed_trap_avg`, `{p}_valid_laps_count`, `{p}_speed_trap_delta` |
| `get_laps_on_compound(year, gp, session_type, compound)` | `{p}_laps_on_{c}` |

`get_best_lap_delta` — `(driver_best - field_best) / field_best`. Zero for the
session leader.

`get_compound_avg` — degradation-corrected average lap time. Requires >= 3
clean laps on the compound. Constants: SOFT 0.12 s/lap, MEDIUM 0.08 s/lap,
HARD 0.05 s/lap.

`get_sector_deltas` — average sector time minus the session's fastest individual
sector time. All three sectors are computed in a single session load.

---

### Long run / race simulation

| Function | Key output columns |
|---|---|
| `get_longrun_avg_pace(year, gp, session_type, compound, window_size=8)` | `{p}_longrun_{c}_avg_pace` |
| `get_longrun_deg_rate(year, gp, session_type, compound, min_laps=5)` | `{p}_longrun_{c}_deg_rate` |
| `get_longrun_deg_total(year, gp, session_type, compound, min_laps=5, expected_stint_length=15)` | `{p}_longrun_{c}_deg_rate`, `{p}_longrun_{c}_deg_total` |
| `get_longrun_consistency(year, gp, session_type, compound, min_laps=5)` | `{p}_longrun_{c}_consistency` |
| `get_longrun_compound(year, gp, session_type, min_laps=5)` | `{p}_longrun_compound`, `{p}_longrun_stint_laps` |
| `get_fuel_corrected_pace(year, gp, session_type, compound, window_size=5)` | `{p}_{c}_fuel_corrected_pace`, `{p}_{c}_est_start_fuel_kg` |

**`get_longrun_avg_pace`** — best rolling `N` strictly consecutive laps within a
stint, degradation-corrected. Consecutiveness validated by
`max(LapNumber) - min(LapNumber) == N - 1` inside the rolling window.

**`get_longrun_deg_rate`** — `np.polyfit` slope (s/lap). Selected from the stint
with the most laps. Positive = tyre degrading, negative = fuel burn outpacing
tyre wear.

**`get_longrun_deg_total`** — extrapolated total time loss:
`deg_rate * expected_stint_length`.

**`get_longrun_consistency`** — standard deviation of degradation-corrected lap
times in the primary stint. Lower = more consistent.

**`get_fuel_corrected_pace`** — removes both tyre degradation and estimated fuel
load from each lap, then finds the best rolling 5-lap average. Fuel physics
constants: 1.5 kg/lap consumption, 5.0 kg safety margin, 0.03 s/kg time penalty.

---

### Power unit / ERS (2026 regulations)

These functions require `telemetry=True` and are meaningful only for 2026 data.
The 2026 regulations introduced ~350 kW electrical deployment (vs ~120 kW in
2025), making ERS management a primary differentiator. For pre-2026 rows they
return `NaN`; XGBoost handles missing values natively via Sparsity-Aware Split
Finding, learning a separate split direction for the missing-value branch.

| Function | Key output columns |
|---|---|
| `get_ers_efficiency_proxy(year, gp, session_type)` | `braking_time_s`, `full_throttle_time_s`, `{p}_ers_efficiency_proxy` |
| `get_lift_and_coast_laps(year, gp, session_type, min_laps=5)` | `{p}_avg_lift_coast_time_s` |
| `get_pu_deployment_asymmetry(year, gp, session_type)` | `{p}_best_s1_s`, `{p}_best_s3_s`, `{p}_s1_vs_s3_ratio`, `{p}_pu_asymmetry_delta` |
| `get_speed_trap_variance(year, gp, session_type, min_laps=5)` | `{p}_speed_trap_std_kmh`, `{p}_speed_trap_mean_kmh` |

**`get_ers_efficiency_proxy`** — deploy/recover ratio on the fastest lap, delta
to the field median. Deploy = seconds at throttle >= 95%; recover = seconds with
brake > 0. Positive = efficient, negative = clipping or poor recovery.

**`get_lift_and_coast_laps`** — average L&C time per lap during the primary long
run. L&C definition: `Speed > 250 km/h AND Throttle < 20% AND Brake == 0`.
High values indicate battery starvation or PU thermal management.

**`get_pu_deployment_asymmetry`** — S1/S3 ratio on the fastest lap vs field
median. Positive delta = saving battery for S3; negative = deploying aggressively
in S1 with S3 clipping risk.

**`get_speed_trap_variance`** — standard deviation of per-lap maximum speed
across the primary long run. Low variance = stable ERS map and predictable race
pace; high variance = clipping or heavy traffic.

---

### FP3 / setup tuning

| Function | Key output columns |
|---|---|
| `get_qualy_sim_delta(year, gp, session_type, compound)` | `{p}_{c}_best_lap_s`, `{p}_{c}_best_lap_delta` |
| `get_fp3_vs_fp2_improvement(year, gp, compound)` | `fp3_vs_fp2_{c}_improvement` |
| `get_sector_improvement_vs_fp2(year, gp, compound)` | `fp3_s{1,2,3}_delta_vs_fp2` |
| `get_fp3_qualy_sim_context(year, gp, session_type)` | `{p}_best_lap_compound`, `{p}_best_lap_tyre_age`, `{p}_is_true_qualy_sim` |
| `get_track_evolution(year, gp, session_type)` | `{p}_track_evolution_s` |

**`get_fp3_vs_fp2_improvement`** — `FP2_best - FP3_best`. Positive = car got
faster overnight (setup improvement + track evolution).

**`get_sector_improvement_vs_fp2`** — sector-by-sector overnight improvement.
Useful for detecting aero trade-offs (e.g., drag reduction improves S1/S3 at
the cost of S2).

**`get_fp3_qualy_sim_context`** — true qualifying simulation flag: compound must
be SOFT and tyre life <= 3 laps.

**`get_track_evolution`** — `best_early_lap - best_late_lap`, split at the
session midpoint by elapsed time. Broadcast as a constant to all drivers.

---

### Qualifying

| Function | Key output columns |
|---|---|
| `get_qualy_deltas(year, gp)` | `q3_time_s`, `q3_delta_to_pole`, `best_q_time_s`, `best_q_delta_to_pole`, `reached_session` |
| `get_q3_participation_flag(year, gp)` | `q3_participation` |
| `get_qualy_vs_fp3_improvement(year, gp)` | `best_q_vs_fp3_improvement` |
| `get_best_quali_relative_sector(year, gp)` | `quali_best_relative_sector`, `s{1,2,3}_delta` |

**`get_qualy_vs_fp3_improvement`** — `FP3_best - qualy_best`. A large positive
value indicates significant sandbagging in practice.

---

### Grid position — `get_quali_grid_position(year, gp)`

Race starting positions with fallback chain:

```
Q (Saturday qualifying) -> SQ (Sprint Qualifying) -> R (GridPosition column)
```

This chain handles sprint weekends correctly: the race grid comes from Saturday
Q, not Friday SQ. Session 3 is used for sprint detection
(`event.get_session_name(3) == "Sprint"`), not session 4 — session 4 is always
"Qualifying" regardless of weekend format.

Output columns: `grid_position`, `is_front_row`, `started_top_10`.

---

### Sprint features — `extract_sprint_features()`

| Column | Description |
|---|---|
| `sq_best_lap_delta` | SQ best lap delta to field leader |
| `sq_sector{1,2,3}_delta` | SQ sector deltas |
| `sq_speed_trap_delta` | SQ speed trap gap to fastest driver |
| `sq_vs_teammate` | SQ pace delta vs teammate |
| `s_finish_position` | Sprint race finishing position |
| `s_grid_position` | Sprint race starting position |
| `s_positions_gained` | `grid - finish` in the Sprint race |
| `s_classified` | Binary: finished classified |

---

### Teammate delta features — `add_teammate_deltas(df)`

For each driver, computes pace delta relative to their teammate in FP1 and SQ.
Eliminates car performance from the signal, isolating driver contribution:

```python
delta = driver_best_lap_delta - mean(teammate_best_lap_delta)
# Positive = faster than teammate, negative = slower
```

Output: `fp1_vs_teammate`, `sq_vs_teammate`. Applied to every row in training
and inference.

---

### Season form (rolling, no leakage)

Computed from prior 2026 races only. Injected in a second pass over the training
cache — each race row only sees results from rounds before its own.

| Function | Output columns |
|---|---|
| `get_season_form(labels_list)` | `recent_form_avg`, `recent_form_wins` |
| `get_team_pace_rank(labels_list, features_list)` | `team_2026_pace_rank` |

---

### Derived feature — ERS clipping penalty index

Assembled in `build_race_features()`:

```python
ers_clipping_penalty_index = abs(pu_asymmetry_delta) * abs(ers_efficiency_proxy)
```

Active only when **both** components are negative — the driver is deploying
aggressively in S1 (asymmetry < 0) and is ERS-inefficient overall (proxy < 0).
Zero otherwise. Captures combined S3 battery starvation and ERS clipping risk.

---

## Model architecture

### Ranker: `XGBRanker` with `rank:pairwise`

Each race is a **ranking group**. The model learns pairwise driver preferences
within each group, which is mathematically more appropriate for position
prediction than regression. Label encoding:

```
label = 22 - finish_position   (higher score = better finish)
P1  -> 21   P10 -> 12   DNF -> 0
```

DNF, crash, and Accident are all assigned `finish_position = 22` before
computing the label.

### Hyperparameters (shared by all models)

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 400 | Sufficient capacity for small datasets |
| `learning_rate` | 0.02 | Low rate for better generalisation |
| `max_depth` | 3 | Prevents overfitting with few training races |
| `subsample` | 0.8 | Row bagging per tree |
| `colsample_bytree` | 0.7 | Column bagging per tree |
| `min_child_weight` | 2 | Regularises leaf splits |
| `reg_alpha` | 2.0 | L1 sparsity |
| `reg_lambda` | 4.0 | L2 weight shrinkage |
| `gamma` | 0.1 | Minimum gain required to make a split |
| `random_state` | 42 | Reproducibility |

### Sample weights

`XGBRanker` with `group=` expects one weight per query group (one per race), not
one per driver row. `group_weights` has one entry per race in training order:

```python
model.fit(X_proc, y, group=groups, sample_weight=group_weights)
```

### Temporal decay for historical races

```
weight = exp(-1.5 * delta_years)
```

| Delta years | Weight |
|---|---|
| 0 (current season) | 1.00 |
| 1 | 0.22 |
| 2 | 0.05 |

Seasons 3+ years old are effectively noise and are not included.

### Sprint vs standard weekend detection

```python
is_sprint = event.get_session_name(3) == "Sprint"
```

Session 3 is "Sprint" on sprint weekends across all format variants (2023
sprint_shootout, 2024+ sprint_qualifying). Session 4 is always "Qualifying"
regardless of weekend format.

- Standard weekend: `FP1 + FP2 + FP3 + Q` features (SQ/S columns = NaN)
- Sprint weekend: `FP1 + SQ + S + Q` features (FP2/FP3 columns = NaN)

XGBoost handles the column mismatch natively via Sparsity-Aware Split Finding.

### Preprocessing

1. Categorical columns (e.g., `quali_best_relative_sector`) -> `pd.get_dummies(drop_first=True)`
2. All columns cast to `float`; non-numeric -> `NaN`
3. Sprint/FP2/FP3 mismatch -> `NaN` (handled natively by XGBoost)
4. At inference, training schema is enforced: any missing column is filled with `0.0`

---

## Prediction pipeline

Each race model follows the same phase structure:

```
Phase 0: Race discovery
  ├── FastF1 calendar -> all 2026 rounds before target GP
  ├── Circuit-specific historical entries (2024, 2025)
  └── Full historical seasons via discover_full_season() [if INCLUDE_FULL_HISTORICAL]

Phase 1: Build training set (3 passes)
  ├── Pass 1: extract features + labels for every race -> race_cache
  ├── Pass 2: inject rolling 2026 season form + team pace (no leakage)
  └── Pass 3: assemble df_train

Phase 1.5: Parameter calibration
  ├── Sprint models: calibrate_sigma() grid-searches (upgrade_sigma, sprint_sigma)
  └── Standard models: calibrate_params() grid-searches (upgrade_sigma, anchor_weight)

Phase 2: LOO-CV evaluation
  └── Held-out one race at a time -> NDCG + Spearman rho vs grid_position baseline

Phase 2.5: Ablation study
  └── Measures rho for: Base / +Upgrade / +Sprint (or +Anchor) / All combined

Phase 3: Build target race feature matrix (no labels — inference only)
  └── FP1 + (FP2/FP3 OR SQ/S) + Q + team features + season form

Phase 4: Train XGBRanker on full training set
  └── group=[n_drivers_per_race], sample_weight=[one_weight_per_race]

Phase 5: Predict + apply adjustment layers
  ├── Raw model score -> ranked list
  ├── + Upgrade adjustment (calibrated sigma)
  └── + Sprint adjustment (sprint weekends) OR grid-anchor blend (standard weekends)
```

---

## Post-prediction adjustment layers

Neither layer is a training feature — both encode race-week information that
only becomes available after the training dataset is frozen.

### 1. Upgrade adjustment

```python
adjusted_score = rank_score + upgrade_score * upgrade_sigma * score_std
```

Each team receives an `upgrade_score` (0-3) sourced from F1.com technical
reports and specialist press. The boost is in units of the field's score
standard deviation, so it scales with how spread the raw predictions are.

```
Scale: 0 = none  1 = minor/cooling  2 = significant floor/wing  3 = major package
```

`upgrade_sigma` defaults to 0.18 and is calibrated by `calibrate_sigma()`.

### 2a. Sprint race adjustment (sprint weekends only)

```python
sprint_z = (median_pos - sprint_pos) / median_pos   # range: +1 (P1) to -1 (last)
sprint_adjusted_score = adjusted_score + sprint_z * sprint_sigma * adjusted_std
```

The Sprint race result is the strongest available real-world pace signal on
sprint weekends. Applied after the upgrade adjustment.

| Sprint result | Approximate sprint_z |
|---|---|
| P1 | +0.91 |
| Median finisher | ~0 |
| Last / DNF | -1.0 |

`sprint_sigma` defaults to 0.20 and is calibrated by `calibrate_sigma()`.

### 2b. Grid-anchor blend (standard weekends — Monaco, Barcelona)

```python
grid_score = (n_drivers + 1) - grid_position   # P1 gets highest score
final_score = alpha * grid_score + (1 - alpha) * adjusted_score
```

Applied at circuits where the qualifying grid strongly determines the race
result and on-track overtaking is limited. `alpha` is calibrated by
`calibrate_params()`. Monaco uses a higher alpha (0.50) than Barcelona (0.30),
reflecting Monaco's near-zero overtaking probability.

---

## Sigma / parameter calibration

### `calibrate_sigma()` — sprint weekend models

Grid-searches 25 combinations of `(upgrade_sigma, sprint_sigma)` on completed
sprint weekends with known race results, maximising mean Spearman rho:

```
upgrade_sigmas = [0.10, 0.15, 0.18, 0.20, 0.25]
sprint_sigmas  = [0.10, 0.15, 0.20, 0.25, 0.30]
```

Sprint weekend calibration pool:

| Season | Weekends |
|---|---|
| 2026 | China, Miami (plus subsequent sprint rounds) |
| 2024 | China, Miami, Austria, Azerbaijan, USA, Sao Paulo, Qatar |
| 2023 | Austria, Belgium, Brazil, Qatar |

Fewer than 2 available sprint weekends with known results falls back to defaults
(`upgrade_sigma=0.18`, `sprint_sigma=0.20`).

### `calibrate_params()` — standard weekend models

Grid-searches combinations of `(upgrade_sigma, grid_anchor_weight)` on completed
standard-weekend races, maximising mean Spearman rho in a LOO loop.

---

## Ablation study — `ablation_study()`

Measures the incremental Spearman rho contribution of each adjustment layer on
completed 2026 races (held out one at a time):

| Variant | Description |
|---|---|
| Base only | Raw XGBRanker output |
| + Upgrade | Raw + upgrade adjustment |
| + Sprint/Anchor | Raw + sprint adjustment or grid-anchor only |
| All combined | Raw + upgrade + sprint/anchor |

Threshold: delta_rho > +0.01 => helps, delta_rho < -0.01 => hurts, otherwise neutral.

Canada 2026 reference (4 folds):

| Variant | Mean rho | vs Base |
|---|---|---|
| Base only | +0.553 | — |
| + Upgrade | +0.563 | +0.010 |
| + Sprint | +0.651 | +0.099 |
| All combined | +0.701 | +0.148 |

---

## LOO-CV evaluation

Leave-One-Out Cross-Validation is run before every final prediction.

For each race in the training set:
1. Remove it from training.
2. Train on the remaining races.
3. Predict the held-out race.
4. Score against actual finishing positions.

Metrics:
- **NDCG** (Normalised Discounted Cumulative Gain): ranking quality, 1.0 = perfect.
- **Spearman rho**: rank correlation, 1.0 = perfect, 0 = random.
- **Delta vs baseline**: NDCG gain over a naive `grid_position` predictor.

Canada 2026 (40 folds — 4 races 2026, Canada 2025, full 2024, partial 2023):

```
Model    NDCG: 0.950 +/- 0.042   delta vs baseline: -0.006
Baseline NDCG: 0.956 +/- 0.040  (grid_position only)
Spearman rho:  0.716 +/- 0.157
```

The model is within noise of the grid-position baseline on NDCG (a top-heavy
metric) but meaningfully better on full-field ordering (Spearman). Consistently
low-rho circuits such as Australia (~0.4) reflect inherent race unpredictability
(safety cars, attrition), not model failure.

---

## Power unit scoring — `team_mappings.json`

| Constructor | PU Score | Is Works |
|---|---|---|
| Mercedes | 5 | Yes |
| McLaren | 5 | No (customer) |
| Williams | 5 | No |
| Alpine | 5 | No |
| Haas | 4 | No |
| Ferrari | 3.5 | Yes |
| Red Bull Racing | 3 | Yes |
| Racing Bulls | 3 | No |
| Audi | 2 | Yes |
| Cadillac | 2 | No |
| Aston Martin | 1 | Yes |

Interaction term: `pu_is_works = pu_score * is_works`

Separates works teams (full PU access, free deployment maps) from customer teams
using the same hardware. For example, Mercedes works = 5, McLaren-Mercedes
customer = 0.

---

## Feature column reference

### FP1 (all weekends)

```
fp1_best_lap_delta, fp1_clean_laps_count,
fp1_sector{1,2,3}_delta, fp1_max_speed_trap, fp1_speed_trap_delta,
fp1_compound_medium_avg, fp1_laps_on_hard,
fp1_vs_teammate
```

### FP2 — race simulation (standard weekends only)

```
fp2_best_lap_delta, fp2_clean_laps_count,
fp2_sector{1,2,3}_delta, fp2_max_speed_trap,
fp2_longrun_medium_avg_pace, fp2_longrun_medium_deg_rate,
fp2_longrun_medium_deg_total, fp2_longrun_medium_consistency,
fp2_longrun_hard_avg_pace, fp2_longrun_hard_deg_rate,
fp2_medium_fuel_corrected_pace,
fp2_pu_asymmetry_delta,       (2026 only — NaN for historical rows)
fp2_speed_trap_std_kmh,       (2026 only)
fp2_avg_lift_coast_time_s,    (2026 only)
fp2_ers_efficiency_proxy      (2026 only)
```

### FP3 — final setup tune (standard weekends only)

```
fp3_best_lap_delta,
fp3_sector{1,2,3}_delta,
fp3_soft_best_lap_delta, fp3_vs_fp2_soft_improvement,
fp3_s{1,2,3}_delta_vs_fp2, fp3_is_true_qualy_sim,
fp3_track_evolution_s
```

### Sprint (sprint weekends only — NaN for standard weekends)

```
sq_best_lap_delta, sq_sector{1,2,3}_delta, sq_speed_trap_delta,
sq_vs_teammate,
s_finish_position, s_positions_gained, s_classified
```

### Qualifying (always available)

```
grid_position, is_front_row, started_top_10,
q3_delta_to_pole, best_q_delta_to_pole, q3_participation,
best_q_vs_fp3_improvement, quali_best_relative_sector
```

### Team / power unit

```
pu_score, is_works, pu_is_works
```

### Season form (2026 rolling — NaN for first race)

```
recent_form_avg, recent_form_wins, team_2026_pace_rank
```

### Engineered

```
ers_clipping_penalty_index
```

---

## Model summary

| Model | Circuit type | Weekend | Grid anchor | Sprint adj | Full historical |
|---|---|---|---|---|---|
| [Miami 2026](miami/README.md) | Permanent, moderate speed | Sprint | No | Yes (sigma=0.35) | No |
| [Canada 2026](canada/README.md) | Semi-street, heavy braking | Sprint | No | Yes (calibrated) | Yes |
| [Monaco 2026](monaco/README.md) | Street, ultra-low overtaking | Standard | Yes (alpha=0.50, calibrated) | No | Yes |
| [Barcelona 2026](barcelona/README.md) | Permanent, high-speed | Standard | Yes (alpha=0.30, calibrated) | No | Yes |

See each model's `README.md` for circuit profile, upgrade scores, pipeline
phases, and feature coverage at inference time.

---

## Dependencies

Managed with [uv](https://docs.astral.sh/uv/) (PEP 621).

| Package | Constraint | Role |
|---|---|---|
| Python | `>=3.13` | Pinned via `.python-version` |
| `fastf1` | `>=3.8.3` | Session data, timing, telemetry via OpenF1/Ergast |
| `xgboost` | `>=3.2.0` | `XGBRanker` pairwise ranking model |
| `lightgbm` | `>=4.6.0` | Gradient-boosted baseline for ablation |
| `scikit-learn` | `>=1.8.0` | Preprocessing, `ndcg_score` for LOO-CV |
| `scipy` | `>=1.17.1` | `spearmanr` for LOO-CV |
| `rich` | `>=15.0.0` | Terminal UI (progress bars, tables, panels) |

Full transitive dependency graph is pinned in `uv.lock`.
