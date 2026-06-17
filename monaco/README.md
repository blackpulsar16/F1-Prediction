# Monaco 2026 Grand Prix — Model Documentation

**File:** `monaco/monaco_model.py`
**Run:** `uv run python monaco/monaco_model.py` (from repo root)

---

## Circuit profile

Circuit de Monaco, a street circuit winding through the Principality of Monaco.
The narrowest permanent circuit in F1: no room for side-by-side racing in most
corners, and overtaking is practically impossible except on pit strategy.

```python
MONACO_CIRCUIT_PROFILE = {
    "high_speed_ratio":      0.10,   # almost no high-speed sections
    "overtaking_difficulty": 0.95,   # effectively zero overtaking on track
    "brake_intensity":       0.70,   # heavy braking into Mirabeau, Portier, Nouvelle
    "pu_sensitivity":        0.20,   # short straights limit PU advantage
    "street_circuit":        1,
    "traction_zones":        0.90,   # slow-speed exits dominate laptime
    "active_aero_enabled":   0,      # Straight Mode banned by FIA at Monaco 2026
}
```

### 2026 regulation note — Active Aero disabled

The FIA banned Straight Mode (Active Aero) at Monaco 2026 on safety grounds:
the tunnel exit run-off, traction and braking zones, and the 3-second minimum
activation zone rule make it infeasible. Monaco is the first 2026 race where
all cars run in fixed Corner Mode (maximum downforce) for the entire weekend.

**Consequence for features:** Straight-line speed advantages and PU deployment
strategies that differentiate teams at other circuits (`fp2_ers_efficiency_proxy`,
`fp2_pu_asymmetry_delta`, `fp2_speed_trap_std_kmh`) are neutralised here.
`pu_sensitivity = 0.20` reflects this in the circuit profile, but the model
cannot explicitly zero out those features — it learns their reduced weight from
Monaco training data (2023, 2024, 2025 Monaco rows share similar characteristics).

---

## Grid-anchor blend

Monaco's extremely low overtaking probability means the qualifying grid order
is the dominant predictor of the race result. A pure XGBRanker score is
blended with an inverted grid-position score:

```python
grid_score = (n_drivers + 1) - grid_position   # P1 gets highest score
final_score = alpha * grid_score + (1 - alpha) * model_score
```

`GRID_ANCHOR_WEIGHT = 0.50` is the initial value. It is calibrated by
`calibrate_params()` (Phase 1.5) to find the alpha that maximises Spearman rho
on completed Monaco-like races in the training set.

The anchor is applied after the upgrade adjustment and before the final ranking.

---

## Weekend format

Standard weekend: `FP1 -> FP2 -> FP3 -> Q -> Race`

Full free practice coverage is available. FP2 long-run data is the most
valuable session for race pace characterisation, though at Monaco the correlation
between long-run pace and race result is weaker than at other circuits due to
the near-impossibility of overtaking.

---

## Training data

| Source | Races | Weight | Rationale |
|---|---|---|---|
| 2026 season | Prior 2026 rounds (dynamic) | 1.00 | Same regs, same season |
| Monaco 2025 | Monaco | 0.22 | Same circuit, previous regs |
| Monaco 2024 | Monaco | 0.05 | Same circuit, two seasons back |
| Full 2024 season | ~24 races | 0.22 | Regulation context; gated by `INCLUDE_FULL_HISTORICAL` |
| Full 2023 season | ~22 races | 0.05 | Weak historical prior; gated by `INCLUDE_FULL_HISTORICAL` |

`INCLUDE_FULL_HISTORICAL = True` by default.

The full historical seasons provide a larger training corpus that helps the
ranker learn general feature-to-position relationships. Monaco-specific rows
receive the same weight as any other circuit in those seasons, but the
historical Monaco entries (2024, 2025) additionally receive circuit-specific
weight multipliers.

---

## Configuration constants

### Upgrade scores (Monaco 2026)

Sourced from F1.com, The Race, and Motorsport.com Monaco 2026 technical reports.

Nearly every team fitted a mainplane winglet in the volume previously used by
the Straight Mode actuator — this is a field-wide adaptation to the Active Aero
ban rather than a genuine downforce gain. Scoring differentiates by real
downforce additions and Monaco-specific mechanical gains (e.g., increased
steering lock).

| Team | Score | Technical summary |
|---|---|---|
| McLaren | 2 | Front suspension for steering lock, beam wing, diffuser stay |
| Ferrari | 2 | Floor + diffuser aero-load devices (real downforce gain) |
| Red Bull Racing | 1 | 4 items, mostly reliability / thermal management |
| Audi | 1 | 4 items, no-active-aero adaptation + cooling |
| Mercedes | 1 | Rear-wing winglet (local downforce) |
| Aston Martin | 1 | Exhaust wing, front suspension steer range, cooling louvres |
| Haas | 1 | Winglet, front suspension, exhaust wing |
| Racing Bulls | 1 | Front suspension + rear-wing central winglet |
| Williams | 1 | Exhaust tailpipe step + front suspension tweak |
| Alpine | 0 | Winglet only — field-wide adaptation, no performance gain |
| Cadillac | 0 | Winglet only |

```
Scale: 0 = none/winglet-only  1 = minor  2 = significant  3 = major package
```

### Calibrated constants

```python
GRID_ANCHOR_WEIGHT = 0.50        # initial; overwritten by calibrate_params()
UPGRADE_SIGMA_PER_POINT = 0.18   # initial; overwritten by calibrate_params()
```

Both are calibrated jointly by `calibrate_params()` in Phase 1.5.

---

## Pipeline phases

### Phase 0 — Race discovery

`discover_2026_races_before_target("Monaco")` queries the 2026 FastF1 calendar.
Historical Monaco entries and full historical seasons are appended from
`HISTORICAL_MONACO` and `FULL_HISTORICAL_YEARS`.

### Phase 1 — Training set assembly (3 passes)

Same three-pass structure as Canada: feature extraction, rolling season form
injection (no leakage), and final concatenation into `df_train`.

### Phase 1.5 — Parameter calibration

`calibrate_params(race_cache, feature_cols)` grid-searches combinations of
`(upgrade_sigma, grid_anchor_weight)` on completed 2026 standard-weekend races
and all available historical races, maximising mean Spearman rho in a LOO loop.

The grid anchor is specific to Monaco because it captures the structural
reality that position changes are rare. It is not applied in sprint-weekend
models (Canada, Miami) where the race is more dynamic.

### Phase 2 — LOO-CV

Leave-one-out cross-validation over all training races. Reports NDCG and
Spearman rho vs `grid_position` baseline.

### Phase 2.5 — Ablation study

Measures the incremental Spearman rho contribution of:
- Base XGBRanker output
- Base + upgrade adjustment
- Base + grid anchor
- All combined

### Phase 3 — Target feature matrix

`build_race_features(2026, "Monaco")` extracts inference features from
`FP1 + FP2 + FP3 + Q`. All sessions are available for a standard weekend.

### Phase 4 — Train final model

`train_model(df_train, feature_cols)` trains on all data.

### Phase 5 — Predict and adjust

1. `predict_top()` — raw XGBRanker scores.
2. Upgrade adjustment with calibrated `best_upgrade_sigma`.
3. Grid-anchor blend: final score = `best_alpha * grid_score + (1 - best_alpha) * adjusted_score`.

The grid-anchor blend step replaces the sprint adjustment found in Canada and
Miami models. There is no sprint race at Monaco.

---

## Feature coverage at inference

| Session | Status | Notes |
|---|---|---|
| FP1 | Available | Baseline pace, tyre data |
| FP2 | Available | Long-run race simulation (primary pace signal) |
| FP3 | Available | Final setup tune, qualy-sim delta |
| SQ | Not applicable | Standard weekend |
| Sprint race | Not applicable | Standard weekend |
| Q | Available | Grid position, delta to pole |

Monaco has full session coverage. The ERS and PU deployment features
(`fp2_ers_efficiency_proxy`, `fp2_pu_asymmetry_delta`, `fp2_speed_trap_std_kmh`,
`fp2_avg_lift_coast_time_s`) are populated but carry less predictive weight than
at power-unit-sensitive circuits, given the Active Aero ban.

---

## Known limitations

- The Active Aero ban is encoded only in the circuit profile constant
  (`active_aero_enabled: 0`); it is not explicitly zeroed in the feature matrix.
  The model relies on learning from historical Monaco data (where PU deployment
  features also had low importance) to downweight those columns automatically.
- Overtaking probability is effectively zero in dry conditions. If a safety car
  creates a free pit-stop window, the qualifying grid order — which the anchor
  weights heavily — can become misleading.
- No weather data. Rain at Monaco produces extreme result shuffles that no
  telemetry-based model can anticipate.
