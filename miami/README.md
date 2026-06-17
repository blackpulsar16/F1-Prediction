# Miami 2026 Grand Prix — Model Documentation

**File:** `miami/miami_model.py`
**Run:** `uv run python miami/miami_model.py` (from repo root)

---

## Circuit profile

Hard Rock Stadium, Miami Gardens. A permanent circuit built in 2022 around the
NFL stadium. Moderate-speed with a mix of high-speed sweepers and tight
technical sections.

```python
MIAMI_CIRCUIT_PROFILE = {
    "high_speed_ratio":      0.55,   # significant high-speed content
    "overtaking_difficulty": 0.45,   # DRS-assisted passes possible
    "brake_intensity":       0.65,   # moderate-heavy braking demands
    "pu_sensitivity":        0.70,   # long DRS zones reward PU output
    "street_circuit":        0,
    "traction_zones":        0.60,
}
```

---

## Weekend format

Sprint weekend: `FP1 -> SQ -> Sprint -> Q -> Race`

FP2 and FP3 do not exist. This has two consequences:

1. Long-run race simulation data (normally from FP2) is replaced by Sprint
   Qualifying and Sprint race telemetry, which is inherently shorter.
2. The sprint adjustment layer is the primary real-world pace signal and carries
   a higher sigma than models trained on more complete free practice data.

---

## Training data

| Source | GP | Weight | Rationale |
|---|---|---|---|
| 2026 season | Australia, China, Japan (discovered dynamically) | 1.00 | Same regs, same season |
| Miami 2025 | Miami | 0.22 | Same circuit, previous regs — temporal decay exp(-1.5 * 1) |
| Miami 2024 | Miami | 0.05 | Same circuit, older regs — exp(-1.5 * 2) |

`INCLUDE_FULL_HISTORICAL` is not enabled for Miami. The model uses only circuit-
specific historical races rather than full seasons. This keeps the feature matrix
smaller and avoids diluting the sprint-weekend signal with non-sprint historical
rows that would always have `NaN` in all SQ/S columns.

---

## Configuration constants

### Upgrade scores (Miami 2026)

Sourced from F1.com technical reports and specialist press (May 2026).
Applied as a post-prediction sigma adjustment — not a training feature.

| Team | Score | Technical summary |
|---|---|---|
| McLaren | 2 | Floor update + front wing revision |
| Ferrari | 2 | Sidepod and floor revision |
| Mercedes | 1 | Minor cooling and aero tweaks |
| Red Bull Racing | 1 | Floor edge update |
| Williams | 1 | Front wing and brake duct update |
| Racing Bulls | 1 | Sidepod revision |
| Alpine | 0 | No performance upgrade |
| Haas | 0 | No performance upgrade |
| Cadillac | 0 | No performance upgrade |
| Aston Martin | 0 | No performance upgrade |
| Audi | 0 | No performance upgrade |

```
Scale: 0 = none  1 = minor/cooling  2 = significant floor/wing  3 = major package
```

### Sigma constants

```python
UPGRADE_SIGMA_PER_POINT = 0.18   # not calibrated — no sprint weekends in training for phase 1.5
SPRINT_BOOST_SIGMA = 0.35        # higher than Canada (0.20) because FP2/FP3 data is absent
```

Miami does not run `calibrate_sigma()` because the training set contains no
completed sprint weekends with known results at the time of prediction. Both
sigmas are applied at their defaults.

---

## Pipeline phases

### Phase 0 — Race discovery

`discover_2026_races_before_target("Miami")` queries the 2026 FastF1 calendar
and returns all rounds with `RoundNumber < target_round`. Falls back to
`[Australia, China, Japan]` if the API is unavailable.

Historical Miami entries (2024, 2025) are appended manually from
`HISTORICAL_MIAMI`.

### Phase 1 — Training set assembly

For each race in the combined list:
- `build_race_features(year, gp)` extracts all session features via `f1_fe`.
- `get_race_labels(year, gp)` loads the race result and computes the label.
- Rows are merged on `Driver` and concatenated into `df_train`.

Sprint weekends in training: `build_race_features()` detects via
`event.get_session_name(4) == "Sprint"` (session 4, not session 3, to match
the FastF1 naming convention used here) and dispatches to either the FP2/FP3
path or the SQ/S path.

Season form (`recent_form_avg`, `recent_form_wins`, `team_2026_pace_rank`) is
injected in a rolling second pass over the 2026 race cache — each row only
sees results from races before its own round, preventing leakage.

### Phase 2 — LOO-CV

Leave-one-out cross-validation over all training races. Reports NDCG and
Spearman rho vs a naive `grid_position` baseline. No calibration or ablation
step runs here (insufficient sprint weekends with known results).

### Phase 3 — Target feature matrix

`build_race_features(2026, "Miami")` runs without labels. Session coverage:
`FP1 + SQ + Sprint + Q` (standard for a sprint weekend).

Sprint features (`s_finish_position`, `s_positions_gained`, `sq_*`) are
populated only after the Sprint race is complete on Saturday.

### Phase 4 — Train final model

`train_model(df_train, feature_cols)` trains on all available data including the
target race's prior 2026 rounds. Group sizes and per-group weights are derived
from the training set structure.

### Phase 5 — Predict and adjust

1. `predict_top()` — raw XGBRanker scores -> ranked driver list.
2. `apply_upgrade_adjustment()` — adds `upgrade_score * UPGRADE_SIGMA_PER_POINT * score_std`
   per team.
3. `apply_sprint_adjustment()` — adds sprint-z * `SPRINT_BOOST_SIGMA * adj_std` per driver
   based on their Sprint race finishing position.

---

## Feature coverage at inference

| Session | Status | Notes |
|---|---|---|
| FP1 | Always available | Baseline pace and tyre data |
| FP2 | Not applicable | Sprint weekend — no FP2 |
| FP3 | Not applicable | Sprint weekend — no FP3 |
| SQ | Available | Sprint Qualifying lap times and sectors |
| Sprint race | Available after Saturday | `s_finish_position`, `s_positions_gained` |
| Q | Available after Saturday | Grid position, delta to pole |

When run before the Sprint race, `s_finish_position` and `s_positions_gained`
will be `NaN` and the sprint adjustment step is skipped (no signal to apply).

---

## Known limitations

- No weather data. `session.load(..., weather=False)` is used throughout.
- The model has never seen a Miami race with 2026 regulations; all 2026 training
  data comes from different circuits.
- `INCLUDE_FULL_HISTORICAL = False` means the model has fewer than 10 training
  races. LOO-CV variance is high; individual fold rho values should be
  interpreted with caution.
