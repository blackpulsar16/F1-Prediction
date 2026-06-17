"""Monaco 2026 Grand Prix race-result prediction model.

Circuit: Circuit de Monaco, a tight street circuit with almost no straights
where overtaking is near-impossible, so the qualifying grid dominates the race
result. The prediction is therefore grid-anchored: model scores are blended
with grid position via a calibrated anchor weight (alpha).

2026 regulation note: Active Aero (Straight Mode) is disabled at Monaco by the
FIA on safety grounds. With no Straight Mode, top-speed and PU-deployment
advantages stop differentiating the field, so straight-line features are
neutralised for this race.

Weekend format: standard weekend (FP1 -> FP2 -> FP3 -> Q -> Race).

Training data:
    - 2026 season races before Monaco, weight 1.00.
    - Monaco 2025 / 2024 as circuit-specific priors (weights 0.22 / 0.05).
    - Full 2024 / 2023 seasons for regulation context (weights 0.22 / 0.05),
      gated by INCLUDE_FULL_HISTORICAL.

Pipeline:
    1. Discover prior 2026 races and full historical seasons via the FastF1
       calendar.
    2. Build the training set, inject rolling 2026 season form / team pace
       (no leakage), and assemble df_train.
    3. Calibrate the upgrade sigma and the grid-anchor weight.
    4. Evaluate with leave-one-out CV and run the adjustment-layer ablation.
    5. Train an XGBRanker (rank:pairwise), predict the target race, apply the
       upgrade adjustment, and blend with the grid anchor.
"""

import contextlib
import io
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib
import json
import logging
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import ndcg_score as sklearn_ndcg_score
from xgboost import XGBRanker
from rich.console import Console
from rich.progress import (
    Progress, SpinnerColumn, TextColumn,
    BarColumn, MofNCompleteColumn, TimeElapsedColumn,
)
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

warnings.filterwarnings("ignore")

for _name in ["fastf1", "fastf1.core", "fastf1.api", "fastf1.ergast",
              "urllib3", "requests", "httpx"]:
    logging.getLogger(_name).setLevel(logging.CRITICAL)
os.environ["TQDM_DISABLE"] = "1"

console = Console(
    file=getattr(sys, "__stdout__", sys.stdout),
    force_terminal=True,
    highlight=False,
)

_spec = importlib.util.spec_from_file_location(
    "f1_fe",
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "f1-api", "f1_fe.py")
    ),
)
f1_fe = importlib.util.module_from_spec(_spec)
sys.modules["f1_fe"] = f1_fe
_spec.loader.exec_module(f1_fe)

import fastf1
fastf1.Cache.enable_cache("./f1_cache")

with open("team_mappings.json", "r") as f:
    MAPS = json.load(f)

PU_MAP    = MAPS["PU_MAP"]
WORKS_MAP = MAPS["WORKS_MAP"]

HISTORICAL_MONACO = [
    {"year": 2025, "gp": "Monaco", "weight": 0.22},
    {"year": 2024, "gp": "Monaco", "weight": 0.05},
]

FULL_HISTORICAL_YEARS = [
    {"year": 2024, "weight": 0.22},
    {"year": 2023, "weight": 0.05},
]
INCLUDE_FULL_HISTORICAL = True

TARGET_RACE = {"year": 2026, "gp": "Monaco"}

MONACO_CIRCUIT_PROFILE = {
    "high_speed_ratio":      0.10,
    "overtaking_difficulty": 0.95,
    "brake_intensity":       0.70,
    "pu_sensitivity":        0.20,
    "street_circuit":        1,
    "traction_zones":        0.90,
    "active_aero_enabled":   0,
}

UPGRADE_SCORE = {
    "McLaren":         2,
    "Ferrari":         2,
    "Red Bull Racing": 1,
    "Audi":            1,
    "Mercedes":        1,
    "Aston Martin":    1,
    "Haas":            1,
    "Racing Bulls":    1,
    "Williams":        1,
    "Alpine":          0,
    "Cadillac":        0,
}
UPGRADE_SIGMA_PER_POINT = 0.18

UPGRADE_DESC = {
    3: "Major rebuild — floor+chassis+wings+bodywork",
    2: "Significant — floor / front wing / mechanical",
    1: "Minor — cooling / aero tweaks / winglet",
    0: "Winglet only — no performance gain",
}

TEAM_NORM_MAP = {
    "Mercedes-AMG Petronas F1 Team": "Mercedes",
    "McLaren F1 Team":               "McLaren",
    "Scuderia Ferrari":              "Ferrari",
    "Williams Racing":               "Williams",
    "BWT Alpine F1 Team":            "Alpine",
    "Aston Martin Aramco F1 Team":   "Aston Martin",
    "Oracle Red Bull Racing":        "Red Bull Racing",
    "Visa Cash App RB F1 Team":      "Racing Bulls",
    "MoneyGram Haas F1 Team":        "Haas",
    "Audi F1 Team":                  "Audi",
    "Cadillac F1 Team":              "Cadillac",
    "Kick Sauber":                   "Audi",
    "AlphaTauri":                    "Racing Bulls",
    "Alfa Romeo":                    "Audi",
    "Aston Martin Aramco Mercedes":  "Aston Martin",
    "Haas F1 Team":                  "Haas",
    "Scuderia AlphaTauri":           "Racing Bulls",
}

GRID_ANCHOR_WEIGHT = 0.50


@contextlib.contextmanager
def silent():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def discover_2026_races_before_target(
    target_gp: str, fallback: list[dict] = None
) -> list[dict]:
    try:
        schedule = fastf1.get_event_schedule(2026, include_testing=False)
        target_mask = schedule["EventName"].str.contains(target_gp, case=False, na=False)
        if not target_mask.any():
            target_mask = schedule["Country"].str.contains(target_gp, case=False, na=False)
        if not target_mask.any():
            raise ValueError(f"'{target_gp}' not found in the 2026 calendar")
        target_round = schedule.loc[target_mask, "RoundNumber"].iloc[0]
        prior = schedule[schedule["RoundNumber"] < target_round]
        return [{"year": 2026, "gp": row["EventName"], "weight": 1.0}
                for _, row in prior.iterrows()]
    except Exception as exc:
        print(f"Discovery failed ({exc}) — using fallback")
        return fallback or []


def discover_full_season(year: int, weight: float) -> list[dict]:
    """All completed race weekends for a given historical season."""
    try:
        schedule = fastf1.get_event_schedule(year, include_testing=False)
        return [
            {"year": year, "gp": row["EventName"], "weight": weight}
            for _, row in schedule.iterrows()
        ]
    except Exception:
        return []


def safe_merge(base: pd.DataFrame, df: pd.DataFrame, on: str = "Driver") -> pd.DataFrame:
    if df is None or df.empty:
        return base
    cols = [c for c in df.columns if c not in base.columns or c == on]
    return base.merge(df[cols], on=on, how="left")


def get_base_drivers(year: int, gp: str) -> pd.DataFrame:
    for session_type in ["Q", "SQ", "R", "S"]:
        try:
            s = fastf1.get_session(year, gp, session_type)
            s.load(telemetry=False, weather=False, messages=False)
            drivers = s.results["Abbreviation"].tolist()
            if drivers:
                return pd.DataFrame({"Driver": drivers, "year": year, "gp": gp})
        except Exception:
            continue
    return pd.DataFrame()


def get_quali_grid_position(year: int, gp: str) -> pd.DataFrame:
    """Grid positions from Q qualifying → SQ → R fallback.

    For sprint weekends the race grid comes from Saturday Q, not Friday SQ.
    """
    for stype in ["Q", "SQ"]:
        try:
            s = fastf1.get_session(year, gp, stype)
            s.load(telemetry=False, weather=False, messages=False)
            res = s.results[["Abbreviation", "Position"]].copy()
            res.columns = ["Driver", "grid_position"]
            res["grid_position"] = pd.to_numeric(res["grid_position"], errors="coerce")
            if res["grid_position"].notna().any():
                res["is_front_row"]   = (res["grid_position"] <= 2).astype(int)
                res["started_top_10"] = (res["grid_position"] <= 10).astype(int)
                return res
        except Exception:
            continue
    try:
        s = fastf1.get_session(year, gp, "R")
        s.load(telemetry=False, weather=False, messages=False)
        res = s.results[["Abbreviation", "GridPosition"]].copy()
        res.columns = ["Driver", "grid_position"]
        res["grid_position"] = pd.to_numeric(res["grid_position"], errors="coerce")
        res["is_front_row"]   = (res["grid_position"] <= 2).astype(int)
        res["started_top_10"] = (res["grid_position"] <= 10).astype(int)
        return res
    except Exception:
        return pd.DataFrame()


def get_season_form(labels_list: list[pd.DataFrame]) -> pd.DataFrame:
    """Driver season form: avg finish position and wins across supplied races."""
    if not labels_list:
        return pd.DataFrame()
    combined = pd.concat(labels_list, ignore_index=True)
    return (
        combined.groupby("Driver")["finish_pos"]
        .agg(
            recent_form_avg="mean",
            recent_form_wins=lambda x: int((x == 1).sum()),
        )
        .reset_index()
    )


def get_team_pace_rank(
    labels_list: list[pd.DataFrame],
    features_list: list[pd.DataFrame],
) -> pd.DataFrame:
    """Team average finishing position keyed on Team_norm."""
    if not labels_list or not features_list:
        return pd.DataFrame()
    driver_team_dfs = [
        f[["Driver", "Team"]].copy()
        for f in features_list if "Team" in f.columns
    ]
    if not driver_team_dfs:
        return pd.DataFrame()
    driver_team = (
        pd.concat(driver_team_dfs, ignore_index=True)
        .drop_duplicates("Driver", keep="last")
    )
    driver_team["Team_norm"] = driver_team["Team"].replace(TEAM_NORM_MAP).fillna(driver_team["Team"])
    combined = (
        pd.concat(labels_list, ignore_index=True)
        .merge(driver_team[["Driver", "Team_norm"]], on="Driver", how="left")
    )
    return (
        combined.groupby("Team_norm")["finish_pos"].mean()
        .reset_index()
        .rename(columns={"finish_pos": "team_2026_pace_rank"})
    )


def add_teammate_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Add pace delta vs teammate for FP1 and SQ sessions.

    Positive = faster than teammate, negative = slower.
    Removes car performance bias, isolating driver contribution.
    """
    df = df.copy()
    for prefix in ["fp1", "sq"]:
        src = f"{prefix}_best_lap_delta"
        dst = f"{prefix}_vs_teammate"
        df[dst] = np.nan
        if src not in df.columns or "Team" not in df.columns:
            continue
        for _, group in df.groupby("Team"):
            if len(group) < 2:
                continue
            for idx in group.index:
                my_val = df.at[idx, src]
                others = group.loc[group.index != idx, src].dropna()
                if pd.notna(my_val) and len(others) > 0:
                    df.at[idx, dst] = float(my_val) - float(others.mean())
    return df


def extract_fp1_features(year: int, gp: str) -> pd.DataFrame:
    dfs = []
    for fn, kwargs in [
        (f1_fe.get_best_lap_delta,   {"session_type": "FP1"}),
        (f1_fe.get_clean_laps_count, {"session_type": "FP1"}),
        (f1_fe.get_sector_deltas,    {"session_type": "FP1"}),
        (f1_fe.get_max_speed_trap,   {"session_type": "FP1"}),
        (f1_fe.get_compound_avg,     {"session_type": "FP1", "compound": "MEDIUM"}),
        (f1_fe.get_laps_on_compound, {"session_type": "FP1", "compound": "HARD"}),
    ]:
        try:
            dfs.append(fn(year, gp, **kwargs))
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()
    result = dfs[0]
    for df in dfs[1:]:
        result = safe_merge(result, df)
    return result


def extract_fp2_features(year: int, gp: str) -> pd.DataFrame:
    dfs = []
    for fn, kwargs in [
        (f1_fe.get_best_lap_delta,   {"session_type": "FP2"}),
        (f1_fe.get_clean_laps_count, {"session_type": "FP2"}),
        (f1_fe.get_sector_deltas,    {"session_type": "FP2"}),
        (f1_fe.get_max_speed_trap,   {"session_type": "FP2"}),
    ]:
        try:
            dfs.append(fn(year, gp, **kwargs))
        except Exception:
            pass
    for compound in ["MEDIUM", "HARD"]:
        for fn in [
            f1_fe.get_longrun_avg_pace,
            f1_fe.get_longrun_deg_rate,
            f1_fe.get_longrun_deg_total,
            f1_fe.get_longrun_consistency,
        ]:
            try:
                dfs.append(fn(year, gp, "FP2", compound=compound))
            except Exception:
                pass
    for fn, kwargs in [
        (f1_fe.get_fuel_corrected_pace,     {"session_type": "FP2", "compound": "MEDIUM"}),
        (f1_fe.get_pu_deployment_asymmetry, {"session_type": "FP2"}),
        (f1_fe.get_speed_trap_variance,     {"session_type": "FP2"}),
        (f1_fe.get_lift_and_coast_laps,     {"session_type": "FP2"}),
        (f1_fe.get_ers_efficiency_proxy,    {"session_type": "FP2"}),
    ]:
        try:
            dfs.append(fn(year, gp, **kwargs))
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()
    result = dfs[0]
    for df in dfs[1:]:
        result = safe_merge(result, df)
    return result


def extract_fp3_features(year: int, gp: str) -> pd.DataFrame:
    dfs = []
    for fn, kwargs in [
        (f1_fe.get_best_lap_delta,            {"session_type": "FP3"}),
        (f1_fe.get_clean_laps_count,          {"session_type": "FP3"}),
        (f1_fe.get_sector_deltas,             {"session_type": "FP3"}),
        (f1_fe.get_max_speed_trap,            {"session_type": "FP3"}),
        (f1_fe.get_qualy_sim_delta,           {"session_type": "FP3", "compound": "SOFT"}),
        (f1_fe.get_fp3_vs_fp2_improvement,    {}),
        (f1_fe.get_sector_improvement_vs_fp2, {}),
        (f1_fe.get_fp3_qualy_sim_context,     {"session_type": "FP3"}),
        (f1_fe.get_track_evolution,           {"session_type": "FP3"}),
    ]:
        try:
            dfs.append(fn(year, gp, **kwargs))
        except Exception:
            pass
    if not dfs:
        return pd.DataFrame()
    result = dfs[0]
    for df in dfs[1:]:
        result = safe_merge(result, df)
    return result


def extract_sprint_features(year: int, gp: str) -> pd.DataFrame:
    dfs = []
    for fn, kwargs in [
        (f1_fe.get_best_lap_delta,   {"session_type": "SQ"}),
        (f1_fe.get_sector_deltas,    {"session_type": "SQ"}),
        (f1_fe.get_max_speed_trap,   {"session_type": "SQ"}),
        (f1_fe.get_clean_laps_count, {"session_type": "SQ"}),
    ]:
        try:
            dfs.append(fn(year, gp, **kwargs))
        except Exception:
            pass
    try:
        session_s = fastf1.get_session(year, gp, "S")
        session_s.load(telemetry=False, weather=False, messages=False)
        res = session_s.results
        dfs.append(pd.DataFrame({
            "Driver":             res["Abbreviation"],
            "s_finish_position":  pd.to_numeric(res["Position"], errors="coerce"),
            "s_grid_position":    pd.to_numeric(res["GridPosition"], errors="coerce"),
            "s_positions_gained": (
                pd.to_numeric(res["GridPosition"], errors="coerce")
                - pd.to_numeric(res["Position"], errors="coerce")
            ),
            "s_classified": res["Status"].str.contains("Finished|Lap", na=False).astype(int),
        }))
    except Exception:
        pass
    if not dfs:
        return pd.DataFrame()
    result = dfs[0]
    for df in dfs[1:]:
        result = safe_merge(result, df)
    return result


def extract_quali_features(year: int, gp: str) -> pd.DataFrame:
    dfs = []
    for fn, kwargs in [
        (f1_fe.get_qualy_deltas,               {}),
        (f1_fe.get_q3_participation_flag,      {}),
        (f1_fe.get_qualy_vs_fp3_improvement,   {}),
        (f1_fe.get_best_quali_relative_sector, {}),
    ]:
        try:
            dfs.append(fn(year, gp, **kwargs))
        except Exception:
            pass
    grid_df = get_quali_grid_position(year, gp)
    if not grid_df.empty:
        dfs.append(grid_df)
    if not dfs:
        return pd.DataFrame()
    result = dfs[0]
    for df in dfs[1:]:
        result = safe_merge(result, df)
    return result


def extract_team_features(year: int, gp: str) -> pd.DataFrame:
    try:
        df_team = f1_fe.get_team_info(year, gp, "Q")
        df_team["Team_norm"]   = df_team["Team"].replace(TEAM_NORM_MAP)
        df_team["pu_score"]    = df_team["Team_norm"].map(PU_MAP).fillna(2).astype(float)
        df_team["is_works"]    = df_team["Team_norm"].map(WORKS_MAP).fillna(0).astype(float)
        df_team["pu_is_works"] = df_team["pu_score"] * df_team["is_works"]
        return df_team[["Driver", "Team", "pu_score", "is_works", "pu_is_works"]]
    except Exception:
        return pd.DataFrame()


def add_circuit_context(df: pd.DataFrame) -> pd.DataFrame:
    for key, val in MONACO_CIRCUIT_PROFILE.items():
        df[f"circuit_{key}"] = val
    return df


def build_race_features(
    year: int, gp: str, include_circuit_context: bool = False
) -> pd.DataFrame:
    try:
        event    = fastf1.get_event(year, gp)
        is_sprint = event.get_session_name(3) == "Sprint"
    except Exception:
        is_sprint = False

    base = get_base_drivers(year, gp)
    if base.empty:
        return pd.DataFrame()

    base = safe_merge(base, extract_fp1_features(year, gp))

    if is_sprint:
        base = safe_merge(base, extract_sprint_features(year, gp))
    else:
        base = safe_merge(base, extract_fp2_features(year, gp))
        base = safe_merge(base, extract_fp3_features(year, gp))

    base = safe_merge(base, extract_quali_features(year, gp))
    base = safe_merge(base, extract_team_features(year, gp))

    if include_circuit_context:
        base = add_circuit_context(base)

    if "fp2_ers_efficiency_proxy" in base.columns and "fp2_pu_asymmetry_delta" in base.columns:
        ar = np.where(base["fp2_pu_asymmetry_delta"] < 0, np.abs(base["fp2_pu_asymmetry_delta"]), 0)
        ir = np.where(base["fp2_ers_efficiency_proxy"] < 0, np.abs(base["fp2_ers_efficiency_proxy"]), 0)
        base["ers_clipping_penalty_index"] = ar * ir
    else:
        base["ers_clipping_penalty_index"] = np.nan

    base = add_teammate_deltas(base)

    return base


def get_race_labels(year: int, gp: str) -> pd.DataFrame:
    try:
        session = fastf1.get_session(year, gp, "R")
        session.load(telemetry=False, weather=False, messages=False)
        res = session.results
        df = pd.DataFrame({
            "Driver":     res["Abbreviation"],
            "finish_pos": pd.to_numeric(res["Position"], errors="coerce").fillna(22),
            "status":     res["Status"],
        })
        dnf_mask = ~df["status"].str.contains("Finished|Lap", na=False)
        df.loc[dnf_mask, "finish_pos"] = 22
        df["label"] = (22 - df["finish_pos"]).clip(lower=0).astype(int)
        return df[["Driver", "finish_pos", "label"]]
    except Exception:
        return pd.DataFrame()


def preprocess(
    df: pd.DataFrame,
    feature_cols: list[str],
    known_cols: list[str] = None,
) -> tuple[pd.DataFrame, list[str]]:
    cat_cols = [c for c in feature_cols if c in df.columns and df[c].dtype == object]
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan
    df_enc = pd.get_dummies(df[feature_cols], columns=cat_cols, drop_first=True)
    df_enc = df_enc.apply(pd.to_numeric, errors="coerce")
    df_enc = df_enc.select_dtypes(include=[np.number])
    if known_cols is not None:
        for col in known_cols:
            if col not in df_enc.columns:
                df_enc[col] = 0.0
        df_enc = df_enc[known_cols]
    final_cols = df_enc.columns.tolist()
    return pd.DataFrame(df_enc.to_numpy(dtype=float), columns=final_cols), final_cols


def train_model(df_train: pd.DataFrame, feature_cols: list[str], n_jobs: int = -1):
    df_train = df_train.sort_values(["year", "gp"]).reset_index(drop=True)
    X_raw = df_train[feature_cols]
    y     = df_train["label"].values.astype(int)
    X_proc, final_cols = preprocess(X_raw, feature_cols)

    race_order    = df_train[["year", "gp"]].drop_duplicates()
    groups        = []
    group_weights = []
    for _, row in race_order.iterrows():
        mask = (df_train["year"] == row["year"]) & (df_train["gp"] == row["gp"])
        groups.append(mask.sum())
        group_weights.append(df_train.loc[mask, "sample_weight"].iloc[0])

    model = XGBRanker(
        objective="rank:pairwise",
        n_estimators=400,
        learning_rate=0.02,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=2,
        reg_alpha=2.0,
        reg_lambda=4.0,
        gamma=0.1,
        random_state=42,
        verbosity=0,
        n_jobs=n_jobs,
    )
    model.fit(X_proc, y, group=groups, sample_weight=group_weights)
    return model, final_cols


def _train_predict_fold(
    df_cv_train: pd.DataFrame, feature_cols: list[str], df_target: pd.DataFrame,
) -> pd.DataFrame | None:
    """Train + predict a single CV fold — runs inside a worker process.

    Pinned to n_jobs=1: the caller fans many of these out across CPU cores
    at once, so each individual XGBRanker must not also spawn its own thread
    pool (that would oversubscribe the machine and slow everything down).
    """
    try:
        with silent():
            model, cols = train_model(df_cv_train, feature_cols, n_jobs=1)
            return predict_top(model, cols, df_target, feature_cols)
    except Exception:
        return None


def predict_top(
    model, final_cols: list[str], df_target: pd.DataFrame, feature_cols: list[str],
) -> pd.DataFrame:
    df_inf = df_target.copy()
    for col in feature_cols:
        if col not in df_inf.columns:
            df_inf[col] = np.nan
    X_raw = df_inf[feature_cols]
    X_proc, _ = preprocess(X_raw, feature_cols, known_cols=final_cols)
    scores = model.predict(X_proc.values)
    df_result = df_target[["Driver"]].copy()
    if "Team" in df_target.columns:
        df_result["Team"] = df_target["Team"].values
    if "grid_position" in df_target.columns:
        df_result["grid_position"] = df_target["grid_position"].values
    df_result["rank_score"] = scores
    df_result = df_result.sort_values("rank_score", ascending=False).reset_index(drop=True)
    df_result["predicted_position"] = df_result.index + 1
    return df_result.head(22)


def apply_upgrade_adjustment(
    df_pred: pd.DataFrame,
    upgrade_sigma: float | None = None,
) -> pd.DataFrame:
    sigma = upgrade_sigma if upgrade_sigma is not None else UPGRADE_SIGMA_PER_POINT
    df = df_pred.copy()
    df["Team_norm"]     = df["Team"].replace(TEAM_NORM_MAP).fillna(df["Team"])
    df["upgrade_score"] = df["Team_norm"].map(UPGRADE_SCORE).fillna(0)
    score_std            = df["rank_score"].std()
    df["upgrade_boost"]  = df["upgrade_score"] * sigma * score_std
    df["adjusted_score"] = df["rank_score"] + df["upgrade_boost"]
    df = df.sort_values("adjusted_score", ascending=False).reset_index(drop=True)
    df["adjusted_position"] = df.index + 1
    return df


def apply_grid_anchor(
    df_pred: pd.DataFrame,
    year: int,
    gp: str,
    anchor_weight: float | None = None,
) -> pd.DataFrame:
    """Blend the model's order with the qualifying grid order.

    Monaco is the hardest track on the calendar to overtake on, and with
    Straight Mode (active aero) banned in 2026 there is even less of a top-speed
    delta to make a pass — track position from qualifying dominates the race
    result. We standardise both signals and blend:

        anchored = (1 - α)·z(model_score) + α·z(grid)

    α = 0 → pure model · α = 1 → pure grid (qualifying order).
    """
    alpha = anchor_weight if anchor_weight is not None else GRID_ANCHOR_WEIGHT
    df = df_pred.copy()

    score_col = "adjusted_score" if "adjusted_score" in df.columns else "rank_score"

    if "grid_position" not in df.columns or df["grid_position"].isna().all():
        grid = get_quali_grid_position(year, gp)
        if not grid.empty:
            df = df.merge(grid[["Driver", "grid_position"]], on="Driver", how="left",
                          suffixes=("", "_g"))
            if "grid_position_g" in df.columns:
                df["grid_position"] = df["grid_position"].fillna(df["grid_position_g"])
                df.drop(columns=["grid_position_g"], inplace=True)

    n     = len(df)
    s     = df[score_col].astype(float)
    s_z   = (s - s.mean()) / (s.std() + 1e-9)

    grid  = pd.to_numeric(df.get("grid_position"), errors="coerce")
    median_pos = (n + 1) / 2
    grid_z = (median_pos - grid) / median_pos
    grid_z = grid_z.fillna(0.0)

    df["grid_z"]          = grid_z
    df["anchored_score"]  = (1.0 - alpha) * s_z + alpha * grid_z
    df = df.sort_values("anchored_score", ascending=False).reset_index(drop=True)
    df["anchored_position"] = df.index + 1
    return df


def calibrate_params(
    race_cache: list[dict],
    feature_cols: list[str],
) -> tuple[float, float]:
    """Grid-search upgrade_sigma and grid-anchor α on completed 2026 races.

    LOO-CV over every completed 2026 race (sprint or conventional) that has a
    qualifying grid, maximising mean Spearman ρ between the grid-anchored
    prediction and the actual finishing order.
    Returns (best_upgrade_sigma, best_anchor_weight).
    """
    test_races = [
        e for e in race_cache
        if e["race"]["year"] == 2026
        and not e["labels"].empty
        and "grid_position" in e["features"].columns
        and e["race"]["gp"] != TARGET_RACE["gp"]
    ]

    if len(test_races) < 2:
        console.print(
            f"  [yellow]Only {len(test_races)} calibration race(s) — using defaults "
            f"(σ={UPGRADE_SIGMA_PER_POINT:.2f}, α={GRID_ANCHOR_WEIGHT:.2f}).[/yellow]"
        )
        return UPGRADE_SIGMA_PER_POINT, GRID_ANCHOR_WEIGHT

    upgrade_sigmas = [0.10, 0.15, 0.18, 0.20, 0.25]
    anchor_weights = [0.00, 0.25, 0.40, 0.55, 0.70, 0.85]
    best_rho       = -np.inf
    best_u, best_a = UPGRADE_SIGMA_PER_POINT, GRID_ANCHOR_WEIGHT
    grid_results   = {}

    total_fits = len(upgrade_sigmas) * len(anchor_weights) * len(test_races)
    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}[/cyan]"),
        BarColumn(bar_width=25),
        MofNCompleteColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Calibrating σ × α...", total=total_fits)
        fold_models = {}
        fold_jobs = []
        for entry in test_races:
            train_parts = [e["df"] for e in race_cache if e is not entry]
            if not train_parts:
                continue
            df_cv = pd.concat(train_parts, ignore_index=True)
            avail = [c for c in feature_cols if c in df_cv.columns]
            fold_jobs.append((entry, df_cv, avail))

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {
                executor.submit(_train_predict_fold, df_cv, avail, entry["features"]): entry
                for entry, df_cv, avail in fold_jobs
            }
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    fold_models[id(entry)] = future.result()
                except Exception:
                    fold_models[id(entry)] = None

        for u_sigma in upgrade_sigmas:
            for a_weight in anchor_weights:
                rhos = []
                for entry in test_races:
                    df_pred = fold_models.get(id(entry))
                    if df_pred is None:
                        progress.advance(task)
                        continue
                    try:
                        with silent():
                            df_adj = apply_upgrade_adjustment(df_pred, upgrade_sigma=u_sigma)
                            df_anc = apply_grid_anchor(
                                df_adj, entry["race"]["year"],
                                entry["race"]["gp"], anchor_weight=a_weight,
                            )
                    except Exception:
                        progress.advance(task)
                        continue
                    df_eval = df_anc.merge(
                        entry["labels"][["Driver", "finish_pos"]], on="Driver", how="inner"
                    )
                    if len(df_eval) >= 5:
                        rho, _ = spearmanr(df_eval["anchored_position"], df_eval["finish_pos"])
                        rhos.append(rho)
                    progress.advance(task)

                if rhos:
                    mean_rho = float(np.mean(rhos))
                    grid_results[(u_sigma, a_weight)] = mean_rho
                    if mean_rho > best_rho:
                        best_rho = mean_rho
                        best_u, best_a = u_sigma, a_weight

    t = Table(title="σ × α Calibration results (top 5)", box=box.SIMPLE_HEAVY, border_style="dim")
    t.add_column("Upgrade σ", justify="center")
    t.add_column("Anchor α",  justify="center")
    t.add_column("Mean ρ",    justify="center")
    sorted_grid = sorted(grid_results.items(), key=lambda x: -x[1])
    for (u, a), rho in sorted_grid[:5]:
        c = "green" if (u, a) == (best_u, best_a) else "white"
        t.add_row(f"[{c}]{u:.2f}[/{c}]", f"[{c}]{a:.2f}[/{c}]", f"[{c}]{rho:+.3f}[/{c}]")
    console.print(t)
    console.print(
        f"  Best: upgrade σ=[bold green]{best_u:.2f}[/bold green]  "
        f"anchor α=[bold green]{best_a:.2f}[/bold green]  "
        f"ρ=[bold]{best_rho:+.3f}[/bold]  "
        f"(calibrated on {len(test_races)} completed 2026 races)"
    )
    return best_u, best_a


def ablation_study(race_cache: list[dict], feature_cols: list[str]) -> None:
    """Measure Spearman ρ for each adjustment layer on completed 2026 races.

    Shows whether upgrade and grid-anchor adjustments actually improve
    predictions vs the raw model output — the honest sanity check.
    """
    test_entries = [
        e for e in race_cache
        if e["race"]["year"] == 2026
        and not e["labels"].empty
        and e["race"]["gp"] != TARGET_RACE["gp"]
    ]

    if not test_entries:
        console.print("  [yellow]No completed 2026 races for ablation. Skipping.[/yellow]")
        return

    variants: dict[str, list[float]] = {
        "Base only":     [],
        "+ Upgrade":     [],
        "+ Grid anchor": [],
        "All combined":  [],
    }

    jobs = []
    for entry in test_entries:
        train_parts = [e["df"] for e in race_cache if e is not entry]
        if not train_parts:
            continue
        df_cv = pd.concat(train_parts, ignore_index=True)
        avail = [c for c in feature_cols if c in df_cv.columns]
        jobs.append((entry, df_cv, avail))

    fold_preds: dict[int, pd.DataFrame | None] = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}[/cyan]"),
        BarColumn(bar_width=25),
        MofNCompleteColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(
            f"Ablation folds (×{os.cpu_count()} parallel)...", total=len(jobs)
        )
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {
                executor.submit(_train_predict_fold, df_cv, avail, entry["features"]): entry
                for entry, df_cv, avail in jobs
            }
            for future in as_completed(futures):
                entry = futures[future]
                progress.update(
                    task,
                    description=f"Ablation — done: {entry['race']['gp']} {entry['race']['year']}",
                )
                try:
                    fold_preds[id(entry)] = future.result()
                except Exception:
                    fold_preds[id(entry)] = None
                progress.advance(task)

    for entry, _df_cv, _avail in jobs:
        df_pred = fold_preds.get(id(entry))
        if df_pred is None:
            continue
        labels = entry["labels"][["Driver", "finish_pos"]]

        def _rho(pos_col: str, df: pd.DataFrame) -> float | None:
            ev = df.merge(labels, on="Driver", how="inner")
            if len(ev) < 5:
                return None
            r, _ = spearmanr(ev[pos_col], ev["finish_pos"])
            return float(r)

        r = _rho("predicted_position", df_pred)
        if r is not None:
            variants["Base only"].append(r)

        df_adj = apply_upgrade_adjustment(df_pred)
        r = _rho("adjusted_position", df_adj)
        if r is not None:
            variants["+ Upgrade"].append(r)

        df_anc = apply_grid_anchor(
            df_pred, entry["race"]["year"], entry["race"]["gp"]
        )
        r = _rho("anchored_position", df_anc)
        if r is not None:
            variants["+ Grid anchor"].append(r)

        df_all = apply_grid_anchor(
            df_adj, entry["race"]["year"], entry["race"]["gp"]
        )
        r = _rho("anchored_position", df_all)
        if r is not None:
            variants["All combined"].append(r)

    base_mean = float(np.mean(variants["Base only"])) if variants["Base only"] else None

    t = Table(
        title=f"Ablation Study — Spearman ρ  [dim]({len(test_entries)} races)[/dim]",
        box=box.ROUNDED,
        border_style="dim",
    )
    t.add_column("Variant",     style="cyan", width=16)
    t.add_column("Mean ρ",      justify="center", width=9)
    t.add_column("± Std",       justify="center", width=7)
    t.add_column("vs Base",     justify="center", width=9)
    t.add_column("Verdict",     justify="center", width=12)

    for name, rhos in variants.items():
        if not rhos:
            t.add_row(name, "N/A", "—", "—", "[dim]no data[/dim]")
            continue
        mean_rho = float(np.mean(rhos))
        std_rho  = float(np.std(rhos))
        if base_mean is None or name == "Base only":
            delta_str  = "—"
            verdict    = ""
            color      = "white"
        else:
            delta = mean_rho - base_mean
            sign  = "+" if delta >= 0 else ""
            color      = "green" if delta > 0.01 else ("red" if delta < -0.01 else "yellow")
            delta_str  = f"[{color}]{sign}{delta:.3f}[/{color}]"
            verdict    = (
                "[green]helps[/green]"   if delta > 0.01 else
                "[red]hurts[/red]"       if delta < -0.01 else
                "[yellow]~ neutral[/yellow]"
            )
        t.add_row(
            name,
            f"[{color}]{mean_rho:+.3f}[/{color}]",
            f"{std_rho:.3f}",
            delta_str,
            verdict,
        )
    console.print(t)


def evaluate_loo_cv(race_cache: list[dict], feature_cols: list[str]) -> None:
    if len(race_cache) < 2:
        console.print("[yellow]LOO-CV requires >= 2 races. Skipping.[/yellow]")
        return

    ndcg_model, ndcg_base, spearman_vals = [], [], []
    fold_rows: list[tuple | None] = [None] * len(race_cache)

    jobs: list[tuple[int, str, pd.DataFrame, list[str], pd.DataFrame]] = []
    for i, held_out in enumerate(race_cache):
        held_out_race = held_out["race"]
        label = f"{held_out_race['gp']} {held_out_race['year']}"

        train_parts = [r["df"] for j, r in enumerate(race_cache) if j != i]
        if not train_parts:
            fold_rows[i] = (label, "—", "—", "—")
            continue
        df_cv_train  = pd.concat(train_parts, ignore_index=True)
        df_cv_feats  = held_out["features"]
        df_cv_labels = held_out["labels"]

        if df_cv_train.empty or df_cv_feats.empty or df_cv_labels.empty:
            fold_rows[i] = (label, "—", "—", "—")
            continue

        df_test = (
            df_cv_feats
            .merge(df_cv_labels[["Driver", "label", "finish_pos"]], on="Driver", how="inner")
            .dropna(subset=["label"])
        )
        if df_test.empty:
            fold_rows[i] = (label, "—", "—", "—")
            continue

        available = [c for c in feature_cols if c in df_cv_train.columns]
        jobs.append((i, label, df_cv_train, available, df_test))

    fold_preds: dict[int, pd.DataFrame | None] = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}[/cyan]"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task(f"LOO-CV (×{os.cpu_count()} parallel folds)", total=len(jobs))
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = {
                executor.submit(_train_predict_fold, df_cv_train, available, df_test): (i, label)
                for i, label, df_cv_train, available, df_test in jobs
            }
            for future in as_completed(futures):
                i, label = futures[future]
                progress.update(task, description=f"LOO-CV — done: [bold]{label}[/bold]")
                try:
                    fold_preds[i] = future.result()
                except Exception:
                    fold_preds[i] = None
                progress.advance(task)

    test_lookup = {i: (label, df_test) for i, label, _, _, df_test in jobs}
    for i in range(len(race_cache)):
        if fold_rows[i] is not None:
            continue
        label, df_test = test_lookup[i]
        df_pred = fold_preds.get(i)
        if df_pred is None or df_pred.empty:
            fold_rows[i] = (label, "—", "—", "—")
            continue

        df_pred = df_pred.merge(
            df_test[["Driver", "label", "finish_pos"]], on="Driver", how="inner"
        )
        if df_pred.empty:
            fold_rows[i] = (label, "—", "—", "—")
            continue

        y_true    = df_pred["label"].values
        y_score   = df_pred["rank_score"].values
        fold_ndcg = sklearn_ndcg_score(y_true.reshape(1, -1), y_score.reshape(1, -1))
        fold_sp, _ = spearmanr(df_pred["predicted_position"], df_pred["finish_pos"])
        ndcg_model.append(fold_ndcg)
        spearman_vals.append(fold_sp)

        fold_base_str = "N/A"
        if "grid_position" in df_test.columns:
            df_base = df_test[["grid_position", "label"]].dropna()
            if not df_base.empty:
                y_t = df_base["label"].values
                y_s = (-df_base["grid_position"]).values
                fold_base = sklearn_ndcg_score(y_t.reshape(1, -1), y_s.reshape(1, -1))
                ndcg_base.append(fold_base)
                fold_base_str = f"{fold_base:.3f}"

        sp_color = "green" if fold_sp > 0.6 else ("yellow" if fold_sp > 0.4 else "red")
        fold_rows[i] = (
            label,
            f"{fold_ndcg:.3f}",
            fold_base_str,
            f"[{sp_color}]{fold_sp:+.3f}[/{sp_color}]",
        )

    t = Table(title="LOO-CV Results", box=box.ROUNDED, border_style="dim")
    t.add_column("Race",          style="cyan", no_wrap=True)
    t.add_column("NDCG Model",    justify="center")
    t.add_column("NDCG Baseline", justify="center", style="dim")
    t.add_column("Spearman ρ",    justify="center")
    for row in fold_rows:
        t.add_row(*row)
    console.print(t)

    if ndcg_model:
        delta_str = ""
        if ndcg_base:
            delta = np.mean(ndcg_model) - np.mean(ndcg_base)
            sign  = "+" if delta >= 0 else ""
            color = "green" if delta >= 0 else "red"
            delta_str = f"  [{color}]Δ vs baseline: {sign}{delta:.3f}[/{color}]"
        console.print(
            f"  [bold]Model[/bold]    NDCG: [bold]{np.mean(ndcg_model):.3f}[/bold] "
            f"± {np.std(ndcg_model):.3f}{delta_str}"
        )
        if ndcg_base:
            console.print(
                f"  [dim]Baseline NDCG: {np.mean(ndcg_base):.3f} "
                f"± {np.std(ndcg_base):.3f}  (grid_position only)[/dim]"
            )
        console.print(
            f"  Spearman ρ: [bold]{np.mean(spearman_vals):.3f}[/bold] "
            f"± {np.std(spearman_vals):.3f}"
        )


def print_feature_importance(model, final_cols: list[str], top_n: int = 20) -> None:
    imp = (
        pd.DataFrame({"feature": final_cols, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    imp["session"] = imp["feature"].str.split("_").str[0].str.upper()
    session_totals = imp.groupby("session")["importance"].sum().sort_values(ascending=False)

    t_session = Table(title="Importance by Session", box=box.SIMPLE_HEAVY, border_style="dim")
    t_session.add_column("Session", style="cyan", width=8)
    t_session.add_column("Total",   justify="right", width=8)
    t_session.add_column("Bar",     width=30)
    max_val = session_totals.max()
    for sess, val in session_totals.items():
        bar_len = int((val / max_val) * 28)
        t_session.add_row(sess, f"{val:.3f}", f"[green]{'█' * bar_len}[/green]")
    console.print(t_session)

    t_feat = Table(
        title=f"Top {top_n} Individual Features",
        box=box.SIMPLE_HEAVY,
        border_style="dim",
    )
    t_feat.add_column("#",       justify="right", width=4)
    t_feat.add_column("Feature", style="cyan",    width=45)
    t_feat.add_column("Imp.",    justify="right",  width=8)
    t_feat.add_column("Bar",     width=30)
    for i, row in imp.head(top_n).iterrows():
        bar_len = int((row["importance"] / imp["importance"].max()) * 28)
        t_feat.add_row(
            str(i + 1),
            row["feature"],
            f"{row['importance']:.4f}",
            f"[yellow]{'█' * bar_len}[/yellow]",
        )
    console.print(t_feat)


if __name__ == "__main__":

    FEATURE_COLS = [
        "fp1_best_lap_delta", "fp1_clean_laps_count",
        "fp1_sector1_delta", "fp1_sector2_delta", "fp1_sector3_delta",
        "fp1_compound_medium_avg", "fp1_laps_on_hard",
        "fp1_vs_teammate",
        "fp2_best_lap_delta", "fp2_clean_laps_count",
        "fp2_sector1_delta", "fp2_sector2_delta", "fp2_sector3_delta",
        "fp2_longrun_medium_avg_pace", "fp2_longrun_medium_deg_rate",
        "fp2_longrun_medium_deg_total", "fp2_longrun_medium_consistency",
        "fp2_longrun_hard_avg_pace", "fp2_longrun_hard_deg_rate",
        "fp2_medium_fuel_corrected_pace",
        "fp2_avg_lift_coast_time_s",
        "fp3_best_lap_delta",
        "fp3_sector1_delta", "fp3_sector2_delta", "fp3_sector3_delta",
        "fp3_soft_best_lap_delta", "fp3_vs_fp2_soft_improvement",
        "fp3_s1_delta_vs_fp2", "fp3_s2_delta_vs_fp2", "fp3_s3_delta_vs_fp2",
        "fp3_is_true_qualy_sim", "fp3_track_evolution_s",
        "sq_best_lap_delta",
        "sq_sector1_delta", "sq_sector2_delta", "sq_sector3_delta",
        "sq_vs_teammate",
        "s_finish_position", "s_positions_gained", "s_classified",
        "grid_position", "is_front_row", "started_top_10",
        "q3_delta_to_pole", "best_q_delta_to_pole", "q3_participation",
        "best_q_vs_fp3_improvement", "quali_best_relative_sector",
        "is_works",
        "recent_form_avg", "recent_form_wins",
        "team_2026_pace_rank",
    ]

    console.print()
    console.print(Panel.fit(
        "[bold white]F1 MONACO GRAND PRIX 2026[/bold white]\n"
        "[dim]Circuit de Monaco, Monte Carlo[/dim]\n"
        "[yellow]⚠ 2026 reg: Active Aero (Straight Mode) DISABLED — FIA safety ruling[/yellow]",
        border_style="red",
        padding=(1, 6),
    ))

    console.rule("[bold red]PHASE 0 — Race discovery[/bold red]")
    with console.status("[cyan]Querying FastF1 calendars...[/cyan]", spinner="dots"):
        with silent():
            TRAINING_RACES_2026 = discover_2026_races_before_target(
                TARGET_RACE["gp"],
                fallback=[
                    {"year": 2026, "gp": "Australian Grand Prix", "weight": 1.0},
                    {"year": 2026, "gp": "Chinese Grand Prix",    "weight": 1.0},
                    {"year": 2026, "gp": "Japanese Grand Prix",   "weight": 1.0},
                    {"year": 2026, "gp": "Miami Grand Prix",      "weight": 1.0},
                    {"year": 2026, "gp": "Canadian Grand Prix",   "weight": 1.0},
                ],
            )
            HISTORICAL_FULL = []
            if INCLUDE_FULL_HISTORICAL:
                for season in FULL_HISTORICAL_YEARS:
                    HISTORICAL_FULL += discover_full_season(season["year"], season["weight"])

    all_races = TRAINING_RACES_2026 + HISTORICAL_MONACO + HISTORICAL_FULL

    t_races = Table(box=box.ROUNDED, border_style="dim")
    t_races.add_column("Race",   style="cyan")
    t_races.add_column("Year",   justify="center")
    t_races.add_column("Weight", justify="center")
    t_races.add_column("Type",   justify="center")
    for r in TRAINING_RACES_2026:
        t_races.add_row(r["gp"], str(r["year"]), f"{r['weight']:.2f}", "[green]2026[/green]")
    for r in HISTORICAL_MONACO:
        t_races.add_row(r["gp"], str(r["year"]), f"{r['weight']:.2f}", "[dim]Monaco hist.[/dim]")
    shown_hist = HISTORICAL_FULL[:3]
    for r in shown_hist:
        t_races.add_row(r["gp"][:28], str(r["year"]), f"{r['weight']:.2f}", "[dim]full season[/dim]")
    if len(HISTORICAL_FULL) > 3:
        t_races.add_row(f"… +{len(HISTORICAL_FULL)-3} more", "—", "—", "[dim]full season[/dim]")
    console.print(t_races)
    console.print(
        f"  [dim]Total: {len(all_races)} races "
        f"({len(TRAINING_RACES_2026)} 2026 + {len(HISTORICAL_MONACO)} Monaco hist. "
        f"+ {len(HISTORICAL_FULL)} full-season hist.)[/dim]"
    )

    console.rule("[bold red]PHASE 1 — Dataset construction[/bold red]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}[/cyan]"),
        BarColumn(bar_width=25),
        MofNCompleteColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Processing races...", total=len(all_races))
        with silent():
            race_cache = []
            for race in all_races:
                progress.update(task, description=f"Loading {race['gp']} {race['year']}...")
                features = build_race_features(race["year"], race["gp"])
                if features.empty:
                    progress.advance(task)
                    continue
                labels = get_race_labels(race["year"], race["gp"])
                if labels.empty:
                    progress.advance(task)
                    continue
                df = features.merge(labels, on="Driver", how="left").dropna(subset=["label"])
                df["label"]         = df["label"].astype(int)
                df["sample_weight"] = race["weight"]
                df["is_2026"]       = int(race["year"] == 2026)
                race_cache.append({
                    "race":     race,
                    "df":       df,
                    "features": features,
                    "labels":   labels,
                })
                progress.advance(task)

    if not race_cache:
        console.print("[bold red]ERROR: No training data available.[/bold red]")
        sys.exit(1)

    with silent():
        races_2026_entries = [e for e in race_cache if e["race"]["year"] == 2026]
        _feat_sources      = [e["features"] for e in races_2026_entries]
        _cum_labels: list[pd.DataFrame] = []

        for entry in races_2026_entries:
            if _cum_labels:
                _form  = get_season_form(_cum_labels)
                _tpace = get_team_pace_rank(_cum_labels, _feat_sources)

                if not _form.empty:
                    entry["df"]       = entry["df"].merge(_form, on="Driver", how="left")
                    entry["features"] = entry["features"].merge(_form, on="Driver", how="left")

                if not _tpace.empty:
                    for _target in [entry["df"], entry["features"]]:
                        _target["_tnorm"] = _target["Team"].replace(TEAM_NORM_MAP).fillna(_target["Team"])
                        _merged = _target.merge(_tpace, left_on="_tnorm", right_on="Team_norm", how="left")
                        _target["team_2026_pace_rank"] = _merged["team_2026_pace_rank"].values
                        _target.drop(columns=["_tnorm"], errors="ignore", inplace=True)

            if not entry["labels"].empty:
                _cum_labels.append(entry["labels"])

    df_train = pd.concat([e["df"] for e in race_cache], axis=0, ignore_index=True)

    n_2026    = int(df_train["is_2026"].sum())
    n_hist    = int((~df_train["is_2026"].astype(bool)).sum())
    n_sprints = sum(
        1 for e in race_cache if "s_finish_position" in e["features"].columns
    )
    available_features = [c for c in FEATURE_COLS if c in df_train.columns]
    missing_features   = [c for c in FEATURE_COLS if c not in df_train.columns]
    FEATURE_COLS       = available_features

    console.print(
        f"  [green]{len(df_train)} rows[/green] × "
        f"[green]{len(df_train.columns)} columns[/green]  "
        f"([cyan]{n_2026}[/cyan] rows 2026 · "
        f"[dim]{n_hist} historical[/dim] · "
        f"[yellow]{n_sprints} sprint weekends[/yellow])"
    )
    if missing_features:
        console.print(
            f"  [yellow]⚠ {len(missing_features)} features absent from training (ignored)[/yellow]"
        )

    console.rule("[bold red]PHASE 1.5 — Parameter calibration[/bold red]")
    console.print(
        f"  Grid-searching upgrade σ ∈ [0.10…0.25] × anchor α ∈ [0.00…0.85] "
        f"on completed 2026 races…"
    )
    best_upgrade_sigma, best_anchor_weight = calibrate_params(race_cache, FEATURE_COLS)

    console.rule("[bold red]PHASE 2 — LOO-CV Evaluation[/bold red]")
    evaluate_loo_cv(race_cache, FEATURE_COLS)

    console.rule("[bold red]PHASE 2.5 — Ablation Study[/bold red]")
    ablation_study(race_cache, FEATURE_COLS)

    console.rule("[bold red]PHASE 3 — Monaco 2026 features[/bold red]")
    with console.status(
        "[cyan]Loading FP1 / FP2 / FP3 / Q for Monaco 2026...[/cyan]", spinner="dots"
    ):
        with silent():
            df_gp = build_race_features(TARGET_RACE["year"], TARGET_RACE["gp"])

    if df_gp.empty:
        console.print("[bold red]ERROR: Could not build the Monaco 2026 feature matrix.[/bold red]")
        sys.exit(1)

    with silent():
        _2026_cache  = [e for e in race_cache if e["race"]["year"] == 2026]
        _2026_labels = [e["labels"] for e in _2026_cache if not e["labels"].empty]
        _2026_feats  = [e["features"] for e in _2026_cache]

        if _2026_labels:
            _form = get_season_form(_2026_labels)
            if not _form.empty:
                df_gp = df_gp.merge(_form, on="Driver", how="left")

            _tpace = get_team_pace_rank(_2026_labels, _2026_feats)
            if not _tpace.empty:
                df_gp["_tnorm"] = df_gp["Team"].replace(TEAM_NORM_MAP).fillna(df_gp["Team"])
                df_gp = df_gp.merge(_tpace, left_on="_tnorm", right_on="Team_norm", how="left")
                df_gp.drop(columns=["_tnorm", "Team_norm"], errors="ignore", inplace=True)

    console.print(
        f"  [green]{len(df_gp)} drivers[/green] · "
        f"[green]{len(df_gp.columns)} features loaded[/green]"
    )

    console.rule("[bold red]PHASE 4 — Model training[/bold red]")
    with console.status("[cyan]Training XGBRanker (rank:pairwise)...[/cyan]", spinner="dots"):
        with silent():
            model, final_cols = train_model(df_train, FEATURE_COLS)
    console.print(f"  Model trained — [green]{len(final_cols)} encoded features[/green]")

    console.rule("[bold red]PHASE 5 — Prediction[/bold red]")
    with console.status("[cyan]Generating ranking...[/cyan]", spinner="dots"):
        with silent():
            df_top      = predict_top(model, final_cols, df_gp, FEATURE_COLS)
            df_adjusted = apply_upgrade_adjustment(df_top, upgrade_sigma=best_upgrade_sigma)
            df_anchored = apply_grid_anchor(
                df_adjusted,
                TARGET_RACE["year"],
                TARGET_RACE["gp"],
                anchor_weight=best_anchor_weight,
            )

    console.print()
    t_upg = Table(
        title="[bold]Confirmed upgrades — Monaco 2026[/bold]",
        box=box.ROUNDED,
        border_style="dim",
    )
    t_upg.add_column("Team",        style="white",    width=22)
    t_upg.add_column("Points",      justify="center", width=8)
    t_upg.add_column("Level",       justify="center", width=12)
    t_upg.add_column("Description", width=40)

    upgrade_color = {3: "bold green", 2: "yellow", 1: "cyan", 0: "dim"}
    upgrade_label = {3: "* MAJOR", 2: "* SIGNIF.", 1: "* MINOR", 0: "o WINGLET"}
    for team, score in sorted(UPGRADE_SCORE.items(), key=lambda x: -x[1]):
        c = upgrade_color[score]
        t_upg.add_row(
            f"[{c}]{team}[/{c}]",
            f"[{c}]{score}[/{c}]",
            f"[{c}]{upgrade_label[score]}[/{c}]",
            f"[dim]{UPGRADE_DESC[score]}[/dim]",
        )
    console.print(t_upg)

    console.print()
    t_raw = Table(
        title="[bold]Model prediction  [dim](no adjustments)[/dim][/bold]",
        box=box.ROUNDED,
        border_style="dim",
    )
    t_raw.add_column("Pos",    justify="center", width=5)
    t_raw.add_column("Driver", style="bold",     width=7)
    t_raw.add_column("Team",                     width=26)
    t_raw.add_column("Score",  justify="right",  width=10)
    for _, row in df_top.iterrows():
        pos = int(row["predicted_position"])
        c   = "gold1" if pos <= 3 else ("white" if pos <= 10 else "dim")
        t_raw.add_row(
            f"[{c}]P{pos}[/{c}]",
            f"[{c}]{row['Driver']}[/{c}]",
            f"[{c}]{str(row.get('Team', 'N/A'))[:25]}[/{c}]",
            f"[{c}]{row['rank_score']:.3f}[/{c}]",
        )
    console.print(t_raw)

    console.print()
    t_adj = Table(
        title=f"[bold]Upgrade-adjusted  [dim](σ={best_upgrade_sigma:.2f}/pt · calibrated)[/dim][/bold]",
        box=box.ROUNDED,
        border_style="yellow",
    )
    t_adj.add_column("Pos",    justify="center", width=5)
    t_adj.add_column("Driver", style="bold",     width=7)
    t_adj.add_column("Team",                     width=26)
    t_adj.add_column("Upg",    justify="center", width=5)
    t_adj.add_column("Score",  justify="right",  width=10)
    t_adj.add_column("Δ Pos",  justify="center", width=7)
    for _, row in df_adjusted.iterrows():
        adj_pos = int(row["adjusted_position"])
        raw_pos = int(row["predicted_position"])
        delta   = raw_pos - adj_pos
        upg     = int(row.get("upgrade_score", 0))
        c       = "gold1" if adj_pos <= 3 else ("white" if adj_pos <= 10 else "dim")
        uc      = upgrade_color.get(upg, "dim")
        d_str   = (f"[green]+{delta}[/green]" if delta > 0
                   else f"[red]-{abs(delta)}[/red]" if delta < 0
                   else "[dim]—[/dim]")
        t_adj.add_row(
            f"[{c}]P{adj_pos}[/{c}]",
            f"[{c}]{row['Driver']}[/{c}]",
            f"[{c}]{str(row.get('Team', 'N/A'))[:25]}[/{c}]",
            f"[{uc}]{upg}[/{uc}]",
            f"[{c}]{row['adjusted_score']:.3f}[/{c}]",
            d_str,
        )
    console.print(t_adj)

    console.print()
    t_anc = Table(
        title=f"[bold]FINAL — Grid-anchored  [dim](α={best_anchor_weight:.2f} · calibrated · no Straight Mode)[/dim][/bold]",
        box=box.ROUNDED,
        border_style="red",
    )
    t_anc.add_column("Pos",    justify="center", width=5)
    t_anc.add_column("Driver", style="bold",     width=7)
    t_anc.add_column("Team",                     width=26)
    t_anc.add_column("Grid",   justify="center", width=6)
    t_anc.add_column("Score",  justify="right",  width=10)
    t_anc.add_column("Δ Pos",  justify="center", width=7)
    for _, row in df_anchored.iterrows():
        anc_pos = int(row["anchored_position"])
        adj_pos = int(row["adjusted_position"])
        delta   = adj_pos - anc_pos
        grid_p  = row.get("grid_position", float("nan"))
        grid_str = f"P{int(grid_p)}" if pd.notna(grid_p) else "—"
        c       = "gold1" if anc_pos <= 3 else ("white" if anc_pos <= 10 else "dim")
        d_str   = (f"[green]+{delta}[/green]" if delta > 0
                   else f"[red]-{abs(delta)}[/red]" if delta < 0
                   else "[dim]—[/dim]")
        t_anc.add_row(
            f"[{c}]P{anc_pos}[/{c}]",
            f"[{c}]{row['Driver']}[/{c}]",
            f"[{c}]{str(row.get('Team', 'N/A'))[:25]}[/{c}]",
            f"[dim]{grid_str}[/dim]",
            f"[{c}]{row['anchored_score']:.3f}[/{c}]",
            d_str,
        )
    console.print(t_anc)
    console.print(
        f"  [dim]Grid col = qualifying start position. "
        f"Anchor blends model {1-best_anchor_weight:.0%} / grid {best_anchor_weight:.0%} "
        f"(calibrated). Monaco overtaking is near-impossible and Straight Mode is "
        f"banned, so track position dominates.[/dim]"
    )

    podium = df_anchored.head(3)
    console.print()
    console.print(Panel.fit(
        "[bold]Predicted podium — Monaco GP 2026[/bold]\n"
        + "\n".join(
            f"  [gold1]P{int(r['anchored_position'])}[/gold1]  "
            f"[bold]{r['Driver']}[/bold]  [dim]{str(r.get('Team', ''))[:24]}[/dim]"
            for _, r in podium.iterrows()
        ),
        border_style="gold1",
        padding=(1, 4),
    ))

    console.rule("[bold dim]Feature Importance[/bold dim]")
    print_feature_importance(model, final_cols, top_n=20)
    console.print()
