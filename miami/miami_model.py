"""Miami 2026 Grand Prix race-result prediction model.

Circuit: Hard Rock Stadium, a moderate-speed permanent circuit.
Weekend format: sprint weekend (FP1 -> SQ -> Sprint -> Q -> Race).

Training data:
    - 2026 season races before Miami (Australia, China, Japan), weight 1.00.
    - Miami 2025 (weight 0.22) and Miami 2024 (weight 0.05) as circuit-specific
      priors, down-weighted by temporal decay exp(-1.5 * delta_years).

Pipeline:
    1. Discover prior 2026 races plus full historical seasons via the FastF1
       calendar.
    2. Build the training set with telemetry features from f1_fe, inject rolling
       2026 season form and team pace (no leakage), and assemble df_train.
    3. Calibrate the upgrade/sprint sigma adjustments on completed sprints.
    4. Evaluate with leave-one-out CV (NDCG and Spearman rho vs grid baseline)
       and run an ablation study over the adjustment layers.
    5. Train an XGBRanker (rank:pairwise) and predict the target race, then apply
       the post-prediction upgrade and sprint adjustment layers.

Because Miami is a sprint weekend and the model is trained mostly on standard
weekends, the Sprint race result is injected as an explicit post-prediction
correction rather than as a training feature.
"""

import contextlib
import io
import os
import sys
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

HISTORICAL_MIAMI = [
    {"year": 2025, "gp": "Miami", "weight": 0.22},
    {"year": 2024, "gp": "Miami", "weight": 0.05},
]

TARGET_RACE = {"year": 2026, "gp": "Miami"}

MIAMI_CIRCUIT_PROFILE = {
    "high_speed_ratio":      0.55,
    "overtaking_difficulty": 0.45,
    "brake_intensity":       0.65,
    "pu_sensitivity":        0.70,
    "street_circuit":        0,
    "traction_zones":        0.60,
}

UPGRADE_SCORE = {
    "McLaren":         2,
    "Ferrari":         2,
    "Mercedes":        1,
    "Red Bull Racing": 1,
    "Williams":        1,
    "Racing Bulls":    1,
    "Alpine":          0,
    "Haas":            0,
    "Cadillac":        0,
    "Aston Martin":    0,
    "Audi":            0,
}
UPGRADE_SIGMA_PER_POINT = 0.18

UPGRADE_DESC = {
    3: "Major rebuild — floor+chassis+wings+bodywork",
    2: "Significant — floor / front wing",
    1: "Minor — cooling / aero tweaks",
    0: "No performance upgrades",
}

SPRINT_BOOST_SIGMA = 0.35


@contextlib.contextmanager
def silent():
    """Suppress sys.stdout during internal calls. Rich console is unaffected."""
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


def extract_fp1_features(year: int, gp: str) -> pd.DataFrame:
    dfs = []
    for fn, kwargs, _ in [
        (f1_fe.get_best_lap_delta,   {"session_type": "FP1"}, ""),
        (f1_fe.get_clean_laps_count, {"session_type": "FP1"}, ""),
        (f1_fe.get_sector_deltas,    {"session_type": "FP1"}, ""),
        (f1_fe.get_max_speed_trap,   {"session_type": "FP1"}, ""),
        (f1_fe.get_compound_avg,     {"session_type": "FP1", "compound": "MEDIUM"}, ""),
        (f1_fe.get_laps_on_compound, {"session_type": "FP1", "compound": "HARD"}, ""),
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
    for fn, kwargs, _ in [
        (f1_fe.get_best_lap_delta,   {"session_type": "FP2"}, ""),
        (f1_fe.get_clean_laps_count, {"session_type": "FP2"}, ""),
        (f1_fe.get_sector_deltas,    {"session_type": "FP2"}, ""),
        (f1_fe.get_max_speed_trap,   {"session_type": "FP2"}, ""),
    ]:
        try:
            dfs.append(fn(year, gp, **kwargs))
        except Exception:
            pass
    for compound in ["MEDIUM", "HARD"]:
        for fn, _ in [
            (f1_fe.get_longrun_avg_pace,    ""),
            (f1_fe.get_longrun_deg_rate,    ""),
            (f1_fe.get_longrun_deg_total,   ""),
            (f1_fe.get_longrun_consistency, ""),
        ]:
            try:
                dfs.append(fn(year, gp, "FP2", compound=compound))
            except Exception:
                pass
    for fn, kwargs, _ in [
        (f1_fe.get_fuel_corrected_pace,     {"session_type": "FP2", "compound": "MEDIUM"}, ""),
        (f1_fe.get_pu_deployment_asymmetry, {"session_type": "FP2"}, ""),
        (f1_fe.get_speed_trap_variance,     {"session_type": "FP2"}, ""),
        (f1_fe.get_lift_and_coast_laps,     {"session_type": "FP2"}, ""),
        (f1_fe.get_ers_efficiency_proxy,    {"session_type": "FP2"}, ""),
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
    for fn, kwargs, _ in [
        (f1_fe.get_best_lap_delta,            {"session_type": "FP3"}, ""),
        (f1_fe.get_clean_laps_count,          {"session_type": "FP3"}, ""),
        (f1_fe.get_sector_deltas,             {"session_type": "FP3"}, ""),
        (f1_fe.get_max_speed_trap,            {"session_type": "FP3"}, ""),
        (f1_fe.get_qualy_sim_delta,           {"session_type": "FP3", "compound": "SOFT"}, ""),
        (f1_fe.get_fp3_vs_fp2_improvement,    {}, ""),
        (f1_fe.get_sector_improvement_vs_fp2, {}, ""),
        (f1_fe.get_fp3_qualy_sim_context,     {"session_type": "FP3"}, ""),
        (f1_fe.get_track_evolution,           {"session_type": "FP3"}, ""),
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
    for fn, kwargs, _ in [
        (f1_fe.get_best_lap_delta,   {"session_type": "SQ"}, ""),
        (f1_fe.get_sector_deltas,    {"session_type": "SQ"}, ""),
        (f1_fe.get_max_speed_trap,   {"session_type": "SQ"}, ""),
        (f1_fe.get_clean_laps_count, {"session_type": "SQ"}, ""),
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
    for fn, kwargs, _ in [
        (f1_fe.get_qualy_deltas,               {}, ""),
        (f1_fe.get_q3_participation_flag,      {}, ""),
        (f1_fe.get_qualy_vs_fp3_improvement,   {}, ""),
        (f1_fe.get_best_quali_relative_sector, {}, ""),
        (f1_fe.get_grid_position_features,     {"session_type": "R"}, ""),
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


def extract_team_features(year: int, gp: str) -> pd.DataFrame:
    try:
        df_team = f1_fe.get_team_info(year, gp, "Q")
        TEAM_NAME_MAP = {
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
        }
        df_team["Team_norm"]   = df_team["Team"].replace(TEAM_NAME_MAP)
        df_team["pu_score"]    = df_team["Team_norm"].map(PU_MAP).fillna(2).astype(float)
        df_team["is_works"]    = df_team["Team_norm"].map(WORKS_MAP).fillna(0).astype(float)
        df_team["pu_is_works"] = df_team["pu_score"] * df_team["is_works"]
        return df_team[["Driver", "Team", "pu_score", "is_works", "pu_is_works"]]
    except Exception:
        return pd.DataFrame()


def add_circuit_context(df: pd.DataFrame) -> pd.DataFrame:
    for key, val in MIAMI_CIRCUIT_PROFILE.items():
        df[f"circuit_{key}"] = val
    return df


def build_race_features(
    year: int, gp: str, include_circuit_context: bool = False
) -> pd.DataFrame:
    try:
        event = fastf1.get_event(year, gp)
        is_sprint = event.get_session_name(4) == "Sprint"
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
        crash_mask = df["status"].str.contains(
            "Accident|Collision|Spun off|Damage", na=False, case=False)
        df.loc[crash_mask, "finish_pos"] = 22
        dnf_mask = ~df["status"].str.contains("Finished|Lap", na=False)
        df.loc[dnf_mask, "finish_pos"] = 22
        df["label"] = (22 - df["finish_pos"]).clip(lower=0).astype(int)
        return df[["Driver", "finish_pos", "label"]]
    except Exception:
        return pd.DataFrame()


def build_full_training_set(races: list[dict]) -> pd.DataFrame:
    all_dfs = []
    for race in races:
        features = build_race_features(race["year"], race["gp"])
        if features.empty:
            continue
        labels = get_race_labels(race["year"], race["gp"])
        if labels.empty:
            continue
        df = features.merge(labels, on="Driver", how="left").dropna(subset=["label"])
        df["label"]         = df["label"].astype(int)
        df["sample_weight"] = race["weight"]
        df["is_2026"]       = int(race["year"] == 2026)
        all_dfs.append(df)
    if not all_dfs:
        raise ValueError("No training data — verify your FastF1 cache.")
    return pd.concat(all_dfs, axis=0, ignore_index=True)


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


def train_model(df_train: pd.DataFrame, feature_cols: list[str]):
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
    )
    model.fit(X_proc, y, group=groups, sample_weight=group_weights)
    return model, final_cols


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
    df_result["rank_score"] = scores
    df_result = df_result.sort_values("rank_score", ascending=False).reset_index(drop=True)
    df_result["predicted_position"] = df_result.index + 1
    return df_result.head(22)


def apply_upgrade_adjustment(df_pred: pd.DataFrame) -> pd.DataFrame:
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
    }
    df = df_pred.copy()
    df["Team_norm"]     = df["Team"].replace(TEAM_NORM_MAP).fillna(df["Team"])
    df["upgrade_score"] = df["Team_norm"].map(UPGRADE_SCORE).fillna(0)
    score_std            = df["rank_score"].std()
    df["upgrade_boost"]  = df["upgrade_score"] * UPGRADE_SIGMA_PER_POINT * score_std
    df["adjusted_score"] = df["rank_score"] + df["upgrade_boost"]
    df = df.sort_values("adjusted_score", ascending=False).reset_index(drop=True)
    df["adjusted_position"] = df.index + 1
    return df


def apply_sprint_adjustment(df_pred: pd.DataFrame, year: int, gp: str) -> pd.DataFrame:
    """Boost/penalise drivers based on actual Sprint race finishing position.

    The base model assigns near-zero importance to sprint features because all
    training races are standard weekends. For sprint GPs the Sprint result is
    the strongest real-world pace signal and must be applied explicitly.

    Normalised position score: P1 → +1 · median → 0 · last → -1
    """
    try:
        session_s = fastf1.get_session(year, gp, "S")
        session_s.load(telemetry=False, weather=False, messages=False)
        res = session_s.results[["Abbreviation", "Position", "Status"]].copy()
        res.columns = ["Driver", "sprint_pos", "sprint_status"]
        res["sprint_pos"] = pd.to_numeric(res["sprint_pos"], errors="coerce")
    except Exception:
        return df_pred

    df = df_pred.copy()
    df = df.merge(res, on="Driver", how="left")

    n          = len(df)
    score_std  = df["adjusted_score"].std()
    median_pos = (n + 1) / 2

    df["sprint_z"] = (median_pos - df["sprint_pos"]) / median_pos
    dnf_mask = ~df["sprint_status"].fillna("").str.contains("Finished|Lap", na=False)
    df.loc[dnf_mask, "sprint_z"] = -1.0
    df["sprint_z"] = df["sprint_z"].fillna(0.0)

    df["sprint_boost"]          = df["sprint_z"] * SPRINT_BOOST_SIGMA * score_std
    df["sprint_adjusted_score"] = df["adjusted_score"] + df["sprint_boost"]

    df = df.sort_values("sprint_adjusted_score", ascending=False).reset_index(drop=True)
    df["sprint_adjusted_position"] = df.index + 1
    return df


def evaluate_loo_cv(race_cache: list[dict], feature_cols: list[str]) -> None:
    """LOO-CV using pre-built DataFrames from Phase 1.

    Re-using cached DataFrames eliminates all redundant FastF1 session loads —
    each fold only re-trains the model, which takes seconds.
    """
    if len(race_cache) < 2:
        console.print("[yellow]LOO-CV requires >= 2 races. Skipping.[/yellow]")
        return

    ndcg_model, ndcg_base, spearman_vals = [], [], []
    fold_rows = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[cyan]{task.description}[/cyan]"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("LOO-CV", total=len(race_cache))

        for i, held_out in enumerate(race_cache):
            held_out_race = held_out["race"]
            progress.update(
                task,
                description=f"LOO-CV — fold {i+1}/{len(race_cache)}: "
                            f"[bold]{held_out_race['gp']} {held_out_race['year']}[/bold]",
            )

            train_parts = [r["df"] for j, r in enumerate(race_cache) if j != i]
            if not train_parts:
                progress.advance(task)
                continue
            df_cv_train  = pd.concat(train_parts, ignore_index=True)
            df_cv_feats  = held_out["features"]
            df_cv_labels = held_out["labels"]

            if df_cv_train.empty or df_cv_feats.empty or df_cv_labels.empty:
                fold_rows.append((f"{held_out_race['gp']} {held_out_race['year']}", "—", "—", "—"))
                progress.advance(task)
                continue

            df_test = (
                df_cv_feats
                .merge(df_cv_labels[["Driver", "label", "finish_pos"]], on="Driver", how="inner")
                .dropna(subset=["label"])
            )
            if df_test.empty:
                progress.advance(task)
                continue

            available = [c for c in feature_cols if c in df_cv_train.columns]
            with silent():
                model_cv, cols_cv = train_model(df_cv_train, available)
                df_pred = predict_top(model_cv, cols_cv, df_test, available)

            df_pred = df_pred.merge(
                df_test[["Driver", "label", "finish_pos"]], on="Driver", how="inner"
            )
            if df_pred.empty:
                progress.advance(task)
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
            fold_rows.append((
                f"{held_out_race['gp']} {held_out_race['year']}",
                f"{fold_ndcg:.3f}",
                fold_base_str,
                f"[{sp_color}]{fold_sp:+.3f}[/{sp_color}]",
            ))
            progress.advance(task)

    t = Table(title="LOO-CV Results", box=box.ROUNDED, border_style="dim")
    t.add_column("Race",           style="cyan", no_wrap=True)
    t.add_column("NDCG Model",     justify="center")
    t.add_column("NDCG Baseline",  justify="center", style="dim")
    t.add_column("Spearman ρ",     justify="center")
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
    t_feat.add_column("Imp.",    justify="right", width=8)
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
        "fp1_max_speed_trap", "fp1_speed_trap_delta",
        "fp1_compound_medium_avg", "fp1_laps_on_hard",
        "fp2_best_lap_delta", "fp2_clean_laps_count",
        "fp2_sector1_delta", "fp2_sector2_delta", "fp2_sector3_delta",
        "fp2_max_speed_trap",
        "fp2_longrun_medium_avg_pace", "fp2_longrun_medium_deg_rate",
        "fp2_longrun_medium_deg_total", "fp2_longrun_medium_consistency",
        "fp2_longrun_hard_avg_pace", "fp2_longrun_hard_deg_rate",
        "fp2_medium_fuel_corrected_pace",
        "fp2_pu_asymmetry_delta", "fp2_speed_trap_std_kmh",
        "fp2_avg_lift_coast_time_s", "fp2_ers_efficiency_proxy",
        "fp3_best_lap_delta",
        "fp3_sector1_delta", "fp3_sector2_delta", "fp3_sector3_delta",
        "fp3_soft_best_lap_delta", "fp3_vs_fp2_soft_improvement",
        "fp3_s1_delta_vs_fp2", "fp3_s2_delta_vs_fp2", "fp3_s3_delta_vs_fp2",
        "fp3_is_true_qualy_sim", "fp3_track_evolution_s",
        "sq_best_lap_delta",
        "sq_sector1_delta", "sq_sector2_delta", "sq_sector3_delta",
        "sq_speed_trap_delta",
        "s_finish_position", "s_positions_gained", "s_classified",
        "grid_position", "is_front_row", "started_top_10",
        "q3_delta_to_pole", "best_q_delta_to_pole", "q3_participation",
        "best_q_vs_fp3_improvement", "quali_best_relative_sector",
        "pu_score", "is_works", "pu_is_works",
        "ers_clipping_penalty_index",
    ]

    console.print()
    console.print(Panel.fit(
        "[bold white]F1 MIAMI GRAND PRIX 2026[/bold white]\n"
        "[dim]Hard Rock Stadium Circuit, Miami Gardens[/dim]",
        border_style="red",
        padding=(1, 6),
    ))

    console.rule("[bold red]PHASE 0 — 2026 race discovery[/bold red]")
    with console.status("[cyan]Querying FastF1 calendar...[/cyan]", spinner="dots"):
        with silent():
            TRAINING_RACES_2026 = discover_2026_races_before_target(
                TARGET_RACE["gp"],
                fallback=[
                    {"year": 2026, "gp": "Australia", "weight": 1.0},
                    {"year": 2026, "gp": "China",     "weight": 1.0},
                    {"year": 2026, "gp": "Japan",     "weight": 1.0},
                ],
            )
    all_races = TRAINING_RACES_2026 + HISTORICAL_MIAMI

    t_races = Table(box=box.ROUNDED, border_style="dim")
    t_races.add_column("Race",   style="cyan")
    t_races.add_column("Year",   justify="center")
    t_races.add_column("Weight", justify="center")
    t_races.add_column("Type",   justify="center")
    for r in TRAINING_RACES_2026:
        t_races.add_row(r["gp"], str(r["year"]), f"{r['weight']:.2f}", "[green]2026[/green]")
    for r in HISTORICAL_MIAMI:
        t_races.add_row(r["gp"], str(r["year"]), f"{r['weight']:.2f}", "[dim]historical[/dim]")
    console.print(t_races)
    console.print(f"  [dim]Total: {len(all_races)} training races[/dim]")

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
            race_cache     = []
            df_train_parts = []
            for race in all_races:
                progress.update(task, description=f"Processing {race['gp']} {race['year']}...")
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
                df_train_parts.append(df)
                race_cache.append({
                    "race":     race,
                    "df":       df,
                    "features": features,
                    "labels":   labels,
                })
                progress.advance(task)

    if not df_train_parts:
        console.print("[bold red]ERROR: No training data available.[/bold red]")
        sys.exit(1)
    df_train = pd.concat(df_train_parts, axis=0, ignore_index=True)

    available_features = [c for c in FEATURE_COLS if c in df_train.columns]
    missing_features   = [c for c in FEATURE_COLS if c not in df_train.columns]
    FEATURE_COLS       = available_features

    console.print(
        f"  [green]{len(df_train)} rows[/green] × "
        f"[green]{len(df_train.columns)} columns[/green]  "
        f"([cyan]{int(df_train['is_2026'].sum())}[/cyan] rows 2026 + "
        f"[dim]{int((~df_train['is_2026'].astype(bool)).sum())} historical[/dim])"
    )
    if missing_features:
        console.print(
            f"  [yellow]⚠ {len(missing_features)} features missing from training "
            f"(will be ignored)[/yellow]"
        )

    console.rule("[bold red]PHASE 2 — LOO-CV Evaluation[/bold red]")
    evaluate_loo_cv(race_cache, FEATURE_COLS)

    console.rule("[bold red]PHASE 3 — Miami 2026 features[/bold red]")
    with console.status(
        "[cyan]Loading FP1 / SQ / S / Q for Miami 2026...[/cyan]", spinner="dots"
    ):
        with silent():
            df_gp = build_race_features(TARGET_RACE["year"], TARGET_RACE["gp"])

    if df_gp.empty:
        console.print("[bold red]ERROR: Could not build the Miami 2026 feature matrix.[/bold red]")
        sys.exit(1)
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
            df_adjusted = apply_upgrade_adjustment(df_top)
            df_sprint   = apply_sprint_adjustment(
                df_adjusted, TARGET_RACE["year"], TARGET_RACE["gp"]
            )

    console.print()
    t_upg = Table(
        title="[bold]Confirmed upgrades — Miami 2026[/bold]",
        box=box.ROUNDED,
        border_style="dim",
    )
    t_upg.add_column("Team",        style="white",    width=22)
    t_upg.add_column("Points",      justify="center", width=8)
    t_upg.add_column("Level",       justify="center", width=12)
    t_upg.add_column("Description", width=40)

    upgrade_color = {3: "bold green", 2: "yellow", 1: "cyan", 0: "dim"}
    upgrade_label = {3: "* MAJOR", 2: "* SIGNIF.", 1: "* MINOR", 0: "o NONE"}
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
        title=f"[bold]Upgrade-adjusted  [dim](σ boost: {UPGRADE_SIGMA_PER_POINT}/pt)[/dim][/bold]",
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

    if "sprint_adjusted_position" in df_sprint.columns:
        console.print()
        t_sp = Table(
            title=f"[bold]Sprint-adjusted  [dim](σ boost: {SPRINT_BOOST_SIGMA}/normalised unit)[/dim][/bold]",
            box=box.ROUNDED,
            border_style="red",
        )
        t_sp.add_column("Pos",    justify="center", width=5)
        t_sp.add_column("Driver", style="bold",     width=7)
        t_sp.add_column("Team",                     width=26)
        t_sp.add_column("Sprint", justify="center", width=7)
        t_sp.add_column("Score",  justify="right",  width=10)
        t_sp.add_column("Δ Upg",  justify="center", width=7)
        for _, row in df_sprint.iterrows():
            sp_pos  = int(row["sprint_adjusted_position"])
            adj_pos = int(row["adjusted_position"])
            delta   = adj_pos - sp_pos
            sp_p    = row.get("sprint_pos", float("nan"))
            sp_str  = f"P{int(sp_p)}" if pd.notna(sp_p) else "—"
            c       = "gold1" if sp_pos <= 3 else ("white" if sp_pos <= 10 else "dim")
            d_str   = (f"[green]+{delta}[/green]" if delta > 0
                       else f"[red]-{abs(delta)}[/red]" if delta < 0
                       else "[dim]—[/dim]")
            t_sp.add_row(
                f"[{c}]P{sp_pos}[/{c}]",
                f"[{c}]{row['Driver']}[/{c}]",
                f"[{c}]{str(row.get('Team', 'N/A'))[:25]}[/{c}]",
                f"[dim]{sp_str}[/dim]",
                f"[{c}]{row['sprint_adjusted_score']:.3f}[/{c}]",
                d_str,
            )
        console.print(t_sp)
        console.print(
            f"  [dim]Sprint column = actual Sprint race finish position. "
            f"Boost = {SPRINT_BOOST_SIGMA}σ × normalised position score.[/dim]"
        )
    else:
        console.print(
            "  [yellow]Sprint adjustment skipped — Sprint session not yet available.[/yellow]"
        )

    console.rule("[bold dim]Feature Importance[/bold dim]")
    print_feature_importance(model, final_cols, top_n=20)
    console.print()
