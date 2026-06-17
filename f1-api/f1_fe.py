"""F1 feature engineering library (shared by all race models).

Provides pure functions that load a FastF1 session and return a pandas
DataFrame keyed by Driver. Every function applies the same quality filters
before computing anything:

    - 107% rule: laps slower than 1.07 x session_best are discarded.
    - Green flag only: TrackStatus == '1' (no yellow, red, SC, VSC).
    - Clean lap only: PitOutTime and PitInTime must both be NaN.
    - Defensive short-circuit: a session with no valid data returns a
      NaN-filled DataFrame instead of raising.
    - Prefix naming: every output column is prefixed with the session type
      (fp1_, fp2_, sq_, ...) so DataFrames from different sessions merge safely.

Function groups:
    - Circuit baseline: best lap delta, clean lap count, compound averages,
      sector deltas, speed traps, laps on compound.
    - Long run / race simulation: rolling pace, degradation rate/total,
      consistency, stint compound, fuel-corrected pace.
    - Power unit / ERS (2026 regs): ERS efficiency proxy, lift-and-coast,
      PU deployment asymmetry, speed-trap variance.
    - FP3 / setup tuning: qualy-sim delta, FP3 vs FP2 improvement,
      track evolution.
    - Qualifying: Q1/Q2/Q3 deltas, Q3 participation, best relative sector.
    - Grid position: race start order with Q -> SQ -> R fallback.
    - Teammate and season-form deltas.
"""

import fastf1
import pandas as pd
import numpy as np

fastf1.Cache.enable_cache('./f1_cache')

def _keep_representative_laps(valid_laps: pd.DataFrame, factor: float = 1.07) -> pd.DataFrame:
    """Drop in/out/aborted laps from already compound/green/pit-filtered long-run laps.

    A fixed 107% gate against the session's fastest lap (typically a low-fuel
    qualifying simulation) wrongly discards legitimate high-fuel long-run laps,
    which are naturally several percent slower. Instead, within each driver-stint
    keep only laps within ``factor`` of that stint's own median lap time. This
    adapts to fuel load and tyre age and removes only true outliers (in/out laps,
    spins, lift-and-coast aborts).
    """
    if valid_laps.empty or "Stint" not in valid_laps.columns:
        return valid_laps
    df = valid_laps.assign(_lt_s=valid_laps["LapTime"].dt.total_seconds())
    grp = df.groupby(["Driver", "Stint"])["_lt_s"]
    med = grp.transform("median")
    size = grp.transform("size")
    keep = (size < 3) | (df["_lt_s"] <= med * factor)
    return df[keep].drop(columns="_lt_s")


def get_best_lap_delta(year: int, gp: str, session_type: str) -> pd.DataFrame:
    """
    Calculates best_lap_delta for all drivers dynamically based on session_type:
    (driver_best_laptime - field_best_laptime) / field_best_laptime
    
    Only considers clean flying laps without traffic or yellow flags.
    """
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=False, weather=False, messages=False)
    
    prefix = session_type.lower()
    laps = session.laps.copy()

    laps = laps[
        (laps['TrackStatus'] == '1')
        & (laps['PitOutTime'].isna())
        & (laps['PitInTime'].isna())
        & (laps['LapTime'].notna())
        & (laps['LapTime'] > pd.Timedelta(0))
    ]

    driver_best = (
        laps.groupby('Driver')['LapTime']
        .min()
        .reset_index()
        .rename(columns={'LapTime': 'driver_best_laptime'})
    )

    field_best = driver_best['driver_best_laptime'].min()

    driver_best['field_best_laptime'] = field_best
    driver_best[f'{prefix}_best_lap_delta'] = (
        (driver_best['driver_best_laptime'] - field_best).dt.total_seconds() 
        / field_best.total_seconds()
    )

    driver_best[f'{prefix}_driver_best_laptime_s'] = driver_best['driver_best_laptime'].dt.total_seconds()
    driver_best[f'{prefix}_field_best_laptime_s'] = field_best.total_seconds()

    return driver_best[['Driver', f'{prefix}_driver_best_laptime_s', 
                         f'{prefix}_field_best_laptime_s', f'{prefix}_best_lap_delta']]

def get_clean_laps_count(year: int, gp: str, session_type: str) -> pd.DataFrame:
    """
    Counts clean laps per driver in the specified session.
    Filters out aborted laps using the 107% rule threshold.
    """
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=False, weather=False, messages=False)
    
    prefix = session_type.lower()
    laps = session.laps.copy()

    best_time = laps['LapTime'].min()
    max_acceptable = best_time * 1.07

    clean_laps = laps[
        (laps['TrackStatus'] == '1')
        & (laps['PitOutTime'].isna())
        & (laps['PitInTime'].isna())
        & (laps['LapTime'].notna())
        & (laps['LapTime'] > pd.Timedelta(0))
        & (laps['LapTime'] <= max_acceptable)
    ]

    result = clean_laps.groupby('Driver').size().reset_index(name=f'{prefix}_clean_laps_count')

    all_drivers = session.laps['Driver'].unique()
    result = (
        pd.DataFrame({'Driver': all_drivers})
        .merge(result, on='Driver', how='left')
        .fillna({f'{prefix}_clean_laps_count': 0})
    )
    result[f'{prefix}_clean_laps_count'] = result[f'{prefix}_clean_laps_count'].astype(int)

    return result.sort_values(f'{prefix}_clean_laps_count', ascending=False)

def get_compound_avg(year: int, gp: str, session_type: str, compound: str = 'HARD') -> pd.DataFrame:
    """
    Average lap time on a specific compound, normalized by tyre life 
    to correct for degradation evolution. Requires at least 3 clean laps.
    """
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=False, weather=False, messages=False)
    
    prefix = session_type.lower()
    comp_lower = compound.lower()
    laps = session.laps.copy()

    best_time = laps['LapTime'].min()
    max_acceptable = best_time * 1.07

    target_laps = laps[
        (laps['Compound'] == compound.upper())
        & (laps['TrackStatus'] == '1')
        & (laps['PitOutTime'].isna())
        & (laps['PitInTime'].isna())
        & (laps['LapTime'].notna())
        & (laps['LapTime'] > pd.Timedelta(0))
        & (laps['LapTime'] <= max_acceptable)
    ].copy()

    target_laps['LapTime_s'] = target_laps['LapTime'].dt.total_seconds()

    DEG_CORRECTION = 0.08 
    target_laps['LapTime_corrected'] = target_laps['LapTime_s'] - (target_laps['TyreLife'] * DEG_CORRECTION)

    counts = target_laps.groupby('Driver')['LapTime_s'].count()
    valid_drivers = counts[counts >= 3].index
    target_laps = target_laps[target_laps['Driver'].isin(valid_drivers)]

    result = (
        target_laps.groupby('Driver')
        .agg(
            avg=('LapTime_corrected', 'mean'),
            laps_count=('LapTime_s', 'count'),
            best=('LapTime_s', 'min')
        )
        .rename(columns={
            'avg': f'{prefix}_compound_{comp_lower}_avg',
            'laps_count': f'{prefix}_compound_{comp_lower}_laps',
            'best': f'{prefix}_compound_{comp_lower}_best'
        })
        .reset_index()
    )

    all_drivers = session.laps['Driver'].unique()
    result = pd.DataFrame({'Driver': all_drivers}).merge(result, on='Driver', how='left')

    return result.sort_values(f'{prefix}_compound_{comp_lower}_avg')

def get_sector_deltas(year: int, gp: str, session_type: str) -> pd.DataFrame:
    """
    Calculates sector deltas for S1, S2, and S3 simultaneously to avoid
    loading the session data multiple times.
    """
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=False, weather=False, messages=False)
    
    prefix = session_type.lower()
    laps = session.laps.copy()

    best_lap_time = laps['LapTime'].min()
    max_acceptable_lap = best_lap_time * 1.07

    valid_laps = laps[
        (laps['TrackStatus'] == '1')
        & (laps['PitOutTime'].isna())
        & (laps['PitInTime'].isna())
        & (laps['LapTime'].notna())
        & (laps['LapTime'] <= max_acceptable_lap)
    ].copy()

    for sec in [1, 2, 3]:
        valid_laps[f'S{sec}_s'] = valid_laps[f'Sector{sec}Time'].dt.total_seconds()

    driver_stats = valid_laps.groupby('Driver').agg(
        s1_avg=('S1_s', 'mean'), s1_best=('S1_s', 'min'),
        s2_avg=('S2_s', 'mean'), s2_best=('S2_s', 'min'),
        s3_avg=('S3_s', 'mean'), s3_best=('S3_s', 'min')
    ).reset_index()

    for sec in [1, 2, 3]:
        best_sec = driver_stats[f's{sec}_best'].min()
        driver_stats[f'{prefix}_sector{sec}_delta'] = driver_stats[f's{sec}_avg'] - best_sec

    cols_to_keep = ['Driver', f'{prefix}_sector1_delta', f'{prefix}_sector2_delta', f'{prefix}_sector3_delta']
    driver_stats = driver_stats[cols_to_keep]

    all_drivers = session.laps['Driver'].unique()
    result = pd.DataFrame({'Driver': all_drivers}).merge(driver_stats, on='Driver', how='left')

    return result.sort_values(f'{prefix}_sector1_delta').reset_index(drop=True)

def get_max_speed_trap(year: int, gp: str, session_type: str) -> pd.DataFrame:
    """
    Extracts the highest speed trap (SpeedST) value recorded by each driver 
    in km/h during representative laps.
    """
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=False, weather=False, messages=False)
    
    prefix = session_type.lower()
    laps = session.laps.copy()

    best_lap_time = laps['LapTime'].min()
    max_acceptable_lap = best_lap_time * 1.07

    valid_laps = laps[
        (laps['TrackStatus'] == '1')
        & (laps['PitOutTime'].isna())
        & (laps['PitInTime'].isna())
        & (laps['SpeedST'].notna())
        & (laps['SpeedST'] > 0)
        & (laps['LapTime'].notna())
        & (laps['LapTime'] <= max_acceptable_lap)
    ].copy()

    driver_stats = valid_laps.groupby('Driver').agg(
        max_speed=('SpeedST', 'max'),
        avg_speed=('SpeedST', 'mean'),
        count=('SpeedST', 'count')
    ).rename(columns={
        'max_speed': f'{prefix}_max_speed_trap',
        'avg_speed': f'{prefix}_speed_trap_avg',
        'count': f'{prefix}_valid_laps_count'
    }).reset_index()

    absolute_max = driver_stats[f'{prefix}_max_speed_trap'].max()
    driver_stats[f'{prefix}_speed_trap_delta'] = absolute_max - driver_stats[f'{prefix}_max_speed_trap']

    all_drivers = session.laps['Driver'].unique()
    result = pd.DataFrame({'Driver': all_drivers}).merge(driver_stats, on='Driver', how='left')

    return result.sort_values(f'{prefix}_speed_trap_delta').reset_index(drop=True)

def get_laps_on_compound(year: int, gp: str, session_type: str, compound: str = 'HARD') -> pd.DataFrame:
    """
    Calculates the total number of completed laps by each driver 
    using a specific compound during the session.
    """
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=False, weather=False, messages=False)
    
    prefix = session_type.lower()
    comp_lower = compound.lower()
    laps = session.laps.copy()

    target_laps = laps[
        (laps['Compound'] == compound.upper())
        & (laps['LapTime'].notna())
    ].copy()

    driver_stats = target_laps.groupby('Driver').agg(
        laps_count=('LapTime', 'count')
    ).rename(columns={
        'laps_count': f'{prefix}_laps_on_{comp_lower}'
    }).reset_index()

    all_drivers = session.laps['Driver'].unique()
    result = (
        pd.DataFrame({'Driver': all_drivers})
        .merge(driver_stats, on='Driver', how='left')
        .fillna({f'{prefix}_laps_on_{comp_lower}': 0})
    )

    result[f'{prefix}_laps_on_{comp_lower}'] = result[f'{prefix}_laps_on_{comp_lower}'].astype(int)

    return result.sort_values(f'{prefix}_laps_on_{comp_lower}', ascending=False).reset_index(drop=True)

def get_longrun_avg_pace(year: int, gp: str, session_type: str, compound: str = 'MEDIUM', window_size: int = 8) -> pd.DataFrame:
    """
    Calculates the normalized average pace of the best N strictly consecutive laps 
    on a specific compound (Long Run Race Pace).
    
    Includes dynamic tire degradation correction, strict validation, and 
    empty-state handling for sessions where long runs do not occur.
    """
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=False, weather=False, messages=False)
    
    prefix = session_type.lower()
    comp_lower = compound.lower()
    col_name = f'{prefix}_longrun_{comp_lower}_avg_pace'
    
    laps = session.laps.copy()
    all_drivers = laps['Driver'].unique()

    best_lap_time = laps['LapTime'].min()
    max_acceptable_lap = best_lap_time * 1.07

    valid_laps = laps[
        (laps['Compound'] == compound.upper())
        & (laps['TrackStatus'] == '1')
        & (laps['PitOutTime'].isna())
        & (laps['PitInTime'].isna())
        & (laps['LapTime'].notna())
        & (laps['LapTime'] > pd.Timedelta(0))
    ].copy()

    valid_laps = _keep_representative_laps(valid_laps)

    if valid_laps.empty:
        result = pd.DataFrame({'Driver': all_drivers})
        result[col_name] = np.nan
        return result

    valid_laps['LapTime_s'] = valid_laps['LapTime'].dt.total_seconds()

    if compound.upper() == 'SOFT':
        deg_correction = 0.12
    elif compound.upper() == 'HARD':
        deg_correction = 0.05
    else:
        deg_correction = 0.08 

    valid_laps['LapTime_corrected'] = valid_laps['LapTime_s'] - (valid_laps['TyreLife'] * deg_correction)

    valid_laps = valid_laps.sort_values(by=['Driver', 'Stint', 'LapNumber'])

    def get_best_consecutive_window(df: pd.DataFrame, window: int) -> pd.Series:
        if len(df) < window:
            return pd.Series({col_name: np.nan})
            
        rolling_avg = df['LapTime_corrected'].rolling(window=window).mean()
        
        rolling_lap_max = df['LapNumber'].rolling(window=window).max()
        rolling_lap_min = df['LapNumber'].rolling(window=window).min()
        is_strictly_consecutive = (rolling_lap_max - rolling_lap_min) == (window - 1)
        
        valid_windows = rolling_avg[is_strictly_consecutive]
        
        if valid_windows.empty:
            return pd.Series({col_name: np.nan})
            
        return pd.Series({col_name: valid_windows.min()})

    stint_stats = (
        valid_laps.groupby(['Driver', 'Stint'])
        .apply(get_best_consecutive_window, window=window_size, include_groups=False)
        .reset_index()
    )

    if col_name not in stint_stats.columns:
        stint_stats[col_name] = np.nan

    driver_stats = (
        stint_stats.groupby('Driver')
        .agg({col_name: 'min'})
        .reset_index()
    )

    result = (
        pd.DataFrame({'Driver': all_drivers})
        .merge(driver_stats, on='Driver', how='left')
    )

    return result.sort_values(col_name).reset_index(drop=True)

def get_longrun_deg_rate(year: int, gp: str, session_type: str, compound: str = 'MEDIUM', min_laps: int = 5) -> pd.DataFrame:
    """
    Calculates the tire degradation rate (slope of linear regression) 
    in seconds per lap during a long run stint.
    
    A positive slope indicates net degradation (lap times getting slower).
    A negative slope indicates fuel burn outpaces tire wear (lap times getting faster).
    """
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=False, weather=False, messages=False)
    
    prefix = session_type.lower()
    comp_lower = compound.lower()
    col_name = f'{prefix}_longrun_{comp_lower}_deg_rate'
    
    laps = session.laps.copy()
    all_drivers = laps['Driver'].unique()

    best_lap_time = laps['LapTime'].min()
    max_acceptable_lap = best_lap_time * 1.07

    valid_laps = laps[
        (laps['Compound'] == compound.upper())
        & (laps['TrackStatus'] == '1')
        & (laps['PitOutTime'].isna())
        & (laps['PitInTime'].isna())
        & (laps['LapTime'].notna())
        & (laps['LapTime'] > pd.Timedelta(0))
    ].copy()

    valid_laps = _keep_representative_laps(valid_laps)

    if valid_laps.empty:
        result = pd.DataFrame({'Driver': all_drivers})
        result[col_name] = np.nan
        return result

    valid_laps['LapTime_s'] = valid_laps['LapTime'].dt.total_seconds()
    valid_laps = valid_laps.sort_values(by=['Driver', 'Stint', 'LapNumber'])

    def calculate_deg_slope(df: pd.DataFrame) -> pd.Series:
        if len(df) < min_laps:
            return pd.Series({col_name: np.nan, 'laps_in_stint': len(df)})
            
        x = df['TyreLife'].values
        y = df['LapTime_s'].values
        
        slope, _ = np.polyfit(x, y, 1)
        
        return pd.Series({col_name: slope, 'laps_in_stint': len(df)})

    stint_stats = (
        valid_laps.groupby(['Driver', 'Stint'])
        .apply(calculate_deg_slope, include_groups=False)
        .reset_index()
    )

    if col_name not in stint_stats.columns:
        stint_stats[col_name] = np.nan
        stint_stats['laps_in_stint'] = 0

    valid_stints = stint_stats.dropna(subset=[col_name])

    if valid_stints.empty:
        result = pd.DataFrame({'Driver': all_drivers})
        result[col_name] = np.nan
        return result

    best_stints = (
        valid_stints.sort_values(['Driver', 'laps_in_stint'], ascending=[True, False])
        .drop_duplicates('Driver')
    )

    driver_stats = best_stints[['Driver', col_name]]

    result = (
        pd.DataFrame({'Driver': all_drivers})
        .merge(driver_stats, on='Driver', how='left')
    )

    return result.sort_values(col_name).reset_index(drop=True)

def get_longrun_deg_total(year: int, gp: str, session_type: str, compound: str = 'MEDIUM', min_laps: int = 5, expected_stint_length: int = 15) -> pd.DataFrame:
    """
    Calculates the estimated total time loss (in seconds) over a projected full stint.
    Formula: Degradation Rate (slope) * Expected Stint Length.
    
    This translates raw degradation per lap into a strategic variable for 
    pit-window calculations and undercut vulnerability models.
    """
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=False, weather=False, messages=False)
    
    prefix = session_type.lower()
    comp_lower = compound.lower()
    col_rate = f'{prefix}_longrun_{comp_lower}_deg_rate'
    col_total = f'{prefix}_longrun_{comp_lower}_deg_total'
    
    laps = session.laps.copy()
    all_drivers = laps['Driver'].unique()

    best_lap_time = laps['LapTime'].min()
    max_acceptable_lap = best_lap_time * 1.07

    valid_laps = laps[
        (laps['Compound'] == compound.upper())
        & (laps['TrackStatus'] == '1')
        & (laps['PitOutTime'].isna())
        & (laps['PitInTime'].isna())
        & (laps['LapTime'].notna())
        & (laps['LapTime'] > pd.Timedelta(0))
    ].copy()

    valid_laps = _keep_representative_laps(valid_laps)

    if valid_laps.empty:
        result = pd.DataFrame({'Driver': all_drivers})
        result[col_rate] = np.nan
        result[col_total] = np.nan
        return result

    valid_laps['LapTime_s'] = valid_laps['LapTime'].dt.total_seconds()
    valid_laps = valid_laps.sort_values(by=['Driver', 'Stint', 'LapNumber'])

    def calculate_deg_metrics(df: pd.DataFrame) -> pd.Series:
        if len(df) < min_laps:
            return pd.Series({
                col_rate: np.nan, 
                col_total: np.nan, 
                'laps_in_stint': len(df)
            })
            
        x = df['TyreLife'].values
        y = df['LapTime_s'].values
        
        slope, _ = np.polyfit(x, y, 1)
        
        deg_total = slope * expected_stint_length
        
        return pd.Series({
            col_rate: slope, 
            col_total: deg_total, 
            'laps_in_stint': len(df)
        })

    stint_stats = (
        valid_laps.groupby(['Driver', 'Stint'])
        .apply(calculate_deg_metrics, include_groups=False)
        .reset_index()
    )

    if col_total not in stint_stats.columns:
        stint_stats[col_rate] = np.nan
        stint_stats[col_total] = np.nan
        stint_stats['laps_in_stint'] = 0

    valid_stints = stint_stats.dropna(subset=[col_total])

    if valid_stints.empty:
        result = pd.DataFrame({'Driver': all_drivers})
        result[col_rate] = np.nan
        result[col_total] = np.nan
        return result

    best_stints = (
        valid_stints.sort_values(['Driver', 'laps_in_stint'], ascending=[True, False])
        .drop_duplicates('Driver')
    )

    driver_stats = best_stints[['Driver', col_rate, col_total]]

    result = (
        pd.DataFrame({'Driver': all_drivers})
        .merge(driver_stats, on='Driver', how='left')
    )

    return result.sort_values(col_total).reset_index(drop=True)

def get_longrun_consistency(year: int, gp: str, session_type: str, compound: str = 'MEDIUM', min_laps: int = 5) -> pd.DataFrame:
    """
    Calculates the long run consistency (Standard Deviation) of a driver 
    during a stint. 
    
    A lower value indicates a highly consistent driver (metronome). 
    A higher value indicates erratic driving, mistakes, or dealing with traffic.
    Applies tire degradation correction to isolate driver consistency 
    from natural tire wear and fuel burn.
    """
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=False, weather=False, messages=False)
    
    prefix = session_type.lower()
    comp_lower = compound.lower()
    col_name = f'{prefix}_longrun_{comp_lower}_consistency'
    
    laps = session.laps.copy()
    all_drivers = laps['Driver'].unique()

    best_lap_time = laps['LapTime'].min()
    max_acceptable_lap = best_lap_time * 1.07

    valid_laps = laps[
        (laps['Compound'] == compound.upper())
        & (laps['TrackStatus'] == '1')
        & (laps['PitOutTime'].isna())
        & (laps['PitInTime'].isna())
        & (laps['LapTime'].notna())
        & (laps['LapTime'] > pd.Timedelta(0))
    ].copy()

    valid_laps = _keep_representative_laps(valid_laps)

    if valid_laps.empty:
        result = pd.DataFrame({'Driver': all_drivers})
        result[col_name] = np.nan
        return result

    valid_laps['LapTime_s'] = valid_laps['LapTime'].dt.total_seconds()

    if compound.upper() == 'SOFT':
        deg_correction = 0.12
    elif compound.upper() == 'HARD':
        deg_correction = 0.05
    else:
        deg_correction = 0.08 

    valid_laps['LapTime_corrected'] = valid_laps['LapTime_s'] - (valid_laps['TyreLife'] * deg_correction)
    valid_laps = valid_laps.sort_values(by=['Driver', 'Stint', 'LapNumber'])

    def calculate_std_dev(df: pd.DataFrame) -> pd.Series:
        if len(df) < min_laps:
            return pd.Series({col_name: np.nan, 'laps_in_stint': len(df)})
            
        consistency = df['LapTime_corrected'].std()
        
        return pd.Series({col_name: consistency, 'laps_in_stint': len(df)})

    stint_stats = (
        valid_laps.groupby(['Driver', 'Stint'])
        .apply(calculate_std_dev, include_groups=False)
        .reset_index()
    )

    if col_name not in stint_stats.columns:
        stint_stats[col_name] = np.nan
        stint_stats['laps_in_stint'] = 0

    valid_stints = stint_stats.dropna(subset=[col_name])

    if valid_stints.empty:
        result = pd.DataFrame({'Driver': all_drivers})
        result[col_name] = np.nan
        return result

    best_stints = (
        valid_stints.sort_values(['Driver', 'laps_in_stint', col_name], ascending=[True, False, True])
        .drop_duplicates('Driver')
    )

    driver_stats = best_stints[['Driver', col_name]]

    result = (
        pd.DataFrame({'Driver': all_drivers})
        .merge(driver_stats, on='Driver', how='left')
    )

    return result.sort_values(col_name).reset_index(drop=True)

def get_longrun_compound(year: int, gp: str, session_type: str, min_laps: int = 5) -> pd.DataFrame:
    """
    Identifies the tire compound used by each driver for their primary long run stint.
    
    If a driver completed multiple long runs, it selects the compound from 
    the stint with the highest number of valid consecutive laps.
    Outputs a categorical variable (e.g., 'SOFT', 'MEDIUM', 'HARD') for ML normalization.
    """
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=False, weather=False, messages=False)
    
    prefix = session_type.lower()
    col_compound = f'{prefix}_longrun_compound'
    col_laps = f'{prefix}_longrun_stint_laps'
    
    laps = session.laps.copy()
    all_drivers = laps['Driver'].unique()

    best_lap_time = laps['LapTime'].min()
    max_acceptable_lap = best_lap_time * 1.07

    valid_laps = laps[
        (laps['TrackStatus'] == '1')
        & (laps['PitOutTime'].isna())
        & (laps['PitInTime'].isna())
        & (laps['LapTime'].notna())
        & (laps['LapTime'] > pd.Timedelta(0))
        & (laps['Compound'].notna())
    ].copy()

    valid_laps = _keep_representative_laps(valid_laps)

    if valid_laps.empty:
        result = pd.DataFrame({'Driver': all_drivers})
        result[col_compound] = np.nan
        result[col_laps] = 0
        return result

    stint_stats = (
        valid_laps.groupby(['Driver', 'Stint', 'Compound'])
        .agg(laps_in_stint=('LapTime', 'count'))
        .reset_index()
    )

    valid_stints = stint_stats[stint_stats['laps_in_stint'] >= min_laps].copy()

    if valid_stints.empty:
        result = pd.DataFrame({'Driver': all_drivers})
        result[col_compound] = np.nan
        result[col_laps] = 0
        return result

    best_stints = (
        valid_stints.sort_values(['Driver', 'laps_in_stint'], ascending=[True, False])
        .drop_duplicates('Driver')
    )

    driver_stats = best_stints[['Driver', 'Compound', 'laps_in_stint']].rename(
        columns={
            'Compound': col_compound,
            'laps_in_stint': col_laps
        }
    )

    result = (
        pd.DataFrame({'Driver': all_drivers})
        .merge(driver_stats, on='Driver', how='left')
        .fillna({col_laps: 0})
    )
    
    result[col_laps] = result[col_laps].astype(int)

    return result.sort_values(col_laps, ascending=False).reset_index(drop=True)

def get_fuel_corrected_pace(year: int, gp: str, session_type: str, compound: str = 'MEDIUM', window_size: int = 5) -> pd.DataFrame:
    """
    Calculates the 'True Base Pace' by correcting lap times for both 
    dynamic tire degradation and estimated fuel loads.
    
    Fuel load is estimated dynamically based on the total length of the stint.
    Allows fair pace comparison between drivers running different fuel programs.
    """
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=False, weather=False, messages=False)
    
    prefix = session_type.lower()
    comp_lower = compound.lower()
    
    col_pace = f'{prefix}_{comp_lower}_fuel_corrected_pace'
    col_est_fuel = f'{prefix}_{comp_lower}_est_start_fuel_kg'
    
    laps = session.laps.copy()
    all_drivers = laps['Driver'].unique()

    best_lap_time = laps['LapTime'].min()
    max_acceptable_lap = best_lap_time * 1.07

    valid_laps = laps[
        (laps['Compound'] == compound.upper())
        & (laps['TrackStatus'] == '1')
        & (laps['PitOutTime'].isna())
        & (laps['PitInTime'].isna())
        & (laps['LapTime'].notna())
        & (laps['LapTime'] > pd.Timedelta(0))
    ].copy()

    valid_laps = _keep_representative_laps(valid_laps)

    if valid_laps.empty:
        result = pd.DataFrame({'Driver': all_drivers})
        result[col_pace] = np.nan
        result[col_est_fuel] = np.nan
        return result

    valid_laps['LapTime_s'] = valid_laps['LapTime'].dt.total_seconds()
    
    FUEL_CONSUMPTION_PER_LAP = 1.5
    FUEL_MARGIN_KG = 5.0
    FUEL_PENALTY_PER_KG = 0.03
    
    if compound.upper() == 'SOFT':
        deg_correction = 0.12
    elif compound.upper() == 'HARD':
        deg_correction = 0.05
    else:
        deg_correction = 0.08 

    valid_laps = valid_laps.sort_values(by=['Driver', 'Stint', 'LapNumber'])

    def calculate_true_base_pace(df: pd.DataFrame) -> pd.Series:
        if len(df) < window_size:
            return pd.Series({col_pace: np.nan, col_est_fuel: np.nan})
            
        total_stint_laps = len(df)
        start_fuel_kg = (total_stint_laps * FUEL_CONSUMPTION_PER_LAP) + FUEL_MARGIN_KG
        
        corrected_lap_times = []
        for i, row in enumerate(df.itertuples()):
            current_fuel_kg = start_fuel_kg - (i * FUEL_CONSUMPTION_PER_LAP)
            
            fuel_time_penalty = current_fuel_kg * FUEL_PENALTY_PER_KG
            
            tire_time_penalty = row.TyreLife * deg_correction
            
            lap_time_0kg = row.LapTime_s - tire_time_penalty - fuel_time_penalty
            corrected_lap_times.append(lap_time_0kg)
            
        df_calc = pd.DataFrame({'LapTime_0kg': corrected_lap_times, 'LapNumber': df['LapNumber'].values})
        
        rolling_avg = df_calc['LapTime_0kg'].rolling(window=window_size).mean()
        
        rolling_lap_max = df_calc['LapNumber'].rolling(window=window_size).max()
        rolling_lap_min = df_calc['LapNumber'].rolling(window=window_size).min()
        is_strictly_consecutive = (rolling_lap_max - rolling_lap_min) == (window_size - 1)
        
        valid_windows = rolling_avg[is_strictly_consecutive]
        
        if valid_windows.empty:
            return pd.Series({col_pace: np.nan, col_est_fuel: start_fuel_kg})
            
        return pd.Series({
            col_pace: valid_windows.min(), 
            col_est_fuel: start_fuel_kg
        })

    stint_stats = (
        valid_laps.groupby(['Driver', 'Stint'])
        .apply(calculate_true_base_pace, include_groups=False)
        .reset_index()
    )

    if col_pace not in stint_stats.columns:
        stint_stats[col_pace] = np.nan
        stint_stats[col_est_fuel] = np.nan

    driver_stats = (
        stint_stats.groupby('Driver')
        .agg({
            col_pace: 'min',
            col_est_fuel: 'max'
        })
        .reset_index()
    )

    result = (
        pd.DataFrame({'Driver': all_drivers})
        .merge(driver_stats, on='Driver', how='left')
    )

    return result.sort_values(col_pace).reset_index(drop=True)

def get_ers_efficiency_proxy(year: int, gp: str, session_type: str) -> pd.DataFrame:
    """
    Calculates an ERS Efficiency Proxy by analyzing micro-telemetry of the fastest lap.
    It measures the ratio of time spent at Full Throttle (Deploy) vs Braking (Recovery).
    Returns a delta relative to the field median.
    
    Positive proxy = Highly efficient (spends more time deploying relative to recovering).
    Negative proxy = Inefficient (heavy braking, potentially clipping or poor traction).
    """
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=True, weather=False, messages=False)
    
    prefix = session_type.lower()
    col_proxy = f'{prefix}_ers_efficiency_proxy'
    
    laps = session.laps.pick_quicklaps().copy()
    drivers = laps['Driver'].unique()
    
    ers_data = []

    for drv in drivers:
        drv_laps = laps.pick_drivers(drv)
        if drv_laps.empty:
            continue
            
        best_lap = drv_laps.loc[drv_laps['LapTime'].idxmin()]
        
        try:
            tel = best_lap.get_telemetry()
        except:
            continue
            
        if tel.empty:
            continue

        tel['Time_delta'] = tel['Time'].dt.total_seconds().diff().fillna(0)

        braking_time = tel.loc[tel['Brake'] > 0, 'Time_delta'].sum()

        accel_time = tel.loc[tel['Throttle'] >= 95, 'Time_delta'].sum()

        if braking_time > 0:
            ers_ratio = accel_time / braking_time
        else:
            ers_ratio = np.nan

        ers_data.append({
            'Driver': drv,
            'braking_time_s': braking_time,
            'full_throttle_time_s': accel_time,
            'ers_ratio_raw': ers_ratio
        })

    df = pd.DataFrame(ers_data)
    
    if df.empty:
        return df

    median_ratio = df['ers_ratio_raw'].median()
    
    df[col_proxy] = df['ers_ratio_raw'] - median_ratio

    df = df.sort_values(col_proxy, ascending=False).reset_index(drop=True)
    
    return df[['Driver', 'braking_time_s', 'full_throttle_time_s', col_proxy]]

def get_lift_and_coast_laps(year: int, gp: str, session_type: str = 'FP2', min_laps: int = 5) -> pd.DataFrame:
    """
    Calculates the average time (in seconds) per lap a driver spends doing 'Lift & Coast' 
    during their primary long run.
    
    Universal L&C Definition: Speed > 250 km/h AND Throttle < 20% AND Brakes == 0.
    High values indicate battery starvation, severe fuel saving, or PU thermal management.
    """
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=True, weather=False, messages=False)
    
    prefix = session_type.lower()
    col_lc_time = f'{prefix}_avg_lift_coast_time_s'
    
    laps = session.laps.copy()
    all_drivers = laps['Driver'].unique()
    
    valid_laps = laps[
        (laps['TrackStatus'] == '1') & 
        (laps['PitOutTime'].isna()) & 
        (laps['PitInTime'].isna()) &
        (laps['LapTime'].notna()) &
        (laps['LapTime'] > pd.Timedelta(0))
    ].copy()
    
    if valid_laps.empty:
        result = pd.DataFrame({'Driver': all_drivers, col_lc_time: np.nan})
        return result

    stint_counts = valid_laps.groupby(['Driver', 'Stint']).size()
    long_runs = stint_counts[stint_counts >= min_laps].reset_index()
    
    results = []
    
    for drv in all_drivers:
        drv_long_runs = long_runs[long_runs['Driver'] == drv]
        
        if drv_long_runs.empty:
            results.append({'Driver': drv, col_lc_time: np.nan})
            continue
            
        best_stint = drv_long_runs.sort_values(0, ascending=False).iloc[0]['Stint']
        drv_laps = valid_laps[(valid_laps['Driver'] == drv) & (valid_laps['Stint'] == best_stint)]
        
        total_lc_time = 0
        valid_lap_count = 0
        
        for _, lap in drv_laps.iterrows():
            try:
                tel = lap.get_telemetry()
                if tel.empty:
                    continue
                    
                tel['Time_delta'] = tel['Time'].dt.total_seconds().diff().fillna(0)
                
                lc_mask = (tel['Speed'] > 250) & (tel['Throttle'] < 20) & (tel['Brake'] == 0)
                lap_lc_time = tel.loc[lc_mask, 'Time_delta'].sum()
                
                total_lc_time += lap_lc_time
                valid_lap_count += 1
            except Exception:
                continue
        
        if valid_lap_count > 0:
            avg_lc = total_lc_time / valid_lap_count
        else:
            avg_lc = np.nan
            
        results.append({'Driver': drv, col_lc_time: avg_lc})
        
    df = pd.DataFrame(results)
    
    return df.sort_values(col_lc_time, ascending=False).reset_index(drop=True)

def get_pu_deployment_asymmetry(year: int, gp: str, session_type: str = 'FP2') -> pd.DataFrame:
    """
    Calculates the ratio between Sector 1 and Sector 3 times for the fastest lap.
    Acts as a proxy for Power Unit (PU) deployment strategy and aero balance.
    
    A higher ratio indicates more time spent in S1 relative to S3 (saving battery for S3).
    A lower ratio indicates aggressive deployment in S1 and clipping in S3.
    """
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=False, weather=False, messages=False)
    
    prefix = session_type.lower()
    col_s1 = f'{prefix}_best_s1_s'
    col_s3 = f'{prefix}_best_s3_s'
    col_ratio = f'{prefix}_s1_vs_s3_ratio'
    col_asymmetry = f'{prefix}_pu_asymmetry_delta'
    
    laps = session.laps.copy()
    all_drivers = laps['Driver'].unique()

    valid_laps = laps[
        (laps['TrackStatus'] == '1')
        & (laps['PitOutTime'].isna())
        & (laps['PitInTime'].isna())
        & (laps['LapTime'].notna())
        & (laps['Sector1Time'].notna())
        & (laps['Sector3Time'].notna())
    ].copy()

    if valid_laps.empty:
        result = pd.DataFrame({'Driver': all_drivers})
        result[col_ratio] = np.nan
        result[col_asymmetry] = np.nan
        return result

    best_idx = valid_laps.groupby('Driver')['LapTime'].idxmin()
    best_laps = valid_laps.loc[best_idx].copy()
    
    best_laps[col_s1] = best_laps['Sector1Time'].dt.total_seconds()
    best_laps[col_s3] = best_laps['Sector3Time'].dt.total_seconds()
    
    best_laps[col_ratio] = best_laps[col_s1] / best_laps[col_s3]
    
    median_ratio = best_laps[col_ratio].median()
    
    best_laps[col_asymmetry] = best_laps[col_ratio] - median_ratio

    driver_stats = best_laps[['Driver', col_s1, col_s3, col_ratio, col_asymmetry]]

    result = (
        pd.DataFrame({'Driver': all_drivers})
        .merge(driver_stats, on='Driver', how='left')
    )

    return result.sort_values(col_asymmetry, ascending=False).reset_index(drop=True)

def get_speed_trap_variance(year: int, gp: str, session_type: str = 'FP2', min_laps: int = 5) -> pd.DataFrame:
    """
    Calculates the standard deviation (variance proxy) of the maximum speed 
    reached per lap during a driver's primary long run.
    
    High variance indicates erratic ERS deployment, clipping, or heavy traffic.
    Low variance indicates a perfectly tuned ERS map and predictable race pace.
    """
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=True, weather=False, messages=False)
    
    prefix = session_type.lower()
    col_std = f'{prefix}_speed_trap_std_kmh'
    col_mean = f'{prefix}_speed_trap_mean_kmh'
    
    laps = session.laps.copy()
    all_drivers = laps['Driver'].unique()
    
    valid_laps = laps[
        (laps['TrackStatus'] == '1') & 
        (laps['PitOutTime'].isna()) & 
        (laps['PitInTime'].isna()) &
        (laps['LapTime'].notna())
    ].copy()
    
    if valid_laps.empty:
        return pd.DataFrame({'Driver': all_drivers, col_std: np.nan, col_mean: np.nan})

    stint_counts = valid_laps.groupby(['Driver', 'Stint']).size()
    long_runs = stint_counts[stint_counts >= min_laps].reset_index()
    
    results = []
    
    for drv in all_drivers:
        drv_long_runs = long_runs[long_runs['Driver'] == drv]
        
        if drv_long_runs.empty:
            results.append({'Driver': drv, col_std: np.nan, col_mean: np.nan})
            continue
            
        best_stint = drv_long_runs.sort_values(0, ascending=False).iloc[0]['Stint']
        drv_laps = valid_laps[(valid_laps['Driver'] == drv) & (valid_laps['Stint'] == best_stint)]
        
        lap_top_speeds = []
        
        for _, lap in drv_laps.iterrows():
            try:
                tel = lap.get_telemetry()
                if not tel.empty:
                    max_speed = tel['Speed'].max()
                    lap_top_speeds.append(max_speed)
            except Exception:
                continue
        
        if len(lap_top_speeds) > 1:
            speed_std = np.std(lap_top_speeds, ddof=1)
            speed_mean = np.mean(lap_top_speeds)
        else:
            speed_std = np.nan
            speed_mean = np.nan
            
        results.append({'Driver': drv, col_std: speed_std, col_mean: speed_mean})
        
    df = pd.DataFrame(results)
    
    return df.sort_values(col_std, ascending=False).reset_index(drop=True)

def get_qualy_sim_delta(year: int, gp: str, session_type: str = 'FP3', compound: str = 'SOFT') -> pd.DataFrame:
    """
    Calculates the Qualifying Simulation Delta (in seconds) for each driver.
    Finds each driver's absolute fastest lap on the specified compound and 
    compares it to the overall fastest lap of the session.
    
    The leader will have a delta of 0.000.
    """
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=False, weather=False, messages=False)
    
    prefix = session_type.lower()
    comp_lower = compound.lower()
    
    col_laptime = f'{prefix}_{comp_lower}_best_lap_s'
    col_delta = f'{prefix}_{comp_lower}_best_lap_delta'
    
    laps = session.laps.copy()
    all_drivers = laps['Driver'].unique()

    valid_laps = laps[
        (laps['Compound'] == compound.upper())
        & (laps['TrackStatus'] == '1')
        & (laps['PitOutTime'].isna())
        & (laps['PitInTime'].isna())
        & (laps['LapTime'].notna())
    ].copy()

    if valid_laps.empty:
        result = pd.DataFrame({'Driver': all_drivers})
        result[col_laptime] = np.nan
        result[col_delta] = np.nan
        return result

    driver_bests = (
        valid_laps.groupby('Driver')['LapTime']
        .min()
        .reset_index()
    )
    
    overall_fastest_lap = driver_bests['LapTime'].min()

    driver_bests[col_laptime] = driver_bests['LapTime'].dt.total_seconds()
    driver_bests[col_delta] = (driver_bests['LapTime'] - overall_fastest_lap).dt.total_seconds()

    driver_stats = driver_bests[['Driver', col_laptime, col_delta]]

    result = (
        pd.DataFrame({'Driver': all_drivers})
        .merge(driver_stats, on='Driver', how='left')
    )

    return result.sort_values(col_delta).reset_index(drop=True)

def get_fp3_vs_fp2_improvement(year: int, gp: str, compound: str = 'SOFT') -> pd.DataFrame:
    """
    Calculates the overnight setup improvement from FP2 to FP3.
    Formula: FP2 Best Lap - FP3 Best Lap (in seconds).
    
    A positive value indicates the car got faster (improved setup + track evolution).
    A negative value indicates the car got slower (failed setup, bad correlation, or wet track).
    """
    fp2 = fastf1.get_session(year, gp, 'FP2')
    fp3 = fastf1.get_session(year, gp, 'FP3')
    
    fp2.load(telemetry=False, weather=False, messages=False)
    fp3.load(telemetry=False, weather=False, messages=False)
    
    comp_lower = compound.lower()
    
    def get_best_laps(session_laps: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Helper to extract the absolute best valid lap for each driver in a session."""
        valid_laps = session_laps[
            (session_laps['Compound'] == compound.upper())
            & (session_laps['TrackStatus'] == '1')
            & (session_laps['PitOutTime'].isna())
            & (session_laps['PitInTime'].isna())
            & (session_laps['LapTime'].notna())
        ].copy()
        
        if valid_laps.empty:
            return pd.DataFrame({'Driver': session_laps['Driver'].unique(), f'{prefix}_{comp_lower}_best_s': np.nan})
            
        best_laps = (
            valid_laps.groupby('Driver')['LapTime']
            .min()
            .dt.total_seconds()
            .reset_index()
        )
        return best_laps.rename(columns={'LapTime': f'{prefix}_{comp_lower}_best_s'})

    fp2_bests = get_best_laps(fp2.laps, 'fp2')
    fp3_bests = get_best_laps(fp3.laps, 'fp3')
    
    all_drivers = pd.Series(
        list(set(fp2.laps['Driver'].unique()).union(set(fp3.laps['Driver'].unique()))), 
        name='Driver'
    )
    df_all = pd.DataFrame(all_drivers)
    
    result = (
        df_all
        .merge(fp2_bests, on='Driver', how='left')
        .merge(fp3_bests, on='Driver', how='left')
    )
    
    col_improvement = f'fp3_vs_fp2_{comp_lower}_improvement'
    result[col_improvement] = result[f'fp2_{comp_lower}_best_s'] - result[f'fp3_{comp_lower}_best_s']
    
    return result.sort_values(col_improvement, ascending=False).reset_index(drop=True)

def get_sector_improvement_vs_fp2(year: int, gp: str, compound: str = 'SOFT') -> pd.DataFrame:
    """
    Calculates the overnight setup improvement broken down by sector (FP2 to FP3).
    Formula: FP2 Sector Time - FP3 Sector Time (in seconds).
    
    A positive value means the sector got faster (improvement).
    A negative value means the sector got slower (lost time).
    Reveals aerodynamic trade-offs (e.g., stripping wing for straight line speed).
    """
    fp2 = fastf1.get_session(year, gp, 'FP2')
    fp3 = fastf1.get_session(year, gp, 'FP3')
    
    fp2.load(telemetry=False, weather=False, messages=False)
    fp3.load(telemetry=False, weather=False, messages=False)
    
    comp_lower = compound.lower()
    
    def get_best_lap_sectors(session_laps: pd.DataFrame, prefix: str) -> pd.DataFrame:
        """Finds the absolute best valid lap and extracts its sector times."""
        valid_laps = session_laps[
            (session_laps['Compound'] == compound.upper())
            & (session_laps['TrackStatus'] == '1')
            & (session_laps['PitOutTime'].isna())
            & (session_laps['PitInTime'].isna())
            & (session_laps['LapTime'].notna())
            & (session_laps['Sector1Time'].notna())
            & (session_laps['Sector2Time'].notna())
            & (session_laps['Sector3Time'].notna())
        ].copy()
        
        if valid_laps.empty:
            return pd.DataFrame({'Driver': session_laps['Driver'].unique()})
            
        best_idx = valid_laps.groupby('Driver')['LapTime'].idxmin()
        best_laps = valid_laps.loc[best_idx].copy()
        
        res = pd.DataFrame({
            'Driver': best_laps['Driver'],
            f'{prefix}_s1_s': best_laps['Sector1Time'].dt.total_seconds(),
            f'{prefix}_s2_s': best_laps['Sector2Time'].dt.total_seconds(),
            f'{prefix}_s3_s': best_laps['Sector3Time'].dt.total_seconds()
        })
        return res

    fp2_sectors = get_best_lap_sectors(fp2.laps, 'fp2')
    fp3_sectors = get_best_lap_sectors(fp3.laps, 'fp3')
    
    all_drivers = pd.Series(
        list(set(fp2.laps['Driver'].unique()).union(set(fp3.laps['Driver'].unique()))), 
        name='Driver'
    )
    
    df = pd.DataFrame(all_drivers)
    df = (df.merge(fp2_sectors, on='Driver', how='left')
            .merge(fp3_sectors, on='Driver', how='left'))
    
    df['fp3_s1_delta_vs_fp2'] = df['fp2_s1_s'] - df['fp3_s1_s']
    df['fp3_s2_delta_vs_fp2'] = df['fp2_s2_s'] - df['fp3_s2_s']
    df['fp3_s3_delta_vs_fp2'] = df['fp2_s3_s'] - df['fp3_s3_s']
    
    df['total_imp'] = df['fp3_s1_delta_vs_fp2'] + df['fp3_s2_delta_vs_fp2'] + df['fp3_s3_delta_vs_fp2']
    
    final_cols = ['Driver', 'fp3_s1_delta_vs_fp2', 'fp3_s2_delta_vs_fp2', 'fp3_s3_delta_vs_fp2']
    
    return df.sort_values('total_imp', ascending=False)[final_cols].reset_index(drop=True)

def get_fp3_qualy_sim_context(year: int, gp: str, session_type: str = 'FP3') -> pd.DataFrame:
    """
    Extracts high-fidelity context for the qualifying simulation.
    Evaluates the compound and the exact physical degradation (TyreLife) 
    at the moment the fastest lap was set to separate true Qualy sims 
    from compromised runs.
    """
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=False, weather=False, messages=False)
    
    prefix = session_type.lower()
    col_tyre = f'{prefix}_best_lap_compound'
    col_age = f'{prefix}_best_lap_tyre_age'
    col_true_sim_flag = f'{prefix}_is_true_qualy_sim'
    
    laps = session.laps.copy()
    all_drivers = laps['Driver'].unique()

    valid_laps = laps[
        (laps['TrackStatus'] == '1')
        & (laps['PitOutTime'].isna())
        & (laps['PitInTime'].isna())
        & (laps['LapTime'].notna())
    ].copy()

    if valid_laps.empty:
        result = pd.DataFrame({'Driver': all_drivers})
        result[col_tyre] = 'UNKNOWN'
        result[col_age] = np.nan
        result[col_true_sim_flag] = 0
        return result

    best_idx = valid_laps.groupby('Driver')['LapTime'].idxmin()
    best_laps = valid_laps.loc[best_idx].copy()

    best_laps[col_tyre] = best_laps['Compound']
    
    best_laps[col_age] = pd.to_numeric(best_laps['TyreLife'], errors='coerce').fillna(1.0)
    
    best_laps[col_true_sim_flag] = (
        (best_laps['Compound'] == 'SOFT') & 
        (best_laps[col_age] <= 3.0)
    ).astype(int)

    driver_stats = best_laps[['Driver', col_tyre, col_age, col_true_sim_flag]]

    result = (
        pd.DataFrame({'Driver': all_drivers})
        .merge(driver_stats, on='Driver', how='left')
    )
    
    result[col_tyre] = result[col_tyre].fillna('NONE')
    result[col_age] = result[col_age].fillna(100.0)
    result[col_true_sim_flag] = result[col_true_sim_flag].fillna(0).astype(int)

    return result

def get_track_evolution(year: int, gp: str, session_type: str = 'FP3') -> pd.DataFrame:
    """
    Calculates the global track evolution over the course of the session.
    Splits the session dynamically into halves and compares the absolute best 
    lap of the early phase vs the late phase.
    
    A positive value (e.g., 1.5s) means massive rubbering-in (track getting faster).
    A negative value means the track got slower (rain, wind, or cooling temperatures).
    """
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=False, weather=False, messages=False)
    
    prefix = session_type.lower()
    col_evo = f'{prefix}_track_evolution_s'
    
    laps = session.laps.copy()
    all_drivers = laps['Driver'].unique()

    valid_laps = laps[
        (laps['TrackStatus'] == '1') &
        (laps['PitOutTime'].isna()) &
        (laps['PitInTime'].isna()) &
        (laps['LapTime'].notna())
    ].copy()

    if valid_laps.empty:
        return pd.DataFrame({'Driver': all_drivers, col_evo: np.nan})

    session_start = valid_laps['Time'].min()
    session_end = valid_laps['Time'].max()
    midpoint = session_start + (session_end - session_start) / 2

    early_laps = valid_laps[valid_laps['Time'] <= midpoint]
    late_laps = valid_laps[valid_laps['Time'] > midpoint]

    if early_laps.empty or late_laps.empty:
        track_evo = np.nan
    else:
        best_early_s = early_laps['LapTime'].min().total_seconds()
        best_late_s = late_laps['LapTime'].min().total_seconds()
        
        track_evo = best_early_s - best_late_s

    df = pd.DataFrame({'Driver': all_drivers})
    df[col_evo] = track_evo

    return df

def get_grid_position_features(year: int, gp: str, session_type: str = 'R') -> pd.DataFrame:
    """
    Extracts the final starting grid positions for the race and creates 
    highly predictive engineered features for Machine Learning models.
    
    Handles edge cases like pit lane starts (usually represented as 0 in FastF1).
    Note: We use the 'R' (Race) session because it contains the final grid 
    after all grid penalties have been applied.
    """
    session = fastf1.get_session(year, gp, session_type)
    session.load(telemetry=False, weather=False, messages=False)
    
    results = session.results
    
    if results is None or results.empty:
        return pd.DataFrame({'Driver': [], 'grid_position': []})
        
    df = pd.DataFrame({
        'Driver': results['Abbreviation'],
        'raw_grid': results['GridPosition']
    })
    
    df['raw_grid'] = pd.to_numeric(df['raw_grid'], errors='coerce')
    
    df['started_from_pitlane'] = (df['raw_grid'] == 0).astype(int)
    
    df['grid_position'] = np.where(df['raw_grid'] == 0, 21, df['raw_grid'])
    df['grid_position'] = df['grid_position'].fillna(21)
    
    df['is_front_row'] = (df['grid_position'] <= 2).astype(int)
    
    df['started_top_10'] = (df['grid_position'] <= 10).astype(int)
    
    final_cols = [
        'Driver', 
        'grid_position', 
        'is_front_row', 
        'started_top_10', 
        'started_from_pitlane'
    ]
    
    return df[final_cols].sort_values('grid_position').reset_index(drop=True)

def get_qualy_deltas(year: int, gp: str) -> pd.DataFrame:
    """
    Extracts the Qualifying deltas to the Pole Position time.
    Calculates both the strict Q3 delta and an overall Best Qualy delta
    to avoid NaNs for drivers eliminated in Q1/Q2.
    """
    session = fastf1.get_session(year, gp, 'Q')
    session.load(telemetry=False, weather=False, messages=False)

    results = session.results.copy()

    if results is None or results.empty:
        return pd.DataFrame()

    df = pd.DataFrame({'Driver': results['Abbreviation']})

    pole_time = results['Q3'].min()
    pole_time_s = pole_time.total_seconds()

    df['q3_time_s'] = results['Q3'].dt.total_seconds()
    df['q3_delta_to_pole'] = df['q3_time_s'] - pole_time_s

    best_q_times = results[['Q1', 'Q2', 'Q3']].min(axis=1)
    df['best_q_time_s'] = best_q_times.dt.total_seconds()
    
    df['best_q_delta_to_pole'] = df['best_q_time_s'] - pole_time_s

    def get_elimination(row):
        if pd.notna(row['Q3']): return 'Q3'
        if pd.notna(row['Q2']): return 'Q2'
        if pd.notna(row['Q1']): return 'Q1'
        return 'DNQ'
        
    df['reached_session'] = results.apply(get_elimination, axis=1)

    return df

def get_team_info(year: int, gp: str, session_type: str) -> pd.DataFrame:
    """
    Extracts the team and driver mapping using fastf1.
    """
    try:
        session = fastf1.get_session(year, gp, session_type)
        session.load(telemetry=False, weather=False, messages=False)
        res = session.results
        
        teams = []
        for _, row in res.iterrows():
            teams.append({
                'Driver': row['Abbreviation'],
                'Team': row['TeamName']
            })
        return pd.DataFrame(teams)
    except Exception as e:
        print(f"Error obtaining team info for {year} {gp}: {e}")
        return pd.DataFrame(columns=['Driver', 'Team'])

def get_qualy_target_percentile(year: int, gp: str, target_session: str = 'Q') -> pd.DataFrame:
    """
    Gets the Qualifying (Q or SQ) results to calculate a Percentile Target.
    """
    try:
        session = fastf1.get_session(year, gp, target_session)
        session.load(telemetry=False, weather=False, messages=True)
        res = session.results
        
        q_data = []
        fastest_q_time = None
        
        cols_to_check = [c for c in ['Q1','Q2','Q3','SQ1','SQ2','SQ3'] if c in res.columns]
        for _, row in res.iterrows():
            times = [row[c] for c in cols_to_check if c in row and pd.notnull(row[c])]
            if times:
                best_time = min(times)
                if fastest_q_time is None or best_time < fastest_q_time:
                    fastest_q_time = best_time
        
        for _, row in res.iterrows():
            driver = row['Abbreviation']
            pos = row['Position']
            
            times = [row[c] for c in cols_to_check if c in row and pd.notnull(row[c])]
            q_delta = np.nan
            if times and fastest_q_time is not None:
                q_delta = (min(times) - fastest_q_time).total_seconds()
                
            q_data.append({
                'Driver': driver,
                'Qualy_Position': pos,
                'Qualy_Delta': q_delta
            })
            
        df_q = pd.DataFrame(q_data)
        df_q['Target_Percentile'] = df_q['Qualy_Delta'].rank(pct=True)
        return df_q
    except Exception as e:
        print(f"Error extracting {target_session} for {year} {gp}: {e}")
        return None

    final_cols = ['Driver', 'reached_session', 'q3_delta_to_pole', 'best_q_delta_to_pole']
    return df[final_cols].sort_values('best_q_delta_to_pole').reset_index(drop=True)

def get_q3_participation_flag(year: int, gp: str) -> pd.DataFrame:
    """
    Creates a binary feature indicating if a driver reached Q3.
    1 = Reached Q3 (Elite pace/Contender)
    0 = Eliminated in Q1 or Q2
    
    This is highly optimized for the first split in Decision Tree models.
    """
    session = fastf1.get_session(year, gp, 'Q')
    session.load(telemetry=False, weather=False, messages=False)

    results = session.results.copy()

    if results is None or results.empty:
        return pd.DataFrame({'Driver': [], 'q3_participation': []})

    df = pd.DataFrame({'Driver': results['Abbreviation']})

    df['q3_participation'] = results['Q3'].notna().astype(int)

    return df.sort_values('q3_participation', ascending=False).reset_index(drop=True)

def get_qualy_vs_fp3_improvement(year: int, gp: str) -> pd.DataFrame:
    """
    Calculates the 'Time Found' from FP3 to Qualifying.
    Formula: FP3 Best Lap - Qualy Best Lap (in seconds).
    
    A large positive value indicates significant engine mode step-up ("Sandbagging" in FP3).
    A small or negative value indicates the team was already at the limit in FP3, 
    or the driver made a mistake in Qualifying.
    """
    fp3 = fastf1.get_session(year, gp, 'FP3')
    q = fastf1.get_session(year, gp, 'Q')
    
    fp3.load(telemetry=False, weather=False, messages=False)
    q.load(telemetry=False, weather=False, messages=False)
    
    fp3_laps = fp3.laps.copy()
    valid_fp3 = fp3_laps[
        (fp3_laps['TrackStatus'] == '1') & 
        (fp3_laps['PitOutTime'].isna()) & 
        (fp3_laps['PitInTime'].isna()) & 
        (fp3_laps['LapTime'].notna())
    ]
    
    if valid_fp3.empty:
        fp3_bests = pd.DataFrame({'Driver': fp3_laps['Driver'].unique(), 'fp3_best_s': np.nan})
    else:
        fp3_bests = valid_fp3.groupby('Driver')['LapTime'].min().dt.total_seconds().reset_index()
        fp3_bests.rename(columns={'LapTime': 'fp3_best_s'}, inplace=True)
        
    q_results = q.results.copy()
    if q_results is None or q_results.empty:
        q_times = pd.DataFrame({'Driver': fp3_bests['Driver'].unique(), 'q3_time_s': np.nan, 'best_q_time_s': np.nan})
    else:
        q_times = pd.DataFrame({'Driver': q_results['Abbreviation']})
        q_times['q3_time_s'] = q_results['Q3'].dt.total_seconds()
        
        best_q = q_results[['Q1', 'Q2', 'Q3']].min(axis=1)
        q_times['best_q_time_s'] = best_q.dt.total_seconds()
        
    df = pd.merge(fp3_bests, q_times, on='Driver', how='outer')
    
    df['q3_vs_fp3_improvement'] = df['fp3_best_s'] - df['q3_time_s']
    df['best_q_vs_fp3_improvement'] = df['fp3_best_s'] - df['best_q_time_s']
    
    final_cols = ['Driver', 'fp3_best_s', 'best_q_time_s', 'q3_vs_fp3_improvement', 'best_q_vs_fp3_improvement']
    
    return df[final_cols].sort_values('best_q_vs_fp3_improvement', ascending=False).reset_index(drop=True)

def get_best_quali_relative_sector(year: int, gp: str) -> pd.DataFrame:
    """
    Identifies the strongest sector for each driver relative to the field during Qualifying.
    
    1. Finds the absolute best valid lap for each driver across Q1, Q2, Q3.
    2. Extracts S1, S2, S3 times from that lap.
    3. Calculates the delta to the fastest time in each sector across the whole session.
    4. Determines which sector had the smallest delta (their relatively best sector).
    """
    q_session = fastf1.get_session(year, gp, 'Q')
    q_session.load(telemetry=False, weather=False, messages=False)

    laps = q_session.laps.copy()
    all_drivers = laps['Driver'].unique()

    valid_laps = laps[
        (laps['TrackStatus'] == '1') &
        (laps['PitOutTime'].isna()) &
        (laps['PitInTime'].isna()) &
        (laps['LapTime'].notna()) &
        (laps['Sector1Time'].notna()) &
        (laps['Sector2Time'].notna()) &
        (laps['Sector3Time'].notna())
    ].copy()

    if valid_laps.empty:
        return pd.DataFrame({'Driver': all_drivers, 'quali_best_relative_sector': 'NONE'})

    best_lap_idx = valid_laps.groupby('Driver')['LapTime'].idxmin()
    best_laps = valid_laps.loc[best_lap_idx].copy()

    best_laps['s1_s'] = best_laps['Sector1Time'].dt.total_seconds()
    best_laps['s2_s'] = best_laps['Sector2Time'].dt.total_seconds()
    best_laps['s3_s'] = best_laps['Sector3Time'].dt.total_seconds()

    fastest_s1 = best_laps['s1_s'].min()
    fastest_s2 = best_laps['s2_s'].min()
    fastest_s3 = best_laps['s3_s'].min()

    best_laps['s1_delta'] = best_laps['s1_s'] - fastest_s1
    best_laps['s2_delta'] = best_laps['s2_s'] - fastest_s2
    best_laps['s3_delta'] = best_laps['s3_s'] - fastest_s3

    sector_delta_cols = ['s1_delta', 's2_delta', 's3_delta']
    best_laps['best_relative_sector_col'] = best_laps[sector_delta_cols].idxmin(axis=1)

    sector_map = {
        's1_delta': 'S1',
        's2_delta': 'S2',
        's3_delta': 'S3'
    }
    best_laps['quali_best_relative_sector'] = best_laps['best_relative_sector_col'].map(sector_map)

    final_cols = ['Driver', 'quali_best_relative_sector', 's1_delta', 's2_delta', 's3_delta']
    result = best_laps[final_cols]
    
    df_all = pd.DataFrame({'Driver': all_drivers})
    df_final = df_all.merge(result, on='Driver', how='left')
    df_final['quali_best_relative_sector'] = df_final['quali_best_relative_sector'].fillna('NONE')

    return df_final.sort_values(['quali_best_relative_sector', 's1_delta']).reset_index(drop=True)

    
