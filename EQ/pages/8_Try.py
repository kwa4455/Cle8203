# app.py
# Streamlit Air Quality Aggregation App
# - Detects minute / hourly / daily input automatically
# - For minute data: >= 45 one-minute values = valid hour
# - For minute/hourly data: >= 18 valid hours/records = valid day
# - Period averages require >= 75% valid daily capture
# - Includes excluded-data reporting after completeness validation

import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from functools import reduce
from dateutil import parser

# -------------------------
# Page setup
# -------------------------
st.set_page_config(
    page_title="Air Quality Dashboard Aggregator",
    page_icon="🌫️",
    layout="wide"
)

# -------------------------
# Completeness parameters
# -------------------------
DAILY_MIN_OBS = 18
PERIOD_MIN_CAPTURE = 0.75
HOURLY_MIN_MINUTE_OBS = 35

# -------------------------
# Utilities
# -------------------------
def to_csv_download(df: pd.DataFrame) -> BytesIO:
    return BytesIO(df.to_csv(index=False).encode("utf-8"))


def unique_key(tab: str, widget: str, label: str) -> str:
    return f"{widget}_{tab}_{label}"


# -------------------------
# Parsing & Standardization
# -------------------------
def robust_parse(series: pd.Series) -> pd.Series:
    """Apply multiple parsing strategies to maximize datetime detection."""

    # Convert to string and clean
    series = series.astype(str).str.strip()

    # Strategy 1: dayfirst=True
    dt1 = pd.to_datetime(series, errors="coerce", dayfirst=True)

    # Strategy 2: dayfirst=False
    #dt2 = pd.to_datetime(series, errors="coerce", dayfirst=False)

    # Combine both
    dt = dt1.fillna(nan)

    # Strategy 3: Unix timestamps (seconds & milliseconds)
    if dt.isna().mean() > 0.3:
        dt_sec = pd.to_datetime(series, errors="coerce", unit="s")
        dt = dt.fillna(dt_sec)

        dt_ms = pd.to_datetime(series, errors="coerce", unit="ms")
        dt = dt.fillna(dt_ms)

    # Strategy 4: dateutil fallback (handles messy formats)
    if dt.isna().any():
        def safe_parse(x):
            try:
                return parser.parse(x)
            except:
                return pd.NaT

        dt_fallback = series.apply(safe_parse)
        dt = dt.fillna(dt_fallback)

    return dt


def parse_dates(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    """
    Automatically detect the best datetime column and standardize to df['datetime'].
    
    Parameters:
        df (pd.DataFrame): Input dataframe
        debug (bool): If True, prints problematic values
    
    Returns:
        pd.DataFrame: DataFrame with standardized 'datetime' column
    """

    best_col = None
    best_dt = None
    max_valid = 0

    for col in df.columns:
        try:
            dt = robust_parse(df[col])
            valid_count = dt.notna().sum()

            if debug:
                print(f"{col}: {valid_count} valid dates")

            if valid_count > max_valid:
                max_valid = valid_count
                best_col = col
                best_dt = dt

        except Exception as e:
            if debug:
                print(f"Skipping column '{col}': {e}")

    # Apply best column
    if best_col and max_valid > 0:
        df = df.copy()
        df["datetime"] = best_dt

        # Debug failed values
        if debug:
            failed = df.loc[df["datetime"].isna(), best_col]
            if not failed.empty:
                print("\n⚠️ Failed to parse some values:")
                print(failed.unique()[:20])

        # Clean final output
        df = (
            df.dropna(subset=["datetime"])
              .sort_values("datetime")
              .reset_index(drop=True)
        )

        print(f"✅ Using column '{best_col}' as datetime ({max_valid} valid values)")
        return df

    print("⚠️ No valid datetime column found")
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    pm25_cols = ["pm25", "pm_2_5", "pm2.5", "pm25_avg", "pm2_5", "pm 2.5"]
    pm10_cols = ["pm10", "pm_10", "pm10_avg", "pm 10"]
    site_cols = ["site", "station", "location", "site_name", "sitename"]

    df = df.copy()
    rename_map = {}

    for col in df.columns:
        col_lower = col.strip().lower()
        if col_lower in [c.lower() for c in pm25_cols]:
            rename_map[col] = "pm25"
        elif col_lower in [c.lower() for c in pm10_cols]:
            rename_map[col] = "pm10"
        elif col_lower in [c.lower() for c in site_cols]:
            rename_map[col] = "site"

    df.rename(columns=rename_map, inplace=True)
    return df


def removal_box_plot(df: pd.DataFrame, col: str, lower_threshold: float, upper_threshold: float):
    filtered = df[(df[col] >= lower_threshold) & (df[col] <= upper_threshold)]
    return filtered, int((df[col] < lower_threshold).sum()), int((df[col] > upper_threshold).sum())


def cleaned(df: pd.DataFrame):
    df = df.copy()
    df = df.rename(columns=lambda x: x.strip().lower())

    cleaning_report = {
        "rows_input": len(df),
        "removed_missing_datetime_or_site": 0,
        "removed_missing_pollutants": 0,
        "pm25_below_range": 0,
        "pm25_above_range": 0,
        "pm10_below_range": 0,
        "pm10_above_range": 0,
        "rows_output": 0,
    }

    required_columns = ["datetime", "site", "pm25", "pm10"]
    keep = [c for c in required_columns if c in df.columns]
    df = df[keep].dropna(axis=1, how="all")

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    for p in ["pm25", "pm10"]:
        if p in df.columns:
            df[p] = pd.to_numeric(df[p], errors="coerce")

    before = len(df)
    pollutant_cols = [c for c in ["pm25", "pm10"] if c in df.columns]

    if pollutant_cols:
        df = df.dropna(subset=["datetime", "site"], how="any")
        after_datetime_site = len(df)
        cleaning_report["removed_missing_datetime_or_site"] = before - after_datetime_site

        before_poll = len(df)
        df = df.dropna(subset=pollutant_cols, how="all")
        cleaning_report["removed_missing_pollutants"] = before_poll - len(df)
    else:
        df = df.dropna(subset=["datetime", "site"], how="any")
        cleaning_report["removed_missing_datetime_or_site"] = before - len(df)

    if "pm25" in df.columns:
        filtered, low, high = removal_box_plot(df, "pm25", 1, 500)
        cleaning_report["pm25_below_range"] = low
        cleaning_report["pm25_above_range"] = high
        df = filtered

    if "pm10" in df.columns:
        filtered, low, high = removal_box_plot(df, "pm10", 1, 1000)
        cleaning_report["pm10_below_range"] = low
        cleaning_report["pm10_above_range"] = high
        df = filtered

    df["day"] = df["datetime"].dt.floor("D")
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.to_period("M").astype(str)
    df["quarter"] = df["datetime"].dt.to_period("Q").astype(str)
    df["dayofweek"] = df["datetime"].dt.day_name()
    df["weekday_type"] = df["datetime"].dt.weekday.map(lambda x: "Weekend" if x >= 5 else "Weekday")
    df["season"] = df["datetime"].dt.month.map(lambda m: "Harmattan" if m in [12, 1, 2] else "Non-Harmattan")

    daily_counts = df.groupby(["site", "month"])["day"].nunique().reset_index(name="daily_counts")
    sufficient_sites = daily_counts[daily_counts["daily_counts"] >= 1][["site", "month"]]
    df = df.merge(sufficient_sites, on=["site", "month"], how="inner")

    cleaning_report["rows_output"] = len(df)
    return df, cleaning_report


# -------------------------
# Resolution detection
# -------------------------
def detect_data_resolution(df: pd.DataFrame, datetime_col: str = "datetime") -> str:
    if datetime_col not in df.columns:
        return "unknown"

    dt = (
        pd.to_datetime(df[datetime_col], errors="coerce")
        .dropna()
        .sort_values()
        .drop_duplicates()
    )

    if len(dt) < 2:
        return "unknown"

    diffs = dt.diff().dropna().dt.total_seconds()
    if diffs.empty:
        return "unknown"

    typical_step = diffs.median()

    if typical_step <= 90:
        return "minute"
    if typical_step <= 5400:
        return "hourly"
    return "daily"


def get_dataset_resolution_label(df: pd.DataFrame) -> str:
    if "datetime" not in df.columns:
        return "unknown"
    return detect_data_resolution(df, "datetime")


# -------------------------
# Completeness helpers
# -------------------------
def build_valid_daily_means(
    df: pd.DataFrame,
    pollutant: str,
    daily_min_obs: int = DAILY_MIN_OBS,
    hourly_min_minute_obs: int = HOURLY_MIN_MINUTE_OBS,
) -> pd.DataFrame:
    """
    Rules:
    - Minute data: valid hour = >= 45 minute values; valid day = >= 18 valid hours
    - Hourly data: valid day = >= 18 hourly values
    - Daily data: day is already valid if at least one value exists
    """
    if pollutant not in df.columns or "datetime" not in df.columns or "site" not in df.columns:
        return pd.DataFrame(columns=["site", "day", pollutant])

    work = df[["site", "datetime", pollutant]].copy()
    work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce")
    work[pollutant] = pd.to_numeric(work[pollutant], errors="coerce")
    work = work.dropna(subset=["site", "datetime", pollutant])

    if work.empty:
        return pd.DataFrame(columns=["site", "day", pollutant])

    resolution = detect_data_resolution(work, "datetime")

    if resolution == "minute":
        work["hour"] = work["datetime"].dt.floor("h")

        hourly = (
            work.groupby(["site", "hour"], as_index=False)
            .agg(
                n_minute_obs=(pollutant, "count"),
                hourly_mean=(pollutant, "mean")
            )
        )

        hourly_valid = hourly[hourly["n_minute_obs"] >= hourly_min_minute_obs].copy()
        hourly_valid["day"] = hourly_valid["hour"].dt.floor("D")

        base = (
            hourly_valid.groupby(["site", "day"], as_index=False)
            .agg(
                n_obs=("hourly_mean", "count"),
                daily_mean=("hourly_mean", "mean")
            )
        )

        valid = base[base["n_obs"] >= daily_min_obs].copy()
        valid.rename(columns={"daily_mean": pollutant}, inplace=True)

    elif resolution == "hourly":
        work["day"] = work["datetime"].dt.floor("D")

        base = (
            work.groupby(["site", "day"], as_index=False)
            .agg(
                n_obs=(pollutant, "count"),
                daily_mean=(pollutant, "mean")
            )
        )

        valid = base[base["n_obs"] >= daily_min_obs].copy()
        valid.rename(columns={"daily_mean": pollutant}, inplace=True)

    elif resolution == "daily":
        work["day"] = work["datetime"].dt.floor("D")

        daily = (
            work.groupby(["site", "day"], as_index=False)
            .agg(
                n_obs=(pollutant, "count"),
                daily_mean=(pollutant, "mean")
            )
        )

        valid = daily[daily["n_obs"] >= 1].copy()
        valid.rename(columns={"daily_mean": pollutant}, inplace=True)

    else:
        work["day"] = work["datetime"].dt.floor("D")

        base = (
            work.groupby(["site", "day"], as_index=False)
            .agg(
                n_obs=(pollutant, "count"),
                daily_mean=(pollutant, "mean")
            )
        )

        threshold = 1 if base["n_obs"].median() <= 1 else daily_min_obs
        valid = base[base["n_obs"] >= threshold].copy()
        valid.rename(columns={"daily_mean": pollutant}, inplace=True)

    valid["year"] = valid["day"].dt.year
    valid["month"] = valid["day"].dt.to_period("M").astype(str)
    valid["quarter"] = valid["day"].dt.to_period("Q").astype(str)
    valid["dayofweek"] = valid["day"].dt.day_name()
    valid["weekday_type"] = valid["day"].dt.weekday.map(lambda x: "Weekend" if x >= 5 else "Weekday")
    valid["season"] = valid["day"].dt.month.map(lambda m: "Harmattan" if m in [12, 1, 2] else "Non-Harmattan")
    valid["source_resolution"] = resolution

    return valid


def _days_in_quarter(q_str: str) -> int:
    p = pd.Period(q_str, freq="Q")
    start = p.start_time.normalize()
    end = p.end_time.normalize()
    return (end - start).days + 1


def _days_in_year(y: int) -> int:
    y = int(y)
    return 366 if pd.Timestamp(f"{y}-12-31").is_leap_year else 365


def apply_period_completeness(
    daily_valid: pd.DataFrame,
    group_cols: list,
    pollutant: str,
    total_days_fn,
    min_capture: float = PERIOD_MIN_CAPTURE,
) -> pd.DataFrame:
    if daily_valid.empty or pollutant not in daily_valid.columns:
        return pd.DataFrame(columns=group_cols + [pollutant])

    mean_df = daily_valid.groupby(group_cols)[pollutant].mean().reset_index()
    counts = daily_valid.groupby(group_cols)["day"].nunique().reset_index(name="valid_days")
    out = mean_df.merge(counts, on=group_cols, how="left")

    out["total_days"] = out.apply(total_days_fn, axis=1)
    out["capture"] = out["valid_days"] / out["total_days"]
    out.loc[out["capture"] < min_capture, pollutant] = np.nan

    return out


def validate_daily_means_with_report(
    df: pd.DataFrame,
    pollutant: str,
    daily_min_obs: int = DAILY_MIN_OBS,
    hourly_min_minute_obs: int = HOURLY_MIN_MINUTE_OBS,
):
    """
    Returns:
        valid_daily_df,
        summary_df,
        invalid_hours_df,
        invalid_days_df
    """
    empty_valid = pd.DataFrame(columns=["site", "day", pollutant])
    empty_summary = pd.DataFrame(columns=["metric", "value"])
    empty_hours = pd.DataFrame()
    empty_days = pd.DataFrame()

    if pollutant not in df.columns or "datetime" not in df.columns or "site" not in df.columns:
        return empty_valid, empty_summary, empty_hours, empty_days

    work = df[["site", "datetime", pollutant]].copy()
    work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce")
    work[pollutant] = pd.to_numeric(work[pollutant], errors="coerce")
    work = work.dropna(subset=["site", "datetime", pollutant])

    if work.empty:
        return empty_valid, empty_summary, empty_hours, empty_days

    resolution = detect_data_resolution(work, "datetime")
    invalid_hours = pd.DataFrame()
    invalid_days = pd.DataFrame()

    if resolution == "minute":
        work["hour"] = work["datetime"].dt.floor("h")

        hourly = (
            work.groupby(["site", "hour"], as_index=False)
            .agg(
                n_minute_obs=(pollutant, "count"),
                hourly_mean=(pollutant, "mean")
            )
        )

        invalid_hours = hourly[hourly["n_minute_obs"] < hourly_min_minute_obs].copy()
        valid_hours = hourly[hourly["n_minute_obs"] >= hourly_min_minute_obs].copy()
        valid_hours["day"] = valid_hours["hour"].dt.floor("D")

        daily_base = (
            valid_hours.groupby(["site", "day"], as_index=False)
            .agg(
                n_obs=("hourly_mean", "count"),
                daily_mean=("hourly_mean", "mean")
            )
        )

        invalid_days = daily_base[daily_base["n_obs"] < daily_min_obs].copy()
        valid = daily_base[daily_base["n_obs"] >= daily_min_obs].copy()
        valid.rename(columns={"daily_mean": pollutant}, inplace=True)

        summary = pd.DataFrame({
            "metric": [
                "pollutant",
                "resolution_detected",
                "records_after_basic_cleaning",
                "hours_total",
                "hours_excluded_lt_45min",
                "hours_valid",
                "days_total_from_valid_hours",
                "days_excluded_lt_18hours",
                "days_valid",
            ],
            "value": [
                pollutant,
                resolution,
                len(work),
                len(hourly),
                len(invalid_hours),
                len(valid_hours),
                len(daily_base),
                len(invalid_days),
                len(valid),
            ]
        })

    elif resolution == "hourly":
        work["day"] = work["datetime"].dt.floor("D")

        daily_base = (
            work.groupby(["site", "day"], as_index=False)
            .agg(
                n_obs=(pollutant, "count"),
                daily_mean=(pollutant, "mean")
            )
        )

        invalid_days = daily_base[daily_base["n_obs"] < daily_min_obs].copy()
        valid = daily_base[daily_base["n_obs"] >= daily_min_obs].copy()
        valid.rename(columns={"daily_mean": pollutant}, inplace=True)

        summary = pd.DataFrame({
            "metric": [
                "pollutant",
                "resolution_detected",
                "records_after_basic_cleaning",
                "days_total",
                "days_excluded_lt_18hours",
                "days_valid",
            ],
            "value": [
                pollutant,
                resolution,
                len(work),
                len(daily_base),
                len(invalid_days),
                len(valid),
            ]
        })

    elif resolution == "daily":
        work["day"] = work["datetime"].dt.floor("D")

        daily_base = (
            work.groupby(["site", "day"], as_index=False)
            .agg(
                n_obs=(pollutant, "count"),
                daily_mean=(pollutant, "mean")
            )
        )

        invalid_days = daily_base[daily_base["n_obs"] < 1].copy()
        valid = daily_base[daily_base["n_obs"] >= 1].copy()
        valid.rename(columns={"daily_mean": pollutant}, inplace=True)

        summary = pd.DataFrame({
            "metric": [
                "pollutant",
                "resolution_detected",
                "records_after_basic_cleaning",
                "days_total",
                "days_excluded",
                "days_valid",
            ],
            "value": [
                pollutant,
                resolution,
                len(work),
                len(daily_base),
                len(invalid_days),
                len(valid),
            ]
        })

    else:
        work["day"] = work["datetime"].dt.floor("D")

        daily_base = (
            work.groupby(["site", "day"], as_index=False)
            .agg(
                n_obs=(pollutant, "count"),
                daily_mean=(pollutant, "mean")
            )
        )

        threshold = 1 if daily_base["n_obs"].median() <= 1 else daily_min_obs
        invalid_days = daily_base[daily_base["n_obs"] < threshold].copy()
        valid = daily_base[daily_base["n_obs"] >= threshold].copy()
        valid.rename(columns={"daily_mean": pollutant}, inplace=True)

        summary = pd.DataFrame({
            "metric": [
                "pollutant",
                "resolution_detected",
                "records_after_basic_cleaning",
                "days_total",
                f"days_excluded_lt_{threshold}_obs",
                "days_valid",
            ],
            "value": [
                pollutant,
                resolution,
                len(work),
                len(daily_base),
                len(invalid_days),
                len(valid),
            ]
        })

    if not valid.empty:
        valid["year"] = valid["day"].dt.year
        valid["month"] = valid["day"].dt.to_period("M").astype(str)
        valid["quarter"] = valid["day"].dt.to_period("Q").astype(str)
        valid["dayofweek"] = valid["day"].dt.day_name()
        valid["weekday_type"] = valid["day"].dt.weekday.map(lambda x: "Weekend" if x >= 5 else "Weekday")
        valid["season"] = valid["day"].dt.month.map(lambda m: "Harmattan" if m in [12, 1, 2] else "Non-Harmattan")
        valid["source_resolution"] = resolution

    return valid, summary, invalid_hours, invalid_days


def summarize_period_capture(
    daily_valid: pd.DataFrame,
    pollutant: str,
    period_name: str,
    group_cols: list,
    total_days_fn,
    min_capture: float = PERIOD_MIN_CAPTURE,
):
    if daily_valid.empty or pollutant not in daily_valid.columns:
        return pd.DataFrame(columns=group_cols + ["valid_days", "total_days", "capture", "status", "period_type", "pollutant"])

    counts = daily_valid.groupby(group_cols)["day"].nunique().reset_index(name="valid_days")
    counts["total_days"] = counts.apply(total_days_fn, axis=1)
    counts["capture"] = counts["valid_days"] / counts["total_days"]
    counts["status"] = np.where(counts["capture"] >= min_capture, "Included", "Excluded")
    counts["period_type"] = period_name
    counts["pollutant"] = pollutant
    return counts


def generate_dataset_validation_report(df: pd.DataFrame, cleaning_report: dict):
    report = {
        "cleaning_summary": pd.DataFrame([
            {"metric": k, "value": v} for k, v in cleaning_report.items()
        ])
    }

    for pollutant in [p for p in ["pm25", "pm10"] if p in df.columns]:
        daily_valid, summary_df, invalid_hours_df, invalid_days_df = validate_daily_means_with_report(
            df, pollutant, DAILY_MIN_OBS, HOURLY_MIN_MINUTE_OBS
        )

        monthly_capture = summarize_period_capture(
            daily_valid,
            pollutant,
            "Monthly",
            ["month", "site"],
            total_days_fn=lambda r: pd.Period(r["month"], "M").days_in_month
        )

        quarterly_capture = summarize_period_capture(
            daily_valid,
            pollutant,
            "Quarterly",
            ["quarter", "site"],
            total_days_fn=lambda r: _days_in_quarter(r["quarter"])
        )

        yearly_capture = summarize_period_capture(
            daily_valid,
            pollutant,
            "Yearly",
            ["year", "site"],
            total_days_fn=lambda r: _days_in_year(r["year"])
        )

        def season_total_days(row):
            y = int(row["year"])
            s = row["season"]
            if s == "Harmattan":
                return (
                    pd.Period(f"{y}-01", "M").days_in_month +
                    pd.Period(f"{y}-02", "M").days_in_month +
                    pd.Period(f"{y}-12", "M").days_in_month
                )
            else:
                months = [f"{y}-{m:02d}" for m in range(3, 12)]
                return sum(pd.Period(m, "M").days_in_month for m in months)

        seasonal_capture = summarize_period_capture(
            daily_valid,
            pollutant,
            "Season",
            ["season", "year", "site"],
            total_days_fn=season_total_days
        )

        period_capture = pd.concat(
            [monthly_capture, quarterly_capture, yearly_capture, seasonal_capture],
            ignore_index=True
        )

        excluded_periods = period_capture[period_capture["status"] == "Excluded"].copy()

        report[pollutant] = {
            "daily_validation_summary": summary_df,
            "invalid_hours": invalid_hours_df,
            "invalid_days": invalid_days_df,
            "period_capture": period_capture,
            "excluded_periods": excluded_periods,
            "valid_daily": daily_valid,
        }

    return report


def compute_aggregates(df: pd.DataFrame, label: str, pollutant: str):
    aggregates = {}
    daily_valid = build_valid_daily_means(df, pollutant, DAILY_MIN_OBS, HOURLY_MIN_MINUTE_OBS)

    aggregates[f"{label} - Daily Avg ({pollutant})"] = daily_valid[["day", "site", pollutant]].round(1)

    monthly = apply_period_completeness(
        daily_valid, ["month", "site"], pollutant,
        total_days_fn=lambda r: pd.Period(r["month"], "M").days_in_month
    )
    aggregates[f"{label} - Monthly Avg ({pollutant})"] = monthly[["month", "site", pollutant]].round(1)

    quarterly = apply_period_completeness(
        daily_valid, ["quarter", "site"], pollutant,
        total_days_fn=lambda r: _days_in_quarter(r["quarter"])
    )
    aggregates[f"{label} - Quarterly Avg ({pollutant})"] = quarterly[["quarter", "site", pollutant]].round(1)

    yearly = apply_period_completeness(
        daily_valid, ["year", "site"], pollutant,
        total_days_fn=lambda r: _days_in_year(r["year"])
    )
    aggregates[f"{label} - Yearly Avg ({pollutant})"] = yearly[["year", "site", pollutant]].round(1)

    aggregates[f"{label} - Day of Week Avg ({pollutant})"] = (
        daily_valid.groupby(["dayofweek", "site"])[pollutant].mean().reset_index().round(1)
    )
    aggregates[f"{label} - Weekday Type Avg ({pollutant})"] = (
        daily_valid.groupby(["weekday_type", "site"])[pollutant].mean().reset_index().round(1)
    )

    def season_total_days(row):
        y = int(row["year"])
        s = row["season"]
        if s == "Harmattan":
            return (
                pd.Period(f"{y}-01", "M").days_in_month +
                pd.Period(f"{y}-02", "M").days_in_month +
                pd.Period(f"{y}-12", "M").days_in_month
            )
        else:
            months = [f"{y}-{m:02d}" for m in range(3, 12)]
            return sum(pd.Period(m, "M").days_in_month for m in months)

    season = apply_period_completeness(
        daily_valid, ["season", "year", "site"], pollutant,
        total_days_fn=season_total_days
    )
    aggregates[f"{label} - Season Avg ({pollutant})"] = season[["season", "year", "site", pollutant]].round(1)

    return aggregates


# -------------------------
# Exceedances / MinMax
# -------------------------
def calculate_exceedances(df: pd.DataFrame) -> pd.DataFrame:
    pm25_daily = build_valid_daily_means(df, "pm25", DAILY_MIN_OBS, HOURLY_MIN_MINUTE_OBS)
    pm10_daily = build_valid_daily_means(df, "pm10", DAILY_MIN_OBS, HOURLY_MIN_MINUTE_OBS)

    if pm25_daily.empty and pm10_daily.empty:
        return pd.DataFrame()

    daily = pm25_daily[["site", "day", "year", "month", "quarter", "pm25"]].merge(
        pm10_daily[["site", "day", "pm10"]],
        on=["site", "day"],
        how="outer"
    )

    pm25_exceed = (
        daily[daily["pm25"] > 35]
        .groupby(["year", "site"])
        .size()
        .reset_index(name="pm25_Exceedance_Count")
    )
    pm10_exceed = (
        daily[daily["pm10"] > 70]
        .groupby(["year", "site"])
        .size()
        .reset_index(name="pm10_Exceedance_Count")
    )
    total_days = daily.groupby(["year", "site"]).size().reset_index(name="Total_Valid_Days")

    exceedance = (
        total_days.merge(pm25_exceed, on=["year", "site"], how="left")
                  .merge(pm10_exceed, on=["year", "site"], how="left")
    ).fillna(0)

    exceedance["pm25_Exceedance_Percent"] = round(
        (exceedance["pm25_Exceedance_Count"] / exceedance["Total_Valid_Days"]) * 100, 1
    )
    exceedance["pm10_Exceedance_Percent"] = round(
        (exceedance["pm10_Exceedance_Count"] / exceedance["Total_Valid_Days"]) * 100, 1
    )

    return exceedance


def calculate_min_max(df: pd.DataFrame) -> pd.DataFrame:
    pm25_daily = build_valid_daily_means(df, "pm25", DAILY_MIN_OBS, HOURLY_MIN_MINUTE_OBS)
    pm10_daily = build_valid_daily_means(df, "pm10", DAILY_MIN_OBS, HOURLY_MIN_MINUTE_OBS)

    if pm25_daily.empty and pm10_daily.empty:
        return pd.DataFrame()

    daily = pm25_daily[["site", "day", "year", "month", "quarter", "pm25"]].merge(
        pm10_daily[["site", "day", "pm10"]],
        on=["site", "day"],
        how="outer"
    )

    def safe_nanmax(x):
        arr = np.asarray(x, dtype=float)
        return np.nan if np.all(np.isnan(arr)) else round(np.nanmax(arr), 1)

    def safe_nanmin(x):
        arr = np.asarray(x, dtype=float)
        return np.nan if np.all(np.isnan(arr)) else round(np.nanmin(arr), 1)

    out = (
        daily.groupby(["site", "year", "quarter", "month"], as_index=False)
            .agg(
                daily_avg_pm10_max=("pm10", safe_nanmax),
                daily_avg_pm10_min=("pm10", safe_nanmin),
                daily_avg_pm25_max=("pm25", safe_nanmax),
                daily_avg_pm25_min=("pm25", safe_nanmin),
            )
    )
    return out


# -------------------------
# AQI
# -------------------------
def calculate_aqi_and_category(df: pd.DataFrame):
    daily = build_valid_daily_means(df, "pm25", DAILY_MIN_OBS, HOURLY_MIN_MINUTE_OBS)
    if daily.empty:
        return pd.DataFrame(), pd.DataFrame()

    breakpoints = [
        (0.0, 9.0, 0, 50),
        (9.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200),
        (125.5, 225.4, 201, 300),
        (225.5, 325.4, 301, 500),
        (325.5, 99999.9, 501, 999),
    ]

    def calculate_aqi(pm):
        if pd.isna(pm):
            return np.nan
        for low, high, aqi_low, aqi_high in breakpoints:
            if low <= pm <= high:
                return round(((pm - low) * (aqi_high - aqi_low) / (high - low)) + aqi_low)
        return np.nan

    daily_avg = daily[["site", "day", "year", "quarter", "month", "pm25"]].copy()
    daily_avg["pm25"] = daily_avg["pm25"].round(1)
    daily_avg["AQI"] = daily_avg["pm25"].apply(calculate_aqi)

    conditions = [
        (daily_avg["AQI"] >= 0) & (daily_avg["AQI"] <= 50),
        (daily_avg["AQI"] > 50) & (daily_avg["AQI"] <= 100),
        (daily_avg["AQI"] > 100) & (daily_avg["AQI"] <= 150),
        (daily_avg["AQI"] > 150) & (daily_avg["AQI"] <= 200),
        (daily_avg["AQI"] > 200) & (daily_avg["AQI"] <= 300),
        (daily_avg["AQI"] > 300),
    ]
    remarks = [
        "Good",
        "Moderate",
        "Unhealthy for Sensitive Groups",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous"
    ]
    daily_avg["AQI_Remark"] = np.select(conditions, remarks, default="Unknown")

    remarks_counts = daily_avg.groupby(["year", "AQI_Remark"]).size().reset_index(name="Count")
    remarks_counts["Total_Count_Per_Year"] = remarks_counts.groupby("year")["Count"].transform("sum")
    remarks_counts["Percent"] = round((remarks_counts["Count"] / remarks_counts["Total_Count_Per_Year"]) * 100, 1)

    return daily_avg, remarks_counts


# -------------------------
# Daily pollutant helper
# -------------------------
def calculate_day_pollutant(df: pd.DataFrame, pollutant: str) -> pd.DataFrame:
    daily = build_valid_daily_means(df, pollutant, DAILY_MIN_OBS, HOURLY_MIN_MINUTE_OBS)
    if daily.empty:
        return pd.DataFrame(columns=["site", "day", "month", "quarter", "year", pollutant])
    return daily[["site", "day", "month", "quarter", "year", pollutant]].round(1)


# -------------------------
# Tabs
# -------------------------
def render_exceedances_tab(tab, dfs, selected_years):
    with tab:
        st.header("🌫️ Exceedances & Max/Min (valid daily only)")

        for label, df in dfs.items():
            st.subheader(f"Dataset: {label}")
            detected_resolution = df.attrs.get("resolution", "unknown")
            st.caption(
                f"Detected input resolution: {detected_resolution} | "
                f"Valid hour for minute data: ≥ {HOURLY_MIN_MINUTE_OBS} one-minute values/hour | "
                f"Valid day: ≥ {DAILY_MIN_OBS} hourly values/day"
            )

            site_in_tab = st.multiselect(
                f"Select Site(s) for {label}",
                options=sorted(df["site"].unique()),
                key=f"site_exc_{label}",
            )

            available_years = sorted(df["year"].dropna().unique())
            selected_years_in_tab = st.multiselect(
                f"Select Year(s) for {label}",
                options=available_years,
                default=selected_years if selected_years else available_years,
                key=f"year_exc_{label}",
            )

            filtered_df = df.copy()
            if selected_years_in_tab:
                filtered_df = filtered_df[filtered_df["year"].isin(selected_years_in_tab)]
            if site_in_tab:
                filtered_df = filtered_df[filtered_df["site"].isin(site_in_tab)]

            selected_quarters = st.multiselect(
                f"Select Quarter(s) for {label}",
                options=["Q1", "Q2", "Q3", "Q4"],
                default=["Q1", "Q2", "Q3", "Q4"],
                key=f"quarter_exc_{label}",
            )

            if selected_years_in_tab:
                selected_quarter_nums = [f"{year}{q}" for year in selected_years_in_tab for q in selected_quarters]
                filtered_df = filtered_df[filtered_df["quarter"].isin(selected_quarter_nums)]
            else:
                st.warning("No valid years selected!")
                continue

            if filtered_df.empty:
                st.warning(f"No data remaining for {label} after filtering.")
                continue

            exceedances = calculate_exceedances(filtered_df)
            if exceedances.empty:
                st.warning("No exceedance output available for current selection.")
            else:
                st.dataframe(exceedances, use_container_width=True)
                st.download_button(
                    label=f"⬇️ Download Exceedances - {label}",
                    data=to_csv_download(exceedances),
                    file_name=f"Exceedances_{label}.csv",
                    key=f"download_exceedances_{label}",
                )

            min_max = calculate_min_max(filtered_df)
            if min_max.empty:
                st.warning("No min/max output available for current selection.")
            else:
                st.dataframe(min_max, use_container_width=True)
                st.download_button(
                    label=f"⬇️ Download MinMax - {label}",
                    data=to_csv_download(min_max),
                    file_name=f"MinMax_{label}.csv",
                    key=f"download_minmax_{label}",
                )


def render_aqi_tab(tab, selected_years, dfs):
    with tab:
        st.header("🌫️ AQI Stats (valid daily pm25 only)")

        for label, df in dfs.items():
            st.subheader(f"Dataset: {label}")
            detected_resolution = df.attrs.get("resolution", "unknown")
            st.caption(
                f"Detected input resolution: {detected_resolution} | "
                f"Valid hour for minute data: ≥ {HOURLY_MIN_MINUTE_OBS} one-minute values/hour | "
                f"Valid day: ≥ {DAILY_MIN_OBS} hourly values/day"
            )

            all_years = sorted(df["year"].dropna().unique())
            default_years = all_years[-2:] if len(all_years) >= 2 else all_years
            selected_years_in_tab = st.multiselect(
                f"Select Year(s) for {label}",
                options=all_years,
                default=default_years,
                key=f"years_aqi_{label}",
            )

            site_in_tab = st.multiselect(
                f"Select Site(s) for {label}",
                sorted(df["site"].unique()),
                key=f"site_aqi_{label}",
            )

            filtered_df = df.copy()
            if selected_years_in_tab:
                filtered_df = filtered_df[filtered_df["year"].isin(selected_years_in_tab)]
            if site_in_tab:
                filtered_df = filtered_df[filtered_df["site"].isin(site_in_tab)]

            selected_quarters = st.multiselect(
                f"Select Quarter(s) for {label}",
                options=["Q1", "Q2", "Q3", "Q4"],
                default=["Q1", "Q2", "Q3", "Q4"],
                key=unique_key("tab_aqi", "quarter", label),
            ) or []

            selected_quarter_nums = [f"{year}{q}" for year in selected_years_in_tab for q in selected_quarters]
            filtered_df = filtered_df[filtered_df["quarter"].isin(selected_quarter_nums)]

            if filtered_df.empty:
                st.warning(f"No data remaining for {label} after filtering.")
                continue

            daily_avg, remarks_counts = calculate_aqi_and_category(filtered_df)
            if daily_avg.empty:
                st.warning("No valid daily pm25 for AQI calculation.")
                continue

            st.dataframe(remarks_counts, use_container_width=True)
            st.dataframe(daily_avg, use_container_width=True)

            st.download_button(
                f"⬇️ Download Daily Avg - {label}",
                to_csv_download(daily_avg),
                file_name=f"DailyAvg_{label}.csv",
                key=f"download_dailyavg_{label}"
            )
            st.download_button(
                f"⬇️ Download AQI - {label}",
                to_csv_download(remarks_counts),
                file_name=f"AQI_{label}.csv",
                key=f"download_aqi_{label}"
            )


def render_daily_means_tab(tab, dfs, selected_years):
    with tab:
        st.header("📊 Daily Means (valid days only)")

        for label, df in dfs.items():
            st.subheader(f"Dataset: {label}")
            detected_resolution = df.attrs.get("resolution", "unknown")
            st.caption(
                f"Detected input resolution: {detected_resolution} | "
                f"Valid hour for minute data: ≥ {HOURLY_MIN_MINUTE_OBS} one-minute values/hour | "
                f"Valid day: ≥ {DAILY_MIN_OBS} hourly values/day"
            )

            site_in_tab = st.multiselect(
                f"Select Site(s) for {label}",
                sorted(df["site"].unique()),
                key=unique_key("tab_daily", "site", label),
            )

            filtered_df = df.copy()
            if selected_years:
                filtered_df = filtered_df[filtered_df["year"].isin(selected_years)]
            if site_in_tab:
                filtered_df = filtered_df[filtered_df["site"].isin(site_in_tab)]

            selected_quarters = st.multiselect(
                f"Select Quarter(s) for {label}",
                options=["Q1", "Q2", "Q3", "Q4"],
                default=["Q1", "Q2", "Q3", "Q4"],
                key=unique_key("tab_daily", "quarter", label),
            ) or []

            years_use = selected_years if selected_years else sorted(df["year"].dropna().unique())
            selected_quarter_nums = [f"{year}{q}" for year in years_use for q in selected_quarters]
            filtered_df = filtered_df[filtered_df["quarter"].isin(selected_quarter_nums)]

            if filtered_df.empty:
                st.warning(f"No data remaining for {label} after filtering.")
                continue

            valid_pollutants = [p for p in ["pm25", "pm10"] if p in filtered_df.columns]
            selected_display_pollutants = st.multiselect(
                f"Select Pollutants to Display for {label}",
                options=["All"] + valid_pollutants,
                default=["All"],
                key=unique_key("tab_daily", "pollutants", label),
            )
            if "All" in selected_display_pollutants:
                selected_display_pollutants = valid_pollutants

            chart_type = st.radio(
                f"Chart Type for {label}",
                ["Line", "Bar"],
                horizontal=True,
                key=unique_key("tab_daily", "charttype", label),
            )

            df_list = []
            for p in selected_display_pollutants:
                out = calculate_day_pollutant(filtered_df, p)
                if out.empty:
                    continue
                out["pollutant"] = p
                out.rename(columns={p: "value"}, inplace=True)
                df_list.append(out)

            if not df_list:
                st.warning("No valid daily data in selection.")
                continue

            df_avg = pd.concat(df_list, ignore_index=True)

            if chart_type == "Line":
                fig = px.line(
                    df_avg,
                    x="day",
                    y="value",
                    color="pollutant",
                    line_group="site",
                    markers=True,
                    title=f"Daily Means - {label}",
                    hover_data=["site", "pollutant", "value"]
                )
            else:
                fig = px.bar(
                    df_avg,
                    x="day",
                    y="value",
                    color="site",
                    barmode="group",
                    facet_col="pollutant" if len(selected_display_pollutants) > 1 else None,
                    title=f"Daily Means - {label}"
                )

                for pollutant in selected_display_pollutants:
                    threshold = 35 if pollutant.lower() == "pm25" else 70
                    fig.add_trace(go.Scatter(
                        x=sorted(df_avg["day"].unique()),
                        y=[threshold] * len(df_avg["day"].unique()),
                        mode="lines",
                        line=dict(dash="solid", color="red"),
                        name=f"Ghana Standard ({threshold} µg/m³)",
                        showlegend=True
                    ))

            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Show Daily Table"):
                st.dataframe(df_avg, use_container_width=True)


def render_dayofweek_means_tab(tab, dfs, selected_years):
    with tab:
        st.header("📊 Day of Week Means (valid days only)")

        for label, df in dfs.items():
            st.subheader(f"Dataset: {label}")
            detected_resolution = df.attrs.get("resolution", "unknown")
            st.caption(
                f"Detected input resolution: {detected_resolution} | "
                f"Valid hour for minute data: ≥ {HOURLY_MIN_MINUTE_OBS} one-minute values/hour | "
                f"Valid day: ≥ {DAILY_MIN_OBS} hourly values/day"
            )

            site_in_tab = st.multiselect(
                f"Select Site(s) for {label}",
                sorted(df["site"].unique()),
                key=unique_key("tab_dow", "site", label),
            )

            filtered_df = df.copy()
            if selected_years:
                filtered_df = filtered_df[filtered_df["year"].isin(selected_years)]
            if site_in_tab:
                filtered_df = filtered_df[filtered_df["site"].isin(site_in_tab)]

            selected_quarters = st.multiselect(
                f"Select Quarter(s) for {label}",
                options=["Q1", "Q2", "Q3", "Q4"],
                default=["Q1", "Q2", "Q3", "Q4"],
                key=unique_key("tab_dow", "quarter", label),
            ) or []

            years_use = selected_years if selected_years else sorted(df["year"].dropna().unique())
            selected_quarter_nums = [f"{year}{q}" for year in years_use for q in selected_quarters]
            filtered_df = filtered_df[filtered_df["quarter"].isin(selected_quarter_nums)]

            if filtered_df.empty:
                st.warning(f"No data remaining for {label} after filtering.")
                continue

            valid_pollutants = [p for p in ["pm25", "pm10"] if p in filtered_df.columns]
            selected_display_pollutants = st.multiselect(
                f"Select Pollutants to Display for {label}",
                options=["All"] + valid_pollutants,
                default=["All"],
                key=unique_key("tab_dow", "pollutants", label),
            )
            if "All" in selected_display_pollutants:
                selected_display_pollutants = valid_pollutants

            chart_type = st.radio(
                f"Chart Type for {label}",
                ["Line", "Bar"],
                horizontal=True,
                key=unique_key("tab_dow", "charttype", label),
            )

            out_list = []
            for p in selected_display_pollutants:
                daily = build_valid_daily_means(filtered_df, p, DAILY_MIN_OBS, HOURLY_MIN_MINUTE_OBS)
                if daily.empty:
                    continue
                g = daily.groupby(["site", "dayofweek", "quarter", "year"])[p].mean().reset_index().round(1)
                g["pollutant"] = p
                g.rename(columns={p: "value"}, inplace=True)
                out_list.append(g)

            if not out_list:
                st.warning("No valid daily data in selection.")
                continue

            df_avg = pd.concat(out_list, ignore_index=True)

            weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            if "dayofweek" in df_avg.columns:
                df_avg["dayofweek"] = pd.Categorical(df_avg["dayofweek"], categories=weekday_order, ordered=True)
                df_avg = df_avg.sort_values("dayofweek")

            if chart_type == "Line":
                fig = px.line(
                    df_avg,
                    x="dayofweek",
                    y="value",
                    color="pollutant",
                    line_group="site",
                    markers=True,
                    title=f"Day of Week Means - {label}"
                )
            else:
                fig = px.bar(
                    df_avg,
                    x="dayofweek",
                    y="value",
                    color="site",
                    barmode="group",
                    facet_col="pollutant" if len(selected_display_pollutants) > 1 else None,
                    title=f"Day of Week Means - {label}"
                )

            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Show Table"):
                st.dataframe(df_avg, use_container_width=True)


def render_validation_report_tab(tab, dfs):
    with tab:
        st.header("🧾 Validation / Excluded Data Report")
        st.caption(
            f"This report shows records and periods excluded during cleaning and completeness validation. "
            f"Rules applied: valid hour for minute data ≥ {HOURLY_MIN_MINUTE_OBS} one-minute values, "
            f"valid day for minute/hourly data ≥ {DAILY_MIN_OBS} hourly values, "
            f"period capture ≥ {int(PERIOD_MIN_CAPTURE * 100)}% valid days."
        )

        for label, df in dfs.items():
            st.subheader(f"Dataset: {label}")
            validation_report = df.attrs.get("validation_report", {})

            cleaning_summary = validation_report.get("cleaning_summary", pd.DataFrame())
            if not cleaning_summary.empty:
                st.markdown("### Basic Cleaning Summary")
                st.dataframe(cleaning_summary, use_container_width=True)
                st.download_button(
                    f"⬇️ Download Cleaning Summary - {label}",
                    data=to_csv_download(cleaning_summary),
                    file_name=f"{label}_cleaning_summary.csv",
                    key=f"dl_cleaning_summary_{label}",
                )

            for pollutant in [p for p in ["pm25", "pm10"] if p in validation_report]:
                st.markdown(f"## {pollutant.upper()}")
                p_report = validation_report[pollutant]

                st.markdown("#### Daily Completeness Summary")
                st.dataframe(p_report["daily_validation_summary"], use_container_width=True)

                if not p_report["invalid_hours"].empty:
                    st.markdown("#### Excluded Hours")
                    st.dataframe(p_report["invalid_hours"], use_container_width=True)
                    st.download_button(
                        f"⬇️ Download Excluded Hours - {label} - {pollutant}",
                        data=to_csv_download(p_report["invalid_hours"]),
                        file_name=f"{label}_{pollutant}_excluded_hours.csv",
                        key=f"dl_invalid_hours_{label}_{pollutant}",
                    )

                if not p_report["invalid_days"].empty:
                    st.markdown("#### Excluded Days")
                    st.dataframe(p_report["invalid_days"], use_container_width=True)
                    st.download_button(
                        f"⬇️ Download Excluded Days - {label} - {pollutant}",
                        data=to_csv_download(p_report["invalid_days"]),
                        file_name=f"{label}_{pollutant}_excluded_days.csv",
                        key=f"dl_invalid_days_{label}_{pollutant}",
                    )

                if not p_report["period_capture"].empty:
                    st.markdown("#### Period Capture Report")
                    st.dataframe(p_report["period_capture"], use_container_width=True)
                    st.download_button(
                        f"⬇️ Download Period Capture - {label} - {pollutant}",
                        data=to_csv_download(p_report["period_capture"]),
                        file_name=f"{label}_{pollutant}_period_capture.csv",
                        key=f"dl_period_capture_{label}_{pollutant}",
                    )

                if not p_report["excluded_periods"].empty:
                    st.markdown("#### Excluded Periods (< 75% Capture)")
                    st.dataframe(p_report["excluded_periods"], use_container_width=True)
                    st.download_button(
                        f"⬇️ Download Excluded Periods - {label} - {pollutant}",
                        data=to_csv_download(p_report["excluded_periods"]),
                        file_name=f"{label}_{pollutant}_excluded_periods.csv",
                        key=f"dl_excluded_periods_{label}_{pollutant}",
                    )

                st.markdown("---")


# -------------------------
# Main app UI
# -------------------------
st.title("Air Quality Dashboard Aggregator (Completeness-Aware)")

uploaded_files = st.file_uploader(
    "Upload up to 4 datasets (CSV or Excel)",
    type=["csv", "xlsx"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload CSV or Excel files to begin.")
    st.stop()

dfs = {}
site_options = set()
year_options = set()

for file in uploaded_files:
    label = file.name.rsplit(".", 1)[0]
    ext = file.name.rsplit(".", 1)[-1].lower()

    try:
        raw = pd.read_excel(file) if ext == "xlsx" else pd.read_csv(file)
    except Exception as e:
        st.warning(f"⚠️ Could not read {file.name}: {e}")
        continue

    df = parse_dates(raw)
    df = standardize_columns(df)
    df, cleaning_report = cleaned(df)

    if "datetime" not in df.columns or "site" not in df.columns:
        st.warning(f"⚠️ Could not process {label}: missing datetime/site.")
        continue

    if "pm25" not in df.columns and "pm10" not in df.columns:
        st.warning(f"⚠️ Could not process {label}: missing both pm25 and pm10.")
        continue

    if df.empty:
        st.warning(f"⚠️ {label} has no usable data after cleaning.")
        continue

    resolution_label = get_dataset_resolution_label(df)

    dfs[label] = df
    dfs[label].attrs["resolution"] = resolution_label
    dfs[label].attrs["cleaning_report"] = cleaning_report
    dfs[label].attrs["validation_report"] = generate_dataset_validation_report(df, cleaning_report)

    site_options.update(df["site"].dropna().unique().tolist())
    year_options.update(df["year"].dropna().unique().tolist())

if not dfs:
    st.warning("No datasets could be processed. Please check your columns and datetime formatting.")
    st.stop()

with st.sidebar:
    st.header("Filters (global)")
    selected_years = st.multiselect("📅 Filter by Year", sorted(year_options))
    selected_sites = st.multiselect("🏢 Filter by Site", sorted(site_options))
    st.caption(f"Valid hour for minute data: ≥ {HOURLY_MIN_MINUTE_OBS} one-minute values/hour")
    st.caption(f"Valid day for minute/hourly data: ≥ {DAILY_MIN_OBS} hourly values/day")
    st.caption(f"Period completeness: ≥ {int(PERIOD_MIN_CAPTURE * 100)}% valid days")

tabs = st.tabs([
    "Aggregated Means",
    "Exceedances & Max/Min Values",
    "AQI Stats",
    "Daily Means",
    "Day of Week Means",
    "Validation / Excluded Data Report",
])

# -------------------------
# Aggregated Means
# -------------------------
with tabs[0]:
    st.header("📊 Aggregated Means (Completeness Rule Applied)")

    for label, df in dfs.items():
        st.subheader(f"Dataset: {label}")
        detected_resolution = df.attrs.get("resolution", "unknown")
        st.caption(
            f"Detected input resolution: {detected_resolution} | "
            f"Valid hour for minute data: ≥ {HOURLY_MIN_MINUTE_OBS} one-minute values/hour | "
            f"Valid day: ≥ {DAILY_MIN_OBS} hourly values/day"
        )

        site_in_tab = st.multiselect(
            f"Select Site(s) for {label}",
            sorted(df["site"].unique()),
            key=f"site_agg_{label}",
        )

        available_years = sorted(df["year"].dropna().unique().tolist())
        year_choices = ["All"] + available_years
        selected_year_choice = st.multiselect(
            f"Select Year(s) for {label}",
            options=year_choices,
            default=["All"],
            key=f"year_agg_{label}",
        )
        if (not selected_year_choice) or ("All" in selected_year_choice):
            years_to_use = available_years
        else:
            years_to_use = selected_year_choice

        quarter_choices = ["All", "Q1", "Q2", "Q3", "Q4"]
        selected_quarters = st.multiselect(
            f"Select Quarter(s) for {label}",
            options=quarter_choices,
            default=["All"],
            key=f"quarter_agg_{label}",
        )
        if (not selected_quarters) or ("All" in selected_quarters):
            quarters_to_use = ["Q1", "Q2", "Q3", "Q4"]
        else:
            quarters_to_use = selected_quarters

        filtered_df = df.copy()

        if selected_years:
            filtered_df = filtered_df[filtered_df["year"].isin(selected_years)]
        if selected_sites:
            filtered_df = filtered_df[filtered_df["site"].isin(selected_sites)]

        if years_to_use:
            filtered_df = filtered_df[filtered_df["year"].isin(years_to_use)]
        if site_in_tab:
            filtered_df = filtered_df[filtered_df["site"].isin(site_in_tab)]

        quarter_keys = [f"{y}{q}" for y in years_to_use for q in quarters_to_use]
        filtered_df = filtered_df[filtered_df["quarter"].isin(quarter_keys)]

        if filtered_df.empty:
            st.warning(f"No data remaining for {label} after filtering.")
            continue

        valid_pollutants = [p for p in ["pm25", "pm10"] if p in filtered_df.columns]
        if not valid_pollutants:
            st.warning(f"No valid pollutants found in {label}")
            continue

        selected_display_pollutants = st.multiselect(
            f"Select Pollutants to Display for {label}",
            options=["All"] + valid_pollutants,
            default=["All"],
            key=f"pollutants_{label}",
        )
        if "All" in selected_display_pollutants:
            selected_display_pollutants = valid_pollutants

        aggregate_levels = [
            ("Daily Avg", ["day", "site"]),
            ("Monthly Avg", ["month", "site"]),
            ("Quarterly Avg", ["quarter", "site"]),
            ("Yearly Avg", ["year", "site"]),
            ("Day of Week Avg", ["dayofweek", "site"]),
            ("Weekday Type Avg", ["weekday_type", "site"]),
            ("Season Avg", ["season", "year", "site"]),
        ]

        agg_by_pollutant = {p: compute_aggregates(filtered_df, label, p) for p in valid_pollutants}

        for level_name, group_keys in aggregate_levels:
            agg_label = f"{label} - {level_name}"
            st.markdown(f"### {agg_label}")

            frames = []
            for p in valid_pollutants:
                k = f"{label} - {level_name} ({p})"
                frames.append(agg_by_pollutant[p].get(k, pd.DataFrame(columns=group_keys + [p])))

            merged_df = reduce(lambda left, right: pd.merge(left, right, on=group_keys, how="outer"), frames)
            display_cols = group_keys + [p for p in selected_display_pollutants if p in merged_df.columns]
            editable_df = merged_df[display_cols].copy()

            st.data_editor(
                editable_df,
                use_container_width=True,
                column_config={col: {"disabled": True} for col in editable_df.columns},
                num_rows="dynamic",
                key=f"editor_{label}_{level_name}",
            )

            st.download_button(
                label=f"📥 Download {agg_label}",
                data=to_csv_download(editable_df),
                file_name=f"{label}_{level_name.replace(' ', '_')}.csv",
                mime="text/csv",
                key=f"dl_{label}_{level_name}",
            )

            with st.expander(f"📈 Show Charts for {agg_label}", expanded=("Yearly Avg" in level_name)):
                chart_type_choice = st.selectbox(
                    f"Select Chart Type for {agg_label}",
                    options=["line", "bar"],
                    index=0 if level_name in ["Daily Avg", "Monthly Avg", "Quarterly Avg", "Yearly Avg"] else 1,
                    key=f"chart_{label}_{level_name}",
                )

                x_axis = next((c for c in group_keys if c != "site"), None)
                if not x_axis:
                    st.warning("Could not determine x-axis.")
                    continue

                safe_pollutants = [p for p in selected_display_pollutants if p in editable_df.columns]
                if not safe_pollutants:
                    st.warning("No pollutant columns to plot.")
                    continue

                melted = editable_df.melt(
                    id_vars=["site"] + [x_axis],
                    value_vars=safe_pollutants,
                    var_name="pollutant",
                    value_name="value",
                )

                if chart_type_choice == "line":
                    fig = px.line(
                        melted,
                        x=x_axis,
                        y="value",
                        color="pollutant",
                        line_group="site",
                        markers=True,
                        title=f"{agg_label} - {', '.join(safe_pollutants)}"
                    )
                else:
                    fig = px.bar(
                        melted,
                        x=x_axis,
                        y="value",
                        color="pollutant",
                        barmode="group",
                        title=f"{agg_label} - {', '.join(safe_pollutants)}"
                    )

                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

# -------------------------
# Other tabs
# -------------------------
render_exceedances_tab(tabs[1], dfs, selected_years)
render_aqi_tab(tabs[2], selected_years, dfs)
render_daily_means_tab(tabs[3], dfs, selected_years)
render_dayofweek_means_tab(tabs[4], dfs, selected_years)
render_validation_report_tab(tabs[5], dfs)
