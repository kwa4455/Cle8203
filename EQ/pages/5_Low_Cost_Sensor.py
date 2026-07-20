# -*- coding: utf-8 -*-
# app.py
# Streamlit Air Quality Aggregation App (75% completeness rule + Q1–Q4 filter in Aggregated Means + All-years option)

import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from functools import reduce
from dateutil import parser
# -------------------------
# Completeness parameters
# -------------------------
DAILY_MIN_OBS = 18        # >= 18 hourly records for a valid daily mean (set to 1 if data is already daily)
EXPECTED_HOURS_PER_DAY = 24
PERIOD_MIN_CAPTURE = 0.75  # >= 75% of days in the period must be valid


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
    dt = dt1.copy()

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
    return filtered, (df[col] < lower_threshold).sum(), (df[col] > upper_threshold).sum()


def cleaned(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.rename(columns=lambda x: x.strip().lower())

    # Keep only relevant columns if present
    required_columns = ["datetime", "site", "pm25", "pm10"]
    keep = [c for c in required_columns if c in df.columns]
    df = df[keep].dropna(axis=1, how="all").dropna()

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.dropna(subset=["datetime"])

    for p in ["pm25", "pm10"]:
        if p in df.columns:
            df[p] = pd.to_numeric(df[p], errors="coerce")

    df = df.dropna(subset=[c for c in ["site"] if c in df.columns])

    # Outlier filters (adjust to your needs)
    if "pm25" in df.columns:
        df, _, _ = removal_box_plot(df, "pm25", 1, 500)
    if "pm10" in df.columns:
        df, _, _ = removal_box_plot(df, "pm10", 1, 1000)

    # Time fields
    df["day"] = df["datetime"].dt.floor("D")
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.to_period("M").astype(str)      # "YYYY-MM"
    df["quarter"] = df["datetime"].dt.to_period("Q").astype(str)    # "YYYYQ#"
    df["dayofweek"] = df["datetime"].dt.day_name()
    df["weekday_type"] = df["datetime"].dt.weekday.map(lambda x: "Weekend" if x >= 5 else "Weekday")
    df["season"] = df["datetime"].dt.month.map(lambda m: "Harmattan" if m in [12, 1, 2] else "Non-Harmattan")

    # Optional: minimum days/month filter (NOT the 75% rule; extra filter)
    daily_counts = df.groupby(["site", "month"])["day"].nunique().reset_index(name="daily_counts")
    sufficient_sites = daily_counts[daily_counts["daily_counts"] >= 1][["site", "month"]]
    df = df.merge(sufficient_sites, on=["site", "month"], how="inner")

    return df


# -------------------------
# Completeness helpers
# -------------------------
def build_valid_daily_means(df: pd.DataFrame, pollutant: str, daily_min_obs: int = DAILY_MIN_OBS) -> pd.DataFrame:
    """Valid daily mean requires >= daily_min_obs observations/day."""
    if pollutant not in df.columns:
        return pd.DataFrame(columns=["site", "day", pollutant])

    base = (
        df.groupby(["site", "day"])[pollutant]
          .agg(n_obs="count", value="mean")
          .reset_index()
    )

    valid = base[base["n_obs"] >= daily_min_obs].copy()
    valid.rename(columns={"value": pollutant}, inplace=True)

    # Attach time labels
    valid["year"] = valid["day"].dt.year
    valid["month"] = valid["day"].dt.to_period("M").astype(str)
    valid["quarter"] = valid["day"].dt.to_period("Q").astype(str)
    valid["dayofweek"] = valid["day"].dt.day_name()
    valid["weekday_type"] = valid["day"].dt.weekday.map(lambda x: "Weekend" if x >= 5 else "Weekday")
    valid["season"] = valid["day"].dt.month.map(lambda m: "Harmattan" if m in [12, 1, 2] else "Non-Harmattan")

    return valid




def build_daily_completeness_report(
    df: pd.DataFrame,
    pollutant: str,
    daily_min_obs: int = DAILY_MIN_OBS,
) -> pd.DataFrame:
    """
    Build a day-level completeness audit.

    The current app treats each non-missing pollutant record as one hourly
    observation. A day is valid when at least DAILY_MIN_OBS (default 18)
    hourly observations are available. Only valid days contribute to the
    monthly, quarterly, yearly and seasonal means.
    """
    columns = [
        "site", "pollutant", "day", "year", "month", "quarter", "season",
        "hours_available", "required_valid_hours", "hours_used",
        "hour_capture_percent", "valid_day", "daily_mean", "daily_mean_used"
    ]

    if pollutant not in df.columns or df.empty:
        return pd.DataFrame(columns=columns)

    base = (
        df.groupby(["site", "day"])[pollutant]
          .agg(hours_available="count", daily_mean="mean")
          .reset_index()
    )

    if base.empty:
        return pd.DataFrame(columns=columns)

    base["pollutant"] = pollutant
    base["required_valid_hours"] = daily_min_obs
    base["hour_capture_percent"] = (
        base["hours_available"] / EXPECTED_HOURS_PER_DAY * 100
    ).round(1)
    base["valid_day"] = base["hours_available"] >= daily_min_obs

    # Hours and daily means actually admitted to later period calculations.
    base["hours_used"] = np.where(base["valid_day"], base["hours_available"], 0)
    base["daily_mean_used"] = base["daily_mean"].where(base["valid_day"])

    base["year"] = base["day"].dt.year
    base["month"] = base["day"].dt.to_period("M").astype(str)
    base["quarter"] = base["day"].dt.to_period("Q").astype(str)
    base["season"] = base["day"].dt.month.map(
        lambda m: "Harmattan" if m in [12, 1, 2] else "Non-Harmattan"
    )

    base["valid_day"] = base["valid_day"].map({True: "Valid", False: "Invalid"})
    base["daily_mean"] = base["daily_mean"].round(1)
    base["daily_mean_used"] = base["daily_mean_used"].round(1)

    return base[columns].sort_values(["site", "day"]).reset_index(drop=True)


def build_period_completeness_report(
    daily_report: pd.DataFrame,
    period: str,
    min_capture: float = PERIOD_MIN_CAPTURE,
) -> pd.DataFrame:
    """Summarize valid days and hours used for month, quarter, year or season."""
    if daily_report.empty:
        return pd.DataFrame()

    data = daily_report.copy()
    data["is_valid_day"] = data["valid_day"].eq("Valid")
    data["valid_hours_for_analysis"] = np.where(
        data["is_valid_day"], data["hours_available"], 0
    )

    if period == "Month":
        group_cols = ["site", "pollutant", "month"]

        def total_days_fn(row):
            return pd.Period(row["month"], freq="M").days_in_month

    elif period == "Quarter":
        group_cols = ["site", "pollutant", "quarter"]

        def total_days_fn(row):
            return _days_in_quarter(row["quarter"])

    elif period == "Year":
        group_cols = ["site", "pollutant", "year"]

        def total_days_fn(row):
            return _days_in_year(row["year"])

    elif period == "Season":
        group_cols = ["site", "pollutant", "year", "season"]

        def total_days_fn(row):
            y = int(row["year"])
            if row["season"] == "Harmattan":
                return (
                    pd.Period(f"{y}-01", "M").days_in_month
                    + pd.Period(f"{y}-02", "M").days_in_month
                    + pd.Period(f"{y}-12", "M").days_in_month
                )
            return sum(pd.Period(f"{y}-{m:02d}", "M").days_in_month for m in range(3, 12))

    else:
        raise ValueError("period must be one of: Month, Quarter, Year, Season")

    summary = (
        data.groupby(group_cols, dropna=False)
            .agg(
                days_with_data=("day", "nunique"),
                valid_days_used=("is_valid_day", "sum"),
                hours_available=("hours_available", "sum"),
                valid_hours_used=("valid_hours_for_analysis", "sum"),
            )
            .reset_index()
    )

    summary["total_expected_days"] = summary.apply(total_days_fn, axis=1)
    summary["required_valid_days"] = np.ceil(
        summary["total_expected_days"] * min_capture
    ).astype(int)
    summary["invalid_or_missing_days"] = (
        summary["total_expected_days"] - summary["valid_days_used"]
    )
    summary["valid_day_capture_percent"] = (
        summary["valid_days_used"] / summary["total_expected_days"] * 100
    ).round(1)
    summary["minimum_required_percent"] = round(min_capture * 100, 1)
    summary["period_status"] = np.where(
        summary["valid_day_capture_percent"] >= min_capture * 100,
        "Pass",
        "Fail",
    )

    return summary


def render_completeness_tab(tab, dfs, selected_years, selected_sites):
    with tab:
        st.header("✅ Completeness Validation Report")
        st.info(
            f"Validation rules used by the app: a valid day requires at least "
            f"{DAILY_MIN_OBS} hourly observations. Monthly, quarterly, yearly "
            f"and seasonal periods require at least {int(PERIOD_MIN_CAPTURE * 100)}% "
            f"valid days. Only hours from valid days are counted as 'valid hours used'."
        )

        for label, df in dfs.items():
            st.subheader(f"Dataset: {label}")

            filtered_df = df.copy()
            if selected_years:
                filtered_df = filtered_df[filtered_df["year"].isin(selected_years)]
            if selected_sites:
                filtered_df = filtered_df[filtered_df["site"].isin(selected_sites)]

            site_in_tab = st.multiselect(
                f"Select Site(s) for completeness report - {label}",
                options=sorted(filtered_df["site"].dropna().unique()),
                key=f"site_complete_{label}",
            )
            if site_in_tab:
                filtered_df = filtered_df[filtered_df["site"].isin(site_in_tab)]

            available_years = sorted(filtered_df["year"].dropna().unique())
            year_in_tab = st.multiselect(
                f"Select Year(s) for completeness report - {label}",
                options=available_years,
                default=available_years,
                key=f"year_complete_{label}",
            )
            if year_in_tab:
                filtered_df = filtered_df[filtered_df["year"].isin(year_in_tab)]

            pollutants = [p for p in ["pm25", "pm10"] if p in filtered_df.columns]
            selected_pollutants = st.multiselect(
                f"Select Pollutant(s) - {label}",
                options=pollutants,
                default=pollutants,
                key=f"pollutant_complete_{label}",
            )

            if filtered_df.empty or not selected_pollutants:
                st.warning(f"No data available for the selected completeness filters in {label}.")
                continue

            daily_frames = []
            for pollutant in selected_pollutants:
                report = build_daily_completeness_report(
                    filtered_df, pollutant, DAILY_MIN_OBS
                )
                if not report.empty:
                    daily_frames.append(report)

            if not daily_frames:
                st.warning(f"No completeness report could be generated for {label}.")
                continue

            daily_report = pd.concat(daily_frames, ignore_index=True)

            total_evaluated = len(daily_report)
            valid_days = int((daily_report["valid_day"] == "Valid").sum())
            invalid_days = int((daily_report["valid_day"] == "Invalid").sum())
            valid_hours_used = int(daily_report["hours_used"].sum())

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Days evaluated", f"{total_evaluated:,}")
            c2.metric("Valid days used", f"{valid_days:,}")
            c3.metric("Invalid days excluded", f"{invalid_days:,}")
            c4.metric("Valid hours used", f"{valid_hours_used:,}")

            st.markdown("#### Daily validation audit")
            st.caption(
                "`hours_available` is the number of non-missing hourly pollutant records for the day. "
                "`hours_used` equals `hours_available` only when the day reaches the 18-hour validity threshold; "
                "otherwise it is 0 and the daily mean is excluded."
            )
            st.dataframe(daily_report, use_container_width=True)
            st.download_button(
                label=f"⬇️ Download Daily Completeness Audit - {label}",
                data=to_csv_download(daily_report),
                file_name=f"Completeness_Daily_Audit_{label}.csv",
                mime="text/csv",
                key=f"dl_daily_complete_{label}",
            )

            st.markdown("#### Period completeness summaries")
            for period in ["Month", "Quarter", "Year", "Season"]:
                period_report = build_period_completeness_report(
                    daily_report, period, PERIOD_MIN_CAPTURE
                )
                with st.expander(f"{period} completeness", expanded=(period == "Month")):
                    st.dataframe(period_report, use_container_width=True)
                    st.download_button(
                        label=f"⬇️ Download {period} Completeness - {label}",
                        data=to_csv_download(period_report),
                        file_name=f"Completeness_{period}_{label}.csv",
                        mime="text/csv",
                        key=f"dl_{period.lower()}_complete_{label}",
                    )

            st.markdown("---")


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
    """Compute mean only if >= min_capture of days in the period have valid daily means."""
    if daily_valid.empty or pollutant not in daily_valid.columns:
        return pd.DataFrame(columns=group_cols + [pollutant])

    mean_df = daily_valid.groupby(group_cols)[pollutant].mean().reset_index()
    counts = daily_valid.groupby(group_cols)["day"].nunique().reset_index(name="valid_days")
    out = mean_df.merge(counts, on=group_cols, how="left")

    out["total_days"] = out.apply(total_days_fn, axis=1)
    out["capture"] = out["valid_days"] / out["total_days"]

    out.loc[out["capture"] < min_capture, pollutant] = np.nan
    return out


def compute_aggregates(df: pd.DataFrame, label: str, pollutant: str):
    """Returns dict of DataFrames for this pollutant with completeness applied."""
    aggregates = {}
    daily_valid = build_valid_daily_means(df, pollutant, DAILY_MIN_OBS)

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

    # Season completeness per (year, season, site)
    def season_total_days(row):
        y = int(row["year"])
        s = row["season"]
        if s == "Harmattan":
            # Jan + Feb + Dec within same calendar year
            return (pd.Period(f"{y}-01", "M").days_in_month +
                    pd.Period(f"{y}-02", "M").days_in_month +
                    pd.Period(f"{y}-12", "M").days_in_month)
        else:
            months = [f"{y}-{m:02d}" for m in range(3, 12)]  # Mar..Nov
            return sum(pd.Period(m, "M").days_in_month for m in months)

    season = apply_period_completeness(
        daily_valid, ["season", "year", "site"], pollutant,
        total_days_fn=season_total_days
    )
    aggregates[f"{label} - Season Avg ({pollutant})"] = season[["season", "year", "site", pollutant]].round(1)

    return aggregates


# -------------------------
# Exceedances / MinMax (valid daily only)
# -------------------------
def calculate_exceedances(df: pd.DataFrame) -> pd.DataFrame:
    pm25_daily = build_valid_daily_means(df, "pm25", DAILY_MIN_OBS)
    pm10_daily = build_valid_daily_means(df, "pm10", DAILY_MIN_OBS)

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

    exceedance["pm25_Exceedance_Percent"] = round((exceedance["pm25_Exceedance_Count"] / exceedance["Total_Valid_Days"]) * 100, 1)
    exceedance["pm10_Exceedance_Percent"] = round((exceedance["pm10_Exceedance_Count"] / exceedance["Total_Valid_Days"]) * 100, 1)

    return exceedance


def calculate_min_max(df: pd.DataFrame) -> pd.DataFrame:
    pm25_daily = build_valid_daily_means(df, "pm25", DAILY_MIN_OBS)
    pm10_daily = build_valid_daily_means(df, "pm10", DAILY_MIN_OBS)

    daily = pm25_daily[["site", "day", "year", "month", "quarter", "pm25"]].merge(
        pm10_daily[["site", "day", "pm10"]],
        on=["site", "day"],
        how="outer"
    )

    out = (
        daily.groupby(["site", "year", "quarter", "month"], as_index=False)
            .agg(
                daily_avg_pm10_max=("pm10", lambda x: round(np.nanmax(x.values), 1) if np.isfinite(np.nanmax(x.values)) else np.nan),
                daily_avg_pm10_min=("pm10", lambda x: round(np.nanmin(x.values), 1) if np.isfinite(np.nanmin(x.values)) else np.nan),
                daily_avg_pm25_max=("pm25", lambda x: round(np.nanmax(x.values), 1) if np.isfinite(np.nanmax(x.values)) else np.nan),
                daily_avg_pm25_min=("pm25", lambda x: round(np.nanmin(x.values), 1) if np.isfinite(np.nanmin(x.values)) else np.nan),
            )
    )
    return out


# -------------------------
# AQI (valid daily pm25 only)
# -------------------------
def calculate_aqi_and_category(df, site_mode_label="Combined_All_Filtered_Sites", single_site=False):
    work = df[["site", "day", "year", "quarter", "month", "pm25"]].dropna().copy()

    daily_site_avg = (
        work.groupby(["site", "day", "year", "quarter", "month"], as_index=False)
        .agg(pm25=("pm25", lambda x: round(x.mean(), 1)))
        .dropna()
    )

    breakpoints = [
        (0.0, 9.0, 0, 50),
        (9.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200),
        (125.5, 225.4, 201, 300),
        (225.5, 325.4, 301, 500),
        (325.5, 99999.9, 501, 999)
    ]

    def calculate_aqi(pm):
        for low, high, aqi_low, aqi_high in breakpoints:
            if low <= pm <= high:
                return round(((pm - low) * (aqi_high - aqi_low) / (high - low)) + aqi_low)
        return np.nan

    daily_site_avg["AQI"] = daily_site_avg["pm25"].apply(calculate_aqi)

    conditions = [
        (daily_site_avg["AQI"] >= 0) & (daily_site_avg["AQI"] <= 50),
        (daily_site_avg["AQI"] > 50) & (daily_site_avg["AQI"] <= 100),
        (daily_site_avg["AQI"] > 100) & (daily_site_avg["AQI"] <= 150),
        (daily_site_avg["AQI"] > 150) & (daily_site_avg["AQI"] <= 200),
        (daily_site_avg["AQI"] > 200) & (daily_site_avg["AQI"] <= 300),
        (daily_site_avg["AQI"] > 300)
    ]

    remarks = [
        "Good",
        "Moderate",
        "Unhealthy for Sensitive Groups",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous"
    ]

    daily_site_avg["AQI_Remark"] = np.select(conditions, remarks, default="Unknown")
    daily_site_avg = daily_site_avg.dropna()

    if single_site:
        daily_avg = daily_site_avg.copy()
        daily_avg["site_label"] = daily_avg["site"]
    else:
        daily_avg = daily_site_avg.copy()
        daily_avg["site_label"] = site_mode_label

    if single_site:
        remarks_counts = (
            daily_avg.groupby(["site_label", "year", "AQI_Remark"], as_index=False)
            .size()
            .rename(columns={"size": "Count"})
        )
        remarks_counts["Total_Count_Per_Year"] = (
            remarks_counts.groupby(["site_label", "year"])["Count"].transform("sum")
        )
    else:
        remarks_counts = (
            daily_avg.groupby(["year", "AQI_Remark"], as_index=False)
            .size()
            .rename(columns={"size": "Count"})
        )
        remarks_counts["site_label"] = site_mode_label
        remarks_counts["Total_Count_Per_Year"] = (
            remarks_counts.groupby(["year"])["Count"].transform("sum")
        )
        remarks_counts = remarks_counts[
            ["site_label", "year", "AQI_Remark", "Count", "Total_Count_Per_Year"]
        ]

    remarks_counts["Percent"] = (
        remarks_counts["Count"] / remarks_counts["Total_Count_Per_Year"] * 100
    ).round(1)

    remarks_counts = remarks_counts.dropna()

    return daily_avg, remarks_counts



# -------------------------
# Tabs
# -------------------------
def render_exceedances_tab(tab, dfs, selected_years):
    with tab:
        st.header("🌫️ Exceedances & Max/Min (valid daily only)")

        for label, df in dfs.items():
            st.subheader(f"Dataset: {label}")

            site_in_tab = st.multiselect(
                f"Select Site(s) for {label}",
                options=sorted(df["site"].unique()),
                key=f"site_exc_{label}",
            )

            available_years = sorted(df["year"].unique())
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
            st.dataframe(exceedances, use_container_width=True)
            st.download_button(
                label=f"⬇️ Download Exceedances - {label}",
                data=to_csv_download(exceedances),
                file_name=f"Exceedances_{label}.csv",
                key=f"download_exceedances_{label}",
            )

            min_max = calculate_min_max(filtered_df)
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
                st.warning(f"No valid daily pm25 (>= {DAILY_MIN_OBS} obs/day) for AQI calculation.")
                continue

            st.dataframe(remarks_counts, use_container_width=True)
            st.dataframe(daily_avg, use_container_width=True)

            st.download_button(f"⬇️ Download Daily Avg - {label}", to_csv_download(daily_avg), file_name=f"DailyAvg_{label}.csv")
            st.download_button(f"⬇️ Download AQI - {label}", to_csv_download(remarks_counts), file_name=f"AQI_{label}.csv")


def calculate_day_pollutant(df: pd.DataFrame, pollutant: str) -> pd.DataFrame:
    daily = build_valid_daily_means(df, pollutant, DAILY_MIN_OBS)
    if daily.empty:
        return pd.DataFrame(columns=["site", "day", "month", "quarter", "year", pollutant])
    return daily[["site", "day", "month", "quarter", "year", pollutant]].round(1)


def render_daily_means_tab(tab, dfs, selected_years):
    with tab:
        st.header("📊 Daily Means (valid days only)")

        for label, df in dfs.items():
            st.subheader(f"Dataset: {label}")

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

            years_use = selected_years if selected_years else sorted(df["year"].unique())
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
                st.warning(f"No valid daily data (>= {DAILY_MIN_OBS} obs/day) in selection.")
                continue

            df_avg = pd.concat(df_list, ignore_index=True)

            if chart_type == "Line":
                fig = px.line(df_avg, x="day", y="value", color="pollutant", line_group="site",
                              markers=True, title=f"Daily Means - {label}",
                              hover_data=["site", "pollutant", "value"])
            else:
                fig = px.bar(df_avg, x="day", y="value", color="site", barmode="group",
                             facet_col="pollutant" if len(selected_display_pollutants) > 1 else None,
                             title=f"Daily Means - {label}")

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

            years_use = selected_years if selected_years else sorted(df["year"].unique())
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
                daily = build_valid_daily_means(filtered_df, p, DAILY_MIN_OBS)
                if daily.empty:
                    continue
                g = daily.groupby(["site", "dayofweek", "quarter", "year"])[p].mean().reset_index().round(1)
                g["pollutant"] = p
                g.rename(columns={p: "value"}, inplace=True)
                out_list.append(g)

            if not out_list:
                st.warning(f"No valid daily data (>= {DAILY_MIN_OBS} obs/day) in selection.")
                continue

            df_avg = pd.concat(out_list, ignore_index=True)

            if chart_type == "Line":
                fig = px.line(df_avg, x="dayofweek", y="value", color="pollutant", line_group="site",
                              markers=True, title=f"Day of Week Means - {label}")
            else:
                fig = px.bar(df_avg, x="dayofweek", y="value", color="site", barmode="group",
                             facet_col="pollutant" if len(selected_display_pollutants) > 1 else None,
                             title=f"Day of Week Means - {label}")

            st.plotly_chart(fig, use_container_width=True)
            with st.expander("Show Table"):
                st.dataframe(df_avg, use_container_width=True)


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
    label = file.name.split(".")[0]
    ext = file.name.split(".")[-1].lower()

    try:
        raw = pd.read_excel(file) if ext == "xlsx" else pd.read_csv(file)
    except Exception as e:
        st.warning(f"⚠️ Could not read {file.name}: {e}")
        continue

    df = parse_dates(raw)
    df = standardize_columns(df)
    df = cleaned(df)

    if "datetime" not in df.columns or "site" not in df.columns:
        st.warning(f"⚠️ Could not process {label}: missing datetime/site.")
        continue

    if "pm25" not in df.columns and "pm10" not in df.columns:
        st.warning(f"⚠️ Could not process {label}: missing both pm25 and pm10.")
        continue

    dfs[label] = df
    site_options.update(df["site"].dropna().unique().tolist())
    year_options.update(df["year"].dropna().unique().tolist())

if not dfs:
    st.warning("No datasets could be processed. Please check your columns and datetime formatting.")
    st.stop()

with st.sidebar:
    st.header("Filters (global)")
    selected_years = st.multiselect("📅 Filter by Year", sorted(year_options))
    selected_sites = st.multiselect("🏢 Filter by Site", sorted(site_options))
    st.caption(f"Daily validity: ≥ {DAILY_MIN_OBS} obs/day")
    st.caption(f"Period completeness: ≥ {int(PERIOD_MIN_CAPTURE*100)}% valid days")

tabs = st.tabs([
    "Aggregated Means",
    "Exceedances & Max/Min Values",
    "AQI Stats",
    "Daily Means",
    "Day of Week Means",
    "Completeness Validation Report",
])

# ---- Aggregated Means (with Q1-Q4 filter + All year option) ----
with tabs[0]:
    st.header("📊 Aggregated Means (Completeness Rule Applied)")

    for label, df in dfs.items():
        st.subheader(f"Dataset: {label}")

        # --- Tab-specific site filter ---
        site_in_tab = st.multiselect(
            f"Select Site(s) for {label}",
            sorted(df["site"].unique()),
            key=f"site_agg_{label}",
        )

        # --- Tab-specific year filter with "All" ---
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

        # --- Tab-specific quarter filter with "All" ---
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

        # --- Apply global + tab filters ---
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

        # Precompute aggregates per pollutant (completeness-aware)
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
                    fig = px.line(melted, x=x_axis, y="value", color="pollutant", line_group="site",
                                  markers=True, title=f"{agg_label} - {', '.join(safe_pollutants)}")
                else:
                    fig = px.bar(melted, x=x_axis, y="value", color="pollutant", barmode="group",
                                 title=f"{agg_label} - {', '.join(safe_pollutants)}")

                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

# ---- Other tabs ----
render_exceedances_tab(tabs[1], dfs, selected_years)
render_aqi_tab(tabs[2], selected_years, dfs)
render_daily_means_tab(tabs[3], dfs, selected_years)
render_dayofweek_means_tab(tabs[4], dfs, selected_years)
render_completeness_tab(tabs[5], dfs, selected_years, selected_sites)


