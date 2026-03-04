# app.py
# Streamlit Air Quality Aggregation App
# (75% completeness rule + Q1–Q4 filter + valid daily counts shown)

import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from functools import reduce

# -------------------------
# Completeness parameters
# -------------------------
DAILY_MIN_OBS = 18
PERIOD_MIN_CAPTURE = 0.75


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
def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        cl = col.strip().lower()
        if any(k in cl for k in ["date", "time", "datetime", "timestamp"]):
            dt = pd.to_datetime(df[col], errors="coerce")
            if dt.notna().any():
                df = df.copy()
                df["datetime"] = dt
                df = df.dropna(subset=["datetime"])
                return df
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


def cleaned(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().rename(columns=lambda x: x.strip().lower())

    required = ["datetime", "site", "pm25", "pm10"]
    keep = [c for c in required if c in df.columns]
    df = df[keep].dropna()

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    for p in ["pm25", "pm10"]:
        if p in df.columns:
            df[p] = pd.to_numeric(df[p], errors="coerce")

    df["day"] = df["datetime"].dt.floor("D")
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.to_period("M").astype(str)
    df["quarter"] = df["datetime"].dt.to_period("Q").astype(str)
    df["dayofweek"] = df["datetime"].dt.day_name()
    df["weekday_type"] = df["datetime"].dt.weekday.map(lambda x: "Weekend" if x >= 5 else "Weekday")
    df["season"] = df["datetime"].dt.month.map(lambda m: "Harmattan" if m in [12,1,2] else "Non-Harmattan")

    return df


# -------------------------
# Completeness Logic
# -------------------------
def build_valid_daily_means(df: pd.DataFrame, pollutant: str):
    if pollutant not in df.columns:
        return pd.DataFrame()

    base = (
        df.groupby(["site", "day"])[pollutant]
        .agg(n_obs="count", value="mean")
        .reset_index()
    )

    valid = base[base["n_obs"] >= DAILY_MIN_OBS].copy()
    valid.rename(columns={"value": pollutant}, inplace=True)

    valid["year"] = valid["day"].dt.year
    valid["month"] = valid["day"].dt.to_period("M").astype(str)
    valid["quarter"] = valid["day"].dt.to_period("Q").astype(str)
    valid["dayofweek"] = valid["day"].dt.day_name()
    valid["weekday_type"] = valid["day"].dt.weekday.map(lambda x: "Weekend" if x >= 5 else "Weekday")
    valid["season"] = valid["day"].dt.month.map(lambda m: "Harmattan" if m in [12,1,2] else "Non-Harmattan")

    return valid


def apply_period_completeness(daily_valid, group_cols, pollutant, total_days_fn):
    mean_df = daily_valid.groupby(group_cols)[pollutant].mean().reset_index()
    counts = daily_valid.groupby(group_cols)["day"].nunique().reset_index(name="n_valid_days")
    out = mean_df.merge(counts, on=group_cols, how="left")

    out["total_days"] = out.apply(total_days_fn, axis=1)
    out["capture"] = out["n_valid_days"] / out["total_days"]

    out.loc[out["capture"] < PERIOD_MIN_CAPTURE, pollutant] = np.nan
    return out


# -------------------------
# Aggregation Engine
# -------------------------
def compute_aggregates(df: pd.DataFrame, label: str, pollutant: str):
    aggregates = {}
    daily_valid = build_valid_daily_means(df, pollutant)

    # Daily
    daily_counts = (
        daily_valid.groupby(["day","site"])[pollutant]
        .count()
        .reset_index(name="n_valid_days")
    )

    daily_df = daily_valid[["day","site",pollutant]].merge(
        daily_counts, on=["day","site"]
    )

    aggregates[f"{label} - Daily Avg ({pollutant})"] = daily_df

    # Monthly
    monthly = apply_period_completeness(
        daily_valid,
        ["month","site"],
        pollutant,
        lambda r: pd.Period(r["month"],"M").days_in_month
    )

    aggregates[f"{label} - Monthly Avg ({pollutant})"] = monthly

    # Quarterly
    quarterly = apply_period_completeness(
        daily_valid,
        ["quarter","site"],
        pollutant,
        lambda r: pd.Period(r["quarter"],"Q").days_in_quarter
    )

    aggregates[f"{label} - Quarterly Avg ({pollutant})"] = quarterly

    # Yearly
    yearly = apply_period_completeness(
        daily_valid,
        ["year","site"],
        pollutant,
        lambda r: 366 if pd.Timestamp(f"{int(r['year'])}-12-31").is_leap_year else 365
    )

    aggregates[f"{label} - Yearly Avg ({pollutant})"] = yearly

    return aggregates


# -------------------------
# UI
# -------------------------
st.title("Air Quality Dashboard Aggregator (Completeness-Aware)")

uploaded_files = st.file_uploader(
    "Upload CSV or Excel files",
    type=["csv","xlsx"],
    accept_multiple_files=True
)

if not uploaded_files:
    st.stop()

dfs = {}

for file in uploaded_files:
    label = file.name.split(".")[0]
    ext = file.name.split(".")[-1]

    raw = pd.read_excel(file) if ext=="xlsx" else pd.read_csv(file)
    df = cleaned(standardize_columns(parse_dates(raw)))

    dfs[label] = df


tabs = st.tabs(["Aggregated Means"])

# -------------------------
# Aggregated Means Tab
# -------------------------
with tabs[0]:

    for label, df in dfs.items():

        st.subheader(f"Dataset: {label}")

        valid_pollutants = [p for p in ["pm25","pm10"] if p in df.columns]
        agg_by_pollutant = {p: compute_aggregates(df,label,p) for p in valid_pollutants}

        aggregate_levels = [
            ("Daily Avg",["day","site"]),
            ("Monthly Avg",["month","site"]),
            ("Quarterly Avg",["quarter","site"]),
            ("Yearly Avg",["year","site"]),
        ]

        for level_name, group_keys in aggregate_levels:

            st.markdown(f"### {label} - {level_name}")

            frames = []
            for p in valid_pollutants:
                k = f"{label} - {level_name} ({p})"
                frames.append(agg_by_pollutant[p][k])

            merged_df = reduce(lambda l,r: pd.merge(l,r,on=group_keys,how="outer"),frames)

            value_cols = [c for c in merged_df.columns if c in valid_pollutants]
            count_cols = [c for c in merged_df.columns if "n_valid_days" in c]

            display_cols = group_keys + value_cols
            editable_df = merged_df[display_cols]

            st.dataframe(editable_df,use_container_width=True)

            # ---- NEW FEATURE ----
            if count_cols:
                total_valid = merged_df[count_cols].sum().sum()
                st.caption(f"Total valid daily means used: {int(total_valid)} days")

                with st.expander("Show valid daily counts per group"):
                    st.dataframe(merged_df[group_keys+count_cols],use_container_width=True)

            st.markdown("---")
