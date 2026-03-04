# app.py
# Streamlit Air Quality Aggregation App
# Completeness-aware + Valid Daily Count Display

import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
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


# -------------------------
# Date Helpers (FIXED)
# -------------------------
def days_in_month(m_str):
    return pd.Period(m_str, freq="M").days_in_month


def days_in_quarter(q_str):
    p = pd.Period(q_str, freq="Q")
    return (p.end_time.normalize() - p.start_time.normalize()).days + 1


def days_in_year(y):
    y = int(y)
    return 366 if pd.Timestamp(f"{y}-12-31").is_leap_year else 365


# -------------------------
# Parsing
# -------------------------
def parse_dates(df: pd.DataFrame):
    for col in df.columns:
        cl = col.lower()
        if any(k in cl for k in ["date", "time", "datetime", "timestamp"]):
            dt = pd.to_datetime(df[col], errors="coerce")
            if dt.notna().any():
                df["datetime"] = dt
                df = df.dropna(subset=["datetime"])
                return df
    return df


def standardize_columns(df):
    rename_map = {}
    for col in df.columns:
        c = col.lower().strip()
        if c in ["pm25","pm_2_5","pm2.5","pm 2.5"]:
            rename_map[col] = "pm25"
        elif c in ["pm10","pm_10","pm 10"]:
            rename_map[col] = "pm10"
        elif c in ["site","station","location","site_name"]:
            rename_map[col] = "site"
    df.rename(columns=rename_map, inplace=True)
    return df


def cleaned(df):
    df = df.copy()
    df = df.rename(columns=lambda x: x.strip().lower())

    required = ["datetime","site"]
    df = df[[c for c in df.columns if c in required or c in ["pm25","pm10"]]]
    df = df.dropna(subset=["datetime","site"])

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime"])

    for p in ["pm25","pm10"]:
        if p in df.columns:
            df[p] = pd.to_numeric(df[p], errors="coerce")

    df["day"] = df["datetime"].dt.floor("D")
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.to_period("M").astype(str)
    df["quarter"] = df["datetime"].dt.to_period("Q").astype(str)

    return df


# -------------------------
# Completeness Logic
# -------------------------
def build_valid_daily_means(df, pollutant):
    if pollutant not in df.columns:
        return pd.DataFrame()

    base = (
        df.groupby(["site","day"])[pollutant]
        .agg(n_obs="count", value="mean")
        .reset_index()
    )

    valid = base[base["n_obs"] >= DAILY_MIN_OBS].copy()
    valid.rename(columns={"value": pollutant}, inplace=True)

    valid["year"] = valid["day"].dt.year
    valid["month"] = valid["day"].dt.to_period("M").astype(str)
    valid["quarter"] = valid["day"].dt.to_period("Q").astype(str)

    return valid


def apply_period_completeness(daily_valid, group_cols, pollutant, total_days_func):
    if daily_valid.empty:
        return pd.DataFrame(columns=group_cols + [pollutant, "n_valid_days"])

    mean_df = daily_valid.groupby(group_cols)[pollutant].mean().reset_index()
    counts = daily_valid.groupby(group_cols)["day"].nunique().reset_index(name="n_valid_days")

    out = mean_df.merge(counts, on=group_cols)

    out["total_days"] = out.apply(lambda r: total_days_func(r[group_cols[0]]), axis=1)
    out["capture"] = out["n_valid_days"] / out["total_days"]

    out.loc[out["capture"] < PERIOD_MIN_CAPTURE, pollutant] = np.nan

    return out


# -------------------------
# Aggregation Engine
# -------------------------
def compute_aggregates(df, label, pollutant):

    aggregates = {}
    daily_valid = build_valid_daily_means(df, pollutant)

    # Daily
    daily = daily_valid.copy()
    daily["n_valid_days"] = 1
    aggregates[f"{label} - Daily Avg ({pollutant})"] = daily

    # Monthly
    monthly = apply_period_completeness(
        daily_valid,
        ["month","site"],
        pollutant,
        days_in_month
    )
    aggregates[f"{label} - Monthly Avg ({pollutant})"] = monthly

    # Quarterly (FIXED)
    quarterly = apply_period_completeness(
        daily_valid,
        ["quarter","site"],
        pollutant,
        days_in_quarter
    )
    aggregates[f"{label} - Quarterly Avg ({pollutant})"] = quarterly

    # Yearly
    yearly = apply_period_completeness(
        daily_valid,
        ["year","site"],
        pollutant,
        days_in_year
    )
    aggregates[f"{label} - Yearly Avg ({pollutant})"] = yearly

    return aggregates


# -------------------------
# STREAMLIT APP
# -------------------------
st.title("Air Quality Dashboard (Completeness-Aware)")

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


st.header("Aggregated Means")

for label, df in dfs.items():

    st.subheader(f"Dataset: {label}")

    valid_pollutants = [p for p in ["pm25","pm10"] if p in df.columns]
    agg_by_pollutant = {p: compute_aggregates(df,label,p) for p in valid_pollutants}

    levels = [
        ("Daily Avg",["day","site"]),
        ("Monthly Avg",["month","site"]),
        ("Quarterly Avg",["quarter","site"]),
        ("Yearly Avg",["year","site"]),
    ]

    for level_name, group_keys in levels:

        st.markdown(f"### {label} - {level_name}")

        frames = []
        for p in valid_pollutants:
            key = f"{label} - {level_name} ({p})"
            frames.append(agg_by_pollutant[p][key])

        merged_df = reduce(lambda l,r: pd.merge(l,r,on=group_keys,how="outer"),frames)

        value_cols = [c for c in merged_df.columns if c in valid_pollutants]
        count_cols = [c for c in merged_df.columns if "n_valid_days" in c]

        display_cols = group_keys + value_cols
        editable_df = merged_df[display_cols]

        st.dataframe(editable_df, use_container_width=True)

        # -------- Show Count Used --------
        if count_cols:
            total_valid = merged_df[count_cols].sum().sum()
            st.caption(f"Total valid daily means used for this aggregation: {int(total_valid)} days")

            with st.expander("Show valid daily counts per group"):
                st.dataframe(merged_df[group_keys + count_cols], use_container_width=True)

        st.download_button(
            f"Download {level_name}",
            to_csv_download(editable_df),
            file_name=f"{label}_{level_name.replace(' ','_')}.csv"
        )

        st.markdown("---")
