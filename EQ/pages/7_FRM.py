# app.py
# Air Quality Dashboard Aggregator
# Completeness-aware + Advanced Filters + Valid Count Display

import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
from functools import reduce

# -------------------------
# Settings
# -------------------------
DAILY_MIN_OBS = 18
PERIOD_MIN_CAPTURE = 0.75


# -------------------------
# Utilities
# -------------------------
def to_csv_download(df):
    return BytesIO(df.to_csv(index=False).encode("utf-8"))


def days_in_month(m):
    return pd.Period(m, freq="M").days_in_month


def days_in_quarter(q):
    p = pd.Period(q, freq="Q")
    return (p.end_time.normalize() - p.start_time.normalize()).days + 1


def days_in_year(y):
    y = int(y)
    return 366 if pd.Timestamp(f"{y}-12-31").is_leap_year else 365


# -------------------------
# Cleaning
# -------------------------
def parse_dates(df):
    for col in df.columns:
        if any(k in col.lower() for k in ["date","time","timestamp"]):
            dt = pd.to_datetime(df[col], errors="coerce")
            if dt.notna().any():
                df["datetime"] = dt
                return df.dropna(subset=["datetime"])
    return df


def standardize_columns(df):
    rename = {}
    for col in df.columns:
        c = col.lower().strip()
        if c in ["pm25","pm_2_5","pm2.5","pm 2.5"]:
            rename[col] = "pm25"
        elif c in ["pm10","pm_10","pm 10"]:
            rename[col] = "pm10"
        elif c in ["site","station","location","site_name"]:
            rename[col] = "site"
    return df.rename(columns=rename)


def cleaned(df):
    df = df.copy().rename(columns=lambda x: x.strip().lower())
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
    df["season"] = df["datetime"].dt.month.map(
        lambda m: "Harmattan" if m in [12,1,2] else "Non-Harmattan"
    )
    df["weekday_type"] = df["datetime"].dt.weekday.map(
        lambda x: "Weekend" if x >= 5 else "Weekday"
    )

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
        return pd.DataFrame(columns=group_cols+[pollutant,"n_valid_days"])

    mean_df = daily_valid.groupby(group_cols)[pollutant].mean().reset_index()
    counts = daily_valid.groupby(group_cols)["day"].nunique().reset_index(name="n_valid_days")

    out = mean_df.merge(counts, on=group_cols)

    out["total_days"] = out[group_cols[0]].apply(total_days_func)
    out["capture"] = out["n_valid_days"] / out["total_days"]

    out.loc[out["capture"] < PERIOD_MIN_CAPTURE, pollutant] = np.nan

    return out


def compute_aggregates(df, label, pollutant):

    aggregates = {}
    daily_valid = build_valid_daily_means(df, pollutant)

    # Daily
    daily = daily_valid.copy()
    daily["n_valid_days"] = 1
    aggregates[f"{label} - Daily Avg ({pollutant})"] = daily

    # Monthly
    aggregates[f"{label} - Monthly Avg ({pollutant})"] = apply_period_completeness(
        daily_valid, ["month","site"], pollutant, days_in_month
    )

    # Quarterly
    aggregates[f"{label} - Quarterly Avg ({pollutant})"] = apply_period_completeness(
        daily_valid, ["quarter","site"], pollutant, days_in_quarter
    )

    # Yearly
    aggregates[f"{label} - Yearly Avg ({pollutant})"] = apply_period_completeness(
        daily_valid, ["year","site"], pollutant, days_in_year
    )

    return aggregates


# -------------------------
# STREAMLIT UI
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
all_sites = set()
all_years = set()

for file in uploaded_files:
    label = file.name.split(".")[0]
    ext = file.name.split(".")[-1]

    raw = pd.read_excel(file) if ext=="xlsx" else pd.read_csv(file)
    df = cleaned(standardize_columns(parse_dates(raw)))

    dfs[label] = df
    all_sites.update(df["site"].unique())
    all_years.update(df["year"].unique())


# -------------------------
# Global Sidebar Filters
# -------------------------
with st.sidebar:
    st.header("Global Filters")
    selected_years = st.multiselect("Filter Year", sorted(all_years))
    selected_sites = st.multiselect("Filter Site", sorted(all_sites))


tabs = st.tabs(["Aggregated Means"])


# -------------------------
# Aggregated Means Tab
# -------------------------
with tabs[0]:

    st.header("📊 Aggregated Means (Completeness Rule Applied)")

    for label, df in dfs.items():

        st.subheader(f"Dataset: {label}")

        # -------- Tab Filters --------
        site_in_tab = st.multiselect(
            f"Select Site(s)",
            sorted(df["site"].unique()),
            key=f"site_{label}"
        )

        # Year
        available_years = sorted(df["year"].unique())
        year_choices = ["All"] + available_years
        selected_year_choice = st.multiselect(
            "Select Year(s)",
            year_choices,
            default=["All"],
            key=f"year_{label}"
        )

        years_to_use = available_years if ("All" in selected_year_choice or not selected_year_choice) else selected_year_choice

        # Quarter
        quarter_choices = ["All","Q1","Q2","Q3","Q4"]
        selected_quarters = st.multiselect(
            "Select Quarter(s)",
            quarter_choices,
            default=["All"],
            key=f"quarter_{label}"
        )

        quarters_to_use = ["Q1","Q2","Q3","Q4"] if ("All" in selected_quarters or not selected_quarters) else selected_quarters

        # Month
        available_months = sorted(df["month"].unique())
        month_choices = ["All"] + available_months
        selected_months = st.multiselect(
            "Select Month(s)",
            month_choices,
            default=["All"],
            key=f"month_{label}"
        )

        months_to_use = available_months if ("All" in selected_months or not selected_months) else selected_months

        # Season
        season_choices = ["All","Harmattan","Non-Harmattan"]
        selected_seasons = st.multiselect(
            "Select Season(s)",
            season_choices,
            default=["All"],
            key=f"season_{label}"
        )

        seasons_to_use = ["Harmattan","Non-Harmattan"] if ("All" in selected_seasons or not selected_seasons) else selected_seasons

        # Weekday Type
        weekday_choices = ["All","Weekday","Weekend"]
        selected_weekdays = st.multiselect(
            "Select Weekday Type(s)",
            weekday_choices,
            default=["All"],
            key=f"weekday_{label}"
        )

        weekday_types_to_use = ["Weekday","Weekend"] if ("All" in selected_weekdays or not selected_weekdays) else selected_weekdays

        # -------- Apply Filters --------
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

        filtered_df = filtered_df[filtered_df["month"].isin(months_to_use)]
        filtered_df = filtered_df[filtered_df["season"].isin(seasons_to_use)]
        filtered_df = filtered_df[filtered_df["weekday_type"].isin(weekday_types_to_use)]

        if filtered_df.empty:
            st.warning("No data remaining after filtering.")
            continue

        valid_pollutants = [p for p in ["pm25","pm10"] if p in filtered_df.columns]
        agg_by_pollutant = {p: compute_aggregates(filtered_df,label,p) for p in valid_pollutants}

        levels = [
            ("Daily Avg",["day","site"]),
            ("Monthly Avg",["month","site"]),
            ("Quarterly Avg",["quarter","site"]),
            ("Yearly Avg",["year","site"]),
        ]

        for level_name, group_keys in levels:

            st.markdown(f"### {level_name}")

            frames = []
            for p in valid_pollutants:
                key = f"{label} - {level_name} ({p})"
                frames.append(agg_by_pollutant[p][key])

            merged_df = reduce(lambda l,r: pd.merge(l,r,on=group_keys,how="outer"),frames)

            value_cols = [c for c in merged_df.columns if c in valid_pollutants]
            count_cols = [c for c in merged_df.columns if "n_valid_days" in c]

            display_df = merged_df[group_keys + value_cols]
            st.dataframe(display_df, use_container_width=True)

            if count_cols:
                total_valid = merged_df[count_cols].sum().sum()
                st.caption(f"Total valid daily means used: {int(total_valid)} days")

                with st.expander("Show valid daily counts per group"):
                    st.dataframe(merged_df[group_keys + count_cols], use_container_width=True)

            st.download_button(
                f"Download {level_name}",
                to_csv_download(display_df),
                file_name=f"{label}_{level_name.replace(' ','_')}.csv"
            )

            st.markdown("---")
