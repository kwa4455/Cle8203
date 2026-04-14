import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
from functools import reduce

# --- Page Configuration ---
st.set_page_config(
    page_title="Gravimetric Data Analyser Tool",
    page_icon="🛠️",
    layout="wide"
)

st.title("📊 Gravimetric Data Analyser Tool")

# ----------------------------
# Helper functions
# ----------------------------

@st.cache_data(ttl=600)
def cleaned(df):
    df = df.rename(columns=lambda x: str(x).strip().lower())

    required_columns = ["date", "site", "pm25", "pm10"]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df[required_columns].copy()
    df = df.dropna(axis=1, how="all")
    df = df.dropna(subset=["date", "site"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["date"])

    df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")
    df["pm10"] = pd.to_numeric(df["pm10"], errors="coerce")

    df = df.dropna(subset=["pm25", "pm10"], how="all")

    df["site"] = df["site"].astype(str).str.strip()

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["quarter"] = df["date"].dt.to_period("Q").astype(str)
    df["day"] = df["date"].dt.date
    df["dayofweek"] = df["date"].dt.day_name()
    df["weekday_type"] = np.where(df["date"].dt.weekday >= 5, "Weekend", "Weekday")
    df["season"] = np.where(df["date"].dt.month.isin([12, 1, 2]), "Harmattan", "Non-Harmattan")

    return df


def parse_dates(df):
    df = df.copy()
    for col in df.columns:
        if "date" in str(col).lower():
            parsed = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            if parsed.notna().sum() > 0:
                df[col] = parsed
    return df


def standardize_columns(df):
    df = df.copy()

    pm25_cols = ["pm25", "pm2.5", "pm_2_5", "pm25_avg", "pm 2.5", "pm2_5"]
    pm10_cols = ["pm10", "pm_10", "pm 10"]
    site_cols = ["site", "station", "location"]

    rename_map = {}
    for col in df.columns:
        col_lower = str(col).strip().lower()
        if col_lower in [c.lower() for c in pm25_cols]:
            rename_map[col] = "pm25"
        elif col_lower in [c.lower() for c in pm10_cols]:
            rename_map[col] = "pm10"
        elif col_lower in [c.lower() for c in site_cols]:
            rename_map[col] = "site"

    df = df.rename(columns=rename_map)
    return df


def to_csv_download(df):
    return BytesIO(df.to_csv(index=False).encode("utf-8"))


def unique_key(tab: str, widget: str, label: str) -> str:
    return f"{widget}_{tab}_{label}"


# ----------------------------
# Completeness validation
# ----------------------------

def expected_weekly_samples_for_period(period_str, period_type):
    """
    Estimate expected weekly samples for a period assuming monitoring occurs once every week.
    """
    if period_type == "month":
        p = pd.Period(period_str, freq="M")
    elif period_type == "quarter":
        p = pd.Period(period_str, freq="Q")
    elif period_type == "year":
        p = pd.Period(str(period_str), freq="Y")
    else:
        raise ValueError("period_type must be one of: month, quarter, year")

    start = p.start_time.normalize()
    end = p.end_time.normalize()

    # Count unique calendar weeks touching the period
    week_starts = pd.date_range(start=start, end=end, freq="W-MON")
    expected = max(len(week_starts), 1)

    # Make sure very short edge cases do not return zero
    return expected


def aggregate_with_weekly_completeness(df, pollutant, period_type, threshold=0.75):
    """
    Aggregate pollutant by site and period with 75% completeness check
    for weekly monitoring.
    """
    if period_type not in ["month", "quarter", "year"]:
        raise ValueError("period_type must be month, quarter, or year")

    work = df.copy()
    work["obs_day"] = pd.to_datetime(work["day"])

    if period_type == "month":
        group_cols = ["site", "month", "quarter", "year"]
        period_col = "month"
    elif period_type == "quarter":
        group_cols = ["site", "quarter", "year"]
        period_col = "quarter"
    else:
        group_cols = ["site", "year"]
        period_col = "year"

    grouped = (
        work.groupby(group_cols)
        .agg(
            value=(pollutant, "mean"),
            observed_samples=("obs_day", "nunique")
        )
        .reset_index()
    )

    grouped["expected_samples"] = grouped[period_col].apply(
        lambda x: expected_weekly_samples_for_period(x, period_type)
    )
    grouped["completeness_percent"] = (
        grouped["observed_samples"] / grouped["expected_samples"] * 100
    ).round(1)
    grouped["is_valid"] = grouped["completeness_percent"] >= (threshold * 100)

    grouped["value"] = grouped["value"].round(1)

    valid_df = grouped[grouped["is_valid"]].copy()
    invalid_df = grouped[~grouped["is_valid"]].copy()

    return valid_df, invalid_df


# ----------------------------
# Core calculations
# ----------------------------

def calculate_day_pollutant(df, pollutant):
    return (
        df.groupby(["site", "day", "month", "quarter", "year"])[pollutant]
        .mean()
        .reset_index()
        .round(1)
    )


def calculate_month_pollutant(df, pollutant):
    valid_df, invalid_df = aggregate_with_weekly_completeness(df, pollutant, "month", threshold=0.75)
    return valid_df, invalid_df


def calculate_quarter_pollutant(df, pollutant):
    valid_df, invalid_df = aggregate_with_weekly_completeness(df, pollutant, "quarter", threshold=0.75)
    return valid_df, invalid_df


def calculate_year_pollutant(df, pollutant):
    valid_df, invalid_df = aggregate_with_weekly_completeness(df, pollutant, "year", threshold=0.75)
    return valid_df, invalid_df


def calculate_dayofweek_pollutant(df, pollutant):
    return (
        df.groupby(["site", "dayofweek", "quarter", "year"])[pollutant]
        .mean()
        .reset_index()
        .round(1)
    )


def calculate_exceedances(df):
    daily_avg = df.groupby(["site", "day", "year", "quarter", "month"], as_index=False).agg({
        "pm25": "mean",
        "pm10": "mean"
    })

    pm25_exceed = (
        daily_avg[daily_avg["pm25"] > 35]
        .groupby(["year", "site"])
        .size()
        .reset_index(name="pm25_Exceedance_Count")
    )

    pm10_exceed = (
        daily_avg[daily_avg["pm10"] > 70]
        .groupby(["year", "site"])
        .size()
        .reset_index(name="pm10_Exceedance_Count")
    )

    total_days = daily_avg.groupby(["year", "site"]).size().reset_index(name="Total_Records")

    exceedance = (
        total_days.merge(pm25_exceed, on=["year", "site"], how="left")
        .merge(pm10_exceed, on=["year", "site"], how="left")
    )

    exceedance = exceedance.fillna(0)
    exceedance["pm25_Exceedance_Percent"] = (
        exceedance["pm25_Exceedance_Count"] / exceedance["Total_Records"] * 100
    ).round(1)
    exceedance["pm10_Exceedance_Percent"] = (
        exceedance["pm10_Exceedance_Count"] / exceedance["Total_Records"] * 100
    ).round(1)

    return exceedance


def calculate_min_max(df):
    daily_avg = df.groupby(["site", "day", "year", "quarter", "month"], as_index=False).agg({
        "pm25": "mean",
        "pm10": "mean"
    })

    return daily_avg.groupby(["site", "year", "quarter", "month"], as_index=False).agg(
        daily_avg_pm10_max=("pm10", lambda x: round(x.max(), 1)),
        daily_avg_pm10_min=("pm10", lambda x: round(x.min(), 1)),
        daily_avg_pm25_max=("pm25", lambda x: round(x.max(), 1)),
        daily_avg_pm25_min=("pm25", lambda x: round(x.min(), 1))
    )


def calculate_aqi_and_category(df):
    daily_avg = df.groupby(["day", "year", "quarter", "month"], as_index=False).agg({
        "pm25": lambda x: round(x.mean(), 1)
    })

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

    daily_avg["AQI"] = daily_avg["pm25"].apply(calculate_aqi)

    conditions = [
        (daily_avg["AQI"] >= 0) & (daily_avg["AQI"] <= 50),
        (daily_avg["AQI"] > 50) & (daily_avg["AQI"] <= 100),
        (daily_avg["AQI"] > 100) & (daily_avg["AQI"] <= 150),
        (daily_avg["AQI"] > 150) & (daily_avg["AQI"] <= 200),
        (daily_avg["AQI"] > 200) & (daily_avg["AQI"] <= 300),
        (daily_avg["AQI"] > 300)
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
    remarks_counts["Percent"] = (
        remarks_counts["Count"] / remarks_counts["Total_Count_Per_Year"] * 100
    ).round(1)

    return daily_avg, remarks_counts


# ----------------------------
# Rendering tabs
# ----------------------------

def render_exceedances_tab(tab, dfs, selected_years):
    with tab:
        st.header("🌫️ Exceedances & Max/Min")

        for label, df in dfs.items():
            st.subheader(f"Dataset: {label}")

            site_in_tab = st.multiselect(
                f"Select Site(s) for {label}",
                options=sorted(df["site"].unique()),
                key=f"site_exc_{label}"
            )

            available_years = sorted(df["year"].unique())
            selected_years_in_tab = st.multiselect(
                f"Select Year(s) for {label}",
                options=available_years,
                default=selected_years if selected_years else available_years,
                key=f"year_exc_{label}"
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
                key=f"quarter_exc_{label}"
            )

            selected_quarter_nums = [
                f"{year}{q}" for year in selected_years_in_tab for q in selected_quarters
            ] if selected_years_in_tab and selected_quarters else []

            if selected_quarter_nums:
                filtered_df = filtered_df[filtered_df["quarter"].isin(selected_quarter_nums)]
            else:
                st.warning("No valid quarters to filter.")
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
                key=f"download_exceedances_{label}"
            )

            min_max = calculate_min_max(filtered_df)
            st.dataframe(min_max, use_container_width=True)
            st.download_button(
                label=f"⬇️ Download MinMax - {label}",
                data=to_csv_download(min_max),
                file_name=f"MinMax_{label}.csv",
                key=f"download_minmax_{label}"
            )


def render_aqi_tab(tab, selected_years, dfs):
    with tab:
        st.header("🌫️ AQI Stats")

        for label, df in dfs.items():
            st.subheader(f"Dataset: {label}")

            all_years = sorted(df["year"].dropna().unique())
            default_years = all_years[-2:] if len(all_years) >= 2 else all_years

            selected_years_in_tab = st.multiselect(
                f"Select Year(s) for {label}",
                options=all_years,
                default=selected_years if selected_years else default_years,
                key=f"years_aqi_{label}"
            )

            site_in_tab = st.multiselect(
                f"Select Site(s) for {label}",
                sorted(df["site"].unique()),
                key=f"site_aqi_{label}"
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

            selected_quarter_nums = [
                f"{year}{q}" for year in selected_years_in_tab for q in selected_quarters
            ]

            if selected_quarter_nums:
                filtered_df = filtered_df[filtered_df["quarter"].isin(selected_quarter_nums)]

            if filtered_df.empty:
                st.warning(f"No data remaining for {label} after filtering.")
                continue

            daily_avg, remarks_counts = calculate_aqi_and_category(filtered_df)

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
        st.header("📊 Daily Means")

        for label, df in dfs.items():
            st.subheader(f"Dataset: {label}")

            site_in_tab = st.multiselect(
                f"Select Site(s) for {label}",
                sorted(df["site"].unique()),
                key=unique_key("tab5", "site", label)
            )

            filtered_df = df.copy()

            years_to_use = selected_years if selected_years else sorted(df["year"].unique())

            if years_to_use:
                filtered_df = filtered_df[filtered_df["year"].isin(years_to_use)]
            if site_in_tab:
                filtered_df = filtered_df[filtered_df["site"].isin(site_in_tab)]

            selected_quarters = st.multiselect(
                f"Select Quarter(s) for {label}",
                options=["Q1", "Q2", "Q3", "Q4"],
                default=["Q1", "Q2", "Q3", "Q4"],
                key=unique_key("tab5", "quarter", label)
            ) or []

            selected_quarter_nums = [f"{year}{q}" for year in years_to_use for q in selected_quarters]

            if selected_quarter_nums:
                filtered_df = filtered_df[filtered_df["quarter"].isin(selected_quarter_nums)]
            else:
                st.warning("No valid quarters to filter.")
                continue

            if filtered_df.empty:
                st.warning(f"No data remaining for {label} after filtering.")
                continue

            valid_pollutants = [p for p in ["pm25", "pm10"] if p in filtered_df.columns]

            selected_display_pollutants = st.multiselect(
                f"Select Pollutants to Display for {label}",
                options=["All"] + valid_pollutants,
                default=["All"],
                key=unique_key("tab5", "pollutants", label)
            )

            if "All" in selected_display_pollutants:
                selected_display_pollutants = valid_pollutants

            chart_type = st.radio(
                f"Chart Type for {label}",
                ["Line", "Bar"],
                horizontal=True,
                key=unique_key("tab5", "charttype", label)
            )

            df_avg_list = []
            for pollutant in selected_display_pollutants:
                avg_df = calculate_day_pollutant(filtered_df, pollutant)
                avg_df["pollutant"] = pollutant
                avg_df = avg_df.rename(columns={pollutant: "value"})
                df_avg_list.append(avg_df)

            if not df_avg_list:
                st.warning(f"No data available for selected pollutants in {label}")
                continue

            df_avg = pd.concat(df_avg_list, ignore_index=True)

            x_axis = "day"
            y_title = "µg/m³"
            plot_title = f"Aggregated {chart_type} Chart - {label}"

            if chart_type == "Line":
                fig = px.line(
                    df_avg,
                    x=x_axis,
                    y="value",
                    color="pollutant",
                    line_group="pollutant",
                    markers=True,
                    title=plot_title,
                    labels={"value": y_title, x_axis: x_axis.capitalize()}
                )
            else:
                fig = px.bar(
                    df_avg,
                    x=x_axis,
                    y="value",
                    color="site",
                    barmode="group",
                    facet_col="pollutant" if len(selected_display_pollutants) > 1 else None,
                    title=plot_title,
                    labels={"value": y_title, x_axis: x_axis.capitalize()}
                )

            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Show Aggregated Data Table"):
                st.dataframe(df_avg, use_container_width=True)


def render_monthly_means_tab(tab, dfs, selected_years):
    with tab:
        st.header("📊 Monthly Means (75% completeness applied)")

        for label, df in dfs.items():
            st.subheader(f"Dataset: {label}")

            site_in_tab = st.multiselect(
                f"Select Site(s) for {label}",
                sorted(df["site"].unique()),
                key=unique_key("tab2", "site", label)
            )

            filtered_df = df.copy()
            years_to_use = selected_years if selected_years else sorted(df["year"].unique())

            if years_to_use:
                filtered_df = filtered_df[filtered_df["year"].isin(years_to_use)]
            if site_in_tab:
                filtered_df = filtered_df[filtered_df["site"].isin(site_in_tab)]

            selected_quarters = st.multiselect(
                f"Select Quarter(s) for {label}",
                options=["Q1", "Q2", "Q3", "Q4"],
                default=["Q1", "Q2", "Q3", "Q4"],
                key=unique_key("tab2", "quarter", label)
            ) or []

            selected_quarter_nums = [f"{year}{q}" for year in years_to_use for q in selected_quarters]

            if selected_quarter_nums:
                filtered_df = filtered_df[filtered_df["quarter"].isin(selected_quarter_nums)]
            else:
                st.warning("No valid quarters to filter.")
                continue

            if filtered_df.empty:
                st.warning(f"No data remaining for {label} after filtering.")
                continue

            valid_pollutants = [p for p in ["pm25", "pm10"] if p in filtered_df.columns]

            selected_display_pollutants = st.multiselect(
                f"Select Pollutants to Display for {label}",
                options=["All"] + valid_pollutants,
                default=["All"],
                key=unique_key("tab2", "pollutants", label)
            )

            if "All" in selected_display_pollutants:
                selected_display_pollutants = valid_pollutants

            chart_type = st.radio(
                f"Chart Type for {label}",
                ["Line", "Bar"],
                horizontal=True,
                key=unique_key("tab2", "charttype", label)
            )

            valid_list = []
            invalid_list = []

            for pollutant in selected_display_pollutants:
                valid_df, invalid_df = calculate_month_pollutant(filtered_df, pollutant)
                valid_df["pollutant"] = pollutant
                valid_df = valid_df.rename(columns={"value": "value"})
                valid_list.append(valid_df)

                if not invalid_df.empty:
                    invalid_df["pollutant"] = pollutant
                    invalid_list.append(invalid_df)

            if not valid_list:
                st.warning(f"No monthly data met the 75% completeness rule for {label}.")
                continue

            df_avg = pd.concat(valid_list, ignore_index=True)

            if invalid_list:
                with st.expander(f"Show excluded monthly periods for {label}"):
                    st.dataframe(pd.concat(invalid_list, ignore_index=True), use_container_width=True)

            if chart_type == "Line":
                fig = px.line(
                    df_avg,
                    x="month",
                    y="value",
                    color="pollutant",
                    line_group="pollutant",
                    markers=True,
                    title=f"Monthly Means - {label}",
                    labels={"value": "µg/m³", "month": "Month"}
                )
            else:
                fig = px.bar(
                    df_avg,
                    x="month",
                    y="value",
                    color="site",
                    barmode="group",
                    facet_col="pollutant" if len(selected_display_pollutants) > 1 else None,
                    title=f"Monthly Means - {label}",
                    labels={"value": "µg/m³", "month": "Month"}
                )

            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Show Monthly Aggregated Data Table"):
                st.dataframe(df_avg, use_container_width=True)

            st.download_button(
                f"⬇️ Download Monthly Avg - {label}",
                to_csv_download(df_avg),
                file_name=f"MonthlyAvg_{label}.csv",
                key=f"download_monthly_{label}"
            )


def render_quarter_means_tab(tab, dfs, selected_years):
    with tab:
        st.header("📊 Quarterly Means (75% completeness applied)")

        for label, df in dfs.items():
            st.subheader(f"Dataset: {label}")

            site_in_tab = st.multiselect(
                f"Select Site(s) for {label}",
                sorted(df["site"].unique()),
                key=unique_key("tab1", "site", label)
            )

            filtered_df = df.copy()
            years_to_use = selected_years if selected_years else sorted(df["year"].unique())

            if years_to_use:
                filtered_df = filtered_df[filtered_df["year"].isin(years_to_use)]
            if site_in_tab:
                filtered_df = filtered_df[filtered_df["site"].isin(site_in_tab)]

            selected_quarters = st.multiselect(
                f"Select Quarter(s) for {label}",
                options=["Q1", "Q2", "Q3", "Q4"],
                default=["Q1", "Q2", "Q3", "Q4"],
                key=unique_key("tab1", "quarter", label)
            ) or []

            selected_quarter_nums = [f"{year}{q}" for year in years_to_use for q in selected_quarters]

            if selected_quarter_nums:
                filtered_df = filtered_df[filtered_df["quarter"].isin(selected_quarter_nums)]
            else:
                st.warning("No valid quarters to filter.")
                continue

            if filtered_df.empty:
                st.warning(f"No data remaining for {label} after filtering.")
                continue

            valid_pollutants = [p for p in ["pm25", "pm10"] if p in filtered_df.columns]

            selected_display_pollutants = st.multiselect(
                f"Select Pollutants to Display for {label}",
                options=["All"] + valid_pollutants,
                default=["All"],
                key=unique_key("tab1", "pollutants", label)
            )

            if "All" in selected_display_pollutants:
                selected_display_pollutants = valid_pollutants

            chart_type = st.radio(
                f"Chart Type for {label}",
                ["Line", "Bar"],
                horizontal=True,
                key=unique_key("tab1", "charttype", label)
            )

            valid_list = []
            invalid_list = []

            for pollutant in selected_display_pollutants:
                valid_df, invalid_df = calculate_quarter_pollutant(filtered_df, pollutant)
                valid_df["pollutant"] = pollutant
                valid_list.append(valid_df)

                if not invalid_df.empty:
                    invalid_df["pollutant"] = pollutant
                    invalid_list.append(invalid_df)

            if not valid_list:
                st.warning(f"No quarterly data met the 75% completeness rule for {label}.")
                continue

            df_avg = pd.concat(valid_list, ignore_index=True)

            if invalid_list:
                with st.expander(f"Show excluded quarterly periods for {label}"):
                    st.dataframe(pd.concat(invalid_list, ignore_index=True), use_container_width=True)

            if chart_type == "Line":
                fig = px.line(
                    df_avg,
                    x="quarter",
                    y="value",
                    color="pollutant",
                    line_group="pollutant",
                    markers=True,
                    title=f"Quarterly Means - {label}",
                    labels={"value": "µg/m³", "quarter": "Quarter"}
                )
            else:
                fig = px.bar(
                    df_avg,
                    x="quarter",
                    y="value",
                    color="site",
                    barmode="group",
                    facet_col="pollutant" if len(selected_display_pollutants) > 1 else None,
                    title=f"Quarterly Means - {label}",
                    labels={"value": "µg/m³", "quarter": "Quarter"}
                )

            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Show Quarterly Aggregated Data Table"):
                st.dataframe(df_avg, use_container_width=True)

            st.download_button(
                f"⬇️ Download Quarterly Avg - {label}",
                to_csv_download(df_avg),
                file_name=f"QuarterlyAvg_{label}.csv",
                key=f"download_quarterly_{label}"
            )


def render_dayofweek_means_tab(tab, dfs, selected_years):
    with tab:
        st.header("📊 Days of Week Means")

        for label, df in dfs.items():
            st.subheader(f"Dataset: {label}")

            site_in_tab = st.multiselect(
                f"Select Site(s) for {label}",
                sorted(df["site"].unique()),
                key=unique_key("tab6", "site", label)
            )

            filtered_df = df.copy()
            years_to_use = selected_years if selected_years else sorted(df["year"].unique())

            if years_to_use:
                filtered_df = filtered_df[filtered_df["year"].isin(years_to_use)]
            if site_in_tab:
                filtered_df = filtered_df[filtered_df["site"].isin(site_in_tab)]

            selected_quarters = st.multiselect(
                f"Select Quarter(s) for {label}",
                options=["Q1", "Q2", "Q3", "Q4"],
                default=["Q1", "Q2", "Q3", "Q4"],
                key=unique_key("tab6", "quarter", label)
            ) or []

            selected_quarter_nums = [f"{year}{q}" for year in years_to_use for q in selected_quarters]

            if selected_quarter_nums:
                filtered_df = filtered_df[filtered_df["quarter"].isin(selected_quarter_nums)]
            else:
                st.warning("No valid quarters to filter.")
                continue

            if filtered_df.empty:
                st.warning(f"No data remaining for {label} after filtering.")
                continue

            valid_pollutants = [p for p in ["pm25", "pm10"] if p in filtered_df.columns]

            selected_display_pollutants = st.multiselect(
                f"Select Pollutants to Display for {label}",
                options=["All"] + valid_pollutants,
                default=["All"],
                key=unique_key("tab6", "pollutants", label)
            )

            if "All" in selected_display_pollutants:
                selected_display_pollutants = valid_pollutants

            chart_type = st.radio(
                f"Chart Type for {label}",
                ["Line", "Bar"],
                horizontal=True,
                key=unique_key("tab6", "charttype", label)
            )

            df_avg_list = []
            for pollutant in selected_display_pollutants:
                avg_df = calculate_dayofweek_pollutant(filtered_df, pollutant)
                avg_df["pollutant"] = pollutant
                avg_df = avg_df.rename(columns={pollutant: "value"})
                df_avg_list.append(avg_df)

            if not df_avg_list:
                st.warning(f"No data available for selected pollutants in {label}")
                continue

            df_avg = pd.concat(df_avg_list, ignore_index=True)

            if chart_type == "Line":
                fig = px.line(
                    df_avg,
                    x="dayofweek",
                    y="value",
                    color="pollutant",
                    line_group="pollutant",
                    markers=True,
                    title=f"Day of Week Means - {label}",
                    labels={"value": "µg/m³", "dayofweek": "Day of Week"}
                )
            else:
                fig = px.bar(
                    df_avg,
                    x="dayofweek",
                    y="value",
                    color="site",
                    barmode="group",
                    facet_col="pollutant" if len(selected_display_pollutants) > 1 else None,
                    title=f"Day of Week Means - {label}",
                    labels={"value": "µg/m³", "dayofweek": "Day of Week"}
                )

            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Show Aggregated Data Table"):
                st.dataframe(df_avg, use_container_width=True)


# ----------------------------
# Main app
# ----------------------------

uploaded_files = st.file_uploader(
    "Upload up to 4 datasets",
    type=["csv", "xlsx"],
    accept_multiple_files=True
)

if uploaded_files:
    site_options = set()
    year_options = set()
    dfs = {}

    for file in uploaded_files:
        label = file.name.rsplit(".", 1)[0]
        ext = file.name.rsplit(".", 1)[-1].lower()

        try:
            df = pd.read_excel(file) if ext == "xlsx" else pd.read_csv(file)

            df = parse_dates(df)
            df = standardize_columns(df)
            df = cleaned(df)

            if not all(col in df.columns for col in ["date", "pm25", "pm10", "site"]):
                st.warning(f"⚠️ Could not process {label}: missing required columns.")
                continue

            dfs[label] = df
            site_options.update(df["site"].unique())
            year_options.update(df["year"].unique())

        except Exception as e:
            st.error(f"Error processing {label}: {e}")

    if dfs:
        with st.sidebar:
            selected_years = st.multiselect("📅 Filter by Year", sorted(year_options))
            selected_sites = st.multiselect("🏢 Filter by Site", sorted(site_options))

        # apply sidebar site filter globally if chosen
        if selected_sites:
            for label in list(dfs.keys()):
                dfs[label] = dfs[label][dfs[label]["site"].isin(selected_sites)].copy()

        tabs = st.tabs([
            "Aggregated Means",
            "Quarterly Means",
            "Monthly Means",
            "Exceedances & Max/Min Values",
            "AQI Stats",
            "Daily Means",
            "Day of Week Means"
        ])

        render_quarter_means_tab(tabs[1], dfs, selected_years)
        render_monthly_means_tab(tabs[2], dfs, selected_years)
        render_exceedances_tab(tabs[3], dfs, selected_years)
        render_aqi_tab(tabs[4], selected_years, dfs)
        render_daily_means_tab(tabs[5], dfs, selected_years)
        render_dayofweek_means_tab(tabs[6], dfs, selected_years)

        with tabs[0]:
            st.header("📊 Aggregated Means")

            for label, df in dfs.items():
                st.subheader(f"Dataset: {label}")

                site_in_tab = st.multiselect(
                    f"Select Site(s) for {label}",
                    sorted(df["site"].unique()),
                    key=f"site_agg_{label}"
                )

                filtered_df = df.copy()
                if selected_years:
                    filtered_df = filtered_df[filtered_df["year"].isin(selected_years)]
                if site_in_tab:
                    filtered_df = filtered_df[filtered_df["site"].isin(site_in_tab)]

                valid_pollutants = [p for p in ["pm25", "pm10"] if p in filtered_df.columns]
                if not valid_pollutants:
                    st.warning(f"No valid pollutants found in {label}")
                    continue

                selected_display_pollutants = st.multiselect(
                    f"Select Pollutants to Display for {label}",
                    options=["All"] + valid_pollutants,
                    default=["All"],
                    key=f"pollutants_{label}"
                )

                if "All" in selected_display_pollutants:
                    selected_display_pollutants = valid_pollutants

                aggregate_levels = [
                    ("Daily Avg", ["day", "site"], None),
                    ("Monthly Avg", ["month", "site"], "month"),
                    ("Quarterly Avg", ["quarter", "site"], "quarter"),
                    ("Yearly Avg", ["year", "site"], "year"),
                    ("Day of Week Avg", ["dayofweek", "site"], None),
                    ("Weekday Type Avg", ["weekday_type", "site"], None),
                    ("Season Avg", ["season", "site"], None),
                ]

                for level_name, group_keys, period_type in aggregate_levels:
                    agg_label = f"{label} - {level_name}"
                    agg_dfs = []

                    excluded_frames = []

                    for pollutant in selected_display_pollutants:
                        if period_type == "month":
                            valid_df, invalid_df = aggregate_with_weekly_completeness(filtered_df, pollutant, "month", 0.75)
                            temp = valid_df[group_keys + ["value"]].rename(columns={"value": pollutant})
                            agg_dfs.append(temp)
                            if not invalid_df.empty:
                                invalid_df["pollutant"] = pollutant
                                excluded_frames.append(invalid_df)

                        elif period_type == "quarter":
                            valid_df, invalid_df = aggregate_with_weekly_completeness(filtered_df, pollutant, "quarter", 0.75)
                            temp = valid_df[group_keys + ["value"]].rename(columns={"value": pollutant})
                            agg_dfs.append(temp)
                            if not invalid_df.empty:
                                invalid_df["pollutant"] = pollutant
                                excluded_frames.append(invalid_df)

                        elif period_type == "year":
                            valid_df, invalid_df = aggregate_with_weekly_completeness(filtered_df, pollutant, "year", 0.75)
                            temp = valid_df[group_keys + ["value"]].rename(columns={"value": pollutant})
                            agg_dfs.append(temp)
                            if not invalid_df.empty:
                                invalid_df["pollutant"] = pollutant
                                excluded_frames.append(invalid_df)

                        else:
                            temp = (
                                filtered_df.groupby(group_keys)[pollutant]
                                .mean()
                                .reset_index()
                                .round(1)
                            )
                            agg_dfs.append(temp)

                    if not agg_dfs:
                        continue

                    merged_df = reduce(
                        lambda left, right: pd.merge(left, right, on=group_keys, how="outer"),
                        agg_dfs
                    )

                    display_cols = group_keys + [p for p in selected_display_pollutants if p in merged_df.columns]
                    editable_df = merged_df[display_cols]

                    st.data_editor(
                        editable_df,
                        use_container_width=True,
                        disabled=True,
                        key=f"editor_{label}_{agg_label}"
                    )

                    st.download_button(
                        label=f"📥 Download {agg_label}",
                        data=to_csv_download(editable_df),
                        file_name=f"{label}_{agg_label.replace(' ', '_')}.csv",
                        mime="text/csv",
                        key=f"download_{label}_{agg_label}"
                    )

                    if excluded_frames:
                        with st.expander(f"Show excluded periods for {agg_label}"):
                            st.dataframe(pd.concat(excluded_frames, ignore_index=True), use_container_width=True)

                    st.markdown("---")

                    with st.expander(f"📈 Show Charts for {agg_label}", expanded=("Yearly Avg" in agg_label)):
                        x_axis = next((col for col in editable_df.columns if col not in ["site"] + valid_pollutants), None)

                        if not x_axis:
                            st.warning(f"Could not determine x-axis for {agg_label}")
                            continue

                        safe_pollutants = [p for p in selected_display_pollutants if p in editable_df.columns]
                        if not safe_pollutants:
                            st.warning(f"No valid pollutant columns to plot for {agg_label}")
                            continue

                        try:
                            df_melted = editable_df.melt(
                                id_vars=["site", x_axis],
                                value_vars=safe_pollutants,
                                var_name="pollutant",
                                value_name="value"
                            )

                            chart_type_choice = st.selectbox(
                                f"Select Chart Type for {agg_label}",
                                options=["line", "bar"],
                                index=0 if "Yearly" in agg_label else 1,
                                key=f"chart_type_{label}_{agg_label}"
                            )

                            if chart_type_choice == "line":
                                fig = px.line(
                                    df_melted,
                                    x=x_axis,
                                    y="value",
                                    color="pollutant",
                                    line_group="site",
                                    markers=True,
                                    title=f"{agg_label} - {', '.join(safe_pollutants)}",
                                    hover_data=["site", "pollutant", "value"]
                                )
                            else:
                                fig = px.bar(
                                    df_melted,
                                    x=x_axis,
                                    y="value",
                                    color="pollutant",
                                    barmode="group",
                                    title=f"{agg_label} - {', '.join(safe_pollutants)}",
                                    hover_data=["site", "pollutant", "value"]
                                )

                            st.plotly_chart(fig, use_container_width=True)

                        except Exception as e:
                            st.error(f"Error plotting chart: {e}")
    else:
        st.warning("No valid datasets were processed.")
else:
    st.info("Upload CSV or Excel files from different air quality sources to begin.")
