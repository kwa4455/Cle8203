import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# --- Page Configuration ---
st.set_page_config(page_title="Reference Grade Monitor",page_icon="🛠️", layout="wide")
# Inject Dynamic Theme

st.markdown("""
<style>
/* GLOBAL */
* {
    transition: all 0.2s ease-in-out;
    font-family: 'Segoe UI', sans-serif;
}


/* App background */
html, .stApp {
    background: url('https://i.postimg.cc/tCMmM50P/temp-Imageqld-VCj.avif');
    background-size: cover;
    background-position: center;
    min-height: 100vh;
    backdrop-filter: blur(20px);
}

/* Main content container glass style */
.logged_in {
    background: rgba(255, 255, 255, 0.2);
    border-radius: 16px;
    padding: 20px;
    margin: 20px auto;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.3);
    width: 80%;
}

/* Text Input Fields */
div[data-testid="stTextInput"] input {
    background: rgba(255, 255, 255, 0.25) !important;
    border: 1px solid rgba(255, 255, 255, 0.4) !important;
    border-radius: 10px !important;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    color: #000 !important;
}

/* Text Area Fields */
div[data-testid="stTextArea"] textarea {
    background: rgba(255, 255, 255, 0.25) !important;
    border: 1px solid rgba(255, 255, 255, 0.4) !important;
    border-radius: 10px !important;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    color: #000 !important;
}
/* Info alert (Logged in as...) */
div[data-testid="stAlert-info"] {
    background: rgba(255, 255, 255, 0.25) !important;
    color: #000 !important;
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    padding: 16px !important;
    font-weight: 500;
}

/* Optional: info icon color */
div[data-testid="stAlert-info"] svg {
    color: #0099cc !important;
}
/* Buttons */
button[kind="primary"] {
    background: rgba(255, 255, 255, 0.25) !important;
    color: #000 !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
}

button[kind="primary"]:hover {
    background: rgba(255, 255, 255, 0.35) !important;
    box-shadow: 0 6px 14px rgba(0, 0, 0, 0.25);
}
div[data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.18);
    border-radius: 16px;
    padding: 24px;
    margin-top: 10px;
    margin-bottom: 30px;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
    border: 1px solid rgba(255, 255, 255, 0.25);
}
/* Selectbox */
div[data-testid="stSelectbox"] {
    background: rgba(255, 255, 255, 0.25) !important;
    border-radius: 12px;
    padding: 6px 12px;
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.3);
    color: #000 !important;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}

/* Slider */
div[data-testid="stSlider"] {
    background: rgba(255, 255, 255, 0.2);
    border-radius: 12px;
    padding: 10px;
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    box-shadow: 0 3px 8px rgba(0, 0, 0, 0.2);
}

/* Slider handle and track text */
div[data-testid="stSlider"] .stSlider > div {
    color: #000 !important;
}

/* Dataframe / Table Styling */
.css-1d391kg, .css-1r6slb0, .stDataFrame, .stTable {
    background: rgba(255, 255, 255, 0.2) !important;
    border-radius: 10px !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    color: black !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
}
hr {
    border: none;
    border-top: 1px solid rgba(255, 255, 255, 0.3);
}

.footer-text {
    color: rgba(255, 255, 255, 0.7);
}
/* Table Cells */
thead, tbody, tr, th, td {
    background: rgba(255, 255, 255, 0.15) !important;
    color: #000 !important;
    backdrop-filter: blur(4px);
}

/* Sidebar Styling */
section[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.15);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-right: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 2px 0 10px rgba(0, 0, 0, 0.1);
}
</style>
""", unsafe_allow_html=True)


# --- Title and Logo ---
st.title("📊 Reference Grade Monitor Data Analysis")
# --- Helper Functions ---

@st.cache_data(ttl=600)

def cleaned(df):
    df = df.rename(columns=lambda x: x.strip().lower())
    required_columns = ['datetime', 'site', 'pm25', 'pm10']
    df = df[[col for col in required_columns if col in df.columns]]
    df = df.dropna(axis=1, how='all').dropna()
    df = df[df.groupby('pm25')['pm25'].transform('count') <= 2]

    def removal_box_plot(df, col, lower_threshold, upper_threshold):
        filtered = df[(df[col] >= lower_threshold) & (df[col] <= upper_threshold)]
        return filtered, (df[col] < lower_threshold).sum(), (df[col] > upper_threshold).sum()

    if 'pm25' in df.columns:
        df, _, _ = removal_box_plot(df, 'pm25', 1, 500)

    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.to_period('M').astype(str)
    df['quarter'] = df['datetime'].dt.to_period('Q').astype(str)
    df['day'] = df['datetime'].dt.date
    df['dayofweek'] = df['datetime'].dt.day_name()
    df['weekday_type'] = df['datetime'].dt.weekday.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
    df['season'] = df['datetime'].dt.month.apply(lambda x: 'Harmattan' if x in [12, 1, 2] else 'Non-Harmattan')

    daily_counts = df.groupby(['site', 'month'])['day'].nunique().reset_index(name='daily_counts')
    sufficient_sites = daily_counts[daily_counts['daily_counts'] >= 15][['site', 'month']]
    df = df.merge(sufficient_sites, on=['site', 'month'])
    return df

def parse_dates(df):
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                df['datetime'] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                df = df.dropna(subset=['datetime'])
                return df
            except:
                continue
    return df

def standardize_columns(df):
    pm25_cols = ['pm25', 'PM25', 'pm_2_5', 'pm25_avg', 'pm2.5']
    pm10_cols = ['pm10', 'PM10', 'pm_10']
    site_cols = ['site', 'station', 'location']

    for col in df.columns:
        col_lower = col.strip().lower()
        if col_lower in [c.lower() for c in pm25_cols]:
            df.rename(columns={col: 'pm25'}, inplace=True)
        if col_lower in [c.lower() for c in pm10_cols]:
            df.rename(columns={col: 'pm10'}, inplace=True)
        if col_lower in [c.lower() for c in site_cols]:
            df.rename(columns={col: 'site'}, inplace=True)
    return df

def compute_aggregates(df, label, pollutant,unique_key):
    aggregates = {}
    aggregates[f'{label} - Daily Avg ({pollutant})'] = df.groupby(['day', 'site'])[pollutant].mean().reset_index().round(1)
    aggregates[f'{label} - Monthly Avg ({pollutant})'] = df.groupby(['month', 'site'])[pollutant].mean().reset_index().round(1)
    aggregates[f'{label} - Quarterly Avg ({pollutant})'] = df.groupby(['quarter', 'site'])[pollutant].mean().reset_index().round(1)
    aggregates[f'{label} - Yearly Avg ({pollutant})'] = df.groupby(['year', 'site'])[pollutant].mean().reset_index().round(1)
    aggregates[f'{label} - Day of Week Avg ({pollutant})'] = df.groupby(['dayofweek', 'site'])[pollutant].mean().reset_index().round(1)
    aggregates[f'{label} - Weekday Type Avg ({pollutant})'] = df.groupby(['weekday_type', 'site'])[pollutant].mean().reset_index().round(1)
    aggregates[f'{label} - Season Avg ({pollutant})'] = df.groupby(['season', 'site'])[pollutant].mean().reset_index().round(1)
    return aggregates

def calculate_exceedances(df):
    daily_avg = df.groupby(['site', 'day', 'year','quarter', 'month'], as_index=False).agg({
        'pm25': 'mean',
        'pm10': 'mean'
    })
    pm25_exceed = daily_avg[daily_avg['pm25'] > 35].groupby(['year', 'site']).size().reset_index(name='pm25_Exceedance_Count')
    pm10_exceed = daily_avg[daily_avg['pm10'] > 70].groupby(['year', 'site']).size().reset_index(name='pm10_Exceedance_Count')
    total_days = daily_avg.groupby(['year', 'site']).size().reset_index(name='Total_Records')

    exceedance = total_days.merge(pm25_exceed, on=['year', 'site'], how='left') \
                           .merge(pm10_exceed, on=['year', 'site'], how='left')
    exceedance.fillna(0, inplace=True)
    exceedance['pm25_Exceedance_Percent'] = round((exceedance['pm25_Exceedance_Count'] / exceedance['Total_Records']) * 100, 1)
    exceedance['pm10_Exceedance_Percent'] = round((exceedance['pm10_Exceedance_Count'] / exceedance['Total_Records']) * 100, 1)

    return exceedance

def render_exceedances_tab(tab, dfs, selected_years, calculate_exceedances, calculate_min_max):
    with tab:
        st.header("🌫️ Exceedances & Max/Min")

        for label, df in dfs.items():
            st.subheader(f"Dataset: {label}")

            # --- Select Sites ---
            site_in_tab = st.multiselect(
                f"Select Site(s) for {label}",
                options=sorted(df['site'].unique()),
                key=f"site_exc_{label}"
            )

            # --- Select Years ---
            available_years = sorted(df['year'].unique())
            selected_years_in_tab = st.multiselect(
                f"Select Year(s) for {label}",
                options=available_years,
                default=selected_years if selected_years else available_years,
                key=f"year_exc_{label}"
            )

            # --- Filter by Site and Year ---
            filtered_df = df.copy()
            if selected_years_in_tab:
                filtered_df = filtered_df[filtered_df['year'].isin(selected_years_in_tab)]
            if site_in_tab:
                filtered_df = filtered_df[filtered_df['site'].isin(site_in_tab)]

            # --- Select Quarters ---
            selected_quarters = st.multiselect(
                f"Select Quarter(s) for {label}",
                options=['Q1', 'Q2', 'Q3', 'Q4'],
                default=['Q1', 'Q2', 'Q3', 'Q4'],
                key=f"quarter_exc_{label}"
            )

            # --- Filter by Quarter or Year (fallback) ---
            if selected_years_in_tab:
                if selected_quarters:
                    selected_quarter_nums = [f"{year}{q}" for year in selected_years_in_tab for q in selected_quarters]
                    filtered_df = filtered_df[filtered_df['quarter'].isin(selected_quarter_nums)]
                else:
                    # Fallback: filter only by year
                    filtered_df = filtered_df[filtered_df['quarter'].str[:4].isin(selected_years_in_tab)]
            else:
                st.warning("No valid years selected!")
                continue

            # --- Handle Empty DataFrame ---
            if filtered_df.empty:
                st.warning(f"No data remaining for {label} after filtering.")
                continue

            # --- Calculate & Display Exceedances ---
            exceedances = calculate_exceedances(filtered_df)
            st.dataframe(exceedances, use_container_width=True)
            st.download_button(
                label=f"⬇️ Download Exceedances - {label}",
                data=to_csv_download(exceedances),
                file_name=f"Exceedances_{label}.csv",
                key=f"download_exceedances_{label}"
            )

            # --- Calculate & Display Min/Max ---
            min_max = calculate_min_max(filtered_df)
            st.dataframe(min_max, use_container_width=True)
            st.download_button(
                label=f"⬇️ Download MinMax - {label}",
                data=to_csv_download(min_max),
                file_name=f"MinMax_{label}.csv",
                key=f"download_minmax_{label}"
            )



def calculate_min_max(df):
    daily_avg = df.groupby(['site', 'day', 'year','quarter', 'month'], as_index=False).agg({
        'pm25': 'mean',
        'pm10': 'mean'
    })
    df_min_max = daily_avg.groupby(['site','year', 'quarter', 'month'], as_index=False).agg(
        daily_avg_pm10_max=('pm10', lambda x: round(x.max(), 1)),
        daily_avg_pm10_min=('pm10', lambda x: round(x.min(), 1)),
        daily_avg_pm25_max=('pm25', lambda x: round(x.max(), 1)),
        daily_avg_pm25_min=('pm25', lambda x: round(x.min(), 1))
    )
    return df_min_max


def calculate_aqi_and_category(df):
    # Group by day, year, quarter, and month across all selected sites
    daily_avg = df.groupby(['day', 'year', 'quarter', 'month'], as_index=False).agg({
        'pm25': lambda x: round(x.mean(), 1)
    })

    # Define AQI breakpoints
    breakpoints = [
        (0.0, 9.0, 0, 50),
        (9.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200),
        (125.5, 225.4, 201, 300),
        (225.5, 325.4, 301, 500),
        (325.5, 99999.9, 501, 999)
    ]

    # AQI calculation function
    def calculate_aqi(pm):
        for low, high, aqi_low, aqi_high in breakpoints:
            if low <= pm <= high:
                return round(((pm - low) * (aqi_high - aqi_low) / (high - low)) + aqi_low)
        return np.nan

    # Apply AQI calculation
    daily_avg['AQI'] = daily_avg['pm25'].apply(calculate_aqi)

    # Define AQI categories
    conditions = [
        (daily_avg['AQI'] > 300),
        (daily_avg['AQI'] > 200),
        (daily_avg['AQI'] > 150),
        (daily_avg['AQI'] > 100),
        (daily_avg['AQI'] > 50),
        (daily_avg['AQI'] >= 0)
    ]
    remarks = ['Hazardous', 'Very Unhealthy', 'Unhealthy', 'Unhealthy for Sensitive Groups', 'Moderate', 'Good']
    daily_avg['AQI_Remark'] = np.select(conditions, remarks, default='Unknown')

    # Count AQI categories per year (not per site)
    remarks_counts = daily_avg.groupby(['year', 'AQI_Remark']).size().reset_index(name='Count')

    # Calculate total counts per year and percentage
    remarks_counts['Total_Count_Per_Year'] = remarks_counts.groupby('year')['Count'].transform('sum')
    remarks_counts['Percent'] = round((remarks_counts['Count'] / remarks_counts['Total_Count_Per_Year']) * 100, 1)

    return daily_avg, remarks_counts

    


def render_aqi_tab(tab, selected_years, calculate_aqi_and_category, unique_key, dfs, to_csv_download):
    with tab:
        st.header("🌫️ AQI Stats")

        for label, df in dfs.items():
            st.subheader(f"Dataset: {label}")

            all_years = sorted(df['year'].dropna().unique())
            default_years = all_years[-2:] if len(all_years) >= 2 else all_years
            selected_years_in_tab = st.multiselect(
                f"Select Year(s) for {label}",
                options=all_years,
                default=default_years,
                key=f"years_aqi_{label}"
            )

            site_in_tab = st.multiselect(
                f"Select Site(s) for {label}",
                sorted(df['site'].unique()),
                key=f"site_aqi_{label}"
            )

            filtered_df = df.copy()

            if selected_years_in_tab:
                filtered_df = filtered_df[filtered_df['year'].isin(selected_years_in_tab)]

            if site_in_tab:
                filtered_df = filtered_df[filtered_df['site'].isin(site_in_tab)]

            selected_quarters = st.multiselect(
                f"Select Quarter(s) for {label}",
                options=['Q1', 'Q2', 'Q3', 'Q4'],
                default=['Q1', 'Q2', 'Q3', 'Q4'],
                key=unique_key("tab4", "quarter", label)
            ) or []

            if not selected_years_in_tab:
                selected_years_in_tab = sorted(df['year'].unique())

            selected_quarter_nums = [f"{year}{q}" for year in selected_years_in_tab for q in selected_quarters]

            if selected_quarter_nums:
                filtered_df = filtered_df[filtered_df['quarter'].isin(selected_quarter_nums)]
            else:
                st.warning("No valid quarters to filter!")
                continue

            if filtered_df.empty:
                st.warning(f"No data remaining for {label} after filtering.")
                continue

            # ✅ Aggregate AQI across selected sites
            daily_avg, remarks_counts = calculate_aqi_and_category(filtered_df)

            st.markdown('<div class="glass-container">', unsafe_allow_html=True)
            st.dataframe(remarks_counts, use_container_width=True)
            st.dataframe(daily_avg, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.download_button(f"⬇️ Download Daily Avg - {label}", to_csv_download(daily_avg), file_name=f"DailyAvg_{label}.csv")
            st.download_button(f"⬇️ Download AQI - {label}", to_csv_download(remarks_counts), file_name=f"AQI_{label}.csv")

            aqi_colors = {
                'Good': '#00e400',
                'Moderate': '#ffff00',
                'Unhealthy for Sensitive Groups': '#ff7e00',
                'Unhealthy': '#ff0000',
                'Very Unhealthy': '#8f3f97',
                'Hazardous': '#7e0023'
            }

            remarks_counts['Color'] = remarks_counts['AQI_Remark'].map(aqi_colors)

            if len(selected_years_in_tab) >= 2:
                current_year = max(selected_years_in_tab)
                previous_year = current_year - 1

                current_df = remarks_counts[remarks_counts["year"] == current_year]
                prev_df = remarks_counts[remarks_counts["year"] == previous_year]

                col1, col2 = st.columns(2)
                with st.container():
                    with col1:
                        st.markdown(f"**{current_year} AQI (All Selected Sites)**")
                        fig_current = px.bar(
                            current_df,
                            x="Percent",
                            y="AQI_Remark",
                            color="AQI_Remark",
                            orientation="h",
                            color_discrete_map=aqi_colors,
                            hover_data=["Percent"],
                        )
                        fig_current.update_layout(
                            xaxis_title="% Time in AQI Category",
                            yaxis=dict(categoryorder="total ascending"),
                            showlegend=False,
                            height=400
                        )
                        st.plotly_chart(fig_current, use_container_width=True)

                    with col2:
                        st.markdown(f"**{previous_year} AQI (All Selected Sites)**")
                        fig_prev = px.bar(
                            prev_df,
                            x="Percent",
                            y="AQI_Remark",
                            color="AQI_Remark",
                            orientation="h",
                            color_discrete_map=aqi_colors,
                            hover_data=["Percent"],
                        )
                        fig_prev.update_layout(
                            xaxis_title="% Time in AQI Category",
                            yaxis=dict(categoryorder="total ascending"),
                            showlegend=False,
                            height=400
                        )
                        st.plotly_chart(fig_prev, use_container_width=True)
            else:
                st.warning("Please select at least two years to compare AQI.")




def to_csv_download(df):
    return BytesIO(df.to_csv(index=False).encode('utf-8'))

def calculate_day_pollutant(df, pollutant):
    df_grouped = df.groupby(['site', 'day', 'month', 'quarter', 'year'])[pollutant].mean().reset_index().round(1)
    return df_grouped


def render_daily_means_tab(tab, dfs, selected_years, calculate_day_pollutant, unique_key):
    with tab:
        st.header("📊 Daily Means")

        for label, df in dfs.items():
            st.subheader(f"Dataset: {label}")
            
            site_in_tab = st.multiselect(
                f"Select Site(s) for {label}",
                sorted(df['site'].unique()),
                key=unique_key("tab5", "site", label)
            )

            filtered_df = df.copy()

            # Filter by year
            if selected_years:
                filtered_df = filtered_df[filtered_df['year'].isin(selected_years)]
                
            # Filter by site
            if site_in_tab:
                filtered_df = filtered_df[filtered_df['site'].isin(site_in_tab)]
                
            # Filter by quarter
            selected_quarters = st.multiselect(
                f"Select Quarter(s) for {label}",
                options=['Q1', 'Q2', 'Q3', 'Q4'],
                default=['Q1', 'Q2', 'Q3', 'Q4'],
                key=unique_key("tab5", "quarter", label)
            ) or []

            # Ensure selected_years is defined and not empty
            if not selected_years:
                selected_years = sorted(df['year'].unique())  # or handle this upstream

            # Correct mapping to full quarter strings like '2021Q1'
            selected_quarter_nums = [f"{year}{q}" for year in selected_years for q in selected_quarters]

            
            if selected_quarter_nums:
                filtered_df = filtered_df[filtered_df['quarter'].isin(selected_quarter_nums)]
                
            else:
                st.warning("No valid quarters to filter!")
                continue

            # Check if no data remains after filtering
            if filtered_df.empty:
                st.warning(f"No data remaining for {label} after filtering.")
                continue

            # Check available pollutants in the dataset
            selected_pollutants = ['pm25', 'pm10']
            valid_pollutants = [p for p in selected_pollutants if p in filtered_df.columns]

            if not valid_pollutants:
                st.warning(f"No valid pollutants found in {label}")
                continue

            selected_display_pollutants = st.multiselect(
                f"Select Pollutants to Display for {label}",
                options=["All"] + valid_pollutants,
                default=["All"],
                key=unique_key("tab5", "pollutants", label)
            )

            if "All" in selected_display_pollutants:
                selected_display_pollutants = valid_pollutants

            # Choose chart type
            chart_type = st.radio(
                f"Chart Type for {label}",
                ["Line", "Bar"],
                horizontal=True,
                key=unique_key("tab5", "charttype", label)
            )

            df_avg_list = []
            for pollutant in selected_display_pollutants:
                if pollutant in filtered_df.columns:
                    avg_df = calculate_day_pollutant(filtered_df, pollutant)
                    avg_df["pollutant"] = pollutant
                    avg_df.rename(columns={pollutant: "value"}, inplace=True)
                    df_avg_list.append(avg_df)

            # Check if no data is available for the selected pollutants
            if not df_avg_list:
                st.warning(f"No data available for selected pollutants in {label}")
                continue

            # Combine the dataframes into one
            df_avg = pd.concat(df_avg_list, ignore_index=True)

            # Default to "day" as the x-axis, if the column exists
            x_axis = "day" if "day" in filtered_df.columns else "month"
            y_title = "µg/m³"
            plot_title = f"Aggregated {chart_type} Chart - {label}"

            # Generate the chart based on the selected chart type
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

                # Add the Ghana standard threshold lines
                for pollutant in selected_display_pollutants:
                    threshold = 35 if pollutant.lower() == "pm25" else 70
                    fig.add_trace(
                        go.Scatter(
                            x=sorted(df_avg[x_axis].unique()),
                            y=[threshold] * len(df_avg[x_axis].unique()),
                            mode="lines",
                            line=dict(dash="solid", color="red"),
                            name=f"Ghana Standard ({threshold} µg/m³)",
                            showlegend=True
                        )
                    )

                fig.update_layout(
                    yaxis_title=y_title,
                    height=500,
                    legend_title="Site",
                    margin=dict(t=40, b=40)
                )

            # Display the chart
            st.markdown('<div class="glass-container">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)

            # Optionally, display the aggregated data table
            with st.expander("Show Aggregated Data Table"):
                st.dataframe(df_avg)

            st.markdown('</div>', unsafe_allow_html=True)



def calculate_month_pollutant(df, pollutant):
    df_grouped = df.groupby(['site', 'month', 'quarter', 'year'])[pollutant].mean().reset_index().round(1)
   
    return df_grouped
def render_monthly_means_tab(tab, dfs, selected_years, calculate_month_pollutant, unique_key):
    with tab:
        st.header("📊 Monthly Means")

        for label, df in dfs.items():
            st.subheader(f"Dataset: {label}")
            
            site_in_tab = st.multiselect(
                f"Select Site(s) for {label}",
                sorted(df['site'].unique()),
                key=unique_key("tab2", "site", label)
            )

            filtered_df = df.copy()

            # Filter by year
            if selected_years:
                filtered_df = filtered_df[filtered_df['year'].isin(selected_years)]
                
            # Filter by site
            if site_in_tab:
                filtered_df = filtered_df[filtered_df['site'].isin(site_in_tab)]
                
            # Filter by quarter
            selected_quarters = st.multiselect(
                f"Select Quarter(s) for {label}",
                options=['Q1', 'Q2', 'Q3', 'Q4'],
                default=['Q1', 'Q2', 'Q3', 'Q4'],
                key=unique_key("tab2", "quarter", label)
            ) or []

            # Ensure selected_years is defined and not empty
            if not selected_years:
                selected_years = sorted(df['year'].unique())  # or handle this upstream

            # Correct mapping to full quarter strings like '2021Q1'
            selected_quarter_nums = [f"{year}{q}" for year in selected_years for q in selected_quarters]

            
            if selected_quarter_nums:
                filtered_df = filtered_df[filtered_df['quarter'].isin(selected_quarter_nums)]
                
            else:
                st.warning("No valid quarters to filter!")
                continue

            # Check if no data remains after filtering
            if filtered_df.empty:
                st.warning(f"No data remaining for {label} after filtering.")
                continue

            # Check available pollutants in the dataset
            selected_pollutants = ['pm25', 'pm10']
            valid_pollutants = [p for p in selected_pollutants if p in filtered_df.columns]

            if not valid_pollutants:
                st.warning(f"No valid pollutants found in {label}")
                continue

            selected_display_pollutants = st.multiselect(
                f"Select Pollutants to Display for {label}",
                options=["All"] + valid_pollutants,
                default=["All"],
                key=unique_key("tab2", "pollutants", label)
            )

            if "All" in selected_display_pollutants:
                selected_display_pollutants = valid_pollutants

            # Choose chart type
            chart_type = st.radio(
                f"Chart Type for {label}",
                ["Line", "Bar"],
                horizontal=True,
                key=unique_key("tab2", "charttype", label)
            )

            df_avg_list = []
            for pollutant in selected_display_pollutants:
                if pollutant in filtered_df.columns:
                    avg_df = calculate_month_pollutant(filtered_df, pollutant)
                    avg_df["pollutant"] = pollutant
                    avg_df.rename(columns={pollutant: "value"}, inplace=True)
                    df_avg_list.append(avg_df)

            # Check if no data is available for the selected pollutants
            if not df_avg_list:
                st.warning(f"No data available for selected pollutants in {label}")
                continue

            # Combine the dataframes into one
            df_avg = pd.concat(df_avg_list, ignore_index=True)

            # Default to "day" as the x-axis, if the column exists
            x_axis = "month" if "month" in filtered_df.columns else "quarter"
            y_title = "µg/m³"
            plot_title = f"Aggregated {chart_type} Chart - {label}"

            # Generate the chart based on the selected chart type
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


                fig.update_layout(
                    yaxis_title=y_title,
                    height=500,
                    legend_title="Site",
                    margin=dict(t=40, b=40)
                )

            # Display the chart
            st.markdown('<div class="glass-container">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)

            # Optionally, display the aggregated data table
            with st.expander("Show Aggregated Data Table"):
                st.dataframe(df_avg)

            st.markdown('</div>', unsafe_allow_html=True)


def calculate_quarter_pollutant(df, pollutant):
    df_grouped = df.groupby(['site', 'quarter', 'year'])[pollutant].mean().reset_index().round(1)
   
    return df_grouped
    
def render_quarter_means_tab(tab, dfs, selected_years, calculate_quarter_pollutant, unique_key):
    with tab:
        st.header("📊 Quarterly Means")

        for label, df in dfs.items():
            st.subheader(f"Dataset: {label}")
            
            site_in_tab = st.multiselect(
                f"Select Site(s) for {label}",
                sorted(df['site'].unique()),
                key=unique_key("tab1", "site", label)
            )

            filtered_df = df.copy()

            # Filter by year
            if selected_years:
                filtered_df = filtered_df[filtered_df['year'].isin(selected_years)]
                
            # Filter by site
            if site_in_tab:
                filtered_df = filtered_df[filtered_df['site'].isin(site_in_tab)]
                
            # Filter by quarter
            selected_quarters = st.multiselect(
                f"Select Quarter(s) for {label}",
                options=['Q1', 'Q2', 'Q3', 'Q4'],
                default=['Q1', 'Q2', 'Q3', 'Q4'],
                key=unique_key("tab1", "quarter", label)
            ) or []

            # Ensure selected_years is defined and not empty
            if not selected_years:
                selected_years = sorted(df['year'].unique())  # or handle this upstream

            # Correct mapping to full quarter strings like '2021Q1'
            selected_quarter_nums = [f"{year}{q}" for year in selected_years for q in selected_quarters]

            
            if selected_quarter_nums:
                filtered_df = filtered_df[filtered_df['quarter'].isin(selected_quarter_nums)]
                
            else:
                st.warning("No valid quarters to filter!")
                continue

            # Check if no data remains after filtering
            if filtered_df.empty:
                st.warning(f"No data remaining for {label} after filtering.")
                continue

            # Check available pollutants in the dataset
            selected_pollutants = ['pm25', 'pm10']
            valid_pollutants = [p for p in selected_pollutants if p in filtered_df.columns]

            if not valid_pollutants:
                st.warning(f"No valid pollutants found in {label}")
                continue

            selected_display_pollutants = st.multiselect(
                f"Select Pollutants to Display for {label}",
                options=["All"] + valid_pollutants,
                default=["All"],
                key=unique_key("tab1", "pollutants", label)
            )

            if "All" in selected_display_pollutants:
                selected_display_pollutants = valid_pollutants

            # Choose chart type
            chart_type = st.radio(
                f"Chart Type for {label}",
                ["Line", "Bar"],
                horizontal=True,
                key=unique_key("tab1", "charttype", label)
            )

            df_avg_list = []
            for pollutant in selected_display_pollutants:
                if pollutant in filtered_df.columns:
                    avg_df = calculate_quarter_pollutant(filtered_df, pollutant)
                    avg_df["pollutant"] = pollutant
                    avg_df.rename(columns={pollutant: "value"}, inplace=True)
                    df_avg_list.append(avg_df)

            # Check if no data is available for the selected pollutants
            if not df_avg_list:
                st.warning(f"No data available for selected pollutants in {label}")
                continue

            # Combine the dataframes into one
            df_avg = pd.concat(df_avg_list, ignore_index=True)

            # Default to "day" as the x-axis, if the column exists
            x_axis = "quarter" if "quarter" in filtered_df.columns else "year"
            y_title = "µg/m³"
            plot_title = f"Aggregated {chart_type} Chart - {label}"

            # Generate the chart based on the selected chart type
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


                fig.update_layout(
                    yaxis_title=y_title,
                    height=500,
                    legend_title="Site",
                    margin=dict(t=40, b=40)
                )

            # Display the chart
            st.markdown('<div class="glass-container">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)

            # Optionally, display the aggregated data table
            with st.expander("Show Aggregated Data Table"):
                st.dataframe(df_avg)

            st.markdown('</div>', unsafe_allow_html=True)

def calculate_dayofweek_pollutant(df,pollutant):
    df_grouped = df.groupby(['site', 'dayofweek', 'quarter', 'year'])[pollutant].mean().reset_index().round(1)
   
    return df_grouped

def render_dayofweek_means_tab(tab, dfs, selected_years, calculate_dayofweek_pollutant, unique_key):
    with tab:
        st.header("📊 Days of Week Means")

        for label, df in dfs.items():
            st.subheader(f"Dataset: {label}")
            
            site_in_tab = st.multiselect(
                f"Select Site(s) for {label}",
                sorted(df['site'].unique()),
                key=unique_key("tab6", "site", label)
            )

            filtered_df = df.copy()

            # Filter by year
            if selected_years:
                filtered_df = filtered_df[filtered_df['year'].isin(selected_years)]
                
            # Filter by site
            if site_in_tab:
                filtered_df = filtered_df[filtered_df['site'].isin(site_in_tab)]
                
            # Filter by quarter
            selected_quarters = st.multiselect(
                f"Select Quarter(s) for {label}",
                options=['Q1', 'Q2', 'Q3', 'Q4'],
                default=['Q1', 'Q2', 'Q3', 'Q4'],
                key=unique_key("tab6", "quarter", label)
            ) or []

            # Ensure selected_years is defined and not empty
            if not selected_years:
                selected_years = sorted(df['year'].unique())  # or handle this upstream

            # Correct mapping to full quarter strings like '2021Q1'
            selected_quarter_nums = [f"{year}{q}" for year in selected_years for q in selected_quarters]

            
            if selected_quarter_nums:
                filtered_df = filtered_df[filtered_df['quarter'].isin(selected_quarter_nums)]
                
            else:
                st.warning("No valid quarters to filter!")
                continue

            # Check if no data remains after filtering
            if filtered_df.empty:
                st.warning(f"No data remaining for {label} after filtering.")
                continue

            # Check available pollutants in the dataset
            selected_pollutants = ['pm25', 'pm10']
            valid_pollutants = [p for p in selected_pollutants if p in filtered_df.columns]

            if not valid_pollutants:
                st.warning(f"No valid pollutants found in {label}")
                continue

            selected_display_pollutants = st.multiselect(
                f"Select Pollutants to Display for {label}",
                options=["All"] + valid_pollutants,
                default=["All"],
                key=unique_key("tab6", "pollutants", label)
            )

            if "All" in selected_display_pollutants:
                selected_display_pollutants = valid_pollutants

            # Choose chart type
            chart_type = st.radio(
                f"Chart Type for {label}",
                ["Line", "Bar"],
                horizontal=True,
                key=unique_key("tab6", "charttype", label)
            )

            df_avg_list = []
            for pollutant in selected_display_pollutants:
                if pollutant in filtered_df.columns:
                    avg_df = calculate_dayofweek_pollutant(filtered_df, pollutant)
                    avg_df["pollutant"] = pollutant
                    avg_df.rename(columns={pollutant: "value"}, inplace=True)
                    df_avg_list.append(avg_df)

            # Check if no data is available for the selected pollutants
            if not df_avg_list:
                st.warning(f"No data available for selected pollutants in {label}")
                continue

            # Combine the dataframes into one
            df_avg = pd.concat(df_avg_list, ignore_index=True)

            # Default to "day" as the x-axis, if the column exists
            x_axis = "dayofweek" if "dayofweek" in filtered_df.columns else "month"
            y_title = "µg/m³"
            plot_title = f"Aggregated {chart_type} Chart - {label}"

            # Generate the chart based on the selected chart type
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


                fig.update_layout(
                    yaxis_title=y_title,
                    height=500,
                    legend_title="Site",
                    margin=dict(t=40, b=40)
                )

            # Display the chart
            st.markdown('<div class="glass-container">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)

            # Optionally, display the aggregated data table
            with st.expander("Show Aggregated Data Table"):
                st.dataframe(df_avg)

            st.markdown('</div>', unsafe_allow_html=True)



def unique_key(tab: str, widget: str, label: str) -> str:
    return f"{widget}_{tab}_{label}"

def plot_chart(df, x, y, color, chart_type="line", title="", bar_mode="group"):
    streamlit_theme = st.get_option("theme.base")
    theme = streamlit_theme if streamlit_theme else "Light"

    background = '#1c1c1c' if theme == 'dark' else 'white'
    font_color = 'white' if theme == 'dark' else 'black'
    grid_color = '#444' if theme == 'dark' else '#ccc'

    # Define consistent color mapping
    predefined_colors = {
        "pm25": "#1f77b4",   # blue
        "pm10": "#ff7f0e",   # orange
        # Add more pollutants here if needed
    }

    if color and df[color].nunique() > 1:
        selector_param = alt.param(
            name="selector",
            bind=alt.binding_select(options=sorted(df[color].unique()), name=f"Filter {color}: "),
            value=df[color].iloc[0]
        )
        filter_condition = alt.datum[color] == selector_param
    else:
        selector_param = None
        filter_condition = True

    # Axis encoding
    x_encoding = alt.X(f"{x}:N", title=x, axis=alt.Axis(labelAngle=-45)) if chart_type == "bar" else alt.X(x, title=x)

    # Apply consistent color mapping
    if color:
        unique_vals = df[color].unique().tolist()
        color_scale = alt.Scale(domain=list(predefined_colors.keys()), range=list(predefined_colors.values()))
        color_encoding = alt.Color(color, scale=color_scale, legend=alt.Legend(title=color))
    else:
        color_encoding = alt.value("steelblue")

    base = alt.Chart(df).encode(
        x=x_encoding,
        y=alt.Y(y, title=y),
        color=color_encoding,
        tooltip=[x, y, color] if color else [x, y]
    ).properties(
        title=alt.TitleParams(text=title, color=font_color),
        width=700,
        height=400
    ).configure(
        background=background,
        view={"stroke": None},
        axis={
            "labelColor": font_color,
            "titleColor": font_color,
            "gridColor": grid_color,
            "domainColor": grid_color
        },
        legend={
            "labelColor": font_color,
            "titleColor": font_color
        }
    )

    chart = base.mark_line(point=True) if chart_type == "line" else base.mark_bar()

    if selector_param:
        chart = chart.add_params(selector_param).transform_filter(filter_condition)

    return chart.interactive()





# --- Main App ---

uploaded_files = st.file_uploader("Upload up to 4 datasets", type=['csv', 'xlsx'], accept_multiple_files=True)

if uploaded_files:
    all_outputs = {}
    site_options = set()
    year_options = set()
    dfs = {}

    for file in uploaded_files:
        label = file.name.split('.')[0]
        ext = file.name.split('.')[-1]
        df = pd.read_excel(file) if ext == 'xlsx' else pd.read_csv(file)

        df = parse_dates(df)
        df = standardize_columns(df)
        df = cleaned(df)

        if 'datetime' not in df.columns or 'pm25' not in df.columns or 'pm10' not in df.columns or 'site' not in df.columns:
            st.warning(f"⚠️ Could not process {label}: missing columns.")
            continue

        dfs[label] = df
        site_options.update(df['site'].unique())
        year_options.update(df['year'].unique())

    with st.sidebar:
        selected_years = st.multiselect("📅 Filter by Year", sorted(year_options))
        selected_sites = st.multiselect("🏢 Filter by Site", sorted(site_options))

    
    tabs = st.tabs([
        "Aggregated Means",
        "Quartely Means",
        "Monthly Means",
        "Exceedances & Max/Min Values", 
        "AQI Stats", 
        "Daily Means", 
        "Day of Week Means"
    ])

    render_quarter_means_tab(tabs[1], dfs, selected_years, calculate_quarter_pollutant, unique_key)
    render_monthly_means_tab(tabs[2], dfs, selected_years, calculate_month_pollutant, unique_key)
    render_aqi_tab(tabs[4], selected_years,  calculate_aqi_and_category, unique_key)
    render_daily_means_tab(tabs[5], dfs, selected_years, calculate_day_pollutant, unique_key)
    render_dayofweek_means_tab(tabs[6], dfs, selected_years, calculate_dayofweek_pollutant, unique_key)
    render_exceedances_tab(tabs[3], dfs, selected_years, calculate_exceedances,calculate_min_max)
    
    with tabs[0]:  # Aggregated Means
        st.header("📊 Aggregated Means")
        for label, df in dfs.items():
            st.subheader(f"Dataset: {label}")
            site_in_tab = st.multiselect(f"Select Site(s) for {label}", sorted(df['site'].unique()), key=f"site_agg_{label}")
            filtered_df = df.copy()
            if selected_years:
                filtered_df = filtered_df[filtered_df['year'].isin(selected_years)]
            if site_in_tab:
                filtered_df = filtered_df[filtered_df['site'].isin(site_in_tab)]
            
                
            selected_pollutants = ['pm25', 'pm10']
            valid_pollutants = [p for p in selected_pollutants if p in filtered_df.columns]
            if not valid_pollutants:
                st.warning(f"No valid pollutants found in {label}")

            selected_display_pollutants = st.multiselect(
                f"Select Pollutants to Display for {label}",
                options=["All"] + valid_pollutants,
                default=["All"],
                key=f"pollutants_{label}"
            )
            if "All" in selected_display_pollutants:
                selected_display_pollutants = valid_pollutants

            aggregate_levels = [
                ('Daily Avg', ['day', 'site']),
                ('Monthly Avg', ['month', 'site']),
                ('Quarterly Avg', ['quarter', 'site']),
                ('Yearly Avg', ['year', 'site']),
                ('Day of Week Avg', ['dayofweek', 'site']),
                ('Weekday Type Avg', ['weekday_type', 'site']),
                ('Season Avg', ['season', 'site'])
            ]
            for level_name, group_keys in aggregate_levels:
                agg_label = f"{label} - {level_name}"
                agg_dfs = []
                for pollutant in valid_pollutants:
                    agg_df = filtered_df.groupby(group_keys)[pollutant].mean().reset_index().round(1)
                    agg_dfs.append(agg_df)
                from functools import reduce
                merged_df = reduce(lambda left, right: pd.merge(left, right, on=group_keys, how='outer'), agg_dfs)
                display_cols = group_keys + [p for p in selected_display_pollutants if p in merged_df.columns]
                editable_df = merged_df[display_cols]
                
                st.data_editor(
                    editable_df,
                    use_container_width=True,
                    column_config={col: {"disabled": True} for col in editable_df.columns},
                    num_rows="dynamic",
                    key=f"editor_{label}_{agg_label}"
                )
                st.download_button(
                    label=f"📥 Download {agg_label}",
                    data=to_csv_download(editable_df),
                    file_name=f"{label}_{agg_label.replace(' ', '_')}.csv",
                    mime="text/csv"
                )
                st.markdown("---")
                auto_expand = "Yearly Avg" in agg_label
                with st.expander(f"📈 Show Charts for {agg_label}", expanded=auto_expand):
                    default_chart_type = "line" if any(keyword in level_name for keyword in ['Daily', 'Monthly', 'Quarterly', 'Yearly']) else "bar"
                    chart_type_choice = st.selectbox(
                        f"Select Chart Type for {agg_label}",
                        options=["line", "bar"],
                        index=0 if "Yearly" in agg_label else 1,
                        key=f"chart_type_{label}_{agg_label}"
                    )
                    x_axis = next(
                        (col for col in editable_df.columns if col not in ["site"] + valid_pollutants),
                        None
                    )
                    if not x_axis or x_axis not in editable_df.columns:
                        st.warning(f"Could not determine x-axis column for {agg_label}")
                        continue
                    safe_pollutants = [
                        p for p in selected_display_pollutants if p in editable_df.columns
                    ]
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
                        if "pollutant" not in df_melted.columns:
                            st.error(f"'pollutant' column missing in melted DataFrame for {agg_label}")
                            st.dataframe(df_melted.head())
                            continue
                        color_map = alt.Scale(domain=["pm25", "pm10"], range=["#1f77b4", "#ff7f0e"])
                        if chart_type_choice == "line":
                            fig = px.line(
                                df_melted,
                                x=x_axis,
                                y="value",
                                color="pollutant",  # Changed to allow side-by-side comparison of pm10 and pm25
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
                            st.markdown('<div class="glass-container">', unsafe_allow_html=True)
                            st.plotly_chart(fig, use_container_width=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error plotting chart: {e}")

    

                       
                    
                    
                
            

else:
    st.info("Upload CSV or Excel files from different air quality sources to begin.")
