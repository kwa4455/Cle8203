import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from io import BytesIO

# --- Page Configuration ---
st.set_page_config(page_title="Clarity LCS Data Analysis", page_icon="🔬",layout="wide")


# Insert your CSS here
st.markdown("""
    <style>
    /* --- your CSS --- */
    body, .stApp {
        font-family: 'Poppins', sans-serif;
        transition: all 0.5s ease;
    }
    
    /* Light Mode */
    body.light-mode, .stApp.light-mode {
        background: linear-gradient(135deg, #f8fdfc, #d8f3dc);
        color: #1b4332;
    }
    
    /* Dark Mode */
    body.dark-mode, .stApp.dark-mode {
        background: linear-gradient(135deg, #0e1117, #161b22);
        color: #e6edf3;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(12px);
        border-right: 2px solid #74c69d;
        transition: all 0.5s ease;
    }

    /* Buttons */
    .stButton>button, .stDownloadButton>button {
        background: linear-gradient(135deg, #40916c, #52b788);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7em 1.5em;
        font-weight: bold;
        font-size: 1rem;
        box-shadow: 0 0 15px #52b788;
        transition: 0.3s ease;
    }
    
    .stButton>button:hover, .stDownloadButton>button:hover {
        background: linear-gradient(135deg, #2d6a4f, #40916c);
        box-shadow: 0 0 25px #74c69d, 0 0 35px #74c69d;
        transform: scale(1.05);
    }

    /* Custom Scrollbars */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-thumb {
        background: #74c69d;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #52b788;
    }

    /* Glowing Title */
    .glow-text {
        text-align: center;
        font-size: 3em;
        color: #52b788;
        text-shadow: 0 0 5px #52b788, 0 0 10px #52b788, 0 0 20px #52b788;
        margin-bottom: 20px;
    }

    /* Smooth theme transition */
    html, body, .stApp {
        transition: background 0.5s ease, color 0.5s ease;
    }

    /* Download Button Specific */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #1b4332, #2d6a4f);
        box-shadow: 0 0 10px #1b4332;
    }

    /* Button Press Animation */
    .stButton>button:active, .stDownloadButton>button:active {
        transform: scale(0.97);
    }

    /* Tables */
    .stDataFrame, .stTable {
        background: rgba(255, 255, 255, 0.6);
        border-radius: 12px;
        backdrop-filter: blur(10px);
        padding: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        overflow: hidden;
        font-size: 15px;
    }

    /* Table Headers */
    thead tr th {
        background: linear-gradient(135deg, #52b788, #74c69d);
        color: white;
        font-weight: bold;
        text-align: center;
        padding: 0.5em;
    }

    /* Table Rows */
    tbody tr:nth-child(even) {
        background-color: #e9f7ef;
    }
    tbody tr:nth-child(odd) {
        background-color: #ffffff;
    }
    tbody tr:hover {
        background-color: #b7e4c7;
        transition: background-color 0.3s ease;
    }

    /* Graph iframe Glass Effect */
    .element-container iframe {
        background: rgba(255, 255, 255, 0.5) !important;
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 10px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
    }

    /* Dark Mode Table */
    body.dark-mode .stDataFrame, body.dark-mode .stTable {
        background: #161b22cc;
        border-radius: 10px;
        backdrop-filter: blur(8px);
        font-size: 15px;
        overflow: hidden;
    }
    body.dark-mode thead tr th {
        background: linear-gradient(135deg, #238636, #2ea043);
        color: #ffffff;
        font-weight: bold;
        text-align: center;
    }
    body.dark-mode tbody tr:nth-child(even) {
        background: linear-gradient(90deg, #21262d, #30363d);
        color: #e6edf3;
        transition: all 0.3s ease;
    }
    body.dark-mode tbody tr:nth-child(odd) {
        background: linear-gradient(90deg, #161b22, #21262d);
        color: #e6edf3;
        transition: all 0.3s ease;
    }
    body.dark-mode tbody tr:hover {
        background: linear-gradient(90deg, #21262d, #30363d);
        box-shadow: 0 0 15px #58a6ff;
        transform: scale(1.01);
    }

    /* Dark Mode Graph Glow */
    body.dark-mode .element-container iframe {
        background: rgba(22, 27, 34, 0.5) !important;
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 10px;
        border: 2px solid #58a6ff;
        box-shadow: 0 0 15px #58a6ff, 0 0 30px #79c0ff;
        animation: pulse-glow-dark 3s infinite ease-in-out;
    }

    /* Glow Animations */
    @keyframes pulse-glow {
      0% { box-shadow: 0 0 15px #74c69d, 0 0 30px #52b788; }
      50% { box-shadow: 0 0 25px #40916c, 0 0 45px #2d6a4f; }
      100% { box-shadow: 0 0 15px #74c69d, 0 0 30px #52b788; }
    }
    @keyframes pulse-glow-dark {
      0% { box-shadow: 0 0 15px #58a6ff, 0 0 30px #79c0ff; }
      50% { box-shadow: 0 0 25px #3b82f6, 0 0 45px #2563eb; }
      100% { box-shadow: 0 0 15px #58a6ff, 0 0 30px #79c0ff; }
    }
    </style>
""", unsafe_allow_html=True)





# --- Title and Logo ---
st.title("📊 Clarity LCS Data Analysis")


# --- Helper Functions ---

@st.cache_data(ttl=600)

def cleaned(df):
    df = df.rename(columns=lambda x: x.strip().lower())
    required_columns = ['datetime', 'site', 'corrected_pm25', 'pm10']
    df = df[[col for col in required_columns if col in df.columns]]
    df = df.dropna(axis=1, how='all').dropna()


    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.to_period('M').astype(str)
    df['quarter'] = df['datetime'].dt.to_period('Q').astype(str)
    df['day'] = df['datetime'].dt.date
    df['dayofweek'] = df['datetime'].dt.day_name()
    df['weekday_type'] = df['datetime'].dt.weekday.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
    df['season'] = df['datetime'].dt.month.apply(lambda x: 'Harmattan' if x in [12, 1, 2] else 'Non-Harmattan')

    daily_counts = df.groupby(['site', 'month'])['day'].nunique().reset_index(name='daily_counts')
    sufficient_sites = daily_counts[daily_counts['daily_counts'] >= 20][['site', 'month']]
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
    pm25_cols = ['PM2.5', 'corrected_pm25', 'pm25_avg', 'pm2.5']
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

def compute_aggregates(df, label, pollutant):
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
    daily_avg = df.groupby(['site', 'day', 'year', 'month'], as_index=False).agg({
        'corrected_pm25': 'mean',
        'pm10': 'mean'
    })
    pm25_exceed = daily_avg[daily_avg['corrected_pm25'] > 35].groupby(['year', 'site']).size().reset_index(name='PM25_Exceedance_Count')
    pm10_exceed = daily_avg[daily_avg['pm10'] > 70].groupby(['year', 'site']).size().reset_index(name='PM10_Exceedance_Count')
    total_days = daily_avg.groupby(['year', 'site']).size().reset_index(name='Total_Records')

    exceedance = total_days.merge(pm25_exceed, on=['year', 'site'], how='left') \
                           .merge(pm10_exceed, on=['year', 'site'], how='left')
    exceedance.fillna(0, inplace=True)
    exceedance['PM25_Exceedance_Percent'] = round((exceedance['PM25_Exceedance_Count'] / exceedance['Total_Records']) * 100, 1)
    exceedance['PM10_Exceedance_Percent'] = round((exceedance['PM10_Exceedance_Count'] / exceedance['Total_Records']) * 100, 1)

    return exceedance

def calculate_min_max(df):
    daily_avg = df.groupby(['site', 'day', 'year', 'month'], as_index=False).agg({
        'pm25': 'mean',
        'pm10': 'mean'
    })
    df_min_max = daily_avg.groupby(['year', 'site', 'month'], as_index=False).agg(
        daily_avg_pm10_max=('pm10', lambda x: round(x.max(), 1)),
        daily_avg_pm10_min=('pm10', lambda x: round(x.min(), 1)),
        daily_avg_pm25_max=('pm25', lambda x: round(x.max(), 1)),
        daily_avg_pm25_min=('pm25', lambda x: round(x.min(), 1))
    )
    return df_min_max

def calculate_aqi_and_category(df):
    daily_avg = df.groupby(['site', 'day', 'year', 'month'], as_index=False).agg({
        'corrected_pm25': 'mean'
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

    daily_avg['AQI'] = daily_avg['corrected_pm25'].apply(calculate_aqi)
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

    remarks_counts = daily_avg.groupby(['site', 'year', 'AQI_Remark']).size().reset_index(name='Count')
    remarks_counts['Total_Count_Per_Site_Year'] = remarks_counts.groupby(['site', 'year'])['Count'].transform('sum')
    remarks_counts['Percent'] = round((remarks_counts['Count'] / remarks_counts['Total_Count_Per_Site_Year']) * 100, 1)

    return daily_avg, remarks_counts

def to_csv_download(df):
    return BytesIO(df.to_csv(index=False).encode('utf-8'))

def plot_chart(df, x, y, color, chart_type="line", title=""):
    # Automatically detect Streamlit theme
    streamlit_theme = st.get_option("theme.base")
    theme = streamlit_theme if streamlit_theme else "Light"
    
    background = '#1c1c1c' if theme == 'dark' else 'white'
    font_color = 'white' if theme == 'dark' else 'black'
    
    base = alt.Chart(df).encode(
        x=x,
        y=y,
        color=color,
        tooltip=[x, y, color]
    ).properties(
        width=700,
        height=400,
        title=alt.TitleParams(text=title, color=font_color)
    ).configure(
        background=background,
        view={"stroke": None},
        axis={"labelColor": font_color, "titleColor": font_color}
    )

    if chart_type == "line":
        return base.mark_line(point=True)
    else:
        return base.mark_bar()
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

        if 'datetime' not in df.columns or 'corrected_pm25' not in df.columns or 'pm10' not in df.columns or 'site' not in df.columns:
            st.warning(f"⚠️ Could not process {label}: missing columns.")
            continue

        dfs[label] = df
        site_options.update(df['site'].unique())
        year_options.update(df['year'].unique())

    with st.sidebar:
        selected_years = st.multiselect("📅 Filter by Year", sorted(year_options))
        selected_sites = st.multiselect("🏢 Filter by Site", sorted(site_options))

    tabs = st.tabs(["Aggregated Means", "Exceedances", "AQI Stats", "Min/Max Values"])

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

            for pollutant in ['corrected_pm25', 'pm10']:
                if pollutant not in filtered_df.columns:
                    continue
                aggregates = compute_aggregates(filtered_df, label, pollutant)
                for agg_label, agg_df in aggregates.items():
                    st.markdown(f"**{agg_label}**")
                    st.dataframe(agg_df, use_container_width=True)
                    st.download_button(label=f"📥 Download {agg_label}", data=to_csv_download(agg_df), file_name=f"{label}_{agg_label.replace(' ', '_')}.csv", mime="text/csv")
                    chart_type = "line" if any(t in agg_label for t in ['Daily', 'Monthly','Quarterly', 'Yearly']) else "bar"
                    x_axis = agg_df.columns[0]
                    chart = plot_chart(agg_df, x=x_axis, y=pollutant, color="site", chart_type=chart_type, title=agg_label)
                    st.altair_chart(chart, use_container_width=True)

    with tabs[1]:  # Exceedances
        st.header("🚨 Exceedances")
        for label, df in dfs.items():
            st.subheader(f"Dataset: {label}")
            site_in_tab = st.multiselect(f"Select Site(s) for {label}", sorted(df['site'].unique()), key=f"site_exc_{label}")
            filtered_df = df.copy()
            if selected_years:
                filtered_df = filtered_df[filtered_df['year'].isin(selected_years)]
            if site_in_tab:
                filtered_df = filtered_df[filtered_df['site'].isin(site_in_tab)]

            exceedances = calculate_exceedances(filtered_df)
            st.dataframe(exceedances, use_container_width=True)
            st.download_button(f"⬇️ Download Exceedances - {label}", to_csv_download(exceedances), file_name=f"Exceedances_{label}.csv")

    with tabs[2]:  # AQI
        st.header("🌫️ AQI Stats")
        for label, df in dfs.items():
            st.subheader(f"Dataset: {label}")
            site_in_tab = st.multiselect(f"Select Site(s) for {label}", sorted(df['site'].unique()), key=f"site_aqi_{label}")
            filtered_df = df.copy()
            if selected_years:
                filtered_df = filtered_df[filtered_df['year'].isin(selected_years)]
            if site_in_tab:
                filtered_df = filtered_df[filtered_df['site'].isin(site_in_tab)]

            daily_avg, remarks_counts = calculate_aqi_and_category(filtered_df)
            st.dataframe(remarks_counts, use_container_width=True)
            st.dataframe(daily_avg, use_container_width=True)
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
            aqi_chart = alt.Chart(remarks_counts).mark_bar().encode(
                x=alt.X('Percent:Q', title='% Time in AQI Category'),
                y=alt.Y('AQI_Remark:N', sort='-x', title='AQI Category'),
                color=alt.Color('AQI_Remark:N', scale=alt.Scale(domain=list(aqi_colors.keys()), range=list(aqi_colors.values())), legend=None),
                tooltip=['site', 'year', 'AQI_Remark', 'Percent']
            ).properties(width=700, height=400, title="AQI Remark Percentages")
            st.altair_chart(aqi_chart, use_container_width=True)

    with tabs[3]:  # Min/Max
        st.header("🔥 Min/Max Values")
        for label, df in dfs.items():
            st.subheader(f"Dataset: {label}")
            site_in_tab = st.multiselect(f"Select Site(s) for {label}", sorted(df['site'].unique()), key=f"site_minmax_{label}")
            filtered_df = df.copy()
            if selected_years:
                filtered_df = filtered_df[filtered_df['year'].isin(selected_years)]
            if site_in_tab:
                filtered_df = filtered_df[filtered_df['site'].isin(site_in_tab)]

            min_max = calculate_min_max(filtered_df)
            st.dataframe(min_max, use_container_width=True)
            st.download_button(f"⬇️ Download MinMax - {label}", to_csv_download(min_max), file_name=f"MinMax_{label}.csv")

else:
    st.info("Upload CSV or Excel files from different air quality sources to begin.")
