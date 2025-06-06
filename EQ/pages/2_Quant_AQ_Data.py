import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

# --- Page Configuration ---
st.set_page_config(page_title="Quant AQ LCS Data Analyzer",page_icon="🛠️", layout="wide")

if "theme" not in st.session_state:
    st.session_state.theme = "Light"
if "font_size" not in st.session_state:
    st.session_state.font_size = "Medium"
theme_choice = st.sidebar.selectbox(
    "Choose Theme",
    ["Light", "Dark", "Blue", "Green", "Purple"],
    index=["Light", "Dark", "Blue", "Green", "Purple"].index(st.session_state.theme)
)
st.session_state.theme = theme_choice

# Font size selection
font_choice = st.sidebar.radio("Font Size", ["Small", "Medium", "Large"],
                               index=["Small", "Medium", "Large"].index(st.session_state.font_size))
st.session_state.font_size = font_choice

# Reset to default
if st.sidebar.button("🔄 Reset to Defaults"):
    st.session_state.theme = "Light"
    st.session_state.font_size = "Medium"
    st.success("Reset to Light theme and Medium font!")
    st.rerun()

# Theme settings dictionary
themes = {
    "Light": {
        "background": "rgba(255, 255, 255, 0.4)",
        "text": "#004d40",
        "button": "#00796b",
        "hover": "#004d40",
        "input_bg": "rgba(255, 255, 255, 0.6)"
    },
    "Dark": {
        "background":"rgba(22, 27, 34, 0.4)",
        "text": "#e6edf3",
        "button": "#238636",
        "hover": "#2ea043",
        "input_bg": "rgba(33, 38, 45, 0.6)"
    },
    "Blue": {
        "background": "rgba(210, 230, 255, 0.4)",
        "text": "#0a2540",
        "button": "#1e88e5",
        "hover": "#1565c0",
        "input_bg": "rgba(255, 255, 255, 0.6)"
    },
    "Green": {
        "background": "rgba(223, 255, 231, 0.4)", 
        "text": "#1b5e20",
        "button": "#43a047",
        "hover": "#2e7d32",
        "input_bg": "rgba(255, 255, 255, 0.6)"
    },
    "Purple": {
        "background": "rgba(240, 225, 255, 0.4)",
        "text": "#4a148c",
        "button": "#8e24aa",
        "hover": "#6a1b9a",
        "input_bg": "rgba(255, 255, 255, 0.6)"
    }
}

# Font size mapping
font_map = {"Small": "14px", "Medium": "16px", "Large": "18px"}

# Apply theme and inject CSS
theme = themes[st.session_state.theme]
font_size = font_map[st.session_state.font_size]
def generate_css(theme: dict, font_size: str) -> str:
    return f"""
    <style>
    html, body, .stApp, [class^="css"], button, input, label, textarea, select {{
        font-size: {font_size} !important;
        color: {theme["text"]} !important;
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }}
    .stApp {{
        background-color: {theme["background"]} !important;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        background-attachment: fixed;
        transition: background 0.5s ease, color 0.5s ease;
    }}
    html, body, [class^="css"] {{
        background-color: transparent !important;
    }}
    h1, h2, h3 {{
        font-weight: bold;
    }}
    .stTextInput > div > input,
    .stSelectbox > div > div,
    .stRadio > div,
    textarea {{
        background-color: {theme["input_bg"]} !important;
        color: {theme["text"]} !important;
        border: 1px solid {theme["button"]};
        backdrop-filter: blur(6px);
        -webkit-backdrop-filter: blur(6px);
    }}
    div.stButton > button {{
        background-color: {theme["button"]};
        color: white;
        padding: 0.5em 1.5em;
        border-radius: 8px;
        transition: background-color 0.3s ease;
    }}
    div.stButton > button:hover {{
        background-color: {theme["hover"]};
    }}

    body, .stApp {{
        font-family: 'Poppins', sans-serif;
        transition: all 0.5s ease;
    }}

    body.light-mode, .stApp.light-mode {{
        background: linear-gradient(135deg, #f8fdfc, #d8f3dc);
        color: #1b4332;
    }}

    body.dark-mode, .stApp.dark-mode {{
        background: rgba(22, 27, 34, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        color: #e6edf3;
    }}

    [data-testid="stSidebar"] {{
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(12px);
        border-right: 2px solid #74c69d;
        transition: all 0.5s ease;
    }}

    .stButton>button, .stDownloadButton>button {{
        background: background-color;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7em 1.5em;
        font-weight: bold;
        font-size: 1rem;
        box-shadow: 0 0 15px #52b788;
        transition: 0.3s ease;
    }}

    .stButton>button:hover, .stDownloadButton>button:hover {{
        background: background-color);
        box-shadow: 0 0 25px #74c69d, 0 0 35px #74c69d;
        transform: scale(1.05);
    }}
    .glass-container {{
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 15px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        padding: 1.5rem;
        margin-bottom: 2rem;
    }}
    .stContainer {{
        font-size: {font_size}px;
        color: {theme.get('text_color', '#000')};
    }}
    ::-webkit-scrollbar {{
        width: 8px;
    }}
    ::-webkit-scrollbar-thumb {{
        background: #74c69d;
        border-radius: 10px;
    }}
    ::-webkit-scrollbar-thumb:hover {{
        background: #52b788;
    }}

    .glow-text {{
        text-align: center;
        font-size: 3em;
        color: #52b788;
        text-shadow: 0 0 5px #52b788, 0 0 10px #52b788, 0 0 20px #52b788;
        margin-bottom: 20px;
    }}

    html, body, .stApp {{
        transition: background 0.5s ease, color 0.5s ease;
    }}

    .stDownloadButton>button {{
        background: background-color;
        box-shadow: 0 0 10px #1b4332;
    }}

    .stButton>button:active, .stDownloadButton>button:active {{
        transform: scale(0.97);
    }}

    .stDataFrame, .stTable {{
        background: rgba(255, 255, 255, 0.6);
        border-radius: 12px;
        backdrop-filter: blur(10px);
        padding: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        overflow: hidden;
        font-size: 15px;
    }}

    thead tr th {{
        background: background-color;
        color: white;
        font-weight: bold;
        text-align: center;
        padding: 0.5em;
    }}

    tbody tr:nth-child(even) {{
        background-color: #eeeeee;
    }}
    tbody tr:nth-child(odd) {{
        background-color: #ffffff;
    }}
    tbody tr:hover {{
        background-color: #b7e4c7;
        transition: background-color 0.3s ease;
    }}
    .altair-glass {{
        background: rgba(255, 255, 255, 0.2); 
        backdrop-filter: blur(12px);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease-in-out;
    }}
    body.dark-mode .altair-glass {{
        background: rgba(22, 27, 34, 0.3);
        box-shadow: 0 8px 24px rgba(88, 166, 255, 0.2);
    }}

    .element-container iframe {{
        background: rgba(255, 255, 255, 0.5) !important;
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 10px;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
    }}

    body.dark-mode .stDataFrame, body.dark-mode .stTable {{
        background: rgba(33, 38, 45, 0.6);
        border-radius: 12px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        font-size: 15px;
        overflow: hidden;
    }}

    body.dark-mode thead tr th {{
        background: linear-gradient(135deg, #238636, #2ea043);
        color: #ffffff;
        font-weight: bold;
        text-align: center;
    }}

    body.dark-mode tbody tr:nth-child(even) {{
        background: linear-gradient(90deg, #21262d, #30363d);
        color: #e6edf3;
        transition: all 0.3s ease;
    }}

    body.dark-mode tbody tr:nth-child(odd) {{
        background: linear-gradient(90deg, #161b22, #21262d);
        color: #e6edf3;
        transition: all 0.3s ease;
    }}

    body.dark-mode tbody tr:hover {{
        background: linear-gradient(90deg, #21262d, #30363d);
        box-shadow: 0 0 15px #58a6ff;
        transform: scale(1.01);
    }}

    body.dark-mode .element-container iframe {{
        background: rgba(33, 38, 45, 0.5) !important;
        backdrop-filter: blur(10px);
        border: 2px solid #58a6ff;
        padding: 10px;
        border-radius: 16px;
        box-shadow: 0 0 15px #58a6ff, 0 0 30px #79c0ff;
        animation: pulse-glow-dark 3s infinite ease-in-out;
    }}

    @keyframes pulse-glow {{
      0% {{ box-shadow: 0 0 15px #74c69d, 0 0 30px #52b788; }}
      50% {{ box-shadow: 0 0 25px #40916c, 0 0 45px #2d6a4f; }}
      100% {{ box-shadow: 0 0 15px #74c69d, 0 0 30px #52b788; }}
    }}
    @keyframes pulse-glow-dark {{
      0% {{ box-shadow: 0 0 15px #58a6ff, 0 0 30px #79c0ff; }}
      50% {{ box-shadow: 0 0 25px #3b82f6, 0 0 45px #2563eb; }}
      100% {{ box-shadow: 0 0 15px #58a6ff, 0 0 30px #79c0ff; }}
    }}

    .footer {{
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: {theme["background"]};
        color: {theme["text"]};
        text-align: center;
        padding: 12px 0;
        font-size: 14px;
        font-weight: bold;
        box-shadow: 0px -2px 10px rgba(0,0,0,0.1);
        backdrop-filter: blur(8px);
    }}
    </style>
    """

# Inject Dynamic Theme
st.markdown(generate_css(theme, font_size), unsafe_allow_html=True)



# --- Title and Logo ---
st.title("📊 Quant AQ LCS Data Analyzer")
# --- Helper Functions ---

@st.cache_data(ttl=600)

def cleaned(df):
    df = df.rename(columns=lambda x: x.strip().lower())
    required_columns = ['datetime', 'site', 'pm25', 'pm10','sample_temp', 'sample_rh']
    df = df[[col for col in required_columns if col in df.columns]]
    df = df.dropna(axis=1, how='all').dropna()

    # Keep only rows with all required numeric columns > 0
    for col in ['pm1', 'pm25', 'pm10', 'sample_temp', 'sample_rh']:
        if col in df.columns:
            df = df[df[col] > 0]

    # Apply correction formula for PM2.5 if applicable
    if all(col in df.columns for col in ['pm25', 'sample_temp', 'sample_rh']):
        df['pm25'] = 0.94 * df['pm25'] - 0.34 * df['sample_temp'] - 0.08 * df['sample_rh'] + 19.82

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
                df['datetime'] = pd.to_datetime(df[col], utc=True, errors='coerce')
                df = df.dropna(subset=['datetime'])
                return df
            except:
                continue
    return df

def standardize_columns(df):
    pm25_cols = ['pm25', 'PM2.5', 'pm25_avg', 'pm2.5']
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

            # --- Generate Quarter Identifiers ---
            selected_quarter_nums = [
                f"{year}{q}" for year in selected_years_in_tab for q in selected_quarters
            ] if selected_years_in_tab and selected_quarters else []

            # --- Filter by Quarter ---
            if selected_quarter_nums:
                filtered_df = filtered_df[filtered_df['quarter'].isin(selected_quarter_nums)]
            else:
                st.warning("No valid quarters to filter!")
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

def render_aqi_tab(tab, selected_years, calculate_aqi_and_category, unique_key):
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

            site_in_tab = st.multiselect(f"Select Site(s) for {label}", sorted(df['site'].unique()), key=f"site_aqi_{label}")

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
                        st.markdown(f"**{current_year} AQI**")
                        fig_current = px.bar(
                            current_df,
                            x="Percent",
                            y="AQI_Remark",
                            color="AQI_Remark",
                            orientation="h",
                            color_discrete_map=aqi_colors,
                            hover_data=["site", "Percent"],
                        )
                        fig_current.update_layout(
                            xaxis_title="% Time in AQI Category",
                            yaxis_title="AQI Category",
                            yaxis=dict(categoryorder="total ascending"),
                            showlegend=False,
                            height=400
                        )
                        
                        st.plotly_chart(fig_current, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"**{previous_year} AQI**")
                        fig_prev = px.bar(
                            prev_df,
                            x="Percent",
                            y="AQI_Remark",
                            color="AQI_Remark",
                            orientation="h",
                            color_discrete_map=aqi_colors,
                            hover_data=["site", "Percent"],
                        )
                        fig_prev.update_layout(
                            xaxis_title="% Time in AQI Category",
                            yaxis_title="AQI Category",
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
            selected_pollutants = ['corrected_pm25', 'pm10']
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
                    threshold = 35 if pollutant.lower() == "corrected_pm25" else 70
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
            selected_pollutants = ['corrected_pm25', 'pm10']
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
            selected_pollutants = ['corrected_pm25', 'pm10']
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
            selected_pollutants = ['corrected_pm25', 'pm10']
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
            
                
            selected_pollutants = ['corrected_pm25', 'pm10']
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
                        color_map = alt.Scale(domain=['corrected_pm25', 'pm10'], range=["#1f77b4", "#ff7f0e"])
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
