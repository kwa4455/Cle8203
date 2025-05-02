import pandas as pd
import numpy as np
from scipy.stats import kruskal
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# --- Page Config ---
st.set_page_config(page_title="Air Quality Dashboard", layout="wide")
st.title("🌍 Air Quality Data Explorer")

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
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        background-attachment: fixed;
        box-shadow: inset 0 0 50px rgba(255,255,255,0.1);
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
        border-radius: 8px;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);
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

    [data-testid="stSidebar"] {{
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        box-shadow: 4px 0 20px rgba(0, 0, 0, 0.2);
        border-right: 2px solid #74c69d;
    }}

    .stButton>button, .stDownloadButton>button {{
        background: {theme["button"]};
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7em 1.5em;
        font-weight: bold;
        font-size: 1rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3), 0 0 20px {theme["button"]};
        transition: all 0.3s ease;
    }}

    .stButton>button:hover, .stDownloadButton>button:hover {{
        background: {theme["hover"]};
        transform: scale(1.05);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4), 0 0 30px {theme["hover"]};
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
        background: {theme["button"]};
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

st.markdown(generate_css(theme, font_size), unsafe_allow_html=True)


# Helper: Column mappings
column_mappings = {
    'date': ['date', 'sampling date', 'datetime'],
    'id': ['id', 'station id', 'site id'],
    'cd': ['cd(ng/m3)', 'cadmium', 'cd'],
    'cd_error': ['cd_error', 'cd err'],
    'cr': ['cr(ng/m3)', 'chromium', 'cr'],
    'cr_error': ['cr_error', 'cr err'],
    'hg': ['hg(ng/m3)', 'mercury', 'hg'],
    'hg_error': ['hg_error', 'hg err'],
    'al': ['al(ug/m3)', 'aluminium', 'al'],
    'al_error': ['al_error', 'al err'],
    'as': ['as(ng/m3)', 'arsenic', 'as'],
    'as_error': ['as_error', 'as err'],
    'mn': ['mn(ng/m3)', 'manganese', 'mn'],
    'mn_error': ['mn_error', 'mn err'],
    'pb': ['pb(ng/m3)', 'lead', 'pb'],
    'pb_error': ['pb_error', 'pb err'],
    'site': ['site', 'site_location', 'source']
}

metals = ['cd', 'cr', 'hg', 'al', 'as', 'mn', 'pb']
errors = [f'{metal}_error' for metal in metals]
pollutant_cols = metals + errors

required_columns = ['date', 'site'] + pollutant_cols

# --- 1. Parse dates ---
def parse_dates(df):
    for col in df.columns:
        if 'date' in col.lower():
            try:
                df['date'] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')
                df = df.dropna(subset=['date'])
                return df
            except:
                continue
    return df

# --- 2. Standardize column names ---
def standardize_columns(df):
    rename_dict = {}
    for standard_col, aliases in column_mappings.items():
        for col in df.columns:
            if col.strip().lower() in [alias.lower() for alias in aliases]:
                rename_dict[col] = standard_col
    df.rename(columns=rename_dict, inplace=True)
    df.columns = [col.strip().lower() for col in df.columns]
    return df

# --- 3. Cleaned data ---
def cleaned(df):
    df = df[[col for col in required_columns if col in df.columns]].copy()
    df = df.dropna(axis=1, how='all').dropna()

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.to_period('M').astype(str)
    df['day'] = df['date'].dt.date
    df['dayofweek'] = df['date'].dt.day_name()
    df['weekday_type'] = df['date'].dt.weekday.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
    df['season'] = df['date'].dt.month.apply(lambda x: 'Harmattan' if x in [12, 1, 2] else 'Non-Harmattan')

    columns_to_select = ['site', 'day', 'year', 'month', 'dayofweek', 'season'] + pollutant_cols
    df = df[[col for col in columns_to_select if col in df.columns]]
    return df

# --- 4. Compute aggregated data ---
def compute_all_data(df):
    agg_dict = {
        'count': ('site', 'count')
    }

    for col in df.columns:
        if col in pollutant_cols:
            agg_dict[f'{col}_mean'] = (col, lambda x: round(x.mean(skipna=True), 2))
            agg_dict[f'{col}_sd'] = (col, lambda x: round(x.std(skipna=True), 2))
            agg_dict[f'{col}_median'] = (col, lambda x: round(x.median(skipna=True), 2))

    try:
        total_df = df.groupby('site').agg(**agg_dict).reset_index()
    except KeyError as e:
        st.error(f"Missing expected column(s) for aggregation: {e}")
        return pd.DataFrame()
    return total_df

# --- 5. Compute yearly data ---
def compute_yearly_data(df):
    agg_dict = {}
    for col in df.columns:
        if col in pollutant_cols:
            agg_dict[f'{col}_mean'] = (col, lambda x: round(x.mean(skipna=True), 2))
            agg_dict[f'{col}_sd'] = (col, lambda x: round(x.std(skipna=True), 2))
            agg_dict[f'{col}_median'] = (col, lambda x: round(x.median(skipna=True), 2))
            agg_dict[f'{col}_error_median'] = (col, lambda x: round(x.sem(skipna=True), 2))

    try:
        yearly_df = df.groupby(['site', 'year']).agg(**agg_dict).reset_index()
    except KeyError as e:
        st.error(f"Missing expected column(s) for yearly summary: {e}")
        return pd.DataFrame()
    return yearly_df

# --- 6. Compute correlation ---
def calculate_site_correlation(df):
    correlation_results = {}
    try:
        for metal in metals:
            if metal in df.columns:
                pivot = df.pivot_table(index='date', columns='site', values=metal)
                correlation = pivot.corr()
                correlation_results[metal] = correlation
    except KeyError as e:
        st.error(f"Missing required column for correlation: {e}")
    return correlation_results
    
def compute_monthly_data(df):
    agg_dict = {}
    for col in df.columns:
        if col in pollutant_cols:
            agg_dict[f'{col}_median'] = (col, lambda x: round(x.median(skipna=True), 2))

    try:
        monthly_df = df.groupby(['site', 'month']).agg(**agg_dict).reset_index()
    except KeyError as e:
        st.error(f"Missing expected column(s) for monthly summary: {e}")
        return pd.DataFrame()
    return monthly_df
# --- 8. Metal Filter UI ---
def metal_filter():
    return st.selectbox("Select Metal to View", metals)

# --- 9. File upload and Streamlit tabs ---
uploaded_files = st.file_uploader("Upload up to 4 datasets", type=['csv', 'xlsx'], accept_multiple_files=True)

if uploaded_files:
    selected_metal = metal_filter()

    all_outputs = {}
    dfs = {}

    for file in uploaded_files:
        label = file.name.split('.')[0]
        ext = file.name.split('.')[-1].lower()
        df = pd.read_excel(file) if ext == 'xlsx' else pd.read_csv(file)

        df = parse_dates(df)
        df = standardize_columns(df)
        df = cleaned(df)

        dfs[label] = df
        all_outputs[label] = {
            'All Sites Summary': compute_all_data(df),
            'Yearly Summary': compute_yearly_data(df),
            'Monthly Summary': compute_monthly_data(df),
            'Day-of-Week Summary': compute_dayofweek_data(df),
            'Min-Max Averages': calculate_min_max(df),
            'Kruskal-Wallis Test': calculate_kruskal_wallis(df),
            'Correlation': calculate_site_correlation(df)
        }

    metals_color = {
        "pb": "#ffff00", "cd": "green", "cr": "red",
        "mn": "purple", "al": "orange", "as": "maroon", "hg": "blue"
    }

    for label, outputs in all_outputs.items():
        st.header(f"Results for: {label}")
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "\U0001F4C8 Trends", "\U0001F4CA Box & Bar Plots", "\U0001F4A0 Kruskal & T-Test",
            "\U0001F517 Correlation", "\U0001F4C9 Theil-Sen Trend", "Temporal Variation"])

        with tab1:
            st.subheader("All Sites Summary")
            st.dataframe(outputs['All Sites Summary'])
            csv = outputs['All Sites Summary'].to_csv(index=False).encode('utf-8')
            st.download_button("Download Summary CSV", csv, f"{label}_summary.csv", "text/csv")

            df = dfs[label]
            summary_stats = outputs['All Sites Summary']
            if selected_metal in df.columns:
                fig = px.violin(
                    df,
                    x='site',
                    y=selected_metal,
                    box=True,
                    points='all',
                    color='site',
                    hover_data=df.columns
                )
                for i, row in summary_stats.iterrows():
                    fig.add_trace(go.Scatter(
                        x=[row['site']],
                        y=[row[f'{selected_metal}_mean']],
                        mode='markers+text',
                        name='Mean ± SD',
                        marker=dict(color='red', size=10, symbol='x'),
                        error_y=dict(
                            type='data',
                            array=[row[f'{selected_metal}_sd']],
                            visible=True,
                            thickness=1.5,
                            width=3
                        ),
                        text=[f"μ={round(row[f'{selected_metal}_mean'], 2)}"],
                        textposition='top center',
                        showlegend=False
                    ))
                fig.update_layout(
                    title=f"Distribution of {selected_metal.upper()} (ng/m³) by Site",
                    yaxis_title=f"{selected_metal.upper()} (ng/m³)",
                    xaxis_title=None,
                    font=dict(size=14),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(l=40, r=40, t=40, b=40),
                    showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

        with tab2:
            df = dfs[label]
            yearly_df = outputs['Yearly Summary']
            latest_year = yearly_df['year'].max()
            summary_data = yearly_df[yearly_df['year'] == latest_year]

            mean_col = f"{selected_metal}_median"
            error_col = f"{selected_metal}_error_median"
            if mean_col in summary_data.columns and error_col in summary_data.columns:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=summary_data['site'],
                    y=summary_data[mean_col],
                    name=selected_metal.upper(),
                    marker_color=metals_color[selected_metal],
                    error_y=dict(
                        type='data',
                        array=summary_data[error_col],
                        visible=True,
                        thickness=1.5,
                        width=3
                    ),
                    text=summary_data[mean_col],
                    textposition='outside'
                ))
                y_label = f"{selected_metal.upper()} (ug/m³)" if selected_metal == 'al' else f"{selected_metal.upper()} (ng/m³)"
                fig.update_layout(
                    title=f"{selected_metal.upper()} by Site - {latest_year}",
                    xaxis_title=None,
                    yaxis_title=y_label,
                    yaxis=dict(range=[0, summary_data[mean_col].max() * 1.2]),
                    plot_bgcolor='white',
                    showlegend=False,
                    font=dict(size=14),
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                fig.update_xaxes(tickangle=45, tickfont=dict(size=12))
                fig.update_yaxes(tickfont=dict(size=14))
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("Kruskal-Wallis Test")
            st.dataframe(outputs['Kruskal-Wallis Test'])

        with tab4:
            st.subheader("Correlation Between Sites")
            correlation_data = outputs['Correlation']
            if selected_metal in correlation_data:
                st.write(f"**{selected_metal.upper()} Correlation Matrix**")
                corr_df = correlation_data[selected_metal]
                st.dataframe(corr_df)
                fig = px.imshow(corr_df, text_auto=True, title=f"{selected_metal.upper()} Correlation Between Sites")
                st.plotly_chart(fig, use_container_width=True)

        with tab5:
            st.subheader("Min-Max Averages")
            st.dataframe(outputs['Min-Max Averages'])

        with tab6:
            st.subheader("Day-of-Week Summary")
            dow_df = outputs['Day-of-Week Summary']
            median_col = f"{selected_metal}_median"
            if median_col in dow_df.columns:
                fig = px.bar(
                    dow_df,
                    x='dayofweek',
                    y=median_col,
                    color='site',
                    barmode='group',
                    title=f"{selected_metal.upper()} by Day of Week",
                    labels={median_col: f"{selected_metal.upper()} (ng/m³)"}
                )
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Monthly Summary")
            monthly_df = outputs['Monthly Summary']
            if median_col in monthly_df.columns:
                fig = px.bar(
                    monthly_df,
                    x='month',
                    y=median_col,
                    color='site',
                    barmode='group',
                    title=f"{selected_metal.upper()} Monthly Summary",
                    labels={median_col: f"{selected_metal.upper()} (ng/m³)"}
                )
                st.plotly_chart(fig, use_container_width=True)
