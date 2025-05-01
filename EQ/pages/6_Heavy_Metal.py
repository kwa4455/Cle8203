import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.stats import kruskal, ttest_ind
from sklearn.linear_model import TheilSenRegressor

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

def cleaned(df):
    

    # Normalize column names: lowercase, strip spaces
    df.columns = [col.strip().lower() for col in df.columns]

    # Map common alias variations to standard names
    column_aliases = {
        'date': ['date', 'sampling date', 'datetime'],
        'id': ['id', 'station id', 'site id'],
        'cd(ng/m3)': ['cd(ng/m3)', 'cadmium', 'cd'],
        'cd_error': ['cd_error', 'cd err'],
        'cr(ng/m3)': ['cr(ng/m3)', 'chromium', 'cr'],
        'cr_error': ['cr_error', 'cr err'],
        'hg(ng/m3)': ['hg(ng/m3)', 'mercury', 'hg'],
        'hg_error': ['hg_error', 'hg err'],
        'al(ug/m3)': ['al(ug/m3)', 'aluminium', 'al'],
        'al_error': ['al_error', 'al err'],
        'as(ng/m3)': ['as(ng/m3)', 'arsenic', 'as'],
        'as_error': ['as_error', 'as err'],
        'mn(ng/m3)': ['mn(ng/m3)', 'manganese', 'mn'],
        'mn_error': ['mn_error', 'mn err'],
        'pb(ng/m3)': ['pb(ng/m3)', 'lead', 'pb'],
        'pb_error': ['pb_error', 'pb err'],
        'site': ['site', 'site_location', 'source']
    }

    # Reverse alias mapping: variant -> standard
    alias_lookup = {alias.lower(): std for std, aliases in column_aliases.items() for alias in aliases}

    # Rename columns using alias mapping
    df.rename(columns=lambda x: alias_lookup.get(x.lower(), x), inplace=True)

    # Ensure required columns exist
    if 'date' not in df.columns or 'id' not in df.columns:
        raise ValueError("Missing required columns: 'date' or 'id'")

    # Parse 'date' column safely
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['date'])  # Drop rows where 'date' could not be parsed

    # Select only known relevant columns (drop unexpected extras)
    required_columns = list(column_aliases.keys())
    df = df[[col for col in required_columns if col in df.columns]].copy()

    # Drop empty rows/columns
    df = df.dropna(axis=1, how='all').dropna()

    # Extract datetime features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.to_period('M').astype(str)
    df['day'] = df['date'].dt.date
    df['dayofweek'] = df['date'].dt.day_name()
    df['weekday_type'] = df['date'].dt.weekday.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
    df['season'] = df['date'].dt.month.apply(lambda x: 'Harmattan' if x in [12, 1, 2] else 'Non-Harmattan')

    # Final column selection
    columns_to_select = [
        'site', 'day', 'year', 'month', 'dayofweek', 'season',
        'cd(ng/m3)', 'cd_error',
        'cr(ng/m3)', 'cr_error',
        'hg(ng/m3)', 'hg_error',
        'al(ug/m3)', 'al_error',
        'as(ng/m3)', 'as_error',
        'mn(ng/m3)', 'mn_error',
        'pb(ng/m3)', 'pb_error'
    ]
    df = df[[col for col in columns_to_select if col in df.columns]]
    return df
def compute_aggregates_all_metals(df, label):
    aggregates = {}

    # Define groups of columns
    pollutant_cols = ['cd(ng/m3)', 'cr(ng/m3)', 'hg(ng/m3)', 'al(ug/m3)', 'as(ng/m3)', 'mn(ng/m3)', 'pb(ng/m3)']
    error_cols = ['cd_error', 'cr_error', 'hg_error', 'al_error', 'as_error', 'mn_error', 'pb_error']

    # Combine for convenience
    all_cols = pollutant_cols + error_cols

    # Grouping levels
    groupings = {
        'Monthly': 'month',
        'Yearly': 'year',
        'Day of Week': 'dayofweek',
        'Weekday Type': 'weekday_type',
        'Season': 'season'
    }

    # Loop through each time grouping
    for name, group_col in groupings.items():
        # Perform aggregation: mean, median, std
        grouped = df.groupby([group_col, 'site'])[all_cols].agg(['mean', 'median', 'std']).round(3)

        # Flatten the MultiIndex columns
        grouped.columns = ['_'.join([col[0], col[1]]) for col in grouped.columns]
        grouped = grouped.reset_index()

        # Save in dictionary
        aggregates[f'{label} - {name} Stats'] = grouped

    return aggregates
def calculate_min_max(df):
    pollutant_cols = ['cd(ng/m3)', 'cr(ng/m3)', 'hg(ng/m3)', 
                      'al(ug/m3)', 'as(ng/m3)', 'mn(ng/m3)', 'pb(ng/m3)']
    
    valid_pollutants = [p for p in pollutant_cols if p in df.columns]

    # Compute daily statistics
    daily_avg = (
        df.groupby(['site', 'day', 'year', 'month'])[valid_pollutants]
        .agg(['mean', 'median', 'std'])
        .reset_index()
        .round(3)
    )

    # Flatten MultiIndex columns
    daily_avg.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in daily_avg.columns]

    # Prepare min/max aggregation dictionary
    agg_dict = {}
    for pollutant in valid_pollutants:
        mean_col = f"{pollutant}_mean"
        if mean_col in daily_avg.columns:
            agg_dict[f"{pollutant}_daily_avg_max"] = (mean_col, lambda x: round(x.max(), 3))
            agg_dict[f"{pollutant}_daily_avg_min"] = (mean_col, lambda x: round(x.min(), 3))

    # Aggregate by year and site
    df_min_max = daily_avg.groupby(['year', 'site'], as_index=False).agg(**agg_dict)

    return df_min_max

def calculate_kruskal_wallis(df, group_col='site'):
    pollutant_cols = ['cd(ng/m3)', 'cr(ng/m3)', 'hg(ng/m3)', 
                      'al(ug/m3)', 'as(ng/m3)', 'mn(ng/m3)', 'pb(ng/m3)']

    required_cols = ['site', 'season', 'weekday_type'] + pollutant_cols
    p_filtered = df[required_cols].dropna(subset=pollutant_cols, how='all')
    
    
    results = []

    for pollutant in pollutant_cols:
        if pollutant in df.columns:
            grouped_data = [group[pollutant].dropna().values for _, group in df.groupby(group_col)]
            
            # Kruskal-Wallis requires at least 2 groups with more than one observation
            if len(grouped_data) >= 2 and all(len(g) > 1 for g in grouped_data):
                stat, p_value = kruskal(*grouped_data)
                results.append({
                    'Pollutant': pollutant,
                    'Grouping': group_col,
                    'H-statistic': round(stat, 4),
                    'p-value': round(p_value, 4),
                    'Significant (p<0.05)': p_value < 0.05
                })

    return p_filtered(results)

    
# --- Upload Data ---
uploaded_file = st.file_uploader("Upload your air quality dataset (.csv)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = cleaned(df)

    value_cols = [col for col in df.columns if col.endswith('(ng/m3)') or col.endswith('(ug/m3)')]
    
    dfs[label] = df
        site_options.update(df['site'].unique())
        year_options.update(df['year'].unique())
    with st.sidebar:
        selected_years = st.multiselect("📅 Filter by Year", sorted(year_options))
        selected_sites = st.multiselect("🏢 Filter by Site", sorted(site_options))

    df = df[df['site'].isin(selected_sites) & df['year'].isin(selected_years)]


    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Trends", "📊 Box & Bar Plots", "📐 Kruskal & T-Test", "🔗 Correlation", "📉 Theil-Sen Trend"])

    with tabs[0]:  # Aggregated Means
        st.header("📊 Aggregated Means(Mean, Median, Std)")
        for label, df in dfs.items():
            st.subheader(f"Dataset: {label}")
            site_in_tab = st.multiselect(f"Select Site(s) for {label}", sorted(df['site'].unique()), key=f"site_agg_{label}")
            filtered_df = df.copy()
            if selected_years:
                filtered_df = filtered_df[filtered_df['year'].isin(selected_years)]
            if site_in_tab:
                filtered_df = filtered_df[filtered_df['site'].isin(site_in_tab)]
            selected_pollutants = ['cd(ng/m3)', 'cr(ng/m3)', 'hg(ng/m3)', 'al(ug/m3)', 'as(ng/m3)', 'mn(ng/m3)', 'pb(ng/m3)']
            error_pollutants = ['cd_error', 'cr_error', 'hg_error', 'al_error', 'as_error', 'mn_error', 'pb_error']
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
                ('Monthly Avg', ['month', 'site']),
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
                
                with st.expander(f"📈 Show Charts for {agg_label}", expanded=none):
                    for pollutant in selected_display_pollutants:
                        if pollutant in filtered_df.columns and f"{pollutant}_std" in merged_df.columns:
                            st.plotly_chart(
                                px.bar(
                                    merged_df,
                                    x=group_keys[0],
                                    y=f"{pollutant}_mean",
                                    color='site',
                                    error_y=f"{pollutant}_std",
                                    barmode="group",
                                    title=f"{pollutant} Distribution by {level_name}"
                                ),
                                use_container_width=True
                            )
                    for pollutant in selected_display_pollutants:
                        if pollutant in filtered_df.columns:
                            st.plotly_chart(
                                px.box(
                                   filtered_df,
                                   x=group_keys[0],
                                   y=pollutant,
                                   color='site',
                                   title=f"{pollutant} Distribution by {level_name}"
                                ),
                                use_container_width=True
                            )
                    for pollutant in selected_display_pollutants:
                        if pollutant in filtered_df.columns:
                            st.plotly_chart( 
                                px.box(  
                                   filtered_df,
                                   x=group_keys[0],
                                   y=pollutant,
                                   color='site',
                                   title=f"{pollutant} Distribution by {level_name}"
                               ),
                               use_container_width=True
                            )
                    for pollutant in selected_display_pollutants:
                        if f"{pollutant}_mean" in merged_df.columns:
                            st.plotly_chart(
                                px.line(
                                   x=group_keys[0],
                                   y=f"{pollutant}_mean",
                                   color='site',
                                   title=f"{pollutant} Mean Trend by {level_name}"
                                ),
                                use_container_width=True
                            )   
    
