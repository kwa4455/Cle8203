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
    alias_lookup = {}
    for std_name, aliases in column_aliases.items():
        for alias in aliases:
            alias_lookup[alias.lower()] = std_name

    # Rename columns using alias mapping
    df.rename(columns=lambda x: alias_lookup.get(x.lower(), x), inplace=True)

    # Ensure required columns exist
    if 'date' not in df.columns or 'id' not in df.columns:
        raise ValueError("Missing required columns: 'date' or 'id'")

   

    required_columns = list(column_aliases.keys())
    df = df[[col for col in required_columns if col in df.columns]].copy()
    df = df.dropna(axis=1, how='all').dropna()


    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.to_period('M').astype(str)
    df['day'] = df['date'].dt.date
    df['dayofweek'] = df['date'].dt.day_name()
    df['weekday_type'] = df['date'].dt.weekday.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')
    df['season'] = df['date'].dt.month.apply(lambda x: 'Harmattan' if x in [12, 1, 2] else 'Non-Harmattan')

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
def parse_dates(df):
    for col in df.columns:
        if 'date' in col.lower():  # Focus only on columns with "date" in the name
            try:
                # Parse the date column to datetime
                df[col] = pd.to_datetime(df[col], format='%d-%b-%y', errors='coerce')
                # Drop rows where the parsed date is NaT
                df = df.dropna(subset=[col])
            except Exception as e:
                print(f"Error parsing column {col}: {e}")
                continue
    return df
# --- Upload Data ---
uploaded_file = st.file_uploader("Upload your air quality dataset (.csv)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    df = cleaned(df)
    value_cols = [col for col in df.columns if col.endswith('(ng/m3)') or col.endswith('(ug/m3)')]


    # --- Sidebar Filters ---
    st.sidebar.header("🔎 Filters")
    selected_sites = st.sidebar.multiselect("Select Sites", options=df['site'].unique(), default=df['site'].unique())
    selected_years = st.sidebar.multiselect("Select Years", options=sorted(df['year'].unique()), default=sorted(df['year'].unique()))
    selected_metals = st.sidebar.multiselect("Select Metals", options=value_cols, default=value_cols)

    df = df[df['site'].isin(selected_sites) & df['year'].isin(selected_years)]

    # --- Summary Function ---
    def summarize_stats(df, group_by):
        if group_by is None:
            return df.groupby('site')[selected_metals].agg(['mean', 'median', 'std'])
        else:
            return df.groupby(['site', group_by])[selected_metals].agg(['mean', 'median', 'std'])
        

    df_all = summarize_stats(df, None)
    df_year = summarize_stats(df, 'year')
    df_month = summarize_stats(df, 'month')
    df_dow = summarize_stats(df, 'dayofweek')

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Trends", "📊 Box & Bar Plots", "📐 Kruskal & T-Test", "🔗 Correlation", "📉 Theil-Sen Trend"])

    # --- Plotly Trend Plot ---
    with tab1:
        stat = st.selectbox("Statistic", ['mean', 'median'])
        metal = st.selectbox("Select Metal", selected_metals)

        def plot_line_trend(df_summary, group_by):
            col = (metal, stat)
            df_plot = df_summary.copy()
            df_plot[group_by] = df_plot[group_by].astype(str)

            fig = px.line(
                df_plot, x=group_by, y=col, color='site', markers=True,
                title=f"{metal} - {stat.title()} Trend by {group_by.capitalize()}"
            )
            return fig

        st.plotly_chart(plot_line_trend(df_year, 'year'), use_container_width=True)
        st.plotly_chart(plot_line_trend(df_month, 'month'), use_container_width=True)

    # --- Box and Bar Plots ---
    with tab2:
        st.subheader("Box Plot by Year")
        st.plotly_chart(px.box(df, x='year', y=metal, color='site', points='all'), use_container_width=True)

        st.subheader("Bar Plot by Day of Week")
        st.plotly_chart(
            px.bar(df_dow, x='dayofweek', y=(metal, stat), color='site', barmode='group'),
            use_container_width=True
        )

    # --- Kruskal and T-test ---
    with tab3:
        st.subheader("Kruskal-Wallis Test")

        def kruskal_by_year(df, metal):
            results = []
            years = df['year'].unique()
            for year in years:
                subset = df[df['year'] == year]
                groups = [g[metal].dropna().values for _, g in subset.groupby('site') if not g[metal].isnull().all()]
                if len(groups) > 1:
                    stat, p = kruskal(*groups)
                    results.append({'year': year, 'statistic': stat, 'pvalue': p, 'df': len(groups)-1})
            all_groups = [g[metal].dropna().values for _, g in df.groupby('site') if not g[metal].isnull().all()]
            if len(all_groups) > 1:
                stat, p = kruskal(*all_groups)
                results.append({'year': 'All', 'statistic': stat, 'pvalue': p, 'df': len(all_groups)-1})
            return pd.DataFrame(results)

        kruskal_df = kruskal_by_year(df, metal)
        st.dataframe(kruskal_df)

        st.subheader("T-test: Harmattan vs Non-Harmattan")
        df['month_num'] = df['date'].dt.month
        harmattan = df[df['month_num'].isin([12,1,2])][metal]
        non_harmattan = df[~df['month_num'].isin([12,1,2])][metal]
        t_stat, p_val = ttest_ind(harmattan.dropna(), non_harmattan.dropna(), equal_var=False)
        st.write(f"**T-statistic**: {t_stat:.4f}, **P-value**: {p_val:.4f}")

    # --- Correlation Heatmap ---
    with tab4:
        st.subheader("Correlation Between Metals")
        corr_df = df[selected_metals].corr()
        fig = ff.create_annotated_heatmap(
            z=corr_df.values.round(2), x=list(corr_df.columns), y=list(corr_df.index),
            colorscale='Viridis', showscale=True
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- Theil-Sen Trend Analysis ---
    with tab5:
        st.subheader("Theil-Sen Trend Analysis")

        def theil_sen_trend(df, metal):
            results = []
            for site, group in df.groupby("site"):
                group = group.sort_values("day")
                X = group['day'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
                y = group[metal].values
                model = TheilSenRegressor()
                model.fit(X, y)
                results.append({'site': site, 'slope': model.coef_[0], 'intercept': model.intercept_})
            return pd.DataFrame(results)

        trend_df = theil_sen_trend(df, metal)
        st.dataframe(trend_df)

        for site in df['site'].unique():
            site_df = df[df['site'] == site].sort_values('day')
            fig = px.scatter(site_df, x='day', y=metal, title=f"{metal} Trend - {site}")
            X = site_df['day'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
            y = site_df[metal].values
            model = TheilSenRegressor().fit(X, y)
            y_pred = model.predict(X)
            fig.add_trace(go.Scatter(x=site_df['day'], y=y_pred, mode='lines', name='Trend'))
            st.plotly_chart(fig, use_container_width=True)





