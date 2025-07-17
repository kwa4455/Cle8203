import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from scipy.stats import kruskal, ttest_ind
import plotly.express as px 
from scipy import stats
from plotly.subplots import make_subplots


# --- Page Config ---
st.set_page_config(page_title="Air Quality Dashboard", layout="wide")

st.markdown("""
<style>

/* Universal styles */
* {
    font-family: 'Segoe UI', sans-serif;
    transition: all 0.2s ease-in-out;
}

/* =======================
   LIGHT MODE
======================= */
@media (prefers-color-scheme: light) {
    html, body, .stApp {
        background: url('https://source.unsplash.com/1600x900/?clouds,day') no-repeat center center fixed;
        background-size: cover;
        min-height: 100vh;
        backdrop-filter: blur(15px);
    }

    section.main > div {
        background: rgba(255, 255, 255, 0.55);
        color: #111;
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }

    section[data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.3);
        border-right: 1px solid rgba(0, 0, 0, 0.1);
    }

    input, textarea, select {
        background: rgba(255, 255, 255, 0.6) !important;
        color: #111 !important;
        border-radius: 8px !important;
        padding: 6px 10px !important;
        font-size: 1rem !important;
    }

    thead, tbody, tr, th, td {
        background: rgba(255, 255, 255, 0.4) !important;
        color: #111 !important;
        font-size: 0.95rem !important;
    }

    button[data-testid="baseButton-primary"] {
        background: linear-gradient(to right, #ff5f6d, #ffc371) !important;
        color: white !important;
        border-radius: 12px;
        font-weight: bold;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transition: transform 0.3s, box-shadow 0.3s;
    }

    button[data-testid="baseButton-primary"]:hover {
        transform: scale(1.03);
        background: linear-gradient(to right, #ff3d5a, #ffb347) !important;
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2);
    }

    div[data-testid="stAlert-info"] {
        background: rgba(0, 0, 0, 0.05);
        color: #111;
        border-radius: 10px;
    }

    a {
        color: #0056cc !important;
        text-decoration: underline;
        font-weight: 500;
    }
}

/* =======================
   DARK MODE
======================= */
@media (prefers-color-scheme: dark) {
    html, body, .stApp {
        background: url('https://source.unsplash.com/1600x900/?night,sky') no-repeat center center fixed;
        background-size: cover;
        min-height: 100vh;
        backdrop-filter: blur(8px);
        color: #f5f5f5 !important;
    }

    section.main > div {
        background: rgba(20, 20, 20, 0.85) !important;
        color: #f5f5f5 !important;
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.6);
    }

    section[data-testid="stSidebar"] {
        background: rgba(30, 30, 30, 0.95) !important;
        color: #ffffff !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    input, textarea, select {
        background: rgba(255, 255, 255, 0.07) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px !important;
        font-weight: 500;
        padding: 6px 10px !important;
        font-size: 1rem !important;
    }

    thead, tbody, tr, th, td {
        background: rgba(255, 255, 255, 0.05) !important;
        color: #ffffff !important;
        font-size: 0.95rem !important;
    }

    button[data-testid="baseButton-primary"] {
        background: linear-gradient(to right, #0099f7, #0652dd) !important;
        color: #fff !important;
        border-radius: 12px;
        font-weight: bold;
        box-shadow: 0 6px 14px rgba(0, 0, 0, 0.3);
        transition: transform 0.3s, box-shadow 0.3s;
    }

    button[data-testid="baseButton-primary"]:hover {
        transform: scale(1.03);
        background: linear-gradient(to right, #007be5, #0044aa) !important;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
    }

    div[data-testid="stAlert-info"] {
        background: rgba(255, 255, 255, 0.15);
        color: white !important;
        border-radius: 10px;
    }

    a {
        color: #66ccff !important;
        font-weight: 500;
        text-decoration: underline;
    }

    h1, h2, h3, h4, h5, h6, p, li, label, span {
        color: #f5f5f5 !important;
    }

    .element-container {
        background: rgba(0, 0, 0, 0.6) !important;
        color: #f5f5f5 !important;
    }
}

/* =======================
   SHARED STYLES
======================= */

section.main > div,
.stDataFrame, .stTable,
.element-container {
    border-radius: 16px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.25);
    padding: 1rem;
    transition: transform 0.3s, box-shadow 0.3s;
}

/* Hover effects for tables and graphs */
.stDataFrame:hover, .stTable:hover, .element-container:hover {
    transform: scale(1.02);
    box-shadow: 0 10px 36px rgba(0, 0, 0, 0.3);
}

/* Optional: lighter hover for performance */
button:hover, input:hover, select:hover, textarea:hover {
    box-shadow: 0 0 8px rgba(255, 255, 255, 0.2);
}

/* HR and spacing */
hr {
    border: none;
    border-top: 1px solid rgba(255, 255, 255, 0.2);
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

st.title("🌍 Air Quality Data Explorer")

def apply_glass_style(fig, theme=None, font_size=14):
    default_text_color = "#000000"
    if theme is None:
        theme = {"text": default_text_color}
    fig.update_layout(
        paper_bgcolor="rgba(255, 255, 255, 0.1)",
        plot_bgcolor="rgba(255, 255, 255, 0.1)",
        font=dict(color=theme.get("text", default_text_color), size=int(font_size)),
        margin=dict(l=40, r=40, t=60, b=40),
        title_font=dict(size=20, color=theme.get("text", default_text_color), family="Arial"),
        hoverlabel=dict(bgcolor="rgba(255,255,255,0.3)", font_size=13, font_family="Arial")
    )
    return fig

# Plotly table with theme

def plotly_table(df, font_size="16px", theme=None):
    if theme is None:
        theme = {"text": "#000000", "button": "#4CAF50"}

    headers = list(df.columns)
    cells = [df[col].astype(str).tolist() for col in headers]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=headers,
            fill_color=theme["button"],
            font=dict(color='white', size=int(font_size.replace("px", ""))),
            align='center',
            line_color='darkslategray',
            height=32
        ),
        cells=dict(
            values=cells,
            fill_color='rgba(255,255,255,0.4)',
            font=dict(color=theme["text"], size=int(font_size.replace("px", ""))),
            align='left',
            line_color='lightgray',
            height=28
        )
    )])

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=20, b=10)
    )
    return fig



def cleaned(df):
    # Parse and clean date
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['date'])  # Drop rows with invalid dates
    df['date'] = df['date'].dt.tz_localize(None)
    df = df.set_index('date')

    # Add time features
    df['year'] = df.index.year
    df['month'] = pd.Categorical(df.index.strftime('%b'),
                                 categories=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                                 ordered=True)
    df['dayofweek'] = pd.Categorical(df.index.day_name(),
                                     categories=['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                                                 'Friday', 'Saturday', 'Sunday'],
                                     ordered=True)

    return df

import plotly.graph_objects as go
import pandas as pd

def yearly_plot_bar(df, metal_sel):
    if metal_sel not in df.columns:
        return go.Figure(), pd.DataFrame()

    error_col = f"{metal_sel}_error"
    has_error = error_col in df.columns

    # Aggregation
    agg_funcs = {metal_sel: ['mean', 'std', 'median']}
    if has_error:
        agg_funcs[error_col] = ['mean', 'std', 'median']

    summary_data = (
        df.groupby(['site', 'year'])
        .agg(agg_funcs)
        .reset_index()
    )
    summary_data.columns = ['_'.join(col).strip('_') for col in summary_data.columns]
    summary_data['count'] = df.groupby(['site', 'year']).size().values
    summary_data = summary_data.round(3)
    summary_data['year'] = summary_data['year'].astype(str)

    year_colors = {
        "2018": "#008000",
        "2019": "#b30000",
        "2020": "blue",
        "2021": "yellow",
        "2022": "purple"
    }

    metal_limits = {
        "pb": 0.5,  # Ghana
        "cr": 12,   # US EPA
        "cd": 5     # EU AQS
    }

    limit_value = metal_limits.get(metal_sel.lower())
    metal_label = metal_sel.capitalize()

    fig = go.Figure()

    for year in summary_data['year'].unique():
        subset = summary_data[summary_data['year'] == year]
        fig.add_trace(go.Bar(
            x=subset['site'],
            y=subset.get(f'{metal_sel}_median', [0]),
            name=year,
            error_y=dict(
                type='data',
                array=subset.get(f'{error_col}_median', [0]) if has_error else None,
                visible=has_error
            ),
            marker_color=year_colors.get(year, 'gray'),
        ))

    # Add limit line and legend
    if limit_value is not None:
        # Add the visible shape line
        fig.add_shape(
            type="line",
            x0=-0.5, x1=len(summary_data['site'].unique()) - 0.5,
            y0=limit_value, y1=limit_value,
            line=dict(color="red", dash="solid"),
            xref="x", yref="y"
        )
        # Add a dummy trace for the legend
        fig.add_trace(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(color="red", dash="solid"),
            name=f"Standard limit ({limit_value})"
        ))

    for i in range(len(summary_data['site'].unique()) - 1):
        fig.add_vline(x=i + 0.5, line_dash="dash", line_color="black")

    unit = "μg/m³" if metal_sel.lower() == "al" else "ng/m³"

    fig.update_layout(
        barmode='group',
        title={
            'text': f" Annual Median Concentrations of {metal_label} by Site (2018–2022)",
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis_title="Site",
        yaxis_title=f"{metal_label} ({unit})",
        xaxis_tickangle=45,
        legend_title_text='Year',
        template="plotly_white",
        font=dict(size=12, family="Arial"),
        plot_bgcolor='white',
        margin=dict(t=80, b=100),
    )

    return fig, summary_data

def correlation_analysis(df, metals, selected_sites, title="Correlation Heatmap"):
    site_corrs = {}

    for site in df['site'].unique():
        if site not in selected_sites:
            continue
    
        site_df = df[df['site'] == site][metals]
        corr_matrix = site_df.corr(method='pearson')
        site_corrs[site] = corr_matrix

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            colorbar=dict(title="Correlation"),
            hovertemplate="x: %{x}<br>y: %{y}<br>Correlation: %{z:.2f}<extra></extra>",
        ))

        fig.update_layout(
            title=f"{title} - {site}",
            xaxis_title="Variables",
            yaxis_title="Variables",
            title_font=dict(size=16),
            xaxis=dict(tickangle=45, tickfont=dict(size=12)),
            yaxis=dict(tickfont=dict(size=12)),
            plot_bgcolor='white',
            height=600,
            width=600,
        )

        st.plotly_chart(fig, use_container_width=True)  # ✅ Display here

    return site_corrs  # Optional, if you need to use the matrices elsewhere



def plot_box_plot(df, metal):
    colors = {
        "Kaneshie First Light": "#ffff00", "Mallam Market": "green", "East Legon": "red",
        "Amasaman": "purple", "Tetteh Quarshie Roundabout": "orange",
        "Dansoman": "maroon", "North Industrial Area": "blue"
    }

    unit = "µg/m³" if metal.lower() == "al" else "ng/m³"
    fig = go.Figure()

    for site in df['site'].unique():
        site_data = df[df['site'] == site]

        fig.add_trace(go.Box(
            y=site_data[metal],
            name=site,
            boxpoints=False,              # disables scatter points (outliers)
            line=dict(color=colors.get(site, 'gray')),
            fillcolor=colors.get(site, 'lightgray'),
            marker_color=colors.get(site, 'gray'),
            opacity=0.6,
            showlegend=False
        ))

    fig.update_layout(
        title=f"{metal.upper()} ({unit}) by Site",
        yaxis_title=f"{metal.upper()} ({unit})",
        xaxis_title="Site",
        template="plotly_white",
        xaxis_tickangle=45,
        font=dict(size=12, family="Arial", color="black"),
        plot_bgcolor='white',
        margin=dict(t=50, b=100),
    )

    return fig




def kruskal_wallis_by_test(df, metals, site_column, n_bootstrap=1000, ci_level=0.95):

    results = []

    # Bootstrapping confidence interval for the median
    def bootstrap_ci(data, n_bootstrap, ci_level):
        if len(data) == 0:
            return np.nan, np.nan
        bootstrapped_medians = [np.median(np.random.choice(data, size=len(data), replace=True))
                                for _ in range(n_bootstrap)]
        lower_bound = np.percentile(bootstrapped_medians, (1 - ci_level) / 2 * 100)
        upper_bound = np.percentile(bootstrapped_medians, (1 + ci_level) / 2 * 100)
        return lower_bound, upper_bound

    for metal in metals:
        # Collect groups by site
        grouped_data = {
            site: df[df[site_column] == site][metal].dropna()
            for site in df[site_column].unique()
        }

        # Filter out empty groups
        non_empty_groups = [data for data in grouped_data.values() if len(data) > 0]

        # Perform test only if at least 2 non-empty groups
        if len(non_empty_groups) >= 2:
            statistic, p_value = stats.kruskal(*non_empty_groups)
        else:
            statistic, p_value = np.nan, np.nan

        # Confidence intervals for each site group
        ci_dict = {
            site: (f"[{lower:.2f}, {upper:.2f}]" if not np.isnan(lower) else "N/A")
            for site, data in grouped_data.items()
            for lower, upper in [bootstrap_ci(data, n_bootstrap, ci_level)]
        }

        # Store result
        results.append({
            'Variable': metal.capitalize(),
            'Statistic': statistic,
            'p_value': p_value,
            'df': len(grouped_data) - 1,
            **ci_dict
        })

    return pd.DataFrame(results)


# Function to aggregate data by month or dayofweek
def aggregate_metals(df, time_col):
    metals = ["cd", "cr", "hg", "al", "as", "mn", "pb","co","cu","ni","zn"]
    agg_funcs = {col: ['mean', 'std', 'median'] for col in metals}
    summary = df.groupby(['site', time_col]).agg(agg_funcs).reset_index()
    
    # Flatten MultiIndex columns
    summary.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in summary.columns]
    return summary

# Time Variation Plotting Function
def timeVariation(df, pollutants=["pb"], statistic="median", colors=None):
    if colors is None:
        colors = px.colors.qualitative.Plotly

    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=["Time Variation by Month", "Time Variation by Day of Week"],
        shared_yaxes=True
    )

    df_month_agg = aggregate_metals(df, time_col="month")
    for i, pollutant in enumerate(pollutants):
        col_name = f"{pollutant}_{statistic}"
        if col_name not in df_month_agg.columns:
            raise ValueError(f"'{col_name}' not found in DataFrame.")
        df_clean = df_month_agg.dropna(subset=[col_name])
        fig.add_trace(go.Scatter(
            x=df_clean['month'],
            y=df_clean[col_name],
            mode='lines+markers',
            name=f'{pollutant} by Month',
            line=dict(color=colors[i % len(colors)]),
        ), row=1, col=1)

    df_dayofweek_agg = aggregate_metals(df, time_col="dayofweek")
    for i, pollutant in enumerate(pollutants):
        col_name = f"{pollutant}_{statistic}"
        if col_name not in df_dayofweek_agg.columns:
            raise ValueError(f"'{col_name}' not found in DataFrame.")
        df_clean = df_dayofweek_agg.dropna(subset=[col_name])
        fig.add_trace(go.Scatter(
            x=df_clean['dayofweek'],
            y=df_clean[col_name],
            mode='lines+markers',
            name=f'{pollutant} by Day of Week',
            line=dict(color=colors[i % len(colors)]),
        ), row=1, col=2)

    fig.update_layout(
        title=f"Time Variation of Pollutants ({statistic.capitalize()})",
        template="plotly",
        showlegend=True,
        height=500
    )

    fig.update_xaxes(title_text="Month", type="category", row=1, col=1)
    fig.update_xaxes(title_text="Day", type="category", row=1, col=2)

    # Unified Y-axis label
    if any(p.lower() == "al" for p in pollutants):
        y_label = "Concentration (µg/m³)"
    else:
        y_label = "Concentration (ng/m³)"
    fig.update_yaxes(title_text=y_label, row=1, col=1)
    fig.update_yaxes(title_text=y_label, row=1, col=2)

    return fig




uploaded_files = st.file_uploader("Upload CSV files", accept_multiple_files=True, type=["csv"])
if not uploaded_files:
    st.warning("Please upload at least one CSV file.")
    st.stop()

# Define required columns
required_columns = [
    'date', 'site', 'id', "cd", "cr", "hg", "al", "as", "mn", "pb", "co", "cu", "ni", "zn",
    "cd_error", "cr_error", "hg_error", "al_error", "as_error", "mn_error", "pb_error", 
    "co_error", "cu_error", "ni_error", "zn_error"
]

dataframes = []
file_names = []

for uploaded_file in uploaded_files:
    try:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip().str.lower()
        missing = set(required_columns) - set(df.columns)
        if missing:
            st.warning(f"File '{uploaded_file.name}' is missing required columns: {', '.join(sorted(missing))}")
            continue
        df = df[required_columns].copy()
        file_names.append(uploaded_file.name)
        df_cleaned = cleaned(df)
        dataframes.append(df_cleaned)
        file_names.append(uploaded_file.name)
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {e}")
        st.stop()

# Sidebar filters

# Identify metal columns (exclude non-metal ones)
non_metal_columns = {'site', 'year', 'dayofweek', 'month' 'date',"cd_error", "cr_error", "hg_error", "al_error", "as_error", "mn_error", "pb_error","co_error","cu_error","ni_error","zn_error"}
all_columns = set().union(*[df.columns for df in dataframes])
metal_columns = ["cd", "cr", "hg", "al", "as", "mn", "pb","co","cu","ni","zn"]
error_columns = [f"{m}_error" for m in metal_columns]
errors = sorted([col for col in all_columns if col.lower() in error_columns])
non_metal_columns = {'site', 'year', 'dayofweek', 'month', 'date'}
metals = sorted([col for col in all_columns if col.lower() in metal_columns])
sites = sorted(
    set().union(*[df['site'].unique() for df in dataframes if 'site' in df.columns])
)




    
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Yearly Trends", "📊 Correlation Plots", "📐 Violin Plot",
    "🔗 Kruskal-Wallis Test", "📉 Time Variation"
])

# --- Tab 1: Yearly Trends ---
with tab1:
    for df, name in zip(dataframes, file_names):
        st.subheader(f"Yearly Trend: {name}")
        metals = [m for m in metal_columns if m in df.columns]
        if not metals:
            st.warning(f"No metals found in {name}. Skipping.")
            continue

        selected_metal = st.selectbox(
            f"Select metal for {name}", metals, key=f"metals_{name}"
        )

        
        fig, summary_df = yearly_plot_bar(df, selected_metal)
        fig = apply_glass_style(fig)
        summary = plotly_table(summary_df)
        st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(summary, use_container_width=True)

# --- Tab 2: Correlation Analysis ---
with tab2:
    for df, name in zip(dataframes, file_names):
        st.subheader(f"Correlation: {name}")
        metals = [m for m in metal_columns if m in df.columns]
        site_sel = st.selectbox(
            f"Sites for {name}", sites, key=f"site_corr_{name}"
        )
        df_sub = df[df['site'] == site_sel]
        correlation_analysis(df_sub, metals, site_sel, title=name)

# --- Tab 3: Box Plot (Median Only, No Outliers) ---
with tab3:
    for df, name in zip(dataframes, file_names):
        st.subheader(f"Box Plot: {name}")
        metals = [m for m in metal_columns if m in df.columns]
        metal_sel = st.selectbox(f"Metal for {name}", metals, key=f"metal2_{name}")
        fig = plot_box_plot(df, metal_sel)  # <-- new function
        fig = apply_glass_style(fig)
        st.plotly_chart(fig, use_container_width=True)


# --- Tab 4: Kruskal-Wallis Test ---
with tab4:
    for df, name in zip(dataframes, file_names):
        st.subheader(f"Kruskal-Wallis Test: {name}")
        
        sites = sorted(df['site'].dropna().unique())
        metals = [m for m in metal_columns if m in df.columns]

        selected_sites = st.multiselect(
            f"Sites for {name}", sites,  key=f"site3_{name}"
        )

        df_sub = df[df['site'].isin(selected_sites)]

        kruskal_df = kruskal_wallis_by_test(
            df_sub, metals, site_column="site", n_bootstrap=1000, ci_level=0.95
        )

        st.write("Kruskal-Wallis Test Results (includes bootstrapped 95% CI for group medians):")
        fig = plotly_table(kruskal_df)
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 5: Time Variation ---
with tab5:
    for df, name in zip(dataframes, file_names):
        st.subheader(f"Time Variation: {name}")

        try:
            df = cleaned(df)
            except Exception as e:
                st.warning(f"Could not clean data for {name}: {e}")
                continue
        
        sites = sorted(df['site'].unique())
        metals = [m for m in metal_columns if m in df.columns]
        
        site_sel = st.selectbox(
            f"Sites for {name}", sites,  key=f"site5_{name}"
        )
        metal_sel = st.selectbox(
            f"Metals for {name}", metals,  key=f"metal5_{name}"
        )

        statistic = st.selectbox(
            f"Select statistic for {name}",
            options=["mean", "std", "median"],
            index=2,
            key=f"stat5_{name}"
        )

        df_sub = df[df['site'] == site_sel].copy()
        if not df_sub.empty and metal_sel:
            try:
                fig = timeVariation(df_sub, pollutants=metal_sel, statistic=statistic)
                fig = apply_glass_style(fig)
                st.plotly_chart(fig)
            except ValueError as e:
                st.warning(f"Plotting error: {e}")
        else:
            st.info("Please select at least one site and one metal to generate the plot.")
