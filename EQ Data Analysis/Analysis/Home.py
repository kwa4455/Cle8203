import streamlit as st
import streamlit.components.v1 as components


# Page Configuration
st.set_page_config(
    page_title="Air Quality Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)



# ----------------------------
# SESSION STATE DEFAULTS
# ----------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "Light"
if "font_size" not in st.session_state:
    st.session_state.font_size = "Medium"

# ----------------------------
# SIDEBAR SETTINGS
# ----------------------------
st.sidebar.header("🎨 Appearance Settings")

# Theme toggle
dark_mode = st.sidebar.toggle("🌙 Dark Mode", value=(st.session_state.theme == "Dark"))
st.session_state.theme = "Dark" if dark_mode else "Light"

# Font size radio
font_choice = st.sidebar.radio("🔠 Font Size", ["Small", "Medium", "Large"], index=["Small", "Medium", "Large"].index(st.session_state.font_size))
st.session_state.font_size = font_choice

# Reset to defaults
if st.sidebar.button("🔄 Reset to Defaults"):
    st.session_state.theme = "Light"
    st.session_state.font_size = "Medium"
    st.rerun()

# ----------------------------
# THEME AND FONT SETTINGS
# ----------------------------
theme = st.session_state.theme
font_size = st.session_state.font_size

font_map = {
    "Small": "14px",
    "Medium": "16px",
    "Large": "18px"
}

if theme == "Light":
    background = "linear-gradient(135deg, #e0f7fa, #ffffff)"
    text_color = "#004d40"
    button_color = "#00796b"
    button_hover = "#004d40"
else:
    background = "linear-gradient(135deg, #263238, #37474f)"
    text_color = "#e0f2f1"
    button_color = "#26a69a"
    button_hover = "#00897b"

# ----------------------------
# CUSTOM CSS INJECTION
# ----------------------------
st.markdown(
    f"""
    <style>
    body {{
        background: {background};
        background-attachment: fixed;
    }}

    .stApp {{
        background-color: transparent;
        animation: fadeIn 1s ease-in;
    }}

    html, body, [class*="css"] {{
        font-family: 'Segoe UI', 'Roboto', sans-serif;
        font-size: {font_map[font_size]};
        color: {text_color};
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }}

    h1, h2, h3 {{
        font-weight: bold;
        color: {text_color};
    }}

    div.stButton > button {{
        background-color: {button_color};
        color: white;
        padding: 0.5em 1.5em;
        border: none;
        border-radius: 8px;
        font-size: {font_map[font_size]};
        font-weight: bold;
        transition: background-color 0.3s ease;
    }}

    div.stButton > button:hover {{
        background-color: {button_hover};
        cursor: pointer;
    }}

    .stTextInput input, .stTextArea textarea {{
        border: 2px solid {text_color};
        border-radius: 8px;
        padding: 0.5em;
    }}

    .stSelectbox div[data-baseweb="select"] > div {{
        border: 2px solid {text_color};
        border-radius: 8px;
    }}

    .stSlider > div[data-baseweb="slider"] > div {{
        background: {text_color};
    }}

    .stFileUploader label {{
        background-color: {text_color};
        color: white;
        padding: 0.5em;
        border-radius: 8px;
        font-weight: bold;
    }}

    .stCheckbox > label {{
        font-weight: bold;
        color: {text_color};
    }}

    .stTabs [role="tablist"] > div {{
        border: 2px solid {text_color};
        border-radius: 10px;
        padding: 0.25em;
        margin-bottom: 1em;
    }}

    .stTabs [role="tab"] {{
        color: {text_color};
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1em;
        transition: all 0.2s ease-in-out;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: {text_color};
        color: white;
    }}

    .st-expander {{
        border: 2px solid {text_color} !important;
        border-radius: 10px !important;
    }}

    .st-expander summary {{
        color: {text_color};
        font-weight: bold;
    }}

    @keyframes fadeIn {{
        0% {{ opacity: 0; transform: translateY(20px); }}
        100% {{ opacity: 1; transform: translateY(0); }}
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# MAIN UI DEMO
# ----------------------------
st.title("🎛️ Stylish Streamlit App")
st.write(f"Theme: **{theme}**, Font Size: **{font_size}**")
st.button("Test Button")



# -------- Sidebar Content --------
with st.sidebar:
    try:
        st.image("epa-logo.png", width=150)
    except:
        pass

    st.markdown("### 🧑‍💻 Developer Information")
    st.markdown("""
    - **Developed by:** Clement Mensah Ackaah
    - **Email:** clement.ackaah@epa.gov.gh / clementackaah70@gmail.com
    - **GitHub:** [Visit GitHub](https://github.com/kwa4455)
    - **LinkedIn:** [Visit LinkedIn](https://www.linkedin.com/in/clementmensahackaah)
    - **Project Repo:** [Air Quality Dashboard](https://github.com/kwa4455/air-quality-analysis-dashboard)
    """)

# -------- Welcome Banner --------
components.html(
    """
    <div style="background: linear-gradient(90deg, #00bcd4, #009688); 
                padding: 30px; 
                border-radius: 12px; 
                color: white; 
                text-align: center; 
                font-size: 42px; 
                font-weight: bold;
                animation: fadeIn 2s ease-in-out;">
        👋 Welcome to the Air Quality Dashboard!
    </div>

    <style>
    @keyframes fadeIn {
        0% {opacity: 0;}
        100% {opacity: 1;}
    }
    </style>
    """,
    height=120
)

st.markdown("## 🌍 About the Dashboard")
st.markdown("""
### 📈 Air Quality Analysis Tool
Upload, visualize, and monitor air quality data collected from:

- 🏛️ Reference Grade Instruments
- 🛰️ Quant AQ Monitors
- 🌡️ Gravimetric Samplers
- 🧪 Clarity Sensors
- 📟 AirQo Devices

Get insights, analyze seasonal trends, and make informed decisions!
""")
# -------- AQI Education Section (Ultra Fancy Version) --------
st.markdown("---")
st.markdown("## 📚 Understanding AQI (Air Quality Index)")

# Custom CSS for styling
st.markdown("""
<style>
.aqi-card {
  background: linear-gradient(to right, #e0f7fa, #ffffff);
  border: 2px solid #00acc1;
  border-radius: 15px;
  padding: 20px;
  transition: 0.3s;
  box-shadow: 2px 2px 10px rgba(0, 172, 193, 0.2);
  margin-bottom: 20px;
}
.aqi-card:hover {
  transform: scale(1.02);
  box-shadow: 4px 4px 20px rgba(0, 172, 193, 0.4);
}

.aqi-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 15px;
}
.aqi-table th, .aqi-table td {
  border: 1px solid #00acc1;
  padding: 8px;
  text-align: center;
}
.aqi-table th {
  background-color: #00acc1;
  color: white;
}
.fade-in {
  animation: fadeIn 2s ease-in;
}
@keyframes fadeIn {
  from {opacity: 0;}
  to {opacity: 1;}
}

/* Colored backgrounds for AQI levels */
.good {background-color: #00e400; color: white;}
.moderate {background-color: #ffff00; color: black;}
.sensitive {background-color: #ff7e00; color: white;}
.unhealthy {background-color: #ff0000; color: white;}
.veryunhealthy {background-color: #8f3f97; color: white;}
.hazardous {background-color: #7e0023; color: white;}
</style>
""", unsafe_allow_html=True)

# Main Card
st.markdown("""
<div class="aqi-card fade-in">
<p>The <strong>Air Quality Index (AQI)</strong> measures the quality of air and provides important health-related information. It helps you understand when to take action to protect your health!</p>

<h4>🧮 How AQI is Calculated</h4>
<ul>
<li>Each major pollutant (PM₂.₅, PM₁₀, O₃, CO, SO₂, NO₂) gets its own index.</li>
<li>The final AQI is the <strong>highest</strong> individual pollutant index.</li>
</ul>

<h4>🔢 Basic AQI Formula</h4>
<p style="text-align:center;">
<em>
AQI = ((I<sub>high</sub> - I<sub>low</sub>) / (C<sub>high</sub> - C<sub>low</sub>)) × (C - C<sub>low</sub>) + I<sub>low</sub>
</em>
</p>

</div>
""", unsafe_allow_html=True)

# Expanders for AQI Levels (colored)
with st.expander("📗 Good (0-50)"):
    st.markdown('<div class="good"><b>Air quality is satisfactory and poses little or no risk.</b></div>', unsafe_allow_html=True)

with st.expander("📒 Moderate (51-100)"):
    st.markdown('<div class="moderate"><b>Air quality is acceptable; some pollutants may be a concern for a small number of sensitive individuals.</b></div>', unsafe_allow_html=True)

with st.expander("📙 Unhealthy for Sensitive Groups (101-150)"):
    st.markdown('<div class="sensitive"><b>Members of sensitive groups may experience health effects. The general public is not likely to be affected.</b></div>', unsafe_allow_html=True)

with st.expander("📕 Unhealthy (151-200)"):
    st.markdown('<div class="unhealthy"><b>Everyone may begin to experience health effects; sensitive groups may experience more serious effects.</b></div>', unsafe_allow_html=True)

with st.expander("📓 Very Unhealthy (201-300)"):
    st.markdown('<div class="veryunhealthy"><b>Health warnings of emergency conditions. The entire population is more likely to be affected.</b></div>', unsafe_allow_html=True)

with st.expander("📘 Hazardous (301-500)"):
    st.markdown('<div class="hazardous"><b>Health alert: everyone may experience more serious health effects. Emergency conditions.</b></div>', unsafe_allow_html=True)

# Fancy Table
st.markdown("""
<div class="aqi-card fade-in">
<h4>📊 AQI Categories Summary</h4>
<table class="aqi-table">
<thead>
<tr>
<th>AQI Range</th>
<th>Category</th>
<th>Health Effects</th>
</tr>
</thead>
<tbody>
<tr>
<td>0-50</td>
<td>Good</td>
<td>Little or no risk.</td>
</tr>
<tr>
<td>51-100</td>
<td>Moderate</td>
<td>Acceptable, slight concern for sensitive individuals.</td>
</tr>
<tr>
<td>101-150</td>
<td>Unhealthy for Sensitive Groups</td>
<td>Health effects possible for sensitive groups.</td>
</tr>
<tr>
<td>151-200</td>
<td>Unhealthy</td>
<td>Everyone may experience health effects.</td>
</tr>
<tr>
<td>201-300</td>
<td>Very Unhealthy</td>
<td>Health warnings for the entire population.</td>
</tr>
<tr>
<td>301-500</td>
<td>Hazardous</td>
<td>Serious health effects for everyone.</td>
</tr>
</tbody>
</table>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# -------- Quick Links --------
st.markdown("### 🔗 Quick Links")
st.markdown(
    """
    <div style="display: flex; gap: 20px; flex-wrap: wrap;">
        <a href="https://www.epa.gov.gh/" target="_blank" style="padding: 10px 20px; background: #00796b; color: white; border-radius: 8px; text-decoration: none;">🌐 EPA Website</a>
        <a href="https://www.airnow.gov/aqi/aqi-basics/" target="_blank" style="padding: 10px 20px; background: #00695c; color: white; border-radius: 8px; text-decoration: none;">📖 Learn about AQI</a>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# -------- Instructions Card --------
st.markdown(
    """
    <style>
    .instruction-card {
        background-color: #ffffff;
        padding: 30px;
        border-radius: 12px;
        border: 2px solid #00bfa5;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
    }
    .instruction-card:hover {
        transform: translateY(-5px);
    }
    .instruction-card h3 {
        text-align: center;
        color: #00796b;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="instruction-card">', unsafe_allow_html=True)
st.markdown("### 📋 How to Upload Data")
st.markdown("""
✅ Please upload your files in the following formats:

- **Reference Grade Data:** `datetime`, `pm25`, `pm10`, `site`
- **Quant AQ Data:** `datetime`, `temp`, `rh`, `pm1`, `pm25`, `pm10`, `site`
- **Gravimetric Data:** `date`, `pm25`, `pm10`, `site`
- **Clarity Data:** `datetime`, `corrected_pm25`, `pm10`, `site`
- **AirQo Data:** `datetime`, `pm25`, `pm10`, `site`

⚠️ Notes:
- Use `YYYY-MM-DD HH:MM:SS' format for date/time.
- Make sure column names are lowercase and match exactly.
- Avoid spaces or special characters in column names.
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
prompt = st.chat_input(
    "Say something and/or attach an image",
    accept_file=True,
    file_type=["jpg", "jpeg", "png"],
)
if prompt and prompt.text:
    st.markdown(prompt.text)
if prompt and prompt["files"]:
    st.image(prompt["files"][0])
    
sentiment_mapping = ["one", "two", "three", "four", "five"]
selected = st.feedback("stars")
if selected is not None:
    st.markdown(f"You selected {sentiment_mapping[selected]} star(s).")
    
# -------- Info Banner --------
st.success("📢 New updates coming soon! Stay tuned for enhanced analysis features and interactive visualizations.")

# -------- Floating Footer --------
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #e0f2f1;
        color: #004d40;
        text-align: center;
        padding: 12px 0;
        font-size: 14px;
        font-weight: bold;
        box-shadow: 0px -2px 10px rgba(0,0,0,0.1);
    }
    </style>

    <div class="footer">
        Made with ❤️ by Clement Mensah Ackaah
    </div>
    """,
    unsafe_allow_html=True
)
