import streamlit as st
import streamlit.components.v1 as components
from PIL import Image 

# Page Configuration
st.set_page_config(
    page_title="Air Quality Data Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------- Custom Background and Font --------
st.markdown(
    """
    <style>
    /* Background gradient */
    body {
        background: linear-gradient(135deg, #e0f7fa, #ffffff);
        background-attachment: fixed;
    }

    /* Main app styling */
    .stApp {
        background-color: transparent;
    }

    /* Smooth fonts and size */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', 'Roboto', sans-serif;
        font-size: 16px;
        font-smooth: always;
    }

    /* Headings */
    h1, h2, h3 {
        font-weight: bold;
        color: #004d40;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------- Custom Sidebar Styling --------
st.markdown(
    """
    <style>
    /* Sidebar background and padding */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #e0f2f1, #ffffff);
        padding: 20px;
        border-right: 2px solid #d3d3d3;
    }

    /* Sidebar header text styling */
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #00695c;
    }

    /* Sidebar link styling */
    [data-testid="stSidebar"] a {
        color: #00796b;
        font-weight: bold;
        text-decoration: none;
    }
    [data-testid="stSidebar"] a:hover {
        color: #004d40;
    }
    </style>
    """,
    unsafe_allow_html=True
)

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

# -------- Introduction Section --------
st.markdown("## 🌍 About the Dashboard")
col1, col2 = st.columns([1, 2])

with col1:
    st.image("https://images.unsplash.com/photo-1581092580492-0b0f87e9592d?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=MnwzNjUyOXwwfDF8c2VhcmNofDF8fGFpciUyMHF1YWxpdHl8ZW58MHx8fHwxNjg2MTA0MTA0&ixlib=rb-4.0.3&q=80&w=400", use_column_width=True)

with col2:
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

st.markdown("---")

# -------- Quick Links --------
st.markdown("### 🔗 Quick Links")
st.markdown(
    """
    <div style="display: flex; gap: 20px; flex-wrap: wrap;">
        <a href="https://www.epa.gov/" target="_blank" style="padding: 10px 20px; background: #00796b; color: white; border-radius: 8px; text-decoration: none;">🌐 EPA Website</a>
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
- Use `YYYY-MM-DD HH:MM:SS', `YYYY-MM-DD' format for date/time.
- Make sure column names are lowercase and match exactly.
- Avoid spaces or special characters in column names.
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

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
