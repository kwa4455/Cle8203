import streamlit as st
import streamlit.components.v1 as components
from PIL import Image 

# Page Configuration
st.set_page_config(
    page_title="Air Quality Data Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# -------- Custom Background and Font --------
st.markdown(
    """
    <style>
    /* Background gradient */
    body {
        background: linear-gradient(to bottom right, #e0f7fa, #ffffff);
        background-attachment: fixed;
    }

    /* Main app styling */
    .stApp {
        background-color: transparent;
    }

    /* Smooth fonts */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', 'Roboto', sans-serif;
        font-smooth: always;
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
        background: linear-gradient(to bottom right, #e0f2f1, #ffffff);
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
    # Logo (Optional - if you have a logo file)
    try:
        st.image("epa-logo.png", width=150)
    except:
        pass  # Skip if logo not found

    st.markdown("### 🧑‍💻 Developer Information")
    st.markdown("""
    - **Developed by:** Clement Mensah Ackaah
    - **Email:** clement.ackaah@epa.gov.gh / clementackaah70@gmail.com
    - **GitHub:** [Visit GitHub](https://github.com/yourusername)
    - **LinkedIn:** [Visit LinkedIn](https://www.linkedin.com/in/yourname)
    - **Project Repo:** [Air Quality Dashboard](https://github.com/yourusername/air-quality-dashboard)
    """)

# -------- Welcome Section --------
components.html(
    """
    <div style="font-size:40px; font-weight:bold; text-align:center; animation: fadeInMove 2s ease-in-out;">
        👋 Welcome to the Air Quality Dashboard!
    </div>
    <style>
    @keyframes fadeInMove {
      0% {opacity: 0; transform: translateY(50px);}
      100% {opacity: 1; transform: translateY(0);}
    }
    </style>
    """,
    height=100
)

# -------- Main Image --------

# -------- Introduction Section --------
col1, col2 = st.columns(2)

with col1:
    st.markdown("## 🌍 Air Quality Data Analysis")
    st.markdown("""
    Analyze, visualize, and monitor air quality data collected from a variety of sensors including:
    Reference Grade Instruments, Quant AQ, Gravimetric Samplers, Clarity Monitors, and AirQo Devices.
    """)

with col2:
    st.markdown("### 🔗 Quick Links")
    st.markdown("""
    - [Visit EPA Website](https://www.epa.gov/)
    - [Learn about AQI](https://www.airnow.gov/aqi/aqi-basics/)
    """)

st.markdown("---")

# -------- Instructions Section --------
st.markdown(
    """
    <style>
    .instruction-card {
        background-color: #f9f9f9;
        padding: 25px;
        border-radius: 12px;
        border: 1px solid #ddd;
        margin-bottom: 20px;
        transition: box-shadow 0.3s ease-in-out;
    }
    .instruction-card:hover {
        box-shadow: 0 6px 18px rgba(0, 0, 0, 0.2);
    }
    .instruction-card h3 {
        text-align: center;
        color: #4CAF50;
        font-weight: bold;
    }
    .instruction-card ul {
        padding-left: 20px;
    }
    .instruction-card li {
        margin-bottom: 10px;
    }
    .instruction-card p {
        font-size: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="instruction-card">', unsafe_allow_html=True)
st.markdown("### 📋 **Instructions for Uploading Data**")
st.markdown("""
<br>

**Please upload files according to the correct format:**

- **Reference Grade Data**  
  ➔ Columns: `datetime`, `pm25`, `pm10`, `site`
  
- **Quant AQ Data**  
  ➔ Columns: `datetime`, `temp`, `rh`, `pm1`, `pm25`, `pm10`, `site`
  
- **Gravimetric Data**  
  ➔ Columns: `date`, `pm25`, `pm10`, `site`
  
- **Clarity Data**  
  ➔ Columns: `datetime`, `corrected_pm25`, `pm10`, `site`
  
- **AirQo Data**  
  ➔ Columns: `datetime`, `pm25`, `pm10`, `site`

<br>

✅ Ensure `datetime` or `date` columns are formatted as `YYYY-MM-DD HH:MM:SS`  
✅ Column names should be lowercase and match exactly  
✅ Avoid spaces or special characters in column names

""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# -------- Info Banner --------
st.info("📢 Stay tuned for new data uploads, dashboard improvements, and additional features!")

# -------- Floating Footer --------
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: #555;
        text-align: center;
        padding: 10px 0;
        font-size: 14px;
        font-weight: 500;
    }
    </style>

    <div class="footer">
        Made with ❤️ by Clement Mensah Ackaah
    </div>
    """,
    unsafe_allow_html=True
)
