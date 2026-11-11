import streamlit as st
import pandas as pd
import zipfile
import io

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Sensor Data Processing Tool",
    layout="centered"
)

st.title("📊 Sensor Data Processing Tool")

st.markdown("""
Upload a **TXT, CSV, LOG, or ZIP** sensor file.  
Download **cleaned raw data** or **aggregated averages**.
""")

st.info("⚠️ Large files may take time to upload and process.")

# ======================
# CANONICAL COLUMN NAMES
# ======================
TIME_LOCAL = "Date & Time (Local)"
TIME_UTC   = "Date & Time (UTC)"

PM10_COL = "PM10 Conc"
PM25_COL = "PM2.5 Conc"

# ======================
# USER OPTIONS
# ======================
time_choice = st.radio(
    "Select time reference",
    ["Local", "UTC"],
    horizontal=True
)

TIME_COL = TIME_LOCAL if time_choice == "Local" else TIME_UTC

aggregation_level = st.selectbox(
    "Select averaging interval",
    ["Hourly", "Daily", "Monthly", "Quarterly", "Yearly"]
)

uploaded_file = st.file_uploader(
    "Upload file (ZIP, CSV, TXT, LOG)",
    type=["zip", "csv", "txt", "log"]
)

# ======================
# SIDEBAR OPTIONS
# ======================
st.sidebar.header("⚙️ Export Options")
site_name = st.sidebar.text_input("Site name")

# ======================
# RESAMPLE MAP
# ======================
RESAMPLE_MAP = {
    "Hourly": "h",
    "Daily": "D",
    "Monthly": "ME",
    "Quarterly": "QE",
    "Yearly": "YE"
}

# ======================
# FILE ITERATOR
# ======================
def iterate_files(uploaded):
    if uploaded.name.lower().endswith(".zip"):
        with zipfile.ZipFile(uploaded) as z:
            for name in z.namelist():
                if name.lower().endswith((".csv", ".txt", ".log")):
                    yield io.TextIOWrapper(z.open(name))
    else:
        yield uploaded

# ======================
# PROCESSING
# ======================
if uploaded_file:
    try:
        CHUNK_SIZE = 500_000

        progress = st.progress(0)
        status = st.empty()

        # ======================
        # READ SAMPLE
        # ======================
        file_obj = next(iterate_files(uploaded_file))
        file_obj.seek(0)

        sample = pd.read_csv(file_obj, nrows=5)
        sample.columns = sample.columns.str.strip()

        required_cols = [TIME_COL, PM10_COL, PM25_COL]
        missing = [c for c in required_cols if c not in sample.columns]

        if missing:
            st.error("Missing required columns:\n" + "\n".join(missing))
            st.stop()

        # ======================
        # STORAGE
        # ======================
        cleaned_chunks = []
        agg_chunks = []
        total_rows = 0

        status.text("📥 Cleaning data...")

        # ======================
        # READ & CLEAN
        # ======================
        for file_obj in iterate_files(uploaded_file):
            file_obj.seek(0)

            for chunk in pd.read_csv(
                file_obj,
                chunksize=CHUNK_SIZE,
                na_values=["-----"],
                skipinitialspace=True,
                low_memory=False,
                on_bad_lines="skip"
            ):
                chunk.columns = chunk.columns.str.strip()
                chunk = chunk[[TIME_COL, PM10_COL, PM25_COL]]

                chunk[TIME_COL] = pd.to_datetime(
                    chunk[TIME_COL],
                    errors="coerce",
                    utc=(TIME_COL == TIME_UTC)
                )

                chunk[PM10_COL] = pd.to_numeric(chunk[PM10_COL], errors="coerce")
                chunk[PM25_COL] = pd.to_numeric(chunk[PM25_COL], errors="coerce")

                chunk = chunk.dropna(subset=[TIME_COL])

                if chunk.empty:
                    continue

                total_rows += len(chunk)

                cleaned_chunks.append(chunk.copy())

                # ---- aggregate later, AFTER date filtering
                agg_chunks.append(chunk)

                progress.progress(min(total_rows / 5_000_000, 0.7))

        if not cleaned_chunks:
            st.error("No valid data found.")
            st.stop()

        # ======================
        # COMBINE CLEANED DATA
        # ======================
        cleaned_df = pd.concat(cleaned_chunks, ignore_index=True)
        cleaned_df["site"] = site_name if site_name else "UNKNOWN"

        # ======================
        # DATE RANGE (APPLIES TO BOTH)
        # ======================
        st.sidebar.header("📅 Date Range")

        min_date = cleaned_df[TIME_COL].min().date()
        max_date = cleaned_df[TIME_COL].max().date()

        start_date = st.sidebar.date_input(
            "Start date",
            value=min_date,
            min_value=min_date,
            max_value=max_date
        )

        end_date = st.sidebar.date_input(
            "End date",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )

        if start_date > end_date:
            st.sidebar.error("Start date must be before end date.")
            st.stop()

        mask = (
            (cleaned_df[TIME_COL].dt.date >= start_date) &
            (cleaned_df[TIME_COL].dt.date <= end_date)
        )

        cleaned_df = cleaned_df.loc[mask]

        if cleaned_df.empty:
            st.warning("No data in selected date range.")
            st.stop()

        # ======================
        # CLEANED DATA OUTPUT
        # ======================
        st.subheader("🧹 Cleaned Data Preview (No Aggregation)")
        st.dataframe(cleaned_df.head(100), use_container_width=True)

        st.download_button(
            "⬇️ Download Cleaned CSV",
            cleaned_df.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_sensor_data.csv",
            mime="text/csv"
        )

        # ======================
        # AGGREGATION (AFTER DATE FILTER)
        # ======================
        status.text("📊 Aggregating data...")

        df_agg = (
            cleaned_df
            .set_index(TIME_COL)
            .resample(RESAMPLE_MAP[aggregation_level])
            .mean(numeric_only=True)
            .reset_index()
        )

        p = df_agg.rename(columns={
            TIME_COL: "datetime",
            PM25_COL: "pm25",
            PM10_COL: "pm10"
        })

        p["site"] = site_name if site_name else "UNKNOWN"
        p = p[["datetime", "pm25", "pm10", "site"]]

        progress.progress(1.0)
        status.text("✅ Processing complete")

        # ======================
        # AGG OUTPUT
        # ======================
        st.subheader("📈 Aggregated Output Preview")
        st.dataframe(p.head(100), use_container_width=True)

        st.download_button(
            "⬇️ Download Aggregated CSV",
            p.to_csv(index=False).encode("utf-8"),
            file_name=f"{aggregation_level.lower()}_average_final.csv",
            mime="text/csv"
        )

        st.success(f"Processing completed ({len(cleaned_df):,} rows in range).")

    except Exception as e:
        st.error(f"❌ Processing failed: {e}")
