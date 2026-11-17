import streamlit as st
import zipfile
import os
import pandas as pd
import io
import csv

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Multi-File Combiner",
    layout="centered"
)

st.title("📊 Multi-File Combiner (ZIP / CSV / TXT / LOG)")

# =========================
# FILE UPLOAD
# =========================
uploaded_files = st.file_uploader(
    "Upload one or more files",
    type=["zip", "csv", "txt", "log"],
    accept_multiple_files=True
)

# =========================
# FILE READER
# =========================
def read_dataframe(file_obj):
    for encoding in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            file_obj.seek(0)
            sample = file_obj.read(4096).decode(encoding, errors="ignore")
            delimiter = csv.Sniffer().sniff(sample).delimiter
            file_obj.seek(0)
            return pd.read_csv(file_obj, encoding=encoding, delimiter=delimiter)
        except Exception:
            continue
    raise ValueError("Unsupported encoding or delimiter")

# =========================
# PROCESS FILES
# =========================
combined_frames = []
per_file_data = {}
errors = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name
        ext = filename.split(".")[-1].lower()

        # -------- ZIP --------
        if ext == "zip":
            try:
                with zipfile.ZipFile(uploaded_file) as z:
                    for name in z.namelist():
                        if (
                            name.startswith("__MACOSX")
                            or os.path.basename(name).startswith("._")
                            or not name.lower().endswith((".csv", ".txt", ".log"))
                        ):
                            continue
                        try:
                            with z.open(name) as f:
                                df = read_dataframe(io.BytesIO(f.read()))
                                df["source_file"] = os.path.basename(name)
                                df["source_path"] = name
                                combined_frames.append(df)
                                per_file_data[name] = df
                        except Exception as e:
                            errors.append(f"{name}: {e}")
            except zipfile.BadZipFile:
                errors.append(f"{filename}: Invalid ZIP file")

        # -------- SINGLE FILE --------
        else:
            try:
                df = read_dataframe(uploaded_file)
                df["source_file"] = filename
                df["source_path"] = filename
                combined_frames.append(df)
                per_file_data[filename] = df
            except Exception as e:
                errors.append(f"{filename}: {e}")

    # =========================
    # COMBINE & HARMONIZE
    # =========================
    if combined_frames:
        combined_df = pd.concat(
            combined_frames,
            ignore_index=True,
            sort=False
        )

        st.subheader("🔍 Preview")
        st.dataframe(combined_df.head(100), use_container_width=True)

        st.markdown(f"**Total rows:** {len(combined_df)}")
        st.markdown(f"**Total columns:** {len(combined_df.columns)}")

        # =========================
        # DOWNLOAD OPTIONS
        # =========================
        st.subheader("💾 Download Options")

        col1, col2 = st.columns(2)

        with col1:
            output_format = st.radio(
                "Format",
                ["CSV", "TXT", "ZIP"]
            )

        with col2:
            zip_mode = st.radio(
                "ZIP Mode",
                ["Single combined file", "One file per source"],
                disabled=(output_format != "ZIP")
            )

        txt_delim_label = st.selectbox(
            "TXT delimiter",
            ["Tab", "Comma", "Pipe"]
        )

        txt_delim = {
            "Tab": "\t",
            "Comma": ",",
            "Pipe": "|"
        }[txt_delim_label]

        # =========================
        # EXPORT PREP
        # =========================
        csv_bytes = combined_df.to_csv(index=False).encode("utf-8")
        txt_bytes = combined_df.to_csv(index=False, sep=txt_delim).encode("utf-8")

        # -------- CSV --------
        if output_format == "CSV":
            st.download_button(
                "⬇️ Download CSV",
                csv_bytes,
                "combined_output.csv",
                "text/csv"
            )

        # -------- TXT --------
        elif output_format == "TXT":
            st.download_button(
                "⬇️ Download TXT",
                txt_bytes,
                "combined_output.txt",
                "text/plain"
            )

        # -------- ZIP --------
        elif output_format == "ZIP":
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:

                if zip_mode == "Single combined file":
                    zf.writestr("combined_output.csv", csv_bytes)

                else:
                    for name, df in per_file_data.items():
                        safe = os.path.basename(name).rsplit(".", 1)[0]
                        zf.writestr(
                            f"{safe}.csv",
                            df.to_csv(index=False).encode("utf-8")
                        )
                        zf.writestr(
                            f"{safe}.txt",
                            df.to_csv(index=False, sep=txt_delim).encode("utf-8")
                        )

            zip_buffer.seek(0)

            st.download_button(
                "⬇️ Download ZIP",
                zip_buffer,
                "combined_output.zip",
                "application/zip"
            )

    else:
        st.warning("No valid data found.")

    # =========================
    # ERRORS
    # =========================
    if errors:
        with st.expander("⚠️ Files with Errors"):
            for e in errors:
                st.text(e)
