"""
Air Monitor Data Tool (Streamlit)
Run with:  streamlit run app.py
Requires:  pip install streamlit pandas openpyxl xlrd
"""
import io
import os
import zipfile

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Air Monitor Data Tool", page_icon="🌫️", layout="wide")

# ---------------------------------------------------------------------------
# Monitor definitions
# ---------------------------------------------------------------------------
MONITORS = {
    "QuantAQ": [
        "timestamp", "timestamp_local", "id", "sn", "sample_rh", "sample_temp",
        "sample_pres", "lat", "lon", "device_state", "pm1", "pm25", "pm10",
        "pm1_model_id", "pm25_model_id", "pm10_model_id",
    ],
    "Air Gradient": [
        "Location ID", "Location Name", "Location Group", "Location Type",
        "Sensor ID", "Place Open", "Local Date/Time", "UTC Date/Time",
        "# of aggregated records", "PM2.5 (µg/m³) raw", "PM2.5 (µg/m³) corrected",
        "0.3µm particle count", "CO2 (ppm) raw", "CO2 (ppm) corrected",
        "Temperature (°C) raw", "Temperature (°C) corrected", "Heat Index (°C)",
        "Humidity (%) raw", "Humidity (%) corrected", "TVOC (ppb)", "TVOC index",
        "NOX index", "PM1 (µg/m³)", "PM10 (µg/m³)",
    ],
    "Airqo": [
        "site_name", "datetime", "frequency", "network", "latitude", "longitude",
        "temperature", "humidity", "pm25", "pm10",
    ],
    "T640": ["Timestamp (UTC+0)", "PM10", "PM25"],
}

# Key headings used to check that the right monitor type was chosen
KEY_HEADINGS = {
    "QuantAQ": ["timestamp", "pm25"],
    "Air Gradient": ["Local Date/Time"],
    "Airqo": ["datetime", "pm25"],
    "T640": ["Timestamp (UTC+0)", "PM25"],
}

SITE_COLS = ["site", "Location Name", "site_name"]


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_state():
    defaults = {
        "raw": {},          # id -> dict(name, monitor, df, site_created, ver)
        "cleaned": {},      # id -> dict(df, log, flat, stats)
        "merged_raw": {},   # name -> df
        "merged_clean": {}, # name -> df
        "next_id": 1,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()
S = st.session_state


# ---------------------------------------------------------------------------
# File reading helpers
# ---------------------------------------------------------------------------
def read_csv_bytes(b: bytes) -> pd.DataFrame:
    last_err = None
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(b), encoding=enc, low_memory=False)
        except UnicodeDecodeError as e:
            last_err = e
    raise last_err


def read_excel_bytes(b: bytes, label: str):
    sheets = pd.read_excel(io.BytesIO(b), sheet_name=None)
    if len(sheets) == 1:
        return [(label, next(iter(sheets.values())))]
    return [(f"{label} [{sn}]", df) for sn, df in sheets.items()]


def tables_from_bytes(name: str, b: bytes):
    ext = os.path.splitext(name)[1].lower()
    if ext == ".csv":
        return [(name, read_csv_bytes(b))]
    if ext in (".xlsx", ".xlsm", ".xls"):
        return read_excel_bytes(b, name)
    return []


def extract_tables(uploaded):
    """Return list of (display_name, df) for csv / excel / zip uploads."""
    out = []
    name = uploaded.name
    data = uploaded.getvalue()
    if name.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for member in zf.namelist():
                if member.endswith("/") or "__MACOSX" in member:
                    continue
                if os.path.basename(member).startswith("."):
                    continue
                try:
                    for n, df in tables_from_bytes(os.path.basename(member), zf.read(member)):
                        out.append((f"{name}::{n}", df))
                except Exception as e:  # noqa: BLE001
                    st.warning(f"Could not read {member} in {name}: {e}")
    else:
        out = tables_from_bytes(name, data)
    return out


def unique_name(name: str) -> str:
    existing = {m["name"] for m in S["raw"].values()}
    if name not in existing:
        return name
    i = 2
    stem, ext = os.path.splitext(name)
    while f"{stem} ({i}){ext}" in existing:
        i += 1
    return f"{stem} ({i}){ext}"


def add_raw(name, monitor, df):
    df = df.copy()
    site_created = False
    if not any(c in df.columns for c in SITE_COLS):
        stem = os.path.splitext(os.path.basename(name.split("::")[-1]))[0]
        df["site"] = stem
        site_created = True
    fid = S["next_id"]
    S["next_id"] += 1
    S["raw"][fid] = {
        "name": unique_name(name),
        "monitor": monitor,
        "df": df,
        "site_created": site_created,
        "ver": 0,
    }


def suggest_renames(monitor, cols):
    """Prefill suggested new names (datetime, pm25, pm10); user can edit."""
    m = {}
    if monitor == "QuantAQ":
        if "timestamp" in cols:
            m["timestamp"] = "datetime"
    elif monitor == "Air Gradient":
        if "Local Date/Time" in cols:
            m["Local Date/Time"] = "datetime"
        pm25_cols = [c for c in cols if str(c).startswith("PM2.5")]
        corrected = [c for c in pm25_cols if "corrected" in str(c).lower()]
        pick = (corrected or pm25_cols or [None])[0]
        if pick:
            m[pick] = "pm25"
        pm10_cols = [c for c in cols if str(c).startswith("PM10")]
        if pm10_cols:
            m[pm10_cols[0]] = "pm10"
    elif monitor == "T640":
        if "Timestamp (UTC+0)" in cols:
            m["Timestamp (UTC+0)"] = "datetime"
        if "PM25" in cols:
            m["PM25"] = "pm25"
        if "PM10" in cols:
            m["PM10"] = "pm10"
    return m


def site_col_of(df):
    for c in SITE_COLS:
        if c in df.columns:
            return c
    return None


def site_series(fid, df):
    """Site value for every row, taken from site / Location Name / site_name,
    or from the file's site name if none of those columns exists."""
    sc = site_col_of(df)
    if sc:
        return df[sc].astype(str)
    stem = os.path.splitext(os.path.basename(S["raw"][fid]["name"].split("::")[-1]))[0]
    return pd.Series(stem, index=df.index)


def with_site(fid, df, cols):
    """Selected columns plus a 'site' column (placed first)."""
    out = df[cols].copy()
    out["site"] = site_series(fid, df)
    return out[["site"] + [c for c in out.columns if c != "site"]]


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_df(df, monitor, flatline_threshold, remove_flatline_data,
             lower_threshold, upper_threshold, apply_correction):
    summary_log = []
    df = df.copy()

    if not {"datetime", "pm25"}.issubset(df.columns):
        return None, ["Missing required headings: 'datetime' and 'pm25' must exist "
                      "(rename columns in step 2)."], None, None

    # Parse types
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    if getattr(df["datetime"].dt, "tz", None) is not None:
        df["datetime"] = df["datetime"].dt.tz_localize(None)
    bad_dt = int(df["datetime"].isna().sum())
    df = df.dropna(subset=["datetime"])
    summary_log.append(f"Rows with invalid datetime dropped: {bad_dt}")

    df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")
    if "pm10" in df.columns:
        df["pm10"] = pd.to_numeric(df["pm10"], errors="coerce")
    else:
        summary_log.append("No 'pm10' heading found - PM10 not processed")

    site_col = site_col_of(df)
    sort_cols = ([site_col] if site_col else []) + ["datetime"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # =========================
    # DUPLICATE HANDLING
    # =========================
    exact_dup_count = int(df.duplicated().sum())
    df = df[~df.duplicated()]
    summary_log.append(f"Exact row duplicates removed: {exact_dup_count}")

    # =========================
    # FLAT-LINE QA/QC
    # =========================
    new_run = df["pm25"] != df["pm25"].shift()
    if site_col:  # a run never spans two different sites
        new_run = new_run | (df[site_col] != df[site_col].shift())
    df["flatline_group"] = new_run.cumsum()
    df["flatline_run_length"] = df.groupby("flatline_group")["pm25"].transform("size")
    df["flatline_flag"] = df["flatline_run_length"] >= flatline_threshold
    total_flagged = int(df["flatline_flag"].sum())
    events = int(df.loc[df["flatline_flag"], "flatline_group"].nunique())

    flat_detail = None
    if total_flagged > 0:
        flat_detail = df.loc[
            df["flatline_flag"], ["datetime", "pm25", "flatline_run_length", "flatline_flag"]
        ]

    if remove_flatline_data:
        df = df[~df["flatline_flag"]].copy()
        summary_log.append(f"Flat-line rows removed: {total_flagged}")
    else:
        summary_log.append(f"Flat-line rows flagged only: {total_flagged}")

    df = df.drop(columns=["flatline_group"])
    if remove_flatline_data:
        df = df.drop(columns=["flatline_run_length", "flatline_flag"])

    # =========================
    # OUTLIER REMOVAL
    # =========================
    below = int((df["pm25"] < lower_threshold).sum())
    above = int((df["pm25"] > upper_threshold).sum())
    df = df[(df["pm25"] >= lower_threshold) & (df["pm25"] <= upper_threshold)]
    summary_log.append(f"Outliers removed below {lower_threshold}: {below}")
    summary_log.append(f"Outliers removed above {upper_threshold}: {above}")

    # =========================
    # QuantAQ CORRECTION
    # =========================
    if monitor == "QuantAQ" and apply_correction:
        if not {"sample_temp", "sample_rh"}.issubset(df.columns):
            return None, ["Missing sample_temp or sample_rh for QuantAQ correction"], None, None
        df["corrected_pm25"] = (
            0.94 * df["pm25"]
            - 0.34 * df["sample_temp"]
            - 0.08 * df["sample_rh"]
            + 19.82
        )
        summary_log.append("LCS PM2.5 correction applied")

    stats = {"events": events, "flagged": total_flagged}
    return df.reset_index(drop=True), summary_log, flat_detail, stats


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------
def to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


def to_excel_bytes(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="data")
    return buf.getvalue()


def fmt_name(fid):
    m = S["raw"][fid]
    return f"{m['name']}  ({m['monitor']})"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🌫️ Air Monitor Data Tool")
st.caption("Upload → set site & headings → clean → merge → download")

tab_up, tab_cols, tab_clean, tab_merge, tab_dl = st.tabs(
    ["1 · Upload", "2 · Site & headings", "3 · Clean", "4 · Merge", "5 · Download"]
)

# ------------------------------- UPLOAD ------------------------------------
with tab_up:
    monitor = st.selectbox("Monitor type", list(MONITORS.keys()))
    with st.expander("Expected headings for this monitor type"):
        st.code(", ".join(MONITORS[monitor]))

    files = st.file_uploader(
        "Upload files (csv, zip, excel) - multiple allowed",
        type=["csv", "zip", "xlsx", "xlsm", "xls"],
        accept_multiple_files=True,
    )

    c1, c2 = st.columns([1, 1])
    if c1.button("Load files", type="primary", disabled=not files):
        loaded = 0
        for f in files:
            try:
                tables = extract_tables(f)
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read {f.name}: {e}")
                continue
            if not tables:
                st.warning(f"No csv/excel data found in {f.name}")
            for name, df in tables:
                dup = any(
                    m["monitor"] == monitor and m["name"] == name and m["df"].shape[0] == df.shape[0]
                    for m in S["raw"].values()
                )
                if dup:
                    st.info(f"{name} already loaded - skipped.")
                    continue
                missing = [h for h in KEY_HEADINGS[monitor] if h not in df.columns]
                if missing:
                    st.warning(
                        f"{name}: headings not found for {monitor}: {', '.join(missing)}. "
                        "Check the monitor type or rename columns in step 2."
                    )
                add_raw(name, monitor, df)
                loaded += 1
        if loaded:
            st.success(f"Loaded {loaded} dataset(s). Continue to step 2.")

    if c2.button("Clear everything"):
        for k in ("raw", "cleaned", "merged_raw", "merged_clean"):
            S[k] = {}
        st.rerun()

    if S["raw"]:
        st.subheader("Loaded datasets")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "File": m["name"],
                        "Monitor": m["monitor"],
                        "Rows": len(m["df"]),
                        "Columns": len(m["df"].columns),
                        "Site column": site_col_of(m["df"]) or "-",
                    }
                    for m in S["raw"].values()
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

# ------------------------- SITE & HEADINGS ---------------------------------
with tab_cols:
    if not S["raw"]:
        st.info("Upload and load files first.")
    else:
        st.write(
            "Set the site name and edit column headings. Cleaning needs headings named "
            "**datetime** and **pm25** (and **pm10** if available)."
        )
        for fid, m in list(S["raw"].items()):
            with st.expander(f"{m['name']}  ·  {m['monitor']}", expanded=False):
                df = m["df"]
                new_site = None
                if m["site_created"] and "site" in df.columns:
                    cur = str(df["site"].iloc[0]) if len(df) else ""
                    new_site = st.text_input(
                        "Site name (written to the 'site' column)",
                        value=cur,
                        key=f"site_{fid}_{m['ver']}",
                    )
                else:
                    sc = site_col_of(df)
                    st.caption(f"Existing site column found: **{sc}** - no new column created.")

                sug = suggest_renames(m["monitor"], list(df.columns))
                ren_df = pd.DataFrame(
                    {
                        "Current name": list(df.columns),
                        "New name": [sug.get(c, c) if m["ver"] == 0 else c for c in df.columns],
                    }
                )
                edited = st.data_editor(
                    ren_df,
                    disabled=["Current name"],
                    hide_index=True,
                    use_container_width=True,
                    key=f"ren_{fid}_{m['ver']}",
                )
                if sug and m["ver"] == 0:
                    st.caption("Suggested names for datetime / pm25 / pm10 are prefilled - edit if needed.")

                if st.button("Apply", key=f"apply_{fid}"):
                    new_names = [str(n).strip() for n in edited["New name"]]
                    if any(n == "" for n in new_names):
                        st.error("Column names cannot be empty.")
                    elif len(set(new_names)) != len(new_names):
                        st.error("Column names must be unique.")
                    else:
                        df2 = df.copy()
                        df2.columns = new_names
                        if new_site is not None and "site" in df2.columns:
                            df2["site"] = new_site
                        m["df"] = df2
                        m["ver"] += 1
                        S["cleaned"].pop(fid, None)  # old cleaned result is now stale
                        st.rerun()
                st.dataframe(df.head(5), use_container_width=True)

# ------------------------------- CLEAN -------------------------------------
with tab_clean:
    if not S["raw"]:
        st.info("Upload and load files first.")
    else:
        st.subheader("Cleaning settings")
        c1, c2, c3 = st.columns(3)
        flatline_threshold = c1.number_input("Flat-line threshold (consecutive identical PM2.5 values)",
                                             min_value=2, value=6, step=1)
        remove_flatline_data = c1.checkbox("Remove flat-line data (otherwise flag only)", value=True)
        lower_threshold = c2.number_input("Lower PM2.5 threshold", value=0.0)
        upper_threshold = c3.number_input("Upper PM2.5 threshold", value=1000.0)

        selected = st.multiselect(
            "Files to clean",
            options=list(S["raw"].keys()),
            default=list(S["raw"].keys()),
            format_func=fmt_name,
        )
        has_quantaq = any(S["raw"][i]["monitor"] == "QuantAQ" for i in selected)
        apply_correction = False
        if has_quantaq:
            apply_correction = st.checkbox(
                "Apply QuantAQ correction (corrected_pm25 = 0.94·pm25 − 0.34·temp − 0.08·RH + 19.82)",
                value=True,
            )

        if st.button("Run cleaning", type="primary", disabled=not selected):
            for fid in selected:
                m = S["raw"][fid]
                res, log, flat, stats = clean_df(
                    m["df"], m["monitor"], int(flatline_threshold), remove_flatline_data,
                    lower_threshold, upper_threshold, apply_correction,
                )
                if res is None:
                    st.error(f"{m['name']}: {log[0]}")
                    S["cleaned"].pop(fid, None)
                else:
                    S["cleaned"][fid] = {"df": res, "log": log, "flat": flat, "stats": stats}

        for fid, r in S["cleaned"].items():
            if fid not in S["raw"]:
                continue
            m = S["raw"][fid]
            with st.expander(f"✅ {m['name']}  -  {len(r['df']):,} rows after cleaning"):
                st.markdown("**Cleaning summary**")
                for line in r["log"]:
                    st.write("• " + line)
                st.markdown("**📊 PM2.5 Flat-line QA/QC Summary**")
                st.write(f"Events detected: **{r['stats']['events']}**")
                st.write(f"Points flagged: **{r['stats']['flagged']}**")
                if r["flat"] is not None:
                    with st.expander("🔍 View flat-line flagged data"):
                        st.dataframe(r["flat"], use_container_width=True)
                st.dataframe(r["df"].head(100), use_container_width=True)

# ------------------------------- MERGE -------------------------------------
with tab_merge:
    st.write("Merge files that share the same headings. Raw and cleaned data are merged separately.")
    source = st.radio("Data to merge", ["Raw data", "Cleaned data"], horizontal=True)

    if source == "Raw data":
        options = {fid: m["df"] for fid, m in S["raw"].items()}
    else:
        options = {fid: r["df"] for fid, r in S["cleaned"].items() if fid in S["raw"]}

    if not options:
        st.info("No datasets available for this selection yet.")
    else:
        sel = st.multiselect("Files to merge", list(options.keys()), format_func=fmt_name, key=f"msel_{source}")
        if len(sel) >= 2:
            col_sets = [list(options[i].columns) for i in sel]
            identical = all(set(c) == set(col_sets[0]) for c in col_sets)
            common = [c for c in col_sets[0] if all(c in cs for cs in col_sets)]
            if identical:
                st.success("All selected files have the same headings.")
            else:
                st.warning("Selected files do not have identical headings. Only the headings "
                           "common to all files can be merged.")
            chosen = st.multiselect("Headings to include in the merge", common, default=common,
                                    key=f"mcols_{source}")
            st.caption("A **site** column is always added to the merged data "
                       "(from site / Location Name / site_name, or the file's site name).")
            all_sites = sorted(set().union(*[set(site_series(i, options[i]).unique()) for i in sel]))
            chosen_sites = st.multiselect("Sites to include", all_sites, default=all_sites,
                                          key=f"msites_{source}")
            merge_name = st.text_input("Name for merged dataset", value=f"merged_{source.split()[0].lower()}")

            if st.button("Merge", type="primary"):
                if not chosen and not chosen_sites:
                    st.error("Select at least one heading.")
                elif not chosen_sites:
                    st.error("Select at least one site.")
                elif not merge_name.strip():
                    st.error("Enter a name for the merged dataset.")
                else:
                    merged = pd.concat([with_site(i, options[i], chosen) for i in sel], ignore_index=True)
                    merged = merged[merged["site"].isin(chosen_sites)].reset_index(drop=True)
                    target = S["merged_raw"] if source == "Raw data" else S["merged_clean"]
                    target[merge_name.strip()] = merged
                    st.success(f"Merged {len(sel)} files → {len(merged):,} rows.")
                    st.dataframe(merged.head(100), use_container_width=True)
        else:
            st.info("Select at least two files.")

    if S["merged_raw"] or S["merged_clean"]:
        st.subheader("Merged datasets")
        st.write(
            pd.DataFrame(
                [{"Name": n, "Type": "Raw", "Rows": len(d), "Columns": len(d.columns)} for n, d in S["merged_raw"].items()]
                + [{"Name": n, "Type": "Cleaned", "Rows": len(d), "Columns": len(d.columns)} for n, d in S["merged_clean"].items()]
            )
        )

# ------------------------------ DOWNLOAD -----------------------------------
with tab_dl:
    kind = st.radio(
        "What do you want to download?",
        ["Single raw file", "Single cleaned file", "Merged raw data", "Merged cleaned data"],
        horizontal=True,
    )
    datasets = {}
    if kind == "Single raw file":
        datasets = {S["raw"][i]["name"]: S["raw"][i]["df"] for i in S["raw"]}
    elif kind == "Single cleaned file":
        datasets = {S["raw"][i]["name"]: r["df"] for i, r in S["cleaned"].items() if i in S["raw"]}
    elif kind == "Merged raw data":
        datasets = dict(S["merged_raw"])
    else:
        datasets = dict(S["merged_clean"])

    if not datasets:
        st.info("Nothing available here yet.")
    else:
        name = st.selectbox("Dataset", list(datasets.keys()))
        df = datasets[name]
        cols = st.multiselect("Headings to download", list(df.columns), default=list(df.columns),
                              key=f"dlcols_{kind}_{name}")
        fmt = st.radio("Format", ["CSV", "Excel"], horizontal=True)
        if cols:
            out = df[cols]
            base = os.path.splitext(os.path.basename(name.split("::")[-1]))[0]
            prefix = "cleaned_" if "cleaned" in kind.lower() else ""
            st.caption(f"{len(out):,} rows × {len(cols)} columns")
            st.dataframe(out.head(50), use_container_width=True)
            if fmt == "CSV":
                st.download_button("⬇️ Download CSV", to_csv_bytes(out),
                                   file_name=f"{prefix}{base}.csv", mime="text/csv")
            else:
                st.download_button(
                    "⬇️ Download Excel", to_excel_bytes(out), file_name=f"{prefix}{base}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
        else:
            st.warning("Select at least one heading.")
