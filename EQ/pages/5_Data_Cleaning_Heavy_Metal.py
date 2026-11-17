import io
import re
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st


# ============================================================
# PAGE SETTINGS
# ============================================================
st.set_page_config(
    page_title="Metal Concentration Data Cleaner",
    page_icon="🧹",
    layout="wide",
)

st.title("Metal Concentration Data Cleaner")
st.caption(
    "Upload or paste laboratory results with two-level headers such as "
    "'Cd / Conc / Uncert'. The app creates Cd and Cd_error columns and "
    "extracts the date, site code, site name, and pollutant from sample_ids. Units are retained from headers such as Al (ug/m3) and Cr (ng/m3)."
)


# ============================================================
# CONSTANTS
# ============================================================
SAMPLE_HEADER_KEYS = {
    "sampleid",
    "sampleids",
    "sampleidentifier",
    "sampleidentifiers",
    "sample",
}

CONCENTRATION_KEYS = {
    "conc",
    "concentration",
    "result",
    "value",
    "measuredvalue",
}

ERROR_KEYS = {
    "uncert",
    "uncertainty",
    "error",
    "err",
    "sigma",
    "sd",
    "standarddeviation",
}

ELEMENT_CASE = {
    "AL": "Al",
    "CR": "Cr",
    "MN": "Mn",
    "FE": "Fe",
    "CO": "Co",
    "NI": "Ni",
    "CU": "Cu",
    "ZN": "Zn",
    "AS": "As",
    "CD": "Cd",
    "HG": "Hg",
    "PB": "Pb",
    "MG": "Mg",
    "CA": "Ca",
    "NA": "Na",
    "K": "K",
    "TI": "Ti",
    "V": "V",
    "SE": "Se",
    "BR": "Br",
    "SR": "Sr",
    "MO": "Mo",
    "SN": "Sn",
    "SB": "Sb",
    "BA": "Ba",
}


# ============================================================
# GENERAL HELPERS
# ============================================================
def normalize_key(value: object) -> str:
    """Normalize a header value for reliable comparison."""
    if pd.isna(value):
        return ""
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def safe_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def make_unique(names: List[str]) -> List[str]:
    """Ensure output column names are unique."""
    counts: Dict[str, int] = {}
    output: List[str] = []

    for name in names:
        base = name or "unnamed"
        counts[base] = counts.get(base, 0) + 1
        if counts[base] == 1:
            output.append(base)
        else:
            output.append(f"{base}_{counts[base]}")
    return output


def canonical_analyte_name(header: object) -> str:
    """
    Convert headers such as 'Al (ug/m3)' to 'Al'.
    Other analyte names are cleaned but retained.
    """
    text = safe_text(header)
    text = re.sub(r"\([^)]*\)", "", text)  # remove unit in parentheses
    text = re.sub(
        r"\b(conc(?:entration)?|uncert(?:ainty)?|error|err|sigma|sd|result|value)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_-.")

    if not text:
        return "analyte"

    upper = text.upper()
    if upper in ELEMENT_CASE:
        return ELEMENT_CASE[upper]

    # Keep conventional capitalization for short chemical symbols.
    if re.fullmatch(r"[A-Za-z]{1,3}", text):
        return text[0].upper() + text[1:].lower()

    return text


def extract_unit(header: object) -> str:
    """Extract text inside parentheses, e.g. ug/m3 or ng/m3."""
    text = safe_text(header)
    match = re.search(r"\(([^)]*)\)", text)
    return match.group(1).strip() if match else ""


def classify_subheader(value: object) -> str:
    key = normalize_key(value)

    if key in CONCENTRATION_KEYS or key.startswith("conc"):
        return "concentration"

    if (
        key in ERROR_KEYS
        or key.startswith("uncert")
        or key.startswith("error")
    ):
        return "error"

    return "other"


# ============================================================
# INPUT READING
# ============================================================
def read_delimited_bytes(
    content: bytes,
    separator_choice: str,
) -> pd.DataFrame:
    separator_map = {
        "Auto-detect": None,
        "Tab": "\t",
        "Comma": ",",
        "Semicolon": ";",
        "Pipe": "|",
    }
    separator = separator_map[separator_choice]

    last_error: Optional[Exception] = None
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return pd.read_csv(
                io.BytesIO(content),
                sep=separator,
                engine="python",
                header=None,
                dtype=object,
                encoding=encoding,
                keep_default_na=False,
            )
        except Exception as exc:
            last_error = exc

    raise ValueError(f"Could not read the delimited file: {last_error}")


def read_pasted_text(text: str) -> pd.DataFrame:
    first_line = text.splitlines()[0] if text.splitlines() else ""

    if "\t" in first_line:
        separator = "\t"
    elif ";" in first_line:
        separator = ";"
    elif "|" in first_line:
        separator = "|"
    elif "," in first_line:
        separator = ","
    else:
        separator = None

    return pd.read_csv(
        io.StringIO(text),
        sep=separator,
        engine="python",
        header=None,
        dtype=object,
        keep_default_na=False,
    )


# ============================================================
# HEADER DETECTION AND FLATTENING
# ============================================================
def find_sample_header(raw: pd.DataFrame) -> Tuple[int, int]:
    """
    Find the row and column containing sample_ids.
    Searches the first 30 rows.
    """
    search_rows = min(30, len(raw))

    for row_index in range(search_rows):
        for col_index, value in enumerate(raw.iloc[row_index].tolist()):
            if normalize_key(value) in SAMPLE_HEADER_KEYS:
                return row_index, col_index

    raise ValueError(
        "The app could not find a sample_ids column in the first 30 rows."
    )


def second_row_is_subheader(row: pd.Series) -> bool:
    classifications = [classify_subheader(value) for value in row.tolist()]
    recognised = sum(
        item in {"concentration", "error"} for item in classifications
    )
    return recognised >= 2


def parse_one_level_header(header: object) -> str:
    """
    Handle a one-row header such as:
    'Cd Conc', 'Cd Uncert', 'Cd Error', or 'Cd (ng/m3)'.
    """
    text = safe_text(header)
    key = normalize_key(text)

    if key in SAMPLE_HEADER_KEYS:
        return "sample_id"

    is_error = any(
        token in key
        for token in (
            "uncert",
            "uncertainty",
            "error",
            "err",
            "sigma",
            "standarddeviation",
        )
    )

    analyte = canonical_analyte_name(text)
    return f"{analyte}_error" if is_error else analyte


def flatten_lab_table(
    raw: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Convert two-level laboratory headers into a single tidy header.

    Example:
        Cd | Cd
        Conc | Uncert

    becomes:
        Cd | Cd_error
    """
    raw = raw.copy()
    raw = raw.replace(r"^\s*$", pd.NA, regex=True)
    raw = raw.dropna(axis=0, how="all").dropna(axis=1, how="all")

    if raw.empty:
        raise ValueError("The uploaded table contains no usable data.")

    header_row, sample_col = find_sample_header(raw)

    # Remove title rows above the true header and irrelevant columns to the left.
    table = raw.iloc[header_row:, sample_col:].reset_index(drop=True)
    table = table.dropna(axis=1, how="all")

    top_headers_original = table.iloc[0].tolist()
    top_headers_filled = pd.Series(top_headers_original).ffill().tolist()

    has_subheader = (
        len(table) > 1 and second_row_is_subheader(table.iloc[1])
    )

    subheaders = (
        table.iloc[1].tolist()
        if has_subheader
        else [""] * table.shape[1]
    )

    output_names: List[str] = []
    units_records: List[Dict[str, str]] = []
    analyte_columns: List[str] = []

    for column_index in range(table.shape[1]):
        top_original = top_headers_original[column_index]
        top_filled = top_headers_filled[column_index]
        subheader = subheaders[column_index]

        if column_index == 0:
            output_name = "sample_id"
        elif has_subheader:
            analyte = canonical_analyte_name(top_filled)
            subheader_type = classify_subheader(subheader)

            if subheader_type == "error":
                output_name = f"{analyte}_error"
            elif subheader_type == "concentration":
                output_name = analyte
            else:
                extra = canonical_analyte_name(subheader)
                output_name = (
                    f"{analyte}_{extra}"
                    if extra and extra != "analyte"
                    else analyte
                )
        else:
            output_name = parse_one_level_header(top_filled)

        output_names.append(output_name)

        if column_index > 0:
            final_analyte = output_name.removesuffix("_error")
            unit = extract_unit(top_filled) or extract_unit(top_original)

            units_records.append(
                {
                    "analyte": final_analyte,
                    "unit_as_uploaded": unit,
                    "output_column": output_name,
                    "measurement_type": (
                        "error_or_uncertainty"
                        if output_name.endswith("_error")
                        else "concentration"
                    ),
                }
            )
            analyte_columns.append(output_name)

    output_names = make_unique(output_names)

    data_start_row = 2 if has_subheader else 1
    cleaned = table.iloc[data_start_row:].copy()
    cleaned.columns = output_names
    cleaned = cleaned.dropna(axis=0, how="all")
    cleaned = cleaned.reset_index(drop=True)

    # Remove rows without a sample ID.
    cleaned["sample_id"] = cleaned["sample_id"].astype("string").str.strip()
    cleaned = cleaned[
        cleaned["sample_id"].notna()
        & cleaned["sample_id"].ne("")
    ].copy()

    # Reset after filtering so all later Boolean masks align correctly.
    cleaned = cleaned.reset_index(drop=True)

    units = pd.DataFrame(units_records).drop_duplicates().reset_index(drop=True)

    # Rebuild the analyte list after unique-name handling.
    analyte_columns = [
        column for column in cleaned.columns if column != "sample_id"
    ]

    return cleaned, units, analyte_columns


# ============================================================
# SAMPLE ID PARSING
# ============================================================
# Supported sample-ID structures:
# 1. SITE_DATE_POLLUTANT, e.g. APS_170325_PM2.5
# 2. SITE_DATEPOLLUTANT,  e.g. 1_031118PM10
SAMPLE_ID_PATTERNS = [
    re.compile(
        r"^(?P<site>.+?)_(?P<date>\d{6}|\d{8})_(?P<pollutant>.+)$",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"^(?P<site>.+?)_(?P<date>\d{6}|\d{8})(?P<pollutant>[A-Za-z][A-Za-z0-9.]*)$",
        flags=re.IGNORECASE,
    ),
]


def match_sample_id(text: str) -> Optional[re.Match]:
    """Return the first matching supported sample-ID structure."""
    for pattern in SAMPLE_ID_PATTERNS:
        match = pattern.fullmatch(text)
        if match:
            return match
    return None


def normalize_pollutant(value: str) -> str:
    text = str(value).strip()
    compact = re.sub(r"[\s_-]+", "", text.upper())

    if compact in {"PM2.5", "PM25"}:
        return "PM2.5"
    if compact == "PM10":
        return "PM10"
    if compact == "TSP":
        return "TSP"

    return text


def parse_date_token(token: str, date_format_label: str) -> pd.Timestamp:
    formats = {
        "DDMMYY — 170325 = 17 Mar 2025": "%d%m%y",
        "YYMMDD — 250317 = 17 Mar 2025": "%y%m%d",
        "DDMMYYYY — 17032025 = 17 Mar 2025": "%d%m%Y",
        "YYYYMMDD — 20250317 = 17 Mar 2025": "%Y%m%d",
    }

    selected_format = formats[date_format_label]

    # First try the user's selected format.
    parsed = pd.to_datetime(token, format=selected_format, errors="coerce")
    if not pd.isna(parsed):
        return parsed

    # If the token length differs, try the compatible alternatives.
    fallback_formats = (
        ["%d%m%y", "%y%m%d"]
        if len(token) == 6
        else ["%d%m%Y", "%Y%m%d"]
    )

    for date_format in fallback_formats:
        parsed = pd.to_datetime(token, format=date_format, errors="coerce")
        if not pd.isna(parsed):
            return parsed

    return pd.NaT


def parse_sample_ids(
    sample_ids: pd.Series,
    date_format_label: str,
) -> pd.DataFrame:
    records: List[Dict[str, object]] = []

    for sample_id in sample_ids.astype("string").fillna(""):
        text = str(sample_id).strip()
        match = match_sample_id(text)

        if not match:
            records.append(
                {
                    "site_code": pd.NA,
                    "date": pd.NaT,
                    "pollutant": pd.NA,
                    "sample_id_valid": False,
                    "invalid_reason": "Unsupported sample ID structure",
                }
            )
            continue

        date_value = parse_date_token(
            match.group("date"),
            date_format_label,
        )

        date_is_valid = not pd.isna(date_value)

        records.append(
            {
                "site_code": match.group("site").strip(),
                "date": date_value,
                "pollutant": normalize_pollutant(
                    match.group("pollutant")
                ),
                "sample_id_valid": date_is_valid,
                "invalid_reason": (
                    pd.NA
                    if date_is_valid
                    else "Date could not be parsed"
                ),
            }
        )

    return pd.DataFrame(records)


# ============================================================
# NUMERIC CLEANING
# ============================================================
def clean_numeric_series(
    series: pd.Series,
    below_detection_rule: str,
) -> pd.Series:
    """
    Convert laboratory values to numeric.

    Rules for values such as <0.05:
    - Missing: converts to NaN
    - Half limit: converts to 0.025
    - Reporting limit: converts to 0.05
    """
    text = series.astype("string").str.strip()
    text = text.str.replace("−", "-", regex=False)
    text = text.str.replace(",", "", regex=False)

    lower = text.str.lower()
    missing_tokens = {
        "",
        "na",
        "n/a",
        "nan",
        "none",
        "null",
        "nd",
        "n.d.",
        "bdl",
        "below detection",
        "-",
        "--",
    }
    text = text.mask(lower.isin(missing_tokens), pd.NA)

    is_below_limit = text.str.match(r"^\s*<", na=False)

    extracted = text.str.extract(
        r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)",
        expand=False,
    )
    numeric = pd.to_numeric(extracted, errors="coerce")

    if below_detection_rule == "Treat <value as missing":
        numeric = numeric.mask(is_below_limit)
    elif below_detection_rule == "Use half of <value":
        numeric = numeric.mask(is_below_limit, numeric / 2)
    # Otherwise use the reporting-limit number as written.

    return numeric


# ============================================================
# EXPORT
# ============================================================
def create_excel_download(
    cleaned_data: pd.DataFrame,
    long_data: pd.DataFrame,
    units: pd.DataFrame,
    qc_summary: pd.DataFrame,
    invalid_ids: pd.DataFrame,
) -> bytes:
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        cleaned_data.to_excel(
            writer,
            sheet_name="Cleaned_Data_Wide",
            index=False,
        )
        long_data.to_excel(
            writer,
            sheet_name="Cleaned_Data_Long",
            index=False,
        )
        units.to_excel(
            writer,
            sheet_name="Units",
            index=False,
        )
        qc_summary.to_excel(
            writer,
            sheet_name="QC_Summary",
            index=False,
        )
        invalid_ids.to_excel(
            writer,
            sheet_name="Invalid_Sample_IDs",
            index=False,
        )

        # Improve worksheet readability.
        for worksheet in writer.book.worksheets:
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions

            for column_cells in worksheet.columns:
                maximum_length = 0
                column_letter = column_cells[0].column_letter

                for cell in column_cells:
                    value = "" if cell.value is None else str(cell.value)
                    maximum_length = max(maximum_length, len(value))

                worksheet.column_dimensions[column_letter].width = min(
                    maximum_length + 2,
                    35,
                )

    output.seek(0)
    return output.getvalue()


# ============================================================
# SIDEBAR SETTINGS
# ============================================================
with st.sidebar:
    st.header("Cleaning settings")

    date_format_label = st.selectbox(
        "Date format inside sample_id",
        [
            "DDMMYY — 170325 = 17 Mar 2025",
            "YYMMDD — 250317 = 17 Mar 2025",
            "DDMMYYYY — 17032025 = 17 Mar 2025",
            "YYYYMMDD — 20250317 = 17 Mar 2025",
        ],
        index=0,
    )

    below_detection_rule = st.selectbox(
        "How should <value be handled?",
        [
            "Treat <value as missing",
            "Use half of <value",
            "Use the reporting-limit value",
        ],
        index=0,
    )

    drop_invalid_ids = st.checkbox(
        "Drop rows with invalid sample_id",
        value=False,
    )

    duplicate_rule = st.selectbox(
        "Duplicate sample_id handling",
        [
            "Keep all",
            "Keep first",
            "Keep last",
            "Remove every duplicated sample_id",
        ],
        index=0,
    )


# ============================================================
# DATA SOURCE
# ============================================================
source_type = st.radio(
    "Choose an input method",
    ["Upload a file", "Paste tabular data"],
    horizontal=True,
)

raw_data: Optional[pd.DataFrame] = None

if source_type == "Upload a file":
    uploaded_file = st.file_uploader(
        "Upload an Excel, CSV, TSV, or TXT file",
        type=["xlsx", "xls", "csv", "tsv", "txt"],
    )

    if uploaded_file is not None:
        extension = uploaded_file.name.rsplit(".", 1)[-1].lower()
        file_bytes = uploaded_file.getvalue()

        try:
            if extension in {"xlsx", "xls"}:
                excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
                selected_sheet = st.selectbox(
                    "Select worksheet",
                    excel_file.sheet_names,
                )
                raw_data = pd.read_excel(
                    io.BytesIO(file_bytes),
                    sheet_name=selected_sheet,
                    header=None,
                    dtype=object,
                )
            else:
                separator_choice = st.selectbox(
                    "File separator",
                    [
                        "Auto-detect",
                        "Tab",
                        "Comma",
                        "Semicolon",
                        "Pipe",
                    ],
                    index=0,
                )
                raw_data = read_delimited_bytes(
                    file_bytes,
                    separator_choice,
                )
        except Exception as exc:
            st.error(f"Could not read the uploaded file: {exc}")

else:
    example_text = (
        "sample_ids\tAl (ug/m3)\t\tCr (ng/m3)\t\tCd (ng/m3)\t\n"
        "\tConc\tUncert\tConc\tUncert\tConc\tError\n"
        "APS_170325_PM2.5\t0.000\t0.000\t0.00\t0.00\t0.12\t0.02"
    )

    pasted_text = st.text_area(
        "Paste the table, including both header rows",
        height=220,
        placeholder=example_text,
    )

    if pasted_text.strip():
        try:
            raw_data = read_pasted_text(pasted_text)
        except Exception as exc:
            st.error(f"Could not read the pasted table: {exc}")


# ============================================================
# CLEANING WORKFLOW
# ============================================================
if raw_data is None:
    st.info(
        "Upload a file or paste the laboratory table to begin."
    )
    st.stop()

with st.expander("Raw data preview", expanded=False):
    st.dataframe(raw_data.head(20), use_container_width=True)

try:
    flattened, units_table, analyte_columns = flatten_lab_table(raw_data)
except Exception as exc:
    st.error(f"Header-cleaning error: {exc}")
    st.stop()

parsed_ids = parse_sample_ids(
    flattened["sample_id"],
    date_format_label,
)

working = pd.concat(
    [
        flattened.reset_index(drop=True),
        parsed_ids.reset_index(drop=True),
    ],
    axis=1,
)

# Convert concentration and error/uncertainty columns to numeric.
for column in analyte_columns:
    working[column] = clean_numeric_series(
        working[column],
        below_detection_rule,
    )

# Build one unit value for each analyte directly from the uploaded headers.
# Example: Al (ug/m3) -> Al_unit = ug/m3.
analyte_unit_map: Dict[str, str] = {}

if not units_table.empty:
    for analyte_name, unit_group in units_table.groupby(
        "analyte",
        sort=False,
    ):
        detected_units = (
            unit_group["unit_as_uploaded"]
            .astype("string")
            .str.strip()
        )
        detected_units = detected_units[
            detected_units.notna()
            & detected_units.ne("")
        ]

        if not detected_units.empty:
            analyte_unit_map[str(analyte_name)] = str(
                detected_units.iloc[0]
            )

# ============================================================
# USER-DEFINED SITE NAMES
# ============================================================
valid_site_codes = (
    working["site_code"]
    .dropna()
    .astype(str)
    .str.strip()
)
site_codes = list(dict.fromkeys(valid_site_codes.tolist()))

site_mapping: Dict[str, str] = {}

if site_codes:
    first_code = site_codes[0]
    st.subheader("Name the monitoring site")

    st.info(
        f"The first valid sample ID begins with **{first_code}**. "
        "Enter the preferred full site name below. The code is retained "
        "in the site_code column, while the chosen name is stored in id."
    )

    with st.expander(
        "Site-code mapping",
        expanded=True,
    ):
        for position, code in enumerate(site_codes):
            site_mapping[code] = st.text_input(
                f"Preferred site name for {code}",
                value=code,
                key=f"site_name_{position}",
                help=(
                    f"Example: replace {code} with the full station "
                    "or sampling location name."
                ),
            ).strip() or code
else:
    st.warning(
        "No valid site code was extracted from the sample IDs."
    )

working["id"] = working["site_code"].map(site_mapping)
working["id"] = working["id"].fillna(working["site_code"])

# ============================================================
# OPTIONAL ROW CLEANING
# ============================================================
rows_before_rules = len(working)

if drop_invalid_ids:
    working = working[working["sample_id_valid"]].copy()

duplicate_mask = working["sample_id"].duplicated(keep=False)
duplicates_before_rule = int(duplicate_mask.sum())

if duplicate_rule == "Keep first":
    working = working.drop_duplicates(
        subset=["sample_id"],
        keep="first",
    )
elif duplicate_rule == "Keep last":
    working = working.drop_duplicates(
        subset=["sample_id"],
        keep="last",
    )
elif duplicate_rule == "Remove every duplicated sample_id":
    working = working[
        ~working["sample_id"].duplicated(keep=False)
    ].copy()

working = working.reset_index(drop=True)

# ============================================================
# FINAL COLUMN ORDER AND UNIT COLUMNS
# ============================================================
metadata_columns = [
    "sample_id",
    "date",
    "id",
    "site_code",
    "pollutant",
]

# These are the numeric concentration and error/uncertainty columns.
numeric_analyte_columns = [
    column
    for column in working.columns
    if column not in {
        *metadata_columns,
        "sample_id_valid",
    }
]

# Identify each analyte while preserving the order in the uploaded data.
analyte_names: List[str] = []
for column in numeric_analyte_columns:
    analyte_name = (
        column.removesuffix("_error")
        if column.endswith("_error")
        else column
    )
    if analyte_name not in analyte_names:
        analyte_names.append(analyte_name)

# Add a unit column beside every analyte group.
# Example order: Al, Al_error, Al_unit, Cr, Cr_error, Cr_unit.
ordered_measurement_columns: List[str] = []
for analyte_name in analyte_names:
    concentration_column = analyte_name
    error_column = f"{analyte_name}_error"
    unit_column = f"{analyte_name}_unit"

    if concentration_column in working.columns:
        ordered_measurement_columns.append(concentration_column)

    if error_column in working.columns:
        ordered_measurement_columns.append(error_column)

    working[unit_column] = analyte_unit_map.get(
        analyte_name,
        pd.NA,
    )
    ordered_measurement_columns.append(unit_column)

cleaned_output = working[
    metadata_columns + ordered_measurement_columns
].copy()

cleaned_output["date"] = pd.to_datetime(
    cleaned_output["date"],
    errors="coerce",
).dt.date

# Also create a tidy long-format table with one standard unit column.
long_frames: List[pd.DataFrame] = []

for analyte_name in analyte_names:
    concentration_column = analyte_name
    error_column = f"{analyte_name}_error"

    analyte_frame = cleaned_output[metadata_columns].copy()
    analyte_frame["metal"] = analyte_name
    analyte_frame["concentration"] = (
        cleaned_output[concentration_column]
        if concentration_column in cleaned_output.columns
        else pd.NA
    )
    analyte_frame["error"] = (
        cleaned_output[error_column]
        if error_column in cleaned_output.columns
        else pd.NA
    )
    analyte_frame["unit"] = analyte_unit_map.get(
        analyte_name,
        pd.NA,
    )
    long_frames.append(analyte_frame)

if long_frames:
    cleaned_long_output = pd.concat(
        long_frames,
        ignore_index=True,
    )
else:
    cleaned_long_output = pd.DataFrame(
        columns=(
            metadata_columns
            + ["metal", "concentration", "error", "unit"]
        )
    )

# ============================================================
# QUALITY-CONTROL SUMMARY
# ============================================================
sample_id_valid_mask = (
    parsed_ids["sample_id_valid"]
    .fillna(False)
    .astype(bool)
)

invalid_id_count = int((~sample_id_valid_mask).sum())
valid_id_count = int(sample_id_valid_mask.sum())
missing_numeric_count = int(
    cleaned_output[numeric_analyte_columns].isna().sum().sum()
) if numeric_analyte_columns else 0

qc_records = [
    {"check": "Rows read after header cleaning", "value": len(flattened)},
    {"check": "Valid sample IDs", "value": valid_id_count},
    {"check": "Invalid sample IDs", "value": invalid_id_count},
    {
        "check": "Rows matching duplicated sample IDs before duplicate rule",
        "value": duplicates_before_rule,
    },
    {
        "check": "Rows removed by selected rules",
        "value": rows_before_rules - len(working),
    },
    {"check": "Rows in final output", "value": len(cleaned_output)},
    {
        "check": "Missing values across analyte columns",
        "value": missing_numeric_count,
    },
]
qc_summary = pd.DataFrame(qc_records)

st.subheader("Cleaning summary")
metric_columns = st.columns(4)
metric_columns[0].metric("Final rows", len(cleaned_output))
metric_columns[1].metric("Valid sample IDs", valid_id_count)
metric_columns[2].metric("Invalid sample IDs", invalid_id_count)
metric_columns[3].metric("Detected metals", len(analyte_names))

# Build a complete invalid-ID report from the original parsed rows.
invalid_mask = (
    ~parsed_ids["sample_id_valid"]
    .fillna(False)
    .astype(bool)
    .to_numpy()
)

invalid_ids_table = pd.DataFrame(
    {
        "source_row": (
            pd.Series(range(1, len(flattened) + 1))
            .loc[invalid_mask]
            .to_numpy()
        ),
        "sample_id": (
            flattened.loc[invalid_mask, "sample_id"]
            .astype("string")
            .fillna("")
            .to_numpy()
        ),
        "reason": (
            parsed_ids.loc[invalid_mask, "invalid_reason"]
            .astype("string")
            .fillna("Invalid sample ID")
            .to_numpy()
        ),
    }
)

if invalid_id_count:
    st.warning(
        f"{invalid_id_count} sample ID(s) could not be parsed. "
        "Supported structures include APS_170325_PM2.5 and "
        "1_031118PM10."
    )

    with st.expander(
        f"View all invalid sample IDs ({invalid_id_count})",
        expanded=True,
    ):
        st.dataframe(
            invalid_ids_table,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download invalid sample IDs",
            data=invalid_ids_table.to_csv(index=False).encode("utf-8-sig"),
            file_name="invalid_sample_ids.csv",
            mime="text/csv",
            use_container_width=True,
        )
else:
    st.success("All sample IDs were parsed successfully.")

st.subheader("Cleaned data preview")
preview_tab_wide, preview_tab_long = st.tabs(
    ["Wide format", "Long format"]
)

with preview_tab_wide:
    st.dataframe(
        cleaned_output.head(100),
        use_container_width=True,
        hide_index=True,
    )

with preview_tab_long:
    st.dataframe(
        cleaned_long_output.head(500),
        use_container_width=True,
        hide_index=True,
    )

with st.expander("Detected units", expanded=False):
    st.dataframe(
        units_table,
        use_container_width=True,
        hide_index=True,
    )

with st.expander("Quality-control details", expanded=False):
    st.dataframe(
        qc_summary,
        use_container_width=True,
        hide_index=True,
    )

# ============================================================
# DOWNLOADS
# ============================================================
wide_csv_bytes = cleaned_output.to_csv(
    index=False
).encode("utf-8-sig")
long_csv_bytes = cleaned_long_output.to_csv(
    index=False
).encode("utf-8-sig")
excel_bytes = create_excel_download(
    cleaned_output,
    cleaned_long_output,
    units_table,
    qc_summary,
    invalid_ids_table,
)

st.subheader("Download cleaned data")
download_columns = st.columns(3)

download_columns[0].download_button(
    "Download wide CSV",
    data=wide_csv_bytes,
    file_name="cleaned_metal_data_wide.csv",
    mime="text/csv",
    use_container_width=True,
)

download_columns[1].download_button(
    "Download long CSV",
    data=long_csv_bytes,
    file_name="cleaned_metal_data_long.csv",
    mime="text/csv",
    use_container_width=True,
)

download_columns[2].download_button(
    "Download cleaned Excel",
    data=excel_bytes,
    file_name="cleaned_metal_data.xlsx",
    mime=(
        "application/vnd.openxmlformats-officedocument."
        "spreadsheetml.sheet"
    ),
    use_container_width=True,
)
