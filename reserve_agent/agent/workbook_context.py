from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import pandas as pd


MAX_STRING_LENGTH = 180
DEFAULT_MAX_SHEETS = 30
DEFAULT_MAX_SAMPLE_ROWS = 15
DEFAULT_MAX_TEXT_ITEMS = 30
DEFAULT_MAX_COLUMNS = 50
DEFAULT_MAX_SCAN_ROWS = 150


def _trim_string(value: Any, max_length: int = MAX_STRING_LENGTH) -> str:
    """Convert a value to a trimmed string."""
    text = str(value).strip()

    if len(text) <= max_length:
        return text

    return text[: max_length - 3] + "..."


def _is_missing(value: Any) -> bool:
    """Return True if value is empty or missing."""
    if value is None:
        return True

    try:
        if pd.isna(value):
            return True
    except Exception:
        pass

    if isinstance(value, str) and not value.strip():
        return True

    return False


def _excel_column_label(column_number: int) -> str:
    """
    Convert a 1-based Excel column number to Excel-style letters.

    Example:
    1 -> A
    2 -> B
    27 -> AA
    """
    if column_number <= 0:
        return ""

    label = ""
    n = column_number

    while n > 0:
        n, remainder = divmod(n - 1, 26)
        label = chr(65 + remainder) + label

    return label


def _excel_cell_label(row_number: int, column_number: int) -> str:
    """Return an Excel-style cell label such as A1 or C8."""
    col_label = _excel_column_label(column_number)
    if not col_label:
        return str(row_number)
    return f"{col_label}{row_number}"


def _safe_scalar(value: Any) -> Any:
    """
    Convert pandas/numpy scalar values into JSON-friendly Python values.

    The workbook context will later be converted to JSON and sent to the LLM,
    so it should avoid pandas-specific objects.
    """
    if _is_missing(value):
        return None

    try:
        if hasattr(value, "item"):
            value = value.item()
    except Exception:
        pass

    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)

    if isinstance(value, float):
        return round(value, 4)

    if isinstance(value, str):
        return _trim_string(value)

    if isinstance(value, (int, bool)):
        return value

    return _trim_string(value)


def _records_from_dataframe(
    df: pd.DataFrame,
    row_offset: int = 0,
    include_excel_row: bool = False,
) -> list[dict[str, Any]]:
    """
    Convert a DataFrame to JSON-safe row records.

    row_offset is 0-based. If include_excel_row=True, each sample row will
    include its original Excel row number.
    """
    if df is None or df.empty:
        return []

    records: list[dict[str, Any]] = []

    for idx, row in df.iterrows():
        record: dict[str, Any] = {}

        if include_excel_row:
            record["__excel_row"] = int(row_offset + idx + 1)

        for col, val in row.items():
            record[_trim_string(col)] = _safe_scalar(val)

        records.append(record)

    return records


def _json_safe(value: Any) -> Any:
    """Convert common objects into JSON-friendly structures."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]

    if isinstance(value, pd.DataFrame):
        return {
            "shape": [int(value.shape[0]), int(value.shape[1])],
            "columns": [_trim_string(col) for col in value.columns.tolist()],
            "sample_rows": _records_from_dataframe(value.head(DEFAULT_MAX_SAMPLE_ROWS)),
        }

    if isinstance(value, pd.Series):
        return [_json_safe(v) for v in value.tolist()]

    if is_dataclass(value):
        return _json_safe(asdict(value))

    return _safe_scalar(value)


def _make_unique_columns(columns: list[Any]) -> list[str]:
    """Make column names unique and readable."""
    result: list[str] = []
    seen: dict[str, int] = {}

    for idx, col in enumerate(columns, start=1):
        if _is_missing(col):
            base = f"Column {idx}"
        else:
            base = _trim_string(col, max_length=80)

        if base in seen:
            seen[base] += 1
            name = f"{base}_{seen[base]}"
        else:
            seen[base] = 1
            name = base

        result.append(name)

    return result


def _used_range_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """
    Find the non-empty used range of a worksheet.

    Many real Excel sheets do not start from A1. They may have blank rows,
    blank columns, titles, notes, or explanatory text before the real table.

    This function trims only the outer empty margins and records the original
    top-left position. Therefore the summary can still report the real Excel
    row and column positions.

    This implementation avoids DataFrame.applymap for pandas compatibility.
    """
    if df is None or df.empty:
        return pd.DataFrame(), 0, 0

    values = df.to_numpy(dtype=object)

    non_empty_mask = pd.DataFrame(
        [
            [not _is_missing(value) for value in row]
            for row in values
        ],
        index=df.index,
        columns=df.columns,
    )

    if not non_empty_mask.to_numpy().any():
        return pd.DataFrame(), 0, 0

    row_positions = non_empty_mask.any(axis=1)
    col_positions = non_empty_mask.any(axis=0)

    first_row = int(row_positions[row_positions].index[0])
    last_row = int(row_positions[row_positions].index[-1])
    first_col = int(col_positions[col_positions].index[0])
    last_col = int(col_positions[col_positions].index[-1])

    used = df.iloc[first_row : last_row + 1, first_col : last_col + 1].copy()

    # Reset index/columns for easier iloc-based processing, but keep offsets.
    used = used.reset_index(drop=True)
    used.columns = range(used.shape[1])

    return used, first_row, first_col

def _guess_header_row(df: pd.DataFrame, max_scan_rows: int = 30) -> int | None:
    """
    Guess a likely header row in a sheet.

    This is intentionally lightweight. The real Excel detection logic remains
    in the data module. Here we only need a readable summary for the chat agent.
    """
    if df is None or df.empty:
        return None

    limit = min(max_scan_rows, len(df))
    best_idx: int | None = None
    best_score = -1.0

    for row_idx in range(limit):
        row = df.iloc[row_idx]
        values = [v for v in row.tolist() if not _is_missing(v)]

        if len(values) < 2:
            continue

        string_like = sum(isinstance(v, str) for v in values)
        unique_count = len({_trim_string(v, 40).lower() for v in values})

        # Prefer rows with more non-empty and distinct labels.
        # Do not require all cells to be strings because triangle headers may
        # contain development periods such as 12, 24, 36.
        score = len(values) + 0.5 * unique_count + 0.3 * string_like

        if score > best_score:
            best_score = score
            best_idx = row_idx

    return best_idx


def _table_preview_from_raw(
    raw_df: pd.DataFrame,
    header_row: int | None,
    max_sample_rows: int = DEFAULT_MAX_SAMPLE_ROWS,
    max_columns: int = DEFAULT_MAX_COLUMNS,
    row_offset: int = 0,
) -> tuple[list[str], list[dict[str, Any]]]:
    """
    Build a compact preview table from a raw sheet.

    If a header row is found, use it as column names and sample rows below it.
    Otherwise, create generic column names and sample from the first rows.

    row_offset is the original Excel row offset of raw_df.
    """
    if raw_df is None or raw_df.empty:
        return [], []

    limited_df = raw_df.iloc[:, :max_columns].copy()

    if header_row is not None and header_row < len(limited_df):
        columns = _make_unique_columns(limited_df.iloc[header_row].tolist())
        data = limited_df.iloc[header_row + 1 : header_row + 1 + max_sample_rows].copy()
        sample_row_offset = row_offset + header_row + 1
    else:
        columns = [f"Column {i}" for i in range(1, limited_df.shape[1] + 1)]
        data = limited_df.head(max_sample_rows).copy()
        sample_row_offset = row_offset

    data.columns = columns
    data = data.dropna(how="all").reset_index(drop=True)

    return columns, _records_from_dataframe(
        data,
        row_offset=sample_row_offset,
        include_excel_row=True,
    )


def _numeric_summary_from_raw(
    raw_df: pd.DataFrame,
    columns: list[str],
    header_row: int | None,
    max_columns: int = DEFAULT_MAX_COLUMNS,
) -> dict[str, Any]:
    """Summarize numeric columns in the detected table area."""
    if raw_df is None or raw_df.empty or not columns:
        return {}

    data_start = header_row + 1 if header_row is not None else 0
    data = raw_df.iloc[data_start:, : min(len(columns), max_columns)].copy()

    if data.empty:
        return {}

    data.columns = columns[: data.shape[1]]

    summary: dict[str, Any] = {}

    for col in data.columns:
        series = pd.to_numeric(data[col], errors="coerce")
        non_missing = series.dropna()

        if non_missing.empty:
            continue

        summary[str(col)] = {
            "count": int(non_missing.count()),
            "min": round(float(non_missing.min()), 4),
            "max": round(float(non_missing.max()), 4),
            "mean": round(float(non_missing.mean()), 4),
        }

    return summary


def _text_preview_from_raw(
    raw_df: pd.DataFrame,
    max_items: int = DEFAULT_MAX_TEXT_ITEMS,
    max_scan_rows: int = DEFAULT_MAX_SCAN_ROWS,
    max_columns: int = DEFAULT_MAX_COLUMNS,
    row_offset: int = 0,
    col_offset: int = 0,
) -> list[dict[str, Any]]:
    """
    Collect a small number of text cells from a sheet.

    This helps the agent answer questions about notes, descriptions, parameters,
    or assumptions without sending the full workbook.

    row_offset and col_offset are 0-based original Excel offsets.
    """
    if raw_df is None or raw_df.empty:
        return []

    result: list[dict[str, Any]] = []

    scan_rows = min(max_scan_rows, len(raw_df))
    scan_cols = min(max_columns, raw_df.shape[1])

    for r in range(scan_rows):
        for c in range(scan_cols):
            value = raw_df.iat[r, c]

            if _is_missing(value):
                continue

            if not isinstance(value, str):
                continue

            text = value.strip()

            if not text:
                continue

            if len(text) < 2:
                continue

            excel_row = row_offset + r + 1
            excel_col = col_offset + c + 1

            result.append(
                {
                    "row": excel_row,
                    "column": excel_col,
                    "cell": _excel_cell_label(excel_row, excel_col),
                    "text": _trim_string(text),
                }
            )

            if len(result) >= max_items:
                return result

    return result


def _infer_sheet_role(
    sheet_name: str,
    columns: list[str],
    text_preview: list[dict[str, Any]],
    used_for_modeling: bool,
) -> str:
    """
    Infer a rough sheet role for workbook-level Q&A.

    This is not the official Excel detector. It only helps the chat agent
    describe the workbook structure. The rule order matters: explicit sheet
    names such as Claims, Projection, Comparison, Disclaimer should be trusted
    before weaker keyword matches from sample rows.
    """
    if used_for_modeling:
        return "modeling_data"

    name_text = sheet_name.lower()
    column_text = " ".join(columns).lower()
    note_text = " ".join(str(item.get("text", "")) for item in text_preview).lower()
    combined = f"{name_text} {column_text} {note_text}"

    # 1. Strong name-based rules first.
    if any(key in name_text for key in ["disclaimer", "readme", "instruction", "说明", "备注"]):
        return "notes_or_instructions"

    if any(key in name_text for key in ["projection", "method", "ibnr", "预测", "方法"]):
        return "projection_or_methods"

    if any(key in name_text for key in ["comparison", "compare", "summary", "result", "结果", "比较", "汇总"]):
        return "results_or_summary"

    if any(key in name_text for key in ["claims data", "claim data", "claims", "claim", "赔案", "赔款"]):
        return "claims_or_loss_data"

    if any(key in name_text for key in ["exposure", "exposures", "暴露"]):
        return "exposure_or_premium"

    if any(key in name_text for key in ["policy", "policies", "保单"]):
        return "policy_or_premium_data"

    # 2. Then use column/text keywords.
    if any(
        key in combined
        for key in [
            "claim id",
            "claimant",
            "loss date",
            "reporting date",
            "paid",
            "outstanding",
            "incurred",
            "recoveries",
            "claim status",
            "claim description",
            "claim",
            "claims",
            "loss",
            "赔案",
            "赔款",
            "已付",
            "未决",
            "已发生",
        ]
    ):
        return "claims_or_loss_data"

    if any(
        key in combined
        for key in [
            "projection",
            "method 1",
            "method 2",
            "method 3",
            "method 4",
            "ultimate",
            "ibnr",
            "gross up",
            "last diagonal",
            "预测",
            "方法",
            "最终",
        ]
    ):
        return "projection_or_methods"

    if any(
        key in combined
        for key in [
            "comparison",
            "compare",
            "summary",
            "result",
            "output",
            "report",
            "结果",
            "报告",
            "比较",
            "汇总",
        ]
    ):
        return "results_or_summary"

    if any(
        key in combined
        for key in [
            "exposure",
            "turnover",
            "employee",
            "earned premium",
            "premium",
            "gross premium",
            "net premium",
            "brokerage",
            "limit amount",
            "deductible",
            "保费",
            "暴露",
            "营业额",
            "员工",
        ]
    ):
        return "exposure_or_premium"

    if any(
        key in combined
        for key in [
            "policy id",
            "inception date",
            "expiry date",
            "territory",
            "class of business",
            "insured",
            "layer type",
            "保单",
            "起保",
            "到期",
        ]
    ):
        return "policy_or_premium_data"

    if any(
        key in combined
        for key in [
            "assumption",
            "parameter",
            "expected loss",
            "loss ratio",
            "inflation",
            "selected factor",
            "假设",
            "参数",
            "赔付率",
            "通胀",
        ]
    ):
        return "assumptions_or_parameters"

    if any(
        key in combined
        for key in [
            "triangle",
            "development",
            "accident year",
            "policy year",
            "valuation",
            "三角",
            "事故年",
            "保单年",
            "发展期",
        ]
    ):
        return "triangle_or_development_data"

    if any(key in combined for key in ["note", "readme", "instruction", "说明", "备注"]):
        return "notes_or_instructions"

    return "unknown_or_reference"


def _object_summary(obj: Any, max_items: int = 8) -> Any:
    """
    Convert an arbitrary load_result-related object to a small summary.

    This avoids sending large internal objects to the LLM while preserving
    useful metadata such as region boundaries or detector output.
    """
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return _json_safe(obj)

    if isinstance(obj, (list, tuple)):
        return [_object_summary(item, max_items=max_items) for item in list(obj)[:max_items]]

    if isinstance(obj, dict):
        compact: dict[str, Any] = {}
        for key, value in list(obj.items())[:max_items]:
            compact[str(key)] = _object_summary(value, max_items=max_items)
        return compact

    if is_dataclass(obj):
        return _object_summary(asdict(obj), max_items=max_items)

    if isinstance(obj, pd.DataFrame):
        return _json_safe(obj.head(DEFAULT_MAX_SAMPLE_ROWS))

    if hasattr(obj, "__dict__"):
        compact: dict[str, Any] = {}

        for key, value in vars(obj).items():
            if key.startswith("_"):
                continue

            lowered = key.lower()

            # Avoid accidentally embedding large tables or raw data.
            if any(marker in lowered for marker in ["dataframe", "source_table", "raw", "table"]):
                if isinstance(value, pd.DataFrame):
                    compact[key] = {
                        "shape": [int(value.shape[0]), int(value.shape[1])],
                        "columns": [_trim_string(col) for col in value.columns.tolist()],
                    }
                continue

            compact[key] = _object_summary(value, max_items=max_items)

            if len(compact) >= max_items:
                break

        return compact

    return _json_safe(obj)


def _load_result_summary(load_result: Any, selected_sheet: str | None = None) -> dict[str, Any]:
    """
    Summarize the Excel detection result produced by the data loader.

    The code is defensive because load_result is owned by the data module and
    may evolve as teammates improve Excel detection.
    """
    if load_result is None:
        return {}

    source_sheet_name = getattr(load_result, "source_sheet_name", None)
    format_name = getattr(load_result, "format_name", None)
    warnings = getattr(load_result, "warnings", None)
    candidates = getattr(load_result, "candidates", None)
    region = getattr(load_result, "region", None)
    source_table = getattr(load_result, "source_table", None)

    summary: dict[str, Any] = {
        "selected_sheet": selected_sheet,
        "source_sheet_name": source_sheet_name,
        "format_name": format_name,
        "warning_count": len(warnings) if isinstance(warnings, list) else 0,
        "warnings": _json_safe(warnings or []),
        "candidate_count": len(candidates) if isinstance(candidates, list) else None,
        "region": _object_summary(region),
    }

    if isinstance(source_table, pd.DataFrame):
        summary["source_table"] = {
            "shape": [int(source_table.shape[0]), int(source_table.shape[1])],
            "columns": [_trim_string(col) for col in source_table.columns.tolist()],
            "sample_rows": _records_from_dataframe(
                source_table.head(DEFAULT_MAX_SAMPLE_ROWS)
            ),
        }

    if candidates:
        summary["candidate_preview"] = _object_summary(candidates, max_items=5)

    return _json_safe(summary)


def _read_sheet_raw(
    excel_file: pd.ExcelFile,
    sheet_name: str,
) -> pd.DataFrame:
    """
    Read one sheet as raw cells without assuming the first row is a header.
    """
    return excel_file.parse(sheet_name=sheet_name, header=None, dtype=object)


def _build_sheet_summary(
    excel_file: pd.ExcelFile,
    sheet_name: str,
    used_sheet_name: str | None,
    max_sample_rows: int = DEFAULT_MAX_SAMPLE_ROWS,
    max_text_items: int = DEFAULT_MAX_TEXT_ITEMS,
    max_columns: int = DEFAULT_MAX_COLUMNS,
) -> dict[str, Any]:
    """Build a compact summary for one Excel sheet."""
    try:
        raw = _read_sheet_raw(excel_file, sheet_name)
    except Exception as exc:
        return {
            "sheet_name": sheet_name,
            "read_error": str(exc),
            "used_for_modeling": sheet_name == used_sheet_name,
        }

    used_range, row_offset, col_offset = _used_range_frame(raw)
    used_for_modeling = sheet_name == used_sheet_name

    if used_range.empty:
        return {
            "sheet_name": sheet_name,
            "used_for_modeling": used_for_modeling,
            "raw_row_count": int(raw.shape[0]),
            "raw_column_count": int(raw.shape[1]),
            "row_count": 0,
            "column_count": 0,
            "used_range": None,
            "likely_role": "empty_sheet",
            "columns": [],
            "sample_rows": [],
            "text_preview": [],
            "numeric_summary": {},
        }

    header_row = _guess_header_row(used_range)

    columns, sample_rows = _table_preview_from_raw(
        used_range,
        header_row=header_row,
        max_sample_rows=max_sample_rows,
        max_columns=max_columns,
        row_offset=row_offset,
    )

    text_preview = _text_preview_from_raw(
        used_range,
        max_items=max_text_items,
        max_columns=max_columns,
        row_offset=row_offset,
        col_offset=col_offset,
    )

    numeric_summary = _numeric_summary_from_raw(
        used_range,
        columns=columns,
        header_row=header_row,
        max_columns=max_columns,
    )

    likely_role = _infer_sheet_role(
        sheet_name=sheet_name,
        columns=columns,
        text_preview=text_preview,
        used_for_modeling=used_for_modeling,
    )

    top_left_row = row_offset + 1
    top_left_col = col_offset + 1
    bottom_right_row = row_offset + used_range.shape[0]
    bottom_right_col = col_offset + used_range.shape[1]

    return {
        "sheet_name": sheet_name,
        "used_for_modeling": used_for_modeling,
        "likely_role": likely_role,
        "raw_row_count": int(raw.shape[0]),
        "raw_column_count": int(raw.shape[1]),
        "row_count": int(used_range.shape[0]),
        "column_count": int(used_range.shape[1]),
        "used_range": {
            "top_left_row": top_left_row,
            "top_left_column": top_left_col,
            "top_left_cell": _excel_cell_label(top_left_row, top_left_col),
            "bottom_right_row": bottom_right_row,
            "bottom_right_column": bottom_right_col,
            "bottom_right_cell": _excel_cell_label(bottom_right_row, bottom_right_col),
        },
        "guessed_header_row": (
            row_offset + header_row + 1 if header_row is not None else None
        ),
        "guessed_header_row_relative_to_used_range": (
            header_row + 1 if header_row is not None else None
        ),
        "columns": columns,
        "sample_rows": sample_rows,
        "text_preview": text_preview,
        "numeric_summary": numeric_summary,
        "summary_note": (
            "This is a compact sheet-level summary, not the full raw worksheet. "
            "The used_range records where the non-empty area starts and ends in the original Excel sheet. "
            "Specific cell-level questions may require reading the original file."
        ),
    }


def build_workbook_context(
    file_path: str | Path,
    selected_sheet: str | None = None,
    load_result: Any | None = None,
    max_sheets: int = DEFAULT_MAX_SHEETS,
    max_sample_rows: int = DEFAULT_MAX_SAMPLE_ROWS,
    max_text_items: int = DEFAULT_MAX_TEXT_ITEMS,
    max_columns: int = DEFAULT_MAX_COLUMNS,
) -> dict[str, Any]:
    """
    Build a workbook-level context for the follow-up chat agent.

    This function summarizes every sheet in the uploaded Excel workbook so that
    the agent can answer questions about the broader workbook, not only the
    reserving model outputs.

    It does NOT send the full raw workbook to the LLM. It only returns compact
    sheet summaries, sample rows, text previews, numeric summaries, and loader
    metadata.
    """
    path = Path(file_path)

    context: dict[str, Any] = {
        "workbook_name": path.name,
        "workbook_path_note": (
            "The path is used locally by the app only and should not be treated as user-facing evidence."
        ),
        "selected_sheet": selected_sheet,
        "load_result_summary": _load_result_summary(load_result, selected_sheet=selected_sheet),
        "sheets": [],
        "sheet_count": 0,
        "used_sheet_names": [],
        "unused_sheet_names": [],
        "limitations": [
            "The workbook context contains compact summaries rather than the complete raw Excel workbook.",
            "The agent can answer sheet-level and summary-level questions, but may not know every individual cell.",
            "If a question asks for a specific cell not included in sample rows or text previews, the agent should say the context is insufficient.",
        ],
    }

    try:
        excel_file = pd.ExcelFile(path)
    except Exception as exc:
        context["read_error"] = str(exc)
        return _json_safe(context)

    sheet_names = excel_file.sheet_names
    context["sheet_count"] = len(sheet_names)
    context["sheet_names"] = sheet_names

    source_sheet = None
    if load_result is not None:
        source_sheet = getattr(load_result, "source_sheet_name", None)

    used_sheet_name = source_sheet or selected_sheet

    if len(sheet_names) > max_sheets:
        context["sheet_limit_note"] = (
            f"Workbook has {len(sheet_names)} sheets; only the first {max_sheets} are summarized."
        )

    summarized_sheet_names = sheet_names[:max_sheets]

    sheet_summaries: list[dict[str, Any]] = []

    for sheet_name in summarized_sheet_names:
        sheet_summary = _build_sheet_summary(
            excel_file=excel_file,
            sheet_name=sheet_name,
            used_sheet_name=used_sheet_name,
            max_sample_rows=max_sample_rows,
            max_text_items=max_text_items,
            max_columns=max_columns,
        )
        sheet_summaries.append(sheet_summary)

    context["sheets"] = sheet_summaries

    used_sheet_names = [
        sheet["sheet_name"]
        for sheet in sheet_summaries
        if sheet.get("used_for_modeling")
    ]
    unused_sheet_names = [
        sheet["sheet_name"]
        for sheet in sheet_summaries
        if not sheet.get("used_for_modeling")
    ]

    context["used_sheet_names"] = used_sheet_names
    context["unused_sheet_names"] = unused_sheet_names

    role_counts: dict[str, int] = {}
    for sheet in sheet_summaries:
        role = str(sheet.get("likely_role", "unknown_or_reference"))
        role_counts[role] = role_counts.get(role, 0) + 1

    context["sheet_role_counts"] = role_counts

    context["workbook_summary"] = {
        "workbook_name": path.name,
        "sheet_count": len(sheet_names),
        "summarized_sheet_count": len(sheet_summaries),
        "used_sheet_names": used_sheet_names,
        "unused_sheet_names": unused_sheet_names,
        "source_sheet_name": source_sheet,
        "selected_sheet": selected_sheet,
        "format_name": context["load_result_summary"].get("format_name"),
        "candidate_count": context["load_result_summary"].get("candidate_count"),
        "warning_count": context["load_result_summary"].get("warning_count"),
        "sheet_role_counts": role_counts,
    }

    return _json_safe(context)