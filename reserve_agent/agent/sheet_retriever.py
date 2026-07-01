from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd


MAX_DETAIL_SHEETS = 1
MAX_DETAIL_ROWS = 80
MAX_DETAIL_COLUMNS = 50
MAX_TEXT_CELLS = 80
MAX_KEYWORD_REGIONS = 8
REGION_ROW_WINDOW = 6
REGION_COL_WINDOW = 8
MAX_STRING_LENGTH = 160

MAX_TABLE_CANDIDATES = 5
MAX_TABLE_ROWS = 30
MAX_TABLE_COLUMNS = 12
MIN_TABLE_ROWS = 3
MIN_TABLE_COLS = 3


def _trim_string(value: Any, max_length: int = MAX_STRING_LENGTH) -> str:
    text = str(value).strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def _is_missing(value: Any) -> bool:
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


def _safe_scalar(value: Any) -> Any:
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

    if isinstance(value, (int, bool, str)):
        return _trim_string(value)

    return _trim_string(value)


def _excel_column_label(column_number: int) -> str:
    """Convert 1-based column number to Excel letters."""
    if column_number <= 0:
        return ""

    label = ""
    n = column_number

    while n > 0:
        n, remainder = divmod(n - 1, 26)
        label = chr(65 + remainder) + label

    return label


def _excel_cell_label(row_number: int, column_number: int) -> str:
    return f"{_excel_column_label(column_number)}{row_number}"


def _normalize_text(text: Any) -> str:
    return str(text or "").lower().replace(" ", "").replace("_", "").replace("-", "")


def _read_raw_sheet(file_path: str | Path, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        header=None,
        dtype=object,
    )


def _used_range_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """
    Trim outer blank rows/columns while keeping original Excel offsets.

    Returns:
        used_df, row_offset, col_offset

    row_offset and col_offset are 0-based positions in original worksheet.
    """
    if df is None or df.empty:
        return pd.DataFrame(), 0, 0

    values = df.to_numpy(dtype=object)

    non_empty_mask = pd.DataFrame(
        [[not _is_missing(value) for value in row] for row in values],
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
    used = used.reset_index(drop=True)
    used.columns = range(used.shape[1])

    return used, first_row, first_col


def _rows_as_records(
    df: pd.DataFrame,
    row_offset: int,
    col_offset: int,
    max_rows: int = MAX_DETAIL_ROWS,
    max_cols: int = MAX_DETAIL_COLUMNS,
) -> list[dict[str, Any]]:
    """
    Convert raw sheet cells into compact row records.

    Columns are represented by Excel letters, such as A, B, C.
    """
    if df is None or df.empty:
        return []

    rows: list[dict[str, Any]] = []
    max_r = min(max_rows, df.shape[0])
    max_c = min(max_cols, df.shape[1])

    for r in range(max_r):
        record: dict[str, Any] = {
            "__excel_row": row_offset + r + 1,
        }

        has_value = False

        for c in range(max_c):
            value = df.iat[r, c]

            if _is_missing(value):
                continue

            excel_col_number = col_offset + c + 1
            excel_col_label = _excel_column_label(excel_col_number)
            record[excel_col_label] = _safe_scalar(value)
            has_value = True

        if has_value:
            rows.append(record)

    return rows


def _text_cells(
    df: pd.DataFrame,
    row_offset: int,
    col_offset: int,
    max_items: int = MAX_TEXT_CELLS,
    max_rows: int = 200,
    max_cols: int = MAX_DETAIL_COLUMNS,
) -> list[dict[str, Any]]:
    """Collect text cells from the used range."""
    if df is None or df.empty:
        return []

    result: list[dict[str, Any]] = []

    scan_rows = min(max_rows, df.shape[0])
    scan_cols = min(max_cols, df.shape[1])

    for r in range(scan_rows):
        for c in range(scan_cols):
            value = df.iat[r, c]

            if not isinstance(value, str):
                continue

            text = value.strip()

            if not text:
                continue

            excel_row = row_offset + r + 1
            excel_col = col_offset + c + 1

            result.append(
                {
                    "cell": _excel_cell_label(excel_row, excel_col),
                    "row": excel_row,
                    "column": excel_col,
                    "text": _trim_string(text),
                }
            )

            if len(result) >= max_items:
                return result

    return result


def _question_keywords(question: str) -> list[str]:
    """Extract rough keywords from the user question for region retrieval."""
    q = question.lower()

    base_keywords = [
        "method",
        "method 1",
        "method 2",
        "method 3",
        "method 4",
        "projection",
        "ultimate",
        "ibnr",
        "claim",
        "claims",
        "paid",
        "incurred",
        "outstanding",
        "reserve",
        "exposure",
        "turnover",
        "premium",
        "comparison",
        "compare",
        "result",
        "summary",
        "policy",
        "loss",
        "delay",
        "development",
        "方法",
        "预测",
        "最终",
        "结果",
        "比较",
        "暴露",
        "保费",
        "赔案",
        "赔款",
        "准备金",
    ]

    found = [key for key in base_keywords if key in q]

    # Add English words from the question.
    words = re.findall(r"[A-Za-z][A-Za-z0-9/%\-]{2,}", question)
    for word in words:
        lowered = word.lower()
        if lowered not in found:
            found.append(lowered)

    # Keep order and remove duplicates.
    unique: list[str] = []
    for item in found:
        if item not in unique:
            unique.append(item)

    return unique[:20]


def _extract_keyword_regions(
    used_df: pd.DataFrame,
    row_offset: int,
    col_offset: int,
    keywords: list[str],
    max_regions: int = MAX_KEYWORD_REGIONS,
) -> list[dict[str, Any]]:
    """
    Find cells containing question keywords and return nearby rectangular regions.
    """
    if used_df is None or used_df.empty or not keywords:
        return []

    regions: list[dict[str, Any]] = []
    seen_ranges: set[tuple[int, int, int, int]] = set()

    for r in range(used_df.shape[0]):
        for c in range(used_df.shape[1]):
            value = used_df.iat[r, c]

            if _is_missing(value):
                continue

            text = str(value).lower()

            matched = [kw for kw in keywords if kw.lower() in text]

            if not matched:
                continue

            top = max(0, r - 2)
            bottom = min(used_df.shape[0] - 1, r + REGION_ROW_WINDOW)
            left = max(0, c - 2)
            right = min(used_df.shape[1] - 1, c + REGION_COL_WINDOW)

            key = (top, bottom, left, right)

            if key in seen_ranges:
                continue

            seen_ranges.add(key)

            region_df = used_df.iloc[top : bottom + 1, left : right + 1].copy()

            excel_top_row = row_offset + top + 1
            excel_bottom_row = row_offset + bottom + 1
            excel_left_col = col_offset + left + 1
            excel_right_col = col_offset + right + 1

            regions.append(
                {
                    "matched_cell": _excel_cell_label(row_offset + r + 1, col_offset + c + 1),
                    "matched_text": _trim_string(value),
                    "matched_keywords": matched,
                    "range": {
                        "top_left_cell": _excel_cell_label(excel_top_row, excel_left_col),
                        "bottom_right_cell": _excel_cell_label(excel_bottom_row, excel_right_col),
                    },
                    "rows": _rows_as_records(
                        region_df,
                        row_offset=row_offset + top,
                        col_offset=col_offset + left,
                        max_rows=REGION_ROW_WINDOW + 3,
                        max_cols=REGION_COL_WINDOW + 3,
                    ),
                }
            )

            if len(regions) >= max_regions:
                return regions

    return regions


def _numeric_summary(
    used_df: pd.DataFrame,
    max_cols: int = MAX_DETAIL_COLUMNS,
) -> dict[str, Any]:
    """
    Basic numeric summary by raw column position.

    This does not assume a header row. It is mainly used to help the Agent
    identify whether the sheet contains numerical result areas.
    """
    if used_df is None or used_df.empty:
        return {}

    summary: dict[str, Any] = {}
    scan_cols = min(max_cols, used_df.shape[1])

    for c in range(scan_cols):
        series = pd.to_numeric(used_df.iloc[:, c], errors="coerce").dropna()

        if series.empty:
            continue

        summary[f"relative_column_{c + 1}"] = {
            "count": int(series.count()),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "mean": round(float(series.mean()), 4),
        }

    return summary


def _row_non_empty_count(df: pd.DataFrame, row_idx: int) -> int:
    """Count non-empty cells in one row."""
    if df is None or df.empty or row_idx >= df.shape[0]:
        return 0

    count = 0
    for value in df.iloc[row_idx].tolist():
        if not _is_missing(value):
            count += 1

    return count


def _column_non_empty_count(df: pd.DataFrame, col_idx: int, top: int, bottom: int) -> int:
    """Count non-empty cells in one column for a row interval."""
    if df is None or df.empty or col_idx >= df.shape[1]:
        return 0

    count = 0
    for r in range(top, bottom + 1):
        if not _is_missing(df.iat[r, col_idx]):
            count += 1

    return count


def _guess_table_header_row(region_df: pd.DataFrame) -> int | None:
    """
    Guess the header row inside a candidate table region.

    Prefer the first row that has several text-like labels.
    """
    if region_df is None or region_df.empty:
        return None

    scan_rows = min(5, region_df.shape[0])
    best_idx: int | None = None
    best_score = -1.0

    for r in range(scan_rows):
        values = [value for value in region_df.iloc[r].tolist() if not _is_missing(value)]

        if len(values) < 2:
            continue

        text_like = sum(isinstance(value, str) for value in values)
        numeric_like = 0

        for value in values:
            try:
                pd.to_numeric(value)
                numeric_like += 1
            except Exception:
                pass

        score = len(values) + 1.5 * text_like - 0.5 * numeric_like

        if score > best_score:
            best_score = score
            best_idx = r

    return best_idx


def _make_table_column_names(header_values: list[Any], width: int) -> list[str]:
    """Make readable and unique table column names."""
    columns: list[str] = []
    seen: dict[str, int] = {}

    for idx in range(width):
        value = header_values[idx] if idx < len(header_values) else None

        if _is_missing(value):
            base = f"Column {idx + 1}"
        else:
            base = _trim_string(value, 80)

        if base in seen:
            seen[base] += 1
            col_name = f"{base}_{seen[base]}"
        else:
            seen[base] = 1
            col_name = base

        columns.append(col_name)

    return columns


def _table_records_from_region(
    region_df: pd.DataFrame,
    header_row: int | None,
    max_rows: int = MAX_TABLE_ROWS,
) -> list[dict[str, Any]]:
    """Convert a candidate table region into row records."""
    if region_df is None or region_df.empty:
        return []

    if header_row is not None and header_row < region_df.shape[0]:
        header_values = region_df.iloc[header_row].tolist()
        columns = _make_table_column_names(header_values, region_df.shape[1])
        data = region_df.iloc[header_row + 1 : header_row + 1 + max_rows].copy()
    else:
        columns = [f"Column {idx + 1}" for idx in range(region_df.shape[1])]
        data = region_df.head(max_rows).copy()

    data = data.dropna(how="all").reset_index(drop=True)
    data.columns = columns[: data.shape[1]]

    records: list[dict[str, Any]] = []

    for _, row in data.iterrows():
        record: dict[str, Any] = {}

        for col, value in row.items():
            if _is_missing(value):
                continue
            record[str(col)] = _safe_scalar(value)

        if record:
            records.append(record)

    return records


def _extract_table_candidates(
    used_df: pd.DataFrame,
    row_offset: int,
    col_offset: int,
    max_tables: int = MAX_TABLE_CANDIDATES,
) -> list[dict[str, Any]]:
    """
    Extract dense rectangular table-like areas from the used range.

    This is intentionally lightweight. It looks for consecutive rows with
    enough non-empty cells, then trims sparse columns inside that row block.
    """
    if used_df is None or used_df.empty:
        return []

    candidates: list[dict[str, Any]] = []
    visited_row_blocks: set[tuple[int, int]] = set()

    r = 0
    while r < used_df.shape[0]:
        row_count = _row_non_empty_count(used_df, r)

        if row_count < MIN_TABLE_COLS:
            r += 1
            continue

        top = r
        bottom = r

        while bottom + 1 < used_df.shape[0] and _row_non_empty_count(used_df, bottom + 1) >= MIN_TABLE_COLS:
            bottom += 1

        if bottom - top + 1 < MIN_TABLE_ROWS:
            r = bottom + 1
            continue

        if (top, bottom) in visited_row_blocks:
            r = bottom + 1
            continue

        visited_row_blocks.add((top, bottom))

        dense_cols: list[int] = []
        for c in range(used_df.shape[1]):
            non_empty = _column_non_empty_count(used_df, c, top, bottom)
            if non_empty >= max(2, int(0.4 * (bottom - top + 1))):
                dense_cols.append(c)

        if len(dense_cols) < MIN_TABLE_COLS:
            r = bottom + 1
            continue

        left = min(dense_cols)
        right = max(dense_cols)

        # Avoid sending very wide tables.
        right = min(right, left + MAX_TABLE_COLUMNS - 1)

        region_df = used_df.iloc[top : bottom + 1, left : right + 1].copy()

        if region_df.empty:
            r = bottom + 1
            continue

        header_row = _guess_table_header_row(region_df)

        excel_top_row = row_offset + top + 1
        excel_bottom_row = row_offset + bottom + 1
        excel_left_col = col_offset + left + 1
        excel_right_col = col_offset + right + 1

        text_values = []
        for value in region_df.head(5).to_numpy(dtype=object).flatten().tolist():
            if isinstance(value, str) and value.strip():
                text_values.append(_trim_string(value, 60))

        records = _table_records_from_region(
            region_df=region_df,
            header_row=header_row,
            max_rows=MAX_TABLE_ROWS,
        )

        candidates.append(
            {
                "range": {
                    "top_left_cell": _excel_cell_label(excel_top_row, excel_left_col),
                    "bottom_right_cell": _excel_cell_label(excel_bottom_row, excel_right_col),
                },
                "shape": [int(region_df.shape[0]), int(region_df.shape[1])],
                "guessed_header_excel_row": (
                    excel_top_row + header_row if header_row is not None else None
                ),
                "text_preview": text_values[:12],
                "records": records,
            }
        )

        if len(candidates) >= max_tables:
            break

        r = bottom + 1

    return candidates

def _find_relevant_sheet_names(
    question: str,
    workbook_context: dict[str, Any] | None,
    max_sheets: int = MAX_DETAIL_SHEETS,
) -> list[str]:
    """
    Find sheet names likely relevant to the question.

    Preference order:
    1. exact or near-exact sheet name match;
    2. partial name match after removing leading numbering;
    3. role and keyword fallback.
    """
    if not isinstance(workbook_context, dict):
        return []

    sheets = workbook_context.get("sheets", [])
    if not isinstance(sheets, list):
        return []

    q_lower = question.lower()
    q_norm = _normalize_text(question)

    exact_matches: list[str] = []
    scored: list[tuple[int, str]] = []

    for sheet in sheets:
        if not isinstance(sheet, dict):
            continue

        sheet_name = str(sheet.get("sheet_name", ""))
        role = str(sheet.get("likely_role", "")).lower()

        if not sheet_name:
            continue

        sheet_lower = sheet_name.lower()
        sheet_norm = _normalize_text(sheet_name)

        # Remove leading numbering like "3. " or "6. "
        sheet_without_prefix = re.sub(r"^\s*\d+\s*[\.\-_:：]\s*", "", sheet_name)
        sheet_without_prefix_lower = sheet_without_prefix.lower()
        sheet_without_prefix_norm = _normalize_text(sheet_without_prefix)

        # If the user directly mentions the sheet name without the numeric prefix,
        # this is a strong exact match. Return these first and do not mix in nearby sheets.
        if sheet_norm and sheet_norm in q_norm:
            exact_matches.append(sheet_name)
            continue

        if sheet_without_prefix_norm and sheet_without_prefix_norm in q_norm:
            exact_matches.append(sheet_name)
            continue

        score = 0

        # Token overlap match.
        raw_tokens = re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]+", sheet_without_prefix_lower)
        useful_tokens = [
            token
            for token in raw_tokens
            if token not in {"the", "and", "of", "in", "to", "a", "an"}
        ]

        matched_tokens = [token for token in useful_tokens if token in q_lower]

        if useful_tokens:
            token_ratio = len(matched_tokens) / len(useful_tokens)
            if token_ratio >= 0.8:
                score += 70
            elif token_ratio >= 0.6:
                score += 40
            elif matched_tokens:
                score += 8 * len(matched_tokens)

        # Role-based fallback.
        if "comparison" in q_lower or "compare" in q_lower or "比较" in question:
            if "comparison" in sheet_lower or "compare" in sheet_lower:
                score += 90
            if "summary" in role or "result" in role:
                score += 30

        if "projection" in q_lower or "预测" in question or "method" in q_lower or "方法" in question:
            if "projection" in sheet_lower:
                score += 40
            if "projection" in role or "method" in role:
                score += 20

        if "result" in q_lower or "结果" in question:
            if "result" in role or "summary" in role or "projection" in role:
                score += 30
            if "result" in sheet_lower or "comparison" in sheet_lower:
                score += 40

        if "exposure" in q_lower or "暴露" in question:
            if "exposure" in sheet_lower or "exposure" in role:
                score += 90

        if "policy" in q_lower or "保单" in question:
            if "policy" in sheet_lower or "policy" in role:
                score += 90

        if "claims" in q_lower or "claim" in q_lower or "赔案" in question or "赔款" in question:
            if "claim" in sheet_lower or "claims" in role:
                score += 90

        if score > 0:
            scored.append((score, sheet_name))

    if exact_matches:
        result: list[str] = []
        for name in exact_matches:
            if name not in result:
                result.append(name)
            if len(result) >= max_sheets:
                break
        return result

    scored.sort(key=lambda item: item[0], reverse=True)

    result = []
    for _, name in scored:
        if name not in result:
            result.append(name)

        if len(result) >= max_sheets:
            break

    return result


def _build_one_sheet_detail(
    file_path: str | Path,
    sheet_name: str,
    question: str,
) -> dict[str, Any]:
    """Build detailed context for one relevant sheet."""
    try:
        raw_df = _read_raw_sheet(file_path, sheet_name)
    except Exception as exc:
        return {
            "sheet_name": sheet_name,
            "read_error": str(exc),
        }

    used_df, row_offset, col_offset = _used_range_frame(raw_df)

    if used_df.empty:
        return {
            "sheet_name": sheet_name,
            "raw_shape": [int(raw_df.shape[0]), int(raw_df.shape[1])],
            "used_range": None,
            "detail_rows": [],
            "text_cells": [],
            "keyword_regions": [],
            "numeric_summary": {},
        }

    top_left_row = row_offset + 1
    top_left_col = col_offset + 1
    bottom_right_row = row_offset + used_df.shape[0]
    bottom_right_col = col_offset + used_df.shape[1]

    keywords = _question_keywords(question)

    return {
        "sheet_name": sheet_name,
        "raw_shape": [int(raw_df.shape[0]), int(raw_df.shape[1])],
        "used_shape": [int(used_df.shape[0]), int(used_df.shape[1])],
        "used_range": {
            "top_left_cell": _excel_cell_label(top_left_row, top_left_col),
            "bottom_right_cell": _excel_cell_label(bottom_right_row, bottom_right_col),
            "top_left_row": top_left_row,
            "top_left_column": top_left_col,
            "bottom_right_row": bottom_right_row,
            "bottom_right_column": bottom_right_col,
        },
        "detail_rows_note": (
            f"Only the first {MAX_DETAIL_ROWS} rows and {MAX_DETAIL_COLUMNS} columns "
            "of the used range are included in detail_rows."
        ),
        "detail_rows": _rows_as_records(
            used_df,
            row_offset=row_offset,
            col_offset=col_offset,
            max_rows=MAX_DETAIL_ROWS,
            max_cols=MAX_DETAIL_COLUMNS,
        ),
        "text_cells": _text_cells(
            used_df,
            row_offset=row_offset,
            col_offset=col_offset,
            max_items=MAX_TEXT_CELLS,
        ),
        "keyword_regions": _extract_keyword_regions(
            used_df,
            row_offset=row_offset,
            col_offset=col_offset,
            keywords=keywords,
            max_regions=MAX_KEYWORD_REGIONS,
        ),
        "table_candidates": _extract_table_candidates(
            used_df,
            row_offset=row_offset,
            col_offset=col_offset,
        ),
        "numeric_summary": _numeric_summary(used_df),
        "limitations": [
            "This is a detailed local extract for the relevant sheet, not necessarily the complete workbook.",
            "Formula text may not be available because pandas usually reads calculated values rather than Excel formulas.",
            "If the requested information is outside detail_rows and keyword_regions, the agent should say the extract is insufficient.",
        ],
    }


def build_sheet_detail_context(
    file_path: str | Path,
    question: str,
    workbook_context: dict[str, Any] | None = None,
    max_sheets: int = MAX_DETAIL_SHEETS,
) -> dict[str, Any]:
    """
    Build on-demand sheet detail context for workbook-related questions.

    This function is intentionally triggered per question. It reads only the
    sheet(s) likely relevant to the question, rather than sending the entire
    workbook to the LLM.
    """
    relevant_sheet_names = _find_relevant_sheet_names(
        question=question,
        workbook_context=workbook_context,
        max_sheets=max_sheets,
    )

    if not relevant_sheet_names:
        return {
            "triggered": False,
            "reason": "No relevant sheet was identified from the question.",
            "sheets": [],
        }

    details = [
        _build_one_sheet_detail(
            file_path=file_path,
            sheet_name=sheet_name,
            question=question,
        )
        for sheet_name in relevant_sheet_names
    ]

    return {
        "triggered": True,
        "question": question,
        "relevant_sheet_names": relevant_sheet_names,
        "sheets": details,
        "limitations": [
            "Only relevant sheets are read on demand.",
            "Large sheets are truncated to a manageable number of rows and columns.",
            "For very specific cells or formulas outside the extracted regions, further retrieval may still be needed.",
        ],
    }