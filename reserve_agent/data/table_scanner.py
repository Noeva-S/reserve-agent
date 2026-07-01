from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import math
import re
from typing import Any

import pandas as pd

from reserve_agent.data.detector import ROLE_ALIASES, normalise_label, parse_development_label


@dataclass(frozen=True)
class TableRegion:
    """A rectangular table in a raw sheet loaded with header=None.

    `end_row` and `end_col` use Python's exclusive slicing convention.
    """

    header_row: int
    start_col: int
    end_row: int
    end_col: int
    score: float = 0.0

    @property
    def start_row(self) -> int:
        return self.header_row


def _is_empty(value: Any) -> bool:
    if pd.isna(value):
        return True
    return isinstance(value, str) and not value.strip()


def _looks_like_development(value: Any) -> bool:
    return parse_development_label(value) is not None


def _header_score(values: list[Any]) -> float:
    normalised = {normalise_label(value) for value in values if not _is_empty(value)}
    non_empty = [value for value in values if not _is_empty(value)]
    if len(non_empty) < 2:
        return 0.0

    has_claim = bool(normalised & ROLE_ALIASES["claim_id"])
    has_accident = bool(normalised & ROLE_ALIASES["accident_year"])
    has_type = bool(normalised & ROLE_ALIASES["measure"])
    has_development = bool(normalised & ROLE_ALIASES["development"])
    has_amount = bool(normalised & ROLE_ALIASES["amount"])
    has_policy = bool(normalised & ROLE_ALIASES["policy_id"])
    has_inception = bool(normalised & ROLE_ALIASES["inception_date"])
    has_premium = bool(normalised & ROLE_ALIASES["premium"])
    has_exposure = bool(normalised & ROLE_ALIASES["exposure"]) or any(
        any(
            fragment in label
            for fragment in (
                "exposure",
                "turnover",
                "employeenumber",
                "payroll",
                "suminsured",
                "premium",
                "暴露",
                "保费",
                "营业额",
                "员工数",
                "保额",
            )
        )
        for label in normalised
    )
    development_labels = sum(_looks_like_development(value) for value in non_empty)

    if has_claim and has_accident and has_type:
        return 120.0 + min(development_labels, 20)
    if has_accident and has_development and has_amount:
        return 110.0
    if has_accident and development_labels >= 2:
        return 100.0 + min(development_labels, 20)
    if has_policy and (has_inception or has_premium):
        return 85.0
    if has_accident and has_exposure:
        return 80.0

    # Keep tidy unknown tables as low-confidence candidates so the UI can say
    # "unknown format" instead of selecting a title row and so AI fallback has
    # a safe, bounded structure to inspect.
    string_cells = sum(isinstance(value, str) and bool(value.strip()) for value in non_empty)
    if string_cells >= 2:
        return 10.0 + min(string_cells, 10)
    return 0.0


def _non_empty_segments(row: pd.Series) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for col, value in enumerate(row.tolist()):
        if not _is_empty(value):
            if start is None:
                start = col
        elif start is not None:
            segments.append((start, col))
            start = None
    if start is not None:
        segments.append((start, len(row)))
    return segments


def _find_region_end(raw_df: pd.DataFrame, header_row: int, start_col: int, end_col: int) -> int:
    end_row = len(raw_df)
    for row_idx in range(header_row + 1, len(raw_df)):
        values = raw_df.iloc[row_idx, start_col:end_col].tolist()
        if all(_is_empty(value) for value in values):
            end_row = row_idx
            break
        if row_idx > header_row + 1 and _header_score(values) >= 100:
            end_row = row_idx
            break
    return end_row


def find_candidate_table_regions(raw_df: pd.DataFrame) -> list[TableRegion]:
    """Find plausible data tables in a messy worksheet."""

    if raw_df is None or raw_df.empty:
        return []

    candidates: list[TableRegion] = []
    seen: set[tuple[int, int, int]] = set()
    for row_idx in range(len(raw_df)):
        row = raw_df.iloc[row_idx]
        for start_col, end_col in _non_empty_segments(row):
            values = row.iloc[start_col:end_col].tolist()
            score = _header_score(values)
            if score <= 0:
                continue
            if score < 100 and row_idx > 0:
                previous_values = raw_df.iloc[row_idx - 1, start_col:end_col].tolist()
                if any(not _is_empty(value) for value in previous_values):
                    continue
            end_row = _find_region_end(raw_df, row_idx, start_col, end_col)
            if end_row <= row_idx + 1:
                continue
            key = (row_idx, start_col, end_col)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(TableRegion(row_idx, start_col, end_row, end_col, score))

    return sorted(candidates, key=lambda region: (-region.score, region.header_row, region.start_col))


def _unique_headers(values: list[Any]) -> list[Any]:
    headers: list[Any] = []
    counts: dict[str, int] = {}
    for position, value in enumerate(values):
        if _is_empty(value):
            header: Any = f"column_{position + 1}"
        elif isinstance(value, str):
            header = re.sub(r"\s+", " ", value.strip())
        else:
            header = value

        key = str(header)
        count = counts.get(key, 0)
        counts[key] = count + 1
        if count:
            header = f"{key}_{count + 1}"
        headers.append(header)
    return headers


def slice_region_with_header(raw_df: pd.DataFrame, region: TableRegion) -> pd.DataFrame:
    block = raw_df.iloc[region.header_row : region.end_row, region.start_col : region.end_col].copy()
    if block.empty:
        return pd.DataFrame()

    headers = _unique_headers(block.iloc[0].tolist())
    table = block.iloc[1:].copy()
    table.columns = headers
    table = table.dropna(axis=0, how="all").dropna(axis=1, how="all")
    return table.reset_index(drop=True)


def _safe_sample_value(value: Any) -> Any:
    """Return a privacy-preserving sample cell for LLM-assisted structure detection.

    Numeric, boolean and date-like values are retained because they help decide
    whether a column is an accident period, development lag or amount. Free text
    values are masked to avoid sending claimants, notes or other sensitive cells.
    """

    if _is_empty(value):
        return None
    if isinstance(value, (bool,)):
        return bool(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            return int(number) if number.is_integer() else round(number, 6)
        return None
    # pandas/numpy scalar handling.
    try:
        number = float(value)
        if math.isfinite(number) and str(value).strip().replace(".", "", 1).replace("-", "", 1).isdigit():
            return int(number) if number.is_integer() else round(number, 6)
    except Exception:
        pass
    text = str(value)
    return f"<text len={len(text.strip())}>"


def _infer_column_type(series: pd.Series) -> str:
    non_null = series.dropna().head(100)
    if non_null.empty:
        return "empty"
    if pd.api.types.is_datetime64_any_dtype(non_null):
        return "date"
    numeric_ratio = pd.to_numeric(non_null, errors="coerce").notna().mean()
    if numeric_ratio >= 0.8:
        return "numeric"
    date_like_ratio = non_null.map(
        lambda value: hasattr(value, "year") and hasattr(value, "month") and hasattr(value, "day")
    ).mean()
    if date_like_ratio >= 0.8:
        return "date"
    return "text"


def _non_empty_counts(frame: pd.DataFrame) -> tuple[int, int]:
    if frame is None or frame.empty:
        return 0, 0
    mask = frame.map(lambda value: not _is_empty(value)) if hasattr(frame, "map") else frame.applymap(lambda value: not _is_empty(value))
    return int(mask.any(axis=1).sum()), int(mask.any(axis=0).sum())


def build_sheet_structure_summary(
    raw_df: pd.DataFrame,
    sheet_name: str,
    candidates: list[Any] | None = None,
    *,
    max_candidates: int = 8,
    max_columns: int = 60,
    max_sample_rows: int = 5,
) -> dict[str, Any]:
    """Build a bounded worksheet summary for UI display and AI fallback.

    The summary contains sheet-level shape, candidate header rows, column names,
    inferred types and masked sample rows. It intentionally does not include the
    complete workbook or raw free-text cell values.
    """

    candidates = candidates or []
    sheet_rows, sheet_cols = _non_empty_counts(raw_df)
    summaries: list[dict[str, Any]] = []
    for candidate_index, candidate in enumerate(candidates[:max_candidates]):
        region = getattr(candidate, "region", None)
        table = getattr(candidate, "table", pd.DataFrame())
        format_name = getattr(candidate, "format_name", "unknown")
        if table is None:
            table = pd.DataFrame()
        non_empty_rows, non_empty_cols = _non_empty_counts(table)
        columns = list(table.columns)[:max_columns]
        sample_rows = []
        if not table.empty:
            sample_frame = table.loc[:, columns].head(max_sample_rows)
            for _, row in sample_frame.iterrows():
                sample_rows.append({str(column): _safe_sample_value(row[column]) for column in columns})
        summaries.append(
            {
                "candidate_index": candidate_index,
                "sheet_name": sheet_name,
                "header_row": int(region.header_row) if region is not None else None,
                "header_row_excel": int(region.header_row) + 1 if region is not None else None,
                "start_col": int(region.start_col) if region is not None else None,
                "end_row": int(region.end_row) if region is not None else None,
                "end_col": int(region.end_col) if region is not None else None,
                "rule_score": float(region.score) if region is not None else 0.0,
                "rule_format": str(format_name),
                "row_count": int(len(table)),
                "non_empty_rows": non_empty_rows,
                "non_empty_cols": non_empty_cols,
                "candidate_columns": [str(column) for column in columns],
                "columns": [
                    {
                        "name": str(column),
                        "inferred_type": _infer_column_type(table[column]),
                        "non_null_count": int(table[column].notna().sum()),
                    }
                    for column in columns
                ],
                "sample_rows": sample_rows,
            }
        )
    return {
        "sheet_name": sheet_name,
        "shape": [int(raw_df.shape[0]), int(raw_df.shape[1])] if raw_df is not None else [0, 0],
        "non_empty_rows": sheet_rows,
        "non_empty_cols": sheet_cols,
        "candidate_header_rows": [item["header_row"] for item in summaries if item["header_row"] is not None],
        "candidate_count": len(candidates),
        "candidates": summaries,
        "privacy_note": "Free-text cell samples are masked; only structure, column names, numeric/date samples and counts are included.",
    }

# ---------------------------------------------------------------------------
# 兼容队友第 3 部分：手动区域扫描 / 表头行确认接口
# ---------------------------------------------------------------------------

EXCEL_ROW_COL = "__excel_row__"


def _read_raw_excel(file_path: str, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        header=None,
        engine="openpyxl",
        engine_kwargs={"keep_links": False},
    )


def get_excel_sheet_shape(file_path: str, sheet_name: str) -> tuple[int, int]:
    raw_df = _read_raw_excel(file_path, sheet_name)
    return tuple(raw_df.shape) if raw_df is not None else (0, 0)


def _validate_region_bounds(
    raw_df: pd.DataFrame,
    start_row: int,
    end_row: int,
    start_col: int,
    end_col: int,
    header_row: int,
) -> None:
    n_rows, n_cols = raw_df.shape
    if raw_df.empty:
        raise ValueError("【扫描错误】表格完全为空！")
    if start_row < 0 or start_row >= n_rows:
        raise ValueError(f"【区域错误】起始行 {start_row} 超出范围。")
    if end_row <= start_row or end_row > n_rows:
        raise ValueError(f"【区域错误】结束行 {end_row} 超出范围。")
    if start_col < 0 or start_col >= n_cols:
        raise ValueError(f"【区域错误】起始列 {start_col} 超出范围。")
    if end_col <= start_col or end_col > n_cols:
        raise ValueError(f"【区域错误】结束列 {end_col} 超出范围。")
    if header_row < start_row or header_row >= end_row:
        raise ValueError("【区域错误】表头行必须位于所选行区域内。")


def _build_table_from_raw_region(
    raw_df: pd.DataFrame,
    *,
    start_row: int,
    end_row: int,
    start_col: int,
    end_col: int,
    header_row: int,
) -> pd.DataFrame:
    _validate_region_bounds(
        raw_df=raw_df,
        start_row=start_row,
        end_row=end_row,
        start_col=start_col,
        end_col=end_col,
        header_row=header_row,
    )
    header_values = raw_df.iloc[header_row, start_col:end_col].tolist()
    headers = [str(h) for h in _unique_headers(header_values)]
    data_df = raw_df.iloc[header_row + 1 : end_row, start_col:end_col].copy()
    data_df.columns = headers
    data_df = data_df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if data_df.empty:
        raise ValueError("【扫描错误】所选区域表头行之后没有有效数据。")
    data_df[EXCEL_ROW_COL] = data_df.index + 1
    return data_df.reset_index(drop=True)


def scan_excel_table(file_path: str, sheet_name: str, header_row: int = 0) -> pd.DataFrame:
    raw_df = _read_raw_excel(file_path, sheet_name)
    if raw_df.empty:
        raise ValueError("【扫描错误】表格完全为空！")
    if header_row < 0:
        raise ValueError("【扫描错误】表头行不能小于 0。")
    if header_row >= len(raw_df):
        raise ValueError(f"【扫描错误】指定的表头行 {header_row} 超出表格范围！")
    return _build_table_from_raw_region(
        raw_df,
        start_row=header_row,
        end_row=len(raw_df),
        start_col=0,
        end_col=raw_df.shape[1],
        header_row=header_row,
    )


def scan_excel_table_region(
    file_path: str,
    sheet_name: str,
    start_row: int,
    end_row: int,
    start_col: int,
    end_col: int,
    header_row: int | None = None,
) -> pd.DataFrame:
    raw_df = _read_raw_excel(file_path, sheet_name)
    header_row = start_row if header_row is None else header_row
    return _build_table_from_raw_region(
        raw_df,
        start_row=start_row,
        end_row=end_row,
        start_col=start_col,
        end_col=end_col,
        header_row=header_row,
    )
