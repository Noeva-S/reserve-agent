from __future__ import annotations

from typing import Literal

import pandas as pd


ExcelFormat = Literal["claims_snapshot", "triangle", "long_table", "unknown"]


def detect_excel_format(df: pd.DataFrame) -> ExcelFormat:
    """Guess the uploaded sheet format from column names and table shape.

    This is the entry point for teammate A. Keep the return values stable so
    the loader and UI can call this function without knowing the details.
    If the workbook may contain title rows, notes, or blank rows before the
    table, first load the sheet with header=None and use data.table_scanner to
    find a candidate table region, then call this function on the sliced table.
    """

    columns = {str(col).strip().lower() for col in df.columns}
    if {"claim id", "loss year", "type"}.issubset(columns):
        return "claims_snapshot"

    long_table_keywords = [
        {"accident year", "development", "amount"},
        {"accident_year", "development", "amount"},
        {"loss year", "development", "amount"},
        {"事故年", "发展期", "金额"},
    ]
    if any(keys.issubset(columns) for keys in long_table_keywords):
        return "long_table"

    first_col = str(df.columns[0]).strip().lower() if len(df.columns) else ""
    numeric_column_count = sum(pd.to_numeric(pd.Index(df.columns[1:]), errors="coerce").notna())
    if first_col in {"accident year", "accident_year", "loss year", "事故年"} and numeric_column_count >= 2:
        return "triangle"

    return "unknown"
