from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd

from reserve_agent.data.detector import (
    ExcelFormat,
    find_role_column,
    normalise_label,
    parse_development_label,
)
from reserve_agent.data.loader import build_cumulative_triangle


def _coerce_integer_axis(series: pd.Series, axis_name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().all():
        raise ValueError(f"{axis_name}列没有可识别的数字。")
    return numeric


def _find_preferred_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> Any | None:
    columns_by_label = {normalise_label(column): column for column in df.columns}
    for alias in aliases:
        if alias in columns_by_label:
            return columns_by_label[alias]
    return None


def _is_claim_development_table(df: pd.DataFrame) -> bool:
    claim_col = find_role_column(df, "claim_id")
    labels = {normalise_label(column) for column in df.columns}
    return claim_col is not None and bool(labels & {"delayyears", "delayyear", "delaythirds", "delayshifted"})


def _find_amount_column(df: pd.DataFrame, measure: str) -> Any | None:
    if measure.strip().lower() == "incurred":
        priorities = (
            "overallamountrevalued",
            "totalincurred",
            "incurred",
            "overallamountdestinationcurrency",
            "overallamount",
            "amount",
            "value",
            "loss",
            "金额",
            "赔款金额",
        )
    else:
        priorities = ("paid", "paidamount", "amount", "value", "loss", "金额", "赔款金额")
    return _find_preferred_column(df, priorities) or find_role_column(df, "amount")


def _incremental_to_cumulative(triangle: pd.DataFrame) -> pd.DataFrame:
    if triangle.empty:
        return triangle

    latest_calendar_period = max(
        int(accident_year) + int(development)
        for accident_year, row in triangle.iterrows()
        for development, value in row.items()
        if pd.notna(value)
    )
    max_development = max(int(max(triangle.columns)), latest_calendar_period - int(min(triangle.index)))
    all_developments = list(range(0, max_development + 1))
    incremental = triangle.reindex(columns=all_developments)

    cumulative = pd.DataFrame(index=incremental.index, columns=all_developments, dtype=float)
    for accident_year, row in incremental.iterrows():
        latest_observable_development = min(max_development, latest_calendar_period - int(accident_year))
        observed_columns = [
            development for development in all_developments if development <= latest_observable_development
        ]
        if observed_columns:
            cumulative.loc[accident_year, observed_columns] = row[observed_columns].fillna(0.0).cumsum().to_numpy()
    cumulative.index.name = "Accident Year"
    return cumulative


def claims_snapshot_to_triangle(df: pd.DataFrame, measure: str = "Paid") -> pd.DataFrame:
    claim_col = find_role_column(df, "claim_id")
    accident_col = find_role_column(df, "accident_year")
    measure_col = find_role_column(df, "measure")
    if claim_col is None or accident_col is None or measure_col is None:
        raise ValueError("赔案明细格式需要 Claim ID、Loss Year 和 Type 字段。")

    canonical = df.rename(
        columns={
            claim_col: "Claim ID",
            accident_col: "Loss Year",
            measure_col: "Type",
        }
    ).copy()
    return build_cumulative_triangle(canonical, measure=measure)


def triangle_sheet_to_triangle(df: pd.DataFrame) -> pd.DataFrame:
    accident_col = find_role_column(df, "accident_year")
    if accident_col is not None:
        accident_year = _coerce_integer_axis(df[accident_col], "事故年")
        value_df = df.drop(columns=[accident_col])
    elif df.index.name is not None:
        accident_year = _coerce_integer_axis(pd.Series(df.index, index=df.index), "事故年")
        value_df = df.copy()
    else:
        raise ValueError("三角形格式缺少事故年列。")

    columns_by_development: dict[int, list[Any]] = defaultdict(list)
    for column in value_df.columns:
        development = parse_development_label(column)
        if development is not None:
            columns_by_development[development].append(column)
    if not columns_by_development:
        raise ValueError("三角形格式没有可识别的发展期列。")

    valid_rows = accident_year.notna()
    data: dict[int, pd.Series] = {}
    for development, source_columns in columns_by_development.items():
        numeric = value_df.loc[valid_rows, source_columns].apply(pd.to_numeric, errors="coerce")
        data[development] = numeric.sum(axis=1, min_count=1)

    triangle = pd.DataFrame(data)
    triangle.index = accident_year.loc[valid_rows].astype(int).to_numpy()
    triangle.index.name = "Accident Year"
    triangle = triangle.groupby(level=0).sum(min_count=1)
    triangle = triangle.sort_index().sort_index(axis=1).dropna(axis=1, how="all")
    if triangle.empty:
        raise ValueError("三角形数据区域为空。")
    return triangle.astype(float)


def long_table_to_triangle(
    df: pd.DataFrame,
    *,
    is_cumulative: bool = True,
    measure: str = "Paid",
) -> pd.DataFrame:
    claim_development_table = _is_claim_development_table(df)
    if claim_development_table:
        accident_col = _find_preferred_column(
            df,
            ("policyyearshifted", "lossyear", "accidentyear", "originyear", "policyyear"),
        )
        development_col = _find_preferred_column(
            df,
            ("delayshifted", "delayyears", "delayyear", "delaythirds"),
        )
    else:
        accident_col = find_role_column(df, "accident_year")
        development_col = find_role_column(df, "development")
    amount_col = _find_amount_column(df, measure)
    if accident_col is None or development_col is None or amount_col is None:
        raise ValueError("长表格式需要事故年、发展期和金额字段。")

    long_df = df[[accident_col, development_col, amount_col]].copy()
    long_df.columns = ["accident_year", "development", "amount"]
    long_df["accident_year"] = pd.to_numeric(long_df["accident_year"], errors="coerce")
    long_df["development"] = pd.to_numeric(long_df["development"], errors="coerce")
    long_df["amount"] = pd.to_numeric(long_df["amount"], errors="coerce")
    long_df = long_df.dropna(subset=["accident_year", "development", "amount"])
    long_df = long_df[long_df["development"] >= 0]
    if long_df.empty:
        raise ValueError("长表中没有有效的事故年、发展期和金额记录。")

    long_df["accident_year"] = long_df["accident_year"].astype(int)
    long_df["development"] = long_df["development"].astype(int)
    triangle = long_df.pivot_table(
        index="accident_year",
        columns="development",
        values="amount",
        aggfunc="sum",
    ).sort_index().sort_index(axis=1)

    if claim_development_table or not is_cumulative:
        triangle = _incremental_to_cumulative(triangle)
    triangle.index.name = "Accident Year"
    return triangle.astype(float)


def adapt_to_triangle(
    df: pd.DataFrame,
    format_name: ExcelFormat,
    *,
    measure: str = "Paid",
    is_cumulative: bool = True,
) -> pd.DataFrame:
    if format_name == "claims_snapshot":
        return claims_snapshot_to_triangle(df, measure=measure)
    if format_name == "triangle":
        return triangle_sheet_to_triangle(df)
    if format_name == "long_table":
        return long_table_to_triangle(df, is_cumulative=is_cumulative, measure=measure)
    raise ValueError("无法识别 Excel 格式，不能转换为准备金三角形。")
