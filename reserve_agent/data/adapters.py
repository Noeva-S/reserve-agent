from __future__ import annotations

import pandas as pd


def triangle_sheet_to_triangle(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a prepared triangle sheet into the model triangle format.

    Teammate A can extend this function to handle more common classroom Excel
    layouts. The expected output is a cumulative triangle with accident years
    as the index and development ages as numeric columns.
    """

    if df.empty:
        raise ValueError("Triangle sheet is empty.")

    triangle = df.copy()
    triangle = triangle.dropna(axis=0, how="all").dropna(axis=1, how="all")
    accident_year_col = triangle.columns[0]
    triangle = triangle.set_index(accident_year_col)
    triangle.index = pd.to_numeric(triangle.index, errors="coerce")
    triangle = triangle[triangle.index.notna()]
    triangle.index = triangle.index.astype(int)
    triangle.columns = [int(col) if str(col).strip().isdigit() else col for col in triangle.columns]
    return triangle.apply(pd.to_numeric, errors="coerce")


def long_table_to_triangle(
    df: pd.DataFrame,
    accident_year_col: str = "Accident Year",
    development_col: str = "Development",
    amount_col: str = "Amount",
    cumulative: bool = True,
) -> pd.DataFrame:
    """Convert long table data into a cumulative triangle."""

    required = {accident_year_col, development_col, amount_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    work = df[[accident_year_col, development_col, amount_col]].copy()
    work.columns = ["accident_year", "development", "amount"]
    work["accident_year"] = pd.to_numeric(work["accident_year"], errors="coerce")
    work["development"] = pd.to_numeric(work["development"], errors="coerce")
    work["amount"] = pd.to_numeric(work["amount"], errors="coerce")
    work = work.dropna(subset=["accident_year", "development"])

    grouped = work.groupby(["accident_year", "development"], as_index=False)["amount"].sum(min_count=1)
    triangle = grouped.pivot(index="accident_year", columns="development", values="amount")
    triangle = triangle.sort_index().sort_index(axis=1)
    triangle.index = triangle.index.astype(int)
    triangle.columns = triangle.columns.astype(int)
    if not cumulative:
        triangle = triangle.cumsum(axis=1)
    return triangle.astype(float)
