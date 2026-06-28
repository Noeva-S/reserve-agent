from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass
class TableRegion:
    """A likely table area inside an Excel sheet.

    Row and column indexes are zero-based positions in a raw sheet loaded with
    header=None. This module exists because real Excel files often have title
    rows, notes, blank rows, or several small tables before the actual data.
    """

    header_row: int
    start_col: int
    end_col: int
    start_row: int
    end_row: int
    confidence: float
    reason: str


DEFAULT_HEADER_KEYWORDS = {
    "claim",
    "loss",
    "accident",
    "year",
    "development",
    "dev",
    "amount",
    "paid",
    "incurred",
    "type",
    "id",
    "policy",
    "exposure",
}


def _clean_cell(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def _non_empty_positions(row: pd.Series) -> list[int]:
    return [idx for idx, value in enumerate(row.tolist()) if _clean_cell(value)]


def score_header_row(
    row: pd.Series,
    keywords: Iterable[str] = DEFAULT_HEADER_KEYWORDS,
) -> float:
    """Score whether a raw sheet row looks like a table header."""

    values = [_clean_cell(value) for value in row.tolist()]
    non_empty = [value for value in values if value]
    if len(non_empty) < 2:
        return 0.0

    keyword_hits = 0
    for value in non_empty:
        keyword_hits += int(any(keyword in value for keyword in keywords))

    numeric_hits = 0
    for value in non_empty:
        numeric_hits += int(pd.to_numeric(value, errors="coerce") == pd.to_numeric(value, errors="coerce"))

    density_score = min(len(non_empty) / max(len(values), 1), 1.0)
    keyword_score = keyword_hits / len(non_empty)
    numeric_header_score = min(numeric_hits / max(len(non_empty), 1), 0.4)
    return round(0.55 * keyword_score + 0.30 * density_score + 0.15 * numeric_header_score, 4)


def find_candidate_table_regions(raw_df: pd.DataFrame, min_non_empty: int = 2) -> list[TableRegion]:
    """Find likely table regions in a raw Excel sheet.

    Teammate A should extend this function when handling messy workbooks:
    title rows, note rows, merged-cell leftovers, blank separators, and multiple
    tables in one sheet.
    """

    candidates: list[TableRegion] = []
    if raw_df.empty:
        return candidates

    row_scores = []
    for row_index, row in raw_df.iterrows():
        positions = _non_empty_positions(row)
        if len(positions) < min_non_empty:
            continue
        score = score_header_row(row)
        if score <= 0:
            continue
        row_scores.append((row_index, score, min(positions), max(positions)))

    for row_index, score, start_col, end_col in row_scores:
        end_row = row_index
        blank_streak = 0
        for next_row in range(row_index + 1, len(raw_df)):
            row_slice = raw_df.iloc[next_row, start_col : end_col + 1]
            if row_slice.dropna().empty:
                blank_streak += 1
                if blank_streak >= 2:
                    break
            else:
                blank_streak = 0
                end_row = next_row

        candidates.append(
            TableRegion(
                header_row=int(row_index),
                start_col=int(start_col),
                end_col=int(end_col),
                start_row=int(row_index + 1),
                end_row=int(end_row),
                confidence=float(score),
                reason="row has enough populated cells and header-like keywords",
            )
        )

    return sorted(candidates, key=lambda item: item.confidence, reverse=True)


def slice_region_with_header(raw_df: pd.DataFrame, region: TableRegion) -> pd.DataFrame:
    """Convert a detected raw region into a DataFrame with header columns."""

    header = raw_df.iloc[region.header_row, region.start_col : region.end_col + 1].tolist()
    data = raw_df.iloc[region.start_row : region.end_row + 1, region.start_col : region.end_col + 1].copy()
    data.columns = [str(col).strip() if not pd.isna(col) else f"Unnamed {idx}" for idx, col in enumerate(header)]
    return data.dropna(axis=0, how="all").dropna(axis=1, how="all")
