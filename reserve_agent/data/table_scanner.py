from __future__ import annotations

from dataclasses import dataclass
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
    development_labels = sum(_looks_like_development(value) for value in non_empty)

    if has_claim and has_accident and has_type:
        return 120.0 + min(development_labels, 20)
    if has_accident and has_development and has_amount:
        return 110.0
    if has_accident and development_labels >= 2:
        return 100.0 + min(development_labels, 20)

    # Keep tidy unknown tables as low-confidence candidates so the UI can say
    # "unknown format" instead of selecting a title row.
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
