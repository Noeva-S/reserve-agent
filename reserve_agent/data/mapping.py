from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FieldMapping:
    """Column choices used when automatic format detection is not enough."""

    accident_year_col: str | None = None
    development_col: str | None = None
    amount_col: str | None = None
    measure_col: str | None = None
    is_cumulative: bool = True


def mapping_is_complete(mapping: FieldMapping) -> bool:
    return bool(mapping.accident_year_col and mapping.development_col and mapping.amount_col)
