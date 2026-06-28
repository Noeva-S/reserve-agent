from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ValidationIssue:
    level: str
    message: str


def validate_triangle(triangle: pd.DataFrame) -> list[ValidationIssue]:
    """Run basic checks on a cumulative loss triangle.

    Teammate B can add richer actuarial checks here without changing app.py.
    """

    issues: list[ValidationIssue] = []
    if triangle.empty:
        return [ValidationIssue("error", "Triangle is empty.")]

    numeric = triangle.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().all(axis=None):
        issues.append(ValidationIssue("error", "Triangle has no numeric values."))
    if (numeric < 0).any(axis=None):
        issues.append(ValidationIssue("warning", "Triangle contains negative values."))
    if numeric.index.duplicated().any():
        issues.append(ValidationIssue("error", "Triangle contains duplicated accident years."))

    if not issues:
        issues.append(ValidationIssue("info", "No blocking triangle validation issue found."))
    return issues
