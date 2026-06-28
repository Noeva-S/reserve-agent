"""Data loading, format detection, mapping, and validation modules."""

from .loader import (
    DataQualityReport,
    build_cumulative_triangle,
    find_default_workbook,
    latest_diagonal,
    list_excel_sheets,
    load_claims_snapshot,
    load_exposure_data,
    quality_report,
    summarize_claims_by_year,
    triangle_to_display,
    valuation_year_columns,
)
from .table_scanner import TableRegion, find_candidate_table_regions, slice_region_with_header

__all__ = [
    "DataQualityReport",
    "TableRegion",
    "build_cumulative_triangle",
    "find_default_workbook",
    "find_candidate_table_regions",
    "latest_diagonal",
    "list_excel_sheets",
    "load_claims_snapshot",
    "load_exposure_data",
    "quality_report",
    "summarize_claims_by_year",
    "triangle_to_display",
    "slice_region_with_header",
    "valuation_year_columns",
]
