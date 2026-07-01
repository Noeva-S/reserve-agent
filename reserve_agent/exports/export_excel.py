from __future__ import annotations

from dataclasses import asdict, is_dataclass
from io import BytesIO
from typing import Any, Mapping

import pandas as pd

from reserve_agent.data.loader import DataQualityReport
from reserve_agent.models.reserving import ReservingOutputs


def _as_frame(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if isinstance(value, pd.Series):
        return value.reset_index()
    if is_dataclass(value):
        return pd.DataFrame([asdict(value)])
    if isinstance(value, Mapping):
        return pd.DataFrame([dict(value)])
    if isinstance(value, list):
        rows = []
        for item in value:
            if is_dataclass(item):
                rows.append(asdict(item))
            elif isinstance(item, Mapping):
                rows.append(dict(item))
            else:
                rows.append({"value": item})
        return pd.DataFrame(rows)
    return pd.DataFrame([{"value": value}])


def _quality_to_frame(report: DataQualityReport) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Item": "Row Count", "Value": report.row_count},
            {"Item": "Claim / Accident Year Count", "Value": report.claim_count},
            {"Item": "Accident Years", "Value": ", ".join(map(str, report.accident_years))},
            {"Item": "Valuation Years", "Value": ", ".join(map(str, report.valuation_years))},
            {"Item": "Missing Values", "Value": report.missing_values},
            {"Item": "Negative Amount Cells", "Value": report.negative_amount_cells},
            {"Item": "Zero Claim / Accident Year Rows", "Value": report.zero_claim_rows},
            {"Item": "Notes", "Value": " | ".join(report.notes)},
        ]
    )


def _detection_to_frame(detection: Any | None) -> pd.DataFrame:
    if detection is None:
        return pd.DataFrame()
    if is_dataclass(detection):
        detection = asdict(detection)
    if isinstance(detection, Mapping):
        summary = dict(detection)
        candidates = summary.get("candidates")
        if isinstance(candidates, list):
            rows = []
            for item in candidates:
                if isinstance(item, Mapping):
                    rows.append(
                        {
                            "Candidate Index": item.get("candidate_index"),
                            "Sheet": item.get("sheet_name", summary.get("sheet_name")),
                            "Header Row (0-based)": item.get("header_row"),
                            "Header Row (Excel)": item.get("header_row_excel"),
                            "Rule Format": item.get("rule_format"),
                            "Rule Score": item.get("rule_score"),
                            "Rows": item.get("row_count"),
                            "Non-empty Rows": item.get("non_empty_rows"),
                            "Non-empty Cols": item.get("non_empty_cols"),
                            "Candidate Columns": ", ".join(map(str, item.get("candidate_columns", []))),
                        }
                    )
            return pd.DataFrame(rows)
        return pd.DataFrame([summary])
    return _as_frame(detection)


def _write_frame(writer: pd.ExcelWriter, frame: pd.DataFrame, sheet_name: str, *, index: bool = False) -> None:
    if frame is None or frame.empty:
        return
    safe_name = sheet_name[:31]
    frame.to_excel(writer, sheet_name=safe_name, index=index)
    worksheet = writer.sheets[safe_name]
    for column_cells in worksheet.columns:
        header = column_cells[0].value
        length = len(str(header)) if header is not None else 10
        for cell in column_cells[1:100]:
            if cell.value is not None:
                length = max(length, len(str(cell.value)))
        worksheet.column_dimensions[column_cells[0].column_letter].width = min(max(length + 2, 10), 48)
    worksheet.freeze_panes = "A2"


def _optional_output_attr(outputs: ReservingOutputs, *names: str) -> Any | None:
    for name in names:
        if hasattr(outputs, name):
            value = getattr(outputs, name)
            if value is not None:
                return value
    return None


def build_excel_download(
    outputs: ReservingOutputs,
    triangle: pd.DataFrame,
    report: DataQualityReport,
    *,
    detection_summary: Any | None = None,
    validation_issues: Any | None = None,
    mack_results: Any | None = None,
    sensitivity_analysis: Any | None = None,
    processing_log: str | None = None,
) -> bytes:
    """Build an Excel workbook containing all key model and audit outputs."""

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        _write_frame(writer, triangle.reset_index(), "Triangle", index=False)
        factors = outputs.selected_factors.reset_index()
        factors.columns = ["Development Age", "Selected Factor"]
        _write_frame(writer, factors, "Development Factors")

        age_to_ultimate = outputs.age_to_ultimate.reset_index()
        age_to_ultimate.columns = ["Development Age", "Age-to-Ultimate Factor"]
        _write_frame(writer, age_to_ultimate, "Age to Ultimate")

        _write_frame(writer, outputs.comparison, "Model Comparison")
        _write_frame(writer, outputs.chain_ladder, "Chain Ladder")
        _write_frame(writer, outputs.expected_loss_ratio, "ELR")
        _write_frame(writer, outputs.bornhuetter_ferguson, "BF")
        _write_frame(writer, _quality_to_frame(report), "Data Quality")
        _write_frame(writer, _detection_to_frame(detection_summary), "Detection Summary")
        _write_frame(writer, _as_frame(validation_issues), "Validation Issues")

        mack = mack_results or _optional_output_attr(outputs, "mack", "mack_results", "mack_chain_ladder")
        if mack is not None:
            _write_frame(writer, _as_frame(mack), "Mack Results")

        if getattr(outputs, "expected_lr_sensitivity", None) is not None:
            _write_frame(writer, _as_frame(outputs.expected_lr_sensitivity), "ELR Sensitivity")

        if getattr(outputs, "factor_sensitivity", None) is not None:
            _write_frame(writer, _as_frame(outputs.factor_sensitivity), "Factor Sensitivity")

        sensitivity = sensitivity_analysis or _optional_output_attr(outputs, "sensitivity", "sensitivity_analysis")
        if sensitivity is not None:
            if isinstance(sensitivity, Mapping):
                for name, value in sensitivity.items():
                    _write_frame(writer, _as_frame(value), f"Sensitivity {str(name)[:18]}")
            else:
                _write_frame(writer, _as_frame(sensitivity), "Sensitivity Analysis")

        if processing_log:
            log_frame = pd.DataFrame({"Processing Log": processing_log.splitlines() or [processing_log]})
            _write_frame(writer, log_frame, "Processing Log")

        # A compact totals sheet is convenient for quick submission review.
        totals = pd.DataFrame(
            [
                {"Metric": key, "Value": value}
                for key, value in outputs.diagnostics.items()
            ]
        )
        _write_frame(writer, totals, "Summary")
    return buffer.getvalue()
