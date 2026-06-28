from __future__ import annotations

from io import BytesIO

import pandas as pd

from reserve_agent.data.loader import DataQualityReport
from reserve_agent.models.reserving import ReservingOutputs


def build_excel_download(
    outputs: ReservingOutputs,
    triangle: pd.DataFrame,
    report: DataQualityReport,
) -> bytes:
    """Build an Excel workbook containing triangle, result, and diagnostics."""

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        triangle.to_excel(writer, sheet_name="Triangle")
        outputs.comparison.to_excel(writer, sheet_name="Model Results", index=False)
        pd.DataFrame([report.__dict__]).to_excel(writer, sheet_name="Data Quality", index=False)
    return buffer.getvalue()
