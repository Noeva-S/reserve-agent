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
    """Build an Excel workbook containing all key model outputs."""

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        triangle.to_excel(writer, sheet_name="Triangle")
        outputs.selected_factors.reset_index().rename(
            columns={"index": "Development Age", "selected_factor": "Selected Factor"}
        ).to_excel(writer, sheet_name="Development Factors", index=False)
        outputs.chain_ladder.to_excel(writer, sheet_name="Chain Ladder", index=False)
        outputs.expected_loss_ratio.to_excel(writer, sheet_name="ELR", index=False)
        outputs.bornhuetter_ferguson.to_excel(writer, sheet_name="BF", index=False)
        outputs.comparison.to_excel(writer, sheet_name="Model Comparison", index=False)
        pd.DataFrame([report.__dict__]).to_excel(writer, sheet_name="Data Quality", index=False)
    return buffer.getvalue()
