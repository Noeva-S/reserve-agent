from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from docx import Document
from docx.shared import Inches

from reserve_agent.data.loader import DataQualityReport
from reserve_agent.models.reserving import ReservingOutputs


def _format_amount(value: object) -> str:
    try:
        if pd.isna(value):
            return ""
        number = float(value)
    except Exception:
        return str(value)
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:,.2f}m"
    if abs(number) >= 1_000:
        return f"{number / 1_000:,.1f}k"
    return f"{number:,.0f}"


def _add_dataframe_table(doc: Document, df: pd.DataFrame, amount_cols: set[str] | None = None, max_rows: int = 20) -> None:
    amount_cols = amount_cols or set()
    frame = df.head(max_rows).copy()
    table = doc.add_table(rows=1, cols=len(frame.columns))
    table.style = "Table Grid"
    for cell, column in zip(table.rows[0].cells, frame.columns):
        cell.text = str(column)
    for _, row in frame.iterrows():
        cells = table.add_row().cells
        for cell, column in zip(cells, frame.columns):
            value = row[column]
            cell.text = _format_amount(value) if str(column) in amount_cols else ("" if pd.isna(value) else str(value))
    if len(df) > max_rows:
        doc.add_paragraph(f"注：表格仅展示前 {max_rows} 行，完整结果请下载 Excel。")


def _add_key_value_table(doc: Document, rows: list[tuple[str, object]]) -> None:
    table = doc.add_table(rows=1, cols=2)
    table.style = "Table Grid"
    table.rows[0].cells[0].text = "项目"
    table.rows[0].cells[1].text = "数值"
    for label, value in rows:
        cells = table.add_row().cells
        cells[0].text = str(label)
        cells[1].text = str(value)


def _reserve_chart(outputs: ReservingOutputs) -> BytesIO:
    frame = outputs.comparison
    x = range(len(frame))
    years = frame["Accident Year"].astype(int).astype(str).tolist()
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=160)
    ax.bar(x, frame["Selected Reserve"], color="#2E74B5")
    ax.set_xticks(list(x), years, rotation=45, ha="right")
    ax.set_ylabel("Reserve")
    ax.set_title("Selected reserve by accident year")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer


def _ultimate_chart(outputs: ReservingOutputs) -> BytesIO:
    frame = outputs.comparison
    years = frame["Accident Year"].astype(int).tolist()
    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=160)
    ax.plot(years, frame["Latest Cumulative"], marker="o", label="Latest cumulative")
    ax.plot(years, frame["Selected Ultimate"], marker="o", label="Selected ultimate")
    ax.set_xlabel("Accident year")
    ax.set_ylabel("Amount")
    ax.set_title("Latest cumulative vs selected ultimate")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer


def build_word_report(
    explanation_text: str | None = None,
    *,
    source_file: str = "",
    sheet_name: str = "",
    format_name: str = "",
    measure: str = "Paid",
    triangle: pd.DataFrame | None = None,
    outputs: ReservingOutputs | None = None,
    quality: DataQualityReport | None = None,
    method_notes: Mapping[str, str] | None = None,
) -> bytes:
    """Build a complete editable Word report for the current result."""

    document = Document()
    document.add_heading("准备金评估完整报告", level=0)
    document.add_paragraph(f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}")
    if source_file:
        document.add_paragraph(f"数据文件：{source_file}")
    if sheet_name:
        document.add_paragraph(f"工作表：{sheet_name}")
    if format_name:
        document.add_paragraph(f"识别格式：{format_name}")
    document.add_paragraph(f"建模口径：{measure}")

    if quality and outputs is not None:
        document.add_heading("1. 评估概览", level=1)
        diag = outputs.diagnostics
        document.add_paragraph(
            f"系统读取 {quality.row_count:,} 行数据，事故年范围为 "
            f"{min(quality.accident_years)}-{max(quality.accident_years)}。"
            f"累计已观察赔款约 {_format_amount(diag['total_latest'])}，"
            f"展示准备金约 {_format_amount(diag['total_selected_reserve'])}，"
            f"展示最终赔款约 {_format_amount(diag['total_selected_ultimate'])}。"
        )
        _add_key_value_table(
            document,
            [
                ("事故年数量", f"{len(quality.accident_years):,}"),
                ("累计已观察赔款", _format_amount(diag["total_latest"])),
                ("展示准备金", _format_amount(diag["total_selected_reserve"])),
                ("展示最终赔款", _format_amount(diag["total_selected_ultimate"])),
            ],
        )

        document.add_heading("2. 数据质量诊断", level=1)
        _add_key_value_table(
            document,
            [
                ("源数据行数", f"{quality.row_count:,}"),
                ("赔案/事故年数量", f"{quality.claim_count:,}"),
                ("事故年范围", f"{min(quality.accident_years)}-{max(quality.accident_years)}"),
                ("评估年范围", f"{min(quality.valuation_years)}-{max(quality.valuation_years)}"),
                ("缺失值数量", f"{quality.missing_values:,}"),
                ("负金额单元格", f"{quality.negative_amount_cells:,}"),
                ("零赔案/零事故年记录", f"{quality.zero_claim_rows:,}"),
            ],
        )
        for note in quality.notes:
            document.add_paragraph(note, style=None)

    if triangle is not None:
        document.add_heading("3. 累计赔款三角形", level=1)
        triangle_frame = triangle.copy().reset_index()
        triangle_frame.columns = ["Accident Year"] + [f"Dev {int(col)}" for col in triangle.columns]
        _add_dataframe_table(document, triangle_frame, set(triangle_frame.columns[1:]))

    if outputs is not None:
        document.add_heading("4. 模型结果", level=1)
        factor_frame = outputs.selected_factors.reset_index()
        factor_frame.columns = ["Development Age", "Selected Factor"]
        _add_dataframe_table(document, factor_frame, max_rows=30)

        comparison = outputs.comparison.copy()
        _add_dataframe_table(
            document,
            comparison,
            {
                "Latest Cumulative",
                "Chain Ladder Ultimate",
                "Chain Ladder Reserve",
                "Expected Ultimate Loss",
                "ELR Reserve",
                "BF Ultimate Loss",
                "BF Reserve",
                "Mack Ultimate",
                "Mack Reserve",
                "Mack Standard Error",
                "Mack 95% Lower",
                "Mack 95% Upper",
                "Selected Ultimate",
                "Selected Reserve",
            },
        )

        if outputs.mack is not None and not outputs.mack.empty:
            document.add_heading("4.1 Mack Chain Ladder 不确定性分析", level=2)
            mack_diag = outputs.mack_diagnostics or {}
            _add_key_value_table(
                document,
                [
                    ("Mack 准备金", _format_amount(mack_diag.get("total_mack_reserve", 0.0))),
                    ("Mack 标准误", _format_amount(mack_diag.get("total_mack_standard_error", 0.0))),
                    ("Mack CV", f"{float(mack_diag.get('total_mack_cv', 0.0)):.1%}"),
                    (
                        "95% 准备金区间",
                        f"{_format_amount(mack_diag.get('mack_95_lower', 0.0))}-"
                        f"{_format_amount(mack_diag.get('mack_95_upper', 0.0))}",
                    ),
                ],
            )
            _add_dataframe_table(
                document,
                outputs.mack,
                {
                    "Latest Cumulative",
                    "Mack Ultimate",
                    "Mack Reserve",
                    "Mack Standard Error",
                    "Mack 75% Lower",
                    "Mack 75% Upper",
                    "Mack 95% Lower",
                    "Mack 95% Upper",
                },
            )

        if outputs.expected_lr_sensitivity is not None and not outputs.expected_lr_sensitivity.empty:
            document.add_heading("4.2 期望赔付率敏感性分析", level=2)
            _add_dataframe_table(
                document,
                outputs.expected_lr_sensitivity,
                {"ELR Reserve", "ELR Ultimate", "BF Reserve", "BF Ultimate"},
                max_rows=30,
            )

        if outputs.factor_sensitivity is not None and not outputs.factor_sensitivity.empty:
            document.add_heading("4.3 发展因子敏感性分析", level=2)
            _add_dataframe_table(
                document,
                outputs.factor_sensitivity,
                {"Chain Ladder Reserve", "Chain Ladder Ultimate"},
                max_rows=30,
            )

        document.add_heading("5. 可视化", level=1)
        document.add_picture(_reserve_chart(outputs), width=Inches(6.2))
        document.add_paragraph("图 1  各事故年展示准备金")
        document.add_picture(_ultimate_chart(outputs), width=Inches(6.2))
        document.add_paragraph("图 2  已观察赔款与展示最终赔款")

    if explanation_text:
        document.add_heading("6. Agent 分析", level=1)
        for raw_line in explanation_text.splitlines():
            line = raw_line.strip()
            if line:
                document.add_paragraph(line[2:] if line.startswith("- ") else line)

    if method_notes:
        document.add_heading("7. 方法说明", level=1)
        for name, note in method_notes.items():
            document.add_paragraph(f"{name}: {note}")

    document.add_heading("8. 注意事项", level=1)
    document.add_paragraph(
        "本报告由课程项目中的自动化系统生成，适用于课堂展示和方法说明，不构成正式精算签字意见。"
        "实际使用前应复核数据口径、大额赔案、尾部发展和模型假设。"
    )

    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()
