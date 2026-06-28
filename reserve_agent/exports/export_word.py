from __future__ import annotations

from io import BytesIO

from docx import Document


def build_word_report(explanation_text: str) -> bytes:
    """Build a simple Word report from the current Agent explanation."""

    document = Document()
    document.add_heading("准备金评估智能 Agent 解释报告", level=1)
    for paragraph in explanation_text.splitlines():
        if paragraph.strip():
            document.add_paragraph(paragraph)
    buffer = BytesIO()
    document.save(buffer)
    return buffer.getvalue()
