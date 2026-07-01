from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st

from reserve_agent.exports import (
    build_excel_download,
    build_report_chart_images,
    build_word_report,
    build_zip_package,
)


def _safe_stem(text: str) -> str:
    stem = Path(text or "reserve_agent").stem or "reserve_agent"
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem)[:40]


def _validation_rows(issues: Iterable[Any] | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for issue in issues or []:
        rows.append(
            {
                "level": str(getattr(issue, "level", "")),
                "message": str(getattr(issue, "message", issue)),
            }
        )
    return rows


def _processing_log(
    *,
    source_file: str,
    requested_sheet: str,
    source_sheet: str,
    format_name: str,
    measure: str,
    expected_lr: float,
    is_cumulative: bool,
    recognition_source: str,
    warnings: Iterable[str],
    quality_notes: Iterable[str],
    validation_rows: list[dict[str, str]],
) -> str:
    lines = [
        "ReserveAgent processing log",
        f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Source file: {source_file}",
        f"Requested sheet: {requested_sheet}",
        f"Actual sheet: {source_sheet or requested_sheet}",
        f"Detected format: {format_name}",
        f"Measure: {measure}",
        f"Expected LR: {expected_lr:.4f}",
        f"Long-table cumulative flag from UI: {is_cumulative}",
        f"Recognition source: {recognition_source or 'rules'}",
        "",
        "Warnings:",
    ]
    warning_list = list(warnings or [])
    lines.extend([f"- {item}" for item in warning_list] or ["- None"])
    lines.append("")
    lines.append("Data quality notes:")
    lines.extend([f"- {item}" for item in quality_notes] or ["- None"])
    lines.append("")
    lines.append("Validation issues:")
    lines.extend([f"- {row['level']}: {row['message']}" for row in validation_rows] or ["- None"])
    return "\n".join(lines)


def _reset_when_signature_changes(signature: str) -> None:
    if st.session_state.get("download_signature") != signature:
        for key in ["download_excel_bytes", "download_word_bytes", "download_zip_bytes"]:
            st.session_state.pop(key, None)
        st.session_state.download_signature = signature


def render_download_panel(
    *,
    load_result: Any,
    triangle: pd.DataFrame,
    outputs: Any,
    quality: Any,
    validation_issues: Iterable[Any] | None,
    explanation_text: str,
    method_notes: dict[str, str],
    source_file: str,
    requested_sheet_name: str,
    measure: str,
    expected_lr: float,
    is_cumulative: bool,
    signature: str,
) -> None:
    """Render lazy Excel/Word/ZIP generation controls.

    Files are generated only after the user clicks a generation button and then
    stored in ``st.session_state`` until the current data/model signature changes.
    """

    _reset_when_signature_changes(signature)
    stem = _safe_stem(source_file)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    validation_frame_rows = _validation_rows(validation_issues)
    processing_log = _processing_log(
        source_file=source_file,
        requested_sheet=requested_sheet_name,
        source_sheet=getattr(load_result, "source_sheet_name", ""),
        format_name=getattr(load_result, "format_name", ""),
        measure=measure,
        expected_lr=expected_lr,
        is_cumulative=is_cumulative,
        recognition_source=getattr(load_result, "recognition_source", "rules"),
        warnings=getattr(load_result, "warnings", ()),
        quality_notes=getattr(quality, "notes", []),
        validation_rows=validation_frame_rows,
    )

    st.caption("为避免页面刷新卡顿，Excel、Word 和 ZIP 只会在点击生成按钮后创建；数据或参数变化后会自动清空旧下载。")
    col_a, col_b, col_c = st.columns(3)

    with col_a:
        if st.button("生成 Excel 结果", use_container_width=True):
            with st.spinner("正在生成 Excel..."):
                st.session_state.download_excel_bytes = build_excel_download(
                    outputs,
                    triangle,
                    quality,
                    detection_summary=getattr(load_result, "structure_summary", None),
                    validation_issues=validation_frame_rows,
                    processing_log=processing_log,
                )
        excel_bytes = st.session_state.get("download_excel_bytes")
        if excel_bytes:
            st.download_button(
                "下载 Excel 结果",
                data=excel_bytes,
                file_name=f"{stem}_reserve_results_{timestamp}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

    with col_b:
        if st.button("生成 Word 报告", use_container_width=True):
            with st.spinner("正在生成 Word 报告..."):
                st.session_state.download_word_bytes = build_word_report(
                    explanation_text=explanation_text,
                    source_file=source_file,
                    sheet_name=getattr(load_result, "source_sheet_name", "") or requested_sheet_name,
                    format_name=getattr(load_result, "format_name", ""),
                    measure=measure,
                    triangle=triangle,
                    outputs=outputs,
                    quality=quality,
                    method_notes=method_notes,
                    detection_summary=getattr(load_result, "structure_summary", None),
                    validation_issues=validation_frame_rows,
                    recognition_source=getattr(load_result, "recognition_source", "rules"),
                    recognition_reason=getattr(load_result, "recognition_reason", ""),
                    header_row=getattr(getattr(load_result, "region", None), "header_row", None),
                )
        word_bytes = st.session_state.get("download_word_bytes")
        if word_bytes:
            st.download_button(
                "下载 Word 报告",
                data=word_bytes,
                file_name=f"{stem}_agent_report_{timestamp}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
            )

    with col_c:
        if st.button("生成 ZIP 打包", use_container_width=True):
            with st.spinner("正在打包 ZIP..."):
                excel_bytes = st.session_state.get("download_excel_bytes") or build_excel_download(
                    outputs,
                    triangle,
                    quality,
                    detection_summary=getattr(load_result, "structure_summary", None),
                    validation_issues=validation_frame_rows,
                    processing_log=processing_log,
                )
                word_bytes = st.session_state.get("download_word_bytes") or build_word_report(
                    explanation_text=explanation_text,
                    source_file=source_file,
                    sheet_name=getattr(load_result, "source_sheet_name", "") or requested_sheet_name,
                    format_name=getattr(load_result, "format_name", ""),
                    measure=measure,
                    triangle=triangle,
                    outputs=outputs,
                    quality=quality,
                    method_notes=method_notes,
                    detection_summary=getattr(load_result, "structure_summary", None),
                    validation_issues=validation_frame_rows,
                    recognition_source=getattr(load_result, "recognition_source", "rules"),
                    recognition_reason=getattr(load_result, "recognition_reason", ""),
                    header_row=getattr(getattr(load_result, "region", None), "header_row", None),
                )
                chart_images = build_report_chart_images(triangle, outputs)
                files = {
                    "reserve_results.xlsx": excel_bytes,
                    "agent_report.docx": word_bytes,
                    **chart_images,
                }
                st.session_state.download_zip_bytes = build_zip_package(files, processing_log=processing_log)
        zip_bytes = st.session_state.get("download_zip_bytes")
        if zip_bytes:
            st.download_button(
                "下载 ZIP 打包",
                data=zip_bytes,
                file_name=f"{stem}_reserve_agent_outputs_{timestamp}.zip",
                mime="application/zip",
                use_container_width=True,
            )
