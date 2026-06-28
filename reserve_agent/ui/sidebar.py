from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import streamlit as st

from reserve_agent.data.loader import find_default_workbook, list_excel_sheets


@dataclass
class SidebarState:
    data_path: Path
    sheet_name: str
    measure: str
    expected_lr: float
    use_deepseek: bool


def render_sidebar(project_root: Path) -> SidebarState:
    """Render the shared Streamlit sidebar controls."""

    with st.sidebar:
        st.header("数据与参数")
        default_path = find_default_workbook(project_root)
        uploaded = st.file_uploader("上传 Excel 数据", type=["xlsx"])
        if uploaded is not None:
            temp_path = project_root / "reserve_agent" / "_uploaded.xlsx"
            temp_path.write_bytes(uploaded.getvalue())
            data_path = temp_path
        else:
            data_path = default_path
            st.info(f"当前使用默认数据：{default_path.name}")

        sheets = list_excel_sheets(data_path)
        default_sheet_idx = sheets.index("Claims data") if "Claims data" in sheets else 0
        sheet_name = st.selectbox("赔案数据工作表", sheets, index=default_sheet_idx)
        measure = st.selectbox("三角口径", ["Paid", "Incurred"], index=0)
        expected_lr = st.slider("期望赔付率参数", 0.30, 1.20, 0.72, 0.01)
        st.divider()
        st.write("AI 增强解释")
        use_deepseek = st.toggle("启用 DeepSeek API 解释", value=False)

    return SidebarState(
        data_path=data_path,
        sheet_name=sheet_name,
        measure=measure,
        expected_lr=expected_lr,
        use_deepseek=use_deepseek,
    )
