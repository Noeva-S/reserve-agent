from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import plotly.express as px
import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reserve_agent.agent.chat_agent import answer_user_question  # noqa: E402
from reserve_agent.agent.context_builder import build_chat_context  # noqa: E402
from reserve_agent.agent.workbook_context import build_workbook_context  # noqa: E402
from reserve_agent.agent.explanation import (  # noqa: E402
    build_llm_payload,
    generate_agent_explanation,
    generate_data_diagnosis,
    generate_method_notes,
    generate_result_summary,
)
from reserve_agent.agent.llm_client import build_reserving_prompt, call_deepseek, get_deepseek_key  # noqa: E402
from reserve_agent.data.adapters import transform_to_standard_triangle  # noqa: E402
from reserve_agent.data.loader import (  # noqa: E402
    DataQualityReport,
    choose_default_sheet,
    find_default_workbook,
    list_excel_sheets,
    load_excel_to_triangle,
    load_exposure_data,
    quality_report,
    triangle_to_display,
)
from reserve_agent.data.mapping import FieldMapping  # noqa: E402
from reserve_agent.data.table_scanner import scan_excel_table  # noqa: E402
from reserve_agent.data.validator import validate_triangle  # noqa: E402
from reserve_agent.models.reserving import run_reserving_models  # noqa: E402
from reserve_agent.ui.download_panel import render_download_panel  # noqa: E402
from reserve_agent.ui.mapping_panel import render_mapping_panel  # noqa: E402


EXCEL_ROW_COL = "__excel_row__"


def read_secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:
        return None
    return str(value) if value else None


def money_frame(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    money_cols = [
        col
        for col in formatted.columns
        if any(key in str(col) for key in ["Cumulative", "Ultimate", "Reserve", "Loss"])
    ]
    for col in money_cols:
        formatted[col] = pd.to_numeric(formatted[col], errors="coerce")
    return formatted


def _visible_columns(df: pd.DataFrame) -> list[str]:
    return [str(c) for c in df.columns if str(c) != EXCEL_ROW_COL]


def _guess_column(columns: list[str], keywords: list[str]) -> str:
    for col in columns:
        lower = str(col).strip().lower()
        if any(k in lower for k in keywords):
            return col
    return ""


def _guess_amount_cols(columns: list[str], ay_col: str, dev_col: str, type_col: str, *, data_format: str) -> list[str]:
    exclude = {ay_col, dev_col, type_col, EXCEL_ROW_COL, ""}
    numeric_like: list[str] = []
    for col in columns:
        if col in exclude:
            continue
        text = str(col).strip().replace("年", "")
        try:
            numeric_value = float(text)
            if numeric_value.is_integer():
                numeric_like.append(col)
                continue
        except Exception:
            pass
        if text.isdigit():
            numeric_like.append(col)
    if numeric_like and data_format != "long_table":
        return numeric_like

    amount_keywords = ["paid", "incurred", "amount", "loss", "value", "赔款", "金额", "累计", "增量"]
    keyword_cols = [
        col
        for col in columns
        if col not in exclude and any(k in str(col).strip().lower() for k in amount_keywords)
    ]
    if keyword_cols:
        return keyword_cols[:1] if data_format == "long_table" else keyword_cols
    return []


def _format_to_panel_format(format_name: str) -> str:
    return "long_table" if str(format_name) == "long_table" else "wide_table"


def _build_detected_info_from_table(
    preview_df: pd.DataFrame,
    *,
    sheet_name: str,
    default_is_cum: bool,
    format_name: str = "wide_table",
    recognition_source: str = "rules",
    recognition_reason: str = "系统根据字段名称、发展期列和金额关键词进行初步识别。",
    confidence: float | str | None = None,
) -> dict:
    columns = _visible_columns(preview_df)
    data_format = _format_to_panel_format(format_name)
    ay_col = _guess_column(columns, ["accident", "loss year", "origin", "policy year", "事故", "出险", "保单年", "ay"])
    dev_col = _guess_column(columns, ["development", "dev", "delay", "lag", "发展", "进展", "延迟"])
    type_col = _guess_column(columns, ["type", "measure", "口径", "类型"])
    amount_cols = _guess_amount_cols(columns, ay_col, dev_col, type_col, data_format=data_format)
    if ay_col and dev_col and len(amount_cols) == 1:
        data_format = "long_table"
    return {
        "sheet_name": sheet_name,
        "data_format": data_format,
        "accident_year_col": ay_col,
        "development_col": dev_col,
        "amount_cols": amount_cols,
        "type_col_name": type_col,
        "measure_col_value": "Paid",
        "is_cumulative": default_is_cum,
        "confidence": "规则/API" if confidence is None else confidence,
        "recognition_source": recognition_source,
        "reason": recognition_reason,
    }


def _build_detected_info_from_load_result(load_result, *, default_is_cum: bool, measure: str) -> dict:
    info = _build_detected_info_from_table(
        load_result.source_table,
        sheet_name=getattr(load_result, "source_sheet_name", "") or getattr(load_result, "requested_sheet_name", ""),
        default_is_cum=default_is_cum,
        format_name=getattr(load_result, "format_name", "wide_table"),
        recognition_source=getattr(load_result, "recognition_source", "rules"),
        recognition_reason=getattr(load_result, "recognition_reason", ""),
        confidence=getattr(getattr(load_result, "candidates", [None])[-1], "confidence", None) if getattr(load_result, "candidates", None) else None,
    )
    if measure:
        info["measure_col_value"] = measure
    return info


def _mapping_from_detected_info(detected_info: dict) -> FieldMapping:
    return FieldMapping(
        accident_year_col=detected_info.get("accident_year_col", ""),
        development_col=detected_info.get("development_col", ""),
        amount_cols=detected_info.get("amount_cols", []) or [],
        type_col_name=detected_info.get("type_col_name", ""),
        measure_col_value=detected_info.get("measure_col_value", "Paid"),
        is_cumulative=bool(detected_info.get("is_cumulative", True)),
    )


def _show_detection_card(detected_info: dict) -> None:
    with st.container(border=True):
        st.subheader("① AI / 规则识别结果")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("工作表", detected_info.get("sheet_name", ""))
        c2.metric("数据格式", "长表" if detected_info.get("data_format") == "long_table" else "宽表 / 三角形")
        c3.metric("识别来源", "DeepSeek API" if detected_info.get("recognition_source") == "api" else "本地规则")
        c4.metric("识别置信度", str(detected_info.get("confidence") or "无"))
        field_df = pd.DataFrame(
            [
                {"识别项": "事故年列", "识别结果": detected_info.get("accident_year_col") or "未识别"},
                {"识别项": "发展期列", "识别结果": detected_info.get("development_col") or "未识别"},
                {"识别项": "金额列", "识别结果": ", ".join(detected_info.get("amount_cols") or []) or "未识别"},
                {"识别项": "口径列", "识别结果": detected_info.get("type_col_name") or "无"},
                {"识别项": "目标口径", "识别结果": detected_info.get("measure_col_value") or "无"},
                {"识别项": "累计/增量", "识别结果": "累计" if detected_info.get("is_cumulative") else "增量"},
                {"识别项": "识别理由", "识别结果": detected_info.get("reason") or "无"},
            ]
        )
        st.dataframe(field_df, use_container_width=True, hide_index=True)


def _show_issues(issues) -> None:
    for issue in issues:
        if issue.level == "error":
            st.error(issue.message)
        elif issue.level == "warning":
            st.warning(issue.message)
        else:
            st.info(issue.message)


def _show_manual_diagnosis_details(triangle: pd.DataFrame) -> None:
    detail_items = [
        ("negative_rows", "原始负值明细"),
        ("garbage_rows", "无效文字明细"),
        ("gap_exclusions", "空值导致的发展因子剔除记录"),
        ("cumulative_drop_log", "累计下降记录"),
        ("smoothing_log", "累计负值截断记录"),
    ]
    for attr_name, title in detail_items:
        frame = triangle.attrs.get(attr_name)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            with st.expander(title, expanded=False):
                st.dataframe(frame, use_container_width=True)


def _render_visualizations(triangle: pd.DataFrame, outputs, selected_measure: str) -> None:
    st.subheader("结果可视化")
    st.caption("本页展示赔付三角、发展因子、模型差异、准备金贡献、不确定性和敏感性分析。")

    comparison = outputs.comparison.copy()
    measure_name_map = {"Paid": "已付", "Incurred": "已发生"}
    measure_cn = measure_name_map.get(selected_measure, selected_measure)

    st.markdown("### 赔付三角热力图")
    triangle_for_heatmap = triangle.apply(pd.to_numeric, errors="coerce")
    if triangle_for_heatmap.empty:
        st.info("当前没有可用于绘制热力图的赔付三角数据。")
    else:
        fig_heatmap = px.imshow(
            triangle_for_heatmap,
            x=[str(col) for col in triangle_for_heatmap.columns],
            y=[str(idx) for idx in triangle_for_heatmap.index],
            labels={"x": "发展期", "y": "事故年", "color": f"累计{measure_cn}赔款"},
            aspect="auto",
            color_continuous_scale="Blues",
        )
        fig_heatmap.update_layout(height=520, margin=dict(l=40, r=40, t=30, b=40))
        fig_heatmap.update_coloraxes(colorbar_title=f"累计{measure_cn}赔款")
        st.plotly_chart(fig_heatmap, use_container_width=True)
        st.caption("颜色越深表示对应事故年和发展期的累计赔款越高，可用于观察赔款随发展期累积的整体模式。")

    st.markdown("### 发展因子图")
    factor_df = outputs.selected_factors.reset_index()
    factor_df.columns = ["发展期", "发展因子"]
    factor_df["发展期"] = factor_df["发展期"].astype(str)
    factor_df["发展因子"] = pd.to_numeric(factor_df["发展因子"], errors="coerce")
    if factor_df.empty:
        st.info("当前没有可用于绘制发展因子的结果。")
    else:
        fig_factor = px.line(factor_df, x="发展期", y="发展因子", markers=True)
        fig_factor.update_layout(
            xaxis_title="发展期",
            yaxis_title="发展因子",
            height=420,
            margin=dict(l=40, r=40, t=30, b=40),
        )
        st.plotly_chart(fig_factor, use_container_width=True)
        st.caption("发展因子反映相邻发展期累计赔款的增长倍数，越接近 1 表示后续发展空间越小。")

    st.markdown("### 各模型准备金差异图")
    reserve_columns = [
        col
        for col in [
            "Chain Ladder Reserve",
            "ELR Reserve",
            "BF Reserve",
            "Mack Reserve",
        ]
        if col in comparison.columns
    ]
    reserve_name_map = {
        "Chain Ladder Reserve": "链梯法",
        "ELR Reserve": "期望赔付率法",
        "BF Reserve": "BF法",
        "Mack Reserve": "Mack链梯法",
    }
    if not reserve_columns or "Accident Year" not in comparison.columns:
        st.info("当前没有足够的模型准备金结果用于绘制模型差异图。")
    else:
        long_reserve = comparison.melt(
            id_vars=["Accident Year"],
            value_vars=reserve_columns,
            var_name="模型",
            value_name="准备金",
        )
        long_reserve["事故年"] = long_reserve["Accident Year"].astype(str)
        long_reserve["模型"] = long_reserve["模型"].replace(reserve_name_map)
        long_reserve["准备金"] = pd.to_numeric(long_reserve["准备金"], errors="coerce")
        fig_reserve = px.bar(long_reserve, x="事故年", y="准备金", color="模型", barmode="group")
        fig_reserve.update_layout(
            xaxis_title="事故年",
            yaxis_title="准备金",
            height=520,
            margin=dict(l=40, r=40, t=30, b=40),
            legend_title_text="模型",
        )
        st.plotly_chart(fig_reserve, use_container_width=True)
        st.caption("该图比较不同准备金方法在各事故年上的准备金估计差异，可用于观察模型假设变化对结果的影响。")

    st.markdown("### 累计赔款与各模型最终赔款估计图")
    ultimate_columns = [
        col
        for col in [
            "Latest Cumulative",
            "Chain Ladder Ultimate",
            "Expected Ultimate Loss",
            "BF Ultimate Loss",
            "Mack Ultimate",
        ]
        if col in comparison.columns
    ]
    if "Accident Year" not in comparison.columns or len(ultimate_columns) < 2:
        st.info("当前没有足够数据绘制累计赔款与最终赔款估计对比图。")
    else:
        long_ultimate = comparison.melt(
            id_vars=["Accident Year"],
            value_vars=ultimate_columns,
            var_name="指标",
            value_name="金额",
        )
        long_ultimate["事故年"] = long_ultimate["Accident Year"].astype(str)
        long_ultimate["指标"] = long_ultimate["指标"].replace(
            {
                "Latest Cumulative": "当前累计赔款",
                "Chain Ladder Ultimate": "链梯法最终赔款",
                "Expected Ultimate Loss": "期望赔付率法最终赔款",
                "BF Ultimate Loss": "BF法最终赔款",
                "Mack Ultimate": "Mack链梯法最终赔款",
            }
        )
        long_ultimate["金额"] = pd.to_numeric(long_ultimate["金额"], errors="coerce")
        fig_ultimate = px.line(long_ultimate, x="事故年", y="金额", color="指标", markers=True)
        fig_ultimate.update_layout(height=460, margin=dict(l=40, r=40, t=30, b=40), legend_title_text="指标")
        st.plotly_chart(fig_ultimate, use_container_width=True)
        st.caption("该图比较当前累计赔款与各模型最终赔款估计。")

    if outputs.mack is not None and not outputs.mack.empty:
        st.markdown("### Mack 链梯法准备金区间图")
        mack_interval_columns = [
            col for col in ["Mack Reserve", "Mack 95% Lower", "Mack 95% Upper"] if col in outputs.mack.columns
        ]
        if "Accident Year" not in outputs.mack.columns or len(mack_interval_columns) < 2:
            st.info("当前 Mack 链梯法结果中没有足够字段绘制准备金区间图。")
        else:
            mack_chart = outputs.mack.melt(
                id_vars=["Accident Year"],
                value_vars=mack_interval_columns,
                var_name="指标",
                value_name="金额",
            )
            mack_chart["事故年"] = mack_chart["Accident Year"].astype(str)
            mack_chart["指标"] = mack_chart["指标"].replace(
                {"Mack Reserve": "Mack 链梯法准备金", "Mack 95% Lower": "95%下界", "Mack 95% Upper": "95%上界"}
            )
            mack_chart["金额"] = pd.to_numeric(mack_chart["金额"], errors="coerce")
            fig_mack = px.line(mack_chart, x="事故年", y="金额", color="指标", markers=True)
            fig_mack.update_layout(height=460, margin=dict(l=40, r=40, t=30, b=40), legend_title_text="指标")
            st.plotly_chart(fig_mack, use_container_width=True)
            st.caption("该图展示 Mack 链梯法准备金及其近似 95% 不确定性区间，区间越宽表示估计不确定性越高。")

        if "Mack CV" in outputs.mack.columns:
            st.markdown("### Mack 链梯法变异系数图")
            mack_cv_df = outputs.mack[["Accident Year", "Mack CV"]].copy()
            mack_cv_df["事故年"] = mack_cv_df["Accident Year"].astype(str)
            mack_cv_df["变异系数"] = pd.to_numeric(mack_cv_df["Mack CV"], errors="coerce") * 100
            fig_cv = px.bar(mack_cv_df, x="事故年", y="变异系数")
            fig_cv.update_layout(height=420, margin=dict(l=40, r=40, t=30, b=40))
            fig_cv.update_yaxes(ticksuffix="%")
            st.plotly_chart(fig_cv, use_container_width=True)
            st.caption("变异系数衡量准备金标准误相对于准备金估计值的大小，数值越高表示相对不确定性越强。")

    if outputs.expected_lr_sensitivity is not None and not outputs.expected_lr_sensitivity.empty:
        st.markdown("### 期望赔付率敏感性图")
        lr_value_columns = [
            col for col in ["ELR Reserve", "BF Reserve"] if col in outputs.expected_lr_sensitivity.columns
        ]
        if "Expected Loss Ratio" not in outputs.expected_lr_sensitivity.columns or not lr_value_columns:
            st.info("当前期望赔付率敏感性结果中没有足够字段用于绘图。")
        else:
            lr_chart = outputs.expected_lr_sensitivity.melt(
                id_vars=["Expected Loss Ratio"],
                value_vars=lr_value_columns,
                var_name="方法",
                value_name="准备金",
            )
            lr_chart["方法"] = lr_chart["方法"].replace(
                {"ELR Reserve": "期望赔付率法准备金", "BF Reserve": "BF法准备金"}
            )
            lr_chart["准备金"] = pd.to_numeric(lr_chart["准备金"], errors="coerce")
            fig_lr = px.line(lr_chart, x="Expected Loss Ratio", y="准备金", color="方法", markers=True)
            fig_lr.update_layout(height=460, margin=dict(l=40, r=40, t=30, b=40), legend_title_text="方法")
            st.plotly_chart(fig_lr, use_container_width=True)
            st.caption("该图展示期望赔付率参数变化对期望赔付率法和 BF 法准备金的影响。")

    if outputs.factor_sensitivity is not None and not outputs.factor_sensitivity.empty:
        st.markdown("### 发展因子敏感性图")
        if "Factor Shock" not in outputs.factor_sensitivity.columns or "Chain Ladder Reserve" not in outputs.factor_sensitivity.columns:
            st.info("当前发展因子敏感性结果中没有足够字段用于绘图。")
        else:
            factor_sensitivity_df = outputs.factor_sensitivity.copy()
            factor_sensitivity_df["链梯法准备金"] = pd.to_numeric(
                factor_sensitivity_df["Chain Ladder Reserve"],
                errors="coerce",
            )
            fig_factor_sens = px.line(
                factor_sensitivity_df,
                x="Factor Shock",
                y="链梯法准备金",
                markers=True,
            )
            fig_factor_sens.update_layout(height=460, margin=dict(l=40, r=40, t=30, b=40))
            st.plotly_chart(fig_factor_sens, use_container_width=True)
            st.caption("该图展示发展因子整体上调或下调时链梯法准备金的变化。")


def _manual_quality_report(source_table: pd.DataFrame, triangle: pd.DataFrame, data_format: str) -> DataQualityReport:
    try:
        pseudo = pd.DataFrame(
            {
                "Claim ID": range(1, max(len(source_table), 1) + 1),
                "Loss Year": [triangle.index[0] if len(triangle.index) else 0] * max(len(source_table), 1),
                "Type": ["Manual"] * max(len(source_table), 1),
            }
        )
        return quality_report(pseudo, triangle)
    except Exception:
        numeric = triangle.apply(pd.to_numeric, errors="coerce")
        return DataQualityReport(
            row_count=int(len(source_table)),
            claim_count=int(len(triangle.index)),
            accident_years=[int(x) for x in triangle.index.tolist()],
            valuation_years=sorted(
                {int(ay) + int(dev) for ay, row in numeric.iterrows() for dev, val in row.items() if pd.notna(val)}
            ),
            missing_values=int(numeric.isna().sum().sum()),
            negative_amount_cells=int((numeric < 0).sum().sum()),
            zero_claim_rows=int((numeric.fillna(0).sum(axis=1) == 0).sum()),
            notes=[f"当前结果来自字段映射确认模式，原始数据格式：{data_format}。"],
        )


def _build_manual_load_result(
    *,
    source_table: pd.DataFrame,
    triangle: pd.DataFrame,
    quality: DataQualityReport,
    requested_sheet_name: str,
    source_sheet_name: str,
    format_name: str,
    header_row: int,
    detected_info: dict,
    mapping: FieldMapping,
):
    region = SimpleNamespace(header_row=header_row, start_col=0, end_col=len(source_table.columns), end_row=len(source_table) + header_row + 1)
    structure_summary = {
        "sheet_name": source_sheet_name or requested_sheet_name,
        "candidate_count": 1,
        "candidate_header_rows": [header_row],
        "candidates": [
            {
                "candidate_index": 0,
                "sheet_name": source_sheet_name or requested_sheet_name,
                "header_row": header_row,
                "header_row_excel": header_row + 1,
                "rule_format": format_name,
                "rule_score": "manual_confirmed",
                "row_count": int(len(source_table)),
                "non_empty_cols": int(source_table.drop(columns=[EXCEL_ROW_COL], errors="ignore").shape[1]),
                "candidate_columns": [str(c) for c in source_table.columns if str(c) != EXCEL_ROW_COL],
                "mapping": mapping.to_dict(),
            }
        ],
        "manual_detected_info": detected_info,
        "privacy_note": "手动模式导出仅记录结构、列名和确认后的字段映射，不保存完整原始 Excel。",
    }
    return SimpleNamespace(
        source_table=source_table,
        triangle=triangle,
        format_name=format_name,
        region=region,
        candidates=[],
        quality=quality,
        requested_sheet_name=requested_sheet_name,
        source_sheet_name=source_sheet_name or requested_sheet_name,
        recognition_source="manual_mapping",
        recognition_reason="用户在字段映射确认面板中手动确认或修正字段后生成标准赔付三角形。",
        structure_summary=structure_summary,
        warnings=("当前结果来自字段映射确认/手动修正模式。",),
    )


@st.cache_data(show_spinner=False)
def load_pipeline_auto(
    file_path: str,
    sheet_name: str,
    measure: str,
    expected_lr: float,
    is_cumulative: bool,
    use_api_detection: bool = False,
    _api_key: str | None = None,
):
    load_result = load_excel_to_triangle(
        file_path,
        sheet_name,
        measure=measure,
        is_cumulative=is_cumulative,
        api_key=_api_key if use_api_detection else None,
    )
    triangle = load_result.triangle
    exposure = load_exposure_data(file_path)
    report = load_result.quality
    outputs = run_reserving_models(triangle, exposure, expected_lr)
    validation = validate_triangle(triangle)
    return load_result, triangle, exposure, report, outputs, validation


st.set_page_config(page_title="非寿险准备金评估 Agent", layout="wide")
st.title("非寿险准备金评估智能 Agent")
st.caption("支持 Excel 自动/AI 辅助识别、字段映射确认、赔付三角建模、准备金解释、实时问答和增强导出。")

with st.sidebar:
    st.header("数据与参数")
    default_path = find_default_workbook(PROJECT_ROOT)
    uploaded = st.file_uploader("上传 Excel 数据", type=["xlsx"])
    if uploaded is not None:
        temp_path = PROJECT_ROOT / "reserve_agent" / "_uploaded.xlsx"
        temp_path.write_bytes(uploaded.getvalue())
        data_path = temp_path
    else:
        data_path = default_path
        st.info(f"当前使用默认数据：{default_path.name}")

    sheets = list_excel_sheets(data_path)
    sheet_name = st.selectbox("工作表", sheets, index=choose_default_sheet(sheets))
    measure = st.selectbox("默认赔付口径", ["Paid", "Incurred"], index=0)
    is_cumulative = st.toggle("上传长表金额已为累计口径", value=True)
    expected_lr = st.slider("期望赔付率参数", 0.30, 1.20, 0.72, 0.01)
    st.divider()
    parse_mode = st.radio(
        "解析模式",
        ["自动识别 + AI 辅助", "字段映射确认 / 手动修正"],
        help="自动模式优先使用本地规则，低置信度或失败时可由 DeepSeek 辅助；手动模式接入队友第 3 部分的字段映射确认面板。",
    )
    use_deepseek = st.toggle("启用 DeepSeek API", value=False)

api_key = (read_secret("DEEPSEEK_API_KEY") or get_deepseek_key()) if use_deepseek else None
if use_deepseek and not api_key:
    st.sidebar.warning("未检测到 DEEPSEEK_API_KEY；Excel 将只使用本地规则识别，解释也会回退到规则型。")

# 先尝试自动识别；手动模式会把自动识别结果作为默认建议，但不会强依赖它成功。
auto_error: Exception | None = None
auto_bundle = None
try:
    auto_bundle = load_pipeline_auto(
        str(data_path),
        sheet_name,
        measure,
        expected_lr,
        is_cumulative,
        use_deepseek,
        _api_key=api_key,
    )
except Exception as exc:
    auto_error = exc

if parse_mode == "自动识别 + AI 辅助":
    if auto_bundle is None:
        st.error(f"数据处理失败：{auto_error}")
        st.info("可以切换到“字段映射确认 / 手动修正”模式，手动指定表头行、数据区域和字段映射。")
        st.stop()
    load_result, triangle, exposure_df, dq_report, outputs, validation_issues = auto_bundle
    claims_df = load_result.source_table
    selected_measure = measure
    detected_info = _build_detected_info_from_load_result(load_result, default_is_cum=is_cumulative, measure=measure)
else:
    st.info("当前为字段映射确认模式：可以采用自动识别建议，也可以手动修改字段、表头行或表格区域。")
    if auto_bundle is not None:
        auto_load_result = auto_bundle[0]
        detected_info = _build_detected_info_from_load_result(auto_load_result, default_is_cum=is_cumulative, measure=measure)
        default_header_row = int(getattr(getattr(auto_load_result, "region", None), "header_row", 0) or 0)
        if auto_load_result.warnings:
            for warning in auto_load_result.warnings:
                st.warning(warning)
    else:
        st.warning(f"自动识别未成功：{auto_error}。下面将以第 1 行为默认表头进行手动扫描。")
        try:
            preview_df_for_guess = scan_excel_table(str(data_path), sheet_name, header_row=0)
        except Exception as exc:
            st.error(f"手动扫描初始化失败：{exc}")
            st.stop()
        detected_info = _build_detected_info_from_table(
            preview_df_for_guess,
            sheet_name=sheet_name,
            default_is_cum=is_cumulative,
            format_name="wide_table",
            recognition_source="rules",
            recognition_reason="自动识别失败后，系统按第 1 行表头生成手动映射初始建议。",
        )
        default_header_row = 0

    _show_detection_card(detected_info)
    panel_result = render_mapping_panel(
        file_path=str(data_path),
        sheet_name=sheet_name,
        detected_info=detected_info,
        default_header_row=default_header_row,
        key_prefix="manual_mapping",
    )
    if not panel_result.confirmed:
        st.stop()

    triangle = transform_to_standard_triangle(panel_result.preview_df, panel_result.mapping)
    validation_issues = validate_triangle(triangle)
    if triangle.attrs.get("model_blocking_error", False):
        st.error("当前字段映射后的数据存在阻断性问题，无法进入准备金模型。请修改字段映射或原始数据后重新确认。")
        _show_issues(validation_issues)
        _show_manual_diagnosis_details(triangle)
        with st.expander("字段映射后的标准三角", expanded=True):
            st.dataframe(triangle_to_display(triangle), use_container_width=True)
        st.stop()

    exposure_df = load_exposure_data(str(data_path))
    outputs = run_reserving_models(triangle, exposure_df, expected_lr)
    selected_measure = panel_result.selected_measure or measure
    dq_report = _manual_quality_report(panel_result.preview_df, triangle, panel_result.data_format)
    claims_df = panel_result.preview_df
    load_result = _build_manual_load_result(
        source_table=panel_result.preview_df,
        triangle=triangle,
        quality=dq_report,
        requested_sheet_name=sheet_name,
        source_sheet_name=sheet_name,
        format_name=panel_result.data_format,
        header_row=panel_result.header_row,
        detected_info=detected_info,
        mapping=panel_result.mapping,
    )

# 顶部 KPI
diag = outputs.diagnostics
kpi_cols = st.columns(5)
kpi_cols[0].metric("数据行数", f"{dq_report.row_count:,}")
kpi_cols[1].metric("累计已观察赔款", f"{diag['total_latest']:,.0f}")
kpi_cols[2].metric("CL 准备金", f"{diag['total_cl_reserve']:,.0f}")
kpi_cols[3].metric("BF 准备金", f"{diag['total_bf_reserve']:,.0f}")
kpi_cols[4].metric("Mack 准备金", f"{diag.get('total_mack_reserve', 0.0):,.0f}")

tabs = st.tabs(["数据诊断 / 字段映射", "赔付三角", "模型结果", "可视化", "Agent 解释", "下载"])

with tabs[0]:
    st.subheader("Excel 识别与映射结果")
    for warning in getattr(load_result, "warnings", ()):  # type: ignore[arg-type]
        st.warning(warning)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("识别格式", getattr(load_result, "format_name", ""))
    c2.metric("表头行", str(getattr(getattr(load_result, "region", None), "header_row", 0) + 1))
    c3.metric("候选区域数", str(len(getattr(load_result, "candidates", [])) or getattr(getattr(load_result, "structure_summary", {}), "candidate_count", 1)))
    c4.metric("实际数据 Sheet", getattr(load_result, "source_sheet_name", "") or sheet_name)
    c5, c6 = st.columns(2)
    c5.metric("识别来源", "DeepSeek API" if getattr(load_result, "recognition_source", "rules") == "api" else ("手动确认" if getattr(load_result, "recognition_source", "rules") == "manual_mapping" else "本地规则"))
    c6.metric("说明", getattr(load_result, "recognition_reason", "") or "规则识别完成")

    st.write(
        "系统会先扫描说明行、空行之后的有效表格区域；本地规则失败或置信度偏低且启用 API 时，"
        "只提交结构摘要、列名、类型统计和少量脱敏样本进行辅助识别。手动模式下，字段映射确认面板的结果会覆盖自动识别结果。"
    )
    if getattr(load_result, "structure_summary", None):
        with st.expander("查看候选表结构摘要 / 手动映射记录"):
            summary_rows = []
            for item in load_result.structure_summary.get("candidates", []):
                summary_rows.append(
                    {
                        "候选编号": item.get("candidate_index"),
                        "表头行": item.get("header_row_excel"),
                        "规则格式": item.get("rule_format"),
                        "数据行数": item.get("row_count"),
                        "非空列数": item.get("non_empty_cols"),
                        "候选列名": ", ".join(map(str, item.get("candidate_columns", [])[:12])),
                    }
                )
            if summary_rows:
                st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
            mapping = load_result.structure_summary.get("candidates", [{}])[0].get("mapping")
            if mapping:
                st.json(mapping)

    st.subheader("数据质量诊断")
    for item in generate_data_diagnosis(dq_report):
        st.write(f"- {item}")

    st.subheader("结构校验")
    _show_issues(validation_issues)
    _show_manual_diagnosis_details(triangle)

    st.subheader("识别/映射后的数据预览")
    st.dataframe(claims_df.head(30), use_container_width=True)

with tabs[1]:
    st.subheader(f"{selected_measure} 累计赔付进展三角")
    st.dataframe(triangle_to_display(triangle), use_container_width=True)
    st.subheader("发展因子")
    factors = outputs.selected_factors.reset_index()
    factors.columns = ["Development Age", "Selected Factor"]
    st.dataframe(factors, use_container_width=True)

with tabs[2]:
    st.subheader("模型结果对比")
    st.dataframe(money_frame(outputs.comparison), use_container_width=True)
    if outputs.mack is not None and not outputs.mack.empty:
        st.subheader("Mack Chain Ladder 不确定性")
        mack_diag = outputs.mack_diagnostics or {}
        mack_cols = st.columns(4)
        mack_cols[0].metric("Mack 准备金", f"{mack_diag.get('total_mack_reserve', 0.0):,.0f}")
        mack_cols[1].metric("标准误", f"{mack_diag.get('total_mack_standard_error', 0.0):,.0f}")
        mack_cols[2].metric("变异系数", f"{mack_diag.get('total_mack_cv', 0.0):.1%}")
        mack_cols[3].metric(
            "95% 区间",
            f"{mack_diag.get('mack_95_lower', 0.0):,.0f} - {mack_diag.get('mack_95_upper', 0.0):,.0f}",
        )
        st.dataframe(money_frame(outputs.mack), use_container_width=True)

    if outputs.expected_lr_sensitivity is not None and not outputs.expected_lr_sensitivity.empty:
        st.subheader("期望赔付率敏感性")
        st.dataframe(money_frame(outputs.expected_lr_sensitivity), use_container_width=True)

    if outputs.factor_sensitivity is not None and not outputs.factor_sensitivity.empty:
        st.subheader("发展因子敏感性")
        st.dataframe(money_frame(outputs.factor_sensitivity), use_container_width=True)

    for name, note in generate_method_notes().items():
        st.markdown(f"**{name}**：{note}")

with tabs[3]:
    _render_visualizations(triangle, outputs, selected_measure)

with tabs[4]:
    st.subheader("自动解释")
    for item in generate_result_summary(outputs):
        st.write(f"- {item}")

    try:
        rule_text = generate_agent_explanation(dq_report, outputs)
    except Exception:
        rule_text = "【规则型解释生成失败】当前数据结构不完整，已跳过自动解释。"

    current_signature = f"{data_path}|{sheet_name}|{selected_measure}|{expected_lr}|{is_cumulative}|{parse_mode}|{getattr(load_result, 'recognition_source', '')}"
    if st.session_state.get("agent_signature") != current_signature:
        st.session_state.agent_signature = current_signature
        st.session_state.agent_final_text = rule_text
        st.session_state.reserve_chat_history = []

    final_text = st.session_state.get("agent_final_text", rule_text)
    if use_deepseek:
        if st.button("生成 / 刷新 DeepSeek 增强解释"):
            if not api_key:
                st.warning("未检测到 DeepSeek API Key，已保留规则型解释。")
            else:
                with st.spinner("正在调用 DeepSeek 生成增强解释..."):
                    try:
                        messages = build_reserving_prompt(build_llm_payload(dq_report, outputs))
                        final_text = call_deepseek(messages, api_key=api_key)
                        st.session_state.agent_final_text = final_text
                        st.success("DeepSeek 增强解释已生成。")
                    except Exception as exc:
                        st.error(f"DeepSeek 调用失败，已保留规则型解释：{exc}")
                        final_text = rule_text
    else:
        final_text = rule_text
        st.session_state.agent_final_text = final_text

    st.text_area("可复制解释文本", final_text, height=360)

    st.divider()
    st.subheader("实时问答 Agent")
    st.caption(
        "可以继续追问当前模型结果、Mack 不确定性、敏感性分析，以及上传 Excel 中各 sheet 的信息。"
        "目前可保留 20 轮对话结果，请注意及时保存。"
    )
    try:
        workbook_context = build_workbook_context(
            file_path=data_path,
            selected_sheet=sheet_name,
            load_result=load_result,
        )
    except Exception as exc:
        st.warning(f"工作簿上下文构建失败，实时问答将主要基于模型结果回答：{exc}")
        workbook_context = {}
    chat_context = build_chat_context(
        dq_report,
        outputs,
        workbook_context=workbook_context,
        validation_issues=validation_issues,
    )
    if "reserve_chat_history" not in st.session_state:
        st.session_state.reserve_chat_history = []

    clear_col, _ = st.columns([1, 5])
    with clear_col:
        if st.button("清空问答记录"):
            st.session_state.reserve_chat_history = []
            st.rerun()

    for msg in st.session_state.reserve_chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_question = st.chat_input("请输入问题，例如：哪个事故年准备金最高？policy 表里有什么信息？")
    if user_question:
        history_before = st.session_state.reserve_chat_history.copy()
        st.session_state.reserve_chat_history.append({"role": "user", "content": user_question})
        chat_api_key = api_key if use_deepseek else None
        with st.spinner("正在生成回答..."):
            answer = answer_user_question(
                question=user_question,
                context=chat_context,
                api_key=chat_api_key,
                chat_history=history_before,
                workbook_file_path=str(data_path),
            )
        st.session_state.reserve_chat_history.append({"role": "assistant", "content": answer})
        st.rerun()

with tabs[5]:
    st.subheader("下载结果")
    method_notes = generate_method_notes()
    explanation_text = st.session_state.get("agent_final_text", generate_agent_explanation(dq_report, outputs))
    download_signature = f"{data_path}|{sheet_name}|{selected_measure}|{expected_lr}|{is_cumulative}|{parse_mode}|{getattr(load_result, 'source_sheet_name', '')}|{getattr(load_result, 'format_name', '')}"
    render_download_panel(
        load_result=load_result,
        triangle=triangle,
        outputs=outputs,
        quality=dq_report,
        validation_issues=validation_issues,
        explanation_text=explanation_text,
        method_notes=method_notes,
        source_file=Path(data_path).name,
        requested_sheet_name=sheet_name,
        measure=selected_measure,
        expected_lr=expected_lr,
        is_cumulative=is_cumulative,
        signature=download_signature,
    )
