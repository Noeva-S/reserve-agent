from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reserve_agent.agent.chat_agent import answer_user_question  # noqa: E402
from reserve_agent.agent.context_builder import build_chat_context  # noqa: E402
from reserve_agent.agent.explanation import (  # noqa: E402
    build_llm_payload,
    generate_agent_explanation,
    generate_data_diagnosis,
    generate_method_notes,
    generate_result_summary,
)
from reserve_agent.agent.llm_client import build_reserving_prompt, call_deepseek, get_deepseek_key  # noqa: E402
from reserve_agent.data.loader import (  # noqa: E402
    choose_default_sheet,
    find_default_workbook,
    list_excel_sheets,
    load_excel_to_triangle,
    load_exposure_data,
    triangle_to_display,
)
from reserve_agent.data.validator import validate_triangle  # noqa: E402
from reserve_agent.exports import build_excel_download, build_word_report, build_zip_package  # noqa: E402
from reserve_agent.models.reserving import run_reserving_models  # noqa: E402


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
        if any(key in col for key in ["Cumulative", "Ultimate", "Reserve", "Loss", "Error", "Lower", "Upper"])
    ]
    for col in money_cols:
        formatted[col] = pd.to_numeric(formatted[col], errors="coerce")
    return formatted


@st.cache_data(show_spinner=False)
def load_pipeline(file_path: str, sheet_name: str, measure: str, expected_lr: float, is_cumulative: bool):
    load_result = load_excel_to_triangle(file_path, sheet_name, measure=measure, is_cumulative=is_cumulative)
    triangle = load_result.triangle
    exposure = load_exposure_data(file_path)
    report = load_result.quality
    outputs = run_reserving_models(triangle, exposure, expected_lr)
    validation = validate_triangle(triangle)
    return load_result, triangle, exposure, report, outputs, validation


st.set_page_config(page_title="非寿险准备金评估 Agent", layout="wide")
st.title("非寿险准备金评估智能 Agent")
st.caption("支持 Excel 自动识别、赔付三角建模、准备金结果解释、实时问答和结果下载。")

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
    measure = st.selectbox("赔付口径", ["Paid", "Incurred"], index=0)
    is_cumulative = st.toggle("上传长表金额已为累计口径", value=True)
    expected_lr = st.slider("期望赔付率参数", 0.30, 1.20, 0.72, 0.01)
    st.divider()
    use_deepseek = st.toggle("启用 DeepSeek API", value=False)

try:
    load_result, triangle, exposure_df, dq_report, outputs, validation_issues = load_pipeline(
        str(data_path), sheet_name, measure, expected_lr, is_cumulative
    )
except Exception as exc:
    st.error(f"数据处理失败：{exc}")
    st.stop()

diag = outputs.diagnostics
kpi_cols = st.columns(4)
kpi_cols[0].metric("记录数", f"{dq_report.row_count:,}")
kpi_cols[1].metric("累计已观察赔款", f"{diag['total_latest']:,.0f}")
kpi_cols[2].metric("展示准备金", f"{diag['total_selected_reserve']:,.0f}")
kpi_cols[3].metric("展示最终赔款", f"{diag['total_selected_ultimate']:,.0f}")

tabs = st.tabs(["数据诊断", "赔付三角", "模型结果", "可视化", "Agent 解释", "下载"])

with tabs[0]:
    st.subheader("Excel 识别结果")
    c1, c2, c3 = st.columns(3)
    c1.metric("识别格式", load_result.format_name)
    c2.metric("表头行", str(load_result.region.header_row + 1))
    c3.metric("候选区域数", str(len(load_result.candidates)))
    st.write("系统会先扫描标题、说明、空行之后的有效表格区域，再识别明细表、三角形或长表。")

    st.subheader("数据质量诊断")
    for item in generate_data_diagnosis(dq_report):
        st.write(f"- {item}")

    st.subheader("结构校验")
    for issue in validation_issues:
        if issue.level == "error":
            st.error(issue.message)
        elif issue.level == "warning":
            st.warning(issue.message)
        else:
            st.info(issue.message)

    st.subheader("识别后的数据预览")
    st.dataframe(load_result.source_table.head(30), use_container_width=True)

with tabs[1]:
    st.subheader(f"{measure} 累计赔付进展三角")
    st.dataframe(triangle_to_display(triangle), use_container_width=True)
    st.subheader("发展因子")
    factors = outputs.selected_factors.reset_index()
    factors.columns = ["Development Age", "Selected Factor"]
    st.dataframe(factors, use_container_width=True)

with tabs[2]:
    st.subheader("模型结果对比")
    st.dataframe(money_frame(outputs.comparison), use_container_width=True)
    if outputs.mack is not None and not outputs.mack.empty:
        st.subheader("Mack Chain Ladder uncertainty")
        mack_diag = outputs.mack_diagnostics or {}
        mack_cols = st.columns(4)
        mack_cols[0].metric("Mack reserve", f"{mack_diag.get('total_mack_reserve', 0.0):,.0f}")
        mack_cols[1].metric("Standard error", f"{mack_diag.get('total_mack_standard_error', 0.0):,.0f}")
        mack_cols[2].metric("Coefficient of variation", f"{mack_diag.get('total_mack_cv', 0.0):.1%}")
        mack_cols[3].metric(
            "95% interval",
            f"{mack_diag.get('mack_95_lower', 0.0):,.0f} - {mack_diag.get('mack_95_upper', 0.0):,.0f}",
        )
        st.dataframe(money_frame(outputs.mack), use_container_width=True)

    if outputs.expected_lr_sensitivity is not None and not outputs.expected_lr_sensitivity.empty:
        st.subheader("Expected loss ratio sensitivity")
        st.dataframe(money_frame(outputs.expected_lr_sensitivity), use_container_width=True)

    if outputs.factor_sensitivity is not None and not outputs.factor_sensitivity.empty:
        st.subheader("Development factor sensitivity")
        st.dataframe(money_frame(outputs.factor_sensitivity), use_container_width=True)
    for name, note in generate_method_notes().items():
        st.markdown(f"**{name}**：{note}")

with tabs[3]:
    st.subheader("结果可视化")
    comparison = outputs.comparison.copy()
    reserve_columns = ["Chain Ladder Reserve", "ELR Reserve", "BF Reserve", "Selected Reserve"]
    if "Mack Reserve" in comparison.columns:
        reserve_columns.insert(3, "Mack Reserve")
    long_reserve = comparison.melt(
        id_vars=["Accident Year"],
        value_vars=reserve_columns,
        var_name="Method",
        value_name="Reserve",
    )
    fig = px.bar(long_reserve, x="Accident Year", y="Reserve", color="Method", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    long_ultimate = outputs.comparison.melt(
        id_vars=["Accident Year"],
        value_vars=["Latest Cumulative", "Selected Ultimate"],
        var_name="Metric",
        value_name="Amount",
    )
    fig2 = px.line(long_ultimate, x="Accident Year", y="Amount", color="Metric", markers=True)
    st.plotly_chart(fig2, use_container_width=True)

    if outputs.mack is not None and not outputs.mack.empty:
        st.subheader("Mack reserve interval")
        mack_chart = outputs.mack.melt(
            id_vars=["Accident Year"],
            value_vars=["Mack Reserve", "Mack 95% Lower", "Mack 95% Upper"],
            var_name="Metric",
            value_name="Amount",
        )
        fig3 = px.line(mack_chart, x="Accident Year", y="Amount", color="Metric", markers=True)
        st.plotly_chart(fig3, use_container_width=True)

    if outputs.expected_lr_sensitivity is not None and not outputs.expected_lr_sensitivity.empty:
        st.subheader("Expected loss ratio sensitivity")
        lr_chart = outputs.expected_lr_sensitivity.melt(
            id_vars=["Expected Loss Ratio"],
            value_vars=["ELR Reserve", "BF Reserve"],
            var_name="Method",
            value_name="Reserve",
        )
        fig4 = px.line(lr_chart, x="Expected Loss Ratio", y="Reserve", color="Method", markers=True)
        st.plotly_chart(fig4, use_container_width=True)

    if outputs.factor_sensitivity is not None and not outputs.factor_sensitivity.empty:
        st.subheader("Development factor sensitivity")
        fig5 = px.line(
            outputs.factor_sensitivity,
            x="Factor Shock",
            y="Chain Ladder Reserve",
            markers=True,
        )
        st.plotly_chart(fig5, use_container_width=True)

with tabs[4]:
    st.subheader("自动解释")
    for item in generate_result_summary(outputs):
        st.write(f"- {item}")

    rule_text = generate_agent_explanation(dq_report, outputs)
    current_signature = f"{data_path}|{sheet_name}|{measure}|{expected_lr}|{is_cumulative}"
    if st.session_state.get("agent_signature") != current_signature:
        st.session_state.agent_signature = current_signature
        st.session_state.agent_final_text = rule_text
        st.session_state.reserve_chat_history = []

    final_text = st.session_state.get("agent_final_text", rule_text)
    if use_deepseek:
        if st.button("生成 / 刷新 DeepSeek 增强解释"):
            api_key = read_secret("DEEPSEEK_API_KEY") or get_deepseek_key()
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
    st.caption("可以继续追问当前模型结果。未启用 API 时，系统会使用规则型回答。")
    chat_context = build_chat_context(dq_report, outputs)
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

    user_question = st.chat_input("请输入问题，例如：哪个事故年准备金最高？CL 和 BF 差异大吗？")
    if user_question:
        history_before = st.session_state.reserve_chat_history.copy()
        st.session_state.reserve_chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)
        api_key = read_secret("DEEPSEEK_API_KEY") or get_deepseek_key() if use_deepseek else None
        with st.chat_message("assistant"):
            with st.spinner("正在生成回答..."):
                answer = answer_user_question(user_question, chat_context, api_key=api_key, chat_history=history_before)
            st.markdown(answer)
        st.session_state.reserve_chat_history.append({"role": "assistant", "content": answer})

with tabs[5]:
    st.subheader("下载结果")
    method_notes = generate_method_notes()
    explanation_text = st.session_state.get("agent_final_text", generate_agent_explanation(dq_report, outputs))
    excel_bytes = build_excel_download(outputs, triangle, dq_report)
    word_bytes = build_word_report(
        explanation_text,
        source_file=Path(data_path).name,
        sheet_name=sheet_name,
        format_name=load_result.format_name,
        measure=measure,
        triangle=triangle,
        outputs=outputs,
        quality=dq_report,
        method_notes=method_notes,
    )
    zip_bytes = build_zip_package(
        {
            "reserve_results.xlsx": excel_bytes,
            "agent_report.docx": word_bytes,
        }
    )

    col_a, col_b, col_c = st.columns(3)
    col_a.download_button(
        "下载 Excel 结果",
        data=excel_bytes,
        file_name="reserve_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    col_b.download_button(
        "下载 Word 报告",
        data=word_bytes,
        file_name="agent_report.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    col_c.download_button(
        "下载 ZIP 打包",
        data=zip_bytes,
        file_name="reserve_agent_outputs.zip",
        mime="application/zip",
    )
