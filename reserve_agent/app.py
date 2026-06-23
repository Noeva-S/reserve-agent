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

from reserve_agent.data_processing import (  # noqa: E402
    build_cumulative_triangle,
    find_default_workbook,
    list_excel_sheets,
    load_claims_snapshot,
    load_exposure_data,
    quality_report,
    triangle_to_display,
)
from reserve_agent.explanation import (  # noqa: E402
    build_llm_payload,
    generate_agent_explanation,
    generate_data_diagnosis,
    generate_method_notes,
    generate_result_summary,
)
from reserve_agent.llm_client import build_reserving_prompt, call_deepseek, get_deepseek_key  # noqa: E402
from reserve_agent.reserving import run_reserving_models  # noqa: E402


def read_secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:
        return None
    return str(value) if value else None


st.set_page_config(page_title="非寿险准备金评估 Agent", layout="wide")


def money_frame(df: pd.DataFrame) -> pd.DataFrame:
    formatted = df.copy()
    money_cols = [
        col
        for col in formatted.columns
        if any(key in col for key in ["Cumulative", "Ultimate", "Reserve", "Loss"])
    ]
    for col in money_cols:
        formatted[col] = pd.to_numeric(formatted[col], errors="coerce")
    return formatted


@st.cache_data(show_spinner=False)
def load_pipeline(file_path: str, sheet_name: str, measure: str, expected_lr: float):
    claims = load_claims_snapshot(file_path, sheet_name)
    triangle = build_cumulative_triangle(claims, measure=measure)
    exposure = load_exposure_data(file_path)
    report = quality_report(claims, triangle)
    outputs = run_reserving_models(triangle, exposure, expected_lr)
    return claims, triangle, exposure, report, outputs


st.title("非寿险准备金评估智能 Agent 初版")
st.caption("无 API 版本：自动读取数据、生成赔付三角、运行准备金模型，并基于规则生成诊断解释。")

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
    default_sheet_idx = sheets.index("Claims data") if "Claims data" in sheets else 0
    sheet_name = st.selectbox("赔案数据工作表", sheets, index=default_sheet_idx)
    measure = st.selectbox("三角口径", ["Paid", "Incurred"], index=0)
    expected_lr = st.slider("期望赔付率参数", 0.30, 1.20, 0.72, 0.01)
    st.divider()
    st.write("AI 增强解释")
    use_deepseek = st.toggle("启用 DeepSeek API 解释", value=False)

try:
    claims_df, triangle, exposure_df, dq_report, outputs = load_pipeline(
        str(data_path), sheet_name, measure, expected_lr
    )
except Exception as exc:
    st.error(f"数据处理失败：{exc}")
    st.stop()

diag = outputs.diagnostics
kpi_cols = st.columns(4)
kpi_cols[0].metric("赔案数量", f"{dq_report.claim_count:,}")
kpi_cols[1].metric("累计已观测赔款", f"{diag['total_latest']:,.0f}")
kpi_cols[2].metric("展示准备金", f"{diag['total_selected_reserve']:,.0f}")
kpi_cols[3].metric("展示最终赔款", f"{diag['total_selected_ultimate']:,.0f}")

tabs = st.tabs(["数据诊断", "赔付三角", "模型结果", "可视化", "Agent 解释"])

with tabs[0]:
    st.subheader("数据质量诊断")
    for item in generate_data_diagnosis(dq_report):
        st.write(f"- {item}")
    st.write("原始赔案数据预览")
    st.dataframe(claims_df.head(30), use_container_width=True)

with tabs[1]:
    st.subheader(f"{measure} 累计赔付进展三角")
    st.dataframe(triangle_to_display(triangle), use_container_width=True)
    st.write("发展因子")
    factors = outputs.selected_factors.reset_index()
    factors.columns = ["Development Age", "Selected Factor"]
    st.dataframe(factors, use_container_width=True)

with tabs[2]:
    st.subheader("模型结果对比")
    st.dataframe(money_frame(outputs.comparison), use_container_width=True)
    method_notes = generate_method_notes()
    for name, note in method_notes.items():
        st.markdown(f"**{name}**：{note}")

with tabs[3]:
    st.subheader("结果可视化")
    comparison = outputs.comparison.copy()
    long_reserve = comparison.melt(
        id_vars=["Accident Year"],
        value_vars=["Chain Ladder Reserve", "ELR Reserve", "BF Reserve", "Selected Reserve"],
        var_name="Method",
        value_name="Reserve",
    )
    fig = px.bar(long_reserve, x="Accident Year", y="Reserve", color="Method", barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    latest_cols = [col for col in outputs.comparison.columns if col in ["Latest Cumulative", "Selected Ultimate"]]
    long_ultimate = outputs.comparison.melt(
        id_vars=["Accident Year"],
        value_vars=latest_cols,
        var_name="Metric",
        value_name="Amount",
    )
    fig2 = px.line(long_ultimate, x="Accident Year", y="Amount", color="Metric", markers=True)
    st.plotly_chart(fig2, use_container_width=True)

with tabs[4]:
    st.subheader("规则型 Agent 自动解释")
    for item in generate_result_summary(outputs):
        st.write(f"- {item}")
    rule_text = generate_agent_explanation(dq_report, outputs)
    final_text = rule_text
    if use_deepseek:
        api_key = read_secret("DEEPSEEK_API_KEY") or get_deepseek_key()
        if not api_key:
            st.warning("未检测到 DeepSeek API Key，已回退到规则型解释。")
        else:
            with st.spinner("正在调用 DeepSeek 生成增强解释..."):
                try:
                    payload = build_llm_payload(dq_report, outputs)
                    messages = build_reserving_prompt(payload)
                    final_text = call_deepseek(messages, api_key=api_key)
                    st.success("DeepSeek 增强解释已生成。")
                except Exception as exc:
                    st.error(f"DeepSeek 调用失败，已保留规则型解释：{exc}")
                    final_text = rule_text
    st.text_area("可复制解释文本", final_text, height=420)
