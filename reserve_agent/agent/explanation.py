from __future__ import annotations

import pandas as pd

from reserve_agent.data.loader import DataQualityReport
from reserve_agent.models.reserving import ReservingOutputs, format_currency


def generate_data_diagnosis(report: DataQualityReport) -> list[str]:
    messages = [
        f"系统读取到 {report.row_count} 行赔案度量记录，涉及 {report.claim_count} 个赔案或事故年。",
        f"事故年范围为 {min(report.accident_years)}-{max(report.accident_years)}，评估年范围为 "
        f"{min(report.valuation_years)}-{max(report.valuation_years)}。",
    ]
    if report.missing_values > 0:
        messages.append(
            f"数据中存在 {report.missing_values} 个空值，主要来自三角形右下角尚未观测的发展期或源表辅助列。"
        )
    if report.negative_amount_cells > 0:
        messages.append(
            f"系统识别到 {report.negative_amount_cells} 个负金额单元格，应结合追偿、冲回和录入修正进行复核。"
        )
    if report.zero_claim_rows > 0:
        messages.append(
            f"有 {report.zero_claim_rows} 行赔案度量在所有评估年金额均为 0，可视为未发生支付或已关闭零赔案。"
        )
    messages.extend(report.notes)
    return messages


def recommend_model(outputs: ReservingOutputs) -> str:
    total_cl = outputs.diagnostics["total_cl_reserve"]
    total_bf = outputs.diagnostics["total_bf_reserve"]
    total_elr = outputs.diagnostics["total_elr_reserve"]
    latest = outputs.diagnostics["total_latest"]
    reserve_ratio = outputs.diagnostics["total_selected_reserve"] / latest if latest else 0.0

    if reserve_ratio > 0.35:
        return "未决准备金占已观测赔款比例较高，建议以 Bornhuetter-Ferguson 作为主要参考，并保留 Chain Ladder 作为经验发展模式校验。"
    if total_cl > total_bf * 1.4:
        return "Chain Ladder 结果明显高于 BF，说明近期发展模式对最终赔款较敏感，建议复核大额赔案和最新事故年的成熟度。"
    if total_elr > max(total_cl, total_bf) * 1.5:
        return "ELR 结果偏高，可能反映先验赔付率或暴露基准较保守，应结合业务定价假设调整。"
    return "Chain Ladder 与 BF 差异处于可解释范围内，初版建议采用二者均值作为展示口径。"


def generate_result_summary(outputs: ReservingOutputs) -> list[str]:
    diag = outputs.diagnostics
    comparison = outputs.comparison.copy()
    total_selected = comparison["Selected Reserve"].sum()
    comparison["Reserve Share"] = comparison["Selected Reserve"] / total_selected if total_selected else 0.0
    max_year = int(comparison.sort_values("Selected Reserve", ascending=False).iloc[0]["Accident Year"])
    max_share = float(comparison["Reserve Share"].max())

    summary = [
        f"截至当前评估期，累计已观测赔款约为 {format_currency(diag['total_latest'])}。",
        f"Chain Ladder 估计准备金约为 {format_currency(diag['total_cl_reserve'])}，BF 估计准备金约为 "
        f"{format_currency(diag['total_bf_reserve'])}。",
        f"系统选取的展示口径为 Chain Ladder 与 BF 的均值，合计准备金约为 "
        f"{format_currency(diag['total_selected_reserve'])}，对应最终赔款约为 "
        f"{format_currency(diag['total_selected_ultimate'])}。",
        f"准备金贡献最高的事故年为 {max_year}，占总展示准备金约 {max_share:.1%}，应重点解释其赔款成熟度和发展因子影响。",
    ]

    if outputs.mack_diagnostics:
        summary.append(
            "Mack Chain Ladder 估计准备金约为 "
            f"{format_currency(outputs.mack_diagnostics.get('total_mack_reserve', 0.0))}，标准误约为 "
            f"{format_currency(outputs.mack_diagnostics.get('total_mack_standard_error', 0.0))}，95% 准备金区间约为 "
            f"{format_currency(outputs.mack_diagnostics.get('mack_95_lower', 0.0))}-"
            f"{format_currency(outputs.mack_diagnostics.get('mack_95_upper', 0.0))}。"
        )

    if outputs.expected_lr_sensitivity is not None and not outputs.expected_lr_sensitivity.empty:
        lr_low = outputs.expected_lr_sensitivity["ELR Reserve"].min()
        lr_high = outputs.expected_lr_sensitivity["ELR Reserve"].max()
        summary.append(
            f"期望赔付率敏感性显示，在测试参数范围内 ELR 准备金约为 "
            f"{format_currency(lr_low)}-{format_currency(lr_high)}。"
        )

    if outputs.factor_sensitivity is not None and not outputs.factor_sensitivity.empty:
        factor_low = outputs.factor_sensitivity["Chain Ladder Reserve"].min()
        factor_high = outputs.factor_sensitivity["Chain Ladder Reserve"].max()
        summary.append(
            f"发展因子敏感性显示，在因子冲击情景下 Chain Ladder 准备金约为 "
            f"{format_currency(factor_low)}-{format_currency(factor_high)}。"
        )

    summary.append(recommend_model(outputs))
    return summary


def generate_method_notes() -> dict[str, str]:
    return {
        "Chain Ladder": "链梯法假设历史赔款发展模式可以代表未来，通过累计赔款三角计算年龄到年龄发展因子，并将未成熟事故年的最新累计赔款外推至最终赔款。",
        "Expected Loss Ratio": "期望赔付率法基于先验赔付率或暴露量估计最终赔款，在早期事故年信息不足时可以作为稳定的基准模型。",
        "Bornhuetter-Ferguson": "BF 法将先验最终赔款与未报告比例结合，只对未成熟部分使用先验估计，因此比纯链梯法更能缓和早期事故年的波动。",
        "Mack Chain Ladder": "Mack 模型在链梯法基础上估计发展因子波动，并给出准备金标准误和区间，用于说明模型不确定性。",
        "Sensitivity Analysis": "敏感性分析用于观察关键假设变化对准备金的影响，本系统展示期望赔付率和发展因子冲击两类情景。",
    }


def generate_agent_explanation(report: DataQualityReport, outputs: ReservingOutputs) -> str:
    sections = [
        "【数据诊断】",
        *[f"- {item}" for item in generate_data_diagnosis(report)],
        "",
        "【模型结果解释】",
        *[f"- {item}" for item in generate_result_summary(outputs)],
        "",
        "【模型建议】",
        f"- {recommend_model(outputs)}",
        "- 若启用 DeepSeek API，系统会把当前结构化结果传给大模型，生成更自然的审阅意见、风险提示和报告摘要。",
    ]
    return "\n".join(sections)


def build_llm_payload(report: DataQualityReport, outputs: ReservingOutputs) -> dict:
    comparison = outputs.comparison.copy()
    numeric_cols = comparison.select_dtypes(include=["number"]).columns
    comparison[numeric_cols] = comparison[numeric_cols].round(2)
    return {
        "data_quality": {
            "row_count": report.row_count,
            "claim_count": report.claim_count,
            "accident_years": report.accident_years,
            "valuation_years": report.valuation_years,
            "missing_values": report.missing_values,
            "negative_amount_cells": report.negative_amount_cells,
            "zero_claim_rows": report.zero_claim_rows,
            "notes": report.notes,
        },
        "model_totals": {key: round(float(value), 2) for key, value in outputs.diagnostics.items()},
        "selected_factors": outputs.selected_factors.round(6).to_dict(),
        "comparison_by_accident_year": comparison.to_dict(orient="records"),
        "mack_by_accident_year": _round_frame(outputs.mack),
        "mack_diagnostics": {
            key: round(float(value), 2) for key, value in (outputs.mack_diagnostics or {}).items()
        },
        "expected_lr_sensitivity": _round_frame(outputs.expected_lr_sensitivity),
        "factor_sensitivity": _round_frame(outputs.factor_sensitivity),
    }


def _round_frame(frame: pd.DataFrame | None) -> list[dict]:
    if frame is None or frame.empty:
        return []
    result = frame.copy()
    numeric_cols = result.select_dtypes(include=["number"]).columns
    result[numeric_cols] = result[numeric_cols].round(2)
    return result.to_dict(orient="records")
