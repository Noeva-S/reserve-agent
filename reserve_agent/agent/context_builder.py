from __future__ import annotations

from typing import Any

import pandas as pd

from reserve_agent.agent.explanation import build_llm_payload
from reserve_agent.data.loader import DataQualityReport
from reserve_agent.models.reserving import ReservingOutputs


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _money(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return "-"
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:,.2f}m"
    if abs(number) >= 1_000:
        return f"{number / 1_000:,.1f}k"
    return f"{number:,.0f}"


def build_chat_context(report: DataQualityReport, outputs: ReservingOutputs) -> dict[str, Any]:
    """Build compact, JSON-safe context for follow-up Agent chat."""

    payload = build_llm_payload(report, outputs)
    comparison = outputs.comparison.copy()

    reserve_columns = {
        "Chain Ladder": "Chain Ladder Reserve",
        "Expected Loss Ratio": "ELR Reserve",
        "Bornhuetter-Ferguson": "BF Reserve",
        "Selected": "Selected Reserve",
        "Mack Chain Ladder": "Mack Reserve",
    }
    reserve_by_model = {
        name: float(comparison[column].sum())
        for name, column in reserve_columns.items()
        if column in comparison.columns
    }
    highest = max(reserve_by_model, key=reserve_by_model.get) if reserve_by_model else None
    lowest = min(reserve_by_model, key=reserve_by_model.get) if reserve_by_model else None

    top_years = []
    if "Selected Reserve" in comparison.columns and comparison["Selected Reserve"].sum() != 0:
        total_selected = float(comparison["Selected Reserve"].sum())
        ranked = comparison.sort_values("Selected Reserve", ascending=False).head(5)
        for _, row in ranked.iterrows():
            top_years.append(
                {
                    "accident_year": int(row["Accident Year"]),
                    "selected_reserve": round(float(row["Selected Reserve"]), 2),
                    "reserve_share": round(float(row["Selected Reserve"]) / total_selected, 4),
                    "latest_cumulative": round(float(row.get("Latest Cumulative", 0.0)), 2),
                    "selected_ultimate": round(float(row.get("Selected Ultimate", 0.0)), 2),
                    "mack_reserve": round(float(row.get("Mack Reserve", 0.0)), 2),
                }
            )

    mack_diag = outputs.mack_diagnostics or {}
    expected_lr_range = _sensitivity_range(outputs.expected_lr_sensitivity, "ELR Reserve")
    factor_range = _sensitivity_range(outputs.factor_sensitivity, "Chain Ladder Reserve")

    payload.update(
        {
            "task": "non-life unpaid claims reserving explanation",
            "available_model_summaries": [
                {
                    "model_name": name,
                    "total_reserve": round(value, 2),
                    "formatted_total_reserve": _money(value),
                }
                for name, value in reserve_by_model.items()
            ],
            "model_differences": {
                "reserve_by_model": {name: round(value, 2) for name, value in reserve_by_model.items()},
                "highest_reserve_method": highest,
                "lowest_reserve_method": lowest,
                "reserve_range": round(reserve_by_model[highest] - reserve_by_model[lowest], 2)
                if highest and lowest
                else None,
            },
            "top_reserve_years": top_years,
            "mack_summary": {
                "total_reserve": round(float(mack_diag.get("total_mack_reserve", 0.0)), 2),
                "standard_error": round(float(mack_diag.get("total_mack_standard_error", 0.0)), 2),
                "coefficient_of_variation": round(float(mack_diag.get("total_mack_cv", 0.0)), 4),
                "reserve_75_interval": [
                    round(float(mack_diag.get("mack_75_lower", 0.0)), 2),
                    round(float(mack_diag.get("mack_75_upper", 0.0)), 2),
                ],
                "reserve_95_interval": [
                    round(float(mack_diag.get("mack_95_lower", 0.0)), 2),
                    round(float(mack_diag.get("mack_95_upper", 0.0)), 2),
                ],
            },
            "sensitivity_summary": {
                "expected_lr_reserve_range": expected_lr_range,
                "factor_shock_reserve_range": factor_range,
            },
            "capabilities": {
                "has_chain_ladder": True,
                "has_expected_loss_ratio": True,
                "has_bornhuetter_ferguson": True,
                "has_mack_result": bool(outputs.mack is not None and not outputs.mack.empty),
                "has_bootstrap_result": False,
                "has_uncertainty_metrics": bool(outputs.mack_diagnostics),
                "has_sensitivity_analysis": bool(
                    outputs.expected_lr_sensitivity is not None
                    and not outputs.expected_lr_sensitivity.empty
                    and outputs.factor_sensitivity is not None
                    and not outputs.factor_sensitivity.empty
                ),
            },
            "short_summary": {
                "data_scope": f"事故年范围 {min(report.accident_years)}-{max(report.accident_years)}，记录数 {report.row_count}。",
                "model_results": f"展示准备金约 {_money(outputs.diagnostics['total_selected_reserve'])}，展示最终赔款约 "
                f"{_money(outputs.diagnostics['total_selected_ultimate'])}。",
                "mack_uncertainty": (
                    f"Mack 准备金约 {_money(mack_diag.get('total_mack_reserve'))}，95% 区间约 "
                    f"{_money(mack_diag.get('mack_95_lower'))}-{_money(mack_diag.get('mack_95_upper'))}。"
                    if mack_diag
                    else ""
                ),
                "largest_reserve_year": (
                    f"准备金贡献最高的事故年为 {top_years[0]['accident_year']}，占比约 {top_years[0]['reserve_share']:.1%}。"
                    if top_years
                    else ""
                ),
            },
            "glossary": {
                "Chain Ladder": "基于历史累计赔款发展模式外推最终赔款。",
                "Expected Loss Ratio": "使用先验赔付率或暴露量估计最终赔款。",
                "Bornhuetter-Ferguson": "结合先验最终赔款和已报告比例，缓和早期事故年的波动。",
                "Mack Chain Ladder": "在链梯法基础上估计准备金标准误、CV 和置信区间。",
                "Selected Reserve": "当前系统展示口径，默认取 Chain Ladder 与 BF 的平均值。",
            },
        }
    )
    return payload


def _sensitivity_range(frame: pd.DataFrame | None, column: str) -> dict[str, float] | None:
    if frame is None or frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return None
    return {"min": round(float(values.min()), 2), "max": round(float(values.max()), 2)}
