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
                }
            )

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
            "capabilities": {
                "has_chain_ladder": True,
                "has_expected_loss_ratio": True,
                "has_bornhuetter_ferguson": True,
                "has_mack_result": False,
                "has_bootstrap_result": False,
                "has_uncertainty_metrics": False,
            },
            "short_summary": {
                "data_scope": f"事故年范围 {min(report.accident_years)}-{max(report.accident_years)}，记录数 {report.row_count}。",
                "model_results": f"展示准备金约 {_money(outputs.diagnostics['total_selected_reserve'])}，展示最终赔款约 {_money(outputs.diagnostics['total_selected_ultimate'])}。",
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
                "Selected Reserve": "当前系统展示口径，默认取 Chain Ladder 与 BF 的平均值。",
            },
        }
    )
    return payload
