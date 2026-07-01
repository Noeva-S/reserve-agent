from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

import pandas as pd

from reserve_agent.agent.explanation import build_llm_payload
from reserve_agent.data.loader import DataQualityReport
from reserve_agent.models.reserving import ReservingOutputs


MAX_ACCIDENT_YEAR_RECORDS = 80
MAX_SENSITIVITY_ROWS = 30


def _is_missing(value: Any) -> bool:
    """Return True if value is missing or empty."""
    if value is None:
        return True

    try:
        if pd.isna(value):
            return True
    except Exception:
        pass

    if isinstance(value, str) and not value.strip():
        return True

    return False


def _safe_float(value: Any) -> float | None:
    """Convert a value to float when possible."""
    try:
        if _is_missing(value):
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value: Any) -> int | None:
    """Convert a value to int when possible."""
    try:
        if _is_missing(value):
            return None
        return int(value)
    except Exception:
        return None


def _round_float(value: Any, digits: int = 2) -> float | None:
    """Round numeric values safely."""
    number = _safe_float(value)
    if number is None:
        return None
    return round(number, digits)


def _money(value: Any) -> str:
    """Format numeric values compactly for prompt summaries."""
    number = _safe_float(value)

    if number is None:
        return "-"

    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:,.2f}m"

    if abs(number) >= 1_000:
        return f"{number / 1_000:,.1f}k"

    return f"{number:,.0f}"


def _safe_scalar(value: Any) -> Any:
    """Convert scalar values to JSON-friendly Python objects."""
    if _is_missing(value):
        return None

    try:
        if hasattr(value, "item"):
            value = value.item()
    except Exception:
        pass

    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)

    if isinstance(value, float):
        return round(value, 4)

    if isinstance(value, (int, bool, str)):
        return value

    return str(value)


def _records_from_dataframe(
    frame: pd.DataFrame | None,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    """Convert a DataFrame into compact JSON-safe records."""
    if frame is None or frame.empty:
        return []

    data = frame.copy()

    if max_rows is not None:
        data = data.head(max_rows)

    records: list[dict[str, Any]] = []

    for _, row in data.iterrows():
        record: dict[str, Any] = {}

        for col, value in row.items():
            record[str(col)] = _safe_scalar(value)

        records.append(record)

    return records


def _json_safe(value: Any) -> Any:
    """Convert common pandas/dataclass objects to JSON-safe structures."""
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]

    if isinstance(value, pd.DataFrame):
        return {
            "shape": [int(value.shape[0]), int(value.shape[1])],
            "columns": [str(col) for col in value.columns.tolist()],
            "sample_rows": _records_from_dataframe(value, max_rows=10),
        }

    if isinstance(value, pd.Series):
        return {
            "name": str(value.name),
            "items": [
                {"index": str(idx), "value": _safe_scalar(val)}
                for idx, val in value.items()
            ],
        }

    if is_dataclass(value):
        return _json_safe(asdict(value))

    return _safe_scalar(value)


def _series_to_records(series: pd.Series | None) -> list[dict[str, Any]]:
    """Convert selected factors or age-to-ultimate factors to records."""
    if series is None or series.empty:
        return []

    records: list[dict[str, Any]] = []

    for idx, value in series.items():
        records.append(
            {
                "development_age": str(idx),
                "value": _round_float(value, 6),
            }
        )

    return records


def _sensitivity_range(frame: pd.DataFrame | None, column: str) -> dict[str, float] | None:
    """Return min and max for a sensitivity result column."""
    if frame is None or frame.empty or column not in frame.columns:
        return None

    values = pd.to_numeric(frame[column], errors="coerce").dropna()

    if values.empty:
        return None

    return {
        "min": round(float(values.min()), 2),
        "max": round(float(values.max()), 2),
    }


def _sensitivity_preview(
    frame: pd.DataFrame | None,
    max_rows: int = MAX_SENSITIVITY_ROWS,
) -> dict[str, Any] | None:
    """Return a compact sensitivity table preview."""
    if frame is None or frame.empty:
        return None

    return {
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "columns": [str(col) for col in frame.columns.tolist()],
        "rows": _records_from_dataframe(frame, max_rows=max_rows),
    }


def _build_reserve_by_model(comparison: pd.DataFrame) -> dict[str, float]:
    """Aggregate total reserves by available model."""
    if comparison is None or comparison.empty:
        return {}

    reserve_columns = {
        "Chain Ladder": "Chain Ladder Reserve",
        "Expected Loss Ratio": "ELR Reserve",
        "Bornhuetter-Ferguson": "BF Reserve",
        "Mack Chain Ladder": "Mack Reserve",
    }

    reserve_by_model: dict[str, float] = {}

    for model_name, column in reserve_columns.items():
        if column not in comparison.columns:
            continue

        values = pd.to_numeric(comparison[column], errors="coerce").dropna()

        if values.empty:
            continue

        reserve_by_model[model_name] = float(values.sum())

    return reserve_by_model


def _build_top_reserve_years(comparison: pd.DataFrame) -> list[dict[str, Any]]:
    """Identify accident years with the largest model-indicated reserves."""
    if comparison is None or comparison.empty or "Accident Year" not in comparison.columns:
        return []

    reserve_columns = [
        column
        for column in ["Chain Ladder Reserve", "ELR Reserve", "BF Reserve", "Mack Reserve"]
        if column in comparison.columns
    ]
    if not reserve_columns:
        return []

    reserve_frame = comparison[reserve_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    max_model_reserve = reserve_frame.max(axis=1)
    max_model_name = reserve_frame.idxmax(axis=1)
    total_basis = float(max_model_reserve.sum())

    if total_basis == 0:
        return []

    ranked = comparison.assign(_max_model_reserve=max_model_reserve, _max_model_name=max_model_name)
    ranked = ranked.sort_values("_max_model_reserve", ascending=False).head(5)

    top_years: list[dict[str, Any]] = []

    for _, row in ranked.iterrows():
        max_reserve = _safe_float(row.get("_max_model_reserve")) or 0.0

        top_years.append(
            {
                "accident_year": _safe_int(row.get("Accident Year")),
                "largest_model_reserve": round(max_reserve, 2),
                "largest_model": str(row.get("_max_model_name") or ""),
                "reserve_share": round(max_reserve / total_basis, 4),
                "latest_cumulative": _round_float(row.get("Latest Cumulative")),
                "chain_ladder_reserve": _round_float(row.get("Chain Ladder Reserve")),
                "elr_reserve": _round_float(row.get("ELR Reserve")),
                "bf_reserve": _round_float(row.get("BF Reserve")),
                "mack_reserve": _round_float(row.get("Mack Reserve")),
                "mack_standard_error": _round_float(row.get("Mack Standard Error")),
                "mack_cv": _round_float(row.get("Mack CV"), 4),
                "mack_95_lower": _round_float(row.get("Mack 95% Lower")),
                "mack_95_upper": _round_float(row.get("Mack 95% Upper")),
            }
        )

    return top_years


def _build_model_differences(reserve_by_model: dict[str, float]) -> dict[str, Any]:
    """Build a compact model comparison summary."""
    if not reserve_by_model:
        return {
            "reserve_by_model": {},
            "highest_reserve_method": None,
            "lowest_reserve_method": None,
            "reserve_range": None,
        }

    highest = max(reserve_by_model, key=reserve_by_model.get)
    lowest = min(reserve_by_model, key=reserve_by_model.get)

    return {
        "reserve_by_model": {
            name: round(value, 2)
            for name, value in reserve_by_model.items()
        },
        "formatted_reserve_by_model": {
            name: _money(value)
            for name, value in reserve_by_model.items()
        },
        "highest_reserve_method": highest,
        "lowest_reserve_method": lowest,
        "reserve_range": round(reserve_by_model[highest] - reserve_by_model[lowest], 2),
        "formatted_reserve_range": _money(reserve_by_model[highest] - reserve_by_model[lowest]),
    }


def _build_mack_summary(outputs: ReservingOutputs) -> dict[str, Any]:
    """Summarize Mack Chain Ladder outputs and diagnostics."""
    mack_diag = outputs.mack_diagnostics or {}

    has_mack = bool(outputs.mack is not None and not outputs.mack.empty)

    return {
        "available": has_mack,
        "total_reserve": _round_float(mack_diag.get("total_mack_reserve")),
        "formatted_total_reserve": _money(mack_diag.get("total_mack_reserve")),
        "standard_error": _round_float(mack_diag.get("total_mack_standard_error")),
        "formatted_standard_error": _money(mack_diag.get("total_mack_standard_error")),
        "coefficient_of_variation": _round_float(mack_diag.get("total_mack_cv"), 4),
        "reserve_75_interval": [
            _round_float(mack_diag.get("mack_75_lower")),
            _round_float(mack_diag.get("mack_75_upper")),
        ],
        "formatted_reserve_75_interval": [
            _money(mack_diag.get("mack_75_lower")),
            _money(mack_diag.get("mack_75_upper")),
        ],
        "reserve_95_interval": [
            _round_float(mack_diag.get("mack_95_lower")),
            _round_float(mack_diag.get("mack_95_upper")),
        ],
        "formatted_reserve_95_interval": [
            _money(mack_diag.get("mack_95_lower")),
            _money(mack_diag.get("mack_95_upper")),
        ],
        "by_accident_year": _records_from_dataframe(outputs.mack, max_rows=MAX_ACCIDENT_YEAR_RECORDS),
        "interpretation_note": (
            "Mack Chain Ladder provides an analytical uncertainty estimate based on chain ladder assumptions. "
            "It reports reserve standard error, coefficient of variation, and approximate reserve intervals."
            if has_mack
            else "Mack Chain Ladder results are not available in the current run."
        ),
    }


def _build_sensitivity_summary(outputs: ReservingOutputs) -> dict[str, Any]:
    """Summarize expected loss ratio and development factor sensitivities."""
    expected_lr_preview = _sensitivity_preview(outputs.expected_lr_sensitivity)
    factor_preview = _sensitivity_preview(outputs.factor_sensitivity)

    return {
        "has_expected_lr_sensitivity": expected_lr_preview is not None,
        "has_factor_sensitivity": factor_preview is not None,
        "expected_lr_reserve_range": _sensitivity_range(outputs.expected_lr_sensitivity, "ELR Reserve"),
        "expected_lr_bf_reserve_range": _sensitivity_range(outputs.expected_lr_sensitivity, "BF Reserve"),
        "factor_shock_reserve_range": _sensitivity_range(outputs.factor_sensitivity, "Chain Ladder Reserve"),
        "expected_lr_sensitivity_table": expected_lr_preview,
        "factor_sensitivity_table": factor_preview,
        "interpretation_note": (
            "Expected LR sensitivity shows how ELR/BF reserve changes when the selected expected loss ratio changes. "
            "Development factor sensitivity shows how Chain Ladder reserve changes when selected development factors are shocked."
        ),
    }


def _build_validation_summary(validation_issues: list[Any] | None) -> dict[str, Any]:
    """Summarize triangle validation issues for Agent Q&A."""
    if not validation_issues:
        return {
            "available": False,
            "issue_count": 0,
            "error_count": 0,
            "warning_count": 0,
            "info_count": 0,
            "issues": [],
        }

    issues: list[dict[str, Any]] = []

    for issue in validation_issues:
        if isinstance(issue, dict):
            level = issue.get("level")
            message = issue.get("message")
        else:
            level = getattr(issue, "level", None)
            message = getattr(issue, "message", None)

        issues.append(
            {
                "level": str(level) if level is not None else "unknown",
                "message": str(message) if message is not None else str(issue),
            }
        )

    return {
        "available": True,
        "issue_count": len(issues),
        "error_count": sum(1 for item in issues if item["level"] == "error"),
        "warning_count": sum(1 for item in issues if item["level"] == "warning"),
        "info_count": sum(1 for item in issues if item["level"] == "info"),
        "issues": issues,
    }


def _build_data_quality_summary(report: DataQualityReport) -> dict[str, Any]:
    """Summarize basic data quality information."""
    accident_years = list(getattr(report, "accident_years", []) or [])
    development_ages = list(getattr(report, "development_ages", []) or [])

    if accident_years:
        accident_year_range = [min(accident_years), max(accident_years)]
    else:
        accident_year_range = None

    return {
        "row_count": getattr(report, "row_count", None),
        "accident_years": accident_years,
        "accident_year_range": accident_year_range,
        "development_ages": development_ages,
        "missing_cells": getattr(report, "missing_cells", None),
        "negative_values": getattr(report, "negative_values", None),
        "duplicate_records": getattr(report, "duplicate_records", None),
    }


def _build_short_summary(
    report: DataQualityReport,
    outputs: ReservingOutputs,
    top_years: list[dict[str, Any]],
    mack_summary: dict[str, Any],
    workbook_context: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a short natural-language summary for the LLM."""
    accident_years = list(getattr(report, "accident_years", []) or [])

    if accident_years:
        data_scope = (
            f"事故年范围 {min(accident_years)}-{max(accident_years)}，"
            f"记录数 {getattr(report, 'row_count', '-') }。"
        )
    else:
        data_scope = f"记录数 {getattr(report, 'row_count', '-')}。"

    diagnostics = outputs.diagnostics or {}

    workbook_summary_text = ""
    if workbook_context:
        summary = workbook_context.get("workbook_summary", {})
        workbook_summary_text = (
            f"工作簿 {summary.get('workbook_name', '-') } 共 "
            f"{summary.get('sheet_count', '-') } 个 sheet；"
            f"用于建模的 sheet：{summary.get('used_sheet_names', [])}。"
        )

    return {
        "data_scope": data_scope,
        "model_results": (
            f"CL 准备金约 {_money(diagnostics.get('total_cl_reserve'))}，"
            f"BF 准备金约 {_money(diagnostics.get('total_bf_reserve'))}，"
            f"ELR 准备金约 {_money(diagnostics.get('total_elr_reserve'))}。"
        ),
        "mack_uncertainty": (
            f"Mack 准备金约 {mack_summary.get('formatted_total_reserve')}，"
            f"95% 区间约 "
            f"{mack_summary.get('formatted_reserve_95_interval', ['-', '-'])[0]}-"
            f"{mack_summary.get('formatted_reserve_95_interval', ['-', '-'])[1]}。"
            if mack_summary.get("available")
            else ""
        ),
        "largest_reserve_year": (
            f"模型显示准备金较高的事故年为 {top_years[0]['accident_year']}，"
            f"对应口径为 {top_years[0].get('largest_model', '-')}。"
            if top_years
            else ""
        ),
        "workbook": workbook_summary_text,
    }


def build_chat_context(
    report: DataQualityReport,
    outputs: ReservingOutputs,
    workbook_context: dict[str, Any] | None = None,
    validation_issues: list[Any] | None = None,
) -> dict[str, Any]:
    """
    Build compact, JSON-safe context for follow-up Agent chat.

    This context now covers two areas:
    1. Reserving model results, including CL, ELR, BF, Mack, and sensitivities.
    2. Workbook-level information, if workbook_context is provided.

    The function remains backward compatible with the old call:
        build_chat_context(report, outputs)
    """
    payload = build_llm_payload(report, outputs)

    comparison = outputs.comparison.copy() if outputs.comparison is not None else pd.DataFrame()

    reserve_by_model = _build_reserve_by_model(comparison)
    model_differences = _build_model_differences(reserve_by_model)
    top_years = _build_top_reserve_years(comparison)
    mack_summary = _build_mack_summary(outputs)
    sensitivity_summary = _build_sensitivity_summary(outputs)
    validation_summary = _build_validation_summary(validation_issues)
    data_quality_summary = _build_data_quality_summary(report)

    comparison_records = _records_from_dataframe(
        comparison,
        max_rows=MAX_ACCIDENT_YEAR_RECORDS,
    )

    model_context = {
        "data_quality": data_quality_summary,
        "validation_summary": validation_summary,
        "selected_factors": _series_to_records(outputs.selected_factors),
        "age_to_ultimate_factors": _series_to_records(outputs.age_to_ultimate),
        "comparison_by_accident_year": comparison_records,
        "available_model_summaries": [
            {
                "model_name": name,
                "total_reserve": round(value, 2),
                "formatted_total_reserve": _money(value),
            }
            for name, value in reserve_by_model.items()
        ],
        "model_differences": model_differences,
        "top_reserve_years": top_years,
        "mack_summary": mack_summary,
        "sensitivity_summary": sensitivity_summary,
    }

    capabilities = {
        "has_chain_ladder": True,
        "has_expected_loss_ratio": True,
        "has_bornhuetter_ferguson": True,
        "has_selected_result": False,
        "has_mack_result": bool(outputs.mack is not None and not outputs.mack.empty),
        "has_bootstrap_result": False,
        "has_uncertainty_metrics": bool(outputs.mack_diagnostics),
        "has_expected_lr_sensitivity": bool(
            outputs.expected_lr_sensitivity is not None
            and not outputs.expected_lr_sensitivity.empty
        ),
        "has_factor_sensitivity": bool(
            outputs.factor_sensitivity is not None
            and not outputs.factor_sensitivity.empty
        ),
        "has_sensitivity_analysis": bool(
            outputs.expected_lr_sensitivity is not None
            and not outputs.expected_lr_sensitivity.empty
            and outputs.factor_sensitivity is not None
            and not outputs.factor_sensitivity.empty
        ),
        "has_validation_issues": bool(validation_summary.get("available")),
        "has_workbook_context": bool(workbook_context),
        "can_answer_workbook_questions": bool(workbook_context),
        "can_answer_specific_cell_questions": False,
    }

    payload.update(
        {
            "task": "non-life unpaid claims reserving and workbook explanation",
            "design_note": (
                "This context is built for a follow-up chat agent. It contains compact summaries, "
                "not the full raw Excel workbook."
            ),
            "model_context": model_context,
            "workbook_context": workbook_context or {},
            "data_quality": data_quality_summary,
            "validation_summary": validation_summary,
            "selected_factors": model_context["selected_factors"],
            "age_to_ultimate_factors": model_context["age_to_ultimate_factors"],
            "comparison_by_accident_year": comparison_records,
            "available_model_summaries": model_context["available_model_summaries"],
            "model_differences": model_differences,
            "top_reserve_years": top_years,
            "mack_summary": mack_summary,
            "sensitivity_summary": sensitivity_summary,
            "capabilities": capabilities,
            "short_summary": _build_short_summary(
                report=report,
                outputs=outputs,
                top_years=top_years,
                mack_summary=mack_summary,
                workbook_context=workbook_context,
            ),
            "glossary": {
                "Chain Ladder": "基于历史累计赔款发展模式外推最终赔款。",
                "Expected Loss Ratio": "使用先验赔付率或暴露量估计最终赔款。",
                "Bornhuetter-Ferguson": "结合先验最终赔款和已报告比例，缓和早期事故年的波动。",
                "Mack Chain Ladder": "在链梯法基础上估计准备金标准误、CV 和近似区间。",
                "Expected LR sensitivity": "观察期望赔付率参数变化对 ELR 和 BF 准备金的影响。",
                "Development factor sensitivity": "观察发展因子上调或下调时 Chain Ladder 准备金的变化。",
                "Workbook context": "对上传 Excel 中各 sheet 的结构化摘要，包括 sheet 名称、行列规模、有效区域、样例行、文本预览和可能角色。",
                "Used sheet": "系统实际用于准备金建模的数据 sheet。",
                "Unused sheet": "上传工作簿中未直接进入准备金模型计算、但可能包含说明、参数、保费、假设或参考信息的 sheet。",
            },
            "answer_rules": [
                "回答必须基于当前 context，不要编造没有出现在 context 中的 Excel 内容或模型结果。",
                "用户问模型结果、Mack、标准误、区间、敏感性分析时，优先使用 model_context。",
                "用户问 Excel、workbook、sheet、字段、备注、假设、参数时，优先使用 workbook_context。",
                "workbook_context 只是摘要，不是完整原始工作簿；如果用户问具体单元格而 context 中没有该单元格，应说明无法从当前摘要判断。",
                "如果 capabilities.has_mack_result 为 False，不能声称已经运行 Mack。",
                "如果 capabilities.has_bootstrap_result 为 False，不能声称已经运行 Bootstrap 或给出 Bootstrap 置信区间。",
                "如果用户问可靠性、风险、局限或是否可直接采用，应提醒这是课程项目展示结果，不等同于正式精算签字意见。",
                "普通定义类或简单数值类问题可以直接回答，不要每次都机械追加很长的审慎提示。",
            ],
        }
    )

    return _json_safe(payload)
