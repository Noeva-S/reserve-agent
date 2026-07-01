from __future__ import annotations

import re
from typing import Any

from reserve_agent.agent.llm_client import call_deepseek
from reserve_agent.agent.prompts import build_chat_messages
from reserve_agent.agent.sheet_retriever import build_sheet_detail_context


def answer_user_question(
    question: str,
    context: dict[str, Any],
    api_key: str | None = None,
    chat_history: list[dict[str, str]] | None = None,
    workbook_file_path: str | None = None,
) -> str:
    """
    Answer a user's follow-up question.

    Priority:
    1. If API key is provided, use DeepSeek with structured context.
    2. If API is unavailable or fails, return a rule-based fallback answer.

    The fallback is intentionally limited. It covers common workbook/model
    questions but does not try to replace the LLM.
    """
    question = (question or "").strip()

    if not question:
        return "请先输入一个问题。"

    if not context:
        return "当前没有可用的模型或工作簿上下文，请先上传数据并运行准备金模型。"

    context_for_answer = _maybe_add_sheet_detail_context(
        question=question,
        context=context,
        workbook_file_path=workbook_file_path,
    )

    if api_key:
        try:
            messages = build_chat_messages(
                question=question,
                context=context_for_answer,
                chat_history=chat_history,
            )
            response = call_deepseek(messages, api_key=api_key)

            if response and str(response).strip():
                return str(response).strip()

        except Exception as exc:
            fallback = _rule_based_answer(question, context_for_answer)
            return (
                "DeepSeek 调用失败，以下为基于当前结构化上下文生成的规则型回答。\n\n"
                f"调用错误：{exc}\n\n"
                f"{fallback}"
            )

    return _rule_based_answer(question, context_for_answer)


def _maybe_add_sheet_detail_context(
    question: str,
    context: dict[str, Any],
    workbook_file_path: str | None = None,
) -> dict[str, Any]:
    """
    Add on-demand sheet detail context when the user asks about workbook/sheet content.

    This keeps the normal prompt compact, but allows deeper answers for questions
    about Projection, Comparison, Exposure, Policy, Claims, or other sheets.
    """
    if not workbook_file_path:
        print("[sheet_detail_context] skipped: no workbook_file_path")
        return context

    q = question.lower()

    trigger_words = [
        "sheet",
        "worksheet",
        "excel",
        "workbook",
        "projection",
        "method",
        "comparison",
        "policy",
        "exposure",
        "claims",
        "claim",
        "列",
        "行",
        "字段",
        "表",
        "工作簿",
        "数据表",
        "说明",
        "内容",
        "结果",
        "比较",
        "方法",
        "预测",
        "暴露",
        "保单",
        "赔案",
        "赔款",
    ]

    if not any(word in q or word in question for word in trigger_words):
        return context

    try:
        sheet_detail_context = build_sheet_detail_context(
            file_path=workbook_file_path,
            question=question,
            workbook_context=context.get("workbook_context", {}),
        )
    except Exception as exc:
        print("[sheet_detail_context] error:", exc)
        enriched = dict(context)
        enriched["sheet_detail_context_error"] = str(exc)
        return enriched

    if not sheet_detail_context.get("triggered"):
        print("[sheet_detail_context] not triggered:", sheet_detail_context.get("reason"))
        return context

    print(
        "[sheet_detail_context] triggered:",
        sheet_detail_context.get("relevant_sheet_names"),
    )

    enriched = dict(context)
    enriched["sheet_detail_context"] = sheet_detail_context

    return enriched


def _rule_based_answer(question: str, context: dict[str, Any]) -> str:
    """Route common questions to rule-based fallback answers."""
    q = question.lower()

    if _looks_like_specific_cell_question(question):
        return _fallback_specific_cell_answer(question, context)

    if _mentions_missing_or_unavailable_model(q):
        return _fallback_unavailable_model_answer(question, context)

    if _looks_like_sheet_detail_question(question, context):
        return _fallback_workbook_answer(question, context)

    if any(word in q for word in ["sheet", "worksheet", "excel", "workbook", "工作簿", "数据表"]):
        return _fallback_workbook_answer(question, context)

    if any(word in q for word in ["mack", "标准误", "cv", "置信", "区间", "不确定"]):
        return _fallback_mack_answer(question, context)

    if any(word in q for word in ["敏感性", "sensitivity", "expected lr", "期望赔付率", "发展因子", "factor shock"]):
        return _fallback_sensitivity_answer(context)

    if any(word in q for word in ["校验", "validation", "warning", "error", "错误", "警告"]):
        return _fallback_validation_answer(context)

    if any(word in q for word in ["数据质量", "质量", "缺失", "负值", "重复"]):
        return _fallback_data_quality_answer(context)

    if any(word in q for word in ["最高", "最大", "贡献", "top", "哪一年", "事故年"]):
        return _fallback_top_reserve_year_answer(context)

    if any(word in q for word in ["比较", "差异", "cl", "chain ladder", "bf", "elr", "selected", "模型"]):
        return _fallback_model_comparison_answer(context)

    if any(word in q for word in ["是什么", "什么意思", "定义", "解释"]):
        glossary_answer = _fallback_glossary_answer(question, context)
        if glossary_answer:
            return glossary_answer

    return _fallback_summary_answer(context)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _format_money(value: Any) -> str:
    number = _safe_float(value)

    if number is None:
        return "-"

    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:,.2f}m"

    if abs(number) >= 1_000:
        return f"{number / 1_000:,.1f}k"

    return f"{number:,.0f}"


def _format_percent(value: Any) -> str:
    number = _safe_float(value)

    if number is None:
        return "-"

    return f"{number:.1%}"


def _normalize_text(text: Any) -> str:
    return str(text or "").lower().replace(" ", "").replace("_", "").replace("-", "")


def _get_workbook_context(context: dict[str, Any]) -> dict[str, Any]:
    workbook_context = context.get("workbook_context", {})
    return workbook_context if isinstance(workbook_context, dict) else {}


def _get_model_context(context: dict[str, Any]) -> dict[str, Any]:
    model_context = context.get("model_context", {})
    return model_context if isinstance(model_context, dict) else {}


def _get_capabilities(context: dict[str, Any]) -> dict[str, Any]:
    capabilities = context.get("capabilities", {})
    return capabilities if isinstance(capabilities, dict) else {}


def _get_sheets(context: dict[str, Any]) -> list[dict[str, Any]]:
    workbook_context = _get_workbook_context(context)
    sheets = workbook_context.get("sheets", [])

    if not isinstance(sheets, list):
        return []

    return [sheet for sheet in sheets if isinstance(sheet, dict)]


def _find_relevant_sheet(question: str, context: dict[str, Any]) -> dict[str, Any] | None:
    """Find the sheet most likely referenced by the user."""
    q_norm = _normalize_text(question)
    q_lower = question.lower()

    sheets = _get_sheets(context)

    for sheet in sheets:
        sheet_name = str(sheet.get("sheet_name", ""))
        sheet_norm = _normalize_text(sheet_name)

        if not sheet_norm:
            continue

        if sheet_norm in q_norm or q_norm in sheet_norm:
            return sheet

        sheet_without_prefix = re.sub(r"^\s*\d+\s*[\.\-_:：]\s*", "", sheet_name)
        sheet_without_prefix_norm = _normalize_text(sheet_without_prefix)

        if sheet_without_prefix_norm and sheet_without_prefix_norm in q_norm:
            return sheet

        tokens = [
            token
            for token in sheet_without_prefix.lower().replace("_", " ").replace("-", " ").split()
            if token and token not in {"the", "and", "of", "in", "to", "a", "an"}
        ]

        if tokens and all(token in q_lower for token in tokens):
            return sheet

    return None


def _looks_like_specific_cell_question(question: str) -> bool:
    q = question.lower()

    if "单元格" in q:
        return True

    return bool(re.search(r"\b[a-zA-Z]{1,3}\s*\d{1,7}\b", question))


def _extract_cell_label(question: str) -> str | None:
    match = re.search(r"\b([a-zA-Z]{1,3})\s*(\d{1,7})\b", question)

    if not match:
        return None

    return f"{match.group(1).upper()}{match.group(2)}"


def _looks_like_sheet_detail_question(question: str, context: dict[str, Any]) -> bool:
    q = question.lower()

    if any(word in q for word in ["行", "列", "字段", "sheet", "介绍", "说明", "内容"]):
        if _find_relevant_sheet(question, context):
            return True

    return False


def _mentions_missing_or_unavailable_model(q: str) -> bool:
    return any(word in q for word in ["bootstrap", "cape cod", "capecod", "随机模拟"])


def _fallback_specific_cell_answer(question: str, context: dict[str, Any]) -> str:
    cell_label = _extract_cell_label(question)
    sheets = _get_sheets(context)

    if cell_label:
        for sheet in sheets:
            for item in sheet.get("text_preview", []) or []:
                if not isinstance(item, dict):
                    continue

                if str(item.get("cell", "")).upper() == cell_label:
                    return (
                        f"当前摘要中的文本预览包含 {cell_label}："
                        f"{item.get('text')}。\n\n"
                        "但需要注意，workbook_context 只保存摘要和部分预览，"
                        "不是完整原始 Excel。"
                    )

    return (
        "当前 workbook_context 只包含 sheet 摘要、样例行、文本预览和数值概览，"
        "不包含完整原始工作簿的每一个单元格。\n\n"
        "因此，无法可靠回答该具体单元格的内容。"
    )


def _fallback_unavailable_model_answer(question: str, context: dict[str, Any]) -> str:
    capabilities = _get_capabilities(context)
    q = question.lower()

    if "bootstrap" in q:
        if not capabilities.get("has_bootstrap_result"):
            return (
                "当前上下文显示没有 Bootstrap 模型结果。"
                "因此不能提供 Bootstrap 置信区间、模拟分布或对应的随机准备金结果。"
            )

    if "cape" in q or "capecod" in q:
        return (
            "当前上下文没有 Cape Cod 模型结果。"
            "因此不能声称系统已经运行 Cape Cod，也不能给出 Cape Cod 准备金。"
        )

    return "当前上下文没有该模型或方法的结果，不能据此给出对应结论。"


def _fallback_workbook_answer(question: str, context: dict[str, Any]) -> str:
    sheet_detail_context = context.get("sheet_detail_context")

    if isinstance(sheet_detail_context, dict) and sheet_detail_context.get("triggered"):
        return _fallback_sheet_detail_context_answer(sheet_detail_context)

    workbook_context = _get_workbook_context(context)

    if not workbook_context:
        return "当前没有 workbook_context，因此无法回答整个 Excel 或 sheet 相关问题。"

    sheet = _find_relevant_sheet(question, context)

    if sheet:
        return _format_sheet_summary(sheet)

    workbook_summary = workbook_context.get("workbook_summary", {})
    sheet_names = workbook_context.get("sheet_names", [])
    used_sheet_names = workbook_context.get("used_sheet_names", [])
    unused_sheet_names = workbook_context.get("unused_sheet_names", [])

    lines = [
        f"当前工作簿：{workbook_summary.get('workbook_name', workbook_context.get('workbook_name', '-'))}",
        f"sheet 数量：{workbook_context.get('sheet_count', len(sheet_names))}",
        "",
        "sheet 列表：",
    ]

    for sheet_item in _get_sheets(context):
        name = sheet_item.get("sheet_name", "-")
        used = "是" if sheet_item.get("used_for_modeling") else "否"
        role = sheet_item.get("likely_role", "-")
        rows = sheet_item.get("row_count", "-")
        cols = sheet_item.get("column_count", "-")
        lines.append(f"- {name}：用于建模={used}，可能角色={role}，有效区域约 {rows} 行 × {cols} 列")

    lines.extend(
        [
            "",
            f"用于建模的 sheet：{used_sheet_names or '-'}",
            f"未用于建模的 sheet：{unused_sheet_names or '-'}",
            "",
            "注意：这里基于 workbook_context 摘要回答，不等同于完整逐单元格读取原始 Excel。",
        ]
    )

    return "\n".join(lines)


def _fallback_sheet_detail_context_answer(sheet_detail_context: dict[str, Any]) -> str:
    sheets = sheet_detail_context.get("sheets", [])

    if not isinstance(sheets, list) or not sheets:
        return "系统已尝试按需读取相关 sheet，但没有取得可展示的详细内容。"

    lines = [
        "系统已按需读取相关 sheet 的详细内容。摘要如下：",
    ]

    for sheet in sheets:
        if not isinstance(sheet, dict):
            continue

        sheet_name = sheet.get("sheet_name", "-")

        if sheet.get("read_error"):
            lines.extend(
                [
                    "",
                    f"## {sheet_name}",
                    f"读取失败：{sheet.get('read_error')}",
                ]
            )
            continue

        used_range = sheet.get("used_range") or {}
        used_shape = sheet.get("used_shape", ["-", "-"])
        text_cells = sheet.get("text_cells", []) or []
        keyword_regions = sheet.get("keyword_regions", []) or []

        lines.extend(
            [
                "",
                f"## {sheet_name}",
                f"- 有效区域：{used_range.get('top_left_cell', '-')} 到 {used_range.get('bottom_right_cell', '-')}",
                f"- 有效规模：{used_shape[0]} 行 × {used_shape[1]} 列",
            ]
        )

        if text_cells:
            lines.append("")
            lines.append("主要文本单元格预览：")
            for item in text_cells[:12]:
                if isinstance(item, dict):
                    lines.append(f"- {item.get('cell', '-')}: {item.get('text', '-')}")

        if keyword_regions:
            lines.append("")
            lines.append("与问题关键词相关的局部区域：")

            for region in keyword_regions[:3]:
                if not isinstance(region, dict):
                    continue

                region_range = region.get("range") or {}
                lines.append(
                    f"- 匹配单元格 {region.get('matched_cell', '-')}，"
                    f"文本为“{region.get('matched_text', '-')}”，"
                    f"附近区域 {region_range.get('top_left_cell', '-')} 到 "
                    f"{region_range.get('bottom_right_cell', '-')}"
                )

                rows = region.get("rows", []) or []
                for row in rows[:5]:
                    if isinstance(row, dict):
                        lines.append(f"  - {row}")

        lines.append("")
        lines.append(
            "注意：这是按需读取的局部详细内容，仍不是完整逐单元格展示。"
            "如果问题涉及未截取区域或公式，需要进一步精确读取。"
        )

    return "\n".join(lines)


def _format_sheet_summary(sheet: dict[str, Any]) -> str:
    sheet_name = sheet.get("sheet_name", "-")
    used = "是" if sheet.get("used_for_modeling") else "否"
    role = sheet.get("likely_role", "-")
    row_count = sheet.get("row_count", "-")
    column_count = sheet.get("column_count", "-")
    used_range = sheet.get("used_range") or {}
    header_row = sheet.get("guessed_header_row", "-")
    columns = sheet.get("columns", []) or []
    sample_rows = sheet.get("sample_rows", []) or []
    numeric_summary = sheet.get("numeric_summary", {}) or {}

    lines = [
        f"{sheet_name} 这个 sheet 的摘要如下：",
        "",
        f"- 是否用于建模：{used}",
        f"- 可能角色：{role}",
        f"- 有效行数：{row_count}",
        f"- 有效列数：{column_count}",
        f"- 表头行：{header_row}",
    ]

    if used_range:
        lines.append(
            "- 有效区域："
            f"{used_range.get('top_left_cell', '-')}"
            f" 到 {used_range.get('bottom_right_cell', '-')}"
        )

    if columns:
        lines.append("")
        lines.append("主要列名：")
        for col in columns[:30]:
            lines.append(f"- {col}")

        if len(columns) > 30:
            lines.append(f"- ... 其余 {len(columns) - 30} 列未在规则型回答中展开")

    if sample_rows:
        lines.append("")
        lines.append("样例行：")
        first = sample_rows[0]
        if isinstance(first, dict):
            for key, value in list(first.items())[:20]:
                lines.append(f"- {key}: {value}")

    if isinstance(numeric_summary, dict) and numeric_summary:
        lines.append("")
        lines.append("部分数值列概览：")
        for col, summary in list(numeric_summary.items())[:10]:
            if not isinstance(summary, dict):
                continue
            lines.append(
                f"- {col}: count={summary.get('count')}, "
                f"min={summary.get('min')}, max={summary.get('max')}, mean={summary.get('mean')}"
            )

    lines.append("")
    lines.append(
        "注意：这是 sheet 摘要，不是完整原始表。若要回答具体单元格或完整公式，需要进一步读取原始 Excel。"
    )

    return "\n".join(lines)


def _fallback_mack_answer(question: str, context: dict[str, Any]) -> str:
    mack_summary = context.get("mack_summary", {})
    if not isinstance(mack_summary, dict):
        mack_summary = {}

    if not mack_summary.get("available"):
        return "当前上下文显示没有可用的 Mack Chain Ladder 结果。"

    q = question.lower()

    if any(word in q for word in ["哪个", "最大", "最高"]) and any(
        word in q for word in ["不确定", "cv", "标准误", "波动"]
    ):
        rows = mack_summary.get("by_accident_year", []) or []
        best_row = _find_max_mack_uncertainty_row(rows)

        if best_row:
            accident_year = best_row.get("Accident Year", best_row.get("accident_year", "-"))
            se = best_row.get("Mack Standard Error", best_row.get("mack_standard_error"))
            cv = best_row.get("Mack CV", best_row.get("mack_cv"))
            reserve = best_row.get("Mack Reserve", best_row.get("mack_reserve"))

            return (
                f"从当前 Mack 结果看，不确定性最高的事故年大致是 {accident_year}。\n\n"
                f"- Mack Reserve：{_format_money(reserve)}\n"
                f"- Mack Standard Error：{_format_money(se)}\n"
                f"- Mack CV：{_format_percent(cv) if _safe_float(cv) is not None else '-'}\n\n"
                "Mack CV 或标准误越高，说明该事故年准备金估计的不确定性越大。"
            )

    interval_95 = mack_summary.get("formatted_reserve_95_interval") or ["-", "-"]
    interval_75 = mack_summary.get("formatted_reserve_75_interval") or ["-", "-"]

    return (
        "当前有 Mack Chain Ladder 结果。\n\n"
        f"- Mack 总准备金：{mack_summary.get('formatted_total_reserve', '-')}\n"
        f"- Mack 标准误：{mack_summary.get('formatted_standard_error', '-')}\n"
        f"- Mack CV：{_format_percent(mack_summary.get('coefficient_of_variation'))}\n"
        f"- Mack 75% 参考区间：{interval_75[0]} 至 {interval_75[1]}\n"
        f"- Mack 95% 参考区间：{interval_95[0]} 至 {interval_95[1]}\n\n"
        "这里的 Mack 区间应理解为：在 Mack 模型假设和近似分布下，"
        "根据标准误构造的准备金不确定性参考区间。"
        "不应简单表述为“真实准备金有 95% 的概率落在该区间内”。"
    )


def _find_max_mack_uncertainty_row(rows: list[Any]) -> dict[str, Any] | None:
    if not isinstance(rows, list):
        return None

    best_row: dict[str, Any] | None = None
    best_score: float | None = None

    for row in rows:
        if not isinstance(row, dict):
            continue

        cv = _safe_float(row.get("Mack CV", row.get("mack_cv")))
        se = _safe_float(row.get("Mack Standard Error", row.get("mack_standard_error")))

        score = cv if cv is not None else se

        if score is None:
            continue

        if best_score is None or score > best_score:
            best_score = score
            best_row = row

    return best_row


def _fallback_sensitivity_answer(context: dict[str, Any]) -> str:
    sensitivity = context.get("sensitivity_summary", {})
    if not isinstance(sensitivity, dict):
        sensitivity = {}

    has_elr = sensitivity.get("has_expected_lr_sensitivity")
    has_factor = sensitivity.get("has_factor_sensitivity")

    if not has_elr and not has_factor:
        return "当前上下文没有可用的敏感性分析结果。"

    lines = ["当前评估包含敏感性分析："]

    if has_elr:
        lines.append("")
        lines.append("1. Expected LR 敏感性分析")
        lines.append("用于观察期望赔付率变化对 ELR 和 BF 准备金的影响。")

        elr_range = sensitivity.get("expected_lr_reserve_range")
        bf_range = sensitivity.get("expected_lr_bf_reserve_range")

        if isinstance(elr_range, dict):
            lines.append(
                f"- ELR Reserve 范围：{_format_money(elr_range.get('min'))} 至 {_format_money(elr_range.get('max'))}"
            )

        if isinstance(bf_range, dict):
            lines.append(
                f"- BF Reserve 范围：{_format_money(bf_range.get('min'))} 至 {_format_money(bf_range.get('max'))}"
            )

    if has_factor:
        lines.append("")
        lines.append("2. 发展因子敏感性分析")
        lines.append("用于观察 Chain Ladder 结果对发展因子上调或下调的敏感程度。")

        factor_range = sensitivity.get("factor_shock_reserve_range")
        if isinstance(factor_range, dict):
            lines.append(
                f"- Chain Ladder Reserve 范围：{_format_money(factor_range.get('min'))} 至 {_format_money(factor_range.get('max'))}"
            )

    return "\n".join(lines)


def _fallback_validation_answer(context: dict[str, Any]) -> str:
    validation = context.get("validation_summary", {})
    if not isinstance(validation, dict) or not validation.get("available"):
        return "当前上下文没有结构校验问题，或者没有提供 validation_summary。"

    issues = validation.get("issues", []) or []

    lines = [
        "当前结构校验结果如下：",
        f"- issue 总数：{validation.get('issue_count', 0)}",
        f"- error 数量：{validation.get('error_count', 0)}",
        f"- warning 数量：{validation.get('warning_count', 0)}",
        f"- info 数量：{validation.get('info_count', 0)}",
    ]

    if issues:
        lines.append("")
        lines.append("部分校验信息：")
        for item in issues[:8]:
            if not isinstance(item, dict):
                continue
            lines.append(f"- [{item.get('level', '-')}] {item.get('message', '-')}")

    return "\n".join(lines)


def _fallback_data_quality_answer(context: dict[str, Any]) -> str:
    data_quality = context.get("data_quality", {})
    if not isinstance(data_quality, dict):
        data_quality = {}

    return (
        "当前数据质量摘要如下：\n\n"
        f"- 记录数：{data_quality.get('row_count', '-')}\n"
        f"- 事故年范围：{data_quality.get('accident_year_range', '-')}\n"
        f"- 发展期：{data_quality.get('development_ages', '-')}\n"
        f"- 缺失单元格：{data_quality.get('missing_cells', '-')}\n"
        f"- 负值数量：{data_quality.get('negative_values', '-')}\n"
        f"- 重复记录：{data_quality.get('duplicate_records', '-')}"
    )


def _fallback_top_reserve_year_answer(context: dict[str, Any]) -> str:
    top_years = context.get("top_reserve_years", [])

    if not isinstance(top_years, list) or not top_years:
        return "当前上下文没有 top_reserve_years，因此无法判断哪个事故年准备金贡献最高。"

    top = top_years[0]

    if not isinstance(top, dict):
        return "当前 top_reserve_years 格式异常，无法生成规则型回答。"

    return (
        f"当前各模型显示准备金较高的事故年是 {top.get('accident_year', '-')}。\n\n"
        f"- 对应模型口径：{top.get('largest_model', '-')}\n"
        f"- 该口径准备金：{_format_money(top.get('largest_model_reserve'))}\n"
        f"- Latest Cumulative：{_format_money(top.get('latest_cumulative'))}\n"
        f"- Chain Ladder Reserve：{_format_money(top.get('chain_ladder_reserve'))}\n"
        f"- BF Reserve：{_format_money(top.get('bf_reserve'))}\n"
        f"- Mack Reserve：{_format_money(top.get('mack_reserve'))}\n\n"
        "这不是系统最终选择，只表示某些模型在该事故年给出的准备金较高，最终判断还要结合业务和数据质量。"
    )


def _fallback_model_comparison_answer(context: dict[str, Any]) -> str:
    model_differences = context.get("model_differences", {})
    if not isinstance(model_differences, dict):
        model_differences = {}

    reserve_by_model = model_differences.get("reserve_by_model", {})

    if not isinstance(reserve_by_model, dict) or not reserve_by_model:
        summaries = context.get("available_model_summaries", [])
        if not summaries:
            return "当前上下文没有可用的模型比较结果。"

        lines = ["当前可用模型摘要："]
        for item in summaries:
            if isinstance(item, dict):
                lines.append(
                    f"- {item.get('model_name', '-')}: {item.get('formatted_total_reserve', '-')}"
                )
        return "\n".join(lines)

    lines = ["当前各模型准备金总额如下："]

    formatted = model_differences.get("formatted_reserve_by_model", {})

    for model_name, value in reserve_by_model.items():
        display = formatted.get(model_name) if isinstance(formatted, dict) else None
        lines.append(f"- {model_name}: {display or _format_money(value)}")

    lines.extend(
        [
            "",
            f"最高准备金方法：{model_differences.get('highest_reserve_method', '-')}",
            f"最低准备金方法：{model_differences.get('lowest_reserve_method', '-')}",
            f"最高与最低差额：{model_differences.get('formatted_reserve_range', '-')}",
        ]
    )

    return "\n".join(lines)


def _fallback_glossary_answer(question: str, context: dict[str, Any]) -> str | None:
    glossary = context.get("glossary", {})

    if not isinstance(glossary, dict) or not glossary:
        return None

    q_norm = _normalize_text(question)

    for term, explanation in glossary.items():
        if _normalize_text(term) in q_norm:
            return f"{term}：{explanation}"

    return None


def _fallback_summary_answer(context: dict[str, Any]) -> str:
    short_summary = context.get("short_summary", {})

    if not isinstance(short_summary, dict) or not short_summary:
        return (
            "当前可以回答准备金模型结果和 Excel 工作簿摘要相关问题。"
            "你可以询问哪个 sheet 用于建模、某个 sheet 的行列结构、"
            "哪个事故年准备金最高、Mack 区间或敏感性分析等。"
        )

    lines = ["当前评估摘要："]

    for key, value in short_summary.items():
        if value:
            lines.append(f"- {value}")

    return "\n".join(lines)
