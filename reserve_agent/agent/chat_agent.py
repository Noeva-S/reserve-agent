from __future__ import annotations

from typing import Any

from reserve_agent.agent.llm_client import call_deepseek
from reserve_agent.agent.prompts import build_chat_messages


def _format_money(value: Any) -> str:
    try:
        number = float(value)
    except Exception:
        return "-"
    if abs(number) >= 1_000_000:
        return f"{number / 1_000_000:,.2f}m"
    if abs(number) >= 1_000:
        return f"{number / 1_000:,.1f}k"
    return f"{number:,.2f}"


def _fallback_answer(question: str, context: dict[str, Any]) -> str:
    q = question.lower()
    top_years = context.get("top_reserve_years", [])
    differences = context.get("model_differences", {})
    data_quality = context.get("data_quality", {})
    factors = context.get("selected_factors", {})
    summary = context.get("short_summary", {})
    glossary = context.get("glossary", {})

    if any(key in q for key in ["最高", "最大", "top", "哪一年", "哪个事故年"]):
        if top_years:
            top = top_years[0]
            return (
                f"按当前展示口径，准备金贡献最高的是 {top['accident_year']} 年，"
                f"Selected Reserve 约为 {_format_money(top['selected_reserve'])}，"
                f"占总展示准备金约 {top['reserve_share']:.1%}。这通常说明该事故年赔款尚未充分成熟，"
                "或者其最新累计赔款和发展因子共同导致未决估计较高。"
            )
        return "当前结果里没有足够的事故年准备金明细，无法判断最高贡献年份。"

    if any(key in q for key in ["区别", "差异", "比较", "cl", "chain", "bf", "elr"]):
        reserve_by_model = differences.get("reserve_by_model", {})
        if reserve_by_model:
            lines = ["当前各方法总准备金对比如下："]
            for name, value in reserve_by_model.items():
                lines.append(f"- {name}: {_format_money(value)}")
            lines.append(
                "Chain Ladder 更依赖历史发展模式，ELR 更依赖先验赔付率，BF 则把先验估计和已报告比例结合起来。"
            )
            return "\n".join(lines)
        return "当前上下文中没有可比较的模型结果。"

    if any(key in q for key in ["数据", "质量", "空值", "负数", "问题"]):
        if data_quality:
            return (
                "当前数据质量摘要："
                f"记录数 {data_quality.get('row_count')}，赔案/事故年数量 {data_quality.get('claim_count')}，"
                f"缺失值 {data_quality.get('missing_values')}，负金额单元格 {data_quality.get('negative_amount_cells')}。"
                f"系统提示：{'；'.join(data_quality.get('notes', []))}"
            )
        return "当前没有检测到数据质量上下文。"

    if any(key in q for key in ["因子", "发展因子", "factor"]):
        if factors:
            lines = ["当前选定发展因子如下："]
            for dev, value in factors.items():
                lines.append(f"- Dev {dev}: {value}")
            return "\n".join(lines)
        return "当前上下文中没有发展因子信息。"

    if any(key in q for key in ["chain ladder", "链梯", "bf", "bornhuetter", "elr", "术语", "解释"]):
        return "\n".join([f"- {name}: {text}" for name, text in glossary.items()])

    parts = [str(value) for value in summary.values() if value]
    if parts:
        return "根据当前模型结果，可以概括为：\n\n" + "\n\n".join(parts)
    return "当前上下文信息有限，建议先完成数据识别和模型计算后再提问。"


def answer_user_question(
    question: str,
    context: dict[str, Any],
    api_key: str | None = None,
    chat_history: list[dict[str, str]] | None = None,
) -> str:
    """Answer a follow-up question, using DeepSeek when available."""

    if not api_key:
        return _fallback_answer(question, context)
    try:
        messages = build_chat_messages(question, context, chat_history)
        return call_deepseek(messages, api_key=api_key)
    except Exception as exc:
        return _fallback_answer(question, context) + f"\n\n（DeepSeek 调用失败，已使用规则型回答。错误：{exc}）"
