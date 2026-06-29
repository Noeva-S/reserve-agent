from __future__ import annotations

import json
from typing import Any


MAX_CHAT_MESSAGES = 20
MAX_ACCIDENT_YEAR_RECORDS = 80
EXCLUDED_CONTEXT_KEYS = {
    "raw_data",
    "raw_claims",
    "raw_excel",
    "source_dataframe",
    "claims_dataframe",
    "full_dataframe",
    "uploaded_file",
}


def _clean_chat_history(chat_history: list[dict[str, str]] | None) -> list[dict[str, str]]:
    if not chat_history:
        return []
    cleaned: list[dict[str, str]] = []
    for msg in chat_history:
        role = msg.get("role")
        content = msg.get("content")
        if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
            cleaned.append({"role": role, "content": content.strip()})
    return cleaned[-MAX_CHAT_MESSAGES:]


def _limit_records(value: Any, max_records: int = MAX_ACCIDENT_YEAR_RECORDS) -> Any:
    if not isinstance(value, list) or len(value) <= max_records:
        return value
    return {
        "records_included": value[:max_records],
        "records_omitted_count": len(value) - max_records,
    }


def _compact_context(context: dict[str, Any]) -> dict[str, Any]:
    priority_keys = [
        "task",
        "data_quality",
        "model_totals",
        "selected_factors",
        "comparison_by_accident_year",
        "available_model_summaries",
        "model_differences",
        "top_reserve_years",
        "capabilities",
        "short_summary",
        "glossary",
    ]
    compact: dict[str, Any] = {}
    for key in priority_keys:
        if key in context:
            compact[key] = _limit_records(context[key])
    additional = {}
    for key, value in context.items():
        if key in compact or key in EXCLUDED_CONTEXT_KEYS:
            continue
        lowered = key.lower()
        if any(marker in lowered for marker in ["raw", "source_dataframe", "full_dataframe", "claims_dataframe"]):
            continue
        additional[key] = _limit_records(value)
    if additional:
        compact["additional_context_for_future_extensions"] = additional
    return compact


def build_chat_messages(
    question: str,
    context: dict[str, Any],
    chat_history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build DeepSeek messages for reserving follow-up questions."""

    system_prompt = """
你是一个非寿险准备金评估课程项目中的实时问答 Agent。
你只能根据提供的 context 回答，不能编造外部数据、额外模型结果或不存在的计算结果。
如果 context 中没有 Mack、Bootstrap 或不确定性指标，不要声称系统已经运行这些模型。
回答使用中文，语气专业清晰；涉及金额、比例、事故年和模型差异时，优先引用 context 中的具体数值。
如果用户要求报告语言，可以给出正式、连贯的一段文字。
""".strip()
    context_text = json.dumps(_compact_context(context), ensure_ascii=False, indent=2)
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(_clean_chat_history(chat_history))
    messages.append(
        {
            "role": "user",
            "content": f"当前准备金评估 context：\n{context_text}\n\n用户问题：{question.strip()}",
        }
    )
    return messages
