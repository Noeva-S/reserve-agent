from __future__ import annotations

from typing import Any

from reserve_agent.agent.llm_client import call_deepseek
from reserve_agent.agent.prompts import build_chat_messages


def answer_user_question(
    question: str,
    context: dict[str, Any],
    api_key: str,
    chat_history: list[dict[str, str]] | None = None,
) -> str:
    """Answer a follow-up question about the current reserving result."""

    messages = build_chat_messages(question, context, chat_history)
    return call_deepseek(messages, api_key=api_key)
