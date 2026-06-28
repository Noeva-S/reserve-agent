from __future__ import annotations

import json
from typing import Any


def build_chat_messages(
    question: str,
    context: dict[str, Any],
    chat_history: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build DeepSeek messages for a reserving follow-up question."""

    system = (
        "你是非寿险准备金评估课程项目中的智能 Agent。"
        "请只根据提供的模型结果和数据诊断回答，不要编造外部数据。"
    )
    messages = [{"role": "system", "content": system}]
    if chat_history:
        messages.extend(chat_history[-8:])
    user = (
        "当前准备金评估上下文如下：\n"
        f"{json.dumps(context, ensure_ascii=False, indent=2)}\n\n"
        f"用户问题：{question}"
    )
    messages.append({"role": "user", "content": user})
    return messages
