"""Rule-based and LLM-assisted explanation modules."""

from .explanation import (
    build_llm_payload,
    generate_agent_explanation,
    generate_data_diagnosis,
    generate_method_notes,
    generate_result_summary,
    recommend_model,
)
from .llm_client import build_reserving_prompt, call_deepseek, get_deepseek_key

__all__ = [
    "build_llm_payload",
    "build_reserving_prompt",
    "call_deepseek",
    "generate_agent_explanation",
    "generate_data_diagnosis",
    "generate_method_notes",
    "generate_result_summary",
    "get_deepseek_key",
    "recommend_model",
]
