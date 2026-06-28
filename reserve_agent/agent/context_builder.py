from __future__ import annotations

from typing import Any

from reserve_agent.agent.explanation import build_llm_payload
from reserve_agent.data.loader import DataQualityReport
from reserve_agent.models.reserving import ReservingOutputs


def build_chat_context(report: DataQualityReport, outputs: ReservingOutputs) -> dict[str, Any]:
    """Build a compact context object for follow-up Agent chat."""

    return build_llm_payload(report, outputs)
