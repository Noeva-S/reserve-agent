from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import requests


DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_DEEPSEEK_MODEL = "deepseek-chat"

# 如果你不想用 .env 或页面密码框，可以把 DeepSeek API Key 填在这里：
# 注意：填入真实 key 后，不要把这个文件发到公共仓库或群文件里。


def load_env_file(path: str | Path = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def get_deepseek_key(explicit_key: str | None = None) -> str | None:
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    load_env_file()
    key = os.environ.get("DEEPSEEK_API_KEY")
    return key.strip() if key and key.strip() else None


def build_reserving_prompt(payload: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "你是一个非寿险准备金评估助手，熟悉 SOA/CAS 的 unpaid claims reserving 思路。"
        "请基于用户提供的结构化结果，用中文生成精炼、专业、适合课程报告的分析。"
        "不要编造数据，不要声称已经进行人工审计。"
    )
    user = (
        "请解释以下准备金评估结果。要求：\n"
        "1. 先总结数据质量；\n"
        "2. 再解释 Chain Ladder、ELR、Bornhuetter-Ferguson 的结果差异；\n"
        "3. 指出准备金贡献最高的事故年和可能原因；\n"
        "4. 给出模型选择建议和后续复核建议；\n"
        "5. 字数控制在 500-800 字。\n\n"
        f"结构化结果 JSON：\n{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def call_deepseek(
    messages: list[dict[str, str]],
    api_key: str,
    model: str | None = None,
    temperature: float = 0.3,
    timeout: int = 60,
) -> str:
    model_name = model or os.environ.get("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL
    response = requests.post(
        f"{DEEPSEEK_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        },
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"DeepSeek API 请求失败：HTTP {response.status_code} - {response.text[:500]}")
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def make_payload(data_quality: dict[str, Any], diagnostics: dict[str, Any], comparison_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "data_quality": data_quality,
        "model_totals": diagnostics,
        "comparison_by_accident_year": comparison_rows,
    }
