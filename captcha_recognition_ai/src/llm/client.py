import base64
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


load_dotenv()


def has_openai_config() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def image_to_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def parse_json_safely(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return {"raw_response": text}


def call_vision_json(image_path: Path, prompt: str) -> dict[str, Any]:
    if not has_openai_config():
        return {
            "llm_enabled": False,
            "suggestion": "review_manually",
            "note": "OPENAI_API_KEY is not configured; skipped multimodal review.",
        }

    from openai import OpenAI

    client = OpenAI()
    model = os.getenv("OPENAI_VISION_MODEL", "gpt-4o-mini")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                ],
            }
        ],
        temperature=0,
    )
    text = response.choices[0].message.content or "{}"
    result = parse_json_safely(text)
    result["llm_enabled"] = True
    result["model"] = model
    return result


def call_text_markdown(prompt: str) -> str:
    if not has_openai_config():
        return (
            "# 大模型报告生成未启用\n\n"
            "当前未配置 `OPENAI_API_KEY`，因此本报告由本地模板生成。"
            "配置 API Key 后可调用大模型自动生成更完整的训练分析报告。\n"
        )

    from openai import OpenAI

    client = OpenAI()
    model = os.getenv("OPENAI_TEXT_MODEL", "gpt-4o-mini")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return response.choices[0].message.content or ""
