"""错误消息的密钥脱敏；不导入配置，以便配置加载失败时也能使用。"""
from __future__ import annotations

import os
import re


_loaded_api_keys: set[str] = set()


def remember_api_key(value: str) -> None:
    """仅在进程内记住已读取的密钥，用于遮盖上游异常中的原值。"""
    if value:
        _loaded_api_keys.add(value)


def redact_api_keys(message: str) -> str:
    values = _loaded_api_keys.copy()
    values.add(os.environ.get("DEEPSEEK_API_KEY", "").strip())
    for value in sorted(filter(None, values), key=len, reverse=True):
        message = message.replace(value, "[API Key 已隐藏]")
    return re.sub(r"sk-[A-Za-z0-9_-]+", "[API Key 已隐藏]", message)
