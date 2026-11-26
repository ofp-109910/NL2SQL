"""CLI、Python API 与子进程共用的项目配置。

只读取 NL2SQL_HOME 下的 .env（或显式 NL2SQL_ENV_FILE），不向上搜索。
外部环境变量优先于 .env。配置在进程首次导入时固定，修改后重启进程。
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from urllib.parse import urlsplit

from dotenv import load_dotenv

from .privacy import remember_api_key


@dataclass(frozen=True)
class Settings:
    home: Path
    env_file: Path
    output_dir: Path
    key_file: Path
    model: str
    base_url: str


def load_settings() -> Settings:
    home = Path(os.environ.get("NL2SQL_HOME", Path.cwd())).expanduser().resolve()

    def resolve(value: str | Path) -> Path:
        path = Path(value).expanduser()
        return (path if path.is_absolute() else home / path).resolve()

    env_file = resolve(os.environ.get("NL2SQL_ENV_FILE", ".env"))
    load_dotenv(env_file, override=False, interpolate=False)
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash").strip()
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip().rstrip("/")
    url = urlsplit(base_url)
    if not model:
        raise ValueError("DEEPSEEK_MODEL 不能为空")
    if url.scheme not in {"https", "http"} or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("DEEPSEEK_BASE_URL 必须是无凭据、无查询参数的 HTTP(S) API 地址")
    return Settings(home, env_file, resolve(os.environ.get("NL2SQL_OUTPUT_DIR", "outputs")),
                    resolve(os.environ.get("NL2SQL_API_KEY_FILE", ".secrets/deepseek_api")), model, base_url)


SETTINGS = load_settings()


def read_api_key(key_file: Path | None = None) -> str:
    """环境变量（包括 .env）优先，其次为显式/配置的单行密钥文件。"""
    value = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not value:
        source = Path(key_file or SETTINGS.key_file).expanduser()
        try:
            value = source.read_text(encoding="utf-8").strip()
        except OSError:
            raise ValueError("未配置 API Key：请填写项目 .env 的 DEEPSEEK_API_KEY，或设置同名环境变量") from None
    if not value or len(value.splitlines()) != 1 or value.lower() in {"your-api-key", "replace-me", "your_api_key_here"}:
        raise ValueError("API Key 必须为非空单行真实密钥，请修改本地 .env")
    remember_api_key(value)
    return value


def public_settings(*, show_paths: bool = False) -> dict:
    """默认隐藏绝对路径；不验证远程鉴权，也不读取密钥文件内容。"""
    def display_path(path: Path) -> str:
        if show_paths:
            return str(path)
        try:
            relative = path.relative_to(SETTINGS.home)
        except ValueError:
            return "<external path>"
        return "<project>" if relative == Path(".") else f"<project>/{relative.as_posix()}"

    return {"home": display_path(SETTINGS.home), "env_file": display_path(SETTINGS.env_file),
            "env_file_exists": SETTINGS.env_file.is_file(), "output_dir": display_path(SETTINGS.output_dir),
            "model": SETTINGS.model, "base_url": SETTINGS.base_url,
            "api_key_configured": bool(os.environ.get("DEEPSEEK_API_KEY", "").strip()) or SETTINGS.key_file.is_file()}
