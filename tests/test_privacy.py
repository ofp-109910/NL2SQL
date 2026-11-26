"""使用假密钥验证凭据读取到错误输出的脱敏边界，不调用外部 API。"""
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from nl2sql_handbook.cli import main
from nl2sql_handbook.config import read_api_key
from nl2sql_handbook.nl2sql_demo import describe_error


class PrivacyTests(unittest.TestCase):
    def test_loaded_file_key_without_sk_prefix_is_hidden(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"DEEPSEEK_API_KEY": ""}):
            source = Path(directory)/"key.txt"
            source.write_text("offline-file-credential", encoding="utf-8")
            key = read_api_key(source)
            source.unlink()  # 格式化错误不应重新读取密钥文件。
            message = describe_error(RuntimeError(f"provider rejected {key}"))
        self.assertNotIn("offline-file-credential", message)
        self.assertIn("[API Key 已隐藏]", message)
        self.assertIn("RuntimeError: provider rejected", message)

    def test_loaded_environment_key_remains_hidden_after_environment_changes(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "offline-old-credential"}):
            key = read_api_key()
            os.environ["DEEPSEEK_API_KEY"] = "offline-new-credential"
            message = describe_error(RuntimeError(f"{key}; offline-new-credential; sk-dummy-fallback"))
        for value in ("offline-old-credential", "offline-new-credential", "sk-dummy-fallback"):
            self.assertNotIn(value, message)

    def test_cli_hides_environment_key_even_when_config_loading_fails(self):
        error = RuntimeError("config error with offline-cli-credential")
        stderr = io.StringIO()
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "offline-cli-credential"}), \
                patch("nl2sql_handbook.config.public_settings", side_effect=error), \
                contextlib.redirect_stderr(stderr):
            result = main(["doctor"])
        self.assertEqual(result, 1)
        self.assertNotIn("offline-cli-credential", stderr.getvalue())
        self.assertIn("[API Key 已隐藏]", stderr.getvalue())
