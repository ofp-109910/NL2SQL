"""迁移边界：新环境配置、跨目录启动、CLI 子进程与实际 HTTP SDK 请求。"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
import unittest


class DistributionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith(("NL2SQL_", "DEEPSEEK_", "LANGSMITH_", "LANGCHAIN_"))}
        self.env.update(NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost",
                        PYTHONDONTWRITEBYTECODE="1")

    def cli(self, *args, extra=None):
        return subprocess.run([sys.executable, "-m", "nl2sql_handbook", *args],
                              cwd=self.root, env={**self.env, **(extra or {})},
                              capture_output=True, text=True, timeout=40)

    def success(self, *args, extra=None):
        result = self.cli(*args, extra=extra)
        self.assertEqual(0, result.returncode, result.stderr + result.stdout)
        return json.loads(result.stdout)

    def test_env_file_and_external_override_without_disclosing_key(self):
        (self.root/".env").write_text("DEEPSEEK_API_KEY=offline-secret\nDEEPSEEK_MODEL=from-file\nNL2SQL_OUTPUT_DIR=generated\n")
        info = self.success("doctor", extra={"DEEPSEEK_MODEL": "from-environment"})
        self.assertTrue(info["api_key_configured"])
        self.assertEqual(info["model"], "from-environment")
        self.assertEqual(info["output_dir"], "<project>/generated")
        self.assertEqual(info["home"], "<project>")
        self.assertEqual(info["env_file"], "<project>/.env")
        self.assertNotIn(str(self.root), json.dumps(info))
        self.assertNotIn("offline-secret", json.dumps(info))

    def test_home_selects_exact_env_file_from_another_working_directory(self):
        home = self.root/"another-project"
        home.mkdir()
        (home/".env").write_text("DEEPSEEK_MODEL=selected-project\n")
        info = self.success("doctor", extra={"NL2SQL_HOME": str(home)})
        self.assertEqual(info["model"], "selected-project")
        self.assertEqual(info["output_dir"], "<project>/outputs")
        details = self.success("doctor", "--show-paths", extra={"NL2SQL_HOME": str(home),
                                                               "DEEPSEEK_API_KEY": "offline-secret"})
        self.assertEqual(details["output_dir"], str(home/"outputs"))
        self.assertNotIn("offline-secret", json.dumps(details))

    def test_doctor_hides_external_paths(self):
        info = self.success("doctor", extra={"NL2SQL_OUTPUT_DIR": str(self.root.parent/"private-output")})
        self.assertEqual(info["output_dir"], "<external path>")
        self.assertNotIn("private-output", json.dumps(info))

    def test_invalid_endpoint_has_actionable_error_without_traceback(self):
        result = self.cli("doctor", extra={"DEEPSEEK_BASE_URL": "https://user:password@example.test"})
        self.assertEqual(1, result.returncode)
        self.assertIn("DEEPSEEK_BASE_URL", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
        self.assertNotIn("password", result.stderr)

    def test_empty_key_does_not_launch_remote_query(self):
        result = self.cli("query", "-q", "订单数")
        self.assertEqual(1, result.returncode)
        self.assertIn("DEEPSEEK_API_KEY", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_minimal_works_outside_repository_and_generates_local_data(self):
        result = self.success("minimal", "--offline")
        self.assertEqual(result["rows"], [["15887.00"]])
        self.assertEqual(result["model_mode"], "scripted_offline")
        self.assertTrue((self.root/"outputs/data/demo.duckdb").is_file())

    def test_custom_missing_database_is_not_created(self):
        missing = self.root/"typo.duckdb"
        result = self.cli("minimal", "--offline", "--db", str(missing))
        self.assertNotEqual(0, result.returncode)
        self.assertFalse(missing.exists())

    def test_installed_rag_resource_and_cli(self):
        result = self.success("rag", "--retrieve-only", "-q", "销售额怎么计算")
        self.assertTrue(result)
        from nl2sql_handbook.rag_demo.retrieval import DEMO_KNOWLEDGE
        from nl2sql_handbook.ai4db_demo.fixture import ROOT
        self.assertTrue(DEMO_KNOWLEDGE.is_file())
        self.assertTrue((ROOT/"workloads/original.sql").is_file())

    def test_new_workflow_and_key_reach_real_worker_via_local_http(self):
        # A local provider speaks the real SDK protocol; no paid provider is called.
        replies = [json.dumps({"tables": {"main.orders": ["order_id", "status"]},
                               "requirements": "统计已完成订单数"}),
                   "SELECT COUNT(*) AS n FROM orders WHERE status='已完成'",
                   json.dumps({"status": "approved", "feedback": "条件与问题一致"})]
        received = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                received.append((self.path, self.headers.get("Authorization"), data))
                if len(received) > len(replies):
                    self.send_error(500)
                    return
                body = json.dumps({"id": "offline-request", "object": "chat.completion", "created": 0,
                                   "model": data["model"], "choices": [{"index": 0, "finish_reason": "stop",
                                   "message": {"role": "assistant", "content": replies[len(received)-1]}}],
                                   "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            (self.root/".env").write_text(
                "DEEPSEEK_API_KEY=local-test-credential\nDEEPSEEK_MODEL=offline-model\n"
                f"DEEPSEEK_BASE_URL=http://127.0.0.1:{server.server_port}/v1\n")
            result = self.success("query", "-q", "已完成订单有多少笔？")
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertTrue(result["ok"])
        self.assertEqual(result["variant"], "workflow_v2")
        self.assertEqual(result["rows"], [[8]])
        self.assertEqual(result["metrics"]["model_calls"], 3)
        self.assertEqual(len(received), 3)
        self.assertTrue(all(path == "/v1/chat/completions" for path, _, _ in received))
        self.assertTrue(all(auth == "Bearer local-test-credential" for _, auth, _ in received))
        self.assertTrue(all(body["model"] == "offline-model" for _, _, body in received))
        self.assertNotIn("local-test-credential", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
