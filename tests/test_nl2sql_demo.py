"""离线验证 SQL 链路和查询限制；这些测试不会调用 DeepSeek API。"""

import io
import json
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from decimal import Decimal
from pathlib import Path
from unittest.mock import Mock, patch

import duckdb
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from nl2sql_handbook.nl2sql_demo import (
    EXPLANATION_MAX_ROWS,
    NL2SQLDemo,
    SQLExecutionError,
    clean_sql,
    connect_read_only,
    initialize_demo_database,
    load_api_key,
    main,
)


SALES_SELECTION = {
    "main.customers": ["customer_id", "city"],
    "main.orders": ["customer_id", "quantity", "unit_price", "status"],
}


class RecordingHandler(BaseCallbackHandler):
    """记录模型实际收到的消息，检查 schema 裁剪和错误反馈是否真正生效。"""

    def __init__(self):
        self.messages = []

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.messages.extend(messages)


class NL2SQLDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 测试使用临时数据库，不改变当前目录中的实际数据。
        cls.directory = tempfile.TemporaryDirectory()
        cls.db_path = Path(cls.directory.name) / "test.duckdb"
        initialize_demo_database(cls.db_path)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def make_demo(self, sql="SELECT 1", max_rows=100, *, selection=None,
                  responses=None, recorder=None, max_sql_retries=3):
        if responses is None:
            selected = SALES_SELECTION if selection is None else selection
            responses = [json.dumps(selected), sql]
        return NL2SQLDemo(
            self.db_path,
            llm=FakeListChatModel(
                responses=responses,
                callbacks=[recorder] if recorder else None,
            ),
            max_rows=max_rows,
            max_sql_retries=max_sql_retries,
        )

    def test_langchain_pipeline_and_sales_aggregation(self):
        # 假模型只替代远端 API；提示词、输出解析、校验、DuckDB 执行均走真实流程。
        sql = """```sql
SELECT c.city, SUM(o.quantity * o.unit_price) AS sales
FROM orders o JOIN customers c ON o.customer_id = c.customer_id
WHERE o.status = '已完成'
GROUP BY c.city ORDER BY sales DESC
```"""
        with self.make_demo(sql) as demo:
            result = demo.ask("统计每个城市的已完成订单销售额")
            self.assertEqual(result.columns, ["city", "sales"])
            self.assertEqual(result.rows, [
                ("北京", Decimal("7993.00")),
                ("上海", Decimal("5997.00")),
                ("深圳", Decimal("1897.00")),
            ])
            self.assertFalse(result.truncated)
            self.assertIn("订单状态", demo.schema)
            self.assertEqual(result.selected_schema, SALES_SELECTION)
            self.assertEqual(result.attempts, 1)
            self.assertEqual(result.retry_history, [])
            self.assertIsNone(result.explanation)

    def test_sql_generation_receives_only_selected_schema(self):
        recorder = RecordingHandler()
        with self.make_demo("SELECT COUNT(*) FROM orders", recorder=recorder) as demo:
            demo.ask("已完成订单有多少笔？")
        self.assertEqual(len(recorder.messages), 2)
        selection_prompt = recorder.messages[0][0].content
        sql_prompt = recorder.messages[1][0].content
        self.assertIn('"main.products"', selection_prompt)
        self.assertIn('"main.orders"', sql_prompt)
        self.assertIn("成交单价", sql_prompt)
        self.assertNotIn('"main.products"', sql_prompt)
        self.assertNotIn('"order_date"', sql_prompt)

    def test_schema_selection_rejects_unknown_or_malformed_names(self):
        for selection in [
            {"main.nonexistent": ["id"]},
            {"main.orders": ["nonexistent"]},
            {"main.orders": []},
            {"main.orders": "order_id"},
            {"main.orders": [1]},
            ["main.orders"],
        ]:
            recorder = RecordingHandler()
            with self.subTest(selection=selection):
                with self.make_demo(selection=selection, recorder=recorder) as demo:
                    with self.assertRaises(ValueError):
                        demo.ask("查询订单")
                self.assertEqual(len(recorder.messages), 1)

    def test_schema_selection_rejects_non_json(self):
        recorder = RecordingHandler()
        with self.make_demo(responses=["这不是 JSON"], recorder=recorder) as demo:
            with self.assertRaises(OutputParserException):
                demo.ask("查询订单")
        self.assertEqual(len(recorder.messages), 1)

    def test_empty_selection_supports_unanswerable_question(self):
        with self.make_demo(
            "SELECT '无法根据现有表回答该问题' AS message", selection={}
        ) as demo:
            result = demo.ask("明天天气怎么样？")
        self.assertEqual(result.selected_schema, {})
        self.assertEqual(result.rows, [("无法根据现有表回答该问题",)])

    def test_execution_error_is_returned_to_model_for_correction(self):
        recorder = RecordingHandler()
        bad_sql = "SELECT SUM(missing_amount) FROM orders"
        good_sql = "SELECT SUM(quantity * unit_price) FROM orders WHERE status = '已完成'"
        with self.make_demo(responses=[
            json.dumps(SALES_SELECTION), bad_sql, good_sql,
        ], recorder=recorder) as demo:
            result = demo.ask("已完成订单销售额是多少？")
        self.assertEqual(result.rows, [(Decimal("15887.00"),)])
        self.assertEqual(result.sql, good_sql)
        self.assertEqual(result.attempts, 2)
        self.assertEqual(len(result.retry_history), 1)
        correction_messages = recorder.messages[2]
        self.assertIn(bad_sql, correction_messages[-1].content)
        self.assertIn("BinderException", correction_messages[-1].content)
        self.assertNotIn('"main.products"', correction_messages[0].content)

    def test_sql_syntax_error_can_be_corrected(self):
        with self.make_demo(responses=[
            json.dumps(SALES_SELECTION), "SELECT FROM", "SELECT COUNT(*) FROM orders",
        ]) as demo:
            result = demo.ask("有多少订单？")
        self.assertEqual(result.rows, [(10,)])
        self.assertEqual(result.attempts, 2)
        self.assertIn("ParserException", result.retry_history[0]["error"])

    def test_stops_after_three_corrections(self):
        recorder = RecordingHandler()
        bad_sql = "SELECT missing_column FROM orders"
        with self.make_demo(responses=[
            json.dumps(SALES_SELECTION), bad_sql, bad_sql, bad_sql, bad_sql,
        ], recorder=recorder) as demo:
            with self.assertRaises(SQLExecutionError) as caught:
                demo.ask("查询订单")
            self.assertEqual(demo.execute_sql("SELECT COUNT(*) FROM orders").rows, [(10,)])
        self.assertEqual(caught.exception.attempts, 4)
        self.assertEqual(len(caught.exception.history), 4)
        self.assertEqual(caught.exception.last_sql, bad_sql)
        # 选择表列 1 次 + 生成 SQL 4 次，不能再调用第五次 SQL 生成。
        self.assertEqual(len(recorder.messages), 5)

    def test_third_correction_can_succeed(self):
        bad_sql = "SELECT missing_column FROM orders"
        with self.make_demo(responses=[
            json.dumps(SALES_SELECTION), bad_sql, bad_sql, bad_sql, "SELECT 42",
        ]) as demo:
            result = demo.ask("查询订单")
        self.assertEqual(result.rows, [(42,)])
        self.assertEqual(result.attempts, 4)
        self.assertEqual(len(result.retry_history), 3)

    def test_zero_corrections_means_one_sql_attempt(self):
        recorder = RecordingHandler()
        with self.make_demo("SELECT missing_column FROM orders", recorder=recorder,
                            max_sql_retries=0) as demo:
            with self.assertRaises(SQLExecutionError) as caught:
                demo.ask("查询订单")
        self.assertEqual(caught.exception.attempts, 1)
        self.assertEqual(len(recorder.messages), 2)

    def test_invalid_correction_limit_rejected(self):
        for value in [-1, 4, 1.5, "3"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.make_demo(max_sql_retries=value)

    def test_unsafe_sql_is_not_repaired_or_executed(self):
        recorder = RecordingHandler()
        with self.make_demo("DELETE FROM orders", recorder=recorder) as demo:
            with self.assertRaisesRegex(ValueError, "仅允许"):
                demo.ask("删除所有订单")
            self.assertEqual(demo.execute_sql("SELECT COUNT(*) FROM orders").rows, [(10,)])
        self.assertEqual(len(recorder.messages), 2)

    def test_model_request_failure_does_not_trigger_sql_corrections(self):
        with self.make_demo() as demo:
            failed_chain = Mock()
            failed_chain.invoke.side_effect = RuntimeError("API 请求失败")
            demo.sql_chain = failed_chain
            with self.assertRaisesRegex(RuntimeError, "API 请求失败"):
                demo.ask("查询订单")
            failed_chain.invoke.assert_called_once()

    def test_retry_history_is_reset_between_questions(self):
        recorder = RecordingHandler()
        with self.make_demo(responses=[
            json.dumps(SALES_SELECTION), "SELECT missing_column FROM orders", "SELECT 1",
            json.dumps(SALES_SELECTION), "SELECT 2",
        ], recorder=recorder) as demo:
            first = demo.ask("第一个问题")
            second = demo.ask("第二个问题")
        self.assertEqual(first.attempts, 2)
        self.assertEqual(second.attempts, 1)
        self.assertEqual(second.retry_history, [])
        self.assertTrue(recorder.messages[-1][-1].content.endswith("[]"))

    def test_optional_explanation_receives_actual_results(self):
        recorder = RecordingHandler()
        with self.make_demo(responses=[
            json.dumps(SALES_SELECTION), "SELECT COUNT(*) AS count FROM orders", "共有 10 笔订单。",
        ], recorder=recorder) as demo:
            result = demo.ask("共有多少订单？", explain=True)
        self.assertEqual(result.explanation, "共有 10 笔订单。")
        self.assertIsNone(result.explanation_error)
        payload = json.loads(recorder.messages[-1][-1].content)
        self.assertEqual(payload["columns"], ["count"])
        self.assertEqual(payload["rows"], [[10]])
        self.assertFalse(payload["sampled"])
        self.assertEqual(len(recorder.messages), 3)

    def test_explanation_reports_sampling_and_truncation(self):
        recorder = RecordingHandler()
        sql = "SELECT order_id FROM orders CROSS JOIN range(3) ORDER BY order_id"
        with self.make_demo(responses=[
            json.dumps({"main.orders": ["order_id"]}), sql, "这是抽样结果。",
        ], recorder=recorder, max_rows=25) as demo:
            result = demo.ask("列出所有订单", explain=True)
        payload = json.loads(recorder.messages[-1][-1].content)
        self.assertEqual(len(result.rows), 25)
        self.assertEqual(len(payload["rows"]), EXPLANATION_MAX_ROWS)
        self.assertEqual(payload["returned_rows"], 25)
        self.assertTrue(payload["sampled"])
        self.assertTrue(payload["truncated"])

    def test_failed_explanation_preserves_query_result(self):
        with self.make_demo("SELECT COUNT(*) FROM orders") as demo:
            demo.explanation_chain = Mock()
            demo.explanation_chain.invoke.side_effect = RuntimeError("解释 API 暂不可用")
            result = demo.ask("有多少订单？", explain=True)
        self.assertEqual(result.rows, [(10,)])
        self.assertIsNone(result.explanation)
        self.assertIn("解释 API 暂不可用", result.explanation_error)

    def test_empty_explanation_preserves_query_result(self):
        with self.make_demo(responses=[json.dumps(SALES_SELECTION), "SELECT 1", " "]) as demo:
            result = demo.ask("查询订单", explain=True)
        self.assertEqual(result.rows, [(1,)])
        self.assertIn("空的结果解释", result.explanation_error)

    def test_cli_explain_flag_runs_complete_offline_pipeline(self):
        # 替换远端模型和密钥读取，真实运行 CLI 解析、各条链、DuckDB 及 JSON 输出。
        model = FakeListChatModel(responses=[
            json.dumps({"main.orders": ["order_id"]}),
            "SELECT COUNT(*) AS count FROM orders", "共有 10 笔订单。",
        ])
        output = io.StringIO()
        with patch("sys.argv", [
            "nl2sql_demo.py", "--db", str(self.db_path), "--explain",
            "--max-sql-retries", "0", "-q", "共有多少订单？",
        ]), patch("nl2sql_handbook.nl2sql_demo.load_api_key", return_value="test-api-key"), \
                patch("nl2sql_handbook.nl2sql_demo.ChatDeepSeek", return_value=model), redirect_stdout(output):
            exit_code = main()
        self.assertEqual(exit_code, 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result["rows"], [[10]])
        self.assertEqual(result["explanation"], "共有 10 笔订单。")
        self.assertEqual(result["selected_schema"], {"main.orders": ["order_id"]})

    def test_cli_correction_limit_and_failure_exit_code(self):
        recorder = RecordingHandler()
        model = FakeListChatModel(responses=[
            json.dumps({"main.orders": ["order_id"]}),
            "SELECT missing_column FROM orders", "SELECT 1",
        ], callbacks=[recorder])
        errors = io.StringIO()
        with patch("sys.argv", [
            "nl2sql_demo.py", "--db", str(self.db_path),
            "--max-sql-retries", "0", "-q", "查询订单",
        ]), patch("nl2sql_handbook.nl2sql_demo.load_api_key", return_value="test-api-key"), \
                patch("nl2sql_handbook.nl2sql_demo.ChatDeepSeek", return_value=model), redirect_stderr(errors):
            exit_code = main()
        self.assertEqual(exit_code, 1)
        self.assertIn("共尝试 1 次", errors.getvalue())
        self.assertIn("SELECT missing_column FROM orders", errors.getvalue())
        self.assertEqual(len(recorder.messages), 2)

    def test_rejects_mutations_and_multiple_statements(self):
        with self.make_demo() as demo:
            for sql in [
                "DELETE FROM orders",
                "DROP TABLE orders",
                "SELECT 1; DELETE FROM orders",
                "WITH x AS (SELECT 1) DELETE FROM orders",
                "COPY orders TO '/tmp/nl2sql_unexpected_output.csv'",
                "",
            ]:
                with self.subTest(sql=sql), self.assertRaises(ValueError):
                    demo.execute_sql(sql)
            # 拒绝操作后，数据仍完整。
            self.assertEqual(demo.execute_sql("SELECT COUNT(*) FROM orders").rows, [(10,)])

    def test_accepts_cte_and_semicolon_inside_string(self):
        with self.make_demo() as demo:
            result = demo.execute_sql("WITH x AS (SELECT 'a;b' AS text) SELECT * FROM x;")
            self.assertEqual(result.rows, [("a;b",)])
            # 即使末尾有注释，Relation API 的 LIMIT 也必须保持有效。
            self.assertEqual(demo.execute_sql("SELECT 42; -- comment").rows, [(42,)])

    def test_blocks_external_file_access(self):
        csv_path = Path(self.directory.name) / "private.csv"
        csv_path.write_text("value\n42\n", encoding="utf-8")
        with self.make_demo() as demo:
            with self.assertRaises(duckdb.Error):
                demo.execute_sql(f"SELECT * FROM read_csv('{csv_path.as_posix()}')")

    def test_connection_itself_is_read_only(self):
        with self.make_demo() as demo:
            # 即使绕过应用层解析，数据库的只读连接也不能写入。
            with self.assertRaises(duckdb.Error):
                demo.connection.execute("DELETE FROM orders")

    def test_result_limit_and_truncation(self):
        with self.make_demo(max_rows=2) as demo:
            result = demo.execute_sql("SELECT order_id FROM orders ORDER BY order_id")
            self.assertEqual(result.rows, [(1,), (2,)])
            self.assertTrue(result.truncated)
            result = demo.execute_sql("SELECT order_id FROM orders ORDER BY order_id LIMIT 2")
            self.assertFalse(result.truncated)

    def test_initialization_is_idempotent(self):
        initialize_demo_database(self.db_path)
        with self.make_demo() as demo:
            self.assertEqual(demo.execute_sql("SELECT COUNT(*) FROM orders").rows, [(10,)])

    def test_key_file_and_environment_priority(self):
        key_file = Path(self.directory.name) / "test-key"
        key_file.write_text("test-file-token\n", encoding="utf-8")
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": ""}):
            self.assertEqual(load_api_key(key_file), "test-file-token")
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test-env-token"}):
            self.assertEqual(load_api_key(key_file), "test-env-token")

    def test_sqlite_file_has_clear_error(self):
        sqlite_path = Path(self.directory.name) / "source.db"
        with sqlite3.connect(sqlite_path) as source:
            source.execute("CREATE TABLE example(id INTEGER)")
        with self.assertRaisesRegex(ValueError, "SQLite"):
            connect_read_only(sqlite_path)

    def test_empty_input_rejected(self):
        with self.assertRaises(ValueError):
            clean_sql("   ")
        with self.make_demo() as demo, self.assertRaises(ValueError):
            demo.ask("   ")


if __name__ == "__main__":
    unittest.main()
