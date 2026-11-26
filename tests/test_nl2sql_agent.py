"""通过脚本模型验证真正的 create_agent/ToolMessage 循环，不调用外部 API。"""

import io
import itertools
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import duckdb
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from pydantic import Field

from nl2sql_handbook.nl2sql_agent import AgentExecutionError, NL2SQLAgent, main
from nl2sql_handbook.nl2sql_demo import initialize_demo_database


class ToolCallingModel(FakeMessagesListChatModel):
    """只替代 DeepSeek 的响应，Agent 图、工具路由和数据库均使用真实实现。"""

    requests: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tool_names: list[str] = Field(default_factory=list)
    fail_on_call: int | None = None

    def bind_tools(self, tools, **kwargs):
        self.bound_tool_names = [item.name for item in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests.append(list(messages))
        if self.fail_on_call == len(self.requests):
            raise RuntimeError("模拟模型 API 暂时不可用")
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


CALL_IDS = itertools.count()
ORDER_SCHEMA = {"main.orders": ["order_id"]}
SALES_SCHEMA = {"main.orders": ["quantity", "unit_price", "status"]}


def action(name, arguments):
    return AIMessage(content="", tool_calls=[{
        "name": name, "args": arguments, "id": f"call_{next(CALL_IDS)}", "type": "tool_call",
    }])


def schema_action(selection=None):
    return action("get_schema", {"selection": ORDER_SCHEMA if selection is None else selection})


def query_action(sql="SELECT COUNT(*) AS count FROM orders"):
    return action("execute_sql", {"sql": sql})


class NL2SQLAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.db_path = Path(cls.directory.name) / "agent-test.duckdb"
        initialize_demo_database(cls.db_path)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def make_agent(self, responses=None, **kwargs):
        model = ToolCallingModel(responses=responses or [
            action("list_tables", {}), schema_action(), query_action(), AIMessage(content="查询完成。"),
        ])
        return NL2SQLAgent(self.db_path, llm=model, **kwargs), model

    def test_real_agent_routes_tool_calls_and_returns_sql_result(self):
        agent, model = self.make_agent()
        with agent:
            result = agent.ask("共有多少笔订单？")
        self.assertEqual(result.rows, [(10,)])
        self.assertEqual(result.columns, ["count"])
        self.assertEqual(result.selected_schema, ORDER_SCHEMA)
        self.assertEqual(result.attempts, 1)
        self.assertIsNone(result.explanation)
        self.assertEqual([event["tool"] for event in result.tool_trace], [
            "list_tables", "get_schema", "execute_sql",
        ])
        self.assertEqual(model.bound_tool_names, ["list_tables", "get_schema", "execute_sql"])
        self.assertEqual(len(model.requests), 4)
        self.assertTrue(any(isinstance(msg, ToolMessage) for msg in model.requests[-1]))
        # 初始提示词不会直接携带完整的数据库 schema。
        self.assertNotIn('"main.products"', model.requests[0][0].content)

    def test_agent_can_reinspect_schema_after_execution_error(self):
        sql = "SELECT SUM(quantity * unit_price) AS sales FROM orders WHERE status = '已完成'"
        agent, model = self.make_agent([
            schema_action(), query_action("SELECT missing_amount FROM orders"),
            action("list_tables", {}), schema_action(SALES_SCHEMA), query_action(sql),
            AIMessage(content="已完成订单销售额为 15887 元。"),
        ])
        with agent:
            result = agent.ask("已完成订单销售额是多少？", explain=True)
        self.assertEqual(result.rows, [(Decimal("15887.00"),)])
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.selected_schema, SALES_SCHEMA)
        self.assertIn("15887", result.explanation)
        self.assertEqual([event["tool"] for event in result.tool_trace], [
            "get_schema", "execute_sql", "list_tables", "get_schema", "execute_sql",
        ])
        error_messages = [
            json.loads(msg.content) for msg in model.requests[2] if isinstance(msg, ToolMessage)
        ]
        self.assertIn("BinderException", error_messages[-1]["error"])
        self.assertTrue(error_messages[-1]["retry_allowed"])

    def test_bad_schema_selection_is_observed_and_can_be_reselected(self):
        agent, _ = self.make_agent([
            schema_action({"main.orders": ["missing_column"]}),
            action("list_tables", {}), schema_action(), query_action(), AIMessage(content="查询完成。"),
        ])
        with agent:
            result = agent.ask("有多少订单？")
        self.assertFalse(result.tool_trace[0]["observation"]["ok"])
        self.assertEqual(result.rows, [(10,)])
        self.assertEqual(result.attempts, 1)

    def test_execute_requires_real_schema_observation(self):
        agent, _ = self.make_agent([
            query_action(), schema_action(), query_action(), AIMessage(content="查询完成。"),
        ])
        with agent:
            result = agent.ask("有多少订单？")
        self.assertEqual(result.attempts, 2)
        self.assertIn("get_schema", result.retry_history[0]["error"])
        self.assertEqual(result.rows, [(10,)])

    def test_three_corrections_and_four_attempts_are_hard_limit(self):
        bad_sql = "SELECT missing_column FROM orders"
        agent, _ = self.make_agent([
            schema_action(), *[query_action(bad_sql) for _ in range(4)],
            query_action("SELECT 42"), AIMessage(content="无法完成查询。"),
        ])
        with agent:
            with self.assertRaises(AgentExecutionError) as caught:
                agent.ask("查询订单")
        self.assertEqual(caught.exception.attempts, 4)
        self.assertEqual(len(caught.exception.history), 4)
        last_observation = caught.exception.tool_trace[-1]["observation"]
        self.assertFalse(last_observation["retry_allowed"])
        self.assertIn("额度已耗尽", last_observation["error"])

    def test_third_correction_can_succeed(self):
        agent, _ = self.make_agent([
            schema_action(), *[query_action("SELECT missing_column FROM orders") for _ in range(3)],
            query_action(), AIMessage(content="查询完成。"),
        ])
        with agent:
            result = agent.ask("有多少订单？")
        self.assertEqual(result.rows, [(10,)])
        self.assertEqual(result.attempts, 4)
        self.assertEqual(len(result.retry_history), 3)

    def test_zero_retries_allows_only_one_tool_attempt(self):
        agent, _ = self.make_agent([
            schema_action(), query_action("SELECT missing_column FROM orders"),
            query_action(), AIMessage(content="无法完成。"),
        ], max_sql_retries=0)
        with agent:
            with self.assertRaises(AgentExecutionError) as caught:
                agent.ask("查询订单")
        self.assertEqual(caught.exception.attempts, 1)
        self.assertEqual(len(caught.exception.history), 1)

    def test_write_and_multi_statement_sql_are_blocked(self):
        for sql in [
            "DELETE FROM orders", "DROP TABLE orders", "SELECT 1; DELETE FROM orders",
            "WITH x AS (SELECT 1) DELETE FROM orders",
            "COPY orders TO '/tmp/agent-unexpected-output.csv'",
        ]:
            with self.subTest(sql=sql):
                agent, _ = self.make_agent([
                    schema_action(), query_action(sql), AIMessage(content="该操作不被允许。"),
                ])
                with agent:
                    with self.assertRaises(AgentExecutionError) as caught:
                        agent.ask("执行这个操作")
                    self.assertIn("仅允许", caught.exception.history[0]["error"])
                    count = agent.connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0]
                    self.assertEqual(count, 10)

    def test_external_file_access_is_blocked(self):
        csv_path = Path(self.directory.name) / "private.csv"
        csv_path.write_text("value\n42\n", encoding="utf-8")
        agent, _ = self.make_agent([
            schema_action(), query_action(f"SELECT * FROM read_csv('{csv_path.as_posix()}')"),
            AIMessage(content="访问不被允许。"),
        ])
        with agent, self.assertRaises(AgentExecutionError) as caught:
            agent.ask("读取文件")
        self.assertIn("Permission", caught.exception.history[0]["error"])

    def test_duckdb_connection_is_read_only(self):
        agent, _ = self.make_agent()
        with agent, self.assertRaises(duckdb.Error):
            agent.connection.execute("DELETE FROM orders")

    def test_cte_and_semicolon_inside_string_work(self):
        agent, _ = self.make_agent([
            schema_action(), query_action("WITH x AS (SELECT 'a;b' AS value) SELECT * FROM x; -- end"),
            AIMessage(content="查询完成。"),
        ])
        with agent:
            result = agent.ask("查询常量")
        self.assertEqual(result.rows, [("a;b",)])

    def test_raw_records_are_local_by_default(self):
        agent, model = self.make_agent([
            schema_action(), query_action("SELECT order_id FROM orders ORDER BY order_id"),
            AIMessage(content="查询完成。"),
        ])
        with agent:
            result = agent.ask("列出所有订单")
        observation = json.loads([
            msg for msg in model.requests[-1] if isinstance(msg, ToolMessage)
        ][-1].content)
        self.assertEqual(len(result.rows), 10)
        self.assertNotIn("rows", observation)
        self.assertNotIn("rows_preview", observation)
        self.assertFalse(observation["explain"])
        self.assertIsNone(result.explanation)

    def test_explanation_uses_actual_bounded_preview(self):
        agent, model = self.make_agent([
            schema_action(), query_action("SELECT order_id FROM orders CROSS JOIN range(3) ORDER BY order_id"),
            AIMessage(content="返回了 25 行，结果已截断；当前只展示了前 20 行供解释。"),
        ], max_rows=25)
        with agent:
            result = agent.ask("列出所有订单", explain=True)
        observation = json.loads([
            msg for msg in model.requests[-1] if isinstance(msg, ToolMessage)
        ][-1].content)
        self.assertEqual(len(result.rows), 25)
        self.assertEqual(len(observation["rows_preview"]), 20)
        self.assertTrue(observation["sampled"])
        self.assertTrue(observation["truncated"])
        self.assertIn("截断", result.explanation)

    def test_exact_row_limit_does_not_report_truncation(self):
        agent, _ = self.make_agent([
            schema_action(), query_action("SELECT order_id FROM orders ORDER BY order_id LIMIT 2"),
            AIMessage(content="完成。"),
        ], max_rows=2)
        with agent:
            result = agent.ask("前两笔订单")
        self.assertEqual(result.rows, [(1,), (2,)])
        self.assertFalse(result.truncated)

    def test_failed_model_summary_preserves_successful_query(self):
        agent, model = self.make_agent([
            schema_action(), query_action(), AIMessage(content="未被使用的响应"),
        ])
        model.fail_on_call = 3
        with agent:
            result = agent.ask("有多少订单？", explain=True)
        self.assertEqual(result.rows, [(10,)])
        self.assertIn("模拟模型 API", result.agent_error)
        self.assertIn("模拟模型 API", result.explanation_error)
        self.assertIsNone(result.explanation)

    def test_later_failure_cannot_reuse_stale_successful_query(self):
        agent, _ = self.make_agent([
            schema_action(), query_action(), query_action("SELECT missing_column FROM orders"),
            AIMessage(content="查询完成。"),
        ])
        with agent, self.assertRaises(AgentExecutionError) as caught:
            agent.ask("查询订单")
        self.assertEqual(caught.exception.attempts, 2)
        self.assertEqual(len(caught.exception.history), 1)

    def test_model_loop_is_bounded_by_native_middleware(self):
        agent, model = self.make_agent([
            action("list_tables", {}) for _ in range(4)
        ], max_model_calls=3)
        with agent, self.assertRaises(AgentExecutionError) as caught:
            agent.ask("一直调用工具")
        self.assertEqual(len(model.requests), 3)
        self.assertIn("ModelCallLimitExceeded", str(caught.exception))
        self.assertEqual(caught.exception.attempts, 0)

    def test_agent_cannot_claim_success_without_executing_query(self):
        agent, _ = self.make_agent([AIMessage(content="我猜有 100 笔订单。")])
        with agent, self.assertRaises(AgentExecutionError) as caught:
            agent.ask("多少订单？")
        self.assertEqual(caught.exception.attempts, 0)

    def test_sessions_do_not_share_results_or_counters(self):
        agent, _ = self.make_agent()
        with agent:
            first = agent.ask("第一个问题")
            second = agent.ask("第二个问题")
        self.assertEqual(first.attempts, 1)
        self.assertEqual(second.attempts, 1)
        self.assertEqual(len(first.tool_trace), 3)
        self.assertEqual(len(second.tool_trace), 3)
        self.assertEqual(second.retry_history, [])
        self.assertEqual(second.question, "第二个问题")

    def test_private_runtime_is_not_exposed_in_tool_arguments(self):
        agent, _ = self.make_agent()
        with agent:
            arguments = {
                item.name: item.tool_call_schema.model_json_schema()["properties"]
                for item in agent.tools
            }
        self.assertEqual(arguments["list_tables"], {})
        self.assertEqual(set(arguments["get_schema"]), {"selection"})
        self.assertEqual(set(arguments["execute_sql"]), {"sql"})

    def test_parallel_tool_calls_cannot_overrun_sql_budget(self):
        # 故意让模型在一个回复中请求五次查询，工具也必须保证最多执行四次。
        batch = AIMessage(content="", tool_calls=[query_action().tool_calls[0] for _ in range(5)])
        agent, _ = self.make_agent([schema_action(), batch, AIMessage(content="完成。")])
        with agent:
            result = agent.ask("查询订单")
        self.assertEqual(result.attempts, 4)
        observations = [e["observation"] for e in result.tool_trace if e["tool"] == "execute_sql"]
        self.assertEqual(sum(o["ok"] for o in observations), 4)
        self.assertEqual(sum(not o["ok"] for o in observations), 1)
        self.assertEqual(result.rows, [(10,)])

    def test_invalid_limits_and_empty_question_rejected(self):
        for limits in [
            {"max_rows": 0}, {"max_sql_retries": -1}, {"max_sql_retries": 4},
            {"max_model_calls": 0}, {"max_model_calls": 31},
        ]:
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                self.make_agent(**limits)
        agent, model = self.make_agent()
        with agent, self.assertRaises(ValueError):
            agent.ask("   ")
        self.assertEqual(model.requests, [])

    def test_cli_runs_agent_and_serializes_explanation(self):
        model = ToolCallingModel(responses=[
            schema_action(), query_action(), AIMessage(content="共有 10 笔订单。"),
        ])
        output = io.StringIO()
        with patch("sys.argv", [
            "nl2sql_agent.py", "--db", str(self.db_path), "--explain", "-q", "多少订单？",
        ]), patch("nl2sql_handbook.nl2sql_agent.load_api_key", return_value="test-api-key"), \
                patch("nl2sql_handbook.nl2sql_agent.ChatDeepSeek", return_value=model), redirect_stdout(output):
            exit_code = main()
        result = json.loads(output.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(result["rows"], [[10]])
        self.assertEqual(result["explanation"], "共有 10 笔订单。")
        self.assertEqual(len(result["tool_trace"]), 2)

    def test_cli_returns_nonzero_if_no_query_succeeded(self):
        model = ToolCallingModel(responses=[AIMessage(content="无法查询。")])
        errors = io.StringIO()
        with patch("sys.argv", [
            "nl2sql_agent.py", "--db", str(self.db_path), "-q", "多少订单？",
        ]), patch("nl2sql_handbook.nl2sql_agent.load_api_key", return_value="test-api-key"), \
                patch("nl2sql_handbook.nl2sql_agent.ChatDeepSeek", return_value=model), redirect_stderr(errors):
            exit_code = main()
        self.assertEqual(exit_code, 1)
        self.assertIn("Agent 未获得成功", errors.getvalue())

    def test_show_schema_does_not_initialize_model(self):
        output = io.StringIO()
        with patch("sys.argv", [
            "nl2sql_agent.py", "--db", str(self.db_path), "--show-schema",
        ]), patch("nl2sql_handbook.nl2sql_agent.ChatDeepSeek") as model, redirect_stdout(output):
            exit_code = main()
        self.assertEqual(exit_code, 0)
        self.assertIn("main.orders", json.loads(output.getvalue()))
        model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
