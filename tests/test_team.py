"""离线集成测试：只替换模型，四个真实 Agent 图、交接工具与 DuckDB 均正常运行。"""

import io
import itertools
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import duckdb
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import Field

from nl2sql_handbook.nl2sql_demo import initialize_demo_database
from nl2sql_handbook.multi_agent_demo.cli import main
from nl2sql_handbook.multi_agent_demo.runtime import TeamExecutionError
from nl2sql_handbook.multi_agent_demo.team import MultiAgentNL2SQL, ROLES, ROLE_TOOLS


IDS = itertools.count()
SELECTION = {"main.orders": ["quantity", "unit_price", "status"]}
SQL = "SELECT SUM(quantity * unit_price) AS sales FROM main.orders WHERE status = '已完成'"


class ScriptedModel(FakeMessagesListChatModel):
    requests: list[list[BaseMessage]] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)
    fail_on_call: int | None = None

    def bind_tools(self, tools, **kwargs):
        self.tool_names = [item.name for item in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests.append(list(messages))
        if len(self.requests) == self.fail_on_call:
            raise RuntimeError("模拟模型故障")
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def action(name, **arguments):
    return AIMessage(content="", tool_calls=[{
        "name": name, "args": arguments, "id": f"call-{next(IDS)}", "type": "tool_call",
    }])


def delegate(role):
    return action(f"delegate_{role}", assignment="请完整处理原问题，依据证据与实际工具结果。")


def sql_steps(sql=SQL, selection=None):
    return [action("get_schema", selection=selection or SELECTION),
            action("execute_sql", sql=sql), AIMessage(content="查询完成。")]


def review_steps(candidate_id=1, status="approved", feedback="已核对筛选、聚合和问题范围。"):
    return [action("get_candidate"),
            action("submit_review", candidate_id=candidate_id, status=status, feedback=feedback),
            AIMessage(content="核验结束。")]


def models(**overrides):
    scripts = {
        "supervisor": [delegate("retrieval"), delegate("sql"), delegate("review"), AIMessage(content="查询已完成并经过核验。")],
        "retriever": [action("search_knowledge", query="实收成交额"), AIMessage(content="找到口径资料。")],
        "sql": sql_steps(), "reviewer": review_steps(),
    }
    scripts.update(overrides)
    return {role: ScriptedModel(responses=scripts[role]) for role in ROLES}


class TeamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls.directory.name)
        cls.db = cls.path / "sales.duckdb"
        initialize_demo_database(cls.db)
        cls.notes = cls.path / "notes.md"
        cls.notes.write_text("唯一业务证据：实收成交额只统计 status = '已完成'，使用 quantity * unit_price。", encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def make_team(self, scripted=None, **kwargs):
        scripted = scripted or models()
        return MultiAgentNL2SQL(self.db, llms=scripted, knowledge_files=[self.notes], **kwargs), scripted

    def test_real_four_agent_flow_and_tool_boundaries(self):
        team, scripted = self.make_team()
        with team:
            result = team.ask("统计实收成交额")
        self.assertEqual(result.rows, [(Decimal("15887.00"),)])
        self.assertEqual(result.review["status"], "approved")
        self.assertEqual(result.question, "统计实收成交额")
        self.assertEqual(result.model_calls, 12)
        self.assertEqual(result.calls_by_agent, {"supervisor": 4, "retriever": 2, "sql": 3, "reviewer": 3})
        for role, model in scripted.items():
            self.assertEqual(model.tool_names, ROLE_TOOLS[role])
        self.assertEqual(result.delegations, 3)
        self.assertEqual(result.attempts, 1)
        self.assertTrue(result.retrieval_trace[0]["hits"])

    def test_evidence_is_handed_over_without_copying_other_agents_history(self):
        team, scripted = self.make_team()
        with team:
            result = team.ask("统计实收成交额")
        sql_input = json.loads(scripted["sql"].requests[0][-1].content)
        self.assertIn("唯一业务证据", json.dumps(sql_input["evidence"], ensure_ascii=False))
        self.assertEqual(len(scripted["sql"].requests[0]), 2)  # 只包含自己的 system/user。
        self.assertEqual(len(scripted["reviewer"].requests[0]), 2)
        supervisor_text = "\n".join(str(message.content) for call in scripted["supervisor"].requests for message in call)
        self.assertNotIn("唯一业务证据", supervisor_text)  # 主控只接收摘要与来源 ID。

    def test_missing_review_cannot_be_claimed_success(self):
        scripted = models(supervisor=[delegate("retrieval"), delegate("sql"), AIMessage(content="核验通过，成功。")])
        team, _ = self.make_team(scripted)
        with team, self.assertRaises(TeamExecutionError) as raised:
            team.ask("统计实收成交额")
        self.assertIsNotNone(raised.exception.candidate_sql)
        self.assertIsNone(raised.exception.review)

    def test_review_must_read_real_candidate_and_match_id(self):
        for steps in [
            [action("submit_review", candidate_id=1, status="approved", feedback="未经读取就批准"), AIMessage(content="批准")],
            review_steps(candidate_id=999),
        ]:
            with self.subTest(steps=steps):
                team, _ = self.make_team(models(reviewer=steps))
                with team, self.assertRaises(TeamExecutionError) as raised:
                    team.ask("统计实收成交额")
                self.assertIsNone(raised.exception.review)

    def test_semantic_revision_feedback_reaches_sql_agent(self):
        wrong_sql = "SELECT SUM(quantity * unit_price) AS sales FROM main.orders"
        scripted = models(
            supervisor=[delegate("retrieval"), delegate("sql"), delegate("review"),
                        delegate("sql"), delegate("review"), AIMessage(content="已修订并核验")],
            sql=sql_steps(wrong_sql) + sql_steps(),
            reviewer=review_steps(status="revise", feedback="遗漏已完成状态过滤，请补 status 条件。") + review_steps(2),
        )
        team, _ = self.make_team(scripted)
        with team:
            result = team.ask("统计实收成交额")
        self.assertEqual(result.rows, [(Decimal("15887.00"),)])
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.review["candidate_id"], 2)
        second_task = json.loads(scripted["sql"].requests[3][-1].content)
        self.assertIn("遗漏已完成", second_task["last_review"]["feedback"])

    def test_new_sql_invalidates_old_approval(self):
        scripted = models(
            supervisor=[delegate("retrieval"), delegate("sql"), delegate("review"), delegate("sql"), AIMessage(content="结束")],
            sql=sql_steps() + sql_steps("SELECT COUNT(*) AS count FROM orders"),
        )
        team, _ = self.make_team(scripted)
        with team, self.assertRaises(TeamExecutionError) as raised:
            team.ask("统计实收成交额")
        self.assertIsNone(raised.exception.review)
        self.assertIn("COUNT", raised.exception.candidate_sql)

    def test_failed_new_query_does_not_return_old_success(self):
        scripted = models(
            supervisor=[delegate("retrieval"), delegate("sql"), delegate("review"), delegate("sql"), AIMessage(content="结束")],
            sql=sql_steps() + sql_steps("SELECT missing FROM orders"),
        )
        team, _ = self.make_team(scripted)
        with team, self.assertRaises(TeamExecutionError) as raised:
            team.ask("实收成交额")
        self.assertIsNone(raised.exception.candidate_sql)
        self.assertEqual(raised.exception.attempts, 2)
        self.assertIn("BinderException", raised.exception.history[0]["error"])

    def test_new_evidence_invalidates_old_approval(self):
        scripted = models(supervisor=[delegate("retrieval"), delegate("sql"), delegate("review"),
                                     delegate("retrieval"), AIMessage(content="结束")])
        team, _ = self.make_team(scripted)
        with team, self.assertRaises(TeamExecutionError) as raised:
            team.ask("统计实收成交额")
        self.assertIsNone(raised.exception.review)

    def test_sql_correction_and_shared_budget_across_delegations(self):
        scripted = models(
            supervisor=[delegate("retrieval"), delegate("sql"), delegate("review"),
                        delegate("sql"), AIMessage(content="结束")],
            sql=sql_steps() + sql_steps(),
            reviewer=review_steps(status="revise", feedback="需要重查"),
        )
        team, _ = self.make_team(scripted, max_sql_retries=0)
        with team, self.assertRaises(TeamExecutionError) as raised:
            team.ask("实收成交额")
        self.assertEqual(raised.exception.attempts, 1)
        self.assertIsNone(raised.exception.candidate_sql)
        self.assertEqual(len(scripted["sql"].requests), 3)  # 额度耗尽后未重启子 Agent。

    def test_true_sql_error_can_be_corrected_inside_sql_agent(self):
        scripted = models(sql=[action("get_schema", selection=SELECTION),
                               action("execute_sql", sql="SELECT missing FROM orders"),
                               action("execute_sql", sql=SQL), AIMessage(content="已纠错")],
                          reviewer=review_steps(2))
        team, _ = self.make_team(scripted)
        with team:
            result = team.ask("实收成交额")
        self.assertEqual(result.attempts, 2)
        self.assertIn("BinderException", result.retry_history[0]["error"])

    def test_global_model_budget_is_not_per_agent(self):
        team, scripted = self.make_team(max_model_calls=4)
        with team, self.assertRaises(TeamExecutionError) as raised:
            team.ask("实收成交额")
        self.assertEqual(raised.exception.model_calls, 4)
        self.assertEqual(sum(len(model.requests) for model in scripted.values()), 4)

    def test_delegation_budget_is_bounded(self):
        team, _ = self.make_team(max_delegations=2)
        with team, self.assertRaises(TeamExecutionError) as raised:
            team.ask("实收成交额")
        self.assertIsNone(raised.exception.review)
        self.assertTrue(any("委派额度" in str(event["observation"]) for event in raised.exception.agent_trace))

    def test_retriever_must_really_search(self):
        team, scripted = self.make_team(models(retriever=[AIMessage(content="已经检索好了")]))
        with team, self.assertRaises(TeamExecutionError):
            team.ask("实收成交额")
        self.assertEqual(scripted["sql"].requests, [])

    def test_parallel_sql_calls_cannot_exceed_budget(self):
        parallel = AIMessage(content="", tool_calls=[
            {"name": "execute_sql", "args": {"sql": SQL}, "id": f"parallel-{i}", "type": "tool_call"}
            for i in range(2)
        ])
        team, _ = self.make_team(models(sql=[action("get_schema", selection=SELECTION), parallel,
                                            AIMessage(content="完成")]), max_sql_retries=0)
        with team, self.assertRaises(TeamExecutionError) as raised:
            team.ask("实收成交额")
        self.assertEqual(raised.exception.attempts, 1)
        self.assertIsNone(raised.exception.candidate_sql)

    def test_clarification_is_not_silently_approved(self):
        team, _ = self.make_team(models(reviewer=review_steps(status="clarify", feedback="请确认订单状态口径。")))
        with team, self.assertRaisesRegex(TeamExecutionError, "请确认订单状态"):
            team.ask("实收成交额")

    def test_post_approval_supervisor_failure_preserves_verified_result(self):
        scripted = models()
        scripted["supervisor"].fail_on_call = 4
        team, _ = self.make_team(scripted)
        with team:
            result = team.ask("实收成交额", explain=True)
        self.assertEqual(result.rows, [(Decimal("15887.00"),)])
        self.assertEqual(result.review["status"], "approved")
        self.assertIn("模拟模型故障", result.agent_error)
        self.assertIsNone(result.explanation)
        self.assertTrue(result.explanation_error)

    def test_preview_policy_and_explain_tool(self):
        scripted = models(reviewer=[action("get_candidate"), action("explain_candidate")]
                          + review_steps()[1:])
        team, _ = self.make_team(scripted)
        with team:
            result = team.ask("实收成交额")
        self.assertFalse(any("rows_preview" in event["observation"] for event in result.agent_trace))
        explain = next(event["observation"] for event in result.agent_trace if event["tool"] == "explain_candidate")
        self.assertTrue(explain["ok"])
        self.assertFalse(explain["executed"])
        team, scripted = self.make_team()
        with team:
            result = team.ask("实收成交额", explain=True)
        candidate = next(event["observation"] for event in result.agent_trace if event["tool"] == "get_candidate")
        self.assertEqual(candidate["rows_preview"], [(Decimal("15887.00"),)])
        self.assertTrue(result.explanation)

    def test_read_only_and_external_access_still_blocked(self):
        for sql in ["DELETE FROM orders", "SELECT * FROM read_csv('/etc/passwd')", "SELECT 1; SELECT 2"]:
            with self.subTest(sql=sql):
                team, _ = self.make_team(models(sql=sql_steps(sql)), max_sql_retries=0)
                with team, self.assertRaises(TeamExecutionError) as raised:
                    team.ask("实收成交额")
                self.assertEqual(raised.exception.attempts, 1)
                self.assertIsNone(raised.exception.candidate_sql)

    def test_questions_do_not_share_state_or_budgets(self):
        team, scripted = self.make_team()
        with team:
            first = team.ask("实收成交额")
            second = team.ask("实收成交额")
        self.assertEqual(first.model_calls, second.model_calls)
        self.assertEqual(first.attempts, second.attempts)
        self.assertEqual(len(second.retrieval_trace), 1)
        self.assertEqual(second.review["candidate_id"], 1)

    def test_custom_domain_join_and_output_limit(self):
        db = self.path / "library.duckdb"
        with duckdb.connect(str(db)) as connection:
            connection.execute("CREATE TABLE readers(id INTEGER PRIMARY KEY, city VARCHAR)")
            connection.execute("CREATE TABLE loans(id INTEGER, reader_id INTEGER REFERENCES readers(id))")
            connection.execute("INSERT INTO readers VALUES (1, '北京'), (2, '上海')")
            connection.execute("INSERT INTO loans VALUES (1, 1), (2, 1), (3, 2)")
        sql = "SELECT r.city, COUNT(l.id) AS loans FROM readers r JOIN loans l ON r.id = l.reader_id GROUP BY r.city ORDER BY loans DESC"
        scripted = models(
            retriever=[action("search_knowledge", query="readers loans city"), AIMessage(content="找到元数据")],
            sql=sql_steps(sql, {"main.readers": ["city"], "main.loans": ["id"]}),
        )
        with MultiAgentNL2SQL(db, llms=scripted, max_rows=1) as team:
            self.assertFalse(any(doc.metadata["source"].endswith("sales.json") for doc in team.database.knowledge.documents))
            result = team.ask("按城市统计借阅次数")
        self.assertEqual(result.rows, [("北京", 2)])
        self.assertTrue(result.truncated)
        self.assertTrue(result.join_plan["relationships"])

    def test_cli_architecture_and_retrieval_do_not_load_key(self):
        for arguments in [["--show-architecture"],
                          ["--db", str(self.db), "--knowledge-file", str(self.notes), "--retrieve-only", "-q", "实收成交额"]]:
            output = io.StringIO()
            with patch("nl2sql_handbook.multi_agent_demo.team.load_api_key", side_effect=AssertionError("不应读取密钥")), \
                 patch("nl2sql_handbook.multi_agent_demo.team.ChatDeepSeek", side_effect=AssertionError("不应创建模型")), \
                 redirect_stdout(output):
                status = main(arguments)
            self.assertEqual(status, 0)
            self.assertIsInstance(json.loads(output.getvalue()), dict)

    def test_cli_both_modes_with_real_graphs_and_database(self):
        scripted = models()
        single = ScriptedModel(responses=[action("get_schema", selection=SELECTION),
                                           action("execute_sql", sql=SQL), AIMessage(content="完成")])
        common = ["--db", str(self.db), "--knowledge-file", str(self.notes), "-q", "实收成交额"]
        # CLI 的模型工厂唯一替换点，不读取真实密钥；图与查询逻辑仍实际执行。
        def make_multi(db, **options):
            return MultiAgentNL2SQL(db, llms=scripted, **options)
        for mode in ["multi", "single"]:
            output = io.StringIO()
            with patch("nl2sql_handbook.multi_agent_demo.cli.MultiAgentNL2SQL", side_effect=make_multi), \
                 patch("nl2sql_handbook.nl2sql_agent.load_api_key", return_value="offline"), \
                 patch("nl2sql_handbook.nl2sql_agent.ChatDeepSeek", return_value=single), redirect_stdout(output):
                status = main(["--mode", mode] + common)
            self.assertEqual(status, 0)
            result = json.loads(output.getvalue())
            self.assertEqual(result["rows"], [["15887.00"]])
            self.assertEqual(result["mode"], mode)
            self.assertIn("elapsed_ms", result)

    def test_cli_failure_retains_trace_and_nonzero_exit(self):
        scripted = models(supervisor=[AIMessage(content="假装成功")])
        def make_multi(db, **options):
            return MultiAgentNL2SQL(db, llms=scripted, **options)
        output = io.StringIO()
        with patch("nl2sql_handbook.multi_agent_demo.cli.MultiAgentNL2SQL", side_effect=make_multi), redirect_stdout(output):
            status = main(["--db", str(self.db), "-q", "实收成交额"])
        self.assertEqual(status, 1)
        self.assertFalse(json.loads(output.getvalue())["ok"])
        self.assertIn("agent_trace", json.loads(output.getvalue()))


if __name__ == "__main__":
    unittest.main()
