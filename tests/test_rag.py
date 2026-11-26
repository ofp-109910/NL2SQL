"""离线集成验证：假模型 + 真 LangChain 图/检索器/临时 DuckDB，不读取 API Key。

测试证明证据确实进入模型上下文、查询仍经过原执行限制，不代表真实模型准确率。
"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import duckdb
from langchain_core.documents import Document
from langchain_core.language_models.fake_chat_models import FakeListChatModel, FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from pydantic import Field

from nl2sql_handbook.nl2sql_agent import AgentExecutionError
from nl2sql_handbook.nl2sql_demo import initialize_demo_database
from nl2sql_handbook.rag_demo.agent import RAGAgent
from nl2sql_handbook.rag_demo.cli import main
from nl2sql_handbook.rag_demo.retrieval import KnowledgeIndex, LocalBM25Retriever, load_documents, split_text
from nl2sql_handbook.rag_demo.workflow import RAGWorkflow


SALES_SELECTION = {"main.orders": ["quantity", "unit_price", "status"]}
SELECTION = json.dumps({"tables": SALES_SELECTION, "relationship_ids": []})
SALES_SQL = "SELECT SUM(quantity * unit_price) AS sales FROM main.orders WHERE status = '已完成'"


class RecordingWorkflowModel(FakeListChatModel):
    requests: list[list[BaseMessage]] = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class RecordingAgentModel(FakeMessagesListChatModel):
    requests: list[list[BaseMessage]] = Field(default_factory=list)
    tool_names: list[str] = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        self.tool_names = [item.name for item in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def action(name, args):
    return AIMessage(content="", tool_calls=[{
        "name": name, "args": args, "id": f"call-{name}", "type": "tool_call",
    }])


def agent_responses(sql=SALES_SQL):
    return [action("get_schema", {"selection": SALES_SELECTION}),
            action("execute_sql", {"sql": sql}), AIMessage(content="查询完成。")]


class RAGTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.path = Path(cls.directory.name)
        cls.db = cls.path / "sales.duckdb"
        initialize_demo_database(cls.db)
        cls.notes = cls.path / "notes.json"
        cls.notes.write_text(json.dumps([{
            "id": "custom-sales", "tables": ["main.orders"], "kind": "business_note",
            "text": "试验专用术语：实收成交额使用 quantity * unit_price；只统计 status = '已完成'。",
        }], ensure_ascii=False), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def workflow(self, responses=None, **kwargs):
        model = RecordingWorkflowModel(responses=responses or [SELECTION, SALES_SQL])
        return RAGWorkflow(self.db, llm=model, knowledge_files=[self.notes], **kwargs), model

    def agent(self, responses=None, **kwargs):
        model = RecordingAgentModel(responses=responses or agent_responses())
        return RAGAgent(self.db, llm=model, knowledge_files=[self.notes], **kwargs), model

    def test_bm25_chinese_english_and_no_unrelated_fallback(self):
        documents = [Document(page_content="成交金额 quantity unit_price"),
                     Document(page_content="运动员 athletes medals")]
        retriever = LocalBM25Retriever.build(documents, k=1)
        self.assertEqual(retriever.invoke("成交金额")[0].page_content, documents[0].page_content)
        self.assertEqual(retriever.invoke("ATHLETES")[0].page_content, documents[1].page_content)
        self.assertEqual(retriever.invoke("zzzznevermatches"), [])

    def test_chunking_preserves_overlap_and_end(self):
        chunks = split_text("abcdefghijklm", size=6, overlap=2)
        self.assertEqual(chunks, ["abcdef", "efghij", "ijklm"])

    def test_external_knowledge_rejects_wrong_database_references(self):
        with self.assertRaisesRegex(ValueError, "不存在的表"):
            load_documents({}, [], [self.notes])

    def test_duplicate_knowledge_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "重复"):
            load_documents({"main.orders": []}, [], [self.notes, self.notes])

    def test_context_budget_and_empty_search(self):
        file = self.path / "long.md"
        file.write_text("成交金额 " * 400, encoding="utf-8")
        index = KnowledgeIndex({}, [], self.db, knowledge_files=[file], max_context_chars=200)
        result = index.search("成交金额")
        self.assertEqual(result["context_characters"], 200)
        self.assertTrue(result["hits"][0]["truncated"])
        self.assertEqual(index.search("zzzznevermatches")["hits"], [])
        with self.assertRaises(ValueError):
            index.search(" ")

    def test_workflow_evidence_reaches_selection_generation_and_explanation(self):
        demo, model = self.workflow([SELECTION, SALES_SQL, "本次实收成交额为 15887 元。"])
        with demo:
            result = demo.ask("统计实收成交额", explain=True)
        self.assertEqual(result.question, "统计实收成交额")
        self.assertEqual(result.rows, [(Decimal("15887.00"),)])
        self.assertIn("15887", result.explanation)
        self.assertEqual(len(result.retrieval_trace), 1)
        for messages in model.requests:
            self.assertIn("knowledge:custom-sales", messages[-1].content)
            self.assertIn("试验专用术语", messages[-1].content)
        self.assertIsNone(demo._evidence)

    def test_workflow_correction_keeps_evidence_and_true_error(self):
        demo, model = self.workflow([SELECTION, "SELECT missing FROM orders", SALES_SQL])
        with demo:
            result = demo.ask("实收成交额")
        self.assertEqual(result.attempts, 2)
        self.assertIn("BinderException", result.retry_history[0]["error"])
        self.assertIn("knowledge:custom-sales", model.requests[-1][-1].content)

    def test_workflow_questions_do_not_reuse_evidence(self):
        demo, model = self.workflow([SELECTION, SALES_SQL, SELECTION, SALES_SQL])
        with demo:
            first = demo.ask("实收成交额")
            second = demo.ask("zzzznevermatches")
        self.assertTrue(first.retrieval_trace[0]["hits"])
        self.assertEqual(second.retrieval_trace[0]["hits"], [])
        self.assertNotIn("试验专用术语", model.requests[2][-1].content)

    def test_rag_off_workflow_has_no_evidence_and_skips_knowledge_loading(self):
        demo, model = self.workflow(rag_enabled=False)
        with demo:
            result = demo.ask("实收成交额")
        self.assertFalse(result.rag_enabled)
        self.assertEqual(result.retrieval_trace, [])
        self.assertFalse(any("试验专用术语" in msg.content for call in model.requests for msg in call))

    def test_agent_get_schema_includes_evidence_before_execution(self):
        agent, model = self.agent()
        with agent:
            result = agent.ask("实收成交额")
        self.assertEqual(result.rows, [(Decimal("15887.00"),)])
        self.assertEqual(result.attempts, 1)
        self.assertIn("search_knowledge", model.tool_names)
        schema_observation = json.loads(next(msg.content for msg in model.requests[1] if isinstance(msg, ToolMessage)))
        self.assertTrue(schema_observation["retrieval"]["hits"])
        self.assertIn("试验专用术语", json.dumps(schema_observation, ensure_ascii=False))
        self.assertEqual(result.retrieval_trace[0]["query"], "实收成交额")
        self.assertNotIn("rows_preview", result.tool_trace[-1]["observation"])

    def test_agent_can_search_with_its_own_query(self):
        agent, model = self.agent([action("search_knowledge", {"query": "实收成交额"})] + agent_responses())
        with agent:
            result = agent.ask("统计成交额")
        self.assertEqual(len(result.retrieval_trace), 2)
        self.assertEqual(result.retrieval_trace[0]["query"], "实收成交额")
        self.assertEqual(result.retrieval_trace[1]["query"], "统计成交额")
        self.assertEqual(result.tool_trace[0]["tool"], "search_knowledge")

    def test_agent_active_search_is_bounded(self):
        searches = [action("search_knowledge", {"query": "实收成交额"}) for _ in range(5)]
        agent, model = self.agent(searches + agent_responses())
        with agent:
            result = agent.ask("实收成交额")
        self.assertFalse(result.tool_trace[4]["observation"]["ok"])
        self.assertIn("额度", result.tool_trace[4]["observation"]["error"])
        self.assertEqual(result.attempts, 1)

    def test_rag_off_agent_keeps_original_tools(self):
        agent, model = self.agent(rag_enabled=False)
        with agent:
            result = agent.ask("实收成交额")
        self.assertEqual(model.tool_names, ["list_tables", "get_schema", "execute_sql"])
        self.assertEqual(result.retrieval_trace, [])
        self.assertFalse(result.rag_enabled)

    def test_agent_questions_do_not_share_retrieval_history(self):
        agent, model = self.agent(agent_responses() + agent_responses())
        with agent:
            first = agent.ask("实收成交额")
            second = agent.ask("zzzznevermatches")
        self.assertTrue(first.retrieval_trace[0]["hits"])
        self.assertEqual(second.retrieval_trace[0]["hits"], [])
        self.assertEqual(len(second.retrieval_trace), 1)

    def test_both_versions_still_reject_mutations(self):
        demo, model = self.workflow([SELECTION, "DELETE FROM orders"])
        with demo, self.assertRaises(ValueError):
            demo.ask("实收成交额")
        agent, model = self.agent(agent_responses("DELETE FROM orders"), max_sql_retries=0)
        with agent, self.assertRaises(AgentExecutionError):
            agent.ask("实收成交额")

    def test_custom_database_does_not_load_sales_notes_and_supports_join(self):
        # 第二个领域无需修改 RAG 核心。资料来自新库的真实元数据与单独文件。
        db, note = self.path / "library.duckdb", self.path / "library.txt"
        with duckdb.connect(str(db)) as connection:
            connection.execute("CREATE TABLE readers(id INTEGER PRIMARY KEY, city VARCHAR)")
            connection.execute("CREATE TABLE loans(id INTEGER, reader_id INTEGER REFERENCES readers(id))")
            connection.execute("INSERT INTO readers VALUES (1, '北京'), (2, '上海')")
            connection.execute("INSERT INTO loans VALUES (1, 1), (2, 1), (3, 2)")
        note.write_text("借阅热度指 loans 的借阅记录数，读者城市在 readers.city。", encoding="utf-8")
        select = {"tables": {"main.readers": ["city"], "main.loans": ["id"]}, "relationship_ids": []}
        sql = "SELECT r.city, COUNT(l.id) AS loans FROM readers r JOIN loans l ON r.id = l.reader_id GROUP BY r.city ORDER BY loans DESC"
        default = RecordingWorkflowModel(responses=[json.dumps(select), sql])
        with RAGWorkflow(db, llm=default) as demo:
            self.assertFalse(any(doc.metadata["source"].endswith("sales.json") for doc in demo.knowledge.documents))
        for factory, model in [
            (RAGWorkflow, RecordingWorkflowModel(responses=[json.dumps(select), sql])),
            (RAGAgent, RecordingAgentModel(responses=[
                action("get_schema", {"selection": select["tables"]}),
                action("execute_sql", {"sql": sql}), AIMessage(content="完成"),
            ])),
        ]:
            with factory(db, llm=model, knowledge_files=[note]) as demo:
                result = demo.ask("按读者城市统计借阅热度")
            self.assertEqual(result.rows, [("北京", 2), ("上海", 1)])
            self.assertTrue(result.retrieval_trace[0]["hits"])
            self.assertTrue(result.join_plan["relationships"])

    def test_cli_retrieve_only_never_constructs_model(self):
        with patch("nl2sql_handbook.nl2sql_demo.ChatDeepSeek", side_effect=AssertionError("不应创建模型")), \
             patch("nl2sql_handbook.nl2sql_agent.ChatDeepSeek", side_effect=AssertionError("不应创建模型")):
            output = io.StringIO()
            with redirect_stdout(output):
                status = main(["--db", str(self.db), "--knowledge-file", str(self.notes),
                               "--retrieve-only", "-q", "实收成交额"])
        self.assertEqual(status, 0)
        self.assertTrue(json.loads(output.getvalue())["hits"])

    def test_cli_both_modes_return_serializable_real_results(self):
        for mode, model in [
            ("workflow", RecordingWorkflowModel(responses=[SELECTION, SALES_SQL])),
            ("agent", RecordingAgentModel(responses=agent_responses())),
        ]:
            # 只替换模型与密钥读取；CLI、检索器、Agent 图、SQL 执行均走真实路径。
            with self.subTest(mode=mode), \
                 patch("nl2sql_handbook.nl2sql_demo.load_api_key", return_value="offline-test"), \
                 patch("nl2sql_handbook.nl2sql_agent.load_api_key", return_value="offline-test"), \
                 patch("nl2sql_handbook.nl2sql_demo.ChatDeepSeek", return_value=model), \
                 patch("nl2sql_handbook.nl2sql_agent.ChatDeepSeek", return_value=model):
                output = io.StringIO()
                with redirect_stdout(output):
                    status = main(["--mode", mode, "--db", str(self.db), "--knowledge-file", str(self.notes),
                                   "-q", "实收成交额"])
                self.assertEqual(status, 0)
                result = json.loads(output.getvalue())
                self.assertEqual(result["rows"], [["15887.00"]])
                self.assertTrue(result["rag_enabled"])
                self.assertTrue(result["retrieval_trace"][0]["hits"])


if __name__ == "__main__":
    unittest.main()
