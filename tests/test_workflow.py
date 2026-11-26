"""离线检查关键失败边界；假模型不用于报告真实准确率。"""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import duckdb
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from nl2sql_handbook.evaluation_demo.dataset import prepare, cases
from nl2sql_handbook.evaluation_demo.grading import grade
from nl2sql_handbook.workflow_v2.runtime import query
from nl2sql_handbook.workflow_v2.workflow import ImprovedWorkflow, ImprovedJoinWorkflow, ImprovedRAGWorkflow, WorkflowFailed


def selection(tables, requirements="按照原问题查询", **kwargs):
    return json.dumps(dict(tables=tables, requirements=requirements, **kwargs), ensure_ascii=False)


def review(status="approved", feedback="与题意一致"):
    return json.dumps(dict(status=status, feedback=feedback), ensure_ascii=False)


class Recorder(BaseCallbackHandler):
    def __init__(self):
        self.messages = []

    def on_chat_model_start(self, serialized, messages, **kwargs):
        self.messages.append(messages[0])


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        prepare(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def demo(self, responses, *, factory=ImprovedWorkflow, domain="sales", **kwargs):
        return factory(self.root/f"{domain}_0.duckdb", llm=FakeListChatModel(responses=responses), **kwargs)

    def test_missing_foreign_key_is_completed_from_metadata(self):
        case = next(c for c in cases() if c.id == "school-09")
        selected = {"main.enrollments": ["enrollment_id", "score", "term"], "main.students": ["grade"]}
        with self.demo([selection(selected), case.sql, review()], domain="school") as demo:
            result = demo.ask(case.question)
        self.assertIn("student_id", result.selected_schema["main.enrollments"])
        self.assertTrue(grade(case, result.sql, result.__dict__, self.root)["passed"])

    def test_executable_wrong_join_is_rejected_before_review(self):
        selected = {"main.enrollments": ["enrollment_id"], "main.students": ["grade"]}
        wrong = "SELECT s.grade FROM enrollments e JOIN students s ON e.enrollment_id=s.student_id"
        correct = "SELECT s.grade FROM enrollments e JOIN students s ON e.student_id=s.student_id"
        with self.demo([selection(selected), wrong, correct, review()], domain="school") as demo:
            result = demo.ask("查询选课学生年级")
        self.assertEqual(2, result.attempts)
        self.assertEqual("sql_error", result.workflow_trace[1]["stage"])

    def test_semantic_feedback_corrects_executable_sql(self):
        selected = {"main.orders": ["order_id", "status"]}
        wrong, correct = "SELECT COUNT(*) FROM orders", "SELECT COUNT(*) FROM orders WHERE status='已完成'"
        with self.demo([selection(selected), wrong, review("revise", "缺少status已完成过滤"), correct, review()]) as demo:
            result = demo.ask("已完成订单有多少笔？")
        self.assertEqual(correct, result.sql)
        self.assertEqual(2, result.attempts)
        self.assertEqual("semantic_review", result.retry_history[0]["stage"])

    def test_reselection_adds_missing_structure_and_refreshes_rag(self):
        first = {"main.orders": ["order_id"]}
        second = {"main.orders": ["quantity", "unit_price", "status"]}
        with self.demo([selection(first), "SELECT COUNT(*) FROM orders", review("reselect", "需要数量、历史单价及订单状态"),
                        selection(second), "SELECT SUM(quantity*unit_price) FROM orders WHERE status='已完成'", review()],
                       factory=ImprovedRAGWorkflow, knowledge_files=[self.root/"sales_knowledge.json"]) as demo:
            result = demo.ask("统计已完成订单销售额")
        self.assertEqual(2, result.selection_attempts)
        self.assertEqual(2, len(result.retrieval_trace))
        self.assertIsNone(demo._evidence)

    def test_invalid_selection_gets_one_bounded_retry(self):
        with self.demo([selection({"main.invented": ["id"]}), selection({"main.orders": ["order_id"]}),
                        "SELECT COUNT(*) FROM orders", review()]) as demo:
            result = demo.ask("订单数")
        self.assertEqual(2, result.selection_attempts)

    def test_semantic_rejection_cannot_silently_return_final_sql(self):
        selected = selection({"main.orders": ["order_id"]})
        responses = [selected] + [x for _ in range(4) for x in ["SELECT COUNT(*) FROM orders", review("revise", "仍不符合题意")]]
        with self.demo(responses) as demo:
            with self.assertRaises(WorkflowFailed) as caught:
                demo.ask("查询")
        self.assertEqual(4, caught.exception.attempts)
        self.assertEqual(4, len(caught.exception.history))

    def test_invalid_review_is_not_approval(self):
        with self.demo([selection({"main.orders": ["order_id"]}), "SELECT COUNT(*) FROM orders", "{}"], max_sql_retries=0) as demo:
            with self.assertRaises(WorkflowFailed):
                demo.ask("订单数")

    def test_second_reselection_request_exhausts_budget(self):
        chosen = selection({"main.orders": ["order_id"]})
        with self.demo([chosen, "SELECT COUNT(*) FROM orders", review("reselect", "补充结构"), chosen,
                        "SELECT COUNT(*) FROM orders", review("reselect", "继续补充")]) as demo:
            with self.assertRaises(WorkflowFailed) as caught:
                demo.ask("订单数")
        self.assertEqual(2, sum(t["stage"] == "selection" for t in caught.exception.tool_trace))

    def test_write_sql_blocked_and_database_preserved(self):
        path = self.root/"sales_0.duckdb"
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.demo([selection({"main.orders": ["order_id"]}), "DELETE FROM orders"]) as demo:
            with self.assertRaises(ValueError):
                demo.ask("删除订单")
        self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_state_is_reset_between_questions(self):
        chosen = selection({"main.orders": ["order_id"]})
        with self.demo([chosen, "SELECT COUNT(*) FROM orders", review(), chosen, "SELECT COUNT(*) FROM orders", review()],
                       factory=ImprovedRAGWorkflow, knowledge_files=[]) as demo:
            first, second = demo.ask("订单数"), demo.ask("统计订单总数")
        self.assertEqual(1, second.attempts)
        self.assertEqual(1, second.selection_attempts)
        self.assertEqual(1, len(first.retrieval_trace))
        self.assertEqual(1, len(second.retrieval_trace))

    def test_all_new_adapters_use_real_graph_and_meter(self):
        for version in ["workflow_v2", "join_workflow_v2", "rag_workflow_v2"]:
            model = FakeListChatModel(responses=[selection({"main.orders": ["order_id"]}), "SELECT COUNT(*) FROM orders", review()])
            value = query(version, self.root/"sales_0.duckdb", "订单数", llm=model)
            self.assertTrue(value["ok"], value)
            self.assertEqual(3, value["model_calls"])

    def test_ambiguous_roles_are_not_arbitrarily_chosen(self):
        path = self.root/"roles.duckdb"
        with duckdb.connect(str(path)) as db:
            db.execute("CREATE TABLE people(id INTEGER PRIMARY KEY); CREATE TABLE deals(id INTEGER PRIMARY KEY,buyer INTEGER REFERENCES people(id),seller INTEGER REFERENCES people(id));")
        chosen = selection({"main.people": ["id"], "main.deals": ["id"]})
        with ImprovedWorkflow(path, llm=FakeListChatModel(responses=[chosen, chosen])) as demo:
            with self.assertRaises(WorkflowFailed):
                demo.ask("查询交易参与人")

    def test_generating_and_reviewing_receive_only_selected_schema(self):
        recorder = Recorder()
        model = FakeListChatModel(responses=[selection({"main.orders": ["order_id"]}), "SELECT COUNT(*) FROM orders", review()], callbacks=[recorder])
        with ImprovedWorkflow(self.root/"sales_0.duckdb", llm=model) as demo:
            demo.ask("订单数")
        all_selection = "\n".join(m.content for m in recorder.messages[0])
        generation = "\n".join(m.content for m in recorder.messages[1])
        self.assertIn("main.products", all_selection)
        self.assertNotIn("main.products", generation)


if __name__ == "__main__":
    unittest.main()
