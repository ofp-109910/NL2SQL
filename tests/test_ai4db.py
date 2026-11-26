"""使用真实 DuckDB 测执行边界；模型替身仅用于不消耗 API 的流程测试。"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import duckdb
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from nl2sql_handbook.ai4db_demo.cli import main
from nl2sql_handbook.ai4db_demo.engine import Budget, Database, Limits, Snapshot, adoption_reason, benchmark, summarize
from nl2sql_handbook.ai4db_demo.fixture import ROOT, create_fixture
from nl2sql_handbook.ai4db_demo.optimizer import Experiment

ORIGINAL = (ROOT / "workloads/original.sql").read_text(encoding="utf-8")
MERGED = (ROOT / "workloads/merged.sql").read_text(encoding="utf-8")


class AI4DBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.db, cls.edge = cls.root / "main.duckdb", cls.root / "edge.duckdb"
        create_fixture(cls.db, 1000)
        create_fixture(cls.edge, edge=True)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.limits = Limits(warmups=0, repetitions=3)

    def experiment(self, **options):
        return Experiment(self.db, validation_dbs=[self.edge], limits=self.limits, **options)

    def database(self, **options):
        return Database(self.db, options.get("limits", self.limits), options.get("budget", Budget()))

    def fake(self, *outputs):
        prompts = []
        def respond(prompt):
            prompts.append(json.loads(prompt.to_messages()[-1].content))
            output = outputs[min(len(prompts) - 1, len(outputs) - 1)]
            return AIMessage(content=json.dumps(output, ensure_ascii=False),
                             response_metadata={"model_name": "fake"})
        return RunnableLambda(respond), prompts

    def test_fixture_never_overwrites(self):
        before = self.db.read_bytes()
        self.assertFalse(create_fixture(self.db, 2000))
        self.assertEqual(before, self.db.read_bytes())

    def test_valid_merge_equal_on_both_real_databases(self):
        with self.experiment() as experiment:
            report = experiment.run(ORIGINAL, candidate=MERGED)
        attempt = report["attempts"][0]
        self.assertTrue(all(check["equal"] for check in attempt["checks"]))
        self.assertEqual(2, len(attempt["checks"]))
        self.assertEqual(3, len(attempt["measurement"]["pairs"]))
        self.assertEqual(0, report["model_calls"])

    def test_schema_is_strict(self):
        result = Snapshot([("x", "INTEGER")], [(1,)])
        self.assertFalse(result.compare(Snapshot([("x", "BIGINT")], [(1,)]), False)["equal"])
        self.assertFalse(result.compare(Snapshot([("y", "INTEGER")], [(1,)]), False)["equal"])

    def test_null_duplicates_and_order(self):
        a = Snapshot([("x", "INTEGER")], [(None,), (1,), (1,)])
        b = Snapshot(a.columns, [(1,), (None,), (1,)])
        self.assertTrue(a.compare(b, False)["equal"])
        self.assertFalse(a.compare(b, True)["equal"])
        self.assertFalse(a.compare(Snapshot(a.columns, [(None,), (1,)]), False)["equal"])

    def test_nested_values_not_stringified(self):
        a = Snapshot([("x", "STRUCT")], [({"a": [1, None], "b": float("nan")},)])
        b = Snapshot(a.columns, [({"b": float("nan"), "a": [1, None]},)])
        self.assertTrue(a.compare(b, False)["equal"])
        self.assertFalse(Snapshot(a.columns, [(True,)]).compare(Snapshot(a.columns, [(1,)]), False)["equal"])

    def test_wrong_null_join_rejected_before_benchmark(self):
        wrong = ORIGINAL.replace("s.city IS NOT DISTINCT FROM n.city", "s.city = n.city")
        with self.experiment() as experiment:
            report = experiment.run(ORIGINAL, candidate=wrong)
        self.assertFalse(report["adopted"])
        self.assertEqual(report["original_sql"], report["selected_sql"])
        self.assertNotIn("measurement", report["attempts"][0])

    def test_edge_database_catches_distinct_error_main_does_not(self):
        original = "SELECT SUM(quantity) AS total FROM orders WHERE customer_id=1 AND status='已完成'"
        wrong = original.replace("SUM(quantity)", "SUM(DISTINCT quantity)")
        # 主库 customer_id=1 的行全是已取消，二者都得到 NULL；边界库有重复的已完成订单。
        with self.experiment() as experiment:
            report = experiment.run(original, candidate=wrong)
        checks = report["attempts"][0]["checks"]
        self.assertTrue(checks[0]["equal"])
        self.assertFalse(checks[1]["equal"])
        self.assertFalse(report["adopted"])

    def test_result_cap_is_inconclusive(self):
        with Experiment(self.db, limits=Limits(max_rows=1, repetitions=3)) as experiment:
            report = experiment.run("SELECT COUNT(*) AS n FROM orders",
                                    candidate="SELECT COUNT(*) AS n FROM orders GROUP BY customer_id")
        self.assertFalse(report["adopted"])
        self.assertIn("超过比较上限", report["attempts"][0]["error"])

    def test_blocks_write_and_multiple_statements(self):
        database = self.database()
        try:
            for sql in ["DELETE FROM orders", "SELECT 1; SELECT 2", "PRAGMA threads=4",
                        "CREATE TABLE x AS SELECT 1", "EXPLAIN SELECT 1"]:
                with self.subTest(sql=sql), self.assertRaises(ValueError):
                    database.inspect(sql)
        finally:
            database.close()

    def test_external_files_disabled_on_connection(self):
        database = self.database()
        try:
            with self.assertRaises(duckdb.Error):
                database.snapshot("SELECT * FROM read_csv('/etc/passwd')")
            with self.assertRaises(duckdb.Error):
                database.connection.execute("DELETE FROM orders")
        finally:
            database.close()

    def test_nondeterminism_and_unordered_limit_rejected(self):
        database = self.database()
        try:
            for sql in ["SELECT random()", "SELECT current_timestamp", "SELECT uuid()",
                        "SELECT * FROM orders LIMIT 3",
                        "WITH x AS (SELECT * FROM orders LIMIT 2) SELECT * FROM x",
                        "SELECT * FROM orders USING SAMPLE 10%"]:
                with self.subTest(sql=sql), self.assertRaises(ValueError):
                    database.inspect(sql)
            self.assertTrue(database.inspect("SELECT order_id FROM orders ORDER BY order_id LIMIT 2")[1])
        finally:
            database.close()

    def test_profile_has_real_operator_evidence(self):
        database = self.database()
        try:
            profile = database.profile("SELECT SUM(quantity) FROM orders")
            self.assertTrue(profile["physical_plan"])
            self.assertGreater(profile["summary"]["rows_scanned"], 0)
            self.assertTrue(profile["summary"]["operators_by_time"])
        finally:
            database.close()

    def test_actual_measurements_alternate_and_count(self):
        database = self.database()
        try:
            measured = benchmark(database, "SELECT SUM(quantity) FROM orders", "SELECT SUM(quantity) FROM orders")
            self.assertEqual(6, database.budget.used)
            self.assertEqual(["candidate", "original"], measured["pairs"][1]["order"])
            self.assertTrue(all(p["original_ms"] > 0 and p["candidate_ms"] > 0 for p in measured["pairs"]))
        finally:
            database.close()

    def test_adoption_requires_all_thresholds(self):
        good = summarize([{"original_ms": 10, "candidate_ms": 5}] * 7)
        self.assertTrue(adoption_reason(good, self.limits, same_plan=False)[0])
        for changes, same in [({}, True), ({"gain_fraction": 0.01}, False),
                              ({"saving_ms": 0.01}, False), ({"win_fraction": 0.5}, False)]:
            self.assertFalse(adoption_reason({**good, **changes}, self.limits, same_plan=same)[0])

    def test_identical_physical_plan_keeps_original(self):
        with self.experiment() as experiment:
            report = experiment.run("SELECT SUM(quantity) AS n FROM orders",
                                    candidate="SELECT SUM(quantity) AS n FROM orders")
        self.assertFalse(report["adopted"])
        self.assertIn("物理计划相同", report["reason"])

    def test_timeout_recovers_for_next_query(self):
        database = self.database(limits=Limits(timeout=0.02, repetitions=3))
        try:
            with self.assertRaises(duckdb.InterruptException):
                database.snapshot("SELECT SUM(a.i * b.i) FROM range(1000000) a(i), range(1000000) b(i)")
            self.assertEqual([(1,)], database.snapshot("SELECT 1").rows)
        finally:
            database.close()

    def test_shared_execution_budget(self):
        budget = Budget(maximum=1)
        database = self.database(budget=budget)
        try:
            database.snapshot("SELECT 1")
            with self.assertRaisesRegex(RuntimeError, "预算耗尽"):
                database.snapshot("SELECT 2")
        finally:
            database.close()

    def test_model_gets_plan_and_feedback_with_bounded_retries(self):
        fake, prompts = self.fake({"candidate_sql": MERGED.replace("已完成", "已取消")},
                                  {"diagnosis": "合并扫描", "candidate_sql": MERGED})
        with self.experiment(llm=fake) as experiment:
            report = experiment.run(ORIGINAL)
        self.assertEqual(2, report["model_calls"])
        self.assertTrue(prompts[0]["original_physical_plan"])
        self.assertIn("main.orders", prompts[0]["schema"])
        self.assertNotIn("rows", prompts[0])
        self.assertIn("结果不一致", prompts[1]["previous_feedback"][0]["reason"])
        self.assertTrue(all(c["equal"] for c in report["attempts"][1]["checks"]))

    def test_bad_json_cannot_bypass_proposal_budget(self):
        fake = RunnableLambda(lambda _: AIMessage(content="不是 JSON"))
        with self.experiment(llm=fake, max_proposals=2) as experiment:
            report = experiment.run(ORIGINAL)
        self.assertEqual(2, report["model_calls"])
        self.assertFalse(report["adopted"])
        self.assertTrue(all("error" in a for a in report["attempts"]))

    def test_model_can_decline(self):
        fake, _ = self.fake({"candidate_sql": None, "diagnosis": "无可靠改写"})
        with self.experiment(llm=fake) as experiment:
            report = experiment.run(ORIGINAL)
        self.assertEqual(1, report["model_calls"])
        self.assertFalse(report["adopted"])

    def test_candidate_sql_error_is_reported_then_retried(self):
        fake, prompts = self.fake({"candidate_sql": "SELECT no_such_column FROM orders"}, {"candidate_sql": None})
        with self.experiment(llm=fake) as experiment:
            report = experiment.run(ORIGINAL)
        self.assertEqual(2, report["model_calls"])
        self.assertIn("no_such_column", prompts[1]["previous_feedback"][0]["error"])
        self.assertFalse(report["adopted"])

    def test_cli_analyze_without_key_and_result_handoff(self):
        result = self.root / "previous.json"
        result.write_text(json.dumps({"sql": "SELECT COUNT(*) AS n FROM orders", "question": "数量", "rows": [[-999]]}), encoding="utf-8")
        output = self.root / "reports"
        with patch("nl2sql_handbook.ai4db_demo.optimizer.load_api_key", side_effect=AssertionError("不应读取密钥")), patch("builtins.print"):
            code = main(["--db", str(self.db), "--result-file", str(result), "--analyze-only",
                         "--repetitions", "3", "--report-dir", str(output)])
        self.assertEqual(0, code)
        report_file = next(output.glob("*/report.json"))
        report = json.loads(report_file.read_text(encoding="utf-8"))
        self.assertEqual(1, report["original_result"]["rows"])
        self.assertEqual("数量", report["question"])
        self.assertEqual(0, report["model_calls"])
        self.assertTrue((report_file.parent / "selected.sql").exists())
        self.assertTrue((report_file.parent / "REPORT.md").exists())

    def test_offline_candidate_never_reads_key(self):
        with patch("nl2sql_handbook.ai4db_demo.optimizer.load_api_key", side_effect=AssertionError("不应读取密钥")):
            with self.experiment() as experiment:
                report = experiment.run(ORIGINAL, candidate=MERGED)
        self.assertEqual(0, report["model_calls"])

    def test_other_domain_and_empty_result_are_supported(self):
        library = self.root / "library.duckdb"
        with duckdb.connect(str(library)) as connection:
            connection.execute("CREATE TABLE books(id INTEGER, pages INTEGER); INSERT INTO books VALUES (1,20),(2,NULL),(3,20)")
        fake, prompts = self.fake({"candidate_sql": "SELECT id FROM books WHERE id < 0 ORDER BY id"})
        with Experiment(library, llm=fake, limits=self.limits) as experiment:
            report = experiment.run("SELECT id FROM books WHERE id < 0 ORDER BY id")
        self.assertEqual(0, report["original_result"]["rows"])
        self.assertTrue(report["attempts"][0]["checks"][0]["equal"])
        self.assertIn("main.books", prompts[0]["schema"])
        self.assertNotIn("main.orders", prompts[0]["schema"])

    def test_model_failure_keeps_verified_original_and_records_error(self):
        def failure(_):
            raise RuntimeError("synthetic provider failure sk-SECRET123")
        with self.experiment(llm=RunnableLambda(failure), max_proposals=1) as experiment:
            report = experiment.run(ORIGINAL)
        self.assertEqual(1, report["model_calls"])
        self.assertFalse(report["adopted"])
        self.assertEqual(report["original_sql"], report["selected_sql"])
        self.assertNotIn("SECRET123", report["attempts"][0]["error"])

    def test_correct_but_unprofitable_keeps_original(self):
        # 只替换测量返回值，正确性验证和 profile 仍使用真实 DuckDB。
        loss = summarize([{"original_ms": 5, "candidate_ms": 10}] * 3)
        with patch("nl2sql_handbook.ai4db_demo.optimizer.benchmark", return_value=loss):
            with self.experiment() as experiment:
                report = experiment.run(ORIGINAL, candidate=MERGED)
        self.assertTrue(all(check["equal"] for check in report["attempts"][0]["checks"]))
        self.assertFalse(report["adopted"])
        self.assertEqual(report["original_sql"], report["selected_sql"])


if __name__ == "__main__":
    unittest.main()
