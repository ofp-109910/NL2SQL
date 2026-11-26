"""查询示例回归用例：全部离线，使用临时数据库，不读取 API Key。"""
import importlib.util
import json
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import duckdb
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from nl2sql_handbook.nl2sql_demo import initialize_demo_database, connect_read_only, read_schema
from nl2sql_handbook.nl2sql_workflow_join import (
    JoinCatalog, JoinPlanningError, JoinValidationError, validate_join_sql,
    JoinNL2SQLWorkflow,
)
from nl2sql_handbook.nl2sql_agent_join import JoinNL2SQLAgent
from nl2sql_handbook.offline_examples import JOIN_SELECTION, JOIN_SQL, EXPECTED_JOIN, SALES_SQL, run_all
from nl2sql_handbook.result_compare import same_result
from nl2sql_handbook.testing import ToolCallingModel, action


class ExamplesTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "test.duckdb"
        initialize_demo_database(self.db)
        self.connection = connect_read_only(self.db)
        self.schema = json.loads(read_schema(self.connection))
        self.catalog = JoinCatalog(self.connection, self.schema)

    def tearDown(self):
        self.connection.close()
        self.tmp.cleanup()

    def test_minimal_actual_result(self):
        from nl2sql_handbook import minimal as mod
        result = mod.ask_once(self.connection, FakeListChatModel(responses=[SALES_SQL]), mod.QUESTION)
        self.assertEqual(result.rows, [(Decimal("15887.00"),)])

    def test_four_complete_offline_paths(self):
        self.assertEqual(len(run_all(self.db)["results"]), 4)

    def test_reference_totals_and_city_sales(self):
        self.assertEqual(self.connection.execute(
            "SELECT COUNT(*), COUNT(*) FILTER (WHERE status='已完成') FROM orders"
        ).fetchone(), (10, 8))
        self.assertEqual(self.connection.execute(JOIN_SQL).fetchall(), EXPECTED_JOIN)

    def test_bridge_adds_keys_not_missing_business_fields(self):
        plan = self.catalog.plan({"main.customers": ["city"], "main.products": ["category"]})
        self.assertEqual(plan.bridge_tables, ["main.orders"])
        cols = {c["column"] for c in plan.schema["main.orders"]}
        self.assertEqual(cols, {"customer_id", "product_id"})
        self.assertNotIn("unit_price", cols)

    def test_join_workflow_retries_wrong_join(self):
        bad = JOIN_SQL.replace("o.customer_id = c.customer_id", "o.product_id = c.customer_id")
        model = FakeListChatModel(responses=[json.dumps({"tables": JOIN_SELECTION}), bad, JOIN_SQL])
        with JoinNL2SQLWorkflow(self.db, llm=model) as demo:
            result = demo.ask("按城市和商品类别统计已完成订单销售额")
        self.assertEqual(result.rows, EXPECTED_JOIN)
        self.assertEqual(result.attempts, 2)
        self.assertIn("连接键", result.retry_history[0]["error"])

    def test_join_agent_counts_validation_failure(self):
        bad = JOIN_SQL.replace("o.customer_id = c.customer_id", "o.product_id = c.customer_id")
        model = ToolCallingModel(responses=[action("get_schema", {"selection": JOIN_SELECTION}),
            action("execute_sql", {"sql": bad}), action("execute_sql", {"sql": JOIN_SQL}),
            AIMessage(content="完成")])
        with JoinNL2SQLAgent(self.db, llm=model) as demo:
            result = demo.ask("按城市和商品类别统计已完成订单销售额")
        self.assertEqual(result.attempts, 2)
        self.assertEqual(result.rows, EXPECTED_JOIN)

    def test_rejects_cross_join(self):
        with self.assertRaises(JoinValidationError):
            validate_join_sql(self.connection, "SELECT * FROM orders CROSS JOIN customers",
                              self.schema, list(self.catalog.relationships.values()))

    def test_composite_keys(self):
        with duckdb.connect() as c:
            c.execute("""CREATE TABLE parents(tenant_id INT, item_id INT,
                PRIMARY KEY(tenant_id,item_id));
                CREATE TABLE children(child_id INT, tenant_id INT, item_id INT,
                FOREIGN KEY(tenant_id,item_id) REFERENCES parents(tenant_id,item_id));""")
            schema = json.loads(read_schema(c)); catalog = JoinCatalog(c, schema)
            relations = list(catalog.relationships.values())
            bad = "SELECT * FROM children c JOIN parents p ON c.item_id=p.item_id"
            with self.assertRaises(JoinValidationError):
                validate_join_sql(c, bad, schema, relations)
            validate_join_sql(c, bad + " AND c.tenant_id=p.tenant_id", schema, relations)

    def test_ambiguous_roles_need_explicit_relation(self):
        with duckdb.connect() as c:
            c.execute("""CREATE TABLE people(id INT PRIMARY KEY, name VARCHAR);
                CREATE TABLE tasks(id INT, author_id INT REFERENCES people(id),
                reviewer_id INT REFERENCES people(id));""")
            schema = json.loads(read_schema(c)); catalog = JoinCatalog(c, schema)
            selected = {"main.tasks": ["id"], "main.people": ["name"]}
            with self.assertRaises(JoinPlanningError):
                catalog.plan(selected)
            rid = next(r["id"] for r in catalog.relationships.values()
                       if "author_id" in r["left_columns"])
            self.assertEqual(len(catalog.plan(selected, [rid]).relationships), 1)

    def test_left_join_filter_location(self):
        with duckdb.connect() as c:
            c.execute("""CREATE TABLE customers(id INT, name VARCHAR);
                INSERT INTO customers VALUES (1,'甲'),(2,'乙'),(3,'丙');
                CREATE TABLE orders(customer_id INT, amount INT, status VARCHAR);
                INSERT INTO orders VALUES (1,100,'已完成'),(2,50,'已取消');""")
            prefix = "SELECT c.name, COALESCE(SUM(o.amount),0) FROM customers c LEFT JOIN orders o ON c.id=o.customer_id"
            suffix = " GROUP BY c.name ORDER BY c.name"
            bad = c.execute(prefix + " WHERE o.status='已完成'" + suffix).fetchall()
            good = c.execute(prefix + " AND o.status='已完成'" + suffix).fetchall()
            self.assertEqual(bad, [('甲',100)])
            self.assertEqual(dict(good), {'甲':100,'乙':0,'丙':0})

    def test_fanout_and_preaggregation(self):
        with duckdb.connect() as c:
            c.execute("""CREATE TABLE orders(customer_id INT, amount INT);
                INSERT INTO orders VALUES (1,100),(1,200);
                CREATE TABLE refunds(customer_id INT, amount INT);
                INSERT INTO refunds VALUES (1,10),(1,20),(1,30);""")
            bad = c.execute("SELECT SUM(o.amount), SUM(r.amount) FROM orders o JOIN refunds r USING(customer_id)").fetchone()
            good = c.execute("""WITH os AS (SELECT customer_id,SUM(amount) sales FROM orders GROUP BY customer_id),
                rs AS (SELECT customer_id,SUM(amount) refunded FROM refunds GROUP BY customer_id)
                SELECT sales,refunded,sales-refunded FROM os JOIN rs USING(customer_id)""").fetchone()
            self.assertEqual(bad, (900,120))
            self.assertEqual(good, (300,60,240))

    def test_empty_sum_is_null(self):
        self.assertEqual(self.connection.execute(
            "SELECT SUM(quantity*unit_price) FROM orders WHERE 1=0").fetchall(), [(None,)])

    def test_comparison_keeps_duplicates_order_and_truncation(self):
        def r(rows, truncated=False):
            return SimpleNamespace(columns=["x"], rows=rows, truncated=truncated)
        self.assertTrue(same_result(r([(1,), (2,)]), r([(2,), (1,)])))
        self.assertFalse(same_result(r([(1,), (1,)]), r([(1,)])))
        self.assertFalse(same_result(r([(1,), (2,)]), r([(2,), (1,)]), ordered=True))
        self.assertFalse(same_result(r([(1,)], True), r([(1,)])))


if __name__ == "__main__":
    unittest.main()
