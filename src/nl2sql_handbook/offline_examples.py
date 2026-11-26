"""用预设模型回复复现四种流程；实际运行 LangChain、工具和 DuckDB。"""
import json
import tempfile
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage
from nl2sql_handbook.nl2sql_demo import NL2SQLDemo, initialize_demo_database
from nl2sql_handbook.nl2sql_agent import NL2SQLAgent
from nl2sql_handbook.nl2sql_workflow_join import JoinNL2SQLWorkflow
from nl2sql_handbook.nl2sql_agent_join import JoinNL2SQLAgent
# 只复用测试模型适配器；生产实现不依赖此文件。
from nl2sql_handbook.testing import ToolCallingModel, action

SALES_SELECTION = {"main.orders": ["quantity", "unit_price", "status"]}
SALES_SQL = "SELECT SUM(quantity * unit_price) AS sales FROM orders WHERE status = '已完成'"
JOIN_SELECTION = {
    "main.customers": ["city"], "main.products": ["category"],
    "main.orders": ["quantity", "unit_price", "status"],
}
JOIN_SQL = """SELECT c.city, p.category, SUM(o.quantity * o.unit_price) AS sales
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
JOIN products p ON o.product_id = p.product_id
WHERE o.status = '已完成'
GROUP BY c.city, p.category
ORDER BY sales DESC, c.city, p.category"""
EXPECTED_JOIN = [("北京", "数码", Decimal("7094.00")),
                 ("上海", "数码", Decimal("5997.00")),
                 ("深圳", "家具", Decimal("1798.00")),
                 ("北京", "家具", Decimal("899.00")),
                 ("深圳", "数码", Decimal("99.00"))]


def run_all(db):
    models = {
        "workflow": FakeListChatModel(responses=[json.dumps(SALES_SELECTION),
            "SELECT missing_amount FROM orders", SALES_SQL]),
        "agent": ToolCallingModel(responses=[
            action("get_schema", {"selection": {"main.orders": ["order_id"]}}),
            action("execute_sql", {"sql": "SELECT missing_amount FROM orders"}),
            action("list_tables", {}), action("get_schema", {"selection": SALES_SELECTION}),
            action("execute_sql", {"sql": SALES_SQL}), AIMessage(content="查询完成。")]),
        "workflow_join": FakeListChatModel(responses=[json.dumps({"tables": JOIN_SELECTION,
            "relationship_ids": []}), JOIN_SQL]),
        "agent_join": ToolCallingModel(responses=[action("list_tables", {}),
            action("get_schema", {"selection": JOIN_SELECTION}),
            action("execute_sql", {"sql": JOIN_SQL}), AIMessage(content="查询完成。")]),
    }
    classes = {"workflow": NL2SQLDemo, "agent": NL2SQLAgent,
               "workflow_join": JoinNL2SQLWorkflow, "agent_join": JoinNL2SQLAgent}
    out = {"model_mode": "scripted_offline", "results": {}}
    for name, factory in classes.items():
        question = "按城市和商品类别统计已完成订单销售额" if "join" in name else "已完成订单销售额是多少？"
        with factory(db, llm=models[name]) as demo:
            result = demo.ask(question)
        assert result.rows == (EXPECTED_JOIN if "join" in name else [(Decimal("15887.00"),)])
        assert result.attempts == (1 if "join" in name else 2)
        out["results"][name] = asdict(result)
    return out


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        db = Path(directory) / "example.duckdb"
        initialize_demo_database(db)
        print(json.dumps(run_all(db), ensure_ascii=False, indent=2, default=str))
