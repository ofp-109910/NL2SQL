"""真实 DeepSeek 冒烟验收，独立合成库，不修改或发送已有业务数据库。

从项目根目录运行：python -m nl2sql_handbook.live_acceptance.run
--prepare-only 仅准备夹具和本地标准答案，不读取密钥、不联网。
--case 可只重跑某一项；每次使用新的输出目录，保留每次真实运行的证据。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from nl2sql_handbook.config import SETTINGS
from threading import Lock
from time import perf_counter

import duckdb
import httpx
from langchain_core.callbacks import BaseCallbackHandler
from langchain_deepseek import ChatDeepSeek

from nl2sql_handbook.nl2sql_demo import (
    DEFAULT_KEY_FILE, DEFAULT_MODEL, connect_read_only, deepseek_request_options,
    describe_error, initialize_demo_database, load_api_key,
)
from nl2sql_handbook.rag_demo.agent import RAGAgent
from nl2sql_handbook.rag_demo.retrieval import DEMO_KNOWLEDGE
from nl2sql_handbook.rag_demo.workflow import RAGWorkflow
from nl2sql_handbook.multi_agent_demo.team import MultiAgentNL2SQL


GROUP_QUESTION = (
    "按客户所在地和商品大类统计已完成订单的成交额。依次返回客户所在地、商品大类、成交额三列；"
    "按成交额降序，同金额再按客户所在地、商品大类升序。"
)
GROUP_SQL = """SELECT c.city, p.category, SUM(o.quantity * o.unit_price) AS sales
FROM main.orders o
JOIN main.customers c ON o.customer_id = c.customer_id
JOIN main.products p ON o.product_id = p.product_id
WHERE o.status = '已完成'
GROUP BY c.city, p.category ORDER BY sales DESC, c.city, p.category"""
LEFT_QUESTION = (
    "列出所有客户的已完成订单笔数与成交额，包括没有订单或只有取消订单的客户。"
    "依次返回客户ID、客户名称、订单笔数、成交额四列；按客户ID升序，"
    "无已完成订单时笔数与金额都返回0。"
)
LEFT_SQL = """SELECT c.customer_id, c.customer_name, COUNT(o.order_id) AS count,
COALESCE(SUM(o.quantity * o.unit_price), 0) AS sales
FROM main.customers c LEFT JOIN main.orders o
ON c.customer_id = o.customer_id AND o.status = '已完成'
GROUP BY c.customer_id, c.customer_name ORDER BY c.customer_id"""
CASES = {
    "rag_workflow_join": (RAGWorkflow, GROUP_QUESTION, GROUP_SQL),
    "rag_agent_left_join": (RAGAgent, LEFT_QUESTION, LEFT_SQL),
    "multi_agent_join": (MultiAgentNL2SQL, GROUP_QUESTION, GROUP_SQL),
}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


class CallCounter(BaseCallbackHandler):
    """只计数，不记录提示词、密钥、HTTP 头或隐藏推理。"""

    def __init__(self, label):
        self.label, self.calls, self.tokens = label, 0, {}
        self.models, self.lock = set(), Lock()

    def on_chat_model_start(self, serialized, messages, **kwargs):
        with self.lock:
            self.calls += 1
            print(f"{self.label}: model call {self.calls}", flush=True)

    def on_llm_end(self, response, **kwargs):
        with self.lock:
            for generation in response.generations:
                for item in generation:
                    message = getattr(item, "message", None)
                    for name in ("input_tokens", "output_tokens", "total_tokens"):
                        value = (getattr(message, "usage_metadata", None) or {}).get(name)
                        if isinstance(value, int):
                            self.tokens[name] = self.tokens.get(name, 0) + value
                    name = (getattr(message, "response_metadata", None) or {}).get("model_name")
                    if name:
                        self.models.add(name)


def normalize(rows):
    """忽略数值类型的等价差异，仍严格比较行序、维度、NULL 与金额。"""
    return [[str(Decimal(str(value)).quantize(Decimal("0.000001")))
             if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool) else value
             for value in row] for row in rows]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=list(CASES), action="append")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args(argv)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = args.output_dir or SETTINGS.output_dir / "runs" / "live_acceptance" / timestamp
    output.mkdir(parents=True, exist_ok=True)
    database = output / "synthetic.duckdb"
    if database.exists():
        raise ValueError("输出目录已有验收数据库，请使用新目录以保留之前的证据。")
    initialize_demo_database(database)
    # 增加无订单/只有取消订单的客户，避免 INNER JOIN 或右表 WHERE 过滤侥幸过关。
    with duckdb.connect(str(database)) as connection:
        connection.execute("INSERT INTO customers VALUES (5, '测试无订单', '杭州'), (6, '测试仅取消', '南京')")
        connection.execute("INSERT INTO orders VALUES (11, 6, 1, 1, 100.00, '2025-01-01', '已取消')")
    expected = {}
    connection = connect_read_only(database)
    try:
        for identifier, (_, question, reference) in CASES.items():
            expected[identifier] = {"question": question, "reference_sql": reference,
                                    "expected_rows": connection.execute(reference).fetchall()}
    finally:
        connection.close()
    write_json(output / "expected.json", expected)
    print(f"Output directory: {output.resolve()}", flush=True)
    if args.prepare_only:
        print("Prepared synthetic fixture and local reference results; no key read or API call.")
        return 0

    # 先做不带鉴权的 HTTPS 探测；4xx 也能证明网络可达，鉴权交给真实模型调用验收。
    network = {}
    try:
        network["https_status"] = httpx.get("https://api.deepseek.com/", timeout=15).status_code
    except Exception as exc:
        network["error"] = describe_error(exc)
        write_json(output / "network.json", network)
        print(network["error"], flush=True)
        return 1
    write_json(output / "network.json", network)
    outcomes = []
    for identifier in args.case or CASES:
        factory, question, _ = CASES[identifier]
        counter, start = CallCounter(identifier), perf_counter()
        outcome = {"case": identifier, "question": question, "requested_model": args.model,
                   "thinking": "disabled",
                   "expected_rows": expected[identifier]["expected_rows"]}
        try:
            # 与原 Demo 相同的 provider/model/超时/重试参数；这里只增加无内容日志的计数器。
            model = ChatDeepSeek(model=args.model, api_key=load_api_key(DEFAULT_KEY_FILE),
                                 callbacks=[counter], **deepseek_request_options())
            with factory(database, llm=model, knowledge_files=[DEMO_KNOWLEDGE]) as demo:
                result = demo.ask(question, explain=True)
            write_json(output / f"{identifier}.json", asdict(result))
            checks = {
                "rows_equal_reference": normalize(result.rows) == normalize(outcome["expected_rows"]),
                "not_truncated": not result.truncated,
                "retrieval_has_hits": any(report["hits"] for report in result.retrieval_trace),
                "explanation_present": bool(result.explanation),
                "no_explanation_error": not result.explanation_error,
                "no_agent_error": not getattr(result, "agent_error", None),
            }
            if identifier == "multi_agent_join":
                checks.update(review_approved=result.review["status"] == "approved",
                              all_four_agents_called=all(result.calls_by_agent.get(role, 0) > 0
                                                         for role in ("supervisor", "retriever", "sql", "reviewer")))
            outcome.update(passed=all(checks.values()), checks=checks, sql=result.sql, rows=result.rows,
                           sql_attempts=result.attempts, calls_by_agent=getattr(result, "calls_by_agent", None))
        except Exception as exc:
            outcome.update(passed=False, error=describe_error(exc))
            trace = getattr(exc, "agent_trace", getattr(exc, "tool_trace", None))
            if trace is not None:
                write_json(output / f"{identifier}_failure_trace.json", trace)
        outcome.update(elapsed_seconds=round(perf_counter() - start, 2), model_calls=counter.calls,
                       tokens=counter.tokens, response_models=sorted(counter.models))
        outcomes.append(outcome)
        write_json(output / "summary.json", outcomes)  # 每项完成即保存，后续失败也不丢前面的证据。
        print(f"{identifier}: {'PASS' if outcome['passed'] else 'FAIL'}, "
              f"{outcome['model_calls']} calls, {outcome['elapsed_seconds']}s", flush=True)
        if not outcome["passed"]:
            print(json.dumps({key: outcome[key] for key in ("error", "checks") if key in outcome}, ensure_ascii=False), flush=True)
    lines = ["# 真实 DeepSeek 冒烟验收", "", "仅使用独立合成库；未读取或修改原有数据库。",
             f"请求模型：`{args.model}`；显式非思考模式 `thinking=disabled`。", "",
             "| 用例 | 结果 | 模型调用 | SQL 执行 | 耗时/秒 |", "| --- | --- | --- | --- | --- |"]
    lines.extend(f"| {item['case']} | {'PASS' if item['passed'] else 'FAIL'} | {item['model_calls']} | "
                 f"{item.get('sql_attempts', '-')} | {item['elapsed_seconds']} |" for item in outcomes)
    lines.extend(["", "标准 SQL/结果见 expected.json，逐项检查见 summary.json，完整公开工具轨迹见对应用例 JSON。",
                  "这是小规模联网冒烟验收，不是跨数据库准确率评测。"])
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0 if all(item["passed"] for item in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
