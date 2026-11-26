"""Multi-Agent 独立入口，保留上一阶段单 Agent 作为可运行对照。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from nl2sql_handbook.nl2sql_demo import DEFAULT_DB, DEFAULT_KEY_FILE, DEFAULT_MODEL, initialize_demo_database, report_error
from .runtime import DatabaseContext, TeamExecutionError
from .team import MultiAgentNL2SQL, ROLE_TOOLS


def main(argv=None):
    parser = argparse.ArgumentParser(description="第 4 阶段：Supervisor + 检索 / SQL / 核验 Agent")
    parser.add_argument("--mode", choices=["multi", "single"], default="multi",
                        help="single 使用上一阶段 RAGAgent，multi 使用独立的四角色图")
    parser.add_argument("-q", "--question")
    parser.add_argument("--db", type=Path)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-rows", type=int, default=100)
    parser.add_argument("--max-sql-retries", type=int, choices=range(4), default=3)
    parser.add_argument("--max-model-calls", type=int, metavar="N",
                        help="multi 全团队总数，默认 40、最多 100；single 默认 12、最多 30")
    parser.add_argument("--max-delegations", type=int, default=10)
    parser.add_argument("--relationships-file", type=Path)
    files = parser.add_mutually_exclusive_group()
    files.add_argument("--knowledge-file", type=Path, action="append")
    files.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--top-k", type=int, choices=range(1, 11), default=4)
    parser.add_argument("--max-context-chars", type=int, default=2600)
    parser.add_argument("--explain", action="store_true")
    offline = parser.add_mutually_exclusive_group()
    offline.add_argument("--show-architecture", action="store_true", help="显示角色与工具，不读取密钥、不调用模型")
    offline.add_argument("--retrieve-only", action="store_true", help="仅检索资料，不调用模型")
    offline.add_argument("--show-schema", action="store_true")
    offline.add_argument("--show-relationships", action="store_true")
    args = parser.parse_args(argv)
    calls = args.max_model_calls if args.max_model_calls is not None else (40 if args.mode == "multi" else 12)
    if not 1 <= calls <= (100 if args.mode == "multi" else 30):
        parser.error("max-model-calls：multi 需为 1—100，single 需为 1—30。")
    if args.max_rows < 1 or not 1 <= args.max_delegations <= 30 or not 200 <= args.max_context_chars <= 12000:
        parser.error("检查 max-rows、max-delegations（1—30）与 max-context-chars（200—12000）。")
    if args.retrieve_only and (args.question is None or not args.question.strip()):
        parser.error("--retrieve-only 需要非空 -q。")

    def output(value):
        print(json.dumps(value, ensure_ascii=False, indent=2, default=str))

    def failure(exc):
        if isinstance(exc, TeamExecutionError):
            output({"ok": False, "error": str(exc), "attempts": exc.attempts,
                    "model_calls": exc.model_calls, "review": exc.review,
                    "candidate_sql": exc.candidate_sql, "agent_trace": exc.agent_trace})
        else:
            report_error(exc)

    try:
        if args.show_architecture:
            output({"pattern": "supervisor-as-tools", "roles": ROLE_TOOLS,
                    "max_model_calls": calls, "max_sql_attempts": 1 + args.max_sql_retries})
            return 0
        db = args.db if args.db is not None else DEFAULT_DB
        if args.db is None:
            initialize_demo_database(db)
        common = dict(max_rows=args.max_rows, relationships_file=args.relationships_file,
                      knowledge_files=[] if args.metadata_only else args.knowledge_file,
                      top_k=args.top_k, max_context_chars=args.max_context_chars)
        if args.retrieve_only or args.show_schema or args.show_relationships:
            with DatabaseContext(db, **common) as database:
                if args.show_schema:
                    output(database.schema_catalog)
                elif args.show_relationships:
                    output(list(database.catalog.relationships.values()))
                else:
                    output(database.knowledge.search(args.question))
            return 0
        options = dict(**common, key_file=args.api_key_file, model=args.model,
                       max_sql_retries=args.max_sql_retries, max_model_calls=calls)
        factory = MultiAgentNL2SQL
        if args.mode == "multi":
            options["max_delegations"] = args.max_delegations
        else:
            from nl2sql_handbook.rag_demo.agent import RAGAgent
            factory = RAGAgent

        def query(demo, question):
            start = perf_counter()
            result = asdict(demo.ask(question, explain=args.explain))
            result.update(mode=args.mode, elapsed_ms=round((perf_counter() - start) * 1000, 2))
            output(result)

        with factory(db, **options) as demo:
            if args.question is not None:
                query(demo, args.question)
                return 0
            print(f"{args.mode} 已启动，每题独立。输入 exit / quit / 退出 结束。")
            while True:
                try:
                    question = input("\n请输入问题 > ").strip()
                except EOFError:
                    break
                if question.lower() in {"exit", "quit", "退出"}:
                    break
                if question:
                    try:
                        query(demo, question)
                    except Exception as exc:
                        failure(exc)
        return 0
    except KeyboardInterrupt:
        print("\n已退出。")
        return 130
    except Exception as exc:
        failure(exc)
        return 1
