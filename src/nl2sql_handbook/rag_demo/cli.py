"""新阶段自己的 CLI；旧脚本、旧 README 和依赖文件保持原样。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from nl2sql_handbook.nl2sql_demo import (
    DEFAULT_DB, DEFAULT_KEY_FILE, DEFAULT_MODEL, connect_read_only, initialize_demo_database,
    read_schema, report_error,
)
from nl2sql_handbook.nl2sql_workflow_join import JoinCatalog
from .retrieval import KnowledgeIndex


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="独立的 NL2SQL RAG 扩展：LangChain + DeepSeek + DuckDB + 本地 BM25",
        epilog="在父目录运行 python -m nl2sql_handbook.rag_demo；检索诊断不需要密钥或网络。",
    )
    parser.add_argument("--mode", choices=["workflow", "agent"], default="workflow")
    parser.add_argument("-q", "--question")
    parser.add_argument("--db", type=Path, help="已有 DuckDB 文件；默认复用父目录销售示例库")
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-rows", type=int, default=100)
    parser.add_argument("--max-sql-retries", type=int, choices=range(4), default=3)
    parser.add_argument("--max-model-calls", type=int, choices=range(1, 31), default=12,
                        help="仅用于 Agent；默认与前一阶段相同")
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--relationships-file", type=Path)
    knowledge = parser.add_mutually_exclusive_group()
    knowledge.add_argument("--knowledge-file", type=Path, action="append",
                           help="JSON/MD/TXT 资料；可重复指定；覆盖内置销售资料的默认选择")
    knowledge.add_argument("--metadata-only", action="store_true", help="只索引真实元数据，不加载外部资料")
    parser.add_argument("--top-k", type=int, choices=range(1, 11), default=4)
    parser.add_argument("--max-context-chars", type=int, default=2600, help="检索正文总字符数上限，200—12000")
    parser.add_argument("--no-rag", action="store_true", help="关闭检索，作为同一 JOIN 流程的对照")
    diagnostic = parser.add_mutually_exclusive_group()
    diagnostic.add_argument("--retrieve-only", action="store_true", help="只展示本地检索，不调用模型、不执行业务 SQL")
    diagnostic.add_argument("--show-schema", action="store_true")
    diagnostic.add_argument("--show-relationships", action="store_true")
    args = parser.parse_args(argv)
    if args.max_rows < 1 or not 200 <= args.max_context_chars <= 12000:
        parser.error("max-rows 必须为正数，max-context-chars 必须为 200—12000。")
    if args.retrieve_only and (not args.question or args.no_rag):
        parser.error("--retrieve-only 需要非空 -q，且不能同时指定 --no-rag。")

    def print_json(value):
        print(json.dumps(value, ensure_ascii=False, indent=2, default=str))

    try:
        db_path = args.db if args.db is not None else DEFAULT_DB
        if args.db is None:
            initialize_demo_database(db_path)  # 已存在时不会改写，后续连接始终只读。
        files = [] if args.metadata_only else args.knowledge_file
        index_options = dict(knowledge_files=files, top_k=args.top_k,
                             max_context_chars=args.max_context_chars)
        if args.retrieve_only or args.show_schema or args.show_relationships:
            connection = connect_read_only(db_path)
            try:
                schema = json.loads(read_schema(connection))
                if args.show_schema:
                    print_json(schema)
                else:
                    catalog = JoinCatalog(connection, schema, args.relationships_file)
                    relationships = list(catalog.relationships.values())
                    if args.show_relationships:
                        print_json(relationships)
                    else:
                        print_json(KnowledgeIndex(schema, relationships, db_path, **index_options).search(args.question))
            finally:
                connection.close()
            return 0

        # 诊断分支到这里之前已经返回，不会构造 ChatDeepSeek，也不会读取密钥。
        from .agent import RAGAgent
        from .workflow import RAGWorkflow

        options = dict(**index_options, rag_enabled=not args.no_rag, key_file=args.api_key_file,
                       model=args.model, max_rows=args.max_rows, max_sql_retries=args.max_sql_retries,
                       relationships_file=args.relationships_file)
        factory = RAGWorkflow
        if args.mode == "agent":
            factory = RAGAgent
            options["max_model_calls"] = args.max_model_calls
        with factory(db_path, **options) as demo:
            if args.question is not None:
                print_json(asdict(demo.ask(args.question, explain=args.explain)))
                return 0
            print(f"RAG {args.mode} 已启动（每个问题独立），输入 exit / quit / 退出 结束。")
            while True:
                try:
                    question = input("\n请输入问题 > ").strip()
                except EOFError:
                    break
                if question.lower() in {"exit", "quit", "退出"}:
                    break
                if question:
                    try:
                        print_json(asdict(demo.ask(question, explain=args.explain)))
                    except Exception as exc:
                        report_error(exc)
        return 0
    except KeyboardInterrupt:
        print("\n已退出。")
        return 130
    except Exception as exc:
        report_error(exc)
        return 1
