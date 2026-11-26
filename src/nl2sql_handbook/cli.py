"""统一命令行入口，按需导入各阶段实现。"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys

VARIANTS = ("workflow", "agent", "join_workflow", "join_agent", "rag_workflow", "rag_agent",
            "multi_agent", "workflow_v2", "join_workflow_v2", "rag_workflow_v2")
MODULES = {
    "minimal": "minimal", "workflow": "nl2sql_demo", "agent": "nl2sql_agent",
    "join-workflow": "nl2sql_workflow_join", "join-agent": "nl2sql_agent_join",
    "rag": "rag_demo.cli", "multi-agent": "multi_agent_demo.cli",
    "workflow-v2": "workflow_v2.__main__", "evaluate": "evaluation_demo.runner",
    "evaluate-workflow": "workflow_v2.evaluate", "review": "failure_review.review",
    "repair": "failure_review.repair", "ai4db": "ai4db_demo.cli", "serve": "delivery_demo.server",
    "acceptance": "live_acceptance.run",
}


def output(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def initialize(argv):
    parser = argparse.ArgumentParser(prog="nl2sql init", description="生成独立的合成示例库，不调用模型")
    parser.add_argument("--evaluation", action="store_true", help="同时准备三领域评估/HTTP 数据与检查参考 SQL")
    parser.add_argument("--transfer", action="store_true", help="同时准备仓储诊断数据")
    args = parser.parse_args(argv)
    from .nl2sql_demo import DEFAULT_DB, initialize_demo_database
    initialize_demo_database(DEFAULT_DB)
    result = {"demo_database": str(DEFAULT_DB)}
    if args.evaluation:
        from .evaluation_demo.dataset import DATA, prepare
        from .evaluation_demo.runner import validate_references, write_json
        prepared = prepare()
        references = validate_references(DATA)
        write_json(DATA / "dataset.json", prepared)
        write_json(DATA / "reference_checks.json", references)
        result.update(evaluation_data=str(DATA), cases=36, reference_executions=len(references)*2)
    if args.transfer:
        from .workflow_v2.transfer import prepare
        result["transfer_data"] = str(prepare())
    output(result)
    return 0


def query(argv):
    from .config import SETTINGS, read_api_key
    from .nl2sql_demo import DEFAULT_DB, initialize_demo_database
    parser = argparse.ArgumentParser(prog="nl2sql query", description="有预算与整题硬超时的统一查询入口；会调用模型")
    parser.add_argument("-q", "--question", required=True)
    parser.add_argument("--variant", choices=VARIANTS, default="workflow_v2")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--knowledge-file", type=Path)
    parser.add_argument("--model", default=SETTINGS.model)
    parser.add_argument("--max-model-calls", type=int, default=24, help="逻辑模型调用上限，1—30")
    parser.add_argument("--max-rows", type=int, default=100, help="返回行数上限，1—1000")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args(argv)
    if not args.question.strip() or not 1 <= args.timeout <= 600:
        parser.error("问题不能为空，timeout 必须在 1—600 秒之间")
    if not 1 <= args.max_model_calls <= 30 or not 1 <= args.max_rows <= 1000:
        parser.error("max-model-calls 必须在 1—30，max-rows 必须在 1—1000 之间")
    if args.db == DEFAULT_DB:
        initialize_demo_database(DEFAULT_DB)
    database = args.db.expanduser().resolve()
    if not database.is_file():
        parser.error("指定的 DuckDB 数据库不存在；请检查 --db 路径")
    knowledge = args.knowledge_file.expanduser().resolve() if args.knowledge_file else None
    if knowledge and not knowledge.is_file():
        parser.error("knowledge-file 不存在")
    read_api_key()  # 先给出可操作的配置错误，不启动注定失败的工作进程。
    from .workflow_v2.runtime import execute
    outcome = execute({"version": args.variant, "database": str(database), "question": args.question,
                       "knowledge": str(knowledge) if knowledge else None, "model": args.model,
                       "key_file": str(SETTINGS.key_file), "max_calls": args.max_model_calls,
                       "max_rows": args.max_rows}, timeout=args.timeout)
    metrics = {k: outcome.get(k) for k in ("model_calls", "tokens", "elapsed_ms", "harness_wall_ms")}
    if not outcome["ok"]:
        output({"ok": False, "variant": args.variant, "error_type": outcome.get("error_type"),
                "error": outcome.get("error"), "metrics": metrics})
        return 1
    output({"ok": True, "variant": args.variant, **outcome["result"], "metrics": metrics})
    return 0


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="nl2sql", description="NL2SQL Handbook：运行、评估与本地交付")
    parser.add_argument("--version", action="version", version=importlib.metadata.version("nl2sql-handbook"))
    parser.add_argument("command", nargs="?", choices=["doctor", "init", "offline", "query", *MODULES],
                        help="子命令后加 --help 查看各阶段参数")
    if not arguments or arguments[0].startswith("-"):
        parser.parse_args(arguments)
        parser.print_help()
        return 0
    args = parser.parse_args(arguments[:1])
    tail = arguments[1:]
    try:
        if args.command == "init":
            return initialize(tail)
        if args.command == "query":
            return query(tail)
        if args.command == "doctor":
            doctor_parser = argparse.ArgumentParser(prog="nl2sql doctor", description="检查本地配置，不联网、不验证密钥有效性")
            doctor_parser.add_argument("--show-paths", action="store_true", help="显示完整本机路径，仅用于本地排查")
            doctor_args = doctor_parser.parse_args(tail)
            from .config import public_settings
            info = public_settings(show_paths=doctor_args.show_paths)
            info["python"] = sys.version.split()[0]
            info["packages"] = {p: importlib.metadata.version(p) for p in
                                ("nl2sql-handbook", "duckdb", "langchain", "langchain-core", "langchain-deepseek", "python-dotenv")}
            output(info)
            return 0
        if args.command == "offline":
            argparse.ArgumentParser(prog="nl2sql offline", description="预设模型回复，实际执行四种流程，不调用 API").parse_args(tail)
            import tempfile
            from .nl2sql_demo import initialize_demo_database
            from .offline_examples import run_all
            with tempfile.TemporaryDirectory(prefix="nl2sql-offline-") as directory:
                database = Path(directory)/"demo.duckdb"
                initialize_demo_database(database)
                output(run_all(database))
            return 0
        module = importlib.import_module("nl2sql_handbook." + MODULES[args.command])
        # 原阶段 CLI 中部分 main() 读取 sys.argv；统一入口保持其参数语义。
        old = sys.argv
        try:
            sys.argv = ["nl2sql " + args.command, *tail]
            return module.main() or 0
        finally:
            sys.argv = old
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        from .privacy import redact_api_keys
        message = redact_api_keys(str(exc))
        print(f"{type(exc).__name__}: {message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
