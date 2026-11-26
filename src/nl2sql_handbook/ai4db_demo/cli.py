"""独立命令行入口：既可单独实验，也可接收之前版本输出的 SQL。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from nl2sql_handbook.nl2sql_demo import DEFAULT_KEY_FILE, DEFAULT_MODEL, report_error
from nl2sql_handbook.config import SETTINGS
from .engine import Limits
from .fixture import DEFAULT_DB, EDGE_DB, ROOT, create_fixture
from .optimizer import Experiment


def save_report(report: dict, root: Path) -> Path:
    """每次写入新的时间戳目录，不覆盖 SQL 输入、数据库或历史结果。"""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = root.expanduser().resolve() / stamp
    directory.mkdir(parents=True, exist_ok=False)
    report["report_directory"] = str(directory)
    (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    for name, sql in (("original", report["original_sql"]), ("selected", report["selected_sql"])):
        (directory / f"{name}.sql").write_text(sql + "\n", encoding="utf-8")
    lines = ["# AI4DB 实验报告", "", f"- 决策：{report['reason']}",
             f"- 采用改写：{report['adopted']}", f"- DuckDB：{report['duckdb_version']}",
             f"- 主库：{report['database']}", f"- 附加验证库：{report['validation_databases']}",
             f"- 模型调用：{report['model_calls']}；模型耗时：{report['model_elapsed_ms']:.2f} ms",
             f"- 工作负载 SQL 执行：{report['sql_executions']}/{report['sql_execution_budget']}",
             "", "完整计划、逐轮计时、比较结果与阈值见 report.json。", ""]
    for index, attempt in enumerate(report["attempts"], 1):
        lines += [f"## 候选 {index}", "", f"- 结论：{attempt['reason']}"]
        if attempt.get("candidate_sql"):
            (directory / f"candidate_{index}.sql").write_text(attempt["candidate_sql"] + "\n", encoding="utf-8")
        if attempt.get("diagnosis"):
            lines += [f"- 模型诊断：{attempt['diagnosis']}"]
        if attempt.get("error"):
            lines += [f"- 错误：{attempt['error']}"]
        for check in attempt.get("checks", []):
            lines += [f"- 结果检查：{check['database']}：{check['reason']}"]
        measured = attempt.get("measurement")
        if measured:
            lines += [f"- 原 SQL 中位数：{measured['original_median_ms']:.3f} ms",
                      f"- 候选中位数：{measured['candidate_median_ms']:.3f} ms",
                      f"- 收益：{measured['gain_fraction']:.1%}；配对胜率：{measured['win_fraction']:.1%}",
                      "", "| 轮次 | 顺序 | 原 SQL / ms | 候选 / ms |", "|---|---|---:|---:|"]
            for row, pair in enumerate(measured["pairs"], 1):
                lines += [f"| {row} | {' → '.join(pair['order'])} | {pair['original_ms']:.3f} | {pair['candidate_ms']:.3f} |"]
        lines += [""]
    lines += ["## 实验边界", "", report["scope"], "",
              "测量是本机暖缓存、固定线程的执行与完整抓取时间。模型调用及 EXPLAIN ANALYZE 不在配对计时内。",
              "未证明冷缓存、并发负载、未来数据或其他 SQL 的收益；selected.sql 只是报告产物。", ""]
    (directory / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return directory


def load_input(args) -> tuple[str, str]:
    if args.result_file:
        result = json.loads(args.result_file.expanduser().read_text(encoding="utf-8"))
        if not isinstance(result, dict) or result.get("ok") is False:
            raise ValueError("上一阶段结果必须是成功输出的 JSON 对象")
        sql = result.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("上一阶段 JSON 缺少非空 sql 字段；请重定向单题 JSON 输出")
        # 只提取问题和 SQL；旧的结果行、执行状态不作为本轮正确性证据。
        question = result.get("question", "")
        return sql, question if isinstance(question, str) else ""
    path = args.sql_file or ROOT / "workloads" / "original.sql"
    return path.expanduser().read_text(encoding="utf-8"), ""


def main(argv=None):
    parser = argparse.ArgumentParser(description="第 5 阶段：AI4DB 诊断 → 改写 → 结果检查 → 配对性能实验")
    parser.add_argument("--db", type=Path, help="已有 DuckDB 文件；需同时指定 SQL 输入")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--sql-file", type=Path)
    inputs.add_argument("--result-file", type=Path, help="上一阶段单题 JSON（提取 sql 字段）")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--init-only", action="store_true", help="只创建本阶段合成库，不读取密钥")
    modes.add_argument("--analyze-only", action="store_true", help="离线分析与基线测量")
    modes.add_argument("--candidate-file", type=Path, help="离线评估给定候选，不调用模型")
    parser.add_argument("--orders", type=int, default=200_000, help="首次初始化的订单行数；已有文件不变")
    parser.add_argument("--validation-db", type=Path, action="append", default=[])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--max-proposals", type=int, default=2)
    parser.add_argument("--query-timeout", type=float, default=10)
    parser.add_argument("--max-check-rows", type=int, default=100_000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--min-gain", type=float, default=0.05, help="相对收益门槛，默认 0.05 即 5%%")
    parser.add_argument("--min-saving-ms", type=float, default=0.2)
    parser.add_argument("--min-win-fraction", type=float, default=0.8)
    parser.add_argument("--report-dir", type=Path, default=SETTINGS.output_dir / "runs" / "ai4db")
    args = parser.parse_args(argv)
    if bool(args.db) != bool(args.sql_file or args.result_file):
        parser.error("自定义 --db 必须与 --sql-file 或 --result-file 一起指定，以明确 SQL 对应的数据库")
    if args.init_only and (args.db or args.validation_db):
        parser.error("--init-only 只初始化本阶段默认合成库")
    try:
        limits = Limits(timeout=args.query_timeout, max_rows=args.max_check_rows, threads=args.threads,
                        warmups=args.warmups, repetitions=args.repetitions, min_gain=args.min_gain,
                        min_saving_ms=args.min_saving_ms, min_win_fraction=args.min_win_fraction)
        db = args.db or DEFAULT_DB
        validations = list(args.validation_db)
        if args.db is None:
            created = create_fixture(DEFAULT_DB, args.orders)
            edge_created = create_fixture(EDGE_DB, edge=True)
            validations.insert(0, EDGE_DB)
            if args.init_only:
                print(json.dumps({"database": str(DEFAULT_DB), "created": created,
                                  "edge_database": str(EDGE_DB), "edge_created": edge_created}, ensure_ascii=False, indent=2))
                return 0
        sql, question = load_input(args)
        candidate = args.candidate_file.expanduser().read_text(encoding="utf-8") if args.candidate_file else None
        with Experiment(db, validation_dbs=validations, limits=limits, model=args.model,
                        key_file=args.api_key_file, max_proposals=args.max_proposals) as experiment:
            report = experiment.run(sql, candidate=candidate, analyze_only=args.analyze_only, question=question)
        directory = save_report(report, args.report_dir)
        errors = [a["error"] for a in report["attempts"] if "error" in a]
        output = {"adopted": report["adopted"], "reason": report["reason"],
                  "model_calls": report["model_calls"], "response_models": report["response_models"],
                  "errors": errors, "report": str(directory / "REPORT.md"),
                  "details": str(directory / "report.json"), "selected_sql": str(directory / "selected.sql")}
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 2 if errors and len(errors) == len(report["attempts"]) else 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        report_error(exc)
        return 1
