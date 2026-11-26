"""同批次轮转执行旧/新三组Workflow，独立保留全部成绩，不覆盖历史结果。"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path

from nl2sql_handbook.evaluation_demo.dataset import DATA, cases as regression_cases, prepare as prepare_regression
from nl2sql_handbook.evaluation_demo.grading import grade
from nl2sql_handbook.evaluation_demo.runner import aggregate, write_json
from nl2sql_handbook.nl2sql_demo import DEFAULT_MODEL, DEFAULT_KEY_FILE, deepseek_request_options
from .runtime import PROJECT, VERSIONS, execute

from nl2sql_handbook.config import SETTINGS
ROOT = SETTINGS.output_dir / "runs" / "workflow_v2"


def fingerprint(settings, selected, data):
    sources = list(PROJECT.rglob("*.py"))
    payload = dict(settings=settings, cases=[asdict(c) for c in selected],
                   source_hashes={str(p.relative_to(PROJECT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)},
                   fixture_hashes={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(data.iterdir()) if p.suffix in {".duckdb", ".json"}},
                   packages={n: importlib.metadata.version(n) for n in ["duckdb", "langchain", "langchain-core", "langchain-deepseek"]})
    payload["fingerprint"] = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return payload


def report(directory, manifest, outcomes, planned):
    versions = manifest["settings"]["versions"]
    metrics = aggregate(outcomes, versions, planned)
    for version, metric in metrics.items():
        group = [o for o in outcomes if o["version"] == version]
        metric["by_domain"] = {domain: dict(completed=sum(o["domain"] == domain for o in group),
            passed=sum(o["domain"] == domain and o.get("grade", {}).get("passed", False) for o in group))
            for domain in sorted({c["domain"] for c in manifest["cases"]})}
        metric["unknown_token_counts"] = sum(not isinstance(o.get("tokens"), dict) or "total_tokens" not in o["tokens"] for o in group)
    summary = dict(complete=all(m["complete"] for m in metrics.values()), settings=manifest["settings"],
                   fingerprint=manifest["fingerprint"], versions=metrics)
    write_json(directory / "summary.json", summary)
    write_json(directory / "outcomes.json", sorted(outcomes, key=lambda o: (o["case"], o["version"], o["repeat"])))
    lines = ["# Workflow 隔离升级复评", "", f"完成：{summary['complete']}；题集：{manifest['settings']['suite']}。",
             "每个候选SQL在两个独立数据实例上完整执行并评分。", "",
             "| 版本 | 完成/计划 | 与参考一致 | 执行成功 | 平均调用 | 总Token | P50秒 | P95秒 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for version, m in metrics.items():
        fmt = lambda value: "未知" if value is None else f"{value:.2f}"
        lines.append(f"| {version} | {m['completed']}/{m['planned']} | {m['passed']} | {m['execution_succeeded']} | "
                     f"{fmt(m['mean_calls_measured'])} | {m['tokens_reported']} | "
                     f"{fmt(m['p50_ms']/1000 if m['p50_ms'] is not None else None)} | {fmt(m['p95_ms']/1000 if m['p95_ms'] is not None else None)} |")
    lines += ["", "## 原则与边界", "",
              "- 旧题/参考SQL/历史成绩未改。已分析过的30题用于回归，不能再称为独立盲测。",
              "- 仓储迁移题另列，是同轮人工编写的小型诊断题，不是公开跨数据库基准。",
              "- 默认NULL保留SQL语义、默认只统计实际匹配分组；明确题意/业务定义优先。旧题仍可能有口径歧义。",
              "- 新版同时改变信息补齐、提示词、回退和复核；整体对比不能分离每一项的因果贡献。",
              "- 模型不读取参考SQL或第二数据实例；复核批准是模型意见，最终分数由独立执行评分决定。",
              "- 同一模型/API/预算，实际调用和Token可能不同；不声称等算力或架构普遍优势。",
              "- 单次重复不测量随机波动；耗时含连接/ask，不含进程启动和评分。缺失usage不按0成本解释。",
              "", "## 不一致或失败项", ""]
    failures = [o for o in outcomes if not o.get("grade", {}).get("passed", False)]
    lines += [f"- {o['version']} / {o['case']} / {o['repeat']}：{o['grade']['category']}" for o in failures] or ["无。"]
    lines += ["", "## 设置", "", "```json", json.dumps(manifest["settings"], ensure_ascii=False, indent=2), "```"]
    (directory / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["regression", "dev", "transfer"], default="regression")
    parser.add_argument("--versions", nargs="+", choices=VERSIONS, default=VERSIONS)
    parser.add_argument("--case", action="append")
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=2)
    parser.add_argument("--repeats", type=int, choices=range(1, 6), default=1)
    parser.add_argument("--case-timeout", type=float, default=120)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.case_timeout <= 600:
        parser.error("case-timeout必须为1—600秒")
    if args.suite == "transfer":
        from .transfer import cases, prepare
        data, selected = prepare(), cases()
    else:
        prepare_regression()
        data = DATA
        selected = [c for c in regression_cases() if c.split == ("dev" if args.suite == "dev" else "test")]
    if args.case:
        selected = [c for c in selected if c.id in args.case]
    if not selected:
        parser.error("没有匹配的题目")
    settings = dict(suite=args.suite, versions=args.versions, repeats=args.repeats, workers=args.workers,
                    model=args.model, provider_options=deepseek_request_options(), max_model_calls=24,
                    max_sql_attempts=4, max_selection_attempts_v2=2, max_rows=100,
                    case_timeout=args.case_timeout, threads=1, memory="512MB", scoring_instances=2)
    manifest = fingerprint(settings, selected, data)
    directory = args.run_dir or ROOT / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = directory / "manifest.json"
    if path.exists():
        if not args.resume or json.loads(path.read_text())["fingerprint"] != manifest["fingerprint"]:
            parser.error("已有运行，或源码/配置/题集不同；请新建目录，原配置续跑才可--resume")
    elif directory.exists() and any(directory.iterdir()):
        parser.error("输出目录非空且没有完整manifest")
    directory.mkdir(parents=True, exist_ok=True)
    write_json(path, manifest)
    outcomes, tasks = [], []
    for repetition in range(1, args.repeats + 1):
        for case in selected:
            for version in args.versions:
                output = directory / f"{case.id}__{version}__{repetition}.json"
                if output.exists():
                    outcomes.append(json.loads(output.read_text()))
                else:
                    tasks.append((case, version, repetition, output))
    planned = len(selected) * args.repeats
    report(directory, manifest, outcomes, planned)
    print(f"Run: {directory}; pending={len(tasks)}", flush=True)

    def run(task):
        case, version, repetition, output = task
        knowledge = data / f"{case.domain}_knowledge.json"
        request = dict(version=version, database=str(data / f"{case.domain}_0.duckdb"),
                       question=case.question, knowledge=str(knowledge) if knowledge.exists() else None,
                       model=args.model, key_file=str(args.api_key_file), max_calls=24)
        outcome = execute(request, timeout=args.case_timeout)
        outcome.update(case=case.id, domain=case.domain, version=version, repeat=repetition,
                       question=case.question, split=case.split, tags=case.tags)
        if outcome["ok"]:
            outcome["grade"] = grade(case, outcome["result"]["sql"], outcome["result"], data)
        else:
            outcome["grade"] = dict(passed=False, category=outcome.get("error_type", "unknown"))
        write_json(output, outcome)
        return outcome

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for future in as_completed([executor.submit(run, t) for t in tasks]):
            outcome = future.result()
            outcomes.append(outcome)
            report(directory, manifest, outcomes, planned)
            print(f"{len(outcomes)}/{planned*len(args.versions)} {outcome['case']} {outcome['version']} "
                  f"{'PASS' if outcome['grade']['passed'] else 'FAIL'} calls={outcome.get('model_calls')} "
                  f"elapsed={outcome['elapsed_ms']/1000:.2f}s", flush=True)
    summary = report(directory, manifest, outcomes, planned)
    print(f"Report: {directory / 'REPORT.md'}", flush=True)
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
