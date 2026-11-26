"""统一评估 CLI。默认不读取任何旧业务库，断点续跑严格校验配置指纹。"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
from statistics import mean, median
from time import perf_counter

from nl2sql_handbook.nl2sql_demo import DEFAULT_KEY_FILE, DEFAULT_MODEL, deepseek_request_options
from nl2sql_handbook.ai4db_demo.engine import Budget, Database, Limits
from .adapters import FACTORIES
from .dataset import DATA, DATASET_VERSION, ROOT, cases, prepare
from .grading import grade
from .process import execute
from nl2sql_handbook.config import SETTINGS


def write_json(path: Path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    temporary.replace(path)


def validate_references(root):
    """发布题集前先在两个数据实例上执行全部参考 SQL，不读取密钥。"""
    checked = []
    for case in cases():
        counts = []
        for variant in (0,1):
            database = Database(root/f"{case.domain}_{variant}.duckdb",Limits(),Budget())
            try:
                database.inspect(case.sql)
                counts.append(len(database.snapshot(case.sql).rows))
            finally:
                database.close()
        checked.append({"case":case.id,"rows":counts})
    return checked


def fingerprint(settings, selected, root):
    project = ROOT.parent
    sources = list(project.rglob("*.py"))
    hashes = {str(p.relative_to(project)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}
    data = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.iterdir()) if p.is_file() and p.suffix in {".duckdb",".json"}}
    payload = {"dataset_version":DATASET_VERSION,"settings":settings,"cases":[asdict(c) for c in selected],
               "source_hashes":hashes,"fixture_hashes":data,
               "packages":{n:importlib.metadata.version(n) for n in ["duckdb","langchain","langchain-core","langchain-deepseek"]}}
    digest = hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,default=str).encode()).hexdigest()
    return {"fingerprint":digest,**payload}


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered)-1,max(0,__import__("math").ceil(len(ordered)*fraction)-1))]


def aggregate(outcomes: list[dict], versions: list[str], planned: int) -> dict:
    result = {}
    for version in versions:
        rows = [r for r in outcomes if r["version"]==version]
        calls = [r["model_calls"] for r in rows if r.get("model_calls") is not None]
        elapsed = [r["elapsed_ms"] for r in rows]
        passed = sum(r.get("grade",{}).get("passed",False) for r in rows)
        by_domain = {}
        for domain in ("sales","library","school"):
            group = [r for r in rows if r["domain"]==domain]
            by_domain[domain] = {"completed":len(group),"passed":sum(r.get("grade",{}).get("passed",False) for r in group)}
        result[version] = {"planned":planned,"completed":len(rows),"passed":passed,
                           "execution_succeeded":sum(r.get("ok",False) for r in rows),
                           "accuracy_completed":passed/len(rows) if rows else None,
                           "complete":len(rows)==planned,"mean_calls_measured":mean(calls) if calls else None,
                           "unknown_call_counts":len(rows)-len(calls),
                           "p50_ms":median(elapsed) if elapsed else None,"p95_ms":percentile(elapsed,.95),
                           "tokens_reported":sum((r.get("tokens") or {}).get("total_tokens",0) for r in rows),
                           "by_domain":by_domain}
    return result


def report(directory: Path, manifest: dict, outcomes: list[dict], planned: int):
    versions = manifest["settings"]["versions"]
    metrics = aggregate(outcomes,versions,planned)
    complete = all(m["complete"] for m in metrics.values())
    summary = {"complete":complete,"dataset_version":DATASET_VERSION,"fingerprint":manifest["fingerprint"],
               "settings":manifest["settings"],"versions":metrics}
    write_json(directory/"summary.json",summary)
    write_json(directory/"outcomes.json",sorted(outcomes,key=lambda r:(r["case"],r["version"],r["repeat"])))
    lines = ["# NL2SQL 跨版本统一评估", "",f"完成：{complete}；题集：{DATASET_VERSION}。",
             "完整结果评分在两组独立数据上进行；列位置/重复次数/NULL/题目排序参与评分，列别名不参与。",
             "", "| 版本 | 完成/计划 | 正确 | 已完成准确率 | 执行成功 | 平均模型调用* | P50 / s | P95 / s |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name,m in metrics.items():
        fmt = lambda x: "—" if x is None else f"{x:.2f}"
        accuracy = "—" if m["accuracy_completed"] is None else f"{m['accuracy_completed']:.1%}"
        lines.append(f"| {name} | {m['completed']}/{m['planned']} | {m['passed']} | {accuracy} | {m['execution_succeeded']} | {fmt(m['mean_calls_measured'])} | {fmt(m['p50_ms']/1000 if m['p50_ms'] is not None else None)} | {fmt(m['p95_ms']/1000 if m['p95_ms'] is not None else None)} |")
    lines += ["", "*硬超时导致计量丢失时记为未知，不作为 0 次调用；详见 summary.json 的 unknown_call_counts。",
              "Token 仅统计已返回的 provider usage，不按不稳定价格推算费用。",
              "", "## 实验设置", "", "```json",json.dumps(manifest["settings"],ensure_ascii=False,indent=2),"```",
              "", "## 失败项", ""]
    failed = [r for r in outcomes if not r.get("grade",{}).get("passed",False)]
    lines += [f"- {r['version']} / {r['case']} / repeat {r['repeat']}：{r.get('grade',{}).get('category',r.get('error_type','unknown'))}" for r in failed]
    if not failed:
        lines.append("当前已完成项无失败；并不证明未覆盖问题也正确。")
    lines += ["", "## 适用边界", "", "自建小型合成题集，参考答案由本项目编写；执行一致性不等于任意数据上的语义证明。",
              "只有完整运行才能作为本轮最终对照。单次调用没有测量随机波动；可用 --repeats 独立重复。",
              "并发数、模型名、输入文件/源码指纹与预算均记录在 manifest.json；不声称架构具有普遍优势。",
              "延迟包含连接/实例构造和 ask，不包含父进程启动与本地评分；harness_wall_ms 另存。",
              "结果、公开工具轨迹、参考 SQL 与评分细节均仅保存在本地。"]
    (directory/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--versions",nargs="+",choices=list(FACTORIES),default=list(FACTORIES))
    parser.add_argument("--split",choices=["dev","test","all"],default="test")
    parser.add_argument("--case",action="append")
    parser.add_argument("--repeats",type=int,default=1)
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--case-timeout",type=float,default=120)
    parser.add_argument("--max-model-calls",type=int,default=24)
    parser.add_argument("--model",default=DEFAULT_MODEL)
    parser.add_argument("--api-key-file",type=Path,default=DEFAULT_KEY_FILE)
    parser.add_argument("--run-dir",type=Path)
    parser.add_argument("--resume",action="store_true")
    args = parser.parse_args(argv)
    if not 1<=args.workers<=4 or not 1<=args.repeats<=5 or not 1<=args.max_model_calls<=30 or not 1<=args.case_timeout<=600:
        parser.error("workers 1—4、repeats 1—5、max-model-calls 1—30、case-timeout 1—600")
    prepared = prepare()
    references = validate_references(DATA)
    if args.prepare_only:
        write_json(DATA/"dataset.json",prepared)
        write_json(DATA/"reference_checks.json",references)
        print(json.dumps({"cases":36,"dev":6,"test":30,"domains":3,"reference_executions":72,"data":str(DATA)},ensure_ascii=False))
        return 0
    selected = [c for c in cases() if (args.split=="all" or c.split==args.split) and (not args.case or c.id in args.case)]
    if not selected or (args.case and set(args.case)-{c.id for c in selected}):
        parser.error("题号不存在或不属于所选 split")
    versions = list(dict.fromkeys(args.versions))
    settings = {"versions":versions,"split":args.split,"repeats":args.repeats,"workers":args.workers,
                "case_timeout":args.case_timeout,"max_model_calls":args.max_model_calls,"max_sql_attempts":4,
                "max_rows":100,"model":args.model,"provider_options":deepseek_request_options(),
                "knowledge_policy":"same domain definitions for RAG variants; no gold SQL sent",
                "threads":1,"memory":"512MB"}
    manifest = fingerprint(settings,selected,DATA)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    directory = (args.run_dir or SETTINGS.output_dir/"runs"/"evaluation"/stamp).expanduser().resolve()
    if args.resume:
        stored = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
        if stored["fingerprint"]!=manifest["fingerprint"]:
            raise ValueError("配置、题集、数据或源码发生变化，禁止混入原运行；请新建 run-dir")
    else:
        directory.mkdir(parents=True,exist_ok=False)
        write_json(directory/"manifest.json",manifest)
    outcomes, tasks = [],[]
    # 题目轮转版本，避免先测完一种架构、后测另一种造成明显时段偏差。
    for repetition in range(1,args.repeats+1):
        for case in selected:
            for version in versions:
                identifier = f"{case.id}__{version}__{repetition}"
                path = directory/f"{identifier}.json"
                if path.exists():
                    outcomes.append(json.loads(path.read_text(encoding="utf-8")))
                else:
                    tasks.append((case,version,repetition,path))
    planned = len(selected)*args.repeats
    print(f"Run: {directory}; pending {len(tasks)}, completed {len(outcomes)}",flush=True)
    report(directory,manifest,outcomes,planned)
    def run(task):
        case,version,repetition,path = task
        request = {"version":version,"database":str(DATA/f"{case.domain}_0.duckdb"),
                   "question":case.question,"knowledge":str(DATA/f"{case.domain}_knowledge.json"),
                   "model":args.model,"key_file":str(args.api_key_file),"max_calls":args.max_model_calls}
        outcome = execute(request,timeout=args.case_timeout)
        outcome.update(case=case.id,domain=case.domain,split=case.split,tags=case.tags,
                       version=version,repeat=repetition,question=case.question)
        if outcome["ok"]:
            value = outcome["result"]
            outcome["grade"] = grade(case,value["sql"],value,DATA)
        else:
            outcome["grade"] = {"passed":False,"category":outcome.get("error_type","unknown")}
        write_json(path,outcome)
        return outcome
    started = perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for future in as_completed([executor.submit(run,task) for task in tasks]):
            outcome = future.result()
            outcomes.append(outcome)
            report(directory,manifest,outcomes,planned)
            print(f"{len(outcomes)}/{planned*len(versions)} {outcome['version']} {outcome['case']}: "
                  f"{'PASS' if outcome['grade']['passed'] else 'FAIL'} "
                  f"calls={outcome.get('model_calls')} elapsed={outcome['elapsed_ms']/1000:.2f}s",flush=True)
    summary = report(directory,manifest,outcomes,planned)
    print(f"Completed in {perf_counter()-started:.1f}s. Report: {directory/'REPORT.md'}",flush=True)
    return 0 if summary["complete"] else 1
