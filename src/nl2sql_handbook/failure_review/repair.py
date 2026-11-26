"""针对已观察失败的人工诊断补丁回放；不是通用自动修复器或新测试成绩。"""

import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re

from nl2sql_handbook.ai4db_demo.engine import Budget,Database,Limits
from nl2sql_handbook.evaluation_demo.dataset import Case,DATA
from nl2sql_handbook.evaluation_demo.grading import grade
from nl2sql_handbook.evaluation_demo.runner import write_json
from .review import ROOT

PATCHES={
    ("school-09","workflow"):(r"e\.enrollment_id\s*=\s*s\.student_id","e.student_id = s.student_id",
        "enrollment_id是选课记录主键；学生外键是enrollments.student_id。原选表列阶段遗漏了student_id，后续生成器只获得筛选结构。此处仅修正SQL连接键；生成链的修复还需要在开发集验证关联键补齐，不把SQL回放当成模型问题已解决。"),
    ("library-12","join_workflow"):(r"LEFT\s+JOIN\s+main\.loans","JOIN main.loans",
        "题目只统计持有未归还图书的读者，INNER JOIN排除零持有读者，保留原NULL城市分组和DISTINCT计数。"),
    ("sales-08","join_workflow"):(r"COALESCE\(SUM\(quantity\s*\*\s*unit_price\),\s*0\)","SUM(quantity * unit_price)",
        "按冻结参考口径保留全NULL金额分组的SUM(NULL)=NULL。题目未明确NULL转0政策，业务口径仍需人工审查。"),
}


def repair_sql(identifier,version,sql):
    pattern,replacement,reason=PATCHES[(identifier,version)]
    fixed,count=re.subn(pattern,replacement,sql,flags=re.I)
    if count!=1:
        raise ValueError("实际候选与已诊断模式不同，拒绝套用补丁")
    return fixed,reason


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args(argv)
    manifest=json.loads((args.run_dir/"manifest.json").read_text(encoding="utf-8"))
    summary=json.loads((args.run_dir/"summary.json").read_text(encoding="utf-8"))
    if not summary.get("complete"):
        raise ValueError("评估未完成")
    for name,digest in manifest["fixture_hashes"].items():
        if hashlib.sha256((DATA/name).read_bytes()).hexdigest()!=digest:
            raise ValueError("数据与冻结运行不同，拒绝混用回放")
    catalog={c["id"]:Case(**c) for c in manifest["cases"]}
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output=args.output_dir or ROOT/f"repairs_{stamp}"
    output.mkdir(parents=True,exist_ok=False)
    results=[]
    for (identifier,version) in PATCHES:
        path=args.run_dir/f"{identifier}__{version}__1.json"
        if not path.is_file():
            continue
        original=json.loads(path.read_text(encoding="utf-8"))
        if original["grade"]["passed"] or not original.get("ok"):
            continue
        fixed,reason=repair_sql(identifier,version,original["result"]["sql"])
        case=catalog[identifier]
        database=Database(DATA/f"{case.domain}_0.duckdb",Limits(),Budget())
        try:
            value=database.snapshot(fixed)
        finally:
            database.close()
        checked=grade(case,fixed,{"rows":value.rows,"truncated":False},DATA)
        results.append({"case":identifier,"version":version,"reason":reason,"original_passed":False,
                        "original_sql":original["result"]["sql"],"fixed_sql":fixed,"replay_grade":checked,
                        "original_selected_schema":original["result"].get("selected_schema",{}),
                        "scope":"seen-case debugging only; not held-out improvement; no LLM request"})
        (output/f"{identifier}_fixed.sql").write_text(fixed+"\n",encoding="utf-8")
    write_json(output/"repairs.json",results)
    lines=["# 真实失败的局部修复回放", "", "已看过参考答案和失败证据，仅作为排错演示；不覆盖原评分，不宣称泛化改善。", ""]
    for row in results:
        lines += [f"## {row['version']} / {row['case']}","",row["reason"],"",
                  f"原始未通过 → 两组数据回放通过：{row['replay_grade']['passed']}。", "",
                  "```sql",row["fixed_sql"],"```", ""]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"repairs":len(results),"passed":sum(r["replay_grade"]["passed"] for r in results),"report":str((output/"REPORT.md").resolve())},ensure_ascii=False))
    return 0 if results and all(r["replay_grade"]["passed"] for r in results) else 1


if __name__=="__main__":
    raise SystemExit(main())
