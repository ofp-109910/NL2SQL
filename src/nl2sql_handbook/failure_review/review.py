"""失败复盘不是自动归因器。证据、推测、题意边界和改进建议分开保存。"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path

from nl2sql_handbook.ai4db_demo.engine import Budget, Database, Limits
from nl2sql_handbook.evaluation_demo.adapters import query
from nl2sql_handbook.evaluation_demo.dataset import DATA, cases, prepare
from nl2sql_handbook.evaluation_demo.grading import grade
from nl2sql_handbook.evaluation_demo.runner import write_json
from nl2sql_handbook.rag_demo.retrieval import KnowledgeIndex
from nl2sql_handbook.nl2sql_demo import connect_read_only, read_schema
from nl2sql_handbook.nl2sql_workflow_join import JoinCatalog

from nl2sql_handbook.config import SETTINGS
ROOT=SETTINGS.output_dir/"runs"/"failure_review"


def hypotheses(tags):
    """只给排查方向；不能仅凭标签证明某一步导致失败。"""
    hints=[]
    if "left_join" in tags:
        hints.append("核对是否保留了所有左表记录、右表过滤所在位置，以及COUNT是否使用匹配主键。")
    if "null" in tags:
        hints.append("核对NULL与0、COUNT(*)与COUNT(可空列)、SUM/AVG空值语义。")
    if "date" in tags:
        hints.append("核对日期边界、固定参考日期、月份格式与日期输出类型。")
    if "ranking" in tags:
        hints.append("核对LIMIT、并列排序键、HAVING与四舍五入发生的顺序。")
    if "join" in tags or "three_tables" in tags:
        hints.append("核对关联键、一对多重复累计和INNER/LEFT的保留范围。")
    return hints or ["核对所选表列、过滤条件、聚合范围和返回列顺序。"]


def extract_failures(directory: Path) -> list[dict]:
    summary=json.loads((directory/"summary.json").read_text(encoding="utf-8"))
    if not summary.get("complete"):
        raise ValueError("评估尚未全部完成，不能生成最终失败复盘")
    manifest=json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    catalog={c["id"]:c for c in manifest["cases"]}
    outcomes=json.loads((directory/"outcomes.json").read_text(encoding="utf-8"))
    failures=[]
    for outcome in outcomes:
        if outcome.get("grade",{}).get("passed"):
            continue
        case=catalog[outcome["case"]]
        value=outcome.get("result",{})
        record={"source":"live_evaluation","case":outcome["case"],"version":outcome["version"],
                "repeat":outcome["repeat"],"question":case["question"],"reference_sql":case["sql"],
                "candidate_sql":value.get("sql"),"category":outcome.get("grade",{}).get("category"),
                "selected_schema":value.get("selected_schema",{}),
                "error":outcome.get("error",outcome.get("grade",{}).get("error")),
                "checks":outcome.get("grade",{}).get("checks",[]),
                "tool_trace":value.get("agent_trace",value.get("tool_trace",outcome.get("trace",[]))),
                "retrieval_trace":value.get("retrieval_trace",[]),
                "hypotheses":hypotheses(case["tags"]),"manual_review_required":True,
                "observation":"与冻结参考结果不一致或执行未完成；仅据此不能证明根因。"}
        if case["id"]=="school-09" and outcome["version"]=="workflow":
            selected=value.get("selected_schema",{}).get("main.enrollments",[])
            if "enrollment_id" in selected and "student_id" not in selected:
                record["selection_observation"]="第一步选择了enrollments.enrollment_id，却遗漏真正的学生外键student_id；生成SQL只获得筛选后的schema，并使用了错误关联键。局部SQL修复还不足以验证生成链已修复，需要另测选表列阶段的关联键补齐。"
        if case["domain"]=="sales" and any(t in case["tags"] for t in ("aggregate","date","three_tables")):
            record["policy_review"]="检查题目是否明确无匹配分组及全NULL金额的处理。参考SQL保留SQL聚合原生NULL，未要求所有对象时只返回实际匹配分组；若题意容许其他解释，应先修订下一版题集，保留本轮原评分。"
        if value.get("review",{}).get("status")=="approved":
            record["review_observation"]="当前候选虽被核验Agent批准，仍未通过独立参考结果检查。"
        failures.append(record)
    return failures


def controlled(root: Path, output: Path) -> list[dict]:
    """故意构造已知错误。它们是故障检测演示，绝不混入真实模型准确率。"""
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    catalog={c.id:c for c in cases()}
    specifications=[
        ("left_becomes_inner","sales-06",lambda s:s.replace("LEFT JOIN","JOIN"),"INNER JOIN丢弃无匹配客户"),
        ("sum_distinct","sales-10",lambda s:s.replace("SUM(o.quantity)","SUM(DISTINCT o.quantity)"),"DISTINCT误删相同数量的不同订单"),
        ("null_becomes_zero","sales-05",lambda s:s.replace("SUM(o.quantity*o.unit_price)","COALESCE(SUM(o.quantity*o.unit_price),0)"),"未经要求把全NULL分组金额变成0"),
        ("date_boundary","library-06",lambda s:s.replace("l.due_on<DATE","l.due_on<=DATE"),"把恰好到期日误算逾期"),
        ("filter_after_left_join","school-03",lambda s:s.replace("AND e.term='2025春'","").replace("GROUP BY","WHERE e.term='2025春' GROUP BY"),"WHERE右表过滤丢弃零选课学生"),
    ]
    records=[]
    for name,identifier,mutate,cause in specifications:
        case=catalog[identifier]
        sql=mutate(case.sql)
        database=Database(root/f"{case.domain}_0.duckdb",Limits(),Budget())
        try:
            snapshot=database.snapshot(sql)
        finally:
            database.close()
        checked=grade(case,sql,{"rows":snapshot.rows,"truncated":False},root)
        records.append({"source":"controlled_fault","id":name,"case":identifier,
                        "injected_cause":cause,"candidate_sql":sql,"reference_sql":case.sql,
                        "detected":not checked["passed"],"grade":checked})

    # 故意加入冲突知识，只证明检索层会把它返回，不伪称真实模型一定采信它。
    connection=connect_read_only(root/"library_0.duckdb")
    try:
        schema=json.loads(read_schema(connection))
        relations=list(JoinCatalog(connection,schema).relationships.values())
    finally:
        connection.close()
    bad=output/"injected_bad_knowledge.json"
    write_json(bad,[{"id":"deliberately-wrong-overdue","text":"逾期 应还日期 未归还：错误定义，due_on小于等于题目日期都算逾期，包括恰好到期日。"}])
    index=KnowledgeIndex(schema,relations,root/"library_0.duckdb",knowledge_files=[root/"library_knowledge.json",bad])
    evidence=index.search("逾期 应还日期 未归还")
    records.append({"source":"controlled_fault","id":"conflicting_retrieval","injected_cause":"故意将冲突定义加入知识源",
                    "detected":any("deliberately-wrong-overdue" in h["id"] for h in evidence["hits"]),
                    "evidence":evidence,"scope":"证明冲突资料可被召回；未测真实模型采信概率，BM25分数不是可信度。"})

    # 四个真实Agent图运行，但模型响应是显式脚本；演示批准不是独立正确性证明。
    class Scripted(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self
    def call(name,**arguments):
        return AIMessage(content="",tool_calls=[{"name":name,"args":arguments,"id":name,"type":"tool_call"}])
    case=catalog["sales-06"]
    wrong=case.sql.replace("LEFT JOIN","JOIN")
    selection={"main.customers":["customer_id","customer_name"],"main.orders":["order_id","customer_id","quantity","unit_price","status"]}
    fake=Scripted(responses=[call("delegate_retrieval",assignment="检索"),call("search_knowledge",query=case.question),AIMessage(content="已检索"),
        call("delegate_sql",assignment="查询"),call("get_schema",selection=selection),call("execute_sql",sql=wrong),AIMessage(content="完成"),
        call("delegate_review",assignment="核验"),call("get_candidate"),call("submit_review",candidate_id=1,status="approved",feedback="故意错误批准"),AIMessage(content="批准"),AIMessage(content="完成")])
    outcome=query("multi_agent",root/"sales_0.duckdb",case.question,knowledge=root/"sales_knowledge.json",llm=fake)
    value=outcome.get("result",{})
    checked=grade(case,value.get("sql",""),value,root)
    records.append({"source":"controlled_fault","id":"reviewer_false_approval","case":case.id,
                    "injected_cause":"脚本模型故意批准语义错误的已执行SQL",
                    "detected":outcome["ok"] and value.get("review",{}).get("status")=="approved" and not checked["passed"],
                    "grade":checked,"review":value.get("review"),"model_kind":"scripted fake; no API call",
                    "scope":"演示独立评分能识别错误批准，不代表真实模型错误率。"})
    return records


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path)
    parser.add_argument("--controlled-only",action="store_true")
    parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args(argv)
    if bool(args.run_dir)==args.controlled_only:
        parser.error("指定 --run-dir，或单独使用 --controlled-only")
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output=(args.output_dir or ROOT/stamp).resolve()
    output.mkdir(parents=True,exist_ok=False)
    failures=extract_failures(args.run_dir.resolve()) if args.run_dir else []
    prepare(DATA)
    injections=controlled(DATA,output)
    write_json(output/"live_failures.json",failures)
    write_json(output/"controlled_faults.json",injections)
    counts=dict(Counter(r["category"] for r in failures))
    lines=["# 失败案例复盘", "",f"真实评估来源：{args.run_dir or '未使用，仅受控演示'}。",
           f"真实失败项：{len(failures)}；分类：{counts}。", "真实项和受控故障严格分开，受控故障不进入模型成绩。", "",
           "## 真实运行证据", ""]
    for number,item in enumerate(failures,1):
        lines += [f"### {number}. {item['version']} / {item['case']}","",item["question"],"",f"观察：{item['observation']}",
                  f"分类：{item['category']}；仍需人工确认根因。"]
        if item["error"]:
            lines += [f"错误：{item['error']}"]
        if item.get("policy_review"):
            lines += ["",f"题意审查：{item['policy_review']}"]
        if item.get("review_observation"):
            lines += [item["review_observation"]]
        if item.get("selection_observation"):
            lines += ["",f"跨步骤证据：{item['selection_observation']}"]
        if item["selected_schema"]:
            lines += ["", "实际选中的结构：", "```json",json.dumps(item["selected_schema"],ensure_ascii=False,indent=2),"```"]
        lines += ["", "候选SQL：", "```sql",item["candidate_sql"] or "-- 没有成功候选", "```",
                  "参考SQL：", "```sql",item["reference_sql"],"```", "", "排查方向："]
        lines += [f"- {hint}" for hint in item["hypotheses"]]
        lines += ["", "验证改进：另建开发题/数据变体复现；冻结新配置后跑未用于调优的测试题，并对照成本，保留本轮基线。", ""]
    lines += ["## 受控故障", "", "| 故障 | 检测/暴露成功 | 已知注入原因 |", "|---|---|---|"]
    lines += [f"| {r['id']} | {r['detected']} | {r['injected_cause']} |" for r in injections]
    lines += ["", "## 建议的下一次实验", "",
              "1. 先审查有歧义的题目和参考口径；题集修订要升版本，不能覆盖当前分数。",
              "2. 对确证语义错误增加开发集反例；区分模型、检索、工具和评估器问题。",
              "3. 每次只改变一个因素，例如NULL规则或检索资料；在独立测试集比较结果和调用成本。",
              "4. 核验批准作为过程证据，独立参考结果作为本轮评分依据。",
              "", "本工具不自动修改提示词、参考答案或旧实现，也不声称标签启发式完成了因果归因。"]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"live_failures":len(failures),"controlled":len(injections),
                      "all_controlled_detected":all(r["detected"] for r in injections),"report":str(output/"REPORT.md")},ensure_ascii=False))
    return 0 if all(r["detected"] for r in injections) else 1
