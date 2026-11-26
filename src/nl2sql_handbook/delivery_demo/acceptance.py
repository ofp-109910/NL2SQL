"""真实HTTP→旧RAG Workflow→DeepSeek→DuckDB验收；不属于离线单元测试。"""

from dataclasses import replace
from datetime import datetime,timezone
import http.client
import json
from pathlib import Path
from threading import Thread

from nl2sql_handbook.evaluation_demo.dataset import DATA,cases,prepare
from nl2sql_handbook.evaluation_demo.grading import grade
from nl2sql_handbook.evaluation_demo.runner import write_json
from .config import ROOT,load
from .server import Service


def main():
    prepare()
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output=ROOT/"acceptance"/stamp
    output.mkdir(parents=True,exist_ok=False)
    config=replace(load(),port=0,log_file=output/"requests.jsonl")
    server=Service(config)
    thread=Thread(target=server.serve_forever,kwargs={"poll_interval":.05},daemon=True)
    thread.start()
    outcomes=[]
    try:
        for case in [c for c in cases() if c.id in {"sales-06","library-06","school-06"}]:
            connection=http.client.HTTPConnection(*server.server_address,timeout=config.request_timeout+5)
            try:
                body=json.dumps({"database":case.domain,"question":case.question},ensure_ascii=False).encode()
                connection.request("POST","/query",body=body,headers={"Content-Type":"application/json"})
                response=connection.getresponse()
                value=json.loads(response.read())
                checked=grade(case,value.get("sql",""),value,DATA) if response.status==200 else {"passed":False}
                outcome={"case":case.id,"status":response.status,"response":value,"grade":checked}
                outcomes.append(outcome)
                write_json(output/f"{case.id}.json",outcome)
                print(f"{case.id}: HTTP {response.status}, reference match={checked['passed']}",flush=True)
            finally:
                connection.close()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    passed=len(outcomes)==3 and all(r["status"]==200 and r["grade"]["passed"] for r in outcomes)
    write_json(output/"summary.json",{"passed":passed,"cases":outcomes,"config":{"version":config.version,"model":config.model},"server_stopped":True})
    lines=["# HTTP真实联网验收", "", "入口：本机HTTP → 隔离子进程 → 旧RAG Workflow → DeepSeek → 只读DuckDB。", "",
           "| 题目 | HTTP状态 | 两组数据完整结果一致 | 模型调用 |", "|---|---:|---|---:|"]
    lines += [f"| {r['case']} | {r['status']} | {r['grade']['passed']} | {r['response'].get('metrics',{}).get('model_calls')} |" for r in outcomes]
    lines += ["",f"全部通过：{passed}。验收服务器已关闭；结构化事件见requests.jsonl。",
              "仅合成数据；没有把参考SQL/隐藏结果发给模型。Docker镜像未在本机构建（当前没有docker命令）。"]
    (output/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(f"Report: {output/'REPORT.md'}",flush=True)
    return 0 if passed else 1


if __name__=="__main__":
    raise SystemExit(main())
