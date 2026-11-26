"""新版本独立的计量与进程入口，复用旧计量器但不修改旧适配器注册表。"""

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter

from langchain_deepseek import ChatDeepSeek

from nl2sql_handbook.evaluation_demo.adapters import Meter, query as old_query
from nl2sql_handbook.nl2sql_demo import DEFAULT_MODEL, DEFAULT_KEY_FILE, deepseek_request_options, load_api_key, describe_error
from .workflow import ImprovedWorkflow, ImprovedJoinWorkflow, ImprovedRAGWorkflow

FACTORIES = {"workflow_v2": ImprovedWorkflow, "join_workflow_v2": ImprovedJoinWorkflow,
             "rag_workflow_v2": ImprovedRAGWorkflow}
VERSIONS = ["workflow", "workflow_v2", "join_workflow", "join_workflow_v2", "rag_workflow", "rag_workflow_v2"]
PROJECT = Path(__file__).resolve().parent.parent


def query(version, database, question, *, knowledge=None, model=DEFAULT_MODEL, key_file=DEFAULT_KEY_FILE,
          max_calls=24, max_rows=100, llm=None):
    if version not in FACTORIES:
        return old_query(version, Path(database), question, knowledge=Path(knowledge) if knowledge else None,
                         model=model, key_file=key_file, max_calls=max_calls, max_rows=max_rows, llm=llm)
    if not 1 <= max_calls <= 30 or not 1 <= max_rows <= 1000:
        raise ValueError("无效预算")
    meter, started = Meter(max_calls), perf_counter()
    outcome = dict(ok=False, version=version, requested_model=model, thinking="disabled",
                   max_model_calls=max_calls, max_sql_attempts=4)
    try:
        if llm is None:
            llm = ChatDeepSeek(model=model, api_key=load_api_key(Path(key_file)), callbacks=[meter],
                               **deepseek_request_options())
        else:
            llm = llm.model_copy(update={"callbacks": [*(llm.callbacks or []), meter]})
        with FACTORIES[version](Path(database), llm=llm, max_rows=max_rows, max_sql_retries=3,
                               knowledge_files=[Path(knowledge)] if knowledge else []) as demo:
            demo.connection.execute("SET threads=1")
            demo.connection.execute("SET memory_limit='512MB'")
            demo.connection.execute("SET max_temp_directory_size='0B'")
            outcome.update(ok=True, result=asdict(demo.ask(question, explain=False)))
    except Exception as exc:
        outcome.update(error=describe_error(exc), error_type=type(exc).__name__,
                       trace=getattr(exc, "tool_trace", []), retry_history=getattr(exc, "history", []),
                       attempts=getattr(exc, "attempts", None))
    outcome.update(meter.export(), elapsed_ms=(perf_counter() - started) * 1000)
    return outcome


def execute(request, timeout=120):
    """每题独立进程；整个模型等待及SQL执行均受父进程硬超时约束。"""
    started = perf_counter()
    with tempfile.TemporaryDirectory(prefix="workflow-v2-") as directory:
        source, output = Path(directory) / "request.json", Path(directory) / "outcome.json"
        source.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
        process = subprocess.Popen([sys.executable, "-m", "nl2sql_handbook.workflow_v2.worker", str(source), str(output)],
                                   cwd=None, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            return dict(ok=False, error_type="TaskTimeout", error="整题硬超时，进程已回收",
                        model_calls=None, tokens=None, elapsed_ms=(perf_counter() - started) * 1000)
        if process.returncode or not output.is_file():
            return dict(ok=False, error_type="WorkerFailure", error="工作进程没有完整结果",
                        model_calls=None, tokens=None, elapsed_ms=(perf_counter() - started) * 1000)
        result = json.loads(output.read_text(encoding="utf-8"))
        result["harness_wall_ms"] = (perf_counter() - started) * 1000
        return result
