"""为七个旧实现增加统一的调用入口与计量，不改动旧源码或提示词。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from threading import Lock
from time import perf_counter

from langchain_core.callbacks import BaseCallbackHandler
from langchain_deepseek import ChatDeepSeek

from nl2sql_handbook.nl2sql_demo import DEFAULT_KEY_FILE, DEFAULT_MODEL, NL2SQLDemo, deepseek_request_options, describe_error, load_api_key
from nl2sql_handbook.nl2sql_agent import NL2SQLAgent
from nl2sql_handbook.nl2sql_workflow_join import JoinNL2SQLWorkflow
from nl2sql_handbook.nl2sql_agent_join import JoinNL2SQLAgent
from nl2sql_handbook.rag_demo.workflow import RAGWorkflow
from nl2sql_handbook.rag_demo.agent import RAGAgent
from nl2sql_handbook.multi_agent_demo.team import MultiAgentNL2SQL

FACTORIES = {"workflow":NL2SQLDemo, "agent":NL2SQLAgent,
             "join_workflow":JoinNL2SQLWorkflow, "join_agent":JoinNL2SQLAgent,
             "rag_workflow":RAGWorkflow, "rag_agent":RAGAgent, "multi_agent":MultiAgentNL2SQL}


class Meter(BaseCallbackHandler):
    """仅记录次数、时间、Token 和响应模型，不保存提示词或隐藏推理。

    raise_error=True 使调用上限真正中断调用；默认 callback 只打印错误，
    不能作为预算控制。SDK 内部 HTTP 重试仍沿用旧配置，不计成新模型轮次。
    """
    raise_error = True

    def __init__(self, maximum=24):
        self.maximum, self.calls, self.responses = maximum, 0, 0
        self.tokens, self.models = {}, set()
        self.lock = Lock()

    def on_chat_model_start(self, serialized, messages, **kwargs):
        with self.lock:
            if self.calls >= self.maximum:
                raise RuntimeError("统一评估模型调用预算耗尽")
            self.calls += 1

    def on_llm_end(self, response, **kwargs):
        with self.lock:
            self.responses += 1
            for group in response.generations:
                for generation in group:
                    message = getattr(generation, "message", None)
                    for key in ("input_tokens", "output_tokens", "total_tokens"):
                        value = (getattr(message, "usage_metadata", None) or {}).get(key)
                        if isinstance(value, int):
                            self.tokens[key] = self.tokens.get(key, 0) + value
                    name = (getattr(message, "response_metadata", None) or {}).get("model_name")
                    if name:
                        self.models.add(name)

    def export(self):
        return {"model_calls":self.calls, "model_responses":self.responses,
                "tokens":dict(self.tokens), "response_models":sorted(self.models)}


def query(version: str, database: Path, question: str, *, knowledge: Path | None = None,
          model=DEFAULT_MODEL, key_file=DEFAULT_KEY_FILE, max_calls=24, max_rows=100,
          llm=None) -> dict:
    """每个问题独立建实例，无跨题状态；同一份定义只供带 RAG 的版本检索。

    llm 只用于离线测试注入，正常入口始终使用原 DeepSeek 配置。
    子进程外层还会设置整题硬超时，可终止 LLM 等待或 DuckDB 长查询。
    """
    if version not in FACTORIES or not 1 <= max_calls <= 30 or not 1 <= max_rows <= 1000:
        raise ValueError("未知版本或无效预算")
    meter, started = Meter(max_calls), perf_counter()
    outcome = {"ok":False, "version":version, "requested_model":model,
               "thinking":"disabled", "max_model_calls":max_calls, "max_sql_attempts":4}
    try:
        if llm is None:
            llm = ChatDeepSeek(model=model, api_key=load_api_key(Path(key_file)), callbacks=[meter],
                               **deepseek_request_options())
        else:
            # 模拟模型也走同一计量与预算，测试不能只验证真实请求的一半路径。
            llm = llm.model_copy(update={"callbacks":[*(llm.callbacks or []),meter]})
        options = {"llm":llm, "max_rows":max_rows, "max_sql_retries":3}
        if version in {"agent","join_agent","rag_agent","multi_agent"}:
            options["max_model_calls"] = max_calls
        if version in {"rag_workflow","rag_agent","multi_agent"}:
            options["knowledge_files"] = [knowledge] if knowledge else []
        if version == "multi_agent":
            options["max_delegations"] = 10
        with FACTORIES[version](Path(database), **options) as demo:
            connection = demo.database.connection if version == "multi_agent" else demo.connection
            connection.execute("SET threads=1")
            connection.execute("SET memory_limit='512MB'")
            connection.execute("SET max_temp_directory_size='0B'")
            result = demo.ask(question, explain=False)
        outcome.update(ok=True, result=asdict(result))
    except Exception as exc:
        outcome.update(error=describe_error(exc), error_type=type(exc).__name__,
                       trace=getattr(exc,"agent_trace",getattr(exc,"tool_trace",[])),
                       retry_history=getattr(exc,"history",[]), attempts=getattr(exc,"attempts",None))
    outcome.update(meter.export(), elapsed_ms=(perf_counter()-started)*1000)
    return outcome
