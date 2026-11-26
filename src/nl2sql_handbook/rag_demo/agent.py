"""Agent RAG：获取 schema 时附带证据，并提供可自主改写问题的检索工具。

不重建或替换上一阶段的 Agent 图，仍由模型选择工具和纠错；没有新建多 Agent。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from langchain.agents import AgentState
from langchain.tools import ToolRuntime, tool

from nl2sql_handbook.nl2sql_agent import AgentSession
from nl2sql_handbook.nl2sql_agent_join import JoinAgentQueryResult, JoinNL2SQLAgent
from nl2sql_handbook.nl2sql_demo import DEFAULT_DB, describe_error
from .retrieval import EVIDENCE_RULES, KnowledgeIndex


@dataclass
class RAGAgentResult(JoinAgentQueryResult):
    rag_enabled: bool = True
    retrieval_trace: list[dict] = field(default_factory=list)


class RAGAgent(JoinNL2SQLAgent):
    """通过增强工具观察接入证据，保留原生 tool calling 和所有查询限制。"""

    def __init__(self, db_path=DEFAULT_DB, *, knowledge_files=None, top_k=4,
                 max_context_chars=2600, rag_enabled=True, **kwargs):
        self.rag_enabled = rag_enabled
        self._db_path = db_path
        self._index_options = dict(knowledge_files=knowledge_files, top_k=top_k,
                                   max_context_chars=max_context_chars)
        super().__init__(db_path, **kwargs)

    def _build_tools(self):
        original = super()._build_tools()
        self.knowledge = None
        if not self.rag_enabled:
            return original
        self.knowledge = KnowledgeIndex(self.schema_catalog, list(self.catalog.relationships.values()),
                                        self._db_path, **self._index_options)
        original_schema = next(item for item in original if item.name == "get_schema").func

        @tool
        def get_schema(
            selection: dict[str, list[str]],
            runtime: ToolRuntime[AgentSession, AgentState],
            relationship_ids: list[str] | None = None,
        ) -> str:
            """选择真实表列和 JOIN 关系，返回 schema、关联计划及相关检索资料。

            selection 使用完整表名 -> 字段列表；桥接表和连接键自动补齐。
            自动检索原问题；需要更多术语/口径证据时可调用 search_knowledge。
            业务资料不能代替 get_schema，也不能扩大真实的 schema 和关系范围。
            """
            with self._lock:
                observation = json.loads(original_schema(
                    selection=selection, runtime=runtime, relationship_ids=relationship_ids,
                ))
                if observation.get("ok"):
                    observation["retrieval"] = self.knowledge.search(runtime.context.question)
                    observation["evidence_rules"] = EVIDENCE_RULES
                    # 原函数已记录一次观察；替换该观察，避免把同一次工具调用记两遍。
                    runtime.context.tool_trace[-1]["observation"] = observation
                return json.dumps(observation, ensure_ascii=False, default=str)

        @tool
        def search_knowledge(query: str, runtime: ToolRuntime[AgentSession, AgentState]) -> str:
            """按问题或术语检索表结构、关系、说明和已确认的 SQL 样例。

            可在选表前检索，也可重新措辞补查证据。返回正文、来源和相关度分数；
            没有命中不代表数据库没有答案。样例不是本次结果，不能直接照抄。
            每个问题最多主动检索 4 次；该工具不执行 SQL，也不返回业务记录。
            """
            with self._lock:
                session = runtime.context
                used = sum(event["tool"] == "search_knowledge" for event in session.tool_trace)
                if used >= 4:
                    observation = {"ok": False, "error": "主动检索额度已耗尽（最多 4 次）。"}
                else:
                    try:
                        observation = {"ok": True, "retrieval": self.knowledge.search(query),
                                       "evidence_rules": EVIDENCE_RULES}
                    except ValueError as exc:
                        observation = {"ok": False, "error": describe_error(exc)}
                return self._observe(session, "search_knowledge", {"query": query}, observation)

        return [get_schema if item.name == "get_schema" else item for item in original] + [search_knowledge]

    def ask(self, question, *, explain=False):
        result = super().ask(question, explain=explain)
        # 从本次运行上下文的观察收集，不在实例上共享不同问题的检索历史。
        trace = [event["observation"]["retrieval"] for event in result.tool_trace
                 if "retrieval" in event["observation"]]
        return RAGAgentResult(**asdict(result), rag_enabled=self.rag_enabled, retrieval_trace=trace)
