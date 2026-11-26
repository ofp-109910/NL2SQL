"""固定检索的 Workflow：Retrieve -> 原 JOIN 选表/生成/纠错 -> 真实结果。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

from nl2sql_handbook.nl2sql_demo import DEFAULT_DB
from nl2sql_handbook.nl2sql_workflow_join import JoinNL2SQLWorkflow, JoinQueryResult
from .retrieval import KnowledgeIndex, evidence_text


@dataclass
class RAGWorkflowResult(JoinQueryResult):
    rag_enabled: bool = True
    retrieval_trace: list[dict] = field(default_factory=list)


class RAGWorkflow(JoinNL2SQLWorkflow):
    """只增加检索上下文；SQL 校验、纠错额度、连接与解释复用上一阶段。

    前一阶段 Workflow 本身是有状态的，本类同样适合串行 ask()。每次调用会
    清空检索状态，避免下一问题复用上一问题的资料；并发服务应每请求建实例。
    """

    def __init__(self, db_path=DEFAULT_DB, *, knowledge_files=None, top_k=4,
                 max_context_chars=2600, rag_enabled=True, **kwargs):
        self.rag_enabled = rag_enabled
        self._db_path = db_path
        self._index_options = dict(knowledge_files=knowledge_files, top_k=top_k,
                                   max_context_chars=max_context_chars)
        self._evidence = None
        super().__init__(db_path, **kwargs)

    def _build_chains(self, llm):
        def add_evidence(prompt):
            # 输入是 ChatPromptValue，输出是消息列表。保留原始系统规则和问题，
            # 在模型之前追加一个独立的数据消息，不篡改原问题或 schema 字符串。
            messages = prompt.to_messages()
            if self._evidence is not None:
                messages.append(HumanMessage(content=evidence_text(self._evidence)))
            return messages

        # 父类通过 prompt | llm 组合链；这里传入可组合的 Runnable，三个阶段
        # （选表、SQL/纠错、可选解释）因而都能看到同一份有来源的证据。
        enriched_model = RunnableLambda(add_evidence) | llm
        super()._build_chains(enriched_model)
        self.knowledge = KnowledgeIndex(self.schema_catalog, list(self.catalog.relationships.values()),
                                        self._db_path, **self._index_options) if self.rag_enabled else None

    def ask(self, question, *, explain=False):
        self._evidence = None
        try:
            if self.knowledge is not None:
                self._evidence = self.knowledge.search(question)
            result = super().ask(question, explain=explain)
            return RAGWorkflowResult(**asdict(result), rag_enabled=self.rag_enabled,
                                     retrieval_trace=[self._evidence] if self._evidence is not None else [])
        finally:
            self._evidence = None
