"""模型之外的运行边界：只读数据库、每题状态、执行结果与失败信息。

这些对象通过 ToolRuntime 注入，不是模型能够伪造的 JSON 工具参数。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any

from nl2sql_handbook.nl2sql_demo import NL2SQLDemo, QueryResult, connect_read_only, read_schema
from nl2sql_handbook.nl2sql_workflow_join import JoinCatalog, JoinPlan, validate_join_sql
from nl2sql_handbook.rag_demo.retrieval import KnowledgeIndex


class DatabaseContext(NL2SQLDemo):
    """只复用上一阶段的无模型执行能力，不构造 Workflow/ChatDeepSeek。

    父类 execute_sql()、close() 和 with 支持只依赖 connection/max_rows。
    本类不调用父类构造函数；选表与代理编排由团队负责。
    """

    def __init__(self, db_path: Path, *, max_rows=100, relationships_file=None,
                 knowledge_files=None, top_k=4, max_context_chars=2600):
        if not isinstance(max_rows, int) or max_rows < 1:
            raise ValueError("max_rows 必须为正整数。")
        self.max_rows = max_rows
        self.connection = connect_read_only(Path(db_path))
        try:
            self.schema = read_schema(self.connection)
            self.schema_catalog = json.loads(self.schema)
            self.catalog = JoinCatalog(self.connection, self.schema_catalog, relationships_file)
            self.knowledge = KnowledgeIndex(
                self.schema_catalog, list(self.catalog.relationships.values()), db_path,
                knowledge_files=knowledge_files, top_k=top_k, max_context_chars=max_context_chars,
            )
        except Exception:
            self.connection.close()
            raise

    def execute_plan(self, sql: str, plan: JoinPlan, question: str) -> QueryResult:
        validate_join_sql(self.connection, sql, self.schema_catalog, plan.relationships)
        result = super().execute_sql(sql, question=question)
        result.selected_schema = {
            table: [column["column"] for column in columns] for table, columns in plan.schema.items()
        }
        return result


@dataclass
class TeamSession:
    """一次用户提问的共享状态；预算不会随子 Agent 重启而重置。"""

    question: str
    explain: bool = False
    lock: Any = field(default_factory=RLock, repr=False)
    delegation_lock: Any = field(default_factory=RLock, repr=False)
    model_calls: int = 0
    calls_by_agent: dict[str, int] = field(default_factory=dict)
    delegations: int = 0
    attempts: int = 0
    retry_history: list[dict] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)
    evidence_version: int = 0
    candidate: QueryResult | None = None
    candidate_id: int = 0
    candidate_plan: dict = field(default_factory=dict)
    candidate_schema: dict = field(default_factory=dict)
    review: dict | None = None
    trace: list[dict] = field(default_factory=list)


@dataclass
class RoleContext:
    """一次子任务的私有上下文，工具负责控制该角色能读取哪些共享字段。

    每次委派新建此对象；模型不会接收其他角色完整的 messages 历史。
    """

    team: TeamSession
    role: str
    plan: JoinPlan | None = None
    inspected_candidate: int | None = None
    inspected_evidence: int | None = None
    plan_checks: int = 0


@dataclass
class TeamResult(QueryResult):
    join_plan: dict = field(default_factory=dict)
    review: dict = field(default_factory=dict)
    retrieval_trace: list[dict] = field(default_factory=list)
    agent_trace: list[dict] = field(default_factory=list)
    model_calls: int = 0
    calls_by_agent: dict[str, int] = field(default_factory=dict)
    delegations: int = 0
    elapsed_ms: float = 0
    agent_message: str = ""
    agent_error: str | None = None


class TeamExecutionError(RuntimeError):
    """未完成执行与核验协议时明确失败；保留轨迹，而不冒充查询成功。"""

    def __init__(self, team: TeamSession, reason: str):
        self.attempts = team.attempts
        self.history = list(team.retry_history)
        self.agent_trace = list(team.trace)
        self.model_calls = team.model_calls
        self.review = team.review
        self.candidate_sql = team.candidate.sql if team.candidate else None
        super().__init__(f"Multi-Agent 未完成当前查询的执行与核验：{reason}")
