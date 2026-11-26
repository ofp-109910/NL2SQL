"""Supervisor-as-tools Multi-Agent：四个真实 create_agent 图，显式交接证据和结果。

每个角色有独立 messages 与工具集，所有角色沿用同一个 DeepSeek 模型/API 配置。
Python 负责预算、只读执行、版本绑定和成功条件；模型负责分工、查证、生成和核验。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Literal

import duckdb
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import AgentMiddleware, ModelCallLimitMiddleware
from langchain.tools import ToolRuntime, tool
from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek

from nl2sql_handbook.nl2sql_demo import (
    DEFAULT_DB, DEFAULT_KEY_FILE, DEFAULT_MODEL, EXPLANATION_MAX_ROWS, clean_sql,
    deepseek_request_options, describe_error, load_api_key,
)
from nl2sql_handbook.nl2sql_workflow_join import JOIN_RULES, JoinPlanningError
from nl2sql_handbook.rag_demo.retrieval import EVIDENCE_RULES
from . import prompts
from .runtime import DatabaseContext, RoleContext, TeamExecutionError, TeamResult, TeamSession


ROLES = ("supervisor", "retriever", "sql", "reviewer")
ROLE_TOOLS = {"supervisor": ["delegate_retrieval", "delegate_sql", "delegate_review"],
              "retriever": ["search_knowledge"], "sql": ["list_tables", "get_schema", "execute_sql"],
              "reviewer": ["get_candidate", "explain_candidate", "submit_review"]}


def text_reply(state) -> str:
    """仅提取公开的最终文本，不记录 additional_kwargs 中的隐藏推理。"""
    message = state["messages"][-1]
    if not isinstance(message, AIMessage) or message.tool_calls:
        return ""
    if isinstance(message.content, str):
        return message.content.strip()
    return "\n".join(item.get("text", "") for item in message.content
                     if isinstance(item, dict) and item.get("type") == "text").strip()


class SharedModelBudget(AgentMiddleware):
    """所有 Agent 共用每题模型预算，子图重复 invoke() 也不会重置。

    统计的是 SDK 模型调用尝试（包含失败），不包括 SDK 内部 HTTP 重试。
    工具调用轨迹另行记录，避免把模型消息或隐藏推理当成可观测日志。
    """

    def __init__(self, role, limit):
        self.role, self.limit = role, limit

    def wrap_model_call(self, request, handler):
        team = request.runtime.context.team
        with team.lock:
            if team.model_calls >= self.limit:
                raise RuntimeError("整个团队的模型调用额度已耗尽。")
            team.model_calls += 1
            team.calls_by_agent[self.role] = team.calls_by_agent.get(self.role, 0) + 1
        return handler(request)


class MultiAgentNL2SQL:
    """可以注入各角色的假模型做离线验证；正常情况下四角色共用 DeepSeek。

    同一个实例串行处理问题；不同 ask() 的状态独立。数据库连接上的工具调用
    串行加锁，因此即使模型在一轮发出多个 tool_calls，也不会突破总预算。
    """

    def __init__(self, db_path=DEFAULT_DB, *, key_file=DEFAULT_KEY_FILE, model=DEFAULT_MODEL,
                 max_rows=100, max_sql_retries=3, max_model_calls=40, max_delegations=10,
                 relationships_file=None, knowledge_files=None, top_k=4, max_context_chars=2600,
                 llm=None, llms=None):
        if not isinstance(max_sql_retries, int) or max_sql_retries not in range(4):
            raise ValueError("max_sql_retries 必须为 0—3。")
        if not isinstance(max_model_calls, int) or not 1 <= max_model_calls <= 100:
            raise ValueError("max_model_calls 必须为 1—100。")
        if not isinstance(max_delegations, int) or not 1 <= max_delegations <= 30:
            raise ValueError("max_delegations 必须为 1—30。")
        if llms is not None and (llm is not None or set(llms) != set(ROLES)):
            raise ValueError("llms 必须恰好包含四个角色，且不能同时传 llm。")
        self.max_sql_attempts = max_sql_retries + 1
        self.max_model_calls, self.max_delegations = max_model_calls, max_delegations
        self.database = DatabaseContext(
            Path(db_path), max_rows=max_rows, relationships_file=relationships_file,
            knowledge_files=knowledge_files, top_k=top_k, max_context_chars=max_context_chars,
        )
        try:
            if llms is None:
                if llm is None:
                    llm = ChatDeepSeek(model=model, api_key=load_api_key(Path(key_file)),
                                      **deepseek_request_options())
                llms = {role: llm for role in ROLES}
            self.role_tools = self._build_tools()
            # 同一个模型可以 bind 不同工具集；这是四个上下文独立的 Agent，不需四种模型。
            self.agents = {
                role: create_agent(
                    model=llms[role], tools=self.role_tools[role], name=role,
                    system_prompt=getattr(prompts, role.upper()) + "\n" + JOIN_RULES,
                    context_schema=RoleContext,
                    middleware=[SharedModelBudget(role, max_model_calls),
                                ModelCallLimitMiddleware(run_limit=12, exit_behavior="error")],
                ) for role in ROLES
            }
        except Exception:
            self.database.close()
            raise

    @staticmethod
    def _observe(context, name, arguments, observation):
        with context.team.lock:
            context.team.trace.append({"agent": context.role, "tool": name,
                                       "arguments": arguments, "observation": observation})
        return json.dumps(observation, ensure_ascii=False, default=str)

    def _invoke(self, role, team, assignment, last_review=None):
        context = RoleContext(team=team, role=role)
        payload = {"question": team.question, "assignment": assignment[:2000], "explain": team.explain,
                   "sql_attempts_remaining": self.max_sql_attempts - team.attempts}
        if role in {"sql", "reviewer"}:
            # 直接交接真实检索报告，不依赖主控或检索 Agent 是否准确转述。
            payload.update(evidence=team.evidence, evidence_rules=EVIDENCE_RULES,
                           last_review=last_review if last_review is not None else team.review)
        state = self.agents[role].invoke(
            {"messages": [{"role": "user", "content": json.dumps(payload, ensure_ascii=False)}]},
            context=context, config={"recursion_limit": 80},
        )
        return text_reply(state)

    def _candidate_info(self, team, *, full=False):
        result = team.candidate
        if result is None:
            return {"ok": False, "error": "当前没有成功执行的候选查询。"}
        info = {"ok": True, "candidate_id": team.candidate_id, "sql": result.sql,
                "columns": result.columns, "returned_rows": len(result.rows),
                "truncated": result.truncated, "explain": team.explain}
        if full:
            info.update(question=team.question, schema=team.candidate_schema,
                        join_plan=team.candidate_plan, evidence=team.evidence,
                        evidence_version=team.evidence_version)
        if team.explain:
            info.update(rows_preview=result.rows[:EXPLANATION_MAX_ROWS],
                        sampled=len(result.rows) > EXPLANATION_MAX_ROWS)
        return info

    def _delegate(self, role, assignment, context):
        team = context.team
        # 委派锁与状态锁分开：子图的工具可能在另一个工作线程执行，不能让主控
        # 持有状态锁等待子图，否则工具无法取得同一状态锁。委派锁只由主控工具持有。
        with team.delegation_lock:
            name = {"retriever": "delegate_retrieval", "sql": "delegate_sql", "reviewer": "delegate_review"}[role]
            with team.lock:
                last_review = team.review
                # 即使新任务因额度不足而没有启动，也不能以旧结果冒充这次新任务。
                if role == "sql":
                    team.candidate, team.review = None, None
                elif role == "reviewer":
                    team.review = None
                if team.delegations >= self.max_delegations:
                    return self._observe(context, name, {"assignment": assignment},
                                         {"ok": False, "error": "团队委派额度已耗尽。"})
                team.delegations += 1
                if role == "sql":
                    if not team.evidence:
                        return self._observe(context, name, {}, {"ok": False, "error": "先委派检索 Agent 实际检索。"})
                    if team.attempts >= self.max_sql_attempts:
                        return self._observe(context, name, {}, {"ok": False, "error": "整个团队的 SQL 额度已耗尽。"})
                if role == "reviewer":
                    if team.candidate is None:
                        return self._observe(context, name, {}, {"ok": False, "error": "先取得真实 SQL 候选。"})
                count_before = len(team.evidence)
            error, reply = None, ""
            try:
                reply = self._invoke(role, team, assignment, last_review=last_review)
            except Exception as exc:
                error = describe_error(exc)[:2000]
            if role == "retriever":
                observation = {"ok": len(team.evidence) > count_before,
                               "evidence_version": team.evidence_version,
                               "source_ids": sorted({hit["id"] for report in team.evidence for hit in report["hits"]})}
            elif role == "sql":
                observation = self._candidate_info(team)
                observation["attempts"] = team.attempts
                if not observation["ok"] and team.retry_history:
                    observation["last_error"] = team.retry_history[-1]
            else:
                observation = {"ok": team.review is not None, "review": team.review}
            observation.update(agent_message=reply[:2000], agent_error=error)
            return self._observe(context, name, {"assignment": assignment[:2000]}, observation)

    def _build_tools(self):
        @tool
        def delegate_retrieval(assignment: str, runtime: ToolRuntime[RoleContext, AgentState]) -> str:
            """委派检索 Agent 查找结构、关系、说明和样例；补充真实证据后再生成或核验 SQL。"""
            return self._delegate("retriever", assignment, runtime.context)

        @tool
        def delegate_sql(assignment: str, runtime: ToolRuntime[RoleContext, AgentState]) -> str:
            """委派 SQL Agent 依据原问题和证据选表、规划 JOIN、执行或修订查询。新委派作废旧候选。"""
            return self._delegate("sql", assignment, runtime.context)

        @tool
        def delegate_review(assignment: str, runtime: ToolRuntime[RoleContext, AgentState]) -> str:
            """委派独立核验 Agent 检查当前真实 SQL，返回 approved/revise/clarify。"""
            return self._delegate("reviewer", assignment, runtime.context)

        @tool
        def search_knowledge(query: str, runtime: ToolRuntime[RoleContext, AgentState]) -> str:
            """本地 BM25 检索；可改写检索词，全团队每题最多实际检索 4 次，不执行 SQL。"""
            context, team = runtime.context, runtime.context.team
            with team.lock:
                try:
                    if len(team.evidence) >= 4:
                        raise ValueError("整个团队的检索额度已耗尽（4 次）。")
                    report = self.database.knowledge.search(query)
                    team.evidence.append(report)
                    team.evidence_version += 1
                    team.review = None  # 新证据必须重新核验，不能沿用旧批准。
                    observation = {"ok": True, "retrieval": report, "evidence_rules": EVIDENCE_RULES}
                except ValueError as exc:
                    observation = {"ok": False, "error": describe_error(exc)}
                return self._observe(context, "search_knowledge", {"query": query}, observation)

        @tool
        def list_tables(runtime: ToolRuntime[RoleContext, AgentState]) -> str:
            """读取真实表列目录与关系；不读取业务记录。"""
            with runtime.context.team.lock:
                return self._observe(runtime.context, "list_tables", {}, {"ok": True, "tables": {
                    table: [column["column"] for column in columns]
                    for table, columns in self.database.schema_catalog.items()
                }, "relationships": list(self.database.catalog.relationships.values())})

        @tool
        def get_schema(selection: dict[str, list[str]], runtime: ToolRuntime[RoleContext, AgentState],
                       relationship_ids: list[str] | None = None) -> str:
            """获取所选真实字段与 JOIN 计划，selection 是完整表名 -> 字段列表；补齐桥接表和键。"""
            context = runtime.context
            with context.team.lock:
                context.plan = None
                try:
                    if not selection:
                        raise JoinPlanningError("至少选择一张表和相关字段。")
                    context.plan = self.database.catalog.plan(selection, relationship_ids)
                    observation = {"ok": True, "schema": context.plan.schema,
                                   "join_plan": context.plan.summary(), "join_rules": JOIN_RULES}
                except ValueError as exc:
                    observation = {"ok": False, "error": describe_error(exc)}
                return self._observe(context, "get_schema", {"selection": selection,
                                      "relationship_ids": relationship_ids}, observation)

        @tool
        def execute_sql(sql: str, runtime: ToolRuntime[RoleContext, AgentState]) -> str:
            """执行单条只读 SQL；全团队共享首次执行加最多 3 次额外调用，包括校验失败和成功查询。"""
            context, team = runtime.context, runtime.context.team
            with team.lock:
                team.candidate, team.review = None, None
                if team.attempts >= self.max_sql_attempts:
                    return self._observe(context, "execute_sql", {"sql": sql},
                                         {"ok": False, "error": "SQL 总额度已耗尽。", "retry_allowed": False})
                team.attempts += 1
                try:
                    if context.plan is None:
                        raise ValueError("先用 get_schema 获取本次子任务的真实结构。")
                    result = self.database.execute_plan(clean_sql(sql), context.plan, team.question)
                    team.candidate = result
                    team.candidate_id = team.attempts
                    team.candidate_plan = context.plan.summary()
                    team.candidate_schema = context.plan.schema
                    observation = self._candidate_info(team)
                except (ValueError, duckdb.Error) as exc:
                    error = describe_error(exc)[:2000]
                    team.retry_history.append({"sql": sql, "error": error})
                    observation = {"ok": False, "error": error}
                observation.update(attempts=team.attempts, retry_allowed=team.attempts < self.max_sql_attempts)
                return self._observe(context, "execute_sql", {"sql": sql}, observation)

        @tool
        def get_candidate(runtime: ToolRuntime[RoleContext, AgentState]) -> str:
            """查看当前真实候选 SQL、schema、关系、证据与执行元数据；解释开关决定是否返回行预览。"""
            context, team = runtime.context, runtime.context.team
            with team.lock:
                observation = self._candidate_info(team, full=True)
                if observation["ok"]:
                    context.inspected_candidate = team.candidate_id
                    context.inspected_evidence = team.evidence_version
                return self._observe(context, "get_candidate", {}, observation)

        @tool
        def explain_candidate(runtime: ToolRuntime[RoleContext, AgentState]) -> str:
            """仅 EXPLAIN 当前已经成功执行的候选，查看计划而不重跑查询；每次核验最多两次。"""
            context, team = runtime.context, runtime.context.team
            with team.lock:
                try:
                    if team.candidate is None or context.plan_checks >= 2:
                        raise ValueError("没有当前候选，或本次核验的 EXPLAIN 额度已耗尽。")
                    context.plan_checks += 1
                    rows = self.database.connection.execute("EXPLAIN " + team.candidate.sql).fetchall()
                    observation = {"ok": True, "candidate_id": team.candidate_id,
                                   "plan": str(rows)[:4000], "executed": False}
                except (ValueError, duckdb.Error) as exc:
                    observation = {"ok": False, "error": describe_error(exc)}
                return self._observe(context, "explain_candidate", {}, observation)

        @tool
        def submit_review(candidate_id: int, status: Literal["approved", "revise", "clarify"],
                          feedback: str, runtime: ToolRuntime[RoleContext, AgentState]) -> str:
            """提交当前候选的核验意见。必须先 get_candidate；新 SQL/新证据会使旧核验失效。"""
            context, team = runtime.context, runtime.context.team
            with team.lock:
                if (team.candidate is None or candidate_id != team.candidate_id
                    or context.inspected_candidate != candidate_id
                    or context.inspected_evidence != team.evidence_version or not feedback.strip()):
                    observation = {"ok": False, "error": "候选或证据已变化、未读取候选，或反馈为空；请重新核验。"}
                else:
                    team.review = {"candidate_id": candidate_id, "evidence_version": team.evidence_version,
                                   "status": status, "feedback": feedback[:2000]}
                    observation = {"ok": True, "review": team.review}
                return self._observe(context, "submit_review", {"candidate_id": candidate_id,
                                      "status": status, "feedback": feedback[:2000]}, observation)

        return {"supervisor": [delegate_retrieval, delegate_sql, delegate_review],
                "retriever": [search_knowledge], "sql": [list_tables, get_schema, execute_sql],
                "reviewer": [get_candidate, explain_candidate, submit_review]}

    def ask(self, question: str, *, explain=False) -> TeamResult:
        question = question.strip()
        if not question:
            raise ValueError("问题不能为空。")
        team, start = TeamSession(question=question, explain=explain), perf_counter()
        error, reply = None, ""
        try:
            reply = self._invoke("supervisor", team, "协调检索、SQL 和核验，完整回答原问题。")
        except Exception as exc:
            error = describe_error(exc)
        review = team.review
        if (team.candidate is None or review is None or review["status"] != "approved"
            or review["candidate_id"] != team.candidate_id or review["evidence_version"] != team.evidence_version):
            # 成功/失败由真实状态决定，不能用主控最后一句“成功”作为失败原因或成功证明。
            reason = error or (f"核验意见 {review['status']}：{review['feedback']}" if review else
                               "缺少当前成功查询或对当前 SQL/证据的明确核验批准。")
            raise TeamExecutionError(team, reason)
        result = TeamResult(**asdict(team.candidate), join_plan=team.candidate_plan,
                            review=review, retrieval_trace=team.evidence, agent_trace=team.trace,
                            model_calls=team.model_calls, calls_by_agent=team.calls_by_agent,
                            delegations=team.delegations, elapsed_ms=round((perf_counter() - start) * 1000, 2),
                            agent_message=reply, agent_error=error)
        result.attempts, result.retry_history = team.attempts, team.retry_history
        if explain:
            if reply and error is None:
                result.explanation = reply
            else:
                result.explanation_error = error or "主控未返回结果解释。"
        return result

    def close(self):
        self.database.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
