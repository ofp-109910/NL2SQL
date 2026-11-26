#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多表 JOIN 增强版 Agent：不修改 nl2sql_agent.py。

示例（先激活原 langchain_env）：
  python nl2sql_agent_join.py -q "按城市和商品类别统计已完成订单销售额"
  python nl2sql_agent_join.py --explain -q "列出购买过数码商品的客户及商品名称"
  python nl2sql_agent_join.py --show-relationships
  python nl2sql_agent_join.py --db /absolute/path/business.duckdb -q "你的问题"

仍使用 LangChain create_agent、ChatDeepSeek、deepseek-flash 和 duckdb 包，
API 地址、密钥文件和查询限制沿用原 Agent。模型自主选择工具和纠错，
新的 get_schema 会依据真实外键/显式配置补齐桥接表及连接键。

与 nl2sql_workflow_join.py 共享关系目录、路径规划和 JOIN 校验，不需要新依赖。
没有外键时可用 --relationships-file，格式见 --help。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import duckdb
from langchain.agents import AgentState
from langchain.tools import ToolRuntime, tool

from nl2sql_handbook.nl2sql_agent import AgentQueryResult, AgentSession, NL2SQLAgent
from nl2sql_handbook.nl2sql_demo import DEFAULT_DB, clean_sql, describe_error
from nl2sql_handbook.nl2sql_workflow_join import (
    JOIN_RULES, JoinCatalog, JoinPlanningError, run_join_cli, validate_join_sql,
)


@dataclass
class JoinAgentQueryResult(AgentQueryResult):
    join_plan: dict[str, Any] = field(default_factory=dict)


class JoinNL2SQLAgent(NL2SQLAgent):
    """增强工具而非固定 Agent 的执行顺序，复用原 Agent 图和运行边界。"""

    def __init__(self, db_path=DEFAULT_DB, *, relationships_file=None, **kwargs):
        self.relationships_file = relationships_file
        # 父类构造时会调用这里覆盖的 _build_tools，然后创建真正的 create_agent。
        # 不使用 Workflow 的选表链或 SQL 生成链，模型仍通过原生 tool_calls 决策。
        super().__init__(db_path, **kwargs)

    @staticmethod
    def _current_plan(session):
        # 使用本次会话最近一次成功获取的计划，不在 Agent 实例上共享问题状态。
        for event in reversed(session.tool_trace):
            if event["tool"] == "get_schema" and event["observation"].get("ok"):
                return event["observation"]["join_plan"]
        return {"relationships": [], "relationship_ids": [], "bridge_tables": []}

    def _build_tools(self):
        self.catalog = JoinCatalog(self.connection, self.schema_catalog, self.relationships_file)
        original_tools = {item.name: item for item in super()._build_tools()}
        # 调用原工具的 Python 函数，保留原只读执行、返回行数、预览和额度逻辑。
        # runtime 由框架注入，不是允许模型伪造的普通工具参数。
        original_execute = original_tools["execute_sql"].func

        @tool
        def list_tables(runtime: ToolRuntime[AgentSession, AgentState]) -> str:
            """查看表列目录和真实外键/显式配置关系，供多表查询选择关联路径。"""
            with self._lock:
                return self._observe(runtime.context, "list_tables", {}, {
                    "ok": True,
                    "tables": {
                        table: [column["column"] for column in columns]
                        for table, columns in self.schema_catalog.items()
                    },
                    "relationships": list(self.catalog.relationships.values()),
                })

        @tool
        def get_schema(
            selection: dict[str, list[str]],
            runtime: ToolRuntime[AgentSession, AgentState],
            relationship_ids: list[str] | None = None,
        ) -> str:
            """选择业务表列，补齐桥接表、完整连接键，并返回 JOIN 关系计划。

            selection 是完整表名到字段列表的映射，例如
            {"main.customers": ["city"], "main.products": ["category"]}。
            若存在唯一关联路径，将自动补入 orders 等桥接表及两侧连接键。
            多条角色/路径有歧义时，用 relationship_ids 指定目录里的明确关系 ID。
            SQL 失败后可重新选择；新选择替换旧选择，需包含所需的全部业务字段。
            """
            with self._lock:
                session = runtime.context
                try:
                    if not selection:
                        raise JoinPlanningError("请至少选择一张表及相关字段。")
                    plan = self.catalog.plan(selection, relationship_ids)
                    session.selected_schema = plan.schema
                    observation = {
                        "ok": True, "schema": plan.schema, "join_plan": plan.summary(),
                        "join_rules": JOIN_RULES,
                    }
                except JoinPlanningError as exc:
                    observation = {"ok": False, "error": describe_error(exc)}
                return self._observe(session, "get_schema", {
                    "selection": selection, "relationship_ids": relationship_ids,
                }, observation)

        @tool
        def execute_sql(sql: str, runtime: ToolRuntime[AgentSession, AgentState]) -> str:
            """按已获取的 JOIN 计划执行单条只读 SQL，返回结果或可纠错的错误。

            使用显式 ON/USING，直接表 JOIN 必须使用计划关系的全部连接键，
            不允许笛卡尔积。LEFT JOIN 保留未匹配行时注意右表过滤放在 ON；
            多个一对多事实表须防止重复累计，必要时先聚合后连接。
            每次调用（含校验失败）消耗额度。开启解释时才向模型返回结果预览。
            """
            with self._lock:
                session = runtime.context
                plan = self._current_plan(session)
                # 无 schema/额度耗尽等情况由原工具统一处理，不能绕过旧边界。
                if session.attempts < self.max_sql_attempts and session.selected_schema:
                    try:
                        validate_join_sql(
                            self.connection, clean_sql(sql), self.schema_catalog, plan["relationships"],
                        )
                    except (duckdb.Error, ValueError) as exc:
                        session.attempts += 1
                        session.result = None
                        error = describe_error(exc)[:2000]
                        session.retry_history.append({"sql": sql, "error": error})
                        return self._observe(session, "execute_sql", {"sql": sql}, {
                            "ok": False, "error": error, "attempts": session.attempts,
                            "retry_allowed": session.attempts < self.max_sql_attempts,
                        })

                # RLock 支持此处调用同样持锁的原工具；并行调用仍不能突破总额度。
                observation = json.loads(original_execute(sql=sql, runtime=runtime))
                if observation.get("ok"):
                    # 保存“实际执行这条 SQL 时”的计划快照，避免后续重新选表混淆结果。
                    observation["join_plan"] = plan
                    session.tool_trace[-1]["observation"] = observation
                return json.dumps(observation, ensure_ascii=False, default=str)

        return [list_tables, get_schema, execute_sql]

    def ask(self, question, *, explain=False):
        result = super().ask(question, explain=explain)
        plan = {}
        for event in reversed(result.tool_trace):
            if event["tool"] == "execute_sql" and event["observation"].get("ok"):
                plan = event["observation"]["join_plan"]
                break
        return JoinAgentQueryResult(**asdict(result), join_plan=plan)


def main():
    return run_join_cli(JoinNL2SQLAgent, agent=True)


if __name__ == "__main__":
    raise SystemExit(main())
