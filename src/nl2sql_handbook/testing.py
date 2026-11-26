"""用于离线演示的脚本模型；实际 Agent 图、工具和 DuckDB 仍会执行。"""
import itertools
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import Field

class ToolCallingModel(FakeMessagesListChatModel):
    """只替代 DeepSeek 的响应，Agent 图、工具路由和数据库均使用真实实现。"""

    requests: list[list[BaseMessage]] = Field(default_factory=list)
    bound_tool_names: list[str] = Field(default_factory=list)
    fail_on_call: int | None = None

    def bind_tools(self, tools, **kwargs):
        self.bound_tool_names = [item.name for item in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.requests.append(list(messages))
        if self.fail_on_call == len(self.requests):
            raise RuntimeError("模拟模型 API 暂时不可用")
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


CALL_IDS = itertools.count()
ORDER_SCHEMA = {"main.orders": ["order_id"]}
SALES_SCHEMA = {"main.orders": ["quantity", "unit_price", "status"]}


def action(name, arguments):
    return AIMessage(content="", tool_calls=[{
        "name": name, "args": arguments, "id": f"call_{next(CALL_IDS)}", "type": "tool_call",
    }])


def schema_action(selection=None):
    return action("get_schema", {"selection": ORDER_SCHEMA if selection is None else selection})


def query_action(sql="SELECT COUNT(*) AS count FROM orders"):
    return action("execute_sql", {"sql": sql})
