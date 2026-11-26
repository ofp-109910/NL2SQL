# 运行示例

先按项目 README 安装，所有命令从仓库根目录执行。这里的短脚本调用安装包，完整实现位于 `src/nl2sql_handbook`。

| 脚本 | 功能 |
|---|---|
| `minimal.py` | 一次生成与本地执行 |
| `workflow.py` | 带复核和回退的 Workflow |
| `agent.py` | 工具调用 Agent |
| `join.py` | 多表关系与 JOIN 校验 |
| `rag.py` | 业务资料检索与查询 |
| `multi_agent.py` | 多角色查询流程 |
| `evaluate.py` | 多实现统一评估 |
| `failure_review.py` | 失败复盘与故障注入 |
| `ai4db.py` | SQL 优化与性能实验 |
| `serve.py` | 本地 HTTP 服务 |

```bash
python examples/minimal.py --offline
python examples/workflow.py --help
python examples/evaluate.py --prepare-only
python examples/failure_review.py --controlled-only
```

完整参数与版本说明见 [命令文档](../docs/commands.md)。`--help` 不调用模型，其余入口取决于所选模式。默认评估会调用模型，第一次使用先选择开发集单题。
