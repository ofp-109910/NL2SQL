# 命令与版本

完成 README 安装后统一用 `nl2sql`。每个子命令后加 `--help` 查看参数；`python -m nl2sql_handbook` 与 `nl2sql` 等价。

## 统一查询

```bash
nl2sql query --variant workflow_v2 -q "已完成订单有多少笔？"
nl2sql query --variant agent --timeout 90 --max-model-calls 12 -q "已完成订单有多少笔？"
```

统一入口支持十个实现，默认 `workflow_v2`：

| 标识 | 实现 |
|---|---|
| `workflow` | 固定选表、生成、执行错误纠正 |
| `agent` | 自主选择查结构与执行 SQL 的工具 |
| `join_workflow` | 显式关系规划与 JOIN 校验 |
| `join_agent` | 带关系工具的 Agent |
| `rag_workflow` | 固定检索后执行 Workflow |
| `rag_agent` | Agent 使用业务资料检索工具 |
| `multi_agent` | 检索、SQL、复核与调度角色 |
| `workflow_v2` | 结构补齐、语义复核与回退 |
| `join_workflow_v2` | 升级版显式关系选择 |
| `rag_workflow_v2` | 结构回退时刷新检索 |

这些查询都会调用真实模型。成功输出 `ok`、`variant`、实际 `sql/columns/rows` 和计量，失败返回非零退出码。默认限制：24 次逻辑模型调用、4 次 SQL 尝试、120 秒硬超时、显示 100 行。SDK 内部网络重试不等同于逻辑调用次数。`query` 不额外开启自然语言解释。

## 阶段入口

各功能入口直接调用对应实现，便于独立运行和调试。它们保留各阶段参数，**不全部具有 `query` 外层的进程硬超时**。

| 命令 | 用途 | 远端模型 |
|---|---|---|
| `nl2sql minimal --offline` | 最小示例 | 否 |
| `nl2sql offline` | 四种流程的脚本回复演示 | 否 |
| `nl2sql workflow -q "问题"` | 基础 Workflow | 是 |
| `nl2sql agent -q "问题"` | 基础 Agent | 是 |
| `nl2sql join-workflow -q "问题"` | JOIN Workflow | 是 |
| `nl2sql join-agent -q "问题"` | JOIN Agent | 是 |
| `nl2sql rag --mode workflow -q "问题"` | RAG Workflow | 是 |
| `nl2sql rag --mode agent -q "问题"` | RAG Agent | 是 |
| `nl2sql rag --retrieve-only -q "销售额怎么算"` | 检索诊断 | 否 |
| `nl2sql multi-agent -q "问题"` | 多角色协作 | 是 |
| `nl2sql multi-agent --show-architecture` | 角色与工具说明 | 否 |
| `nl2sql workflow-v2 --variant basic -q "问题"` | 升级版详细入口 | 是 |

详细入口的升级版 `--variant` 是 `basic/join/rag`，不同于统一 `query` 的标识。先运行 `nl2sql init` 准备默认数据库。需要自然语言解释时，使用支持 `--explain` 的阶段入口；升级版即使关闭解释，也会在语义复核中发送采样预览。

## 自建数据库

```bash
nl2sql query --variant rag_workflow_v2 \
  --db /path/to/business.duckdb \
  --knowledge-file /path/to/knowledge.json \
  -q "你的业务问题"
```

资料格式见 [sales.json](../src/nl2sql_handbook/rag_demo/knowledge/sales.json)。结构从数据库读取，关系依赖已声明外键或显式关系文件，不根据同名列猜测。没有外键时可用详细入口：

```bash
nl2sql workflow-v2 --variant join \
  --db /path/to/business.duckdb \
  --relationships-file /path/to/relationships.json \
  -q "你的业务问题"
```

关系字段与校验见 [JoinCatalog](../src/nl2sql_handbook/nl2sql_workflow_join.py)。

## 离线复盘与 AI4DB

```bash
nl2sql init --evaluation --transfer
nl2sql review --controlled-only
nl2sql ai4db --init-only
nl2sql ai4db --analyze-only
nl2sql ai4db --candidate-file src/nl2sql_handbook/ai4db_demo/workloads/merged.sql
```

AI4DB 默认生成 20 万订单，离线分析也会实际测量 SQL，性能以本机报告为准。评估与升级版复评默认调用模型，完整用法见 [benchmarks](../benchmarks/README.md)。
