# 架构与代码阅读顺序

入口先读取项目配置，再选择已有实现；统一查询在独立工作进程内执行，HTTP 与评估器有各自的版本注册表。

```text
.env / 环境变量 → config.py
                        ↓
             cli.py → 阶段入口
                │
                └─ query → 工作进程 + 硬超时 + 计量
                              ↓
                  Workflow / Agent / RAG / 多角色
                              ↓
                      只读 DuckDB 连接
                              ↓
                   实际 SQL、结果、公开过程记录
```

## 建议阅读顺序

1. [minimal.py](../src/nl2sql_handbook/minimal.py)：一次生成与本地执行。
2. [nl2sql_demo.py](../src/nl2sql_handbook/nl2sql_demo.py)：结构读取、只读边界、选表与错误纠正。
3. [nl2sql_agent.py](../src/nl2sql_handbook/nl2sql_agent.py)：工具、Agent 图和执行记录。
4. [nl2sql_workflow_join.py](../src/nl2sql_handbook/nl2sql_workflow_join.py)：可信关系、桥接表与 JOIN 校验。
5. [rag_demo/retrieval.py](../src/nl2sql_handbook/rag_demo/retrieval.py)：BM25 检索，不依赖向量服务。
6. [workflow_v2/workflow.py](../src/nl2sql_handbook/workflow_v2/workflow.py)：需求摘要、结构补齐、复核和有界回退。
7. [multi_agent_demo/team.py](../src/nl2sql_handbook/multi_agent_demo/team.py)：多个角色的职责与候选版本检查。
8. [evaluation_demo/grading.py](../src/nl2sql_handbook/evaluation_demo/grading.py)：独立执行评分；不把模型自检作为正确性标签。
9. [ai4db_demo/engine.py](../src/nl2sql_handbook/ai4db_demo/engine.py)：完整结果比较与交替顺序的性能测量。
10. [delivery_demo/server.py](../src/nl2sql_handbook/delivery_demo/server.py)：HTTP 参数、并发、期限与子进程回收。

## 为什么保留不同版本

本项目展示实现如何逐步扩展，因此基础与升级 Workflow 都保留。升级改动同时涉及提示词、信息补齐、回退和复核，历史分数不能归因于某一个模块，也不能推出 Workflow 与 Agent 的普遍优劣。

包内按查询生成、工具调用、检索、评估与服务等功能划分模块。共用配置集中在 `config.py`；数据、日志、报告统一写入 `outputs`。业务知识 JSON 和 SQL 工作负载是只读包资源，随 wheel 一起安装。

新用户从 `nl2sql query` 开始，学习者沿各阶段入口阅读。公共命令、安装路径和内部模块路径的区别见 [commands](commands.md)。
