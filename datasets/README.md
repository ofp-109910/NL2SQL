# 数据与业务资料

本仓库分发生成脚本和小型业务资料，运行后在 `outputs/data` 生成数据库，不打包作者的本地数据库。

| 数据 | 生成命令 | 默认位置 |
|---|---|---|
| 10 笔订单的最小示例 | `nl2sql init` | `outputs/data/demo.duckdb` |
| 销售/图书馆/学校，两个实例 | `nl2sql init --evaluation` | `outputs/data/evaluation/` |
| 仓储诊断，两个实例 | `nl2sql init --transfer` | `outputs/data/transfer/` |
| AI4DB 主负载与边界数据 | `nl2sql ai4db --init-only` | `outputs/data/ai4db/` |

生成逻辑分别见 [基本示例](../src/nl2sql_handbook/nl2sql_demo.py)、[评估夹具](../src/nl2sql_handbook/evaluation_demo/dataset.py)、[仓储夹具](../src/nl2sql_handbook/workflow_v2/transfer.py)、[性能夹具](../src/nl2sql_handbook/ai4db_demo/fixture.py)。

基础 RAG 业务定义为 [sales.json](../src/nl2sql_handbook/rag_demo/knowledge/sales.json)，随安装包提供。评估业务定义由夹具生成器写入各领域 `*_knowledge.json`。它们描述业务口径，不是整库业务记录的复制。

这些数据均为合成数据。最小示例和评估实例虽然都含订单表，但数据量、空值和匹配关系不同，不要跨库套用某一个示例数字。
