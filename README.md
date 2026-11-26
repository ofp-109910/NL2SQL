# NL2SQL Handbook

用 LangChain、DeepSeek 和 DuckDB，逐步实现自然语言查询、Workflow 纠错、Agent 工具调用、多表关联、RAG、评估和本地 HTTP 服务。

这是一个可独立运行的 NL2SQL 实验项目，包含基础实现与加入结构补齐、语义复核和回退的 Workflow 升级版。第一次使用，先运行离线示例，再配置模型。

- [配置规则](docs/configuration.md) · [命令与版本](docs/commands.md)
- [评估与历史成绩](benchmarks/README.md) · [HTTP 服务](docs/http.md)
- [架构与代码阅读顺序](docs/architecture.md) · [常见问题](docs/troubleshooting.md)
- [迁移与验证记录](docs/verification.md)

## 1. 安装

下载并解压仓库，或用 Git 克隆后，进入包含本 README 的 `nl2sql-handbook` 根目录。验证环境为 **Linux x86_64、Python 3.10**；代码声明 Python 3.10 及以上，其他系统和 Python 版本尚未完成验收。

Linux / macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell 对应命令：

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
```

激活后执行：

```bash
python -m pip install -r requirements.lock
python -m pip install --no-build-isolation --no-deps -e .
python -m pip check
nl2sql doctor
```

锁定文件包含运行依赖、传递依赖和构建工具。`-e .` 安装当前源码与 `nl2sql` 命令。`doctor` 显示版本、隐藏绝对路径后的输出目录和是否配置密钥；**不会打印密钥，也不会连接模型**。第一次显示 `api_key_configured: false` 是正常的。仅在本地排查完整路径时使用 `nl2sql doctor --show-paths`。

如命令未找到，用 `python -m nl2sql_handbook` 替代下文的 `nl2sql`。

## 2. 运行离线示例

```bash
nl2sql minimal --offline
```

程序生成三张合成业务表，使用预设 SQL 回复，真正执行 DuckDB 查询。输出应包含：

```json
{
  "model_mode": "scripted_offline",
  "rows": [["15887.00"]]
}
```

这是“已完成订单销售额”的结果。金额以字符串序列化，避免精度损失。示例数据库位于 `outputs/data/demo.duckdb`。

再运行四种流程的完整离线演示：

```bash
nl2sql offline
```

它运行 Workflow、Agent、JOIN Workflow、JOIN Agent，包括错误反馈与纠正。模型回复是脚本预设，LangChain 流程、工具调用和数据库执行是真实的；结果不代表真实模型准确率。

## 3. 配置 API

将 `.env.example` 复制为 `.env`，填写自己的密钥。Linux/macOS 使用 `cp .env.example .env`；PowerShell 使用 `Copy-Item .env.example .env`。

```dotenv
DEEPSEEK_API_KEY=在本地填写自己的密钥
DEEPSEEK_MODEL=deepseek-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com
NL2SQL_OUTPUT_DIR=outputs
```

`.env` 已被 Git 忽略。默认模型沿用原实验配置，并显式关闭 thinking。实际可用模型取决于账号和服务端支持，可修改 `DEEPSEEK_MODEL`。项目使用 `ChatDeepSeek`，自定义 API 地址也应兼容其请求格式。

外部环境变量优先于 `.env`；路径规则与完整优先级见[配置说明](docs/configuration.md)。

## 4. 第一次真实查询

以下命令调用远端模型，会产生 API 用量：

```bash
nl2sql query -q "已完成订单有多少笔？"
```

默认使用 **`workflow_v2`**：选择结构与整理需求 → 补齐关联信息 → 生成并执行 SQL → 语义复核 → 返回或继续修复。合成示例的正确答案为 **8 笔**，但模型仍可能出错，应检查实际 SQL 与结果。

`query` 默认整题硬超时 120 秒、最多 24 次逻辑模型调用、4 次 SQL 尝试、返回 100 行。正常升级版 Workflow 通常调用模型 3 次。复核会把最多前 10 行结果预览发给模型；与旧版未开启解释时的行为不同。

选择不同实现：

```bash
nl2sql query --variant workflow -q "已完成订单有多少笔？"
nl2sql query --variant agent -q "已完成订单有多少笔？"
nl2sql query --variant rag_workflow_v2 -q "按城市统计已完成订单销售额"
```

查询自己的数据库：

```bash
nl2sql query --db /path/to/business.duckdb --variant join_workflow_v2 -q "你的业务问题"
```

文件必须已经存在且为 **DuckDB 格式**；只有内置示例会自动创建，SQLite 文件不能直接替代。业务资料、关系文件与阶段参数见[命令说明](docs/commands.md)。

## 5. 评估与 HTTP 服务

离线生成销售、图书馆、学校三个领域各两个数据实例，并执行 72 次参考 SQL 检查：

```bash
nl2sql init --evaluation
```

题集共 36 题，输出应包含 `reference_executions: 72`。用一题开发集检查模型配置，**这一步会产生 API 用量**：

```bash
nl2sql evaluate-workflow --suite dev --versions workflow_v2 --case sales-01 --workers 1
```

完整评估的运行量、历史成绩和限制见 [benchmarks](benchmarks/README.md)。历史成绩不代表当前安装版本重新跑出的结果。

启动本地服务：

```bash
nl2sql serve
```

另一终端检查健康状态：

```bash
curl --noproxy '*' http://127.0.0.1:8765/health
```

启动和健康检查不调用模型，提交 `/query` 才调用。**HTTP 默认 `rag_workflow` 基础版，只注册原来的七种实现；统一查询 CLI 默认升级版，两者不同。** 请求与配置见 [HTTP 文档](docs/http.md)。

## 6. 测试与目录

```bash
python -m unittest discover -s tests -t . -v
```

测试不调用付费模型。HTTP 与 SDK 集成测试会在本机回环地址开启临时服务；受限环境需要允许本机 socket。

```text
src/nl2sql_handbook/   可安装源码、集中配置、统一 CLI
examples/             按功能组织的运行示例
configs/              HTTP 服务配置模板
datasets/             合成数据与业务资料说明
tests/                原有回归与迁移验证
benchmarks/           评估方法、历史结果摘要
docs/                 使用、配置、架构与验证说明
outputs/              运行后生成的数据、日志、报告（不提交）
```

本项目面向学习、本地实验与受控演示。模型复核不保证任意问题正确；本地 HTTP 服务不包含用户认证或多租户隔离。代码阅读顺序见[架构说明](docs/architecture.md)。

本仓库原创代码及文档采用 [MIT License](LICENSE)。引用资料与第三方依赖遵循各自许可。
