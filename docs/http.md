# 本地 HTTP 服务

先按根目录 README 安装和配置 `.env`，再运行：

```bash
nl2sql init --evaluation
nl2sql serve
```

默认监听 `127.0.0.1:8765`。健康检查不调用模型：

```bash
curl --noproxy '*' http://127.0.0.1:8765/health
```

实际查询会产生模型 API 用量：

```bash
curl --noproxy '*' http://127.0.0.1:8765/query \
  -H 'Content-Type: application/json' \
  --data '{"question":"共有多少笔已完成订单？只返回订单笔数。","database":"sales"}'
```

Windows PowerShell 可以使用 `curl.exe`，或 `Invoke-RestMethod`，避免命令别名差异。服务数据库 ID 为 `sales/library/school`，使用评估合成库；它与最小示例的 10 笔订单数据库不同，不应套用最小示例的结果。请求只接受 `question` 与 `database`，不能传任意文件路径或 SQL。

## 配置

```bash
nl2sql serve --config configs/service.example.json
```

模板路径以 JSON 所在目录为基准，并指向根目录 `outputs/data/evaluation`。改用自定义 `NL2SQL_OUTPUT_DIR` 时，可直接省略 `--config` 使用动态默认值；显式 JSON 则需要同时修改其中路径。

模型与密钥来自统一配置。JSON 可以显式指定 `model` 和单行密钥文件路径 `key_file`，但不放密钥正文。数据库和知识文件缺失时会在启动时报错，先执行初始化。

默认实现仍为 **`rag_workflow` 基础版**，允许 `workflow/agent/join_workflow/join_agent/rag_workflow/rag_agent/multi_agent`。尚未接入三个 `*_v2`，未知版本会被拒绝。CLI 的 `query` 默认升级版，两者应分别说明；未来接入需要新增服务验收，不能直接沿用旧报告。

默认并发 2、整题 90 秒、模型调用 24 次、SQL 尝试 4 次、返回 100 行。每题独立子进程；输出金额和日期为字符串。成功响应包含真实 SQL、列名、数据和计量。

| 状态码 | 含义 |
|---|---|
| 200 | 查询执行完成，仍应核对业务语义 |
| 400/413/415 | 参数、大小或媒体类型不合法 |
| 429 | 并发额度用尽 |
| 502 | 模型或工具执行失败 |
| 504 | 整题超时，工作进程被回收 |
| 500 | 服务内部错误，可用请求 ID 查日志 |

默认日志为 `outputs/logs/requests.jsonl`，只记录请求 ID、状态、耗时和计量，不写问题、SQL、结果或密钥。按 Ctrl+C 关闭服务。本项目没有用户鉴权和多租户隔离，交付范围是本机受控演示。

## 容器材料

仓库附有 Dockerfile，但本次环境未完成 Docker 构建与容器验收。它不是 README 的必经步骤。

```bash
docker build -t nl2sql-handbook .
docker run --rm -p 127.0.0.1:8765:8765 --env-file .env nl2sql-handbook
```

镜像启动时生成合成评估库，不包含作者数据库或密钥；密钥在运行时注入。容器以非 root 用户运行，输出位于 `/app/outputs`。
