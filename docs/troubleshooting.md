# 常见问题

| 现象 | 检查方法 |
|---|---|
| `nl2sql: command not found` | 激活 `.venv` 并完成 `pip install -e .`；也可用 `python -m nl2sql_handbook` |
| 导入不到 `nl2sql_handbook` | 核对 `python` 是否属于新建虚拟环境；不要只复制单个脚本运行 |
| 未配置 API Key | 复制 `.env.example` 为 `.env`，填写真实密钥；运行 `nl2sql doctor` 检查读取位置 |
| 修改 `.env` 没生效 | 外部环境变量优先；重启命令或服务，确认 `NL2SQL_HOME` |
| 鉴权失败、模型不可用 | 检查密钥和账号支持的模型；`doctor` 不验证远端状态 |
| 找不到数据库 | 先 `nl2sql init`；服务和评估用 `nl2sql init --evaluation` |
| SQLite 文件打不开 | 必须传 DuckDB 文件；扩展名 `.db` 不能证明文件格式 |
| 自建库 JOIN 失败 | 检查外键或显式关系文件，不依赖同名列猜测关系 |
| 服务不接受 `rag_workflow_v2` | HTTP 尚未注册升级版；统一查询 CLI 支持，见 HTTP 文档 |
| 本机 curl 走代理 | 使用 `curl --noproxy '*'`，或在客户端配置 `NO_PROXY=127.0.0.1,localhost` |
| 离线测试 socket 被拒绝 | HTTP 测试需要本机回环端口权限，不需要真实模型账号 |
| 续跑评估被拒绝 | 数据、源码、依赖或配置指纹不同应新建运行目录，不能拼接不同实验 |
| 依赖安装失败 | 核对 Python 版本、平台和包源；当前验证组合是 Linux x86_64 / Python 3.10 |

`--help`、`doctor`、离线示例不消耗 API 用量。统一评估默认会调用模型，先运行一题开发集检查配置。模型响应存在波动，SQL 能执行与业务含义正确是两回事。
