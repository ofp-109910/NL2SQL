# 配置规则

`src/nl2sql_handbook/config.py` 是 CLI、Python API、评估器与工作子进程共用的配置入口。项目不依赖作者的虚拟环境或密钥文件。

| 环境变量 | 默认值 | 用途 |
|---|---|---|
| `DEEPSEEK_API_KEY` | 空 | 模型密钥，只在需要调用时检查 |
| `DEEPSEEK_MODEL` | `deepseek-flash` | 模型名称，CLI 的 `--model` 可覆盖 |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | DeepSeek 兼容 API 地址 |
| `NL2SQL_OUTPUT_DIR` | `outputs` | 数据、日志和报告的根目录 |
| `NL2SQL_API_KEY_FILE` | `.secrets/deepseek_api` | 可选单行密钥文件；API Key 环境变量优先 |
| `NL2SQL_HOME` | 启动时工作目录 | 路径基准目录，应在外部环境中设置 |
| `NL2SQL_ENV_FILE` | `.env` | 配置文件，应在外部环境中设置 |

先确定 `NL2SQL_HOME`，再读取它下面的 `.env`，不向父目录搜索。`NL2SQL_ENV_FILE`、输出目录、密钥文件的相对路径都相对于这个目录。`.env` 不做变量插值，不要写 `${变量名}`。

从项目根目录使用最简单：复制 `.env.example`、填写密钥、执行命令。安装后从其他目录启动，会读取那个目录的 `.env` 并在当地生成 `outputs`。要继续使用同一项目配置，在外部环境中指定 `NL2SQL_HOME` 为项目绝对路径。

优先级：

- 模型名称：显式 CLI 参数或服务 JSON > 外部环境变量 > `.env` > 默认值。
- 密钥：外部环境变量 > `.env` > 单行密钥文件。阶段 CLI 的 `--api-key-file` 只覆盖文件位置，不覆盖环境变量中的密钥。
- HTTP JSON 中的数据库、知识文件、日志和 `key_file` 相对路径，以 **JSON 所在目录**为基准。未传 `--config` 时，服务使用统一输出目录下的评估库和日志。模板不写死模型，让 `.env` 生效。

`nl2sql doctor` 只检查本地配置，不显示密钥，不验证远端鉴权、余额或模型是否可用。默认把项目根目录显示为 `<project>`，项目外的路径显示为 `<external path>`，避免在分享诊断输出时暴露本机用户名和绝对目录。需要本地排查完整路径时，使用 `nl2sql doctor --show-paths`；该输出可能包含个人路径。配置在进程启动时加载，修改后应重新启动命令或服务。

程序不会自动获取账号密钥，也不会回退读取作者机器上的密钥。调用模型时，密钥通过 SDK 的鉴权请求头发送给 `DEEPSEEK_BASE_URL` 指定的服务。CLI 和查询错误消息会遮盖已读取的密钥原值，也保留对 `sk-` 格式的遮盖；这不等于对任意业务数据做匿名化。真实查询会按流程把问题、表结构及部分结果预览发送给模型服务。

`.venv`、`.env`、`.secrets`、`outputs` 都被 Git 忽略；仓库提供配置模板和依赖声明，由读者创建自己的环境。
