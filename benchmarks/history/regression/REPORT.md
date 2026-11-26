# Workflow 隔离升级复评

完成：True；题集：regression。
每个候选SQL在两个独立数据实例上完整执行并评分。

| 版本 | 完成/计划 | 与参考一致 | 执行成功 | 平均调用 | 总Token | P50秒 | P95秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| workflow | 30/30 | 30 | 30 | 2.00 | 33539 | 1.79 | 3.39 |
| workflow_v2 | 30/30 | 30 | 30 | 3.00 | 84760 | 3.15 | 4.11 |
| join_workflow | 30/30 | 26 | 30 | 2.00 | 44152 | 1.78 | 2.28 |
| join_workflow_v2 | 30/30 | 30 | 30 | 3.00 | 85065 | 3.21 | 4.46 |
| rag_workflow | 30/30 | 28 | 30 | 2.00 | 81114 | 1.82 | 2.57 |
| rag_workflow_v2 | 30/30 | 30 | 30 | 3.00 | 141007 | 3.30 | 4.14 |

## 原则与边界

- 旧题/参考SQL/历史成绩未改。已分析过的30题用于回归，不能再称为独立盲测。
- 仓储迁移题另列，是同轮人工编写的小型诊断题，不是公开跨数据库基准。
- 默认NULL保留SQL语义、默认只统计实际匹配分组；明确题意/业务定义优先。旧题仍可能有口径歧义。
- 新版同时改变信息补齐、提示词、回退和复核；整体对比不能分离每一项的因果贡献。
- 模型不读取参考SQL或第二数据实例；复核批准是模型意见，最终分数由独立执行评分决定。
- 同一模型/API/预算，实际调用和Token可能不同；不声称等算力或架构普遍优势。
- 单次重复不测量随机波动；耗时含连接/ask，不含进程启动和评分。缺失usage不按0成本解释。

## 不一致或失败项

- join_workflow / sales-04 / 1：result_mismatch
- rag_workflow / sales-04 / 1：result_mismatch
- join_workflow / sales-05 / 1：result_mismatch
- rag_workflow / sales-05 / 1：result_mismatch
- join_workflow / sales-08 / 1：result_mismatch
- join_workflow / library-12 / 1：result_mismatch

## 设置

```json
{
  "suite": "regression",
  "versions": [
    "workflow",
    "workflow_v2",
    "join_workflow",
    "join_workflow_v2",
    "rag_workflow",
    "rag_workflow_v2"
  ],
  "repeats": 1,
  "workers": 2,
  "model": "deepseek-flash",
  "provider_options": {
    "base_url": "https://api.deepseek.com",
    "temperature": 0,
    "max_tokens": 2048,
    "timeout": 60,
    "max_retries": 1,
    "extra_body": {
      "thinking": {
        "type": "disabled"
      }
    }
  },
  "max_model_calls": 24,
  "max_sql_attempts": 4,
  "max_selection_attempts_v2": 2,
  "max_rows": 100,
  "case_timeout": 120,
  "threads": 1,
  "memory": "512MB",
  "scoring_instances": 2
}
```
