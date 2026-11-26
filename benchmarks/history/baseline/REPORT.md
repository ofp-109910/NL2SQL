# NL2SQL 跨版本统一评估

完成：True；题集：synthetic-v1。
完整结果评分在两组独立数据上进行；列位置/重复次数/NULL/题目排序参与评分，列别名不参与。

| 版本 | 完成/计划 | 正确 | 已完成准确率 | 执行成功 | 平均模型调用* | P50 / s | P95 / s |
|---|---:|---:|---:|---:|---:|---:|---:|
| workflow | 30/30 | 29 | 96.7% | 30 | 2.00 | 1.59 | 2.30 |
| agent | 30/30 | 30 | 100.0% | 30 | 4.03 | 3.75 | 4.40 |
| join_workflow | 30/30 | 26 | 86.7% | 30 | 2.00 | 1.81 | 2.42 |
| join_agent | 30/30 | 30 | 100.0% | 30 | 4.07 | 3.73 | 5.37 |
| rag_workflow | 30/30 | 29 | 96.7% | 30 | 2.00 | 1.87 | 2.75 |
| rag_agent | 30/30 | 30 | 100.0% | 30 | 4.03 | 3.66 | 4.67 |
| multi_agent | 30/30 | 30 | 100.0% | 30 | 13.17 | 21.24 | 25.88 |

*硬超时导致计量丢失时记为未知，不作为 0 次调用；详见 summary.json 的 unknown_call_counts。
Token 仅统计已返回的 provider usage，不按不稳定价格推算费用。

## 实验设置

```json
{
  "versions": [
    "workflow",
    "agent",
    "join_workflow",
    "join_agent",
    "rag_workflow",
    "rag_agent",
    "multi_agent"
  ],
  "split": "test",
  "repeats": 1,
  "workers": 2,
  "case_timeout": 120,
  "max_model_calls": 24,
  "max_sql_attempts": 4,
  "max_rows": 100,
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
  "knowledge_policy": "same domain definitions for RAG variants; no gold SQL sent",
  "threads": 1,
  "memory": "512MB"
}
```

## 失败项

- join_workflow / sales-04 / repeat 1：result_mismatch
- rag_workflow / sales-04 / repeat 1：result_mismatch
- join_workflow / sales-05 / repeat 1：result_mismatch
- join_workflow / sales-08 / repeat 1：result_mismatch
- join_workflow / library-12 / repeat 1：result_mismatch
- workflow / school-09 / repeat 1：result_mismatch

## 适用边界

自建小型合成题集，参考答案由本项目编写；执行一致性不等于任意数据上的语义证明。
只有完整运行才能作为本轮最终对照。单次调用没有测量随机波动；可用 --repeats 独立重复。
并发数、模型名、输入文件/源码指纹与预算均记录在 manifest.json；不声称架构具有普遍优势。
延迟包含连接/实例构造和 ask，不包含父进程启动与本地评分；harness_wall_ms 另存。
结果、公开工具轨迹、参考 SQL 与评分细节均仅保存在本地。
