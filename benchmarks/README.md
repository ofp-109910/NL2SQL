# 评估与历史结果

题集与评分器位于 `src/nl2sql_handbook/evaluation_demo`；升级版对照在 `workflow_v2`。两组不同数据实例共同检查结果，比较列位置、重复次数、NULL、值与要求的排序。参考 SQL 供本地评分，不发送给模型。

## 先离线准备

```bash
nl2sql init --evaluation --transfer
```

三个领域共 36 题，其中开发集 6 题、测试集 30 题。主库和变体库各执行一次参考 SQL，共 72 次。仓储诊断另有 6 题与两个数据实例。

## 小规模联网检查

下面会产生 API 用量；先跑单题，再决定是否全量：

```bash
nl2sql evaluate-workflow --suite dev --versions workflow_v2 --case sales-01 --workers 1
nl2sql evaluate --split dev --versions agent --case sales-01 --workers 1
```

## 全量运行

```bash
# 七个基础实现 × 30 题 = 210 个任务
nl2sql evaluate --workers 2

# 三组旧/新 Workflow × 30 题 = 180 个任务
nl2sql evaluate-workflow --suite regression --workers 2

# 三组旧/新 Workflow × 6 道仓储题 = 36 个任务
nl2sql evaluate-workflow --suite transfer --workers 2
```

一个任务可能多次调用模型。默认预算为每题 24 次逻辑模型调用、4 次 SQL 尝试、120 秒整题期限。不要把任务数当成 API 请求数。

报告分别写入 `outputs/runs/evaluation/时间戳` 和 `outputs/runs/workflow_v2/时间戳`。可以用 `--run-dir` 指定新目录，使用原配置和原源码时才可 `--resume`。迁移后的源码和依赖指纹已经改变，不得续跑作者旧实验目录。

每轮保存配置、源码与数据指纹、依赖版本、实际 SQL、公开工具记录、计量和独立评分。阶段评估器的退出码表示运行是否完成；准确率要读取 `summary.json`，不能把退出码 0 理解为所有答案正确。

## 已保存的历史结果

| 记录 | 含义 | 报告 |
|---|---|---|
| 七版基线 | 原始 210 个测试任务 | [baseline](history/baseline/REPORT.md) |
| Workflow 同批复评 | 180 个旧/新对照任务 | [regression](history/regression/REPORT.md) |
| 仓储诊断 | 36 个补充任务 | [transfer](history/transfer/REPORT.md) |

摘要直接来自迁移前保存的真实运行，保留原有指纹；本次只复制 `REPORT.md` 和 `summary.json`，没有将几百份轨迹与生成数据库混入源码。原始完整实验档案仍保留在原实验归档中。这些历史目录不是可续跑的任务目录。

三组升级 Workflow 在历史复评中均为 30/30，但这 30 题已经分析过，只能用于回归。同批复评中新版的第一条 SQL 均已通过，不能据此证明语义修复造成了分数提升。仓储诊断中存在实际纠错，也出现复核建议引入错误后再被纠正的过程；6 题是小型自建诊断，不是独立公开盲测。

新目录的离线测试和安装验证不会更新上述模型成绩。新安装环境的部分传递依赖与历史环境不同；重新实测应新建报告，独立记录模型和成本。

## 失败复盘

```bash
# 全离线故障注入，不代表真实模型错误率
nl2sql review --controlled-only

# 分析自己刚生成的完整七版运行目录
nl2sql review --run-dir outputs/runs/evaluation/你的时间戳
```

`repair` 是对已知失败 SQL 的特定模式回放，不是通用生成器升级；只在原数据指纹匹配且模式相同时使用。真正的生成链改进在 `workflow_v2`。
