"""对受控、同列序的标量查询结果进行比较，不证明任意 SQL 的语义等价。"""
from collections import Counter


def same_result(actual, expected, *, ordered=False):
    # 有截断时不冒充完整结果；别名可不同，但列数和语义列序须由用例先约定。
    if actual.truncated or len(actual.columns) != len(expected.columns):
        return False
    left = [tuple(row) for row in actual.rows]
    right = [tuple(row) for row in expected.rows]
    # Counter 保留重复行的数量；set 会错误地忽略重复行。
    return left == right if ordered else Counter(left) == Counter(right)
