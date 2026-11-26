"""受控的 DuckDB 执行层：模型无权修改限额、连接设置或采用标准。"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from statistics import median
from threading import Timer
from time import perf_counter
from typing import Any, Callable

import duckdb

from nl2sql_handbook.nl2sql_demo import clean_sql, connect_read_only, read_schema


@dataclass
class Limits:
    timeout: float = 10.0
    max_rows: int = 100_000
    threads: int = 1
    warmups: int = 2
    repetitions: int = 7
    min_gain: float = 0.05
    min_saving_ms: float = 0.2
    min_win_fraction: float = 0.8

    def __post_init__(self):
        if not 0.01 <= self.timeout <= 120 or not 1 <= self.max_rows <= 1_000_000:
            raise ValueError("timeout 需为 0.01—120 秒，max_rows 需为 1—1000000")
        if not 1 <= self.threads <= 16 or not 0 <= self.warmups <= 5 or not 3 <= self.repetitions <= 30:
            raise ValueError("threads 需为 1—16，warmups 为 0—5，repetitions 为 3—30")
        if not 0 < self.min_gain < 1 or self.min_saving_ms < 0 or not 0.5 < self.min_win_fraction <= 1:
            raise ValueError("检查收益、绝对节省时间和胜率阈值")


@dataclass
class Budget:
    """全实验共用 SQL 次数上限；元数据/AST 解析不计为工作负载执行。"""

    maximum: int = 300
    used: int = 0

    def consume(self):
        if self.used >= self.maximum:
            raise RuntimeError("实验 SQL 执行次数预算耗尽")
        self.used += 1


def freeze(value: Any) -> Any:
    """把嵌套结果转成可计数的值，不用字符串化或四舍五入掩盖数据差异。"""
    if isinstance(value, float) and math.isnan(value):
        return ("float", "NaN")
    if isinstance(value, dict):
        return ("map", frozenset((freeze(k), freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return (type(value).__name__, tuple(freeze(v) for v in value))
    if isinstance(value, (bytearray, memoryview)):
        return ("bytes", bytes(value))
    # UNION/VARIANT 等列内部仍可能包含不同的 Python 值类型。
    # 显式标注类型，避免 Python 的 True == 1 把差异隐藏掉。
    return (type(value).__name__, value)


@dataclass
class Snapshot:
    columns: list[tuple[str, str]]
    rows: list[tuple]

    def compare(self, other: "Snapshot", ordered: bool) -> dict:
        """无 ORDER BY 用多重集合比较，保留重复次数；有排序则逐行比较。

        列名、顺序和 DuckDB 类型也属于对外接口。即使值看似相同，改变
        DECIMAL 精度或列别名也视为不一致。浮点数默认精确比较，宁可拒绝。
        """
        if self.columns != other.columns:
            return {"equal": False, "reason": "列名、列顺序或类型不一致"}
        left, right = [freeze(r) for r in self.rows], [freeze(r) for r in other.rows]
        equal = left == right if ordered else Counter(left) == Counter(right)
        return {"equal": equal, "reason": "完整结果一致" if equal else "行值、重复次数或要求的顺序不一致",
                "original_rows": len(left), "candidate_rows": len(right),
                "comparison": "ordered" if ordered else "multiset"}


def walk(value):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk(item)


class Database:
    def __init__(self, path: Path, limits: Limits, budget: Budget):
        self.path = path.expanduser().resolve()
        self.limits, self.budget = limits, budget
        self.connection = connect_read_only(self.path)
        try:
            # 设置是固定的宿主代码，不接受模型提供的 SET/PRAGMA。
            self.connection.execute(f"SET threads={limits.threads}")
            self.connection.execute("SET memory_limit='512MB'")
            self.connection.execute("SET max_temp_directory_size='0B'")  # 禁止磁盘溢写。
            self.schema = json.loads(read_schema(self.connection))
            self.version = self.connection.execute("SELECT version()").fetchone()[0]
        except Exception:
            self.close()
            raise

    def close(self):
        self.connection.close()

    def inspect(self, sql: str) -> tuple[str, bool]:
        """使用真实 SQL 解析器；仅接受一个可重复测量的 SELECT。

        常见随机数、时间、序列及采样会破坏结果比较；无排序的 LIMIT 也
        没有稳定的取行语义。这个白名单入口不是任意不可信 SQL 的沙箱：
        自定义宏/视图内的逻辑仍需要数据库所有者了解其确定性。
        """
        sql = clean_sql(sql)
        statements = self.connection.extract_statements(sql)
        if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
            raise ValueError("只允许一条 SELECT，禁止 DDL/DML/SET/EXPLAIN 等语句")
        serialized = json.loads(self.connection.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()[0])
        if serialized.get("error"):
            raise ValueError("此 SELECT 不受本实验 AST 检查支持")
        root = serialized["statements"][0]["node"]
        forbidden = {"random", "gen_random_uuid", "uuid", "uuidv4", "uuidv7", "nextval", "currval",
                     "now", "today", "current_timestamp", "current_date", "current_time",
                     "localtimestamp", "localtime", "get_current_timestamp", "setseed", "set_config"}
        for node in walk(root):
            name = str(node.get("function_name", "")).lower()
            columns = [str(c).lower() for c in node.get("column_names", [])]
            if name in forbidden or (len(columns) == 1 and columns[0] in forbidden):
                raise ValueError("实验拒绝随机、当前时间、序列等非确定性表达式")
            if any(v is not None for k, v in node.items() if k in {"sample", "sample_options"}):
                raise ValueError("实验拒绝采样查询")
            modifiers = {m.get("type") for m in node.get("modifiers", [])}
            if any("LIMIT" in str(m) for m in modifiers) and "ORDER_MODIFIER" not in modifiers:
                raise ValueError("LIMIT/OFFSET 必须在同层有 ORDER BY；排序键还应足以唯一决定取行")
        ordered = any(m.get("type") == "ORDER_MODIFIER" for m in root.get("modifiers", []))
        return sql, ordered

    def _run(self, sql: str, consume: Callable, *, timed: bool = False):
        """Timer 只中断当前语句；结束后 cancel + join，防止误伤下一语句。

        定时器创建/回收和模型调用均在 SQL 计时区间之外。计时包含执行、
        结果传输与完整抓取，不包含 EXPLAIN ANALYZE 的插桩时间。
        """
        self.budget.consume()
        timer = Timer(self.limits.timeout, self.connection.interrupt)
        timer.daemon = True
        timer.start()
        try:
            start = perf_counter()
            cursor = self.connection.execute(sql)
            value = consume(cursor)
            elapsed = (perf_counter() - start) * 1000
            return (value, elapsed) if timed else value
        finally:
            timer.cancel()
            timer.join()

    def _fetch(self, cursor) -> Snapshot:
        columns = [(column[0], str(column[1])) for column in cursor.description]
        rows = []
        while True:
            chunk = cursor.fetchmany(min(1024, self.limits.max_rows + 1 - len(rows)))
            if not chunk:
                break
            rows.extend(chunk)
            if len(rows) > self.limits.max_rows or len(rows) * len(columns) > 1_000_000:
                raise ValueError("完整结果超过比较上限，本次无法判定；不进行截断比较或采用改写")
        return Snapshot(columns, rows)

    def snapshot(self, sql: str) -> Snapshot:
        return self._run(sql, self._fetch)

    def measure(self, sql: str) -> float:
        _, elapsed = self._run(sql, self._fetch, timed=True)
        return elapsed

    def profile(self, sql: str) -> dict:
        physical = self._run("EXPLAIN (FORMAT JSON) " + sql, lambda c: c.fetchone()[1])
        actual = self._run("EXPLAIN (ANALYZE, FORMAT JSON) " + sql, lambda c: c.fetchone()[1])
        plan, profile = json.loads(physical), json.loads(actual)
        operators = [{"operator": n["operator_name"], "ms": n.get("operator_timing", 0) * 1000,
                      "rows": n.get("operator_cardinality"), "rows_scanned": n.get("operator_rows_scanned"),
                      "detail": n.get("extra_info", {})}
                     for n in walk(profile) if "operator_name" in n and n["operator_name"] != "EXPLAIN_ANALYZE"]
        return {"physical_plan": plan, "analyzed_plan": profile,
                "summary": {"latency_ms": profile.get("latency", 0) * 1000,
                            "cpu_ms": profile.get("cpu_time", 0) * 1000,
                            "rows_scanned": profile.get("cumulative_rows_scanned"),
                            "peak_buffer_bytes": profile.get("system_peak_buffer_memory"),
                            "operators_by_time": sorted(operators, key=lambda n: n["ms"], reverse=True)[:10]}}


def benchmark(database: Database, original: str, candidate: str) -> dict:
    """暖缓存 AB/BA 交替配对；不使用单次最快成绩，不关闭数据库优化器。"""
    for _ in range(database.limits.warmups):
        database.measure(original)
        database.measure(candidate)
    pairs = []
    for index in range(database.limits.repetitions):
        order = ["original", "candidate"] if index % 2 == 0 else ["candidate", "original"]
        pair = {"order": order}
        for label in order:
            pair[label + "_ms"] = database.measure(original if label == "original" else candidate)
        pairs.append(pair)
    return summarize(pairs)


def summarize(pairs: list[dict]) -> dict:
    before, after = median(p["original_ms"] for p in pairs), median(p["candidate_ms"] for p in pairs)
    return {"pairs": pairs, "original_median_ms": before, "candidate_median_ms": after,
            "saving_ms": before - after, "gain_fraction": (before - after) / before if before else 0,
            "win_fraction": sum(p["candidate_ms"] < p["original_ms"] for p in pairs) / len(pairs)}


def adoption_reason(measurement: dict, limits: Limits, *, same_plan: bool) -> tuple[bool, str]:
    """模型意见不参与判定。计划相同也拒绝，减少微小抖动造成的误采用。"""
    if same_plan:
        return False, "物理计划相同，不将计时抖动解释为改写收益"
    if measurement["gain_fraction"] < limits.min_gain:
        return False, "中位耗时相对收益未达到阈值"
    if measurement["saving_ms"] < limits.min_saving_ms:
        return False, "中位耗时绝对节省未达到阈值"
    if measurement["win_fraction"] < limits.min_win_fraction:
        return False, "配对胜率未达到阈值"
    return True, "指定数据集完整结果一致，且本轮测量达到全部收益阈值"
