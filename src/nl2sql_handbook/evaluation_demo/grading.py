"""只在本机执行的结果评分：完整结果、重复次数、NULL 及题目要求的排序。"""

from collections import Counter
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import duckdb

from nl2sql_handbook.ai4db_demo.engine import Budget, Database, Limits
from nl2sql_handbook.nl2sql_demo import clean_sql, describe_error
from .dataset import Case


def canonical(value):
    """数值使用十进制表示；日期与旧 CLI 的 ISO 字符串保持可比。

    只对浮点值保留到小数点后 6 位，金额 DECIMAL、整数不做舍入。
    列别名不评分，列的位置和数量评分；测试题要求均明确指定了列顺序。
    """
    if isinstance(value,bool):
        return ("bool",value)
    if isinstance(value,(int,Decimal)):
        return ("number",Decimal(value))
    if isinstance(value,float):
        return ("number",Decimal(str(value)).quantize(Decimal("0.000001")))
    if isinstance(value,(date,datetime)):
        return ("text",value.isoformat())
    if isinstance(value,str):
        # 子进程 JSON 用字符串保留 DECIMAL；只在列类型证明为数值时转回数值。
        return ("text",value)
    if value is None:
        return ("null",)
    if isinstance(value,(list,tuple)):
        return tuple(canonical(item) for item in value)
    return (type(value).__name__,str(value))


def restore(rows, columns):
    """依据真实 SQL 输出类型恢复 CLI 字符串，避免把商品编号 '001' 当数字。"""
    numeric = ("DECIMAL","INTEGER","BIGINT","HUGEINT","SMALLINT","TINYINT","DOUBLE","FLOAT","UBIGINT","UINTEGER","UHUGEINT")
    return [[Decimal(v) if isinstance(v,str) and any(t in typ for t in numeric) else v
             for v,(_,typ) in zip(row,columns)] for row in rows]


def rows_equal(left, right, ordered):
    a,b = [canonical(r) for r in left],[canonical(r) for r in right]
    return a==b if ordered else Counter(a)==Counter(b)


def grade(case: Case, candidate: str, output: dict, root: Path) -> dict:
    checks = []
    result = {"passed":False,"checks":checks,"category":"execution_error"}
    try:
        sql = clean_sql(candidate)
        for variant in (0,1):
            database = Database(root/f"{case.domain}_{variant}.duckdb",Limits(max_rows=1000,timeout=5),Budget())
            try:
                statements = database.connection.extract_statements(sql)
                if len(statements)!=1 or statements[0].type!=duckdb.StatementType.SELECT:
                    raise ValueError("评分只执行一条 SELECT")
                expected, actual = database.snapshot(case.sql),database.snapshot(sql)
                equal = len(expected.columns)==len(actual.columns) and rows_equal(expected.rows,actual.rows,case.ordered)
                checks.append({"variant":variant,"equal":equal,"expected_rows":expected.rows,
                               "actual_rows":actual.rows,"expected_columns":expected.columns,
                               "actual_columns":actual.columns,"ordered":case.ordered})
                if variant==0:
                    returned = output.get("rows",[])
                    shape_ok = all(len(row)==len(actual.columns) for row in returned)
                    result["returned_matches_sql"] = shape_ok and rows_equal(restore(returned,actual.columns),actual.rows,case.ordered)
            finally:
                database.close()
        result["not_truncated"] = not output.get("truncated",False)
        result["passed"] = all(c["equal"] for c in checks) and result["returned_matches_sql"] and result["not_truncated"]
        result["category"] = "pass" if result["passed"] else "result_mismatch" if not all(c["equal"] for c in checks) else "truncated_or_return_mismatch"
    except Exception as exc:
        result["error"] = describe_error(exc)
    return result
