"""生成独立的合成实验库，不读取或修改前几阶段及奥运会数据库。"""

from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent
from nl2sql_handbook.config import SETTINGS
DEFAULT_DB = SETTINGS.output_dir / "data" / "ai4db" / "benchmark.duckdb"
EDGE_DB = SETTINGS.output_dir / "data" / "ai4db" / "edge_cases.duckdb"

DDL = """
CREATE TABLE customers (
    customer_id INTEGER PRIMARY KEY, customer_name VARCHAR, city VARCHAR
);
CREATE TABLE products (
    product_id INTEGER PRIMARY KEY, product_name VARCHAR, category VARCHAR
);
CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id),
    product_id INTEGER REFERENCES products(product_id),
    quantity INTEGER, unit_price DECIMAL(10,2), order_date DATE, status VARCHAR
);
COMMENT ON COLUMN orders.unit_price IS '历史成交单价；销售额为 quantity * unit_price';
COMMENT ON COLUMN orders.status IS '已完成、待付款、已取消';
COMMENT ON TABLE orders IS '每行是一笔单商品订单；不同主键可有相同业务字段';
"""


def create_fixture(path: Path, orders: int = 200_000, *, edge: bool = False) -> bool:
    """只创建新文件；已有文件原样保留。返回是否确实创建。

    先写同目录临时文件，再用硬链接原子发布；即使两个进程同时初始化，
    也不会覆盖先完成的库。初始化失败只删除本次临时文件。
    """
    import os
    import tempfile

    path = path.expanduser().resolve()
    if path.exists():
        return False
    if not 1000 <= orders <= 5_000_000:
        raise ValueError("orders 需在 1000—5000000 之间")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".fixture-", suffix=".duckdb", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    temporary.unlink()  # DuckDB 要求文件不存在或是有效的数据库。
    try:
        with duckdb.connect(str(temporary)) as connection:
            connection.execute("SET threads=1")
            connection.execute(DDL)
            if edge:
                connection.execute("""
                    INSERT INTO customers VALUES
                      (1,'甲','北京'),(2,'乙',NULL),(3,'丙','无订单'),
                      (4,'丁','仅取消'),(5,'戊','全空金额');
                    INSERT INTO products VALUES (1,'商品甲','类别甲'),(2,'商品乙','类别乙');
                    INSERT INTO orders VALUES
                      (1,1,1,2,10,'2025-01-01','已完成'),
                      (2,1,1,2,10,'2025-01-01','已完成'),
                      (3,2,1,1,7,'2025-01-02','已完成'),
                      (4,2,2,NULL,9,'2025-01-03','已完成'),
                      (5,1,2,4,NULL,'2025-01-03','已完成'),
                      (6,4,1,99,100,'2025-01-04','已取消'),
                      (7,1,1,99,100,'2025-01-04','待付款'),
                      (8,5,1,NULL,NULL,'2025-01-04','已完成');
                """)
            else:
                # range 向量化造数，比 Python 逐行插入更适合性能实验。
                # 保留 10 名无订单客户，包含 NULL 城市、空数量和空单价。
                connection.execute("""
                    INSERT INTO customers
                    SELECT i, '客户' || i,
                           CASE WHEN i % 23 = 0 THEN NULL ELSE '城市' || (i % 50) END
                    FROM range(1, 5011) t(i);
                    INSERT INTO products
                    SELECT i, '商品' || i, '类别' || (i % 10) FROM range(1, 501) t(i);
                """)
                connection.execute("""
                    INSERT INTO orders
                    SELECT i, 1 + i % 5000, 1 + i % 500,
                        CASE WHEN i % 101 = 0 THEN NULL ELSE 1 + i % 5 END,
                        CASE WHEN i % 127 = 0 THEN NULL
                             ELSE CAST((1 + i % 20000) / 100.0 AS DECIMAL(10,2)) END,
                        DATE '2025-01-01' + CAST(i % 365 AS INTEGER),
                        CASE WHEN i % 10 = 0 THEN '已取消'
                             WHEN i % 10 = 1 THEN '待付款' ELSE '已完成' END
                    FROM range(1, ?) t(i)
                """, [orders + 1])
            connection.execute("ANALYZE")
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        return True
    finally:
        temporary.unlink(missing_ok=True)
