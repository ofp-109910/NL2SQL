"""版本化合成题集。参考 SQL 仅供本地评分，不加入模型上下文。

三个领域各 12 题：前两题开发，其余十题测试。每个领域准备两组同结构、
不同数值/状态/关联分布的数据，避免一条错误 SQL 仅因当前数据巧合而得分。
这是一套小型自建基准，不宣称代表公开基准或真实业务整体准确率。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent
from nl2sql_handbook.config import SETTINGS
DATA = SETTINGS.output_dir / "data" / "evaluation"
DATASET_VERSION = "synthetic-v1"


@dataclass(frozen=True)
class Case:
    id: str
    domain: str
    split: str
    tags: tuple[str, ...]
    ordered: bool
    question: str
    sql: str


def cases() -> list[Case]:
    result = []
    def add(domain, tags, question, sql, ordered=False):
        number = sum(c.domain == domain for c in result) + 1
        result.append(Case(f"{domain}-{number:02}", domain, "dev" if number <= 2 else "test",
                           tuple(tags.split()), ordered, question, sql))
    add("sales", "aggregate filter", "共有多少笔已完成订单？只返回订单笔数一列。",
        "SELECT COUNT(*) FROM orders WHERE status='已完成'")
    add("sales", "filter order", "列出北京的客户，依次返回客户ID和客户名称，按客户ID升序。",
        "SELECT customer_id,customer_name FROM customers WHERE city='北京' ORDER BY customer_id", True)
    add("sales", "aggregate null", "统计全部已完成订单的销售额，只返回销售额一列。销售额为数量乘以历史成交单价，空值遵守 SQL SUM 语义。",
        "SELECT SUM(quantity*unit_price) FROM orders WHERE status='已完成'")
    add("sales", "join aggregate null", "按城市统计已完成订单销售额和订单笔数，依次返回城市、销售额、订单笔数。保留NULL城市分组，按销售额降序、城市升序且NULL最后。",
        "SELECT c.city,SUM(o.quantity*o.unit_price) sales,COUNT(*) FROM orders o JOIN customers c ON c.customer_id=o.customer_id WHERE o.status='已完成' GROUP BY c.city ORDER BY sales DESC,c.city ASC NULLS LAST", True)
    add("sales", "join three_tables", "按客户城市和商品类别统计已完成订单销售额，依次返回城市、类别、销售额；按销售额降序、城市升序、类别升序，各列NULL最后。",
        "SELECT c.city,p.category,SUM(o.quantity*o.unit_price) sales FROM orders o JOIN customers c ON c.customer_id=o.customer_id JOIN products p ON p.product_id=o.product_id WHERE o.status='已完成' GROUP BY c.city,p.category ORDER BY sales DESC,c.city ASC NULLS LAST,p.category ASC NULLS LAST", True)
    add("sales", "left_join null", "列出所有客户的已完成订单笔数与销售额，包括无订单和仅取消订单的客户。依次返回客户ID、客户名称、笔数、销售额；无已完成订单时笔数和金额为0，按客户ID升序。",
        "SELECT c.customer_id,c.customer_name,COUNT(o.order_id),COALESCE(SUM(o.quantity*o.unit_price),0) FROM customers c LEFT JOIN orders o ON c.customer_id=o.customer_id AND o.status='已完成' GROUP BY c.customer_id,c.customer_name ORDER BY c.customer_id", True)
    add("sales", "subquery anti_join", "列出从未出现在已完成订单中的商品，依次返回商品ID和商品名称，按商品ID升序。",
        "SELECT p.product_id,p.product_name FROM products p WHERE NOT EXISTS(SELECT 1 FROM orders o WHERE o.product_id=p.product_id AND o.status='已完成') ORDER BY p.product_id", True)
    add("sales", "date aggregate", "按订单月份统计已完成订单销售额，依次返回YYYY-MM格式月份和销售额，按月份升序。",
        "SELECT strftime(order_date,'%Y-%m') AS period,SUM(quantity*unit_price) FROM orders WHERE status='已完成' GROUP BY period ORDER BY period", True)
    add("sales", "having aggregate", "找出至少有两笔已完成订单的客户，依次返回客户ID和已完成订单笔数，按客户ID升序。",
        "SELECT customer_id,COUNT(*) FROM orders WHERE status='已完成' GROUP BY customer_id HAVING COUNT(*)>=2 ORDER BY customer_id", True)
    add("sales", "join aggregate duplicates", "按商品类别统计已完成订单的总件数，依次返回类别和总件数；只返回有已完成订单的类别，NULL数量被SUM忽略，按类别升序。",
        "SELECT p.category,SUM(o.quantity) FROM products p JOIN orders o ON p.product_id=o.product_id WHERE o.status='已完成' GROUP BY p.category ORDER BY p.category", True)
    add("sales", "anti_join empty", "列出完全没有任何订单的客户，依次返回客户ID、客户名称，按客户ID升序。",
        "SELECT c.customer_id,c.customer_name FROM customers c WHERE NOT EXISTS(SELECT 1 FROM orders o WHERE o.customer_id=c.customer_id) ORDER BY c.customer_id", True)
    add("sales", "date filter null", "统计2025年第一季度内已完成且数量至少为3的订单，依次返回订单笔数和销售额；日期包含1月1日、不含4月1日。",
        "SELECT COUNT(*),SUM(quantity*unit_price) FROM orders WHERE status='已完成' AND quantity>=3 AND order_date>=DATE '2025-01-01' AND order_date<DATE '2025-04-01'")

    add("library", "aggregate", "图书目录中有多少种图书？只返回图书记录数。", "SELECT COUNT(*) FROM books")
    add("library", "aggregate filter", "有多少条尚未归还的借阅记录？returned_on为NULL表示未归还，只返回记录数。", "SELECT COUNT(*) FROM loans WHERE returned_on IS NULL")
    add("library", "left_join aggregate", "列出所有作者及其图书种数，包括没有图书的作者，依次返回作者ID、姓名、图书种数，按作者ID升序。",
        "SELECT a.author_id,a.name,COUNT(b.book_id) FROM authors a LEFT JOIN books b ON a.author_id=b.author_id GROUP BY a.author_id,a.name ORDER BY a.author_id", True)
    add("library", "left_join aggregate", "列出所有读者的累计借阅次数，包括从未借书的读者，依次返回读者ID、姓名、次数，按读者ID升序。",
        "SELECT m.member_id,m.name,COUNT(l.loan_id) FROM members m LEFT JOIN loans l ON m.member_id=l.member_id GROUP BY m.member_id,m.name ORDER BY m.member_id", True)
    add("library", "join aggregate duplicates", "按图书类别统计累计借阅次数，重复借阅分别计次，只返回有借阅的类别；依次返回类别、次数，按类别升序。",
        "SELECT b.genre,COUNT(*) FROM books b JOIN loans l ON b.book_id=l.book_id GROUP BY b.genre ORDER BY b.genre", True)
    add("library", "join three_tables date", "以固定日期2025-04-01判断逾期：尚未归还且应还日期早于该日。依次返回借阅ID、读者姓名、书名、应还日期，按应还日期和借阅ID升序。",
        "SELECT l.loan_id,m.name,b.title,l.due_on FROM loans l JOIN members m ON m.member_id=l.member_id JOIN books b ON b.book_id=l.book_id WHERE l.returned_on IS NULL AND l.due_on<DATE '2025-04-01' ORDER BY l.due_on,l.loan_id", True)
    add("library", "anti_join subquery", "列出从未被借阅的图书，依次返回图书ID、书名，按图书ID升序。",
        "SELECT b.book_id,b.title FROM books b WHERE NOT EXISTS(SELECT 1 FROM loans l WHERE l.book_id=b.book_id) ORDER BY b.book_id", True)
    add("library", "join three_tables null", "按作者统计当前未归还的借阅记录数，只返回有未归还记录的作者；依次返回作者ID、姓名、记录数，按作者ID升序。",
        "SELECT a.author_id,a.name,COUNT(*) FROM authors a JOIN books b ON a.author_id=b.author_id JOIN loans l ON b.book_id=l.book_id WHERE l.returned_on IS NULL GROUP BY a.author_id,a.name ORDER BY a.author_id", True)
    add("library", "date aggregate", "按借出月份统计借阅次数，依次返回YYYY-MM格式月份、次数，按月份升序。",
        "SELECT strftime(borrowed_on,'%Y-%m') AS period,COUNT(*) FROM loans GROUP BY period ORDER BY period", True)
    add("library", "join having distinct", "找出累计借阅过至少两种图书类别的读者，依次返回读者ID和不同类别数，按读者ID升序。",
        "SELECT l.member_id,COUNT(DISTINCT b.genre) FROM loans l JOIN books b ON b.book_id=l.book_id GROUP BY l.member_id HAVING COUNT(DISTINCT b.genre)>=2 ORDER BY l.member_id", True)
    add("library", "join ranking", "找出累计借阅次数最多的前三种图书，仅考虑借阅过的图书。依次返回图书ID、书名、次数；次数降序、图书ID升序。",
        "SELECT b.book_id,b.title,COUNT(*) n FROM books b JOIN loans l ON b.book_id=l.book_id GROUP BY b.book_id,b.title ORDER BY n DESC,b.book_id LIMIT 3", True)
    add("library", "join distinct null", "按城市统计当前持有未归还图书的不同读者人数，同一个读者只计一次。依次返回城市、人数，保留NULL城市，按城市升序且NULL最后。",
        "SELECT m.city,COUNT(DISTINCT m.member_id) FROM members m JOIN loans l ON m.member_id=l.member_id WHERE l.returned_on IS NULL GROUP BY m.city ORDER BY m.city ASC NULLS LAST", True)

    add("school", "aggregate", "共有多少名学生？只返回学生人数。", "SELECT COUNT(*) FROM students")
    add("school", "aggregate null", "计算全部选课记录的平均成绩，NULL成绩不参与平均，保留两位小数，只返回平均成绩。", "SELECT ROUND(AVG(score),2) FROM enrollments")
    add("school", "left_join filter", "列出所有学生在2025春学期的选课记录数，包括该学期未选课的学生。依次返回学生ID、姓名、记录数，按学生ID升序。",
        "SELECT s.student_id,s.name,COUNT(e.enrollment_id) FROM students s LEFT JOIN enrollments e ON s.student_id=e.student_id AND e.term='2025春' GROUP BY s.student_id,s.name ORDER BY s.student_id", True)
    add("school", "join aggregate null", "按课程统计2025春学期平均成绩，忽略NULL成绩，保留两位小数。只返回该学期有选课的课程；依次返回课程ID、课程名、平均成绩，按课程ID升序。",
        "SELECT c.course_id,c.title,ROUND(AVG(e.score),2) FROM courses c JOIN enrollments e ON c.course_id=e.course_id WHERE e.term='2025春' GROUP BY c.course_id,c.title ORDER BY c.course_id", True)
    add("school", "anti_join null", "列出2025春学期没有任何及格成绩的学生，包括未选课者。及格为score>=60，NULL不算及格。依次返回学生ID、姓名，按学生ID升序。",
        "SELECT s.student_id,s.name FROM students s WHERE NOT EXISTS(SELECT 1 FROM enrollments e WHERE e.student_id=s.student_id AND e.term='2025春' AND e.score>=60) ORDER BY s.student_id", True)
    add("school", "join three_tables left_join", "列出所有学生在2025春学期获得的及格课程学分总和，无及格课程记0。依次返回学生ID、姓名、学分总和，按学生ID升序。",
        "SELECT s.student_id,s.name,COALESCE(SUM(c.credits),0) FROM students s LEFT JOIN enrollments e ON s.student_id=e.student_id AND e.term='2025春' AND e.score>=60 LEFT JOIN courses c ON c.course_id=e.course_id GROUP BY s.student_id,s.name ORDER BY s.student_id", True)
    add("school", "having filter", "找出2025春学期至少有两条及格选课记录的学生，依次返回学生ID、及格记录数，按学生ID升序。",
        "SELECT student_id,COUNT(*) FROM enrollments WHERE term='2025春' AND score>=60 GROUP BY student_id HAVING COUNT(*)>=2 ORDER BY student_id", True)
    add("school", "anti_join filter", "列出2025春学期无人选修的课程，依次返回课程ID、课程名，按课程ID升序。",
        "SELECT c.course_id,c.title FROM courses c WHERE NOT EXISTS(SELECT 1 FROM enrollments e WHERE e.course_id=c.course_id AND e.term='2025春') ORDER BY c.course_id", True)
    add("school", "join aggregate filter", "按年级统计2025春学期的及格选课记录数，仅返回存在及格记录的年级。依次返回年级、记录数，按年级升序。",
        "SELECT s.grade,COUNT(*) FROM students s JOIN enrollments e ON s.student_id=e.student_id WHERE e.term='2025春' AND e.score>=60 GROUP BY s.grade ORDER BY s.grade", True)
    add("school", "aggregate duplicates", "按学期统计选课记录数和有成绩的记录数，依次返回学期、选课记录数、有成绩记录数，按学期升序。",
        "SELECT term,COUNT(*),COUNT(score) FROM enrollments GROUP BY term ORDER BY term", True)
    add("school", "having ranking null", "2025春学期至少有两条非NULL成绩的学生中，找平均成绩最高的前三名。依次返回学生ID、保留两位小数的平均成绩；按未四舍五入的平均成绩降序、学生ID升序。",
        "SELECT student_id,ROUND(AVG(score),2) FROM enrollments WHERE term='2025春' GROUP BY student_id HAVING COUNT(score)>=2 ORDER BY AVG(score) DESC,student_id LIMIT 3", True)
    add("school", "join three_tables null", "列出2025春学期成绩尚未录入的选课记录，依次返回选课记录ID、学生姓名、课程名，按记录ID升序。",
        "SELECT e.enrollment_id,s.name,c.title FROM enrollments e JOIN students s ON s.student_id=e.student_id JOIN courses c ON c.course_id=e.course_id WHERE e.term='2025春' AND e.score IS NULL ORDER BY e.enrollment_id", True)
    return result


SCHEMAS = {
    "sales": """
CREATE TABLE customers(customer_id INTEGER PRIMARY KEY,customer_name VARCHAR,city VARCHAR);
CREATE TABLE products(product_id INTEGER PRIMARY KEY,product_name VARCHAR,category VARCHAR);
CREATE TABLE orders(order_id INTEGER PRIMARY KEY,customer_id INTEGER REFERENCES customers(customer_id),product_id INTEGER REFERENCES products(product_id),quantity INTEGER,unit_price DECIMAL(10,2),order_date DATE,status VARCHAR);
COMMENT ON COLUMN orders.unit_price IS '历史成交单价，销售额=quantity*unit_price';
COMMENT ON COLUMN orders.status IS '已完成、已取消、待付款；成交默认指已完成';
""",
    "library": """
CREATE TABLE authors(author_id INTEGER PRIMARY KEY,name VARCHAR);
CREATE TABLE books(book_id INTEGER PRIMARY KEY,title VARCHAR,author_id INTEGER REFERENCES authors(author_id),genre VARCHAR);
CREATE TABLE members(member_id INTEGER PRIMARY KEY,name VARCHAR,city VARCHAR);
CREATE TABLE loans(loan_id INTEGER PRIMARY KEY,book_id INTEGER REFERENCES books(book_id),member_id INTEGER REFERENCES members(member_id),borrowed_on DATE,due_on DATE,returned_on DATE);
COMMENT ON COLUMN loans.returned_on IS 'NULL表示未归还；每条记录单独计一次借阅';
COMMENT ON COLUMN loans.due_on IS '应还日期；逾期需比较题目指定的固定日期';
""",
    "school": """
CREATE TABLE students(student_id INTEGER PRIMARY KEY,name VARCHAR,grade VARCHAR);
CREATE TABLE courses(course_id INTEGER PRIMARY KEY,title VARCHAR,credits INTEGER);
CREATE TABLE enrollments(enrollment_id INTEGER PRIMARY KEY,student_id INTEGER REFERENCES students(student_id),course_id INTEGER REFERENCES courses(course_id),score DECIMAL(5,2),term VARCHAR);
COMMENT ON COLUMN enrollments.score IS '百分制；>=60为及格；NULL表示尚未录入，AVG忽略NULL';
COMMENT ON COLUMN enrollments.term IS '如2025春、2024秋；统计须按题目限定学期';
""",
}


def records(domain: str, variant: int) -> dict[str, list[tuple]]:
    """小数据也包含一对多、重复业务值、NULL、无匹配行及跨月份/学期记录。"""
    if domain == "sales":
        data = {"customers": [(1,"张甲","北京"),(2,"李乙","上海"),(3,"王丙",None),(4,"无订单","杭州"),(5,"仅取消","深圳"),(6,"赵丁","北京")],
                "products": [(1,"电脑","数码"),(2,"鼠标","数码"),(3,"椅子","家具"),(4,"未成交商品","文具")],
                "orders": [(1,1,1,1,100,"2025-01-01","已完成"),(2,1,2,3,10,"2025-01-02","已完成"),(3,2,3,2,50,"2025-02-01","已完成"),(4,3,2,3,10,"2025-02-02","已完成"),(5,3,2,3,10,"2025-02-02","已完成"),(6,5,1,9,100,"2025-03-01","已取消"),(7,2,1,2,100,"2025-03-02","待付款"),(8,6,3,None,60,"2025-03-03","已完成"),(9,6,2,4,None,"2025-03-04","已完成"),(10,1,1,3,90,"2025-04-01","已完成")]}
        if variant:
            data["customers"] += [(7,"新客户","南京")]
            data["orders"] = [(i,c,p,q,None if price is None else price+7,d,"已完成" if i==7 else status) for i,c,p,q,price,d,status in data["orders"]]
            data["orders"] += [(11,7,4,2,3,"2025-01-03","已完成"),(12,2,1,1,77,"2024-12-31","已完成")]
        return data
    if domain == "library":
        data = {"authors": [(1,"作者甲"),(2,"作者乙"),(3,"无书作者")],
                "books": [(1,"Python入门",1,"技术"),(2,"SQL实践",1,"技术"),(3,"小说甲",2,"文学"),(4,"无人借阅",2,"文学")],
                "members": [(1,"读者甲","北京"),(2,"读者乙","上海"),(3,"未借读者","杭州"),(4,"读者丁",None)],
                "loans": [(1,1,1,"2025-01-01","2025-02-01","2025-01-20"),(2,3,1,"2025-02-01","2025-03-01",None),(3,1,2,"2025-03-01","2025-03-20",None),(4,2,2,"2025-03-02","2025-04-10",None),(5,1,1,"2025-03-03","2025-03-30",None),(6,3,4,"2025-03-04","2025-04-01",None)]}
        if variant:
            data["loans"] += [(7,4,3,"2025-04-01","2025-04-30",None),(8,2,1,"2025-02-04","2025-02-28","2025-03-02")]
            data["loans"][2] = (3,1,2,"2025-03-01","2025-03-20","2025-03-19")
        return data
    data = {"students": [(1,"学生甲","一年级"),(2,"学生乙","一年级"),(3,"学生丙","二年级"),(4,"无选课","二年级"),(5,"仅旧学期","三年级")],
            "courses": [(1,"数据库",3),(2,"Python",2),(3,"算法",4),(4,"未开课",1)],
            "enrollments": [(1,1,1,80,"2025春"),(2,1,2,90,"2025春"),(3,1,3,None,"2025春"),(4,2,1,59,"2025春"),(5,2,2,60,"2025春"),(6,3,1,None,"2025春"),(7,3,3,50,"2025春"),(8,5,1,100,"2024秋"),(9,1,1,70,"2024秋")]}
    if variant:
        data["enrollments"][3] = (4,2,1,85,"2025春")
        data["enrollments"] += [(10,4,4,95,"2025春"),(11,3,2,75,"2025春")]
    return data


KNOWLEDGE = {
    "sales": "成交指已完成订单。销售额使用历史成交单价quantity*unit_price。LEFT JOIN保留无订单客户时把订单状态放在ON；笔数COUNT(order_id)。NULL城市保留为一个分组。",
    "library": "一条loans记录是一次借阅，同一本书可以重复借阅。returned_on为NULL表示未归还。逾期为未归还且due_on早于题目给定固定日期。书目种数COUNT(book_id)，读者去重COUNT(DISTINCT member_id)。",
    "school": "成绩score>=60及格，NULL表示尚未录入。平均成绩忽略NULL。学分使用courses.credits。跨学期可能重复修同一课程，按题目过滤term。统计所有学生时保留零选课学生。",
}


def prepare(root: Path = DATA) -> dict:
    """只初始化新文件，已存在的数据保留；题集和知识文件内容需保持版本一致。"""
    root.mkdir(parents=True, exist_ok=True)
    for domain in SCHEMAS:
        for variant in (0, 1):
            path = root / f"{domain}_{variant}.duckdb"
            if path.exists():
                continue
            # 临时库完成后原子发布；固定数据的初始化不经过 LLM。
            temporary = root / f".{domain}_{variant}.building"
            if temporary.exists():
                raise ValueError(f"发现未完成初始化文件：{temporary}")
            try:
                with duckdb.connect(str(temporary)) as connection:
                    connection.execute(SCHEMAS[domain])
                    for table, rows in records(domain, variant).items():
                        placeholders = ",".join("?" for _ in rows[0])
                        connection.executemany(f"INSERT INTO {table} VALUES ({placeholders})", rows)
                    connection.execute("ANALYZE")
                # 使用链接而非覆盖，保留另一进程可能已完成的文件。
                import os
                os.link(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        knowledge = root / f"{domain}_knowledge.json"
        payload = [{"id":f"{domain}-definitions", "text":KNOWLEDGE[domain]}]
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        if knowledge.exists() and knowledge.read_text(encoding="utf-8") != text:
            raise ValueError(f"知识文件与当前题集版本不一致：{knowledge}")
        if not knowledge.exists():
            knowledge.write_text(text, encoding="utf-8")
    return {"version":DATASET_VERSION, "cases":[asdict(c) for c in cases()], "root":str(root.resolve())}
