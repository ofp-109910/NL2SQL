"""补充的仓储迁移题：新表名、两跳关系和两个不同数据实例。

这是一组人工编写的小型诊断题，不是独立公开盲测；不参与程序或提示词的运行。
参考SQL只由父评估进程读取。新数据库均放在本隔离目录。
"""

from pathlib import Path
import duckdb

from nl2sql_handbook.evaluation_demo.dataset import Case

from nl2sql_handbook.config import SETTINGS
ROOT = SETTINGS.output_dir / "data" / "transfer"


def prepare():
    ROOT.mkdir(parents=True, exist_ok=True)
    for variant in (0, 1):
        path = ROOT / f"warehouse_{variant}.duckdb"
        if path.exists():
            continue
        with duckdb.connect(str(path)) as db:
            db.execute("""
CREATE TABLE regions(region_key INTEGER PRIMARY KEY,label VARCHAR);
CREATE TABLE depots(depot_key INTEGER PRIMARY KEY,region_key INTEGER REFERENCES regions(region_key),title VARCHAR);
CREATE TABLE movements(movement_key INTEGER PRIMARY KEY,depot_key INTEGER REFERENCES depots(depot_key),delta INTEGER,posted_on DATE,state VARCHAR);
COMMENT ON COLUMN movements.delta IS '有符号库存变动量；负数为出库，正数为入库；NULL为数量尚未录入';
COMMENT ON COLUMN movements.state IS '已入账或已撤销；每一行代表一次独立库存变动';
""")
            # 两实例的ID不共享；不仅改数值，也改变匹配范围、空值和分组分布。
            offset = 1000 if variant else 100
            regions = [(offset+1, "东区"), (offset+2, "西区"), (offset+3, None), (offset+4, "空区")]
            depots = [(offset+i, offset+r, f"仓{i}") for i, r in [(1,1),(2,1),(3,2),(4,3),(5,2)]]
            movements = [
                (1,1,10,"2025-01-01","已入账"), (2,1,-4,"2025-01-31","已入账"),
                (3,1,-4,"2025-02-01","已入账"), (4,2,-8,"2025-02-03","已入账"),
                (5,2,None,"2025-02-04","已入账"), (6,3,99,"2025-02-10","已撤销"),
                (7,4,None,"2025-03-01","已入账"), (8,1,-20,"2025-04-01","已入账")]
            if variant:
                movements += [(9,3,-3,"2025-03-04","已入账"), (10,4,5,"2025-03-05","已入账")]
                movements = [(i,d,q*2 if q is not None else None,date,state) for i,d,q,date,state in movements]
            db.executemany("INSERT INTO regions VALUES (?,?)", regions)
            db.executemany("INSERT INTO depots VALUES (?,?,?)", depots)
            db.executemany("INSERT INTO movements VALUES (?,?,?,?,?)",
                           [(offset+i,offset+d,q,date,state) for i,d,q,date,state in movements])
    return ROOT


def cases():
    def case(number, tags, question, sql):
        return Case(f"warehouse-{number:02}", "warehouse", "transfer", tuple(tags.split()), True, question, sql)
    return [
        case(1, "bridge join aggregate null", "按区域名称统计已入账库存变动量的合计，仅返回实际有已入账记录的区域，保留NULL区域名及SUM原生NULL语义。依次返回区域名、合计，区域名升序NULL最后。",
             "SELECT r.label,SUM(m.delta) FROM regions r JOIN depots d ON d.region_key=r.region_key JOIN movements m ON m.depot_key=d.depot_key WHERE m.state='已入账' GROUP BY r.label ORDER BY r.label NULLS LAST"),
        case(2, "left_join aggregate", "列出所有仓库的已入账记录数和库存变动量合计，包括没有已入账记录的仓库，合计为NULL时显示0。依次返回仓库编号、记录数、合计，仓库编号升序。",
             "SELECT d.depot_key,COUNT(m.movement_key),COALESCE(SUM(m.delta),0) FROM depots d LEFT JOIN movements m ON m.depot_key=d.depot_key AND m.state='已入账' GROUP BY d.depot_key ORDER BY d.depot_key"),
        case(3, "anti_join", "找出没有任何已入账库存变动记录的仓库，依次返回仓库编号、名称，编号升序。",
             "SELECT d.depot_key,d.title FROM depots d WHERE NOT EXISTS(SELECT 1 FROM movements m WHERE m.depot_key=d.depot_key AND m.state='已入账') ORDER BY d.depot_key"),
        case(4, "date filter aggregate", "按YYYY-MM月份统计2025年第一季度已入账出库记录的笔数和出库总量。出库为delta小于0，总量取负号后为正数；日期包含1月1日不含4月1日。依次返回月份、笔数、总量，月份升序。",
             "SELECT strftime(posted_on,'%Y-%m') AS period,COUNT(*),SUM(-delta) FROM movements WHERE state='已入账' AND delta<0 AND posted_on>=DATE '2025-01-01' AND posted_on<DATE '2025-04-01' GROUP BY period ORDER BY period"),
        case(5, "bridge distinct null", "按区域名称统计存在已入账变动记录的不同仓库数量，一个仓库只计一次，仅返回存在此类仓库的区域，保留NULL区域名。依次返回区域名、仓库数，区域名升序NULL最后。",
             "SELECT r.label,COUNT(DISTINCT d.depot_key) FROM regions r JOIN depots d ON d.region_key=r.region_key JOIN movements m ON m.depot_key=d.depot_key WHERE m.state='已入账' GROUP BY r.label ORDER BY r.label NULLS LAST"),
        case(6, "having aggregate duplicates", "找出已入账库存变动量合计为负数的仓库，NULL数量遵守SUM语义，每条变动记录独立累计。依次返回仓库编号、合计，编号升序。",
             "SELECT depot_key,SUM(delta) FROM movements WHERE state='已入账' GROUP BY depot_key HAVING SUM(delta)<0 ORDER BY depot_key"),
    ]
