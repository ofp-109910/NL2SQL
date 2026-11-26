"""配置只由服务启动者读取。HTTP 请求不能覆盖模型、路径、版本或预算。"""

from dataclasses import dataclass
import json
from pathlib import Path

from nl2sql_handbook.nl2sql_demo import DEFAULT_KEY_FILE, DEFAULT_MODEL
from nl2sql_handbook.evaluation_demo.adapters import FACTORIES

from nl2sql_handbook.config import SETTINGS
from nl2sql_handbook.evaluation_demo.dataset import DATA
ROOT=SETTINGS.output_dir/"service"


@dataclass(frozen=True)
class Config:
    host: str
    port: int
    version: str
    model: str
    key_file: Path
    request_timeout: float
    max_concurrent: int
    max_model_calls: int
    max_rows: int
    log_file: Path
    databases: dict


def load(path: Path | None=None) -> Config:
    if path is None:
        path = ROOT / "defaults.json"
        values = {"log_file": str(SETTINGS.output_dir/"logs"/"requests.jsonl"),
                  "databases": {name: {"database": str(DATA/f"{name}_0.duckdb"),
                                      "knowledge": str(DATA/f"{name}_knowledge.json")}
                                for name in ("sales", "library", "school")}}
    else:
        path=path.expanduser().resolve()
        values=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values,dict):
        raise ValueError("配置必须为JSON对象")
    allowed={"host","port","version","model","key_file","request_timeout","max_concurrent","max_model_calls","max_rows","log_file","databases"}
    if values.keys()-allowed:
        raise ValueError("存在未知配置项")
    def number(name,default,lower,upper,integer=True):
        value=values.get(name,default)
        valid=type(value) is int if integer else type(value) in (int,float)
        if not valid or not lower<=value<=upper:
            raise ValueError(f"无效配置：{name}")
        return value
    def resolve(value):
        if not isinstance(value,str) or not value.strip():
            raise ValueError("路径必须为非空字符串")
        result=Path(value).expanduser()
        return (result if result.is_absolute() else path.parent/result).resolve()
    host=values.get("host","127.0.0.1")
    if not isinstance(host,str) or host not in {"127.0.0.1","localhost","0.0.0.0"}:
        raise ValueError("host支持127.0.0.1、localhost或显式0.0.0.0")
    version=values.get("version","rag_workflow")
    model=values.get("model",DEFAULT_MODEL)
    if version not in FACTORIES or not isinstance(model,str) or not model.strip():
        raise ValueError("无效版本或模型")
    registry=values.get("databases")
    if not isinstance(registry,dict) or not registry:
        raise ValueError("必须配置数据库注册表")
    databases={}
    for identifier,entry in registry.items():
        if not isinstance(identifier,str) or not identifier.isidentifier() or not isinstance(entry,dict) or entry.keys()-{"database","knowledge"}:
            raise ValueError("无效数据库注册项")
        database=resolve(entry.get("database"))
        knowledge=resolve(entry["knowledge"]) if entry.get("knowledge") else None
        if not database.is_file() or (knowledge and not knowledge.is_file()):
            raise ValueError("数据库或知识文件不存在，请先执行 nl2sql init --evaluation")
        databases[identifier]={"database":str(database),"knowledge":str(knowledge) if knowledge else None}
    return Config(host,number("port",8765,0,65535),version,model,
                  resolve(values.get("key_file",str(DEFAULT_KEY_FILE))),
                  number("request_timeout",90,.1,600,False),number("max_concurrent",2,1,4),
                  number("max_model_calls",24,1,30),number("max_rows",100,1,1000),
                  resolve(values.get("log_file","logs/requests.jsonl")),databases)
