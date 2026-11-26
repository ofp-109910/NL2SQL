"""最小本机HTTP服务，仅用标准库，不改变已有Python环境依赖。

主入口固定选择一个已存在的实现（默认RAG Workflow）。服务请求只传问题
和数据库ID；模型、密钥、文件路径、预算由本地配置控制。每题独立子进程，
硬超时后回收，业务连接仍使用旧版本的只读/外部访问限制。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import BoundedSemaphore, Lock
from time import perf_counter
from uuid import uuid4

from nl2sql_handbook.evaluation_demo.process import execute
from .config import Config, ROOT, load

MAX_BODY=16_384


def validate_body(value, config: Config):
    if not isinstance(value,dict) or value.keys()-{"question","database"}:
        raise ValueError("只接受question和database字段")
    question,database=value.get("question"),value.get("database")
    if not isinstance(question,str) or not 1<=len(question.strip())<=2000:
        raise ValueError("question必须为1—2000字符的非空文本")
    if not isinstance(database,str) or database not in config.databases:
        raise ValueError("database必须使用已配置ID")
    return question.strip(),database


class Service(ThreadingHTTPServer):
    # 关闭服务时等待已接受请求按其硬期限结束，避免把子进程遗留在后台。
    daemon_threads=False
    block_on_close=True

    def __init__(self,config: Config,executor=execute):
        self.config,self.executor=config,executor
        self.slots=BoundedSemaphore(config.max_concurrent)
        self.log_lock=Lock()
        config.log_file.parent.mkdir(parents=True,exist_ok=True)
        with config.log_file.open("a",encoding="utf-8"):
            pass
        super().__init__((config.host,config.port),Handler)

    def event(self,record):
        # 不记录问题、SQL、结果、API Key、请求头或provider原始错误。
        with self.log_lock,self.config.log_file.open("a",encoding="utf-8") as file:
            file.write(json.dumps({"time":datetime.now(timezone.utc).isoformat(),**record},ensure_ascii=False)+"\n")


class Handler(BaseHTTPRequestHandler):
    protocol_version="HTTP/1.0"
    server_version="NL2SQLLearning/1"

    def setup(self):
        super().setup()
        self.connection.settimeout(5)  # 限制不完整请求头/请求体占用连接的时间。

    def log_message(self,*args):
        pass  # 标准库默认访问日志会包含原始URL，统一改用下面的结构化事件。

    def respond(self,status,payload):
        content=json.dumps(payload,ensure_ascii=False,default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Content-Length",str(len(content)))
        self.send_header("Cache-Control","no-store")
        self.end_headers()
        try:
            self.wfile.write(content)
        except (BrokenPipeError,ConnectionResetError):
            pass

    def do_GET(self):
        if self.path!="/health":
            self.respond(404,{"error":"not_found"})
            return
        config=self.server.config
        # 健康检查只证明HTTP服务和注册表可用，不读取密钥，不声称API可达。
        self.respond(200,{"status":"ok","version":config.version,"model":config.model,
                          "databases":sorted(config.databases),"read_only":True})

    def do_POST(self):
        identifier,started=uuid4().hex,perf_counter()
        record={"request_id":identifier,"version":self.server.config.version}
        status,payload=500,{"request_id":identifier,"error":"internal_error"}
        acquired=False
        try:
            if self.path!="/query":
                status,payload=404,{"request_id":identifier,"error":"not_found"}
                return
            if self.headers.get("Transfer-Encoding") or len(self.headers.get_all("Content-Length",[]))!=1:
                status,payload=400,{"request_id":identifier,"error":"需要单一Content-Length"}
                return
            try:
                size=int(self.headers["Content-Length"])
            except (TypeError,ValueError):
                size=-1
            if size<0 or size>MAX_BODY:
                status,payload=(413 if size>MAX_BODY else 400),{"request_id":identifier,"error":"请求体长度无效"}
                return
            if self.headers.get_content_type()!="application/json":
                status,payload=415,{"request_id":identifier,"error":"需要application/json"}
                return
            try:
                raw=self.rfile.read(size)
                if len(raw)!=size:
                    raise ValueError("请求体不完整")
                def unique(pairs):
                    result={}
                    for key,value in pairs:
                        if key in result:
                            raise ValueError("JSON字段重复")
                        result[key]=value
                    return result
                value=json.loads(raw.decode("utf-8"),object_pairs_hook=unique,
                                 parse_constant=lambda _:(_ for _ in ()).throw(ValueError("JSON常量无效")))
                question,database=validate_body(value,self.server.config)
            except (ValueError,UnicodeError,TimeoutError):
                status,payload=400,{"request_id":identifier,"error":"请求字段、JSON或数据库ID无效"}
                return
            record["database"]=database
            if not self.server.slots.acquire(blocking=False):
                status,payload=429,{"request_id":identifier,"error":"服务繁忙，请稍后重试"}
                return
            acquired=True
            config=self.server.config
            request={**config.databases[database],"question":question,"version":config.version,
                     "model":config.model,"key_file":str(config.key_file),
                     "max_calls":config.max_model_calls,"max_rows":config.max_rows}
            outcome=self.server.executor(request,timeout=config.request_timeout)
            metrics={key:outcome.get(key) for key in ["model_calls","model_responses","tokens","elapsed_ms","harness_wall_ms","response_models"]}
            record.update(model_calls=outcome.get("model_calls"),error_type=outcome.get("error_type"))
            if not outcome.get("ok"):
                status=504 if outcome.get("error_type")=="TaskTimeout" else 502
                payload={"request_id":identifier,"error":"查询未完成","error_type":outcome.get("error_type"),"metrics":metrics}
                return
            result=outcome["result"]
            status,payload=200,{"request_id":identifier,"database":database,"version":config.version,
                                **{key:result[key] for key in ["sql","columns","rows","truncated"]},"metrics":metrics}
        except Exception as exc:
            record["error_type"]=type(exc).__name__
        finally:
            if acquired:
                self.server.slots.release()
            record.update(status=status,elapsed_ms=round((perf_counter()-started)*1000,2))
            try:
                self.server.event(record)
            except OSError:
                # 保留请求ID，不把日志I/O错误泄露成私有路径或覆盖已完成的查询。
                print(json.dumps({"request_id":identifier,"error":"request_log_write_failed"}),flush=True)
            self.respond(status,payload)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=None)
    args=parser.parse_args(argv)
    server=Service(load(args.config))
    print(json.dumps({"listening":f"http://{server.server_address[0]}:{server.server_address[1]}",
                      "version":server.config.version,"databases":sorted(server.config.databases)},ensure_ascii=False),flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
