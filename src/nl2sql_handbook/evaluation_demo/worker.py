"""内部进程入口；从本地任务文件读参数，不提供任何网络服务。"""

import json
from pathlib import Path
import sys
import time


def main():
    source, output = map(Path,sys.argv[1:])
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload["target"] == "sleep_test":
        time.sleep(10)  # 只用于验证父进程确实会杀死超时进程，不调用模型。
        result = {"ok":False}
    else:
        from .adapters import query
        result = query(**payload["request"])
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result,ensure_ascii=False,default=str),encoding="utf-8")
    temporary.replace(output)


if __name__ == "__main__":
    main()
