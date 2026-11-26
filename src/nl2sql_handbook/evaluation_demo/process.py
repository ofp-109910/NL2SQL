"""通过独立 Python 子进程设置整题硬超时，超时后回收进程再返回。

    输出用每次任务私有目录内的原子文件交接，避免大 Queue 消息等待 join
    造成死锁。未完成文件永远不能作为成功结果；不向参数传递 API Key。
"""

from pathlib import Path
import json
import subprocess
import sys
import tempfile
from time import perf_counter

PROJECT = Path(__file__).resolve().parent.parent


def execute(request: dict, *, timeout: float = 120, target="query") -> dict:
    if not 0.05 <= timeout <= 600 or target not in {"query","sleep_test"}:
        raise ValueError("无效的进程超时或内部任务类型")
    started = perf_counter()
    with tempfile.TemporaryDirectory(prefix="nl2sql-task-") as directory:
        root = Path(directory)
        input_file, output_file = root/"input.json", root/"output.json"
        input_file.write_text(json.dumps({"target":target,"request":request},ensure_ascii=False), encoding="utf-8")
        # 不收集 provider stderr，防止把底层 HTTP 诊断误当成可公开日志。
        process = subprocess.Popen([sys.executable,"-m","nl2sql_handbook.evaluation_demo.worker",str(input_file),str(output_file)],
                                   cwd=None, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            # 无计量证据时用 None，避免将已消耗但丢失的 API 调用错误记成 0。
            return {"ok":False,"error_type":"TaskTimeout","error":"整题硬超时，工作进程已回收",
                    "model_calls":None,"tokens":None,"elapsed_ms":(perf_counter()-started)*1000,
                    "harness_wall_ms":(perf_counter()-started)*1000}
        if process.returncode != 0 or not output_file.is_file():
            return {"ok":False,"error_type":"WorkerFailure","error":"工作进程未生成完整结果",
                    "model_calls":None,"tokens":None,"elapsed_ms":(perf_counter()-started)*1000,
                    "harness_wall_ms":(perf_counter()-started)*1000}
        result = json.loads(output_file.read_text(encoding="utf-8"))
        result["harness_wall_ms"] = (perf_counter()-started)*1000
        return result
