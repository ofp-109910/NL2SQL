"""仅用于本地复评的子进程；参考答案不进入该进程。"""

import json
from pathlib import Path
import sys

from .runtime import query

if __name__ == "__main__":
    source, output = map(Path, sys.argv[1:])
    result = query(**json.loads(source.read_text(encoding="utf-8")))
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, default=str), encoding="utf-8")
    temporary.replace(output)
