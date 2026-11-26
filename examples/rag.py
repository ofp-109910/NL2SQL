"""rag 运行示例；参数见 python examples/rag.py --help。"""
import sys
from nl2sql_handbook.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["rag", *sys.argv[1:]]))
