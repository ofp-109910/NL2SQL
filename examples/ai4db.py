"""ai4db 运行示例；参数见 python examples/ai4db.py --help。"""
import sys
from nl2sql_handbook.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["ai4db", *sys.argv[1:]]))
