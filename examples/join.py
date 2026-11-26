"""join 运行示例；参数见 python examples/join.py --help。"""
import sys
from nl2sql_handbook.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["join-workflow", *sys.argv[1:]]))
