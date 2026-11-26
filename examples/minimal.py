"""minimal 运行示例；参数见 python examples/minimal.py --help。"""
import sys
from nl2sql_handbook.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["minimal", *sys.argv[1:]]))
