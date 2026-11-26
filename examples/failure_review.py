"""failure_review 运行示例；参数见 python examples/failure_review.py --help。"""
import sys
from nl2sql_handbook.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["review", *sys.argv[1:]]))
