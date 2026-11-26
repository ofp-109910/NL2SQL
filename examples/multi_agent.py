"""multi_agent 运行示例；参数见 python examples/multi_agent.py --help。"""
import sys
from nl2sql_handbook.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["multi-agent", *sys.argv[1:]]))
