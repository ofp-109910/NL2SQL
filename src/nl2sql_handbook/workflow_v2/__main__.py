"""python -m nl2sql_handbook.workflow_v2 --variant basic|join|rag --db 数据库 -q 问题。"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from nl2sql_handbook.nl2sql_demo import DEFAULT_DB, DEFAULT_KEY_FILE, DEFAULT_MODEL, report_error
from .workflow import ImprovedWorkflow, ImprovedJoinWorkflow, ImprovedRAGWorkflow


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["basic", "join", "rag"], default="basic")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("-q", "--question", required=True)
    parser.add_argument("--knowledge-file", type=Path, action="append")
    parser.add_argument("--relationships-file", type=Path)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_KEY_FILE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args(argv)
    factory = dict(basic=ImprovedWorkflow, join=ImprovedJoinWorkflow, rag=ImprovedRAGWorkflow)[args.variant]
    try:
        with factory(args.db, relationships_file=args.relationships_file, knowledge_files=args.knowledge_file,
                     key_file=args.api_key_file, model=args.model) as demo:
            result = demo.ask(args.question, explain=args.explain)
            print(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str))
        return 0
    except Exception as exc:
        report_error(exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
