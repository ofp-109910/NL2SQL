import json
from pathlib import Path
import tempfile
import unittest

from nl2sql_handbook.evaluation_demo.dataset import prepare
from nl2sql_handbook.failure_review.review import controlled, extract_failures, hypotheses


class ReviewTests(unittest.TestCase):
    def test_all_controlled_faults_are_exposed_without_api(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            data,out=root/"data",root/"review"
            prepare(data)
            out.mkdir()
            results=controlled(data,out)
            self.assertEqual(7,len(results))
            self.assertTrue(all(r["source"]=="controlled_fault" and r["detected"] for r in results))
            approval=next(r for r in results if r["id"]=="reviewer_false_approval")
            self.assertEqual("approved",approval["review"]["status"])
            self.assertFalse(approval["grade"]["passed"])

    def test_incomplete_benchmark_cannot_be_final_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/"summary.json").write_text('{"complete":false}')
            with self.assertRaisesRegex(ValueError,"尚未全部完成"):
                extract_failures(root)

    def test_frozen_manifest_and_manual_uncertainty_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            data={"summary.json":{"complete":True},
                  "manifest.json":{"cases":[{"id":"sales-04","domain":"sales","question":"冻结原问题","sql":"SELECT 1","tags":["aggregate","null"]}]},
                  "outcomes.json":[{"case":"sales-04","version":"multi_agent","repeat":1,"grade":{"passed":False,"category":"result_mismatch"},"result":{"sql":"SELECT 2","review":{"status":"approved"}}}]}
            for name,value in data.items():
                (root/name).write_text(json.dumps(value))
            result=extract_failures(root)[0]
            self.assertEqual("冻结原问题",result["question"])
            self.assertTrue(result["manual_review_required"])
            self.assertIn("题",result["policy_review"])
            self.assertIn("批准",result["review_observation"])

    def test_hypotheses_do_not_claim_proven_causes(self):
        hints=hypotheses(["left_join","date","null"])
        self.assertEqual(3,len(hints))
        self.assertTrue(all(h.startswith("核对") for h in hints))


if __name__=="__main__":
    unittest.main()
