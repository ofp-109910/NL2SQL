"""评分器、夹具、真实旧版本适配与子进程超时的离线测试。"""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel, FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from nl2sql_handbook.ai4db_demo.engine import Database, Limits, Budget
from nl2sql_handbook.evaluation_demo.adapters import FACTORIES, Meter, query
from nl2sql_handbook.evaluation_demo.dataset import cases, prepare
from nl2sql_handbook.evaluation_demo.grading import grade, rows_equal
from nl2sql_handbook.evaluation_demo.process import execute
from nl2sql_handbook.evaluation_demo.runner import aggregate, fingerprint, validate_references


class Scripted(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


def action(name, **kwargs):
    return AIMessage(content="",tool_calls=[{"name":name,"args":kwargs,"id":f"call-{name}","type":"tool_call"}])


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temporary.name)
        prepare(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def snapshot(self, case, sql=None):
        database=Database(self.root/f"{case.domain}_0.duckdb",Limits(),Budget())
        try:
            value=database.snapshot(sql or case.sql)
            return {"rows":value.rows,"truncated":False}
        finally:
            database.close()

    def test_dataset_split_and_all_references_executable(self):
        self.assertEqual(36,len(cases()))
        self.assertEqual(30,sum(c.split=="test" for c in cases()))
        self.assertEqual(36,len(validate_references(self.root)))

    def test_every_gold_passes_its_own_score_on_two_instances(self):
        for case in cases():
            with self.subTest(case=case.id):
                self.assertTrue(grade(case,case.sql,self.snapshot(case),self.root)["passed"])

    def test_prepare_preserves_existing_files(self):
        path=self.root/"sales_0.duckdb"
        before=path.read_bytes()
        prepare(self.root)
        self.assertEqual(before,path.read_bytes())

    def test_hardcoded_answer_caught_by_second_instance(self):
        case=cases()[0]
        value=self.snapshot(case)["rows"][0][0]
        wrong=f"SELECT {value}"
        checked=grade(case,wrong,{"rows":[[value]],"truncated":False},self.root)
        self.assertTrue(checked["checks"][0]["equal"])
        self.assertFalse(checked["checks"][1]["equal"])
        self.assertFalse(checked["passed"])

    def test_wrong_left_join_loses_unmatched_customers(self):
        case=next(c for c in cases() if c.id=="sales-06")
        wrong=case.sql.replace("LEFT JOIN","JOIN")
        self.assertFalse(grade(case,wrong,self.snapshot(case,wrong),self.root)["passed"])

    def test_duplicates_order_nulls_and_columns_are_scored(self):
        self.assertFalse(rows_equal([(1,),(1,)],[(1,)],False))
        self.assertFalse(rows_equal([(None,),(1,)],[(1,),(None,)],True))
        self.assertTrue(rows_equal([(None,),(1,)],[(1,),(None,)],False))
        case=cases()[0]
        wrong=case.sql.replace("COUNT(*)","COUNT(*), COUNT(*)")
        self.assertFalse(grade(case,wrong,self.snapshot(case,wrong),self.root)["passed"])

    def test_truncated_or_fabricated_return_is_rejected(self):
        case=cases()[0]
        self.assertFalse(grade(case,case.sql,{"rows":[[-999]],"truncated":False},self.root)["passed"])
        result=self.snapshot(case)
        result["truncated"]=True
        self.assertFalse(grade(case,case.sql,result,self.root)["passed"])

    def test_write_sql_blocked_before_execution(self):
        case=cases()[0]
        self.assertFalse(grade(case,"DELETE FROM orders",{},self.root)["passed"])
        self.assertGreater(self.snapshot(case)["rows"][0][0],0)

    def test_model_meter_enforces_budget_in_callback(self):
        meter=Meter(1)
        model=FakeListChatModel(responses=["ok"],callbacks=[meter])
        model.invoke("test")
        with self.assertRaisesRegex(RuntimeError,"预算耗尽"):
            model.invoke("test")
        self.assertEqual(1,meter.calls)

    def test_all_seven_adapters_execute_real_old_graphs(self):
        case=cases()[0]
        selection={"main.orders":["order_id","status"]}
        tools=[action("get_schema",selection=selection),action("execute_sql",sql=case.sql),AIMessage(content="完成")]
        for version in FACTORIES:
            if version in {"workflow","join_workflow","rag_workflow"}:
                chosen=selection if version=="workflow" else {"tables":selection,"relationship_ids":[]}
                llm=FakeListChatModel(responses=[json.dumps(chosen),case.sql])
            elif version=="multi_agent":
                llm=Scripted(responses=[action("delegate_retrieval",assignment="检索"),
                    action("search_knowledge",query=case.question),AIMessage(content="找到"),
                    action("delegate_sql",assignment="查询"),*tools,
                    action("delegate_review",assignment="核验"),action("get_candidate"),
                    action("submit_review",candidate_id=1,status="approved",feedback="确认"),
                    AIMessage(content="核验完成"),AIMessage(content="完成")])
            else:
                llm=Scripted(responses=tools)
            with self.subTest(version=version),patch("nl2sql_handbook.evaluation_demo.adapters.load_api_key",side_effect=AssertionError("离线不读密钥")):
                value=query(version,self.root/"sales_0.duckdb",case.question,knowledge=self.root/"sales_knowledge.json",llm=llm)
                self.assertTrue(value["ok"],value)
                self.assertTrue(grade(case,value["result"]["sql"],value["result"],self.root)["passed"])

    def test_subprocess_timeout_terminates_without_fake_zero_calls(self):
        result=execute({},timeout=.1,target="sleep_test")
        self.assertEqual("TaskTimeout",result["error_type"])
        self.assertIsNone(result["model_calls"])

    def test_incomplete_runs_and_unknown_cost_not_hidden(self):
        outcome={"version":"workflow","domain":"sales","grade":{"passed":True},"ok":True,"elapsed_ms":1,"model_calls":None}
        result=aggregate([outcome],["workflow"],2)["workflow"]
        self.assertFalse(result["complete"])
        self.assertEqual(1,result["unknown_call_counts"])
        self.assertIsNone(result["mean_calls_measured"])

    def test_resume_fingerprint_captures_settings_and_case_changes(self):
        a=fingerprint({"model":"a"},cases()[:1],self.root)
        b=fingerprint({"model":"b"},cases()[:1],self.root)
        self.assertNotEqual(a["fingerprint"],b["fingerprint"])


if __name__=="__main__":
    unittest.main()
