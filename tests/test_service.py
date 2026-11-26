"""本地真实HTTP协议测试；模型使用替身，硬超时使用真实子进程。"""

from dataclasses import replace
from decimal import Decimal
import http.client
import json
from pathlib import Path
import tempfile
from threading import Event, Thread
import unittest
from unittest.mock import patch

from nl2sql_handbook.evaluation_demo.dataset import prepare
from nl2sql_handbook.evaluation_demo.process import execute
from nl2sql_handbook.delivery_demo.config import load
from nl2sql_handbook.delivery_demo.server import MAX_BODY, Service, validate_body


def success(request,timeout):
    return {"ok":True,"model_calls":2,"elapsed_ms":1,
            "result":{"sql":"SELECT 12.50 AS amount","columns":["amount"],"rows":[[Decimal("12.50")]],"truncated":False}}


class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temporary.name)
        prepare(cls.root/"data")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.config_file=self.root/"config.json"
        self.config_file.write_text(json.dumps({"port":0,"log_file":"events.jsonl","databases":{
            "sales":{"database":"data/sales_0.duckdb","knowledge":"data/sales_knowledge.json"}}}))
        self.config=load(self.config_file)
        self.server=None

    def start(self,executor=success,config=None):
        self.server=Service(config or self.config,executor)
        self.thread=Thread(target=self.server.serve_forever,kwargs={"poll_interval":.02},daemon=True)
        self.thread.start()

    def tearDown(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join()

    def request(self,body=None,method="POST",path="/query",headers=None):
        connection=http.client.HTTPConnection(*self.server.server_address,timeout=5)
        try:
            raw=json.dumps(body,ensure_ascii=False).encode() if isinstance(body,(dict,list)) else body
            connection.request(method,path,body=raw,headers=headers or {"Content-Type":"application/json"})
            response=connection.getresponse()
            return response.status,json.loads(response.read())
        finally:
            connection.close()

    def test_health_reads_no_key_or_model(self):
        self.start()
        with patch("nl2sql_handbook.evaluation_demo.adapters.load_api_key",side_effect=AssertionError("不读密钥")):
            status,value=self.request(method="GET",path="/health")
        self.assertEqual(200,status)
        self.assertTrue(value["read_only"])
        self.assertNotIn("key_file",value)

    def test_result_precision_and_request_id(self):
        self.start()
        status,value=self.request({"question":"销售额","database":"sales"})
        self.assertEqual(200,status)
        self.assertEqual([["12.50"]],value["rows"])
        self.assertEqual(32,len(value["request_id"]))

    def test_invalid_body_never_calls_executor(self):
        calls=[]
        self.start(lambda *a,**k:calls.append(a))
        for body in [{},{"question":" ","database":"sales"},{"question":"x","database":"../../etc"},
                     {"question":"x","database":"sales","model":"override"},{"question":"x"*2001,"database":"sales"},[]]:
            self.assertEqual(400,self.request(body)[0])
        self.assertEqual([],calls)

    def test_duplicate_keys_invalid_json_and_media_type(self):
        self.start()
        self.assertEqual(400,self.request(b'{"question":"a","question":"b","database":"sales"}')[0])
        self.assertEqual(400,self.request(b'{broken')[0])
        self.assertEqual(400,self.request(b'{"question":NaN,"database":"sales"}')[0])
        self.assertEqual(415,self.request(b'{}',headers={"Content-Type":"text/plain"})[0])

    def test_oversized_body_and_unknown_endpoint(self):
        self.start()
        self.assertEqual(413,self.request(b'x'*(MAX_BODY+1))[0])
        self.assertEqual(404,self.request({},path="/unknown")[0])

    def test_concurrency_returns_busy_then_recovers(self):
        entered,release=Event(),Event()
        def slow(request,timeout):
            entered.set()
            release.wait(3)
            return success(request,timeout)
        self.start(slow,replace(self.config,max_concurrent=1))
        values=[]
        thread=Thread(target=lambda:values.append(self.request({"question":"q","database":"sales"})))
        thread.start()
        try:
            self.assertTrue(entered.wait(2))
            self.assertEqual(429,self.request({"question":"q2","database":"sales"})[0])
        finally:
            release.set()
            thread.join()
        self.assertEqual(200,values[0][0])
        self.assertEqual(200,self.request({"question":"q3","database":"sales"})[0])

    def test_real_process_timeout_is_504_and_unknown_cost(self):
        self.start(lambda request,timeout:execute({},timeout=.1,target="sleep_test"))
        status,value=self.request({"question":"q","database":"sales"})
        self.assertEqual(504,status)
        self.assertEqual("TaskTimeout",value["error_type"])
        self.assertIsNone(value["metrics"]["model_calls"])

    def test_provider_error_does_not_expose_secret(self):
        self.start(lambda *a,**k:{"ok":False,"error_type":"ProviderError","error":"sk-SECRET_PRIVATE","model_calls":1})
        status,value=self.request({"question":"q","database":"sales"})
        self.assertEqual(502,status)
        self.assertNotIn("SECRET",json.dumps(value))

    def test_logs_omit_question_sql_rows_and_key(self):
        self.start()
        status,value=self.request({"question":"PRIVATE_QUESTION_TEXT","database":"sales"})
        self.assertEqual(200,status)
        line=[json.loads(line) for line in self.config.log_file.read_text().splitlines() if value["request_id"] in line][0]
        self.assertEqual({"time","request_id","version","database","model_calls","error_type","status","elapsed_ms"},set(line))
        self.assertNotIn("PRIVATE",json.dumps(line))
        self.assertNotIn("SELECT",json.dumps(line))

    def test_unexpected_error_releases_slot_and_is_sanitized(self):
        calls=[]
        def fail_once(request,timeout):
            calls.append(1)
            if len(calls)==1:
                raise RuntimeError("sk-SECRET")
            return success(request,timeout)
        self.start(fail_once,replace(self.config,max_concurrent=1))
        first=self.request({"question":"q","database":"sales"})
        self.assertEqual(500,first[0])
        self.assertNotIn("SECRET",json.dumps(first))
        self.assertEqual(200,self.request({"question":"q","database":"sales"})[0])

    def test_configuration_strictness_and_relative_paths(self):
        self.assertEqual((self.root/"data/sales_0.duckdb").resolve(),Path(self.config.databases["sales"]["database"]))
        for updates in [{"max_rows":True},{"max_concurrent":0},{"request_timeout":float("nan")},{"api_key":"never-store-this"}]:
            value=json.loads(self.config_file.read_text())
            value.update(updates)
            self.config_file.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                load(self.config_file)
            self.setUp()


if __name__=="__main__":
    unittest.main()
