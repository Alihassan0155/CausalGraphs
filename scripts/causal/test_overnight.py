"""Lightweight supervisor tests: fake workers only; never runs causal discovery."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import discover as runner


class FakeWorker:
    def __init__(self, command, **kwargs):
        self.pid=2147483646
        self.returncode=0
        self.polls=0
        algo=command[command.index('--algorithm')+1]
        run_id=command[command.index('--run-id')+1]
        folder=runner.BASE/algo/run_id
        runner.atomic_json(folder/'algorithm_result.json',{'worker_peak_rss_bytes':123,'nodes':2})

    def poll(self):
        self.polls+=1
        return None if self.polls==1 else self.returncode

    def wait(self): return self.returncode


class OvernightTests(unittest.TestCase):
    def test_no_timeout_and_explicit_budget(self):
        for limit in [None,0]:
            self.assertFalse(runner.time_budget_exhausted(limit,10**12))
        self.assertFalse(runner.time_budget_exhausted(3600,3599))
        self.assertTrue(runner.time_budget_exhausted(3600,3600))

    def test_reviewed_legacy_configuration(self):
        saved=json.loads((runner.ROOT/'graphs/causal_split0/pc/safety_pc_resume/config.json').read_text())
        current=copy.deepcopy(saved)
        current.update(wrapper_sha256=runner.sha(Path(runner.__file__)),
                       algorithm_state_contract_sha256=runner.recovery_contract())
        self.assertTrue(runner.compatible_configuration(saved,current))
        current['seed']+=1
        self.assertFalse(runner.compatible_configuration(saved,current))
        current['seed']-=1
        current['algorithm_state_contract_sha256']='changed algorithm'
        self.assertFalse(runner.compatible_configuration(saved,current))

    def invoke_fake(self, algorithm, run_id, extra=()):
        args=['discover.py','--algorithm',algorithm,'--run-id',run_id,'--nodes','2','--cells','4',*extra]
        with patch.object(runner.sys,'argv',args), patch.object(runner.subprocess,'Popen',FakeWorker), \
             patch.object(runner.subprocess,'check_output',return_value='test-commit\n'), \
             patch.object(runner.time,'sleep',lambda seconds:None):
            runner.main()

    def test_unlimited_completion_saves_results_and_protects_them(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runner,'BASE',Path(temp)):
            self.invoke_fake('pc','unlimited')
            folder=Path(temp)/'pc/unlimited'
            status=json.loads((folder/'status.json').read_text())
            self.assertEqual(status['status'],'complete')
            self.assertIsNone(status['time_limit_seconds'])
            self.assertFalse(status['timeout_enabled'])
            self.assertTrue((folder/'statistics.json').exists())
            original=(folder/'statistics.json').read_bytes()
            with self.assertRaisesRegex(SystemExit,'Completed experiment protected'):
                self.invoke_fake('pc','unlimited',('--resume',))
            self.assertEqual(original,(folder/'statistics.json').read_bytes())

    def test_boss_explicit_restart_is_unlimited(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runner,'BASE',Path(temp)):
            self.invoke_fake('boss','restart')
            folder=Path(temp)/'boss/restart'
            state=json.loads((folder/'status.json').read_text())
            state['status']='interrupted'
            runner.atomic_json(folder/'status.json',state)
            self.invoke_fake('boss','restart',('--restart-stage',))
            state=json.loads((folder/'status.json').read_text())
            self.assertEqual(state['status'],'complete')
            self.assertEqual(state['recovery_mode'],'restart')
            self.assertIsNone(state['time_limit_seconds'])

    def test_explicit_timeout_still_stops_worker(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(runner,'BASE',Path(temp)), \
             patch.object(runner,'time_budget_exhausted',return_value=True), \
             patch.object(runner,'stop_worker_tree') as stop:
            with self.assertRaises(SystemExit):
                self.invoke_fake('pc','limited',('--time-limit-seconds','1',))
            stop.assert_called_once()
            state=json.loads((Path(temp)/'pc/limited/status.json').read_text())
            self.assertEqual(state['status'],'timed_out')
            self.assertTrue(state['timeout_enabled'])


if __name__=='__main__': unittest.main()
