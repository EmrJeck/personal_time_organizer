import copy
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from app.recommender import generate_plan
from experiments.run_supervisor_15 import generate_pair, evaluate, run, METHODS
from experiments.run_recency_comparison import planner_inputs


class SupervisorScenarioTests(unittest.TestCase):
    def test_exact_capacity_grid_and_individual_feasibility(self):
        for repeat in range(5):
            case, _ = generate_pair(repeat)
            inputs = planner_inputs(case)
            windows = sum((b-a).total_seconds()/60 for a, b in inputs['available_windows'])
            busy = sum((b-a).total_seconds()/60 for a, b in inputs['busy_intervals'])
            self.assertEqual((windows, busy, windows-busy), (1800, 720, 1080))
            self.assertEqual(len(inputs['tasks']), 15)
            self.assertEqual(case['independently_feasible_count'], 15)
            for t in inputs['tasks']:
                self.assertIn((t.importance-1)/4, (.25, .5, .75))
                self.assertIn(t.duration_min, range(15, 121, 15))
                self.assertLessEqual(t.deadline, inputs['available_windows'][-1][1])

    def test_paired_history_conditions_preserve_tasks_and_hidden_world(self):
        primary, cold = generate_pair(0)
        self.assertEqual((primary, cold), generate_pair(0))
        self.assertNotEqual(primary['planner']['tasks'], generate_pair(1)[0]['planner']['tasks'])
        self.assertEqual(primary['planner']['tasks'], cold['planner']['tasks'])
        self.assertEqual(primary['simulation'], cold['simulation'])
        self.assertEqual(len(primary['planner']['history_observations']), 240)
        self.assertEqual(cold['planner']['history_observations'], [])
        self.assertTrue(all(r['total'] == 0 for r in cold['planner']['history']))
        cutoff = datetime.fromisoformat(primary['planner']['as_of'])
        self.assertTrue(all(datetime.fromisoformat(r['occurred_at']) < cutoff
                            for r in primary['planner']['history_observations']))

    def test_value_and_utilization_are_independent_of_future_execution(self):
        case, _ = generate_pair(0)
        result = generate_plan(**planner_inputs(case), method='importance')
        metrics = evaluate(case, result)
        expected = sum((t['importance']-1)/4 for t in case['planner']['tasks']
                       if t['id'] in {a.task_id for a in result.assignments})
        self.assertEqual(metrics['normalized_value'], expected)
        self.assertEqual(metrics['total_violations'], 0)
        altered = copy.deepcopy(case)
        for row in altered['simulation']['future_task_noise'].values():
            row['noise'] = 20
        new = evaluate(altered, result)
        self.assertEqual(new['normalized_value'], expected)
        self.assertEqual(new['utilization_percent'], metrics['utilization_percent'])
        self.assertEqual(new['simulated_on_time_importance_fraction'], 0)

    def test_cold_start_neutral_and_all_methods_respect_constraints(self):
        _, cold = generate_pair(0)
        results = {m: generate_plan(**planner_inputs(cold), method=m) for m in METHODS}
        signature = lambda r: [(a.task_id, a.start, a.end) for a in r.assignments]
        self.assertEqual(signature(results['fuzzy']), signature(results['personalized']))
        self.assertEqual(signature(results['fuzzy']), signature(results['personalized_recent']))
        for result in results.values():
            self.assertEqual(evaluate(cold, result)['total_violations'], 0)

    def test_export_is_complete_and_refuses_overwriting(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch('app.create_app', side_effect=AssertionError('App forbidden')), \
             patch('smtplib.SMTP', side_effect=AssertionError('Mail forbidden')), \
             patch('smtplib.SMTP_SSL', side_effect=AssertionError('Mail forbidden')):
            path = Path(directory)/'output'
            manifest = run(path, repeats=1, timing_repeats=1, progress=False)
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(manifest['metric_rows'], 16)
            self.assertEqual(manifest['total_violations'], 0)
            self.assertEqual(len((path/'assignments.jsonl').read_text(encoding='utf-8').splitlines()), 16)
            self.assertFalse(json.loads((path/'config.json').read_text(encoding='utf-8'))['exact_supervisor_data_available'])
            with self.assertRaises(FileExistsError):
                run(path, repeats=1, timing_repeats=1, progress=False)


if __name__ == '__main__':
    unittest.main()
