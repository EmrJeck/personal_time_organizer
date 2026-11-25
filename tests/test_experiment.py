import copy
import csv
import hashlib
import json
import smtplib
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.recommender import generate_plan
from experiments.run_comparison import (
    CONDITIONS, HISTORY_SIZES, LOADS, METHODS, evaluate_plan, generate_case,
    individually_feasible_ids, planner_inputs, run_experiment, simulate_duration,
    validate_plan,
)


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.case = generate_case('stable', 0.7, 20, 0)
        self.patches = [patch.object(smtplib, 'SMTP', side_effect=AssertionError('SMTP forbidden')),
                        patch.object(smtplib, 'SMTP_SSL', side_effect=AssertionError('SMTP_SSL forbidden'))]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()

    def test_generator_is_reproducible_and_cells_have_distinct_seeds(self):
        self.assertEqual(self.case, generate_case('stable', 0.7, 20, 0))
        cases = [generate_case(condition, load, n, 0)
                 for condition in CONDITIONS for load in LOADS for n in HISTORY_SIZES]
        self.assertEqual(len({case['seed'] for case in cases}), 18)
        self.assertEqual(len({json.dumps(case['planner']['tasks']) for case in cases}), 18)
        for case in cases:
            self.assertGreaterEqual(case['actual_load'], case['load_target'])
            self.assertLess(case['total_allocated_minutes']-case['load_target']*960, 120)

    def test_history_has_exact_counts_and_only_past_observations(self):
        inputs = planner_inputs(self.case)
        self.assertEqual(len(self.case['history_records']), 240)
        self.assertEqual(sum(row['total'] for row in self.case['planner']['history']), 240)
        for group in self.case['planner']['history']:
            records = [row for row in self.case['history_records']
                       if (row['category_id'], row['daypart']) == (group['category_id'], group['daypart'])]
            self.assertEqual(len(records), 20)
            self.assertEqual(sum(row['success'] for row in records), group['successes'])
            for row in records:
                start = datetime.fromisoformat(row['start'])
                planned_minutes = 15*((row['estimated_minutes']+14)//15)
                self.assertLess(start+timedelta(minutes=planned_minutes), inputs['as_of'])

    def test_hidden_world_and_future_noise_are_not_planner_inputs(self):
        changed = copy.deepcopy(self.case)
        changed['simulation'] = {'forbidden_hidden_information': object()}
        changed['history_records'] = []
        self.assertEqual(planner_inputs(self.case), planner_inputs(changed))
        self.assertEqual(set(planner_inputs(self.case)),
                         {'tasks', 'available_windows', 'busy_intervals', 'history', 'as_of', 'timezone_name'})

    def test_simulator_does_not_read_priority_or_history_scores(self):
        task = self.case['planner']['tasks'][0]
        start = planner_inputs(self.case)['available_windows'][0][0]
        expected = simulate_duration(self.case, task, start)
        changed = copy.deepcopy(self.case)
        changed['planner']['history'] = []
        changed['R'] = float('inf')
        changed['H'] = -999
        task = dict(task, importance=1, priority_score=999, history_score=-999)
        self.assertEqual(simulate_duration(changed, task, start), expected)

    def test_no_effect_control_is_invariant_to_start_and_category(self):
        case = generate_case('no_time_effect', 0.7, 20, 0)
        task = case['planner']['tasks'][0]
        start = planner_inputs(case)['available_windows'][0][0]
        reference = simulate_duration(case, task, start)
        for category in (1, 2, 3):
            for offset in (0, 4, 8, 24):
                changed_task = dict(task, category_id=category)
                self.assertEqual(simulate_duration(case, changed_task, start+timedelta(hours=offset)), reference)

    def test_drift_changes_future_peaks_only_with_eight_hour_offset(self):
        case = generate_case('drift', 0.7, 1, 0)
        hidden = case['simulation']
        for category, peak in hidden['historical_peaks'].items():
            self.assertAlmostEqual((hidden['future_peaks'][category]-peak) % 24, 8)
        stable = self.case['simulation']
        self.assertEqual(stable['historical_peaks'], stable['future_peaks'])

    def test_all_methods_respect_constraints_and_cold_start_matches_fuzzy(self):
        case = generate_case('stable', 1.4, 0, 1)
        inputs = planner_inputs(case)
        results = {method: generate_plan(**inputs, method=method) for method in METHODS}
        for result in results.values():
            self.assertEqual(validate_plan(case, result)['total_violations'], 0)
        positions = lambda result: [(row.task_id, row.start, row.end) for row in result.assignments]
        self.assertEqual(positions(results['personalized']), positions(results['fuzzy']))
        self.assertEqual(case['independently_feasible_count'], len(individually_feasible_ids(case)))

    def test_weighted_simulation_denominator_includes_unplaced_tasks(self):
        case = copy.deepcopy(self.case)
        start = planner_inputs(case)['available_windows'][0][0]
        case['planner']['tasks'] = [
            dict(id=1, title='Placed', category_id=1, importance=1, duration_min=20,
                 earliest_start=None, deadline=None),
            dict(id=2, title='Unplaced', category_id=1, importance=5, duration_min=20,
                 earliest_start=None, deadline=None),
        ]
        case['simulation']['amplitude'] = 0
        case['simulation']['future_task_noise']['1'] = {'difficulty': 0, 'noise': 0}
        result = SimpleNamespace(assignments=[SimpleNamespace(task_id=1, start=start, end=start+timedelta(minutes=30))],
                                 unscheduled=[SimpleNamespace(task_id=2)], candidate_count=1)
        metrics = evaluate_plan(case, result)
        self.assertAlmostEqual(metrics['simulated_on_time_importance_fraction'], 100/6)
        self.assertEqual(metrics['fit_fraction_of_placed'], 100)
        self.assertEqual(metrics['placed_fraction'], 50)
        self.assertEqual(metrics['total_violations'], 0)

    def test_independent_validator_detects_bad_intervals_and_duplicates(self):
        inputs = planner_inputs(self.case)
        result = generate_plan(**inputs, method='edf')
        self.assertTrue(result.assignments)
        changed = copy.deepcopy(result)
        changed.assignments.append(changed.assignments[0])
        violations = validate_plan(self.case, changed)
        self.assertEqual(violations['duplicate_violations'], 1)
        self.assertGreater(violations['conflict_violations'], 0)

    def test_smoke_exports_paired_inputs_and_checksums_without_database(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'result'
            with patch('app.create_app', side_effect=AssertionError('Database/application access forbidden')):
                manifest = run_experiment(output, repeats=1, timing_repeats=2, progress=False)
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(manifest['scenario_count'], 18)
            self.assertEqual(manifest['metric_rows'], 90)
            self.assertEqual(manifest['measured_planner_calls'], 180)
            self.assertEqual(manifest['total_violations'], 0)
            config = json.loads((output/'config.json').read_text(encoding='utf-8'))
            self.assertEqual(config['simulator']['amplitude_no_time_effect'], 0)
            self.assertFalse(config['simulator']['planner_has_access_to_hidden_simulation_fields'])
            with (output/'case_metrics.csv').open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            for scenario in {row['scenario_id'] for row in rows}:
                group = [row for row in rows if row['scenario_id'] == scenario]
                self.assertEqual({row['method'] for row in group}, set(METHODS))
                self.assertEqual(len({row['planner_inputs_sha256'] for row in group}), 1)
                self.assertTrue(all(float(row['elapsed_ms']) >= 0 for row in group))
            for filename, expected in manifest['outputs_sha256'].items():
                self.assertEqual(hashlib.sha256((output/filename).read_bytes()).hexdigest(), expected)
            with self.assertRaises(FileExistsError):
                run_experiment(output, repeats=1, timing_repeats=1, progress=False)


if __name__ == '__main__':
    unittest.main()
