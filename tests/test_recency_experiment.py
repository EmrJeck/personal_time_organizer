import copy
import csv
import hashlib
import json
import math
import smtplib
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from app.recommender import generate_plan
from experiments.run_recency_comparison import (
    CHANGE_AGES, CONDITIONS, HALF_LIFE_DAYS, HISTORY_SIZES, LEGACY_METHODS,
    LOADS, METHODS, ROOT, check_legacy_equivalence, configuration, duration_model,
    evaluate_plan, generate_case, planner_inputs, run_experiment,
    simulate_duration, validate_plan,
)


class RecencyExperimentTests(unittest.TestCase):
    def setUp(self):
        self.case = generate_case('observed_drift_14', 0.7, 20, 0)
        self.smtp = [patch.object(smtplib, 'SMTP', side_effect=AssertionError('SMTP forbidden')),
                     patch.object(smtplib, 'SMTP_SSL', side_effect=AssertionError('SMTP_SSL forbidden'))]
        for item in self.smtp:
            item.start()

    def tearDown(self):
        for item in reversed(self.smtp):
            item.stop()

    def test_configuration_freezes_six_conditions_seven_methods_and_half_life(self):
        config = configuration()
        self.assertEqual(config['scenario_count'], 1440)
        self.assertEqual(config['repeats_per_cell'], 40)
        self.assertEqual(config['master_seed'], 20261106)
        self.assertEqual(config['recency']['half_life_days'], 14)
        self.assertEqual(len(config['conditions']), 6)
        self.assertEqual(len(config['methods']), 7)
        self.assertEqual(config['fixed_weights']['history_weight'], .2)
        self.assertFalse(config['simulator']['planner_has_access_to_hidden_simulation_fields'])

    def test_deterministic_generator_has_independent_cell_seeds(self):
        self.assertEqual(self.case, generate_case('observed_drift_14', 0.7, 20, 0))
        cases = [generate_case(c, load, n, 0) for c in CONDITIONS for load in LOADS for n in HISTORY_SIZES]
        self.assertEqual(len({case['seed'] for case in cases}), 36)
        self.assertEqual(len({json.dumps(case['planner']['tasks']) for case in cases}), 36)
        for case in cases:
            self.assertGreaterEqual(case['actual_load'], case['load_target'])
            self.assertLess(case['total_allocated_minutes']-case['load_target']*960, 120)

    def test_historical_labels_follow_actual_change_date_and_stay_in_past(self):
        for condition in CONDITIONS:
            case = generate_case(condition, .7, 20, 0)
            as_of = datetime.fromisoformat(case['planner']['as_of'])
            hidden = case['simulation']
            change_at = datetime.fromisoformat(hidden['change_at']) if hidden['change_at'] else None
            if condition in CHANGE_AGES:
                self.assertEqual(as_of-change_at, timedelta(days=CHANGE_AGES[condition]))
            else:
                self.assertIsNone(change_at)
            labels = []
            for record in case['history_records']:
                start = datetime.fromisoformat(record['start'])
                end = datetime.fromisoformat(record['planned_end'])
                allocated = 15*math.ceil(record['estimated_minutes']/15)
                self.assertEqual(end-start, timedelta(minutes=allocated))
                self.assertLess(end, as_of)
                expected_new = change_at is not None and start >= change_at
                labels.append(expected_new)
                self.assertEqual(record['post_change'], expected_new)
                expected_peak = hidden['future_peaks' if expected_new else 'historical_peaks'][str(record['category_id'])]
                self.assertEqual(record['peak_used'], expected_peak)
                actual = duration_model(record['estimated_minutes'], record['hour_local'], expected_peak,
                                        hidden['amplitude'], record['difficulty'], record['noise'])
                self.assertEqual(record['actual_minutes'], actual)
                self.assertEqual(record['success'], actual <= allocated)
            self.assertEqual(case['post_change_history_count'], sum(labels))
            if condition.startswith('observed_drift'):
                self.assertIn(True, labels)
                self.assertIn(False, labels)
            else:
                self.assertNotIn(True, labels)

    def test_shared_raw_and_weighted_counts_match_only_observed_records(self):
        cutoff = datetime.fromisoformat(self.case['planner']['as_of'])
        records = self.case['planner']['history_observations']
        self.assertEqual(len(records), 240)
        for row in self.case['planner']['history']:
            selected = [r for r in records if (r['category_id'], r['daypart']) == (row['category_id'], row['daypart'])]
            weights = [2**(-(cutoff-datetime.fromisoformat(r['occurred_at'])).total_seconds()/86400/HALF_LIFE_DAYS)
                       for r in selected]
            self.assertEqual(row['total'], 20)
            self.assertEqual(row['successes'], sum(r['success'] for r in selected))
            self.assertAlmostEqual(row['weighted_total'], math.fsum(weights))
            self.assertAlmostEqual(row['weighted_successes'], math.fsum(w for r, w in zip(selected, weights) if r['success']))
            self.assertLess(row['weighted_total'], row['total'])
            self.assertGreater(row['weighted_total'], 0)
        raw = self.case['history_records']
        for record, public in zip(raw, records):
            self.assertEqual(record['planned_end'], public['occurred_at'])
            self.assertEqual(record['success'], public['success'])

    def test_planner_has_no_access_to_latent_change_or_future_noise(self):
        modified = copy.deepcopy(self.case)
        modified['simulation'] = {'forbidden_future_information': object()}
        modified['history_records'] = []
        modified['post_change_history_count'] = -999
        self.assertEqual(planner_inputs(self.case), planner_inputs(modified))
        self.assertEqual(set(planner_inputs(self.case)),
                         {'tasks', 'available_windows', 'busy_intervals', 'history', 'as_of', 'timezone_name'})

    def test_future_simulation_is_independent_of_scores_and_history(self):
        task = self.case['planner']['tasks'][0]
        start = planner_inputs(self.case)['available_windows'][0][0]
        expected = simulate_duration(self.case, task, start)
        modified = copy.deepcopy(self.case)
        modified['planner']['history'] = []
        modified['planner']['history_observations'] = []
        modified['history_records'] = []
        modified['R'] = float('inf')
        modified['H'] = -999
        self.assertEqual(simulate_duration(modified, dict(task, importance=1), start), expected)

    def test_negative_control_has_no_category_or_time_effect(self):
        case = generate_case('no_time_effect', .7, 20, 0)
        task = case['planner']['tasks'][0]
        start = planner_inputs(case)['available_windows'][0][0]
        expected = simulate_duration(case, task, start)
        for category in (1, 2, 3):
            for offset in (0, 4, 24):
                self.assertEqual(simulate_duration(case, dict(task, category_id=category),
                                                   start+timedelta(hours=offset)), expected)

    def test_all_baselines_match_frozen_engine_and_cold_start_is_neutral(self):
        positions = lambda result: [(a.task_id, a.start, a.end) for a in result.assignments]
        for condition in CONDITIONS:
            for n in HISTORY_SIZES:
                case = generate_case(condition, 1.4, n, 0)
                inputs = planner_inputs(case)
                results = {method: generate_plan(**inputs, method=method) for method in METHODS}
                check_legacy_equivalence(inputs, results)
                for result in results.values():
                    self.assertEqual(validate_plan(case, result)['total_violations'], 0)
                    metrics = evaluate_plan(case, result)
                    self.assertLessEqual(metrics['simulated_on_time_importance_fraction'],
                                         metrics['scheduled_importance_fraction'])
                if n == 0:
                    for method in ('personalized', 'personalized_recent'):
                        self.assertEqual(positions(results[method]), positions(results['fuzzy']))
                    self.assertEqual(positions(results['weighted_recent']), positions(results['weighted']))

    def test_smoke_export_preserves_old_evidence_pairs_and_hashes(self):
        old_files = [ROOT/'experiments/run_comparison.py', ROOT/'experiments/results_v1/manifest.json',
                     ROOT/'experiments/results_v1/case_metrics.csv']
        before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in old_files}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/'smoke'
            with patch('app.create_app', side_effect=AssertionError('Database/application access forbidden')):
                manifest = run_experiment(output, repeats=1, timing_repeats=2, progress=False)
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(manifest['scenario_count'], 36)
            self.assertEqual(manifest['metric_rows'], 252)
            self.assertEqual(manifest['measured_planner_calls'], 504)
            self.assertEqual(manifest['frozen_baseline_equivalence_checks'], 36*len(LEGACY_METHODS))
            self.assertEqual(manifest['total_violations'], 0)
            self.assertFalse(manifest['database_access'])
            with (output/'case_metrics.csv').open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            for scenario in {row['scenario_id'] for row in rows}:
                group = [row for row in rows if row['scenario_id'] == scenario]
                self.assertEqual({row['method'] for row in group}, set(METHODS))
                self.assertEqual(len({row['planner_inputs_sha256'] for row in group}), 1)
            for name, expected in manifest['outputs_sha256'].items():
                self.assertEqual(hashlib.sha256((output/name).read_bytes()).hexdigest(), expected)
            for name, expected in manifest['sources_sha256'].items():
                self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), expected)
            with self.assertRaises(FileExistsError):
                run_experiment(output, repeats=1, timing_repeats=1, progress=False)
        self.assertEqual(before, {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in old_files})


if __name__ == '__main__':
    unittest.main()
