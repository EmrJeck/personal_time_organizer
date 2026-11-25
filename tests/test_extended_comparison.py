"""Integrity guards for extending archived experiments without changing evidence."""
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from app.recommender import generate_plan
from experiments import run_extended_comparison as extension
from experiments import run_recency_comparison as v2


class ExtendedComparisonTests(unittest.TestCase):
    def test_zero_edf_denominator_is_not_invented(self):
        self.assertIsNone(extension.relative_value(0, 0))
        self.assertIsNone(extension.relative_value(5, 0))
        self.assertEqual(extension.relative_value(12, 10), 1.2)

    def test_archive_signature_detects_changed_score_and_rejection(self):
        case = v2.generate_case('stable', .7, 0, 0)
        result = generate_plan(**v2.planner_inputs(case), method='fuzzy')
        archived = {'scenario_id': case['scenario_id'], 'method': 'fuzzy',
                    'assignments': extension.serialized_assignments(result),
                    'unscheduled': [asdict(item) for item in result.unscheduled]}
        metrics = {'candidate_count': result.candidate_count}
        extension.compare_archived_result(result, archived, metrics)
        archived['assignments'][0]['score'] += .001
        with self.assertRaises(AssertionError):
            extension.compare_archived_result(result, archived, metrics)
        archived['assignments'] = extension.serialized_assignments(result)
        archived['unscheduled'].append({'task_id': 9999, 'reason': 'changed'})
        with self.assertRaises(AssertionError):
            extension.compare_archived_result(result, archived, metrics)

    def test_archive_corruption_stops_before_planning(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory/'inputs.jsonl').write_text('changed', encoding='utf-8')
            manifest = {'status': 'complete', 'outputs_sha256': {'inputs.jsonl': 'wrong'},
                        'config_sha256': 'unused'}
            (directory/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                extension.audit_archive(directory)

    def test_reject_invalid_timing_repeat_count(self):
        for invalid in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                extension.run_experiment('unused', 'unused', invalid)


if __name__ == '__main__':
    unittest.main()
