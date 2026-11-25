import math
import unittest
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

from app import db
from app.analytics import build_analytics, parse_period
from app.models import Plan, Category
from app.recommendations import (create_draft, historical_counts, historical_recency_counts,
                                 draft_problem)
from app.recommender import (TaskInput, HistoryCount, WeightedHistoryCount, HistoryObservation,
                            recency_counts, weighted_history_rating, generate_plan)
from experiments.frozen_v1 import recommender as frozen
from tests import test_recommendations as fixture


NOW = datetime(2030, 1, 7, 6)
WINDOWS = [(datetime(2030, 1, 7, 7), datetime(2030, 1, 7, 16))]


class RecencyEngineTests(unittest.TestCase):
    def observation(self, days, success=True, part='morning', category=1):
        return HistoryObservation(category, part, NOW-timedelta(days=days), success)

    def plan(self, history, method='personalized_recent', **changes):
        inputs = dict(tasks=[TaskInput(1, 'Study', 1, 4, 60)], available_windows=WINDOWS,
                      busy_intervals=[], history=history, as_of=NOW, method=method)
        inputs.update(changes)
        return generate_plan(**inputs)

    def test_half_life_and_raw_counts_are_separate(self):
        records = [self.observation(14), self.observation(28, False)]
        count = recency_counts(records, NOW)[(1, 'morning')]
        self.assertEqual((count.successes, count.total), (1, 2))
        self.assertEqual((count.weighted_successes, count.weighted_total), (.5, .75))
        self.assertAlmostEqual(weighted_history_rating(.5, .75), 1.5/2.75)
        self.assertEqual(recency_counts(records, NOW), recency_counts(list(reversed(records)), NOW))

    def test_future_and_exact_cutoff_are_excluded(self):
        rows = [self.observation(-1), self.observation(0), self.observation(14, False)]
        count = recency_counts(rows, NOW)[(1, 'morning')]
        self.assertEqual(count, WeightedHistoryCount(0, 1, 0., .5))

    def test_time_decay_returns_toward_neutral_without_new_observations(self):
        rows = [self.observation(1)]
        recent = recency_counts(rows, NOW)[(1, 'morning')]
        old = recency_counts(rows, NOW+timedelta(days=1400))[(1, 'morning')]
        self.assertEqual(old.total, 1)
        self.assertLess(old.weighted_total, recent.weighted_total)
        self.assertEqual(weighted_history_rating(old.weighted_successes, old.weighted_total), .5)
        self.assertGreater(weighted_history_rating(recent.weighted_successes, recent.weighted_total), .5)

    def test_cold_start_matches_fuzzy_and_weighted_baselines(self):
        for full, baseline in [('personalized_recent', 'fuzzy'), ('weighted_recent', 'weighted')]:
            recent = self.plan({}, full)
            old = self.plan({}, baseline)
            positions = lambda result: [(a.task_id, a.start, a.end) for a in result.assignments]
            self.assertEqual(positions(recent), positions(old))
            self.assertEqual(recent.assignments[0].history_score, .5)
            self.assertEqual(recent.assignments[0].history_weighted_total, 0.)

    def test_recent_observations_can_reverse_old_preference(self):
        rows = ([self.observation(56, True, 'morning') for _ in range(8)]
                + [self.observation(56, False, 'afternoon') for _ in range(8)]
                + [self.observation(1, False, 'morning') for _ in range(2)]
                + [self.observation(1, True, 'afternoon') for _ in range(2)])
        history = recency_counts(rows, NOW)
        legacy = self.plan(history, 'personalized')
        recent = self.plan(history)
        self.assertEqual(legacy.assignments[0].daypart, 'morning')
        self.assertEqual(recent.assignments[0].daypart, 'afternoon')
        self.assertEqual(recent.assignments[0].history_total, 10)
        self.assertLess(recent.assignments[0].history_weighted_total, 3)
        self.assertEqual(recent.assignments[0].start, datetime(2030, 1, 7, 10))

    def test_recent_preference_does_not_override_deadline_or_busy_event(self):
        history = recency_counts([self.observation(1, False, 'morning'),
                                 self.observation(1, True, 'afternoon')], NOW)
        task = TaskInput(1, 'Study', 1, 4, 60, deadline=datetime(2030, 1, 7, 10))
        result = self.plan(history, tasks=[task],
                           busy_intervals=[(datetime(2030, 1, 7, 7), datetime(2030, 1, 7, 8))])
        self.assertEqual(result.assignments[0].start, datetime(2030, 1, 7, 8))
        self.assertLessEqual(result.assignments[0].end, task.deadline)

    def test_unweighted_nonempty_history_is_not_assumed_fresh(self):
        for mode in ('personalized_recent', 'weighted_recent'):
            with self.assertRaises(ValueError):
                self.plan({(1, 'morning'): HistoryCount(1, 2)}, mode)
            self.assertEqual(self.plan({(1, 'morning'): HistoryCount(0, 0)}, mode).assignments[0].history_score, .5)

    def test_invalid_observations_half_lives_and_weights_rejected(self):
        for half_life in (0, -1, True, math.nan, math.inf):
            with self.assertRaises(ValueError): recency_counts([], NOW, half_life)
        for row in (HistoryObservation(True, 'morning', NOW, True),
                    HistoryObservation(1, 'unknown', NOW, True),
                    HistoryObservation(1, 'morning', NOW, 1),
                    HistoryObservation(1, 'morning', NOW.replace(tzinfo=timezone.utc), True)):
            with self.assertRaises(ValueError): recency_counts([row], NOW)
        for count in (WeightedHistoryCount(1, 1, 2., 2.), WeightedHistoryCount(0, 1, .5, .5),
                      WeightedHistoryCount(1, 1, 0., .5), WeightedHistoryCount(1, 1, math.nan, 1.),
                      WeightedHistoryCount(1, 1, True, 1.)):
            with self.assertRaises(ValueError): self.plan({(1, 'morning'): count})

    def test_all_five_legacy_modes_match_frozen_engine(self):
        tasks = [TaskInput(i, f'Task {i}', 1+i%3, 1+i%5, [20, 30, 60, 90][i%4],
                           deadline=NOW+timedelta(hours=4+i%6)) for i in range(1, 19)]
        histories = [self.observation(1+i, i%3 != 0, part, category)
                     for i in range(20) for part in ('morning', 'afternoon') for category in (1, 2, 3)]
        history = recency_counts(histories, NOW)
        kwargs = dict(tasks=tasks, available_windows=WINDOWS,
                      busy_intervals=[(NOW+timedelta(hours=3), NOW+timedelta(hours=4))],
                      history=history, as_of=NOW)
        for mode in frozen.METHODS:
            current = generate_plan(**kwargs, method=mode)
            old = frozen.generate_plan(**kwargs, method=mode)
            self.assertEqual(current.candidate_count, old.candidate_count)
            self.assertEqual([asdict(r) for r in current.unscheduled], [asdict(r) for r in old.unscheduled])
            self.assertEqual([{k: v for k, v in asdict(a).items() if not k.startswith('history_weighted_')}
                              for a in current.assignments], [asdict(a) for a in old.assignments])


class RecencyIntegrationTests(unittest.TestCase):
    setUp = fixture.RecommendationIntegrationTests.setUp
    tearDown = fixture.RecommendationIntegrationTests.tearDown
    task = fixture.RecommendationIntegrationTests.task
    history_attempt = fixture.RecommendationIntegrationTests.history_attempt
    post = fixture.RecommendationIntegrationTests.post
    draft = fixture.RecommendationIntegrationTests.draft

    def test_adapter_uses_planned_end_and_exact_shared_eligibility(self):
        good = self.history_attempt(end=NOW-timedelta(days=14), success=True)
        good.actual_end = good.planned_end-timedelta(minutes=30)
        other_category = Category.query.filter_by(name='Робота').one()
        good.task.category_id = other_category.id
        db.session.commit()
        self.history_attempt(end=NOW-timedelta(days=28), success=False)
        self.history_attempt(end=NOW, success=True)
        self.history_attempt(end=NOW-timedelta(days=1), accepted=False)
        self.history_attempt(end=NOW-timedelta(days=1), owner=self.other_id)
        canceled_end = NOW-timedelta(days=2)
        self.history_attempt(end=canceled_end, canceled=canceled_end-timedelta(hours=2))
        recent, ignored = historical_recency_counts(self.user_id, NOW)
        raw, raw_ignored = historical_counts(self.user_id, NOW)
        self.assertEqual(ignored, raw_ignored)
        self.assertEqual({key: (v.successes, v.total) for key, v in recent.items()},
                         {key: (v.successes, v.total) for key, v in raw.items()})
        count = next(iter(recent.values()))
        self.assertEqual((count.successes, count.total), (1, 2))
        self.assertEqual((count.weighted_successes, count.weighted_total), (.5, .75))
        self.assertEqual(next(iter(recent))[0], self.category_id)

    def test_draft_freezes_weighted_explanation_and_simple_ui(self):
        self.history_attempt(end=NOW-timedelta(days=14))
        self.task()
        plan = create_draft(self.user, WINDOWS, now=NOW)
        db.session.commit()
        self.assertEqual(plan.parameters_json['method'], 'personalized_recent')
        self.assertEqual(plan.parameters_json['history_half_life_days'], 14.)
        self.assertEqual(plan.parameters_json['history_age_timestamp'], 'planned_end')
        self.assertEqual(plan.parameters_json['history_weighted_observations'], .5)
        for item in plan.items:
            stored = item.task_snapshot_json['recommendation']
            self.assertIn('history_weighted_total', stored)
            self.assertIsInstance(stored['history_weighted_total'], float)
        html = self.client.get(f'/recommendations/{plan.id}').text
        self.assertIn('Новіші записи мають більшу вагу.', html)
        self.assertIsNone(draft_problem(plan, self.user, NOW))
        plan.parameters_json = dict(plan.parameters_json, model_version='1.0')
        self.assertIn('оновлено', draft_problem(plan, self.user, NOW))

    def test_analytics_and_recommender_match_independently_of_date_filter(self):
        self.history_attempt(end=NOW-timedelta(days=28), success=True)
        self.history_attempt(end=NOW-timedelta(days=14), success=False)
        counts, _ = historical_recency_counts(self.user_id, NOW)
        one = build_analytics(self.user, *parse_period('2030-01-01', '2030-01-07', self.user.timezone, NOW), now=NOW)
        two = build_analytics(self.user, *parse_period('2029-11-01', '2030-01-07', self.user.timezone, NOW), now=NOW)
        self.assertEqual(one['history'], two['history'])
        self.assertEqual(one['history_total'], 2)
        self.assertEqual(one['planned_count'], 0)
        self.assertEqual(two['planned_count'], 2)
        row = one['history'][0]
        count = counts[(row['category_id'], row['daypart'])]
        self.assertEqual((row['weighted_successes'], row['weighted_total']), (.25, .75))
        self.assertEqual(row['score'], weighted_history_rating(count.weighted_successes, count.weighted_total))
        self.assertAlmostEqual(row['score'], 1.25/2.75)
        response = self.client.get('/analytics?start=2030-01-01&end=2030-01-07')
        self.assertEqual(response.status_code, 200)
        self.assertIn('Новіші записи мають більшу вагу.', response.text)


if __name__ == '__main__':
    unittest.main()
