"""Behavioral checks for the pure-importance comparison baseline."""
import unittest
from datetime import datetime, timedelta

from app.recommender import HistoryCount, TaskInput, generate_plan


START = datetime(2030, 1, 7, 7)  # 09:00 Kyiv.


class ImportanceBaselineTests(unittest.TestCase):
    def task(self, task_id, importance=3, minutes=60, **changes):
        return TaskInput(task_id, f'Task {task_id}', 1, importance, minutes, **changes)

    def plan(self, tasks, hours=1, **changes):
        options = dict(tasks=tasks, available_windows=[(START, START+timedelta(hours=hours))],
                       busy_intervals=[], history={}, as_of=START, method='importance')
        options.update(changes)
        return generate_plan(**options)

    @staticmethod
    def positions(result):
        return [(item.task_id, item.start, item.end) for item in result.assignments]

    def test_importance_wins_over_urgency_when_only_one_task_fits(self):
        tasks = [self.task(1, 5), self.task(2, 1, deadline=START+timedelta(hours=1))]
        result = self.plan(tasks)
        self.assertEqual([item.task_id for item in result.assignments], [1])
        self.assertEqual(result.assignments[0].score, 1.)
        self.assertEqual([item.task_id for item in self.plan(tasks, method='edf').assignments], [2])

    def test_history_and_history_weight_do_not_change_importance_plan(self):
        tasks = [self.task(1, 4), self.task(2, 3)]
        history = {(1, 'morning'): HistoryCount(0, 100),
                   (1, 'afternoon'): HistoryCount(100, 100)}
        reference = self.plan(tasks, hours=9)
        changed = self.plan(tasks, hours=9, history=history, history_weight=1.)
        self.assertEqual(self.positions(reference), self.positions(changed))
        self.assertEqual([item.score for item in changed.assignments], [.75, .5])
        self.assertTrue(all(item.score == item.importance_normalized for item in changed.assignments))

    def test_high_importance_cannot_override_busy_time_release_or_deadline(self):
        tasks = [self.task(1, 5, earliest_start=START+timedelta(minutes=30),
                           deadline=START+timedelta(hours=1)),
                 self.task(2, 2, 30, earliest_start=START+timedelta(minutes=15),
                           deadline=START+timedelta(hours=1))]
        result = self.plan(tasks, busy_intervals=[(START, START+timedelta(minutes=30))])
        self.assertEqual(self.positions(result),
                         [(2, START+timedelta(minutes=30), START+timedelta(hours=1))])
        self.assertEqual([item.task_id for item in result.unscheduled], [1])

    def test_equal_importance_uses_deadline_then_start_then_id(self):
        tasks = [self.task(3, minutes=30),
                 self.task(2, minutes=30, deadline=START+timedelta(hours=1)),
                 self.task(1, minutes=30, deadline=START+timedelta(hours=1))]
        result = self.plan(tasks, hours=2)
        self.assertEqual([item.task_id for item in result.assignments], [1, 2, 3])
        self.assertEqual(self.positions(result), self.positions(self.plan(list(reversed(tasks)), hours=2)))

    def test_minimum_importance_is_still_scheduled_when_feasible(self):
        result = self.plan([self.task(1, 1)])
        self.assertEqual([item.task_id for item in result.assignments], [1])
        self.assertEqual(result.assignments[0].score, 0.)


if __name__ == '__main__':
    unittest.main()
