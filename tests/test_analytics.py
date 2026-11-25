import smtplib
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from app import create_app, db
from app.analytics import build_analytics, parse_period
from app.database import initialize_database
from app.models import Category, Plan, PlanItem, Task, User


NOW = datetime(2030, 1, 15, 12)  # 14:00 Kyiv


class AnalyticsPeriodTests(unittest.TestCase):
    def test_default_uses_local_date_and_includes_thirty_days(self):
        first, last, start, end = parse_period('', '', 'Europe/Kyiv', datetime(2030, 1, 31, 23))
        self.assertEqual((first, last), (date(2030, 1, 3), date(2030, 2, 1)))
        self.assertEqual(start, datetime(2030, 1, 2, 22))
        self.assertEqual(end, datetime(2030, 2, 1, 22))

    def test_dst_days_have_real_elapsed_lengths(self):
        spring = parse_period('2026-03-29', '2026-03-29', 'Europe/Kyiv', NOW)
        autumn = parse_period('2026-10-25', '2026-10-25', 'Europe/Kyiv', NOW)
        self.assertEqual(spring[3]-spring[2], timedelta(hours=23))
        self.assertEqual(autumn[3]-autumn[2], timedelta(hours=25))

    def test_366_days_and_last_supported_date(self):
        first, last, _, _ = parse_period('2028-01-01', '2028-12-31', 'UTC', NOW)
        self.assertEqual((last-first).days+1, 366)
        result = parse_period('2100-12-31', '2100-12-31', 'UTC', NOW)
        self.assertEqual(result[3], datetime(2101, 1, 1))

    def test_invalid_ranges_and_timezones_are_rejected(self):
        for start, end, zone in [('', '2030-01-15', 'UTC'), ('2030-01-15', '', 'UTC'),
                                 ('bad', '2030-01-15', 'UTC'), ('2030-02-30', '2030-03-01', 'UTC'),
                                 ('2030-01-16', '2030-01-15', 'UTC'),
                                 ('2028-01-01', '2029-01-01', 'UTC'),
                                 ('1899-12-31', '1900-01-01', 'UTC'),
                                 ('2101-01-01', '2101-01-01', 'UTC'),
                                 ('2030-01-01', '2030-01-15', 'Missing/Zone')]:
            with self.subTest(start=start, end=end, zone=zone):
                with self.assertRaises(ValueError):
                    parse_period(start, end, zone, NOW)
        with self.assertRaises(ValueError):
            parse_period('', '', 'UTC', NOW.replace(tzinfo=timezone.utc))

    def test_missing_or_ambiguous_midnight_is_not_silently_shifted(self):
        # Samoa skipped this entire civil date; Havana repeated midnight.
        for day, zone in [('2011-12-30', 'Pacific/Apia'), ('2020-11-01', 'America/Havana')]:
            with self.subTest(day=day, zone=zone):
                with self.assertRaises(ValueError):
                    parse_period(day, day, zone, NOW)


class AnalyticsDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.app = create_app({'TESTING': True, 'SECRET_KEY': 'analytics-test',
                              'SQLALCHEMY_DATABASE_URI': 'sqlite:///'+str(Path(self.temp.name)/'test.db')})
        self.ctx = self.app.app_context()
        self.ctx.push()
        initialize_database()
        self.user = User(username='Reader', email='reader@example.com', password_hash='unused')
        self.other = User(username='Other', email='other@example.com', password_hash='unused')
        db.session.add_all([self.user, self.other])
        db.session.commit()
        self.category = Category.query.filter_by(name='Навчання').one()
        self.patches = [patch.object(smtplib, 'SMTP', side_effect=AssertionError('SMTP forbidden')),
                        patch.object(smtplib, 'SMTP_SSL', side_effect=AssertionError('SMTP_SSL forbidden'))]
        for item in self.patches:
            item.start()

    def tearDown(self):
        db.session.remove()
        db.engine.dispose()
        self.ctx.pop()
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def attempt(self, start=None, *, owner=None, task=None, duration=60,
                actual_start=None, actual_end=None, canceled_at=None, accepted=True,
                accepted_at=None, created_at=None, snapshot=None):
        owner = owner or self.user
        start = start or datetime(2030, 1, 14, 8)
        end = start+timedelta(minutes=duration)
        if task is None:
            task = Task(user_id=owner.id, category_id=self.category.id, title='Study',
                        importance=4, duration_min=duration, status='pending')
            db.session.add(task)
        plan = Plan(user_id=owner.id, created_at=created_at or start-timedelta(days=1),
                    accepted_at=(accepted_at or start-timedelta(hours=1)) if accepted else None,
                    horizon_start=start, horizon_end=end,
                    status='canceled' if canceled_at else 'accepted' if accepted else 'draft',
                    source_version=1, parameters_json={})
        item = PlanItem(plan=plan, task=task, planned_start=start, planned_end=end,
                        actual_start=actual_start, actual_end=actual_end, canceled_at=canceled_at,
                        status='canceled' if canceled_at else 'completed' if actual_end else 'scheduled',
                        task_snapshot_json=snapshot if snapshot is not None else {
                            'category_id': self.category.id, 'category': self.category.name,
                            'timezone': 'Europe/Kyiv', 'title': 'Original title'})
        db.session.add(plan)
        db.session.commit()
        return item

    def report(self, first='2030-01-01', last='2030-01-31', now=NOW):
        return build_analytics(self.user, *parse_period(first, last, self.user.timezone, now), now=now)

    def test_empty_report_has_zero_counts_and_no_invented_percentage(self):
        report = self.report()
        self.assertEqual(report['planned_count'], 0)
        self.assertEqual(report['planned_minutes'], 0)
        self.assertIsNone(report['on_time_percent'])
        self.assertEqual(report['history'], [])
        self.assertEqual(len(report['daily']), 31)

    def test_known_completion_breakdown_and_elapsed_recorded_time(self):
        self.attempt(actual_start=datetime(2030, 1, 14, 8, 5), actual_end=datetime(2030, 1, 14, 9))
        self.attempt(datetime(2030, 1, 14, 9), actual_start=datetime(2030, 1, 14, 9),
                     actual_end=datetime(2030, 1, 14, 10, 20))
        self.attempt(datetime(2030, 1, 14, 10))
        report = self.report()
        self.assertEqual((report['planned_count'], report['completed_count'], report['due_count']), (3, 2, 3))
        self.assertEqual((report['on_time_count'], report['late_count'], report['unrecorded_count']), (1, 1, 1))
        self.assertAlmostEqual(report['on_time_percent'], 100/3)
        self.assertEqual((report['planned_minutes'], report['recorded_minutes'], report['measured_count']), (180, 135, 2))
        row = report['daily'][13]
        self.assertEqual((row['date'], row['planned'], row['completed'], row['due'], row['on_time']),
                         (date(2030, 1, 14), 3, 2, 3, 1))

    def test_owner_drafts_and_later_acceptance_or_creation_are_excluded(self):
        self.attempt(owner=self.other, actual_end=datetime(2030, 1, 14, 9))
        self.attempt(accepted=False, actual_end=datetime(2030, 1, 14, 9))
        self.attempt(accepted_at=NOW+timedelta(hours=1))
        self.attempt(created_at=NOW+timedelta(hours=1))
        report = self.report()
        self.assertEqual(report['planned_count'], 0)
        self.assertEqual(report['history_total'], 0)

    def test_early_cancellation_excluded_late_cancellation_remains_an_attempt(self):
        start = datetime(2030, 1, 14, 8)
        self.attempt(start, canceled_at=start-timedelta(seconds=1))
        self.attempt(start, canceled_at=start)
        self.attempt(start, canceled_at=start+timedelta(minutes=30))
        report = self.report()
        self.assertEqual(report['excluded_early_count'], 1)
        self.assertEqual((report['planned_count'], report['unrecorded_count'], report['history_total']), (2, 2, 2))

    def test_future_and_exact_cutoff_do_not_enter_due_denominator(self):
        self.attempt(NOW-timedelta(hours=1), actual_end=NOW)  # planned_end == now
        self.attempt(NOW+timedelta(days=1), accepted_at=NOW-timedelta(hours=1), created_at=NOW-timedelta(days=1))
        report = self.report()
        self.assertEqual((report['planned_count'], report['completed_count'], report['due_count']), (2, 1, 0))
        self.assertIsNone(report['on_time_percent'])
        self.assertEqual(report['history_total'], 0)

    def test_future_completion_is_not_leaked_into_statistics(self):
        self.attempt(actual_start=datetime(2030, 1, 14, 8), actual_end=NOW+timedelta(hours=1))
        report = self.report()
        self.assertEqual((report['completed_count'], report['unrecorded_count'], report['recorded_minutes']), (0, 1, 0))

    def test_local_start_boundaries_and_whole_assignment_duration(self):
        start = datetime(2030, 1, 13, 22)  # 00:00 14 January Kyiv
        end = datetime(2030, 1, 14, 22)    # 00:00 15 January Kyiv
        self.attempt(start-timedelta(minutes=1), duration=120)
        self.attempt(start, duration=45)
        self.attempt(end-timedelta(minutes=30), duration=120)
        self.attempt(end, duration=60)
        report = self.report('2030-01-14', '2030-01-14')
        self.assertEqual(report['planned_count'], 2)
        self.assertEqual(report['planned_minutes'], 165)
        self.assertEqual(report['daily'][0]['planned'], 2)

    def test_repeated_attempts_and_archived_or_reopened_task_keep_history(self):
        first = self.attempt(actual_end=datetime(2030, 1, 14, 9))
        task = first.task
        task.archived_at = NOW
        task.status = 'pending'
        self.attempt(datetime(2030, 1, 13, 8), task=task)
        report = self.report()
        self.assertEqual((report['planned_count'], report['completed_count'], report['history_total']), (2, 1, 2))

    def test_history_uses_all_past_dates_snapshot_category_and_recency_smoothing(self):
        old = self.attempt(datetime(2029, 12, 10, 8), actual_end=datetime(2029, 12, 10, 9))
        other_category = Category.query.filter_by(name='Робота').one()
        old.task.category_id = other_category.id
        db.session.commit()
        self.attempt(datetime(2029, 12, 11, 8))
        report = self.report()
        self.assertEqual(report['planned_count'], 0)
        self.assertEqual(report['history_total'], 2)
        row = report['history'][0]
        self.assertEqual((row['category_id'], row['daypart'], row['successes'], row['total']),
                         (self.category.id, 'morning', 1, 2))
        success_weight = 2**(-((NOW-datetime(2029, 12, 10, 9)).total_seconds()/86400)/14)
        failure_weight = 2**(-((NOW-datetime(2029, 12, 11, 9)).total_seconds()/86400)/14)
        self.assertAlmostEqual(row['score'], (success_weight+1)/(success_weight+failure_weight+2))
        self.assertLess(row['score'], .5)

    def test_history_keeps_snapshot_timezone_after_user_timezone_change(self):
        self.attempt(datetime(2030, 1, 14, 10), actual_end=datetime(2030, 1, 14, 11))
        self.user.timezone = 'America/New_York'
        db.session.commit()
        report = self.report()
        self.assertEqual(report['history'][0]['daypart'], 'afternoon')  # 12:00 in recorded Kyiv zone

    def test_missing_category_history_is_reported_as_ignored(self):
        self.attempt(snapshot={'title': 'Legacy without category'})
        report = self.report()
        self.assertEqual(report['planned_count'], 1)
        self.assertEqual((report['history_total'], report['ignored_history']), (0, 1))

    def test_missing_actual_start_never_becomes_measured_time(self):
        self.attempt(actual_end=datetime(2030, 1, 14, 9))
        report = self.report()
        self.assertEqual(report['completed_count'], 1)
        self.assertEqual((report['recorded_minutes'], report['measured_count']), (0, 0))

    def test_mismatched_utc_boundary_rejected(self):
        first, last, start, end = parse_period('2030-01-14', '2030-01-14', self.user.timezone, NOW)
        with self.assertRaises(ValueError):
            build_analytics(self.user, first, last, start+timedelta(hours=1), end, NOW)

    def client(self):
        client = self.app.test_client()
        with client.session_transaction() as session:
            session['_user_id'] = str(self.user.id)
            session['_fresh'] = True
        return client

    def test_http_requires_login_and_does_not_accept_writes(self):
        self.assertEqual(self.app.test_client().get('/analytics').status_code, 302)
        self.assertEqual(self.client().post('/analytics').status_code, 405)

    def test_http_empty_state_and_navigation_render_without_fake_rate(self):
        with patch('app.routes.utcnow', return_value=NOW):
            response = self.client().get('/analytics')
        self.assertEqual(response.status_code, 200)
        self.assertIn('немає прийнятих призначень', response.text)
        self.assertIn('Історії для рекомендацій ще немає', response.text)
        self.assertIn('Аналітика', response.text)
        self.assertNotIn('0,0<small>%', response.text)

    def test_http_invalid_period_has_clear_error_and_safe_default(self):
        client = self.client()
        for query in ('start=bad&end=2030-01-15', 'start=2030-01-14',
                      'start=2030-01-16&end=2030-01-15',
                      'start=2028-01-01&end=2030-01-01'):
            with self.subTest(query=query), patch('app.routes.utcnow', return_value=NOW):
                response = client.get('/analytics?'+query)
                self.assertEqual(response.status_code, 400)
                self.assertIn('alert danger', response.text)
                self.assertIn('value="2030-01-15"', response.text)

    def test_http_known_data_is_read_only_and_user_scoped(self):
        self.attempt(actual_start=datetime(2030, 1, 14, 8), actual_end=datetime(2030, 1, 14, 9))
        private_category = Category(name='PRIVATE OTHER USER CATEGORY')
        db.session.add(private_category)
        db.session.flush()
        self.attempt(owner=self.other, snapshot={'category_id': private_category.id, 'timezone': 'Europe/Kyiv'})
        before = [(item.id, item.status, item.actual_end) for item in PlanItem.query.order_by(PlanItem.id)]
        version = self.user.data_version
        with patch('app.routes.utcnow', return_value=NOW):
            response = self.client().get('/analytics?start=2030-01-14&end=2030-01-14')
        self.assertEqual(response.status_code, 200)
        self.assertIn('100,0<small>%', response.text)
        weight = 2**(-1.125/14)
        self.assertIn(f'{(weight+1)/(weight+2):.3f}'.replace('.', ','), response.text)
        self.assertIn('Новіші записи мають більшу вагу.', response.text)
        self.assertIn('14.01.2030: заплановано 1, завершено 1', response.text)
        self.assertNotIn('PRIVATE OTHER USER CATEGORY', response.text)
        self.assertEqual(before, [(item.id, item.status, item.actual_end) for item in PlanItem.query.order_by(PlanItem.id)])
        self.assertEqual(version, self.user.data_version)

    def test_http_period_does_not_relabel_history_as_period_data(self):
        self.attempt(datetime(2029, 12, 10, 8), actual_end=datetime(2029, 12, 10, 9))
        with patch('app.routes.utcnow', return_value=NOW):
            response = self.client().get('/analytics?start=2030-01-14&end=2030-01-14')
        self.assertEqual(response.status_code, 200)
        self.assertIn('немає прийнятих призначень', response.text)
        self.assertIn('Усі попередні спостереження · 1 спроб', response.text)
        self.assertIn('Не обмежується фільтром дат', response.text)
        weight = 2**(-((NOW-datetime(2029, 12, 10, 9)).total_seconds()/86400)/14)
        self.assertIn(f'{(weight+1)/(weight+2):.3f}'.replace('.', ','), response.text)


if __name__ == '__main__':
    unittest.main()
