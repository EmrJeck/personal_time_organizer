import re
import smtplib
import sqlite3
import tempfile
import unittest
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from werkzeug.security import generate_password_hash, check_password_hash
from app import create_app, db
from app.database import initialize_database
from app.models import User, Category, Task, CalendarEvent, Plan, PlanItem, Reminder
from app.planning import schedule_task
from app.time_utils import local_to_utc
from app.utils import check_and_send_reminders, send_reminder_email

NOW = datetime(2030, 1, 7, 6)  # Monday, 08:00 Kyiv.
START = '2030-01-07T10:00'

class OrganizerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.app = create_app({'TESTING': True, 'SECRET_KEY': 'test-only',
            'SQLALCHEMY_DATABASE_URI': 'sqlite:///' + str(Path(self.temp.name)/'test.db')})
        self.ctx = self.app.app_context(); self.ctx.push()
        initialize_database()
        self.user = User(username='tester', email='tester@example.com', password_hash=generate_password_hash('Test-password-123'))
        self.other = User(username='other', email='other@example.com', password_hash='unused')
        db.session.add_all([self.user, self.other]); db.session.commit()
        self.user_id, self.other_id = self.user.id, self.other.id
        self.category_id = Category.query.filter_by(name='Навчання').one().id
        self.client = self.app.test_client()
        self.login_session(self.client, self.user_id)
        self.patches = [patch.object(smtplib, 'SMTP', side_effect=AssertionError('Real SMTP forbidden')),
                        patch.object(smtplib, 'SMTP_SSL', side_effect=AssertionError('Real SMTP forbidden')),
                        patch('app.planning.utcnow', return_value=NOW), patch('app.routes.utcnow', return_value=NOW)]
        for p in self.patches: p.start()

    def tearDown(self):
        db.session.remove(); db.engine.dispose(); self.ctx.pop()
        for p in reversed(self.patches): p.stop()
        self.temp.cleanup()

    @staticmethod
    def login_session(client, user_id):
        with client.session_transaction() as session:
            session['_user_id'] = str(user_id); session['_fresh'] = True

    def token(self, path='/create-task', client=None):
        html = (client or self.client).get(path).get_data(as_text=True)
        return re.search(r'name="csrf_token"[^>]*value="([^"]+)"', html).group(1)

    def post(self, path, **values):
        values['csrf_token'] = self.token()
        return self.client.post(path, data=values)

    def task(self, **values):
        task = Task(title=values.pop('title', 'Task'), user_id=values.pop('user_id', self.user_id),
                    category_id=self.category_id, **values)
        db.session.add(task); db.session.commit()
        return task

    def task_values(self, **changes):
        values = dict(title='Study', description='Draft', category_id=self.category_id,
                      importance='4', duration_min='40', earliest_start='', deadline='')
        values.update(changes)
        return values

    def schedule(self, task, start=START, **changes):
        values = dict(task_id=task.id, planned_start=start, date='2030-01-07', reminder_minutes='10')
        values.update(changes)
        return self.post('/calendar/schedule', **values)

    def event(self, start='2030-01-07T09:00', end='2030-01-07T10:00', **changes):
        values = dict(title='Meeting', event_type='fixed', starts_at=start, ends_at=end)
        values.update(changes)
        return self.post('/calendar/events/new', **values)

    def test_create_edit_filters_archive_restore(self):
        self.assertEqual(self.post('/create-task', **self.task_values(deadline='2030-01-07T14:30')).status_code, 302)
        task = Task.query.one()
        self.assertEqual(task.deadline, datetime(2030,1,7,12,30))
        self.assertEqual(task.importance, 4)
        self.assertEqual(self.post(f'/tasks/{task.id}/edit', **self.task_values(title='Updated')).status_code, 302)
        db.session.expire_all(); self.assertIsNone(task.deadline)
        self.assertIn('Updated', self.client.get('/?q=Updated').text)
        self.assertNotIn('Updated', self.client.get('/?q=absent').text)
        self.assertEqual(self.post(f'/tasks/{task.id}/delete').status_code, 302)
        self.assertEqual(Task.query.count(), 1)
        self.assertNotIn('Updated', self.client.get('/').text)
        self.assertIn('Updated', self.client.get('/?archived=1').text)
        self.assertEqual(self.post(f'/tasks/{task.id}/restore').status_code, 302)
        self.assertIn('Updated', self.client.get('/').text)

    def test_invalid_task_does_not_partially_update(self):
        task = self.task()
        for values in [dict(title=''), dict(title='x'*141), dict(deadline='invalid'), dict(importance='6'),
                       dict(duration_min='0'), dict(category_id='999'), dict(earliest_start=START, deadline='2030-01-07T10:15')]:
            self.assertEqual(self.post(f'/tasks/{task.id}/edit', **self.task_values(**values)).status_code, 400)
            db.session.expire_all(); self.assertEqual(task.title, 'Task')

    def test_task_start_automatically_books_and_tracks_edits(self):
        response = self.post('/create-task', **self.task_values(earliest_start='2030-01-07T10:07'))
        self.assertEqual(response.status_code, 302)
        task = Task.query.one()
        marker = f'data-task-start="{task.id}"'
        item = PlanItem.query.one()
        self.assertEqual((item.planned_start, item.planned_end),
                         (datetime(2030,1,7,8,15), datetime(2030,1,7,9)))
        version = db.session.get(User, self.user_id).data_version
        page = self.client.get('/calendar?date=2030-01-07').text
        self.assertNotIn(marker, page)
        self.assertIn('10:15–11:00', page)
        self.assertEqual(db.session.get(User, self.user_id).data_version, version)
        self.assertEqual(Reminder.query.count(), 0)
        self.assertEqual(self.post(f'/tasks/{task.id}/edit', **self.task_values(
            earliest_start='2030-01-15T08:00')).status_code, 302)
        db.session.expire_all()
        self.assertEqual(item.status, 'canceled')
        self.assertIn('08:00–08:45', self.client.get('/calendar?date=2030-01-15').text)
        self.assertEqual(self.post(f'/tasks/{task.id}/edit', **self.task_values()).status_code, 302)
        self.assertEqual(PlanItem.query.filter_by(status='scheduled').count(), 0)

    def test_auto_booking_creation_errors_leave_no_partial_task(self):
        self.event('2030-01-07T10:00', '2030-01-07T11:00')
        version = db.session.get(User, self.user_id).data_version
        for fields in [dict(earliest_start='2030-01-07T10:00'),
                       dict(earliest_start='2029-01-07T10:00'),
                       dict(earliest_start='2030-01-07T11:07', deadline='2030-01-07T11:52')]:
            response = self.post('/create-task', **self.task_values(**fields))
            self.assertEqual(response.status_code, 400)
            self.assertEqual((Task.query.count(), Plan.query.count(), PlanItem.query.count()), (0,0,0))
            self.assertEqual(db.session.get(User, self.user_id).data_version, version)
            self.assertIn(fields['earliest_start'], response.text)

    def test_auto_booking_failed_edit_preserves_task_plan_and_reminder(self):
        task = self.task(title='Original', duration_min=30, earliest_start=datetime(2030,1,7,8))
        self.schedule(task)
        item = PlanItem.query.one()
        reminder = Reminder.query.one()
        self.event('2030-01-07T11:00', '2030-01-07T12:00')
        version = db.session.get(User, self.user_id).data_version
        response = self.post(f'/tasks/{task.id}/edit', **self.task_values(
            title='Changed', earliest_start='2030-01-07T11:00'))
        self.assertEqual(response.status_code, 400)
        db.session.expire_all()
        self.assertEqual((task.title, task.duration_min, task.earliest_start),
                         ('Original', 30, datetime(2030,1,7,8)))
        self.assertEqual((item.status, reminder.status, item.plan.status), ('scheduled','pending','accepted'))
        self.assertEqual(PlanItem.query.count(), 1)
        self.assertEqual(db.session.get(User, self.user_id).data_version, version)

    def test_auto_booking_existing_task_is_idempotent_and_respects_other_tasks(self):
        task = self.task(earliest_start=datetime(2030,1,7,8), duration_min=40)
        values = self.task_values(earliest_start='2030-01-07T10:00')
        self.assertEqual(self.post(f'/tasks/{task.id}/edit', **values).status_code, 302)
        self.assertEqual(self.post(f'/tasks/{task.id}/edit', **values).status_code, 302)
        self.assertEqual(PlanItem.query.count(), 1)
        response = self.post('/create-task', **self.task_values(earliest_start='2030-01-07T10:15'))
        self.assertEqual(response.status_code, 400)
        self.assertEqual(Task.query.count(), 1)
        self.assertEqual(self.post(f'/tasks/{task.id}/edit', **dict(values, duration_min='60')).status_code, 302)
        self.assertEqual(PlanItem.query.filter_by(status='canceled').count(), 1)
        active = PlanItem.query.filter_by(status='scheduled').one()
        self.assertEqual(active.planned_end, datetime(2030,1,7,9))

    def test_task_start_uses_local_day_and_week_boundaries(self):
        first = self.task(title='Monday start', earliest_start=datetime(2030,1,6,22,15))
        last = self.task(title='Sunday start', earliest_start=datetime(2030,1,13,21,45))
        next_week = self.task(earliest_start=datetime(2030,1,13,22))
        previous = self.task(earliest_start=datetime(2030,1,6,21,59))
        page = self.client.get('/calendar?date=2030-01-07').text
        monday = page.split('aria-label="2030-01-07"')[1].split('</section>')[0]
        sunday = page.split('aria-label="2030-01-13"')[1].split('</section>')[0]
        self.assertIn(f'data-task-start="{first.id}"', monday)
        self.assertIn('Від 00:15', monday)
        self.assertIn(f'data-task-start="{last.id}"', sunday)
        for task in (next_week, previous):
            self.assertNotIn(f'data-task-start="{task.id}"', page)

    def test_task_start_filters_owner_archive_and_completion(self):
        visible = self.task(earliest_start=datetime(2030,1,7,8))
        hidden = [self.task(user_id=self.other_id, earliest_start=datetime(2030,1,7,8)),
                  self.task(archived_at=NOW, earliest_start=datetime(2030,1,7,8)),
                  self.task(status='completed', earliest_start=datetime(2030,1,7,8)),
                  self.task(deadline=datetime(2030,1,7,15))]
        page = self.client.get('/calendar?date=2030-01-07').text
        self.assertIn(f'data-task-start="{visible.id}"', page)
        for task in hidden:
            self.assertNotIn(f'data-task-start="{task.id}"', page)

    def test_task_start_disappears_when_planned_and_returns_after_cancellation(self):
        task = self.task(earliest_start=datetime(2030,1,7,8))
        marker = f'data-task-start="{task.id}"'
        self.assertEqual(self.schedule(task, '2030-01-15T10:00').status_code, 302)
        self.assertNotIn(marker, self.client.get('/calendar?date=2030-01-07').text)
        item = PlanItem.query.one()
        self.assertEqual(self.post(f'/calendar/items/{item.id}/cancel').status_code, 302)
        self.assertIn(marker, self.client.get('/calendar?date=2030-01-07').text)
        self.assertEqual(self.schedule(task).status_code, 302)
        page = self.client.get('/calendar?date=2030-01-07').text
        self.assertNotIn(marker, page)
        self.assertIn('10:00–10:30', page)

    def test_task_start_does_not_occupy_time_or_accept_draft(self):
        task = self.task(earliest_start=datetime(2030,1,7,8))
        draft = Plan(user_id=self.user_id, horizon_start=NOW, horizon_end=NOW+timedelta(days=1),
                     status='draft', source_version=1)
        db.session.add(PlanItem(plan=draft, task=task, planned_start=datetime(2030,1,7,8),
                               planned_end=datetime(2030,1,7,8,30)))
        db.session.commit()
        page = self.client.get('/calendar?date=2030-01-07').text
        self.assertIn(f'data-task-start="{task.id}"', page)
        self.assertEqual(self.event('2030-01-07T10:00', '2030-01-07T10:30').status_code, 302)
        db.session.expire_all()
        self.assertEqual(draft.status, 'draft')
        self.assertEqual(Reminder.query.count(), 0)

    def test_schedule_rounds_duration_and_persists_snapshot(self):
        task = self.task(duration_min=40)
        self.assertEqual(self.schedule(task).status_code, 302)
        item = PlanItem.query.one()
        self.assertEqual(item.planned_start, datetime(2030,1,7,8))
        self.assertEqual(item.planned_end-item.planned_start, timedelta(minutes=45))
        self.assertEqual(item.task_snapshot_json['duration_min'], 40)
        self.assertEqual(item.task_snapshot_json['category'], 'Навчання')
        self.assertEqual(item.plan.status, 'accepted')
        self.assertEqual(item.plan.parameters_json['mode'], 'manual')
        self.assertIsNone(item.priority_score)
        self.assertEqual(Reminder.query.one().remind_at, datetime(2030,1,7,7,50))
        self.assertEqual(db.session.get(User,self.user_id).data_version, 2)
        self.assertEqual(item.plan.source_version, 1)
        self.assertIn('10:00–10:45', self.client.get('/calendar?date=2030-01-07').text)

    def test_busy_event_and_task_conflicts_but_adjacent_is_allowed(self):
        self.assertEqual(self.event().status_code, 302)
        task = self.task(duration_min=30)
        self.assertEqual(self.schedule(task, '2030-01-07T09:45').status_code, 400)
        self.assertEqual(PlanItem.query.count(), 0)
        self.assertEqual(self.schedule(task).status_code, 302)
        another = self.task(title='Second')
        self.assertEqual(self.schedule(another, '2030-01-07T10:15').status_code, 400)
        self.assertEqual(self.schedule(another, '2030-01-07T10:30').status_code, 302)
        self.assertEqual(self.event('2030-01-07T10:15','2030-01-07T11:00').status_code, 400)
        self.assertEqual(CalendarEvent.query.count(), 1)

    def test_task_windows_past_time_and_grid(self):
        task = self.task(earliest_start=datetime(2030,1,7,8), deadline=datetime(2030,1,7,9), duration_min=40)
        for start in ['2030-01-07T09:45', '2030-01-07T10:30', '2029-01-07T10:00',
                      '2030-01-07T10:01', '', 'bad']:
            self.assertEqual(self.schedule(task, start).status_code, 400, start)
        self.assertEqual(PlanItem.query.count(), 0)
        self.assertEqual(self.schedule(task, '2030-01-07T10:15').status_code, 302)  # Ends exactly at deadline.

    def test_reschedule_preserves_old_item_snapshot_and_cancels_reminder(self):
        task = self.task()
        self.schedule(task)
        old_id = PlanItem.query.one().id
        self.assertEqual(self.schedule(task, '2030-01-07T11:00').status_code, 302)
        db.session.expire_all()
        old = db.session.get(PlanItem, old_id)
        self.assertEqual(old.status, 'canceled')
        self.assertIsNotNone(old.canceled_at)
        self.assertEqual(old.reminders[0].status, 'canceled')
        self.assertEqual(old.plan.status, 'canceled')
        self.assertEqual(PlanItem.query.count(), 2)
        self.assertEqual(old.task_snapshot_json['title'], 'Task')
        self.assertNotIn('10:00–10:30', self.client.get('/calendar?date=2030-01-07').text)

    def test_failed_reschedule_keeps_existing_plan(self):
        task = self.task()
        self.schedule(task)
        self.event('2030-01-07T11:00','2030-01-07T12:00')
        self.assertEqual(self.schedule(task,'2030-01-07T11:15').status_code, 400)
        item = PlanItem.query.one()
        self.assertEqual(item.status, 'scheduled')
        self.assertEqual(item.reminders[0].status, 'pending')

    def test_edit_description_keeps_plan_and_duration_change_cancels(self):
        task = self.task()
        self.schedule(task)
        self.post(f'/tasks/{task.id}/edit', **self.task_values(title='Changed', duration_min=30))
        db.session.expire_all()
        item = PlanItem.query.one()
        self.assertEqual(item.status, 'scheduled')
        self.assertEqual(item.task_snapshot_json['title'], 'Task')
        self.post(f'/tasks/{task.id}/edit', **self.task_values(duration_min=45))
        db.session.expire_all(); self.assertEqual(item.status, 'canceled')

    def test_archive_cancels_future_placement_and_preserves_history(self):
        task = self.task()
        self.schedule(task)
        self.post(f'/tasks/{task.id}/delete')
        db.session.expire_all()
        self.assertIsNotNone(task.archived_at)
        self.assertEqual(PlanItem.query.one().status, 'canceled')
        self.assertEqual(self.schedule(task).status_code, 400)

    def test_cancel_then_schedule_again(self):
        task = self.task()
        self.schedule(task)
        item = PlanItem.query.one()
        self.assertEqual(self.post(f'/calendar/items/{item.id}/cancel').status_code,302)
        self.assertEqual(self.schedule(task).status_code,302)
        self.assertEqual(PlanItem.query.filter_by(status='scheduled').count(),1)

    def test_start_complete_actual_times_and_protect_in_progress(self):
        task = self.task()
        self.schedule(task)
        item = PlanItem.query.one()
        self.post(f'/calendar/items/{item.id}/start')
        db.session.expire_all(); self.assertIsNone(item.actual_start)
        with patch('app.routes.utcnow', return_value=datetime(2030,1,7,8,1)):
            self.post(f'/calendar/items/{item.id}/start')
        db.session.expire_all()
        self.assertEqual(task.status,'in_progress')
        self.assertEqual(item.actual_start,datetime(2030,1,7,8,1))
        self.assertEqual(self.post(f'/tasks/{task.id}/edit', **self.task_values(duration_min=50)).status_code,400)
        self.post(f'/tasks/{task.id}/delete')
        db.session.expire_all(); self.assertIsNone(task.archived_at)
        with patch('app.routes.utcnow', return_value=datetime(2030,1,7,8,25)):
            self.post(f'/tasks/{task.id}/toggle')
        db.session.expire_all()
        self.assertEqual(task.status,'completed')
        self.assertEqual(item.actual_end,datetime(2030,1,7,8,25))
        self.assertEqual(item.reminders[0].status,'canceled')
        self.post(f'/tasks/{task.id}/toggle')
        db.session.expire_all()
        self.assertEqual(task.status,'pending')
        self.assertEqual(item.status,'completed')  # Completed history is immutable.

    def test_completing_before_scheduled_time_frees_future_slot(self):
        task=self.task(); self.schedule(task)
        self.post(f'/tasks/{task.id}/toggle')
        self.assertEqual(PlanItem.query.one().status,'canceled')
        db.session.expire_all(); self.assertEqual(task.status,'completed')

    def test_only_one_task_can_be_in_progress(self):
        first=self.task(title='First'); second=self.task(title='Second')
        self.schedule(first); self.schedule(second,'2030-01-07T10:30')
        items=PlanItem.query.order_by(PlanItem.id).all()
        with patch('app.routes.utcnow',return_value=datetime(2030,1,7,9)):
            self.post(f'/calendar/items/{items[0].id}/start')
            self.post(f'/calendar/items/{items[1].id}/start')
        db.session.expire_all()
        self.assertEqual(PlanItem.query.filter_by(status='in_progress').count(),1)
        self.assertEqual(items[1].status,'scheduled')
        self.assertIsNone(items[1].actual_start)

    def test_date_range_rejected_before_save_and_boundary_week_renders(self):
        self.assertEqual(self.event('2101-01-07T09:00','2101-01-07T10:00').status_code,400)
        self.assertEqual(CalendarEvent.query.count(),0)
        self.assertEqual(self.post('/create-task',**self.task_values(deadline='2101-01-07T10:00')).status_code,400)
        self.assertEqual(self.schedule(self.task(),'2101-01-07T10:00').status_code,400)
        self.assertEqual(PlanItem.query.count(),0)
        self.assertEqual(self.client.get('/calendar?date=2100-12-31').status_code,200)

    def test_event_edit_delete_and_invalid_intervals(self):
        for start,end in [(START,START),(START,'2030-01-07T09:00'),('',START),('bad',START)]:
            self.assertEqual(self.event(start,end).status_code,400)
        self.event()
        event=CalendarEvent.query.one()
        self.assertEqual(self.post(f'/calendar/events/{event.id}/edit', title='Break',event_type='break',
                                  starts_at='2030-01-07T12:00',ends_at='2030-01-07T12:30').status_code,302)
        db.session.expire_all(); self.assertEqual(event.event_type,'break')
        self.assertEqual(self.post(f'/calendar/events/{event.id}/delete').status_code,302)
        self.assertEqual(CalendarEvent.query.count(),0)

    def test_cross_midnight_event_and_calendar_navigation(self):
        self.assertEqual(self.event('2030-01-06T23:30','2030-01-07T00:30',title='Across midnight').status_code,302)
        html=self.client.get('/calendar?date=2030-01-07').text
        self.assertIn('Across midnight',html)
        self.assertIn('00:00–00:30',html)
        self.assertEqual(self.client.get('/calendar?date=not-a-date').status_code,400)
        self.assertEqual(self.client.get('/calendar?date=9999-12-31').status_code,400)

    def test_dst_gap_ambiguity_and_winter_summer_offsets(self):
        for value in ('2030-03-31T03:30','2030-10-27T03:30'):
            with self.assertRaises(ValueError):
                local_to_utc(value,ZoneInfo('Europe/Kyiv'))
        self.assertEqual(local_to_utc(START,ZoneInfo('Europe/Kyiv')),datetime(2030,1,7,8))
        self.assertEqual(local_to_utc('2030-07-07T10:00',ZoneInfo('Europe/Kyiv')),datetime(2030,7,7,7))

    def test_csrf_and_ownership_for_tasks_events_and_items(self):
        other_task=self.task(title='Private',user_id=self.other_id)
        event=CalendarEvent(user_id=self.other_id,title='Private event',starts_at=NOW,ends_at=NOW+timedelta(hours=1))
        db.session.add(event); db.session.commit()
        with patch('app.planning.utcnow',return_value=NOW):
            schedule_task(db.session.get(User,self.other_id),other_task,NOW+timedelta(hours=2))
            db.session.commit()
        item=PlanItem.query.one()
        self.assertEqual(self.client.post('/create-task',data={'title':'No token'}).status_code,400)
        for action in ('edit','delete','toggle','restore'):
            self.assertEqual(self.post(f'/tasks/{other_task.id}/{action}',**self.task_values()).status_code,403)
        self.assertEqual(self.schedule(other_task).status_code,403)
        for action in ('edit','delete'):
            self.assertEqual(self.post(f'/calendar/events/{event.id}/{action}',title='Changed').status_code,403)
        for action in ('cancel','start'):
            self.assertEqual(self.post(f'/calendar/items/{item.id}/{action}').status_code,403)
        self.assertNotIn('Private',self.client.get('/').text)
        self.assertNotIn('Private',self.client.get('/calendar?date=2030-01-07').text)
        # Another user's busy time does not block the current user.
        self.assertEqual(self.schedule(self.task()).status_code,302)

    def test_constraints_foreign_keys_and_unique_plan_task(self):
        self.assertEqual(db.session.execute(text('PRAGMA foreign_keys')).scalar(),1)
        db.session.add(Task(user_id=999,category_id=self.category_id,title='Orphan'))
        with self.assertRaises(IntegrityError): db.session.commit()
        db.session.rollback()
        db.session.add(Task(user_id=self.user_id,category_id=self.category_id,title='Invalid',importance=6))
        with self.assertRaises(IntegrityError): db.session.commit()
        db.session.rollback()
        task=self.task(); self.schedule(task)
        item=PlanItem.query.one()
        db.session.add(PlanItem(plan_id=item.plan_id,task_id=task.id,planned_start=item.planned_start,planned_end=item.planned_end))
        with self.assertRaises(IntegrityError): db.session.commit()
        db.session.rollback()

    def test_mail_disabled_all_entry_points_and_suppression(self):
        task=self.task(); self.schedule(task)
        for enabled in (False,True):
            self.app.config.update(MAIL_ENABLED=enabled,MAIL_SUPPRESS_SEND=True)
            with patch('app.utils.mail.send') as transport:
                self.assertFalse(send_reminder_email('test@example.com','test','test'))
                self.assertFalse(check_and_send_reminders(self.app)['enabled'])
                self.assertEqual(self.client.get('/send-test-email').status_code,200)
                self.assertEqual(self.app.test_cli_runner().invoke(args=['reminders']).exit_code,0)
                transport.assert_not_called()
        db.session.expire_all()
        self.assertEqual(Reminder.query.one().status,'pending')
        self.assertEqual(Reminder.query.one().attempt_count,0)

    def test_reminder_retry_once_and_skip_canceled_completed_overdue(self):
        task=self.task(); self.schedule(task)
        self.app.config.update(MAIL_ENABLED=True,MAIL_SUPPRESS_SEND=False)
        now=datetime(2030,1,7,7,55)
        with patch('app.utils.send_reminder_email',side_effect=RuntimeError('test')):
            self.assertEqual(check_and_send_reminders(self.app,now)['failed'],1)
        db.session.expire_all()
        self.assertEqual(Reminder.query.one().status,'failed')
        self.assertEqual(Reminder.query.one().attempt_count,1)
        with patch('app.utils.send_reminder_email',return_value=True) as transport:
            self.assertEqual(check_and_send_reminders(self.app,now)['sent'],1)
            self.assertEqual(check_and_send_reminders(self.app,now)['sent'],0)
            self.assertEqual(transport.call_count,1)
        db.session.expire_all()
        self.assertEqual(Reminder.query.one().attempt_count,2)
        for title,start in [('Canceled','2030-01-07T11:00'),('Done','2030-01-07T12:00'),('Overdue','2030-01-07T13:00')]:
            candidate=self.task(title=title); self.schedule(candidate,start)
            if title=='Canceled':
                self.post(f'/calendar/items/{PlanItem.query.order_by(PlanItem.id.desc()).first().id}/cancel')
            elif title=='Done':
                self.post(f'/tasks/{candidate.id}/toggle')
        with patch('app.utils.send_reminder_email',return_value=True) as transport:
            self.assertEqual(check_and_send_reminders(self.app,datetime(2030,1,7,14))['sent'],0)
            transport.assert_not_called()

    def test_registration_login_and_public_pages(self):
        self.assertEqual(self.post('/logout').status_code,302)
        for path in ('/','/login','/register'):
            self.assertEqual(self.client.get(path).status_code,200)
        response=self.client.post('/register',data={'csrf_token':self.token('/register'),
            'username':'newuser','email':'newuser@example.com','password':'New-password-123','confirm_password':'New-password-123'})
        self.assertEqual(response.status_code,302)
        response=self.client.post('/login',data={'csrf_token':self.token('/login'),
            'email':'newuser@example.com','password':'New-password-123'})
        self.assertEqual(response.status_code,302)
        self.assertIn('newuser',self.client.get('/').text)

    def test_start_time_reminder_allows_short_polling_delay(self):
        self.schedule(self.task(), reminder_minutes='0')
        self.app.config.update(MAIL_ENABLED=True,MAIL_SUPPRESS_SEND=False)
        with patch('app.utils.send_reminder_email',return_value=True) as transport:
            self.assertEqual(check_and_send_reminders(self.app,datetime(2030,1,7,8,2))['sent'],1)
            self.assertEqual(transport.call_count,1)

    def test_initialization_is_repeatable(self):
        task=self.task(); task_id=task.id
        initialize_database(); initialize_database()
        self.assertEqual(Task.query.count(),1)
        self.assertEqual(db.session.get(Task,task_id).title,'Task')

    def test_simultaneous_booking_only_one_succeeds(self):
        first=self.task(title='First'); second=self.task(title='Second')
        clients=[self.app.test_client(),self.app.test_client()]
        data=[]
        for client,task in zip(clients,(first,second)):
            self.login_session(client,self.user_id)
            data.append(dict(csrf_token=self.token(client=client),task_id=task.id,planned_start=START,date='2030-01-07'))
        # Each thread has its own request/app context and database connection.
        def book(index):
            return clients[index].post('/calendar/schedule',data=data[index]).status_code
        db.session.remove()
        with ThreadPoolExecutor(max_workers=2) as pool:
            statuses=list(pool.map(book,(0,1)))
        self.assertEqual(sorted(statuses),[302,400])
        self.assertEqual(PlanItem.query.filter_by(status='scheduled').count(),1)

class MigrationTests(unittest.TestCase):
    def make_legacy(self, path, orphan=False):
        password=generate_password_hash('Legacy-password')
        with closing(sqlite3.connect(path)) as conn, conn:
            conn.executescript("""
                CREATE TABLE user(id INTEGER PRIMARY KEY, username TEXT, email TEXT, password TEXT);
                CREATE TABLE task(id INTEGER PRIMARY KEY,title TEXT,description TEXT,created_at DATETIME,
                    user_id INTEGER,is_completed BOOLEAN,remind_at DATETIME,due_date DATETIME,reminder_sent_at DATETIME);
                CREATE TABLE tag(id INTEGER PRIMARY KEY,name TEXT);
                CREATE TABLE task_tags(task_id INTEGER,tag_id INTEGER);
                CREATE TABLE alembic_version(version_num TEXT);
            """)
            conn.execute('INSERT INTO user VALUES(1,?,?,?)',('Legacy','legacy@example.com',password))
            conn.execute("INSERT INTO task VALUES(7,'Теза','Опис','2026-01-01 12:00:00',?,1,'2026-11-03 09:50:00','2026-11-03 10:00:00',NULL)",(999 if orphan else 1,))
            conn.execute("INSERT INTO tag VALUES(1,'Тег А'),(2,'Тег Б')")
            conn.execute('INSERT INTO task_tags VALUES(7,1),(7,2)')
        return password

    def test_legacy_upgrade_preserves_data_backup_and_repeatability(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'legacy.db'
            password=self.make_legacy(path)
            app=create_app({'TESTING':True,'SECRET_KEY':'test','SQLALCHEMY_DATABASE_URI':'sqlite:///'+str(path)})
            with app.app_context():
                backup=initialize_database()
                self.assertTrue(backup.exists())
                user=db.session.get(User,1); task=db.session.get(Task,7)
                self.assertEqual(user.password_hash,password)
                self.assertTrue(check_password_hash(user.password_hash,'Legacy-password'))
                self.assertEqual(task.title,'Теза'); self.assertEqual(task.description,'Опис')
                self.assertEqual(task.status,'completed')
                self.assertEqual(task.created_at,datetime(2026,1,1,12))
                self.assertEqual(task.deadline,datetime(2026,11,3,8))
                self.assertEqual(task.legacy_remind_at,datetime(2026,11,3,7,50))
                self.assertEqual(task.legacy_tags_json,['Тег А','Тег Б'])
                self.assertEqual(task.category.name,'Без категорії')
                initialize_database()
                self.assertEqual(Task.query.count(),1)
                self.assertEqual(Plan.query.count(),0)
                self.assertEqual(db.session.execute(text('PRAGMA foreign_key_check')).all(),[])
                with closing(sqlite3.connect(backup)) as original:
                    self.assertEqual(original.execute('SELECT COUNT(*) FROM task').fetchone()[0],1)
                db.session.remove(); db.engine.dispose()

    def test_invalid_legacy_data_rolls_back_schema_and_records(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'legacy.db'; self.make_legacy(path,orphan=True)
            app=create_app({'TESTING':True,'SECRET_KEY':'test','SQLALCHEMY_DATABASE_URI':'sqlite:///'+str(path)})
            with app.app_context():
                with self.assertRaises(IntegrityError): initialize_database()
                db.session.remove(); db.engine.dispose()
            with closing(sqlite3.connect(path)) as conn:
                tables={row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                self.assertIn('user',tables); self.assertNotIn('users',tables)
                self.assertEqual(conn.execute('SELECT COUNT(*) FROM task').fetchone()[0],1)
                self.assertEqual(conn.execute('PRAGMA user_version').fetchone()[0],0)

if __name__ == '__main__':
    unittest.main()
