import math
import re
import smtplib
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from app import create_app, db
from app.database import initialize_database
from app.models import User, Category, Task, Plan, PlanItem, CalendarEvent, Reminder
from app.recommender import (TaskInput, HistoryCount, WeightedHistoryCount, generate_plan, fuzzy_priority, memberships,
                            history_rating, daypart_of, allocated_minutes, urgency_at, METHODS)
from app.recommendations import historical_counts, available_windows, create_draft, accept_draft
from app.planning import accepted_items

NOW=datetime(2030,1,7,6)  # 08:00 Kyiv
WINDOWS=[(datetime(2030,1,7,7),datetime(2030,1,7,16))]

class EngineTests(unittest.TestCase):
    def task(self, **changes):
        values=dict(id=1,title='Study',category_id=1,importance=4,duration_min=60)
        values.update(changes)
        return TaskInput(**values)

    def run_plan(self,tasks=None,**changes):
        params=dict(tasks=tasks or [self.task()],available_windows=WINDOWS,busy_intervals=[],
                    history={},as_of=NOW)
        params.update(changes)
        return generate_plan(**params)

    def test_memberships_partition_endpoints_and_bounds(self):
        self.assertEqual(memberships(0),(1,0,0))
        self.assertEqual(memberships(.5),(0,1,0))
        self.assertEqual(memberships(1),(0,0,1))
        for n in range(101):
            self.assertAlmostEqual(sum(memberships(n/100)),1)
        for value in (-1,1.1,math.nan,math.inf,True):
            with self.assertRaises(ValueError): memberships(value)

    def test_fuzzy_monotonicity_and_nonadditivity(self):
        self.assertAlmostEqual(fuzzy_priority(.5,.5),.55)
        self.assertAlmostEqual(fuzzy_priority(.75,.5),.70)
        for fixed in [n/20 for n in range(21)]:
            for orientation in (False,True):
                values=[fuzzy_priority(fixed,n/20) if orientation else fuzzy_priority(n/20,fixed) for n in range(21)]
                self.assertTrue(all(0 <= v <= 1 for v in values))
                self.assertTrue(all(a <= b+1e-12 for a,b in zip(values,values[1:])))
        # Any additive f(I)+g(U) would have zero mixed difference.
        mixed=fuzzy_priority(1,1)-fuzzy_priority(1,0)-fuzzy_priority(0,1)+fuzzy_priority(0,0)
        self.assertNotAlmostEqual(mixed,0)

    def test_history_shrinkage_and_proven_influence_bound(self):
        self.assertEqual(history_rating(0,0),.5)
        self.assertAlmostEqual(history_rating(1,1),2/3)
        for total in range(31):
            for successes in range(total+1):
                h=history_rating(successes,total)
                self.assertLessEqual(abs(.2*h-.1),.1*total/(total+2)+1e-12)
        for successes,total in [(2,1),(-1,2),(1,-1),(True,1),(1,1.5)]:
            with self.assertRaises(ValueError): history_rating(successes,total)

    def test_urgency_relative_duration_and_frozen_origin(self):
        task=self.task(deadline=NOW+timedelta(hours=2))
        self.assertEqual(urgency_at(task,NOW),.5)
        self.assertEqual(urgency_at(self.task(),NOW),0)
        self.assertEqual(urgency_at(self.task(deadline=NOW),NOW),1)
        self.assertEqual(urgency_at(self.task(earliest_start=NOW+timedelta(hours=1),deadline=NOW+timedelta(hours=2)),NOW),1)

    def test_cold_start_identical_to_fuzzy_and_order_deterministic(self):
        tasks=[self.task(id=i,importance=1+i%5,duration_min=15+i*10,deadline=NOW+timedelta(hours=4+i%3)) for i in range(1,12)]
        full=self.run_plan(tasks)
        fuzzy=self.run_plan(tasks,method='fuzzy')
        placements=lambda result:[(a.task_id,a.start,a.end) for a in result.assignments]
        self.assertEqual(placements(full),placements(fuzzy))
        self.assertEqual(placements(full),placements(self.run_plan(list(reversed(tasks)))))

    def test_history_changes_daypart_without_changing_constraints(self):
        history={(1,'morning'):HistoryCount(1,10),(1,'afternoon'):HistoryCount(9,10)}
        full=self.run_plan(history=history)
        no_history=self.run_plan(method='fuzzy',history=history)
        self.assertEqual(full.assignments[0].start,datetime(2030,1,7,10))  # noon Kyiv
        self.assertEqual(no_history.assignments[0].start,datetime(2030,1,7,7))
        self.assertEqual(full.assignments[0].history_total,10)
        self.assertEqual(full.assignments[0].daypart,'afternoon')
        self.assertGreater(full.assignments[0].history_score,.5)

    def test_history_cannot_override_deadline_or_busy_period(self):
        history={(1,'morning'):HistoryCount(0,10),(1,'afternoon'):HistoryCount(10,10)}
        result=self.run_plan([self.task(deadline=datetime(2030,1,7,10))],history=history,
                             busy_intervals=[(datetime(2030,1,7,7),datetime(2030,1,7,8))])
        self.assertEqual(result.assignments[0].start,datetime(2030,1,7,8))
        self.assertLessEqual(result.assignments[0].end,datetime(2030,1,7,10))

    def test_rounding_grid_adjacent_intervals_and_nonpreemption(self):
        self.assertEqual(allocated_minutes(40),45)
        result=self.run_plan([self.task(duration_min=40)],as_of=datetime(2030,1,7,7,1))
        self.assertEqual(result.assignments[0].start,datetime(2030,1,7,7,15))
        self.assertEqual(result.assignments[0].end,datetime(2030,1,7,8))
        result=self.run_plan([self.task(duration_min=60)],available_windows=[
            (datetime(2030,1,7,7),datetime(2030,1,7,7,30)),
            (datetime(2030,1,7,8),datetime(2030,1,7,8,30))])
        self.assertEqual(len(result.assignments),0)  # Cannot split one task across windows.

    def test_all_baselines_feasible_and_unique_under_load(self):
        tasks=[self.task(id=i,importance=i%5+1,duration_min=i%4*15+15,
                         deadline=datetime(2030,1,7,10+i%6)) for i in range(1,30)]
        busy=[(datetime(2030,1,7,9),datetime(2030,1,7,10))]
        history={(1,'afternoon'):WeightedHistoryCount(8,10,4.,5.)}
        for method in METHODS:
            result=self.run_plan(tasks,method=method,busy_intervals=busy,history=history)
            self.assertEqual(len({a.task_id for a in result.assignments}),len(result.assignments))
            self.assertEqual(len(result.assignments)+len(result.unscheduled),len(tasks))
            for a in result.assignments:
                task=next(t for t in tasks if t.id==a.task_id)
                self.assertTrue(WINDOWS[0][0] <= a.start < a.end <= min(task.deadline,WINDOWS[0][1]))
                self.assertFalse(any(a.start < end and a.end > start for start,end in busy))
            self.assertTrue(all(a.end <= b.start for a,b in zip(result.assignments,result.assignments[1:])))

    def test_edf_prefers_nearer_deadline(self):
        tasks=[self.task(id=1,importance=5,deadline=datetime(2030,1,7,15)),
               self.task(id=2,importance=1,deadline=datetime(2030,1,7,9))]
        result=self.run_plan(tasks,method='edf')
        self.assertEqual(result.assignments[0].task_id,2)

    def test_unplaceable_reasons_and_empty_input(self):
        result=self.run_plan([self.task(deadline=NOW-timedelta(minutes=1))])
        self.assertEqual(len(result.assignments),0)
        self.assertIn('строк',result.unscheduled[0].reason)
        self.assertEqual(generate_plan([],WINDOWS,[],{},NOW).assignments,[])

    def test_invalid_inputs_fail_explicitly(self):
        cases=[dict(history_weight=math.nan),dict(prior_strength=0),dict(method='invented'),
               dict(timezone_name='Invalid/Zone'),dict(as_of=NOW.replace(tzinfo=timezone.utc)),
               dict(available_windows=[]),dict(available_windows=[(NOW,NOW)]),
               dict(available_windows=[(NOW,NOW+timedelta(days=8))]),
               dict(tasks=[self.task(),self.task()]),dict(tasks=[self.task(importance=0)]),
               dict(tasks=[self.task(duration_min=0)])]
        for params in cases:
            with self.assertRaises(ValueError,msg=str(params)): self.run_plan(**params)

    def test_local_dayparts_and_summer_time(self):
        self.assertEqual(daypart_of(datetime(2030,1,7,4)),'morning')
        self.assertEqual(daypart_of(datetime(2030,1,7,10)),'afternoon')
        self.assertEqual(daypart_of(datetime(2030,7,7,9)),'afternoon')
        self.assertEqual(daypart_of(datetime(2030,1,7,22)),'night')

class RecommendationIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.app=create_app({'TESTING':True,'SECRET_KEY':'test',
            'SQLALCHEMY_DATABASE_URI':'sqlite:///'+str(Path(self.temp.name)/'test.db')})
        self.ctx=self.app.app_context();self.ctx.push();initialize_database()
        self.user=User(username='User',email='test@example.com',password_hash='unused')
        self.other=User(username='Other',email='other@example.com',password_hash='unused')
        db.session.add_all([self.user,self.other]);db.session.commit()
        self.user_id=self.user.id;self.other_id=self.other.id
        self.category=Category.query.filter_by(name='Навчання').one()
        self.category_id=self.category.id
        self.client=self.app.test_client()
        with self.client.session_transaction() as s: s['_user_id']=str(self.user.id);s['_fresh']=True
        self.patches=[patch('app.routes.utcnow',return_value=NOW),
                      patch('app.recommendations.utcnow',return_value=NOW),
                      patch.object(smtplib,'SMTP',side_effect=AssertionError('SMTP forbidden')),
                      patch.object(smtplib,'SMTP_SSL',side_effect=AssertionError('SMTP_SSL forbidden'))]
        for p in self.patches:p.start()

    def tearDown(self):
        db.session.remove();db.engine.dispose();self.ctx.pop()
        for p in reversed(self.patches):p.stop()
        self.temp.cleanup()

    def post(self,path,**values):
        html=self.client.get('/recommendations').text
        values['csrf_token']=re.search(r'name="csrf_token"[^>]*value="([^"]+)"',html).group(1)
        return self.client.post(path,data=values)

    def task(self,**changes):
        values=dict(user_id=self.user_id,category_id=self.category_id,title='Study',importance=4,duration_min=40)
        values.update(changes)
        t=Task(**values);db.session.add(t);db.session.commit();return t

    def draft(self,**changes):
        values=dict(first_day='2030-01-07',days='1',start_clock='09:00',end_clock='18:00',reminder_minutes='10')
        values.update(changes)
        return self.post('/recommendations',**values)

    def history_attempt(self,*,success=True,canceled=None,accepted=True,owner=None,category_id=None,end=None):
        end=end or NOW-timedelta(days=1)
        start=end-timedelta(hours=1)
        task=self.task(user_id=owner or self.user_id)
        plan=Plan(user_id=owner or self.user_id,created_at=start-timedelta(days=1),
                  accepted_at=start-timedelta(hours=2) if accepted else None,
                  horizon_start=start,horizon_end=end,status='canceled' if canceled else 'accepted' if accepted else 'draft',
                  source_version=1,parameters_json={})
        item=PlanItem(plan=plan,task=task,planned_start=start,planned_end=end,
                      actual_end=end if success else None,status='canceled' if canceled else 'completed' if success else 'scheduled',
                      canceled_at=canceled,
                      task_snapshot_json={'category_id':category_id or self.category_id,'timezone':'Europe/Kyiv','title':'History'})
        db.session.add(plan);db.session.commit();return item

    def test_draft_has_no_calendar_effect_then_accepts_atomically(self):
        self.task()
        version=self.user.data_version
        self.assertEqual(self.draft().status_code,302)
        plan=Plan.query.one()
        self.assertEqual(plan.status,'draft')
        self.assertEqual(accepted_items(self.user_id).count(),0)
        self.assertEqual(Reminder.query.count(),0)
        self.assertEqual(self.user.data_version,version)
        html=self.client.get(f'/recommendations/{plan.id}').text
        self.assertIn('Історії виконання ще немає',html)
        self.assertIn('Прийняти план',html)
        self.assertIn('Пріоритет задачі',html)
        self.assertEqual(self.post(f'/recommendations/{plan.id}/accept').status_code,302)
        db.session.expire_all()
        self.assertEqual(plan.status,'accepted')
        self.assertEqual(accepted_items(self.user_id).count(),1)
        self.assertEqual(Reminder.query.count(),1)
        self.assertEqual(self.user.data_version,version+1)
        self.assertEqual(plan.parameters_json['model_version'],'1.1')
        self.assertEqual(plan.parameters_json['method'],'personalized_recent')
        self.assertIn('09:00–09:45',self.client.get('/calendar?date=2030-01-07').text)

    def test_repeated_accept_is_idempotent(self):
        self.task();self.draft();plan=Plan.query.one()
        self.post(f'/recommendations/{plan.id}/accept')
        version=db.session.get(User,self.user_id).data_version
        self.assertEqual(self.post(f'/recommendations/{plan.id}/accept').status_code,302)
        self.assertEqual(Reminder.query.count(),1)
        self.assertEqual(db.session.get(User,self.user_id).data_version,version)

    def test_other_draft_becomes_stale_after_acceptance(self):
        self.task();self.draft();self.draft()
        plans=Plan.query.order_by(Plan.id).all()
        self.post(f'/recommendations/{plans[0].id}/accept')
        self.assertEqual(self.post(f'/recommendations/{plans[1].id}/accept').status_code,409)
        self.assertEqual(accepted_items(self.user_id).count(),1)

    def test_changed_task_or_calendar_rejects_without_partial_accept(self):
        task=self.task();self.task(title='Another');self.draft();plan=Plan.query.one()
        self.post(f'/tasks/{task.id}/edit',title='Changed',category_id=self.category_id,importance=4,duration_min=40)
        self.assertEqual(self.post(f'/recommendations/{plan.id}/accept').status_code,409)
        db.session.expire_all();self.assertEqual(plan.status,'draft')
        self.assertEqual(Reminder.query.count(),0)
        self.assertEqual(accepted_items(self.user_id).count(),0)

    def test_revalidates_conflicts_even_without_data_version_change(self):
        self.task();self.draft();plan=Plan.query.one();item=plan.items[0]
        db.session.add(CalendarEvent(user_id=self.user_id,title='Conflict',starts_at=item.planned_start,ends_at=item.planned_end))
        db.session.commit()
        self.assertEqual(self.post(f'/recommendations/{plan.id}/accept').status_code,409)
        self.assertEqual(accepted_items(self.user_id).count(),0)

    def test_expired_start_rejects(self):
        self.task();self.draft();plan=Plan.query.one()
        with patch('app.routes.utcnow',return_value=datetime(2030,1,7,7,1)):
            self.assertEqual(self.post(f'/recommendations/{plan.id}/accept').status_code,409)
        self.assertEqual(accepted_items(self.user_id).count(),0)

    def test_snapshot_changes_detected_even_without_data_version(self):
        task=self.task();self.draft();plan=Plan.query.one()
        task.importance=1;db.session.commit()
        self.assertEqual(self.post(f'/recommendations/{plan.id}/accept').status_code,409)

    def test_changed_model_version_rejects_old_draft(self):
        self.task();self.draft();plan=Plan.query.one()
        plan.parameters_json=dict(plan.parameters_json,model_version='old');db.session.commit()
        self.assertEqual(self.post(f'/recommendations/{plan.id}/accept').status_code,409)
        self.assertEqual(accepted_items(self.user_id).count(),0)

    def test_evening_form_defaults_to_tomorrow(self):
        with patch('app.routes.utcnow',return_value=datetime(2030,1,7,17)):
            self.assertIn('value="2030-01-08"',self.client.get('/recommendations').text)

    def test_cancel_draft_never_changes_calendar_or_history(self):
        self.task();self.draft();plan=Plan.query.one()
        self.assertEqual(self.post(f'/recommendations/{plan.id}/dismiss').status_code,302)
        self.assertEqual(self.post(f'/recommendations/{plan.id}/accept').status_code,409)
        counts,_=historical_counts(self.user_id,NOW+timedelta(days=2))
        self.assertEqual(counts,{})
        self.assertEqual(Reminder.query.count(),0)

    def test_history_uses_snapshots_excludes_drafts_future_and_early_cancellation(self):
        good=self.history_attempt(success=True)
        # Change current category: the earlier snapshot must remain authoritative.
        second=Category.query.filter_by(name='Робота').one()
        good.task.category_id=second.id;db.session.commit()
        end=NOW-timedelta(hours=2);start=end-timedelta(hours=1)
        self.history_attempt(success=False,end=end,canceled=start-timedelta(minutes=1))
        late=self.history_attempt(success=False,end=end,canceled=start+timedelta(minutes=10))
        self.history_attempt(success=True,accepted=False)
        self.history_attempt(success=True,end=NOW+timedelta(hours=1))
        self.history_attempt(success=True,owner=self.other_id)
        counts,_=historical_counts(self.user_id,NOW)
        self.assertEqual(sum(c.total for c in counts.values()),2)
        self.assertEqual(sum(c.successes for c in counts.values()),1)
        self.assertTrue(all(key[0]==self.category_id for key in counts))
        self.assertEqual(late.plan.status,'canceled')

    def test_history_cutoff_rejects_later_acceptance_and_future_completion(self):
        attempt=self.history_attempt(success=False)
        attempt.actual_end=NOW+timedelta(hours=1)
        db.session.commit()
        counts,_=historical_counts(self.user_id,NOW)
        self.assertEqual(sum(c.successes for c in counts.values()),0)
        attempt.plan.accepted_at=NOW+timedelta(hours=2);db.session.commit()
        self.assertEqual(historical_counts(self.user_id,NOW)[0],{})

    def test_archived_completed_and_other_users_tasks_never_recommended(self):
        self.task(title='Eligible')
        self.task(title='Archived',archived_at=NOW)
        self.task(title='Completed',status='completed')
        self.task(title='Private',user_id=self.other_id)
        self.draft()
        plan=Plan.query.one()
        self.assertEqual([i.task.title for i in plan.items],['Eligible'])
        self.assertNotIn('Private',self.client.get(f'/recommendations/{plan.id}').text)

    def test_already_scheduled_task_excluded_and_busy_event_respected(self):
        task=self.task();self.draft();plan=Plan.query.one()
        self.post(f'/recommendations/{plan.id}/accept')
        self.task(title='New')
        self.draft()
        newer=Plan.query.order_by(Plan.id.desc()).first()
        self.assertEqual(len(newer.items),1)
        self.assertEqual(newer.parameters_json['excluded'][0]['task_id'],task.id)
        self.assertGreaterEqual(newer.items[0].planned_start,plan.items[0].planned_end)

    def test_no_tasks_and_infeasible_draft_have_clear_explanation(self):
        self.draft();plan=Plan.query.one()
        self.assertEqual(len(plan.items),0)
        self.assertEqual(self.post(f'/recommendations/{plan.id}/accept').status_code,400)
        self.task(deadline=NOW-timedelta(hours=1))
        self.draft();new=Plan.query.order_by(Plan.id.desc()).first()
        self.assertIn('Граничний строк',self.client.get(f'/recommendations/{new.id}').text)

    def test_ownership_and_csrf(self):
        self.task(user_id=self.other_id)
        other= db.session.get(User,self.other_id)
        plan=create_draft(other,WINDOWS,now=NOW);db.session.commit()
        self.assertEqual(self.client.get(f'/recommendations/{plan.id}').status_code,403)
        for action in ('accept','dismiss'):
            self.assertEqual(self.post(f'/recommendations/{plan.id}/{action}').status_code,403)
        self.assertEqual(self.client.post('/recommendations',data={}).status_code,400)

    def test_invalid_window_and_parameter_forms_do_not_write(self):
        cases=[dict(first_day='bad'),dict(days='0'),dict(days='8'),dict(days='many'),dict(start_clock='10:07'),
               dict(start_clock='19:00',end_clock='09:00'),dict(reminder_minutes='-1'),
               dict(first_day='2100-12-31',days='2'),dict(first_day='2029-01-07')]
        for values in cases:
            self.assertEqual(self.draft(**values).status_code,400)
        self.assertEqual(Plan.query.count(),0)

    def test_dst_hours_rejected_and_summer_windows_converted(self):
        windows=available_windows('2030-07-07','1','09:00','18:00','Europe/Kyiv')
        self.assertEqual(windows[0][0],datetime(2030,7,7,6))
        with self.assertRaises(ValueError): available_windows('2030-03-31','1','03:00','05:00','Europe/Kyiv')

if __name__=='__main__':
    unittest.main()
