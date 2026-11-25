"""Database adapter for the pure recommender, draft lifecycle, immutable explanations."""
from collections import defaultdict
from dataclasses import asdict
from datetime import date, datetime, time, timedelta
from time import perf_counter
from zoneinfo import ZoneInfo
from app import db
from app.models import Task, Plan, PlanItem, CalendarEvent, Reminder, utcnow
from app.planning import accepted_items, active_items, ensure_free, touch
from app.recommender import (TaskInput, HistoryCount, generate_plan, daypart_of, allocated_minutes,
                             VERSION, RULE_TABLE, DAYPART_LABELS, METHODS, RECENT_METHODS,
                             HistoryObservation, recency_counts, RECENCY_HALF_LIFE_DAYS)
from app.time_utils import local_to_utc

class StalePlanError(ValueError):
    pass

def available_windows(first_day, days, start_clock, end_clock, timezone_name):
    try:
        first = date.fromisoformat(first_day)
        count = int(days)
        start = time.fromisoformat(start_clock)
        end = time.fromisoformat(end_clock)
        zone = ZoneInfo(timezone_name)
    except (ValueError, TypeError, KeyError):
        raise ValueError('Перевірте дату, кількість днів і часовий пояс.') from None
    if not 1 <= count <= 7 or not 1900 <= first.year <= 2100:
        raise ValueError('Оберіть від 1 до 7 днів у межах 1900–2100 років.')
    if (start.tzinfo or end.tzinfo or start.second or end.second or start.microsecond or end.microsecond or
        start.minute % 15 or end.minute % 15 or end <= start):
        raise ValueError('Вкажіть початок і кінець одного дня з кроком 15 хвилин; завершення має бути пізніше.')
    windows = []
    for offset in range(count):
        day = first + timedelta(days=offset)
        if day.year > 2100:
            raise ValueError('Період виходить за межі 2100 року.')
        windows.append((local_to_utc(f'{day}T{start:%H:%M}',zone),
                        local_to_utc(f'{day}T{end:%H:%M}',zone)))
    return windows

def task_snapshot(task, timezone_name):
    return {'title':task.title, 'category_id':task.category_id, 'category':task.category.name,
            'importance':task.importance, 'duration_min':task.duration_min,
            'allocated_min':allocated_minutes(task.duration_min), 'timezone':timezone_name,
            'earliest_start':task.earliest_start.isoformat() if task.earliest_start else None,
            'deadline':task.deadline.isoformat() if task.deadline else None}

def _historical_observations(user_id, as_of):
    """Read accepted attempts whose planned end is strictly before the cutoff.
    A canceled plan may contain a late-canceled attempt and must not disappear.
    Missing completion means no recorded on-time completion, not a proven failure.
    """
    if not isinstance(as_of, datetime) or as_of.tzinfo is not None:
        raise ValueError('Для історії потрібен час у UTC без tzinfo.')
    attempts = PlanItem.query.join(Plan).filter(
        Plan.user_id == user_id, Plan.accepted_at.is_not(None), Plan.accepted_at <= as_of,
        Plan.created_at <= as_of, PlanItem.planned_end < as_of).all()
    observations = []
    excluded = 0
    for item in attempts:
        if item.canceled_at is not None and item.canceled_at < item.planned_start:
            continue
        snapshot = item.task_snapshot_json or {}
        category_id = snapshot.get('category_id')
        if isinstance(category_id,bool) or not isinstance(category_id,int) or category_id <= 0:
            excluded += 1
            continue
        # Legacy assignments were always in Kyiv; never infer from edited tasks.
        zone_name = snapshot.get('timezone','Europe/Kyiv')
        part = snapshot.get('daypart') or daypart_of(item.planned_start,zone_name)
        if part not in DAYPART_LABELS:
            excluded += 1
            continue
        observations.append(HistoryObservation(category_id, part, item.planned_end,
                            item.actual_end is not None and item.actual_end <= item.planned_end))
    return observations, excluded


def historical_counts(user_id, as_of):
    """Legacy counts, using the same eligibility as the recent mode."""
    observations, excluded = _historical_observations(user_id, as_of)
    counts = defaultdict(lambda:[0,0])
    for observation in observations:
        entry = counts[(observation.category_id, observation.daypart)]
        entry[1] += 1
        entry[0] += int(observation.success)
    return {key:HistoryCount(value[0],value[1]) for key,value in counts.items()}, excluded

def historical_recency_counts(user_id, as_of, half_life_days=RECENCY_HALF_LIFE_DAYS):
    observations, excluded = _historical_observations(user_id, as_of)
    return recency_counts(observations, as_of, half_life_days), excluded


def create_draft(user, windows, method='personalized_recent', reminder_minutes=None, now=None):
    now = now or utcnow()
    if method not in METHODS:
        raise ValueError('Невідомий метод планування.')
    if reminder_minutes is not None and reminder_minutes not in (0,5,10,15,30,60):
        raise ValueError('Оберіть допустимий час нагадування.')
    if windows[-1][1] <= now:
        raise ValueError('Обраний час уже минув. Виберіть майбутній період.')
    current = accepted_items(user.id).all()
    already_scheduled = {item.task_id for item in current if item.status in ('scheduled','in_progress')}
    all_pending = Task.query.filter_by(user_id=user.id,status='pending').filter(
        Task.archived_at.is_(None)).order_by(Task.id).all()
    tasks = [task for task in all_pending if task.id not in already_scheduled]
    excluded = [dict(task_id=task.id,title=task.title,reason='Задача вже має прийняте призначення в календарі.')
                for task in all_pending if task.id in already_scheduled]
    inputs = [TaskInput(task.id,task.title,task.category_id,task.importance,task.duration_min,
                        task.earliest_start,task.deadline) for task in tasks]
    events = CalendarEvent.query.filter(CalendarEvent.user_id == user.id,
        CalendarEvent.starts_at < windows[-1][1],CalendarEvent.ends_at > windows[0][0]).all()
    busy = [(event.starts_at,event.ends_at) for event in events]
    busy += [(item.planned_start,item.planned_end) for item in current
             if item.planned_start < windows[-1][1] and item.planned_end > windows[0][0]]
    history, ignored_history = (historical_recency_counts(user.id,now)
                               if method in RECENT_METHODS else historical_counts(user.id,now))
    started = perf_counter()
    result = generate_plan(inputs,windows,busy,history,as_of=now,timezone_name=user.timezone,method=method)
    elapsed = (perf_counter()-started)*1000
    by_id = {task.id:task for task in tasks}
    params = {
        'mode':'recommendation', 'method':method, 'model_version':VERSION,
        'as_of':now.isoformat(), 'timezone':user.timezone, 'slot_minutes':15,
        'history_weight':.2, 'prior_strength':2, 'prior_mean':.5, 'linear_weights':[.6,.4],
        'history_half_life_days': RECENCY_HALF_LIFE_DAYS if method in RECENT_METHODS else None,
        'history_age_timestamp': 'planned_end' if method in RECENT_METHODS else None,
        'rule_table':[list(row) for row in RULE_TABLE],
        'dayparts':{'night':[0,6],'morning':[6,12],'afternoon':[12,18],'evening':[18,24]},
        'available_windows':[[a.isoformat(),b.isoformat()] for a,b in windows],
        'candidate_count':result.candidate_count, 'elapsed_ms':elapsed, 'input_task_count':len(tasks),
        'history_observations':sum(count.total for count in history.values()),
        'history_weighted_observations':sum(count.weighted_total for count in history.values()) if method in RECENT_METHODS else None,
        'ignored_history':ignored_history, 'reminder_minutes':reminder_minutes,
        'unscheduled':[dict(task_id=r.task_id,title=by_id[r.task_id].title,reason=r.reason)
                       for r in result.unscheduled], 'excluded':excluded,
    }
    plan = Plan(user_id=user.id,created_at=now,horizon_start=windows[0][0],horizon_end=windows[-1][1],
                status='draft',parameters_json=params,source_version=user.data_version)
    db.session.add(plan)
    for assignment in result.assignments:
        snapshot = task_snapshot(by_id[assignment.task_id],user.timezone)
        snapshot['daypart'] = assignment.daypart
        explanation = asdict(assignment)
        for key in ('start','end'):
            explanation[key] = explanation[key].isoformat()
        snapshot['recommendation'] = explanation
        db.session.add(PlanItem(plan=plan,task=by_id[assignment.task_id],
                               planned_start=assignment.start,planned_end=assignment.end,
                               priority_score=assignment.score,task_snapshot_json=snapshot))
    return plan

def draft_problem(plan, user, now=None):
    now = now or utcnow()
    if plan.status != 'draft':
        return None
    if plan.parameters_json.get('model_version') != VERSION:
        return 'Модель планування оновлено. Сформуйте новий проєкт плану.'
    if plan.source_version != user.data_version:
        return 'Задачі або календар змінилися. Сформуйте новий проєкт плану.'
    if plan.parameters_json.get('timezone') != user.timezone:
        return 'Часовий пояс змінився. Сформуйте новий проєкт плану.'
    if any(item.planned_start < now for item in plan.items):
        return 'Час початку однієї з рекомендацій уже минув. Сформуйте новий проєкт плану.'
    return None

def accept_draft(user, plan, now=None):
    now = now or utcnow()
    if plan.user_id != user.id:
        raise ValueError('Немає доступу до цього плану.')
    if plan.status == 'accepted':
        return False  # Repeated submission is harmless.
    if plan.status != 'draft' or plan.parameters_json.get('mode') != 'recommendation':
        raise StalePlanError('Цей проєкт плану вже скасовано або він непридатний до прийняття.')
    problem = draft_problem(plan,user,now)
    if problem:
        raise StalePlanError(problem)
    if not plan.items:
        raise ValueError('У проєкті немає задач для додавання до календаря.')
    windows = [(datetime.fromisoformat(a),datetime.fromisoformat(b))
               for a,b in plan.parameters_json['available_windows']]
    new_intervals = []
    for item in sorted(plan.items,key=lambda i:i.planned_start):
        task = item.task
        if task.user_id != user.id or task.archived_at or task.status != 'pending' or active_items(task):
            raise StalePlanError('Стан задач змінився. Сформуйте новий проєкт плану.')
        current_snapshot = task_snapshot(task,user.timezone)
        if any(item.task_snapshot_json.get(key) != value for key,value in current_snapshot.items()):
            raise StalePlanError('Параметри задачі змінилися. Сформуйте новий проєкт плану.')
        if item.status != 'scheduled' or item.planned_end-item.planned_start != timedelta(minutes=allocated_minutes(task.duration_min)):
            raise StalePlanError('Некоректна тривалість призначення. Сформуйте план знову.')
        if (task.earliest_start and item.planned_start < task.earliest_start) or (task.deadline and item.planned_end > task.deadline):
            raise StalePlanError('Призначення не відповідає часовим обмеженням задачі.')
        if not any(a <= item.planned_start and item.planned_end <= b for a,b in windows):
            raise StalePlanError('Призначення виходить за обрані години планування.')
        if any(item.planned_start < b and item.planned_end > a for a,b in new_intervals):
            raise StalePlanError('У проєкті є перетин задач. Сформуйте план знову.')
        try:
            ensure_free(user.id,item.planned_start,item.planned_end)
        except ValueError as error:
            raise StalePlanError(str(error)) from None
        new_intervals.append((item.planned_start,item.planned_end))
    # All checks succeeded; only now change persistent state.
    plan.status = 'accepted'
    plan.accepted_at = now
    minutes = plan.parameters_json.get('reminder_minutes')
    if minutes is not None:
        for item in plan.items:
            db.session.add(Reminder(plan_item=item,remind_at=item.planned_start-timedelta(minutes=minutes)))
    touch(user)
    return True

def dismiss_draft(user, plan, now=None):
    if plan.user_id != user.id or plan.status != 'draft':
        raise ValueError('Скасувати можна лише власний проєкт плану.')
    plan.status = 'canceled'
    for item in plan.items:
        item.status='canceled'
        item.canceled_at=now or utcnow()
