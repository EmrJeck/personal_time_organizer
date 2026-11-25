"""Manual planning with the same hard constraints required by the future recommender."""
from datetime import timedelta
from app import db
from app.models import CalendarEvent, Plan, PlanItem, Reminder, utcnow

def touch(user):
    user.data_version += 1

def accepted_items(user_id):
    return PlanItem.query.join(Plan).filter(
        Plan.user_id == user_id, Plan.status == 'accepted', PlanItem.status != 'canceled')

def active_items(task):
    return accepted_items(task.user_id).filter(
        PlanItem.task_id == task.id, PlanItem.status.in_(('scheduled', 'in_progress'))).all()

def ensure_free(user_id, start, end, exclude_item=None, exclude_event=None):
    if end <= start:
        raise ValueError('Завершення має бути пізніше за початок.')
    events = CalendarEvent.query.filter(
        CalendarEvent.user_id == user_id, CalendarEvent.starts_at < end, CalendarEvent.ends_at > start)
    if exclude_event is not None:
        events = events.filter(CalendarEvent.id != exclude_event)
    if events.first():
        raise ValueError('Цей час перетинається з подією або перервою в календарі.')
    items = accepted_items(user_id).filter(PlanItem.planned_start < end, PlanItem.planned_end > start)
    if exclude_item is not None:
        items = items.filter(PlanItem.id != exclude_item)
    if items.first():
        raise ValueError('Цей час уже зайнятий запланованою задачею.')

def cancel_item(item):
    if item.status != 'scheduled':
        raise ValueError('Скасувати можна лише задачу, виконання якої ще не почалося.')
    item.status = 'canceled'
    item.canceled_at = utcnow()
    for reminder in item.reminders:
        if reminder.status in ('pending', 'failed'):
            reminder.status = 'canceled'
    if all(other.status == 'canceled' for other in item.plan.items):
        item.plan.status = 'canceled'

def schedule_task(user, task, start, reminder_minutes=None):
    if task.archived_at or task.status != 'pending':
        raise ValueError('Планувати можна лише активну невиконану задачу.')
    if start is None or start < utcnow():
        raise ValueError('Оберіть майбутній час початку.')
    if start.minute % 15 or start.second or start.microsecond:
        raise ValueError('Початок задачі має бути кратним 15 хвилинам: 00, 15, 30 або 45.')
    duration = ((task.duration_min + 14) // 15) * 15
    end = start + timedelta(minutes=duration)
    if task.earliest_start and start < task.earliest_start:
        raise ValueError('Початок раніше за дозволений час для цієї задачі.')
    if task.deadline and end > task.deadline:
        raise ValueError('Задача не встигає завершитися до граничного строку.')
    old_items = active_items(task)
    if len(old_items) > 1 or any(item.status != 'scheduled' for item in old_items):
        raise ValueError('Поточний стан плану не дозволяє перенесення.')
    old_item = old_items[0] if old_items else None
    ensure_free(user.id, start, end, exclude_item=old_item.id if old_item else None)
    if reminder_minutes is not None and reminder_minutes not in (0, 5, 10, 15, 30, 60):
        raise ValueError('Оберіть допустимий час нагадування.')
    # Validate everything before canceling the earlier assignment.
    if old_item:
        cancel_item(old_item)
    plan = Plan(user_id=user.id, accepted_at=utcnow(), horizon_start=start, horizon_end=end,
                status='accepted', source_version=user.data_version,
                parameters_json={'mode': 'manual', 'slot_minutes': 15})
    item = PlanItem(plan=plan, task=task, planned_start=start, planned_end=end,
                    task_snapshot_json={'title': task.title, 'category_id': task.category_id,
                                        'timezone': user.timezone,
                                        'category': task.category.name, 'importance': task.importance,
                                        'duration_min': task.duration_min, 'allocated_min': duration,
                                        'earliest_start': task.earliest_start.isoformat() if task.earliest_start else None,
                                        'deadline': task.deadline.isoformat() if task.deadline else None})
    db.session.add(plan)
    if reminder_minutes is not None:
        db.session.add(Reminder(plan_item=item, remind_at=start - timedelta(minutes=reminder_minutes)))
    touch(user)
    return item
