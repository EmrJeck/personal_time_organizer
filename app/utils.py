from datetime import timedelta
from flask import current_app
from flask_mail import Message
from app import db, mail
from app.models import Task, Plan, PlanItem, Reminder, utcnow

def mail_enabled():
    return current_app.config['MAIL_ENABLED'] and not current_app.config['MAIL_SUPPRESS_SEND']

def send_reminder_email(to, subject, body):
    if not mail_enabled():
        return False
    mail.send(Message(subject, recipients=[to], body=body))
    return True

def check_and_send_reminders(app, now=None):
    """Explicit single-worker batch. No scheduler runs at startup; now is naive UTC."""
    with app.app_context():
        if not mail_enabled():
            return {'enabled': False, 'sent': 0, 'failed': 0}
        now = now or utcnow()
        reminders = Reminder.query.join(PlanItem).join(Plan).join(Task, PlanItem.task_id == Task.id).filter(
            Reminder.status.in_(('pending', 'failed')), Reminder.remind_at <= now,
            Plan.status == 'accepted', PlanItem.status == 'scheduled',
            PlanItem.planned_start >= now - timedelta(minutes=5), PlanItem.planned_end > now,
            Task.status == 'pending', Task.archived_at.is_(None)).all()
        sent = failed = 0
        for reminder in reminders:
            reminder_id = reminder.id
            try:
                reminder.attempt_count += 1
                delivered = send_reminder_email(reminder.plan_item.plan.user.email,
                    f'Нагадування: {reminder.plan_item.task.title}',
                    f'Наближається час виконання задачі: {reminder.plan_item.task.title}')
                if delivered:
                    reminder.status = 'sent'
                    reminder.sent_at = now
                    db.session.commit()
                    sent += 1
                else:
                    db.session.rollback()
            except Exception:
                db.session.rollback()
                persisted = db.session.get(Reminder, reminder_id)
                persisted.status = 'failed'
                persisted.attempt_count += 1
                db.session.commit()
                failed += 1
                app.logger.warning('Не вдалося надіслати або зберегти нагадування.')
        return {'enabled': True, 'sent': sent, 'failed': failed}
