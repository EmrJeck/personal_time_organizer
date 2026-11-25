"""Disposable demonstration, never writes to the user's database. Port 5002."""
from datetime import datetime, time, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo
from werkzeug.security import generate_password_hash
from app import create_app, db
from app.database import initialize_database
from app.models import User, Category, Task, CalendarEvent, Plan, PlanItem, utcnow
from app.recommendations import task_snapshot
from app.time_utils import local_to_utc


def prepare_demo(folder):
    app = create_app({'SECRET_KEY': 'disposable-analytics-preview', 'DEMO_MODE': True, 'TEMPLATES_AUTO_RELOAD': True,
                      'SESSION_COOKIE_NAME': 'organizer_analytics_preview',
                      'SQLALCHEMY_DATABASE_URI': 'sqlite:///'+str(Path(folder)/'preview.db')})
    with app.app_context():
        initialize_database()
        user = User(username='Демонстрація', email='ui-check@example.com',
                    password_hash=generate_password_hash('Ui-check-only-123'))
        db.session.add(user)
        db.session.flush()
        categories = {c.name: c for c in Category.query.all()}
        today = datetime.now(ZoneInfo(user.timezone)).date()

        def at(day, clock):
            return local_to_utc(f'{day.isoformat()}T{clock}', ZoneInfo(user.timezone))

        # Explicit synthetic history for UI inspection, not efficacy evidence.
        examples = [(-6, '09:00', 'Навчання', 60, 'on_time'),
                    (-6, '14:00', 'Робота', 45, 'late'),
                    (-5, '09:00', 'Навчання', 45, 'missing'),
                    (-5, '14:00', 'Навчання', 60, 'on_time'),
                    (-4, '09:00', 'Робота', 60, 'on_time'),
                    (-4, '14:00', 'Навчання', 45, 'on_time'),
                    (-3, '09:00', 'Навчання', 30, 'missing'),
                    (-3, '14:00', 'Робота', 60, 'on_time')]
        for i, (offset, clock, category_name, duration, outcome) in enumerate(examples, 1):
            start = at(today+timedelta(days=offset), clock)
            end = start+timedelta(minutes=duration)
            task = Task(user_id=user.id, category=categories[category_name], title=f'Демонстраційна задача {i}',
                        duration_min=duration, importance=3,
                        status='completed' if outcome != 'missing' else 'pending', archived_at=utcnow())
            plan = Plan(user_id=user.id, created_at=start-timedelta(days=1),
                        accepted_at=start-timedelta(hours=2), horizon_start=start, horizon_end=end,
                        status='accepted', parameters_json={'mode': 'manual'}, source_version=1)
            item = PlanItem(plan=plan, task=task, planned_start=start, planned_end=end,
                            actual_start=start+timedelta(minutes=5) if outcome != 'missing' else None,
                            actual_end=(end if outcome == 'on_time' else end+timedelta(minutes=15)) if outcome != 'missing' else None,
                            status='completed' if outcome != 'missing' else 'scheduled',
                            task_snapshot_json=task_snapshot(task, user.timezone))
            # Snapshot category_id is available after flushing new task relationships.
            db.session.add_all([task, plan, item]); db.session.flush()
            item.task_snapshot_json = task_snapshot(task, user.timezone)
        tomorrow = today+timedelta(days=1)
        for title, duration, importance, category_name in [
                ('Написати підрозділ 3.2', 60, 5, 'Навчання'),
                ('Опрацювати наукові джерела', 45, 4, 'Навчання'),
                ('Підготувати робочі матеріали', 30, 3, 'Робота')]:
            db.session.add(Task(user_id=user.id, category=categories[category_name], title=title,
                                duration_min=duration, importance=importance, deadline=at(tomorrow, '18:00')))
        db.session.add_all([
            CalendarEvent(user_id=user.id, title='Зустріч', starts_at=at(tomorrow, '10:00'), ends_at=at(tomorrow, '11:00')),
            CalendarEvent(user_id=user.id, title='Обідня перерва', event_type='break', starts_at=at(tomorrow, '12:00'), ends_at=at(tomorrow, '12:30'))])
        db.session.commit()
    return app


if __name__ == '__main__':
    with TemporaryDirectory(prefix='organizer_analytics_demo_') as folder:
        app = prepare_demo(folder)
        try:
            app.run(host='127.0.0.1', port=5002, debug=False, use_reloader=False)
        finally:
            with app.app_context():
                db.session.remove(); db.engine.dispose()
