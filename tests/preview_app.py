"""Disposable UI-check data; run from the project root: python -m tests.preview_app."""
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from zoneinfo import ZoneInfo
from werkzeug.security import generate_password_hash
from app import create_app, db
from app.database import initialize_database
from app.models import User, Task, Category, CalendarEvent, utcnow
from app.planning import schedule_task
from app.time_utils import local_to_utc

if __name__ == '__main__':
    with TemporaryDirectory(prefix='organizer_ui_check_') as folder:
        app = create_app({'SECRET_KEY': 'disposable-ui-check', 'SESSION_COOKIE_NAME': 'organizer_ui_check',
                          'SQLALCHEMY_DATABASE_URI': 'sqlite:///' + str(Path(folder)/'preview.db')})
        with app.app_context():
            initialize_database()
            user = User(username='Перевірка', email='ui-check@example.com', password_hash=generate_password_hash('Ui-check-only-123'))
            db.session.add(user); db.session.flush()
            category = Category.query.filter_by(name='Навчання').one()
            local_now = datetime.now(ZoneInfo('Europe/Kyiv'))
            day = (local_now + timedelta(days=1)).date().isoformat()
            def at(clock): return local_to_utc(day+'T'+clock, ZoneInfo('Europe/Kyiv'))
            for title, duration, importance, clock in [('Підготувати підрозділ 3.1',60,4,'10:00'),
                                                       ('Опрацювати джерела',40,3,'14:00')]:
                task = Task(user_id=user.id, category=category, title=title, duration_min=duration,
                            importance=importance, deadline=at('18:00'))
                db.session.add(task); db.session.flush()
                schedule_task(user, task, at(clock), 10)
            db.session.add_all([CalendarEvent(user_id=user.id,title='Зустріч із керівником',starts_at=at('09:00'),ends_at=at('10:00')),
                               CalendarEvent(user_id=user.id,title='Обідня перерва',event_type='break',starts_at=at('12:00'),ends_at=at('12:30')),
                               Task(user_id=user.id,category=category,title='Перевірити календар',duration_min=30,importance=3)])
            db.session.commit()
        try:
            app.run(host='127.0.0.1',port=5002,debug=False,use_reloader=False)
        finally:
            with app.app_context():
                db.session.remove(); db.engine.dispose()
