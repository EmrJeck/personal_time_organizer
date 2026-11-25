"""Explicit, atomic SQLite migration with an online backup before legacy upgrade."""
from datetime import datetime, timezone
from contextlib import closing
from pathlib import Path
import sqlite3
from zoneinfo import ZoneInfo
from sqlalchemy import inspect, text
from app import db
from app.time_utils import local_to_utc

SCHEMA_VERSION = 1
DEFAULT_CATEGORIES = ('Без категорії', 'Робота', 'Навчання', 'Особисте', 'Інше')

def legacy_date(raw, local=False):
    if not raw:
        return None
    value = datetime.fromisoformat(raw) if isinstance(raw, str) else raw
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    if not local:
        return value
    # Old forms saved Kyiv wall time. Refuse ambiguous values instead of guessing.
    converted = local_to_utc(value.strftime('%Y-%m-%dT%H:%M'), ZoneInfo('Europe/Kyiv'), validate_range=False)
    return converted.replace(second=value.second, microsecond=value.microsecond)

def backup_database():
    path = Path(db.engine.url.database)
    backup_dir = path.parent / 'backups'
    backup_dir.mkdir(parents=True, exist_ok=True)
    target = backup_dir / ('before_schema_v1_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.db')
    with closing(sqlite3.connect(str(path))) as source, closing(sqlite3.connect(str(target))) as dest:
        source.backup(dest)
    return target

def initialize_database():
    from app.models import User, Category, Task
    db.session.remove()
    existing = set(inspect(db.engine).get_table_names())
    with db.engine.connect() as conn:
        version = conn.exec_driver_sql('PRAGMA user_version').scalar()
    if version > SCHEMA_VERSION:
        raise RuntimeError('База створена новішою версією програми; оновлення скасовано.')
    if 'user' in existing and 'users' in existing:
        raise RuntimeError('Одночасно виявлено стару й нову структури. Потрібна перевірка бази.')
    backup = backup_database() if 'user' in existing else None
    with db.engine.connect() as conn:
        conn.exec_driver_sql('BEGIN IMMEDIATE')
        try:
            db.metadata.create_all(bind=conn)
            names = list(conn.execute(db.select(Category.name)).scalars())
            for name in DEFAULT_CATEGORIES:
                if name not in names:
                    conn.execute(Category.__table__.insert().values(name=name))
            category_ids = dict(conn.execute(db.select(Category.name, Category.id)).tuples().all())
            if 'user' in existing:
                old_users = conn.execute(text('SELECT * FROM user')).mappings().all()
                old_tasks = conn.execute(text('SELECT * FROM task')).mappings().all()
                old_tags = conn.execute(text('SELECT * FROM tag')).mappings().all() if 'tag' in existing else []
                links = conn.execute(text('SELECT * FROM task_tags')).mappings().all() if 'task_tags' in existing else []
                tag_names = {row['id']: row['name'] for row in old_tags}
                for row in old_tags:
                    if row['name'] not in category_ids:
                        result = conn.execute(Category.__table__.insert().values(name=row['name']))
                        category_ids[row['name']] = result.inserted_primary_key[0]
                for row in old_users:
                    conn.execute(User.__table__.insert().values(
                        id=row['id'], username=row['username'], email=row['email'],
                        password_hash=row['password'], timezone='Europe/Kyiv', settings_json={}, data_version=1))
                for row in old_tasks:
                    labels = sorted({tag_names[link['tag_id']] for link in links if link['task_id'] == row['id']})
                    category = labels[0] if len(labels) == 1 else 'Без категорії'
                    conn.execute(Task.__table__.insert().values(
                        id=row['id'], user_id=row['user_id'], category_id=category_ids[category],
                        title=row['title'], description=row['description'] or '', importance=3, duration_min=30,
                        created_at=legacy_date(row['created_at']) or datetime.now(timezone.utc).replace(tzinfo=None),
                        deadline=legacy_date(row.get('due_date'), local=True),
                        status='completed' if row['is_completed'] else 'pending',
                        legacy_tags_json=labels, legacy_remind_at=legacy_date(row.get('remind_at'), local=True),
                        legacy_reminder_sent_at=legacy_date(row.get('reminder_sent_at'))))
                if conn.execute(db.select(db.func.count()).select_from(Task)).scalar() != len(old_tasks):
                    raise RuntimeError('Кількість перенесених задач не збігається.')
                for table in ('task_tags', 'task', 'tag', 'user', 'alembic_version'):
                    if table in existing:
                        conn.exec_driver_sql('DROP TABLE "' + table + '"')
            violations = conn.exec_driver_sql('PRAGMA foreign_key_check').all()
            if violations:
                raise RuntimeError('Порушено зв’язки між таблицями; міграцію скасовано.')
            conn.exec_driver_sql(f'PRAGMA user_version = {SCHEMA_VERSION}')
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return backup
