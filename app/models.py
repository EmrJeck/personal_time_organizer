"""Seven core entities from thesis subsection 2.3. Datetimes are naive UTC."""
from datetime import datetime, timezone
from flask_login import UserMixin
from app import db

def utcnow():
    return datetime.now(timezone.utc).replace(tzinfo=None)

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    email = db.Column(db.String(150), nullable=False, unique=True)
    password_hash = db.Column(db.String(255), nullable=False)
    timezone = db.Column(db.String(64), nullable=False, default='Europe/Kyiv')
    settings_json = db.Column(db.JSON, nullable=False, default=dict)
    data_version = db.Column(db.Integer, nullable=False, default=1)
    tasks = db.relationship('Task', back_populates='author')

class Category(db.Model):
    __tablename__ = 'categories'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False, unique=True)

class Task(db.Model):
    __tablename__ = 'tasks'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    category_id = db.Column(db.Integer, db.ForeignKey('categories.id'), nullable=False)
    title = db.Column(db.String(140), nullable=False)
    description = db.Column(db.Text, nullable=False, default='')
    importance = db.Column(db.Integer, nullable=False, default=3)
    duration_min = db.Column(db.Integer, nullable=False, default=30)
    earliest_start = db.Column(db.DateTime)
    deadline = db.Column(db.DateTime)
    status = db.Column(db.String(20), nullable=False, default='pending')
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    archived_at = db.Column(db.DateTime)
    # Preserve original many-to-many labels and reminder metadata during migration.
    legacy_tags_json = db.Column(db.JSON, nullable=False, default=list)
    legacy_remind_at = db.Column(db.DateTime)
    legacy_reminder_sent_at = db.Column(db.DateTime)
    author = db.relationship('User', back_populates='tasks')
    category = db.relationship('Category')
    plan_items = db.relationship('PlanItem', back_populates='task')
    __table_args__ = (
        db.CheckConstraint('importance BETWEEN 1 AND 5', name='task_importance'),
        db.CheckConstraint('duration_min BETWEEN 1 AND 10080', name='task_duration'),
        db.CheckConstraint("status IN ('pending','in_progress','completed')", name='task_status'),
        db.CheckConstraint('earliest_start IS NULL OR deadline IS NULL OR earliest_start < deadline', name='task_window'),
    )

class CalendarEvent(db.Model):
    __tablename__ = 'calendar_events'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    title = db.Column(db.String(140), nullable=False)
    event_type = db.Column(db.String(20), nullable=False, default='fixed')
    starts_at = db.Column(db.DateTime, nullable=False, index=True)
    ends_at = db.Column(db.DateTime, nullable=False)
    __table_args__ = (
        db.CheckConstraint('starts_at < ends_at', name='event_interval'),
        db.CheckConstraint("event_type IN ('fixed','break')", name='event_type'),
    )

class Plan(db.Model):
    __tablename__ = 'plans'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False, index=True)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    accepted_at = db.Column(db.DateTime)
    horizon_start = db.Column(db.DateTime, nullable=False)
    horizon_end = db.Column(db.DateTime, nullable=False)
    status = db.Column(db.String(20), nullable=False, default='draft')
    parameters_json = db.Column(db.JSON, nullable=False, default=dict)
    source_version = db.Column(db.Integer, nullable=False)
    user = db.relationship('User')
    items = db.relationship('PlanItem', back_populates='plan')
    __table_args__ = (
        db.CheckConstraint('horizon_start < horizon_end', name='plan_horizon'),
        db.CheckConstraint("status IN ('draft','accepted','canceled')", name='plan_status'),
    )

class PlanItem(db.Model):
    __tablename__ = 'plan_items'
    id = db.Column(db.Integer, primary_key=True)
    plan_id = db.Column(db.Integer, db.ForeignKey('plans.id'), nullable=False, index=True)
    task_id = db.Column(db.Integer, db.ForeignKey('tasks.id'), nullable=False, index=True)
    planned_start = db.Column(db.DateTime, nullable=False, index=True)
    planned_end = db.Column(db.DateTime, nullable=False)
    actual_start = db.Column(db.DateTime)
    actual_end = db.Column(db.DateTime)
    status = db.Column(db.String(20), nullable=False, default='scheduled')
    canceled_at = db.Column(db.DateTime)
    task_snapshot_json = db.Column(db.JSON, nullable=False, default=dict)
    priority_score = db.Column(db.Float)
    plan = db.relationship('Plan', back_populates='items')
    task = db.relationship('Task', back_populates='plan_items')
    reminders = db.relationship('Reminder', back_populates='plan_item')
    __table_args__ = (
        db.UniqueConstraint('plan_id', 'task_id', name='plan_task_unique'),
        db.CheckConstraint('planned_start < planned_end', name='item_interval'),
        db.CheckConstraint('actual_start IS NULL OR actual_end IS NULL OR actual_start <= actual_end', name='actual_interval'),
        db.CheckConstraint("status IN ('scheduled','in_progress','completed','canceled')", name='item_status'),
    )

class Reminder(db.Model):
    __tablename__ = 'reminders'
    id = db.Column(db.Integer, primary_key=True)
    plan_item_id = db.Column(db.Integer, db.ForeignKey('plan_items.id'), nullable=False, index=True)
    remind_at = db.Column(db.DateTime, nullable=False, index=True)
    sent_at = db.Column(db.DateTime)
    status = db.Column(db.String(20), nullable=False, default='pending')
    attempt_count = db.Column(db.Integer, nullable=False, default=0)
    plan_item = db.relationship('PlanItem', back_populates='reminders')
    __table_args__ = (
        db.CheckConstraint('attempt_count >= 0', name='reminder_attempts'),
        db.CheckConstraint("status IN ('pending','sent','failed','canceled')", name='reminder_status'),
    )
