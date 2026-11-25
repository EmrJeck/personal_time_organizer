from datetime import date, datetime, time, timedelta
from flask import render_template, redirect, url_for, flash, request, Blueprint, abort
from flask_login import login_user, current_user, logout_user, login_required
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from werkzeug.security import generate_password_hash, check_password_hash
from app import db, login_manager
from app.forms import RegistrationForm, LoginForm
from app.models import User, Task, Category, CalendarEvent, PlanItem, utcnow
from app.time_utils import local_to_utc, to_local, user_zone
from app.planning import touch, accepted_items, active_items, ensure_free, cancel_item, schedule_task

routes = Blueprint('routes', __name__)

@routes.before_request
def serialize_writes():
    # SQLite checks and writes share one reserved lock, preventing overlapping
    # bookings from simultaneous form submissions. GET requests never change data.
    if request.method == 'POST':
        db.session.execute(text('BEGIN IMMEDIATE'))

@routes.teardown_request
def release_transaction(error=None):
    # Also release the reserved lock on validation errors and early returns.
    db.session.rollback()

@routes.errorhandler(IntegrityError)
def invalid_database_write(error):
    db.session.rollback()
    return render_template('error.html', message='Не вдалося зберегти зміни: дані вже існують або порушують обмеження. Оновіть сторінку й повторіть дію.'), 409

@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, int(user_id))
    except (TypeError, ValueError):
        return None

def owned(model, record_id):
    record = db.get_or_404(model, record_id)
    owner = record.plan.user_id if isinstance(record, PlanItem) else record.user_id
    if owner != current_user.id:
        abort(403)
    return record

@routes.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('routes.index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        db.session.add(User(username=form.username.data, email=form.email.data,
                            password_hash=generate_password_hash(form.password.data)))
        db.session.commit()
        flash('Акаунт створено. Тепер увійдіть.', 'success')
        return redirect(url_for('routes.login'))
    return render_template('auth.html', form=form, registering=True)

@routes.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('routes.index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and check_password_hash(user.password_hash, form.password.data):
            login_user(user, remember=form.remember.data)
            return redirect(url_for('routes.index'))
        flash('Перевірте email і пароль.', 'danger')
    return render_template('auth.html', form=form, registering=False)

@routes.route('/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return redirect(url_for('routes.index'))

@routes.route('/')
def index():
    categories = Category.query.order_by(Category.name).all()
    category_id = request.args.get('category', type=int)
    search = request.args.get('q', '').strip()
    archived = request.args.get('archived') == '1'
    tasks, placements = [], {}
    if current_user.is_authenticated:
        query = Task.query.filter_by(user_id=current_user.id)
        query = query.filter(Task.archived_at.is_not(None) if archived else Task.archived_at.is_(None))
        if category_id:
            query = query.filter_by(category_id=category_id)
        if search:
            query = query.filter(Task.title.ilike(f'%{search}%'))
        tasks = query.order_by(Task.created_at.desc()).all()
        placements = {item.task_id: item for item in accepted_items(current_user.id).filter(
            PlanItem.status.in_(('scheduled', 'in_progress'))).all()}
    return render_template('index.html', tasks=tasks, categories=categories,
                           category_id=category_id, search=search, archived=archived, placements=placements)

def task_values():
    title = request.form.get('title', '').strip()
    if not title or len(title) > 140:
        raise ValueError('Назва має містити від 1 до 140 символів.')
    try:
        importance = int(request.form.get('importance', '3'))
        duration = int(request.form.get('duration_min', '30'))
        category_id = int(request.form.get('category_id', ''))
    except ValueError:
        raise ValueError('Перевірте категорію, важливість і тривалість.') from None
    if not 1 <= importance <= 5 or not 1 <= duration <= 10080:
        raise ValueError('Важливість: від 1 до 5. Тривалість: від 1 до 10080 хвилин.')
    if db.session.get(Category, category_id) is None:
        raise ValueError('Оберіть наявну категорію.')
    start = local_to_utc(request.form.get('earliest_start', ''))
    deadline = local_to_utc(request.form.get('deadline', ''))
    allocated = timedelta(minutes=((duration + 14) // 15) * 15)
    if start and deadline and start + allocated > deadline:
        raise ValueError('Між найранішим початком і граничним строком недостатньо часу для задачі.')
    return dict(title=title, description=request.form.get('description', '').strip(),
                importance=importance, duration_min=duration, category_id=category_id,
                earliest_start=start, deadline=deadline)

def task_form(task=None, status=200):
    return render_template('task_form.html', task=task, categories=Category.query.order_by(Category.name).all()), status

def book_task_start(task):
    start = task.earliest_start
    if start < utcnow():
        raise ValueError('Для розміщення задачі оберіть майбутній час початку.')
    if start.minute % 15 or start.second or start.microsecond:
        start = start.replace(second=0, microsecond=0) + timedelta(minutes=15-start.minute % 15)
    return schedule_task(current_user, task, start)

@routes.route('/create-task', methods=['GET', 'POST'])
@login_required
def create_task():
    if request.method == 'POST':
        try:
            values = task_values()
            task = Task(**values, user_id=current_user.id)
            db.session.add(task)
            db.session.flush()
            if task.earliest_start:
                book_task_start(task)
            else:
                touch(current_user)
        except ValueError as error:
            db.session.rollback()
            flash(str(error), 'danger')
            return task_form(status=400)
        db.session.commit()
        flash('Задачу створено.' + (' Час у календарі збережено.' if values['earliest_start'] else ''), 'success')
        return redirect(url_for('routes.index'))
    return task_form()

@routes.route('/tasks/<int:task_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_task(task_id):
    task = owned(Task, task_id)
    if task.archived_at:
        abort(409)
    if request.method == 'POST':
        try:
            values = task_values()
            timing_changed = any(values[key] != getattr(task, key) for key in ('duration_min', 'earliest_start', 'deadline'))
            items = active_items(task)
            if timing_changed and any(item.status == 'in_progress' for item in items):
                raise ValueError('Завершіть поточне виконання перед зміною часових параметрів.')
            for key, value in values.items():
                setattr(task, key, value)
            booked, canceled = False, False
            if task.status == 'pending' and task.earliest_start and (timing_changed or not items):
                book_task_start(task)
                booked = True
            else:
                if timing_changed:
                    for item in items:
                        cancel_item(item)
                        canceled = True
                touch(current_user)
        except ValueError as error:
            db.session.rollback()
            flash(str(error), 'danger')
            return task_form(task, 400)
        db.session.commit()
        flash('Задачу оновлено.' + (' Час у календарі збережено.' if booked else
              (' Попереднє розміщення скасовано.' if canceled else '')), 'success')
        return redirect(url_for('routes.index'))
    return task_form(task)

@routes.route('/tasks/<int:task_id>/delete', methods=['POST'])
@login_required
def delete_task(task_id):
    task = owned(Task, task_id)
    items = active_items(task)
    if any(item.status == 'in_progress' for item in items):
        flash('Спочатку завершіть виконання задачі.', 'danger')
        return redirect(url_for('routes.index'))
    for item in items:
        cancel_item(item)
    task.archived_at = utcnow()
    touch(current_user)
    db.session.commit()
    flash('Задачу перенесено в архів. Історію планування збережено.', 'success')
    return redirect(url_for('routes.index'))

@routes.route('/tasks/<int:task_id>/restore', methods=['POST'])
@login_required
def restore_task(task_id):
    task = owned(Task, task_id)
    task.archived_at = None
    touch(current_user)
    db.session.commit()
    flash('Задачу відновлено з архіву.', 'success')
    return redirect(url_for('routes.index'))

@routes.route('/tasks/<int:task_id>/toggle', methods=['POST'])
@login_required
def toggle_task(task_id):
    task = owned(Task, task_id)
    if task.archived_at:
        abort(409)
    if task.status == 'completed':
        task.status = 'pending'
    else:
        now = utcnow()
        for item in active_items(task):
            if item.status == 'scheduled' and item.planned_start > now:
                cancel_item(item)
            else:
                item.status = 'completed'
                item.actual_end = now
                for reminder in item.reminders:
                    if reminder.status in ('pending', 'failed'):
                        reminder.status = 'canceled'
        task.status = 'completed'
    touch(current_user)
    db.session.commit()
    flash('Статус задачі оновлено.', 'success')
    return redirect(url_for('routes.index'))

def selected_day():
    raw = request.values.get('date', '')
    try:
        selected = date.fromisoformat(raw) if raw else to_local(utcnow()).date()
        if not 1900 <= selected.year <= 2100:
            raise ValueError()
        return selected
    except ValueError:
        raise ValueError('Оберіть коректну дату між 1900 і 2100 роками.') from None

def calendar_page(status=200):
    try:
        selected = selected_day()
    except ValueError as error:
        return render_template('error.html', message=str(error)), 400
    first = selected - timedelta(days=selected.weekday())
    last = first + timedelta(days=7)
    def midnight(day):
        return local_to_utc(datetime.combine(day, time()).strftime('%Y-%m-%dT%H:%M'), validate_range=False)
    start, end = midnight(first), midnight(last)
    events = CalendarEvent.query.filter(CalendarEvent.user_id == current_user.id,
                                       CalendarEvent.starts_at < end, CalendarEvent.ends_at > start).all()
    items = accepted_items(current_user.id).filter(PlanItem.planned_start < end, PlanItem.planned_end > start).all()
    # A release time is a calendar marker, not a booking. Check accepted active
    # assignments across all weeks so a later placement does not duplicate it.
    scheduled_task_ids = accepted_items(current_user.id).filter(
        PlanItem.status.in_(('scheduled', 'in_progress'))).with_entities(PlanItem.task_id)
    starting_tasks = Task.query.filter(
        Task.user_id == current_user.id, Task.archived_at.is_(None), Task.status == 'pending',
        Task.earliest_start >= start, Task.earliest_start < end,
        ~Task.id.in_(scheduled_task_ids)).all()
    days = []
    for offset in range(7):
        day = first + timedelta(days=offset)
        day_start, day_end = midnight(day), midnight(day + timedelta(days=1))
        entries = []
        for kind, records in (('event', events), ('task', items)):
            for record in records:
                a, b = (record.starts_at, record.ends_at) if kind == 'event' else (record.planned_start, record.planned_end)
                if a < day_end and b > day_start:
                    entries.append(dict(kind=kind, record=record, sort=max(a, day_start),
                                        from_time=to_local(max(a, day_start)).strftime('%H:%M'),
                                        to_time='24:00' if b >= day_end else to_local(b).strftime('%H:%M'),
                                        continues=a < day_start or b > day_end))
        for task in starting_tasks:
            if day_start <= task.earliest_start < day_end:
                entries.append(dict(kind='available', record=task, sort=task.earliest_start,
                                    from_time=to_local(task.earliest_start).strftime('%H:%M')))
        entries.sort(key=lambda entry: entry['sort'])
        days.append(dict(date=day, entries=entries))
    default_start = datetime.combine(selected, time(9))
    local_now = to_local(utcnow()).replace(tzinfo=None)
    if selected == local_now.date() and default_start <= local_now:
        default_start = local_now.replace(second=0, microsecond=0) + timedelta(minutes=15 - local_now.minute % 15)
    tasks = Task.query.filter_by(user_id=current_user.id, status='pending').filter(Task.archived_at.is_(None)).order_by(Task.title).all()
    selected_task_id = request.values.get('task_id', type=int)
    chosen_task = next((task for task in tasks if task.id == selected_task_id), None)
    if chosen_task and chosen_task.earliest_start:
        earliest = to_local(chosen_task.earliest_start).replace(tzinfo=None)
        if earliest.date() == selected:
            default_start = max(earliest, local_now) if selected == local_now.date() else earliest
            if default_start.minute % 15 or default_start.second or default_start.microsecond:
                default_start = default_start.replace(second=0, microsecond=0) + timedelta(minutes=15-default_start.minute % 15)
    return render_template('calendar.html', days=days, selected=selected, first=first, end=last-timedelta(days=1),
                           previous=first-timedelta(days=7), following=first+timedelta(days=7),
                           today=to_local(utcnow()).date(), tasks=tasks,
                           default_start=default_start.strftime('%Y-%m-%dT%H:%M'),
                           selected_task=selected_task_id,
                           weekday_names=('Пн','Вт','Ср','Чт','Пт','Сб','Нд')), status

@routes.route('/calendar')
@login_required
def calendar():
    return calendar_page()

@routes.route('/calendar/schedule', methods=['POST'])
@login_required
def schedule():
    try:
        task_id = int(request.form.get('task_id', ''))
    except ValueError:
        flash('Оберіть задачу.', 'danger')
        return calendar_page(400)
    task = owned(Task, task_id)
    try:
        start = local_to_utc(request.form.get('planned_start', ''))
        raw = request.form.get('reminder_minutes', '')
        minutes = int(raw) if raw else None
        schedule_task(current_user, task, start, minutes)
    except (ValueError, OverflowError) as error:
        flash(str(error) or 'Перевірте час планування.', 'danger')
        return calendar_page(400)
    db.session.commit()
    flash('Задачу розміщено в календарі. Листи залишаються вимкненими.', 'success')
    return redirect(url_for('routes.calendar', date=to_local(start).date().isoformat()))

def event_form(event=None, status=200):
    return render_template('event_form.html', event=event, day=request.args.get('date', to_local(utcnow()).date().isoformat())), status

def event_values():
    title = request.form.get('title', '').strip()
    if not title or len(title) > 140:
        raise ValueError('Назва має містити від 1 до 140 символів.')
    kind = request.form.get('event_type', 'fixed')
    if kind not in ('fixed', 'break'):
        raise ValueError('Оберіть тип події.')
    start = local_to_utc(request.form.get('starts_at', ''))
    end = local_to_utc(request.form.get('ends_at', ''))
    if not start or not end:
        raise ValueError('Вкажіть початок і завершення події.')
    return dict(title=title, event_type=kind, starts_at=start, ends_at=end)

@routes.route('/calendar/events/new', methods=['GET', 'POST'])
@login_required
def create_event():
    if request.method == 'POST':
        try:
            values = event_values()
            ensure_free(current_user.id, values['starts_at'], values['ends_at'])
        except ValueError as error:
            flash(str(error), 'danger')
            return event_form(status=400)
        db.session.add(CalendarEvent(user_id=current_user.id, **values))
        touch(current_user)
        db.session.commit()
        flash('Подію додано.', 'success')
        return redirect(url_for('routes.calendar', date=to_local(values['starts_at']).date().isoformat()))
    return event_form()

@routes.route('/calendar/events/<int:event_id>/edit', methods=['GET', 'POST'])
@login_required
def edit_event(event_id):
    event = owned(CalendarEvent, event_id)
    if request.method == 'POST':
        try:
            values = event_values()
            ensure_free(current_user.id, values['starts_at'], values['ends_at'], exclude_event=event.id)
        except ValueError as error:
            flash(str(error), 'danger')
            return event_form(event, 400)
        for key, value in values.items():
            setattr(event, key, value)
        touch(current_user)
        db.session.commit()
        flash('Подію оновлено.', 'success')
        return redirect(url_for('routes.calendar', date=to_local(event.starts_at).date().isoformat()))
    return event_form(event)

@routes.route('/calendar/events/<int:event_id>/delete', methods=['POST'])
@login_required
def delete_event(event_id):
    event = owned(CalendarEvent, event_id)
    day = to_local(event.starts_at).date().isoformat()
    db.session.delete(event)
    touch(current_user)
    db.session.commit()
    flash('Подію видалено.', 'success')
    return redirect(url_for('routes.calendar', date=day))

@routes.route('/calendar/items/<int:item_id>/cancel', methods=['POST'])
@login_required
def cancel_schedule(item_id):
    item = owned(PlanItem, item_id)
    try:
        cancel_item(item)
    except ValueError as error:
        flash(str(error), 'danger')
        return redirect(url_for('routes.calendar', date=to_local(item.planned_start).date().isoformat()))
    touch(current_user)
    db.session.commit()
    flash('Розміщення скасовано. Задача залишається у вашому списку.', 'success')
    return redirect(url_for('routes.calendar', date=to_local(item.planned_start).date().isoformat()))

@routes.route('/calendar/items/<int:item_id>/start', methods=['POST'])
@login_required
def start_task(item_id):
    item = owned(PlanItem, item_id)
    if item.status != 'scheduled' or item.task.status != 'pending' or item.plan.status != 'accepted':
        abort(409)
    now = utcnow()
    if now < item.planned_start:
        flash('Почати виконання можна з часу, зазначеного в календарі.', 'danger')
    elif accepted_items(current_user.id).filter(PlanItem.status == 'in_progress').first():
        flash('Спочатку завершіть задачу, яку вже виконуєте.', 'danger')
    else:
        item.actual_start = now
        item.status = item.task.status = 'in_progress'
        touch(current_user)
        db.session.commit()
        flash('Початок виконання зафіксовано.', 'success')
    return redirect(url_for('routes.calendar', date=to_local(item.planned_start).date().isoformat()))

@routes.route('/send-test-email')
@login_required
def send_test_email():
    return 'Листи вимкнено в локальній версії. Повідомлення не надсилалось.', 200


# Recommendation routes share the transaction/CSRF policy of this blueprint.
from app.models import Plan
from app.recommendations import (available_windows, create_draft, accept_draft, dismiss_draft,
                                 draft_problem, StalePlanError)
from app.recommender import DAYPART_LABELS

def recommendation_form(status=200):
    local_now = to_local(utcnow())
    default_day = local_now.date() + (timedelta(days=1) if local_now.hour >= 18 else timedelta())
    recent = Plan.query.filter(Plan.user_id == current_user.id,
        Plan.parameters_json['mode'].as_string() == 'recommendation').order_by(Plan.created_at.desc(),Plan.id.desc()).limit(10).all()
    return render_template('recommendations_form.html', default_day=default_day,
                           recent=recent), status

@routes.route('/recommendations', methods=['GET','POST'])
@login_required
def recommendations():
    if request.method == 'POST':
        try:
            windows = available_windows(request.form.get('first_day',''),request.form.get('days','1'),
                                        request.form.get('start_clock',''),request.form.get('end_clock',''),current_user.timezone)
            raw = request.form.get('reminder_minutes','')
            reminder = int(raw) if raw else None
            plan = create_draft(current_user,windows,reminder_minutes=reminder,now=utcnow())
            db.session.commit()
        except ValueError as error:
            db.session.rollback()
            flash(str(error),'danger')
            return recommendation_form(400)
        return redirect(url_for('routes.recommendation_detail',plan_id=plan.id))
    return recommendation_form()

def recommendation_details(plan,status=200):
    if plan.parameters_json.get('mode') != 'recommendation':
        abort(404)
    return render_template('recommendation_detail.html',plan=plan,
        problem=draft_problem(plan,current_user,utcnow()),daypart_labels=DAYPART_LABELS), status

@routes.route('/recommendations/<int:plan_id>')
@login_required
def recommendation_detail(plan_id):
    return recommendation_details(owned(Plan,plan_id))

@routes.route('/recommendations/<int:plan_id>/accept', methods=['POST'])
@login_required
def accept_recommendations(plan_id):
    plan = owned(Plan,plan_id)
    try:
        changed = accept_draft(current_user,plan,now=utcnow())
        db.session.commit()
    except StalePlanError as error:
        db.session.rollback()
        flash(str(error),'danger')
        return recommendation_details(plan,409)
    except ValueError as error:
        db.session.rollback()
        flash(str(error),'danger')
        return recommendation_details(plan,400)
    flash('Рекомендований план додано до календаря.' if changed else 'Цей план уже додано до календаря.','success')
    return redirect(url_for('routes.calendar',date=to_local(plan.horizon_start).date().isoformat()))

@routes.route('/recommendations/<int:plan_id>/dismiss', methods=['POST'])
@login_required
def dismiss_recommendations(plan_id):
    plan = owned(Plan,plan_id)
    try:
        dismiss_draft(current_user,plan,now=utcnow())
        db.session.commit()
    except ValueError as error:
        db.session.rollback()
        flash(str(error),'danger')
        return recommendation_details(plan,409)
    flash('Проєкт плану скасовано. Задачі залишилися у вашому списку.','success')
    return redirect(url_for('routes.recommendations'))


@routes.route('/analytics')
@login_required
def analytics():
    from app.analytics import parse_period, build_analytics
    now = utcnow()
    status = 200
    try:
        period = parse_period(request.args.get('start', ''), request.args.get('end', ''),
                              current_user.timezone, now=now)
    except ValueError as error:
        flash(str(error), 'danger')
        period = parse_period('', '', current_user.timezone, now=now)
        status = 400
    first, last, start, end = period
    report = build_analytics(current_user, first, last, start, end, now=now)
    chart_days = [row for row in report['daily'] if row['planned']]
    chart_max = max((row['planned'] for row in chart_days), default=1)
    return render_template('analytics.html', report=report, first=first, last=last,
                           chart_days=chart_days, chart_max=chart_max, as_of=now), status
