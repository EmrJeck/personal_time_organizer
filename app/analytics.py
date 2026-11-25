"""Read-only analytics of accepted scheduling attempts, with explicit time cutoffs.

The selected period is a cohort of planned starts in the user's current time zone.
It is not a measure of all calendar occupancy or of continuous productive work.
Recommendation history uses all eligible prior attempts with declining age weights.
"""
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.models import Category, Plan, PlanItem, utcnow
from app.recommendations import historical_recency_counts
from app.recommender import DAYPART_LABELS, weighted_history_rating, RECENCY_HALF_LIFE_DAYS


def _zone(name):
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, TypeError):
        raise ValueError('Невідомий часовий пояс.') from None


def _now(value):
    value = utcnow() if value is None else value
    if not isinstance(value, datetime) or value.tzinfo is not None:
        raise ValueError('Для розрахунку потрібен час у UTC без tzinfo.')
    return value


def _midnight(day, zone):
    """Reject exceptional ambiguous/nonexistent civil midnights explicitly."""
    wall = datetime.combine(day, time())
    candidates = set()
    for fold in (0, 1):
        instant = wall.replace(tzinfo=zone, fold=fold).astimezone(timezone.utc)
        if instant.astimezone(zone).replace(tzinfo=None) == wall:
            candidates.add(instant.replace(tzinfo=None))
    if len(candidates) != 1:
        raise ValueError('Межа періоду припадає на зміну місцевого часу. Оберіть інші дати.')
    return candidates.pop()


def parse_period(start_raw, end_raw, timezone_name, now=None):
    """Return inclusive local dates and exclusive UTC end; at most 366 dates."""
    now = _now(now)
    zone = _zone(timezone_name)
    if not start_raw and not end_raw:
        last = now.replace(tzinfo=timezone.utc).astimezone(zone).date()
        first = last - timedelta(days=29)
    else:
        if not start_raw or not end_raw:
            raise ValueError('Вкажіть обидві дати періоду.')
        try:
            first, last = date.fromisoformat(start_raw), date.fromisoformat(end_raw)
        except (ValueError, TypeError):
            raise ValueError('Перевірте дати початку й завершення періоду.') from None
    if not (1900 <= first.year <= 2100 and 1900 <= last.year <= 2100):
        raise ValueError('Оберіть дати між 1900 і 2100 роками.')
    if first > last:
        raise ValueError('Початок періоду має передувати завершенню або збігатися з ним.')
    if (last-first).days >= 366:
        raise ValueError('Оберіть період не довший за 366 днів.')
    return first, last, _midnight(first, zone), _midnight(last+timedelta(days=1), zone)


def build_analytics(user, first_date, last_date, start_utc, end_utc, now=None):
    """Summarize the selected cohort; never use mutable task status as history."""
    now = _now(now)
    zone = _zone(user.timezone)
    # Validate the public function's boundaries too, rather than allocating an
    # unbounded daily series if called outside the web route.
    expected = parse_period(first_date.isoformat(), last_date.isoformat(), user.timezone, now)
    if (start_utc, end_utc) != expected[2:]:
        raise ValueError('Межі періоду не відповідають обраному часовому поясу.')

    daily = []
    for offset in range((last_date-first_date).days+1):
        day = first_date + timedelta(days=offset)
        daily.append(dict(date=day, label=day.strftime('%d.%m'), planned=0,
                          completed=0, due=0, on_time=0))
    by_day = {row['date']: row for row in daily}
    report = dict(planned_count=0, completed_count=0, due_count=0,
                  on_time_count=0, late_count=0, unrecorded_count=0,
                  on_time_percent=None, planned_minutes=0.0, recorded_minutes=0.0,
                  measured_count=0, excluded_early_count=0, daily=daily)

    attempts = PlanItem.query.join(Plan).filter(
        Plan.user_id == user.id, Plan.accepted_at.is_not(None),
        Plan.accepted_at <= now, Plan.created_at <= now,
        PlanItem.planned_start >= start_utc, PlanItem.planned_start < end_utc).all()
    for item in attempts:
        if item.canceled_at is not None and item.canceled_at < item.planned_start:
            report['excluded_early_count'] += 1
            continue
        day = item.planned_start.replace(tzinfo=timezone.utc).astimezone(zone).date()
        row = by_day[day]
        report['planned_count'] += 1
        row['planned'] += 1
        report['planned_minutes'] += (item.planned_end-item.planned_start).total_seconds()/60
        completed = item.actual_end is not None and item.actual_end <= now
        if completed:
            report['completed_count'] += 1
            row['completed'] += 1
            if item.actual_start is not None and item.actual_start <= item.actual_end:
                report['measured_count'] += 1
                report['recorded_minutes'] += (item.actual_end-item.actual_start).total_seconds()/60
        if item.planned_end < now:
            report['due_count'] += 1
            row['due'] += 1
            if completed and item.actual_end <= item.planned_end:
                report['on_time_count'] += 1
                row['on_time'] += 1
            elif completed:
                report['late_count'] += 1
            else:
                report['unrecorded_count'] += 1
    if report['due_count']:
        report['on_time_percent'] = 100 * report['on_time_count'] / report['due_count']

    # Use exactly the recommender's eligibility rules and smoothing. These counts
    # are independent of the selected reporting period, as the UI states.
    counts, ignored = historical_recency_counts(user.id, now)
    category_ids = {category_id for category_id, _ in counts}
    categories = {category.id: category.name for category in
                  Category.query.filter(Category.id.in_(category_ids)).all()} if category_ids else {}
    parts = {part: index for index, part in enumerate(DAYPART_LABELS)}
    history = []
    for (category_id, part), count in counts.items():
        history.append(dict(category_id=category_id,
                            category=categories.get(category_id, f'Категорія {category_id}'),
                            daypart=part, daypart_label=DAYPART_LABELS[part],
                            successes=count.successes, total=count.total,
                            weighted_successes=count.weighted_successes,
                            weighted_total=count.weighted_total,
                            score=weighted_history_rating(count.weighted_successes, count.weighted_total)))
    history.sort(key=lambda row: (row['category'].casefold(), row['category_id'], parts[row['daypart']]))
    report.update(history=history, history_total=sum(count.total for count in counts.values()),
                  ignored_history=ignored, history_half_life_days=RECENCY_HALF_LIFE_DAYS)
    return report
