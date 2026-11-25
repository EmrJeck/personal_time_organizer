"""Convert form wall times to UTC; reject DST gaps and ambiguous wall times."""
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from flask import current_app
from flask_login import current_user

def user_zone():
    name = current_user.timezone if current_user.is_authenticated else current_app.config['APP_TIMEZONE']
    return ZoneInfo(name)

def local_to_utc(value, zone=None, *, validate_range=True):
    if not value:
        return None
    try:
        wall = datetime.strptime(value, '%Y-%m-%dT%H:%M')
    except ValueError:
        raise ValueError('Перевірте дату й час.') from None
    if validate_range and not 1900 <= wall.year <= 2100:
        raise ValueError('Оберіть дату між 1900 і 2100 роками.')
    zone = zone or user_zone()
    candidates = set()
    for fold in (0, 1):
        aware = wall.replace(tzinfo=zone, fold=fold)
        instant = aware.astimezone(timezone.utc)
        if instant.astimezone(zone).replace(tzinfo=None) == wall:
            candidates.add(instant.replace(tzinfo=None))
    if len(candidates) != 1:
        raise ValueError('Цей місцевий час не існує або повторюється через переведення годинника. Виберіть інший час.')
    return candidates.pop()

def to_local(value, zone=None):
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc).astimezone(zone or user_zone())

def local_format(value, pattern='%d.%m.%Y %H:%M'):
    return to_local(value).strftime(pattern) if value else ''

def form_datetime(value):
    return local_format(value, '%Y-%m-%dT%H:%M')
