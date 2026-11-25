"""Deterministic, pure planner for model 2.2. Inputs/outputs are naive UTC."""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import isfinite
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

VERSION = '1.0'
RULE_TABLE = ((.05, .25, .55), (.30, .55, .80), (.60, .85, 1.00))
METHODS = ('personalized', 'fuzzy', 'weighted', 'weighted_history', 'edf')
DAYPART_LABELS = {'night': 'ніч', 'morning': 'ранок', 'afternoon': 'день', 'evening': 'вечір'}
MAX_TASKS = 200

@dataclass(frozen=True)
class TaskInput:
    id: int
    title: str
    category_id: int
    importance: int
    duration_min: int
    earliest_start: datetime | None = None
    deadline: datetime | None = None

@dataclass(frozen=True)
class HistoryCount:
    successes: int
    total: int

@dataclass(frozen=True)
class Assignment:
    task_id: int
    start: datetime
    end: datetime
    importance_normalized: float
    urgency: float
    fuzzy_score: float
    base_score: float
    history_score: float
    history_successes: int
    history_total: int
    daypart: str
    score: float

@dataclass(frozen=True)
class Rejection:
    task_id: int
    reason: str

@dataclass
class RecommendationResult:
    assignments: list
    unscheduled: list
    candidate_count: int

def _unit(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'{label}: потрібне число від 0 до 1.')

def _utc(value):
    if not isinstance(value, datetime) or value.tzinfo is not None:
        raise ValueError('Розрахунковий модуль очікує дату й час у UTC без tzinfo.')

def allocated_minutes(duration):
    if isinstance(duration, bool) or not isinstance(duration, int) or not 1 <= duration <= 10080:
        raise ValueError('Тривалість задачі має бути цілим числом від 1 до 10080 хвилин.')
    return ((duration + 14) // 15) * 15

def memberships(x):
    _unit(x, 'Нормована ознака')
    return max(0., 1-2*x), 1-abs(2*x-1), max(0., 2*x-1)

def fuzzy_priority(importance, urgency):
    mi, mu = memberships(importance), memberships(urgency)
    activations = [(mi[a]*mu[b], RULE_TABLE[a][b]) for a in range(3) for b in range(3)]
    return sum(alpha*z for alpha,z in activations) / sum(alpha for alpha,z in activations)

def history_rating(successes, total, prior_strength=2., prior_mean=.5):
    if (isinstance(successes, bool) or isinstance(total, bool) or
        not isinstance(successes, int) or not isinstance(total, int) or not 0 <= successes <= total):
        raise ValueError('Некоректна кількість історичних спостережень.')
    _unit(prior_mean, 'Початкова історична оцінка')
    if isinstance(prior_strength, bool) or not isinstance(prior_strength,(int,float)) or not isfinite(prior_strength) or prior_strength <= 0:
        raise ValueError('Сила згладжування має бути додатним скінченним числом.')
    return (successes + prior_strength*prior_mean)/(total + prior_strength)

def daypart_of(start, timezone_name='Europe/Kyiv'):
    _utc(start)
    hour = start.replace(tzinfo=timezone.utc).astimezone(ZoneInfo(timezone_name)).hour
    return ('night','morning','afternoon','evening')[hour//6]

def _ceil_slot(instant, zone):
    local = instant.replace(tzinfo=timezone.utc).astimezone(zone)
    remainder = timedelta(minutes=local.minute % 15, seconds=local.second, microseconds=local.microsecond)
    return instant - remainder + (timedelta(minutes=15) if remainder else timedelta())

def urgency_at(task, origin):
    if task.deadline is None:
        return 0.
    duration = allocated_minutes(task.duration_min)
    earliest = max(origin, task.earliest_start or origin)
    slack = max(0., (task.deadline-earliest).total_seconds()/60-duration)
    return duration/(duration+slack)

def _overlaps(start, end, intervals):
    return any(start < b and end > a for a,b in intervals)

def generate_plan(tasks, available_windows, busy_intervals, history, as_of,
                  timezone_name='Europe/Kyiv', method='personalized',
                  history_weight=.2, prior_strength=2., prior_mean=.5):
    """Ranks feasible pairs once. Greedy acceptance is equivalent to repeated max selection.
    The score is a preference rating, never a calibrated completion probability.
    """
    _utc(as_of)
    try:
        zone = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError, TypeError):
        raise ValueError('Невідомий часовий пояс.') from None
    if method not in METHODS:
        raise ValueError('Невідомий метод планування.')
    _unit(history_weight, 'Вага історії')
    history_rating(0, 0, prior_strength, prior_mean)
    tasks = list(tasks)
    if len(tasks) > MAX_TASKS:
        raise ValueError(f'За один запуск можна планувати не більше {MAX_TASKS} задач.')
    if len({task.id for task in tasks}) != len(tasks):
        raise ValueError('Перелік містить повторні ідентифікатори задач.')
    for task in tasks:
        allocated_minutes(task.duration_min)
        if isinstance(task.importance,bool) or not isinstance(task.importance,int) or not 1 <= task.importance <= 5:
            raise ValueError('Важливість задачі має бути цілим числом від 1 до 5.')
        for value in (task.earliest_start,task.deadline):
            if value is not None:
                _utc(value)
        if task.earliest_start and task.deadline and task.earliest_start >= task.deadline:
            raise ValueError('Некоректні часові обмеження задачі.')
    windows = sorted(available_windows)
    busy = sorted(busy_intervals)
    if not windows:
        raise ValueError('Оберіть хоча б один доступний часовий інтервал.')
    for start,end in windows + busy:
        _utc(start); _utc(end)
        if start >= end:
            raise ValueError('Початок часового інтервалу має передувати завершенню.')
    # A seven-local-day horizon can contain an extra hour when clocks move back.
    if windows[-1][1]-windows[0][0] > timedelta(days=7,hours=2):
        raise ValueError('Горизонт планування не може перевищувати 7 днів.')
    if any(windows[i][1] > windows[i+1][0] for i in range(len(windows)-1)):
        raise ValueError('Доступні часові інтервали перетинаються.')
    for (category,part), count in history.items():
        if part not in DAYPART_LABELS:
            raise ValueError('Невідома частина доби в історії.')
        history_rating(count.successes,count.total,prior_strength,prior_mean)
    origin = max(as_of, windows[0][0])
    candidates, reasons = [], {}
    for task in tasks:
        minutes = allocated_minutes(task.duration_min)
        duration = timedelta(minutes=minutes)
        importance = (task.importance-1)/4
        urgency = urgency_at(task, origin)
        fuzzy = fuzzy_priority(importance,urgency)
        base = .6*importance + .4*urgency if method in ('weighted','weighted_history') else fuzzy
        scores = {}
        for part in DAYPART_LABELS:
            count = history.get((task.category_id,part),HistoryCount(0,0))
            h = history_rating(count.successes,count.total,prior_strength,prior_mean)
            score = (1-history_weight)*base + history_weight*h if method in ('personalized','weighted_history') else base
            scores[part] = (score,h,count)
        time_fits = any(max(a,as_of,task.earliest_start or a)+duration <= min(b,task.deadline or b) for a,b in windows)
        count_before = len(candidates)
        for a,b in windows:
            start = _ceil_slot(max(a,as_of,task.earliest_start or a),zone)
            limit = min(b,task.deadline or b)
            while start+duration <= limit:
                end = start+duration
                if not _overlaps(start,end,busy):
                    part = daypart_of(start,timezone_name)
                    score,h,count = scores[part]
                    # Round only the sort key to avoid floating-point tie artifacts;
                    # keep the original score for explanation and reproducibility.
                    key = (task.deadline or datetime.max,start,task.id) if method == 'edf' else (
                        -round(score,12), task.deadline or datetime.max,start,task.id)
                    payload = (importance,urgency,fuzzy,base,h,count,part,score)
                    candidates.append((key,task.id,start,end,payload))
                start += timedelta(minutes=15)
        if len(candidates) == count_before:
            if task.deadline and task.deadline <= origin:
                reason = 'Граничний строк минув до доступного початку планування.'
            elif not time_fits:
                reason = 'Тривалість і часові обмеження не дозволяють розміщення в обраному періоді.'
            else:
                reason = 'Немає вільного інтервалу потрібної довжини на сітці 15 хвилин.'
            reasons[task.id] = reason
    candidates.sort(key=lambda candidate:candidate[0])
    chosen, intervals = {}, []
    for key,task_id,start,end,payload in candidates:
        if task_id in chosen or _overlaps(start,end,intervals):
            continue
        i,u,f,base,h,count,part,score = payload
        chosen[task_id] = Assignment(task_id,start,end,i,u,f,base,h,count.successes,count.total,part,score)
        intervals.append((start,end))
    rejected = [Rejection(task.id,reasons.get(task.id,'Доступний час зайняли інші обрані задачі.'))
                for task in tasks if task.id not in chosen]
    return RecommendationResult(sorted(chosen.values(),key=lambda x:(x.start,x.task_id)),rejected,len(candidates))
