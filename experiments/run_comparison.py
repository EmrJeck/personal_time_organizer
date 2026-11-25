"""Frozen synthetic comparison for thesis 3.3, using the unmodified planner.

The evaluator tests whether each independently simulated duration fits its own
planned interval. It does not simulate cascading delays or claim human outcomes.
Run: python -m experiments.run_comparison --output experiments/results_v1
"""
import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

from app import recommender
from app.recommender import HistoryCount, TaskInput, generate_plan


SCHEMA_VERSION = '1.0'
MASTER_SEED = 20261104
CONDITIONS = ('stable', 'no_time_effect', 'drift')
LOADS = (0.7, 1.4)
HISTORY_SIZES = (0, 1, 20)
METHODS = ('edf', 'weighted', 'weighted_history', 'fuzzy', 'personalized')
METHOD_LABELS = {
    'edf': 'Найближчий строк',
    'weighted': 'Лінійна оцінка',
    'weighted_history': 'Лінійна оцінка з історією',
    'fuzzy': 'Нечітка оцінка',
    'personalized': 'Запропонований метод',
}
DURATIONS = (20, 30, 40, 60, 90, 120)
SLACKS = (0, 30, 60, 120, 240, 480, 960)
PARTS = ('night', 'morning', 'afternoon', 'evening')
ZONE_NAME = 'Europe/Kyiv'
FIRST_DAY = datetime(2030, 1, 7)
CAPACITY_MINUTES = 960
VIOLATION_KEYS = (
    'unknown_task_violations', 'duplicate_violations', 'conflict_violations',
    'busy_violations', 'deadline_violations', 'release_violations',
    'window_violations', 'grid_violations', 'duration_violations',
    'past_start_violations', 'partition_violations',
)


def _seed(*components):
    text = '|'.join(map(str, components)).encode('utf-8')
    return int.from_bytes(hashlib.sha256(text).digest()[:8], 'big')


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'),
                      default=lambda item: item.isoformat() if isinstance(item, datetime) else str(item))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _utc_local(day, hour):
    wall = day + timedelta(hours=hour)
    return wall.replace(tzinfo=ZoneInfo(ZONE_NAME)).astimezone(timezone.utc).replace(tzinfo=None)


def _allocated(duration):
    # Deliberately not the planner helper: an independent specification check.
    return 15 * math.ceil(duration / 15)


def duration_model(estimated_duration, hour, peak, amplitude, difficulty, noise):
    """Independent lognormal duration model. No priority/history score input."""
    return estimated_duration * math.exp(
        0.12 - amplitude * math.cos(2 * math.pi * (hour - peak) / 24)
        + difficulty + 0.25 * noise)


def configuration(repeats=30, timing_repeats=3, master_seed=MASTER_SEED):
    return {
        'schema_version': SCHEMA_VERSION,
        'master_seed': master_seed,
        'repeats_per_cell': repeats,
        'timing_repeats': timing_repeats,
        'conditions': list(CONDITIONS), 'load_targets': list(LOADS),
        'history_sizes_per_category_daypart': list(HISTORY_SIZES),
        'methods': list(METHODS), 'method_labels': METHOD_LABELS,
        'scenario_count': len(CONDITIONS) * len(LOADS) * len(HISTORY_SIZES) * repeats,
        'independence': 'Each full condition/load/history_n/repeat cell has a distinct SHA-256-derived seed; paired only across methods.',
        'seed_derivation': 'First 8 SHA-256 bytes, big endian, of master_seed|condition|load_target|history_n|repeat; tagged RNG streams.',
        'first_local_date': FIRST_DAY.date().isoformat(), 'timezone': ZONE_NAME,
        'days': 2, 'daily_window': ['09:00', '18:00'],
        'daily_busy': [['12:00', '12:30'], ['15:00', '15:30']],
        'capacity_minutes': CAPACITY_MINUTES, 'grid_minutes': 15,
        'durations_minutes': list(DURATIONS), 'importance_range_inclusive': [1, 5],
        'categories': 3,
        'task_count_rule': 'Generate until sum(15*ceil(duration/15)) >= load_target*960; retain final whole task.',
        'earliest_start': {'missing_probability': 0.4, 'first_day_probability_if_present': 2/3,
                           'quarter_hour_slots_inclusive': ['09:00', '16:00']},
        'deadline': {'missing_probability': 0.2, 'slack_minutes_uniform': list(SLACKS),
                     'rule': 'min(last_window_end, effective_release + allocated_duration + slack)'},
        'fixed_weights': {'linear': [0.6, 0.4], 'history_weight': 0.2,
                          'fuzzy_weight': 0.8, 'prior_strength': 2, 'prior_mean': 0.5},
        'fixed_rule_table': [list(row) for row in recommender.RULE_TABLE],
        'planner_version': recommender.VERSION,
        'simulator': {
            'formula': 'duration * exp(0.12 - A*cos(2*pi*(hour-peak)/24) + difficulty + 0.25*noise)',
            'amplitude_stable_and_drift': 0.28, 'amplitude_no_time_effect': 0.0,
            'past_category_peaks_uniform_hours': [8, 20], 'future_drift_hours': 8,
            'task_difficulty_uniform': [-0.12, 0.12], 'noise': 'independent standard normal',
            'history_hours': 'uniform inside each six-hour daypart',
            'history_dates': 'uniform over 1 to 60 days before the first planning date',
            'history_success': 'simulated_duration <= 15*ceil(estimated_duration/15)',
            'future_random_numbers': 'one fixed difficulty and noise per task, shared by all methods',
            'evaluation': 'independent interval fit, no cascading delay propagation',
            'no_time_effect_control': 'no systematic time-of-day or category effect',
            'planner_has_access_to_hidden_simulation_fields': False,
        },
        'measurement': {
            'warmup': 'one untimed invocation of each of the five methods on the first scenario',
            'order': 'independently shuffled methods inside each timing round of each scenario',
            'elapsed_ms': 'median of planner-only perf_counter durations',
            'denominator_importance': 'sum of importance of ALL input tasks, including unplaced/infeasible tasks',
            'fraction_units': 'percent, 0 to 100',
            'runtime_scope': 'five modes of a shared implementation; all currently compute fuzzy_priority',
            'parameter_tuning': 'none; previously specified parameters remain fixed',
        },
    }


def generate_case(condition, load_target, history_n, repeat, master_seed=MASTER_SEED):
    if condition not in CONDITIONS or load_target not in LOADS or history_n not in HISTORY_SIZES:
        raise ValueError('Unsupported experimental cell')
    if not isinstance(repeat, int) or repeat < 0:
        raise ValueError('repeat must be non-negative')
    seed = _seed(master_seed, condition, load_target, history_n, repeat)
    task_rng = random.Random(_seed(seed, 'tasks'))
    world_rng = random.Random(_seed(seed, 'world'))
    history_rng = random.Random(_seed(seed, 'history'))
    future_rng = random.Random(_seed(seed, 'future'))
    windows = [(_utc_local(FIRST_DAY + timedelta(days=i), 9),
                _utc_local(FIRST_DAY + timedelta(days=i), 18)) for i in range(2)]
    busy = [(_utc_local(FIRST_DAY + timedelta(days=i), a),
             _utc_local(FIRST_DAY + timedelta(days=i), b))
            for i in range(2) for a, b in ((12, 12.5), (15, 15.5))]
    as_of = _utc_local(FIRST_DAY, 8)
    past_peaks = {str(category): world_rng.uniform(8, 20) for category in (1, 2, 3)}
    future_peaks = {category: (peak + (8 if condition == 'drift' else 0)) % 24
                    for category, peak in past_peaks.items()}
    amplitude = 0 if condition == 'no_time_effect' else 0.28
    tasks, future, minutes = [], {}, 0
    while minutes < load_target * CAPACITY_MINUTES:
        task_id = len(tasks) + 1
        duration = task_rng.choice(DURATIONS)
        allocated = _allocated(duration)
        importance = task_rng.randint(1, 5)
        category = task_rng.randint(1, 3)
        earliest = None
        if task_rng.random() >= 0.4:
            offset = 0 if task_rng.random() < 2/3 else 1
            earliest = _utc_local(FIRST_DAY + timedelta(days=offset), 9 + task_rng.randint(0, 28)/4)
        release = earliest or windows[0][0]
        deadline = None
        if task_rng.random() >= 0.2:
            deadline = min(windows[-1][1], release + timedelta(minutes=allocated + task_rng.choice(SLACKS)))
        tasks.append({'id': task_id, 'title': f'Task {task_id}', 'category_id': category,
                      'importance': importance, 'duration_min': duration,
                      'earliest_start': earliest.isoformat() if earliest else None,
                      'deadline': deadline.isoformat() if deadline else None})
        future[str(task_id)] = {'difficulty': future_rng.uniform(-0.12, 0.12),
                                'noise': future_rng.gauss(0, 1)}
        minutes += allocated
    history, history_records = [], []
    for category in (1, 2, 3):
        for part_index, part in enumerate(PARTS):
            successes = 0
            for observation in range(history_n):
                hour = 6 * part_index + 6 * history_rng.random()
                day = FIRST_DAY - timedelta(days=history_rng.randint(1, 60))
                start = _utc_local(day, hour)
                duration = history_rng.choice(DURATIONS)
                difficulty = history_rng.uniform(-0.12, 0.12)
                noise = history_rng.gauss(0, 1)
                actual = duration_model(duration, hour, past_peaks[str(category)], amplitude, difficulty, noise)
                success = actual <= _allocated(duration)
                successes += int(success)
                history_records.append({'category_id': category, 'daypart': part,
                                        'observation': observation, 'start': start.isoformat(),
                                        'hour_local': hour, 'estimated_minutes': duration,
                                        'difficulty': difficulty, 'noise': noise,
                                        'actual_minutes': actual, 'success': success})
            history.append({'category_id': category, 'daypart': part,
                            'successes': successes, 'total': history_n})
    case = {
        'schema_version': SCHEMA_VERSION,
        'scenario_id': f'{condition}_load{load_target}_n{history_n}_r{repeat:03}',
        'condition': condition, 'load_target': load_target, 'history_n': history_n,
        'repeat': repeat, 'seed': seed,
        'total_allocated_minutes': minutes, 'capacity_minutes': CAPACITY_MINUTES,
        'actual_load': minutes/CAPACITY_MINUTES,
        'planner': {'tasks': tasks, 'available_windows': [[a.isoformat(), b.isoformat()] for a, b in windows],
                    'busy_intervals': [[a.isoformat(), b.isoformat()] for a, b in busy],
                    'history': history, 'as_of': as_of.isoformat(), 'timezone_name': ZONE_NAME},
        'simulation': {'amplitude': amplitude, 'historical_peaks': past_peaks,
                       'future_peaks': future_peaks, 'future_task_noise': future},
        'history_records': history_records,
    }
    feasible_ids = individually_feasible_ids(case)
    case['individually_feasible_ids'] = feasible_ids
    case['independently_feasible_count'] = len(feasible_ids)
    case['independently_feasible_importance'] = sum(t['importance'] for t in tasks if t['id'] in feasible_ids)
    return case


def planner_inputs(case):
    """Explicit allow-list: the planner cannot inspect the latent world/noise."""
    raw = case['planner']
    tasks = [TaskInput(**{**task,
                         'earliest_start': datetime.fromisoformat(task['earliest_start']) if task['earliest_start'] else None,
                         'deadline': datetime.fromisoformat(task['deadline']) if task['deadline'] else None})
             for task in raw['tasks']]
    return {'tasks': tasks,
            'available_windows': [(datetime.fromisoformat(a), datetime.fromisoformat(b)) for a, b in raw['available_windows']],
            'busy_intervals': [(datetime.fromisoformat(a), datetime.fromisoformat(b)) for a, b in raw['busy_intervals']],
            'history': {(row['category_id'], row['daypart']): HistoryCount(row['successes'], row['total'])
                        for row in raw['history']},
            'as_of': datetime.fromisoformat(raw['as_of']), 'timezone_name': raw['timezone_name']}


def individually_feasible_ids(case):
    inputs = planner_inputs(case)
    feasible = []
    for task in inputs['tasks']:
        duration = timedelta(minutes=_allocated(task.duration_min))
        found = False
        for start, end in inputs['available_windows']:
            candidate = start
            while candidate + duration <= end:
                if (candidate >= inputs['as_of']
                        and (task.earliest_start is None or candidate >= task.earliest_start)
                        and (task.deadline is None or candidate + duration <= task.deadline)
                        and not any(candidate < b and candidate + duration > a for a, b in inputs['busy_intervals'])):
                    found = True
                    break
                candidate += timedelta(minutes=15)
            if found:
                break
        if found:
            feasible.append(task.id)
    return feasible


def simulate_duration(case, task, start):
    if isinstance(task, dict):
        task_id, category_id, duration = task['id'], task['category_id'], task['duration_min']
    else:
        task_id, category_id, duration = task.id, task.category_id, task.duration_min
    local = start.replace(tzinfo=timezone.utc).astimezone(ZoneInfo(case['planner']['timezone_name']))
    hour = local.hour + local.minute/60 + local.second/3600 + local.microsecond/3_600_000_000
    hidden = case['simulation']
    noise = hidden['future_task_noise'][str(task_id)]
    return duration_model(duration, hour, hidden['future_peaks'][str(category_id)], hidden['amplitude'],
                          noise['difficulty'], noise['noise'])


def validate_plan(case, result):
    """Validate scheduling constraints independently of score and engine helpers."""
    inputs = planner_inputs(case)
    tasks = {task.id: task for task in inputs['tasks']}
    counts = {key: 0 for key in VIOLATION_KEYS}
    seen, intervals = set(), []
    for item in result.assignments:
        if item.task_id in seen:
            counts['duplicate_violations'] += 1
        seen.add(item.task_id)
        if item.task_id not in tasks:
            counts['unknown_task_violations'] += 1
            continue
        task = tasks[item.task_id]
        if item.end-item.start != timedelta(minutes=_allocated(task.duration_min)):
            counts['duration_violations'] += 1
        if item.start < inputs['as_of']:
            counts['past_start_violations'] += 1
        if task.earliest_start is not None and item.start < task.earliest_start:
            counts['release_violations'] += 1
        if task.deadline is not None and item.end > task.deadline:
            counts['deadline_violations'] += 1
        if not any(a <= item.start < item.end <= b for a, b in inputs['available_windows']):
            counts['window_violations'] += 1
        if any(item.start < b and item.end > a for a, b in inputs['busy_intervals']):
            counts['busy_violations'] += 1
        local = item.start.replace(tzinfo=timezone.utc).astimezone(ZoneInfo(inputs['timezone_name']))
        if local.minute % 15 or local.second or local.microsecond:
            counts['grid_violations'] += 1
        counts['conflict_violations'] += sum(item.start < b and item.end > a for a, b in intervals)
        intervals.append((item.start, item.end))
    rejected = [item.task_id for item in result.unscheduled]
    if (len(rejected) != len(set(rejected)) or seen & set(rejected)
            or seen | set(rejected) != set(tasks)):
        counts['partition_violations'] += 1
    counts['total_violations'] = sum(counts.values())
    return counts


def evaluate_plan(case, result):
    tasks = {task['id']: task for task in case['planner']['tasks']}
    total_importance = sum(task['importance'] for task in tasks.values())
    placed_importance, fit_importance, fit_count = 0, 0, 0
    for item in result.assignments:
        task = tasks[item.task_id]
        placed_importance += task['importance']
        if simulate_duration(case, task, item.start) <= (item.end-item.start).total_seconds()/60:
            fit_importance += task['importance']
            fit_count += 1
    count = len(result.assignments)
    return {
        'input_task_count': len(tasks), 'input_importance': total_importance,
        'placed_count': count, 'placed_importance': placed_importance,
        'simulated_fit_count': fit_count, 'simulated_fit_importance': fit_importance,
        'placed_fraction': 100*count/len(tasks),
        'scheduled_importance_fraction': 100*placed_importance/total_importance,
        'simulated_on_time_importance_fraction': 100*fit_importance/total_importance,
        'fit_fraction_of_placed': 100*fit_count/count if count else None,
        'candidate_count': result.candidate_count,
        **validate_plan(case, result),
    }


def _signature(result):
    return ([(a.task_id, a.start, a.end, a.score) for a in result.assignments],
            [(r.task_id, r.reason) for r in result.unscheduled], result.candidate_count)


def run_experiment(output, repeats=30, timing_repeats=3, master_seed=MASTER_SEED, progress=True):
    if not isinstance(repeats, int) or repeats < 1 or not isinstance(timing_repeats, int) or timing_repeats < 1:
        raise ValueError('Positive integer repeat counts required')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Output must be new or empty; existing evidence will not be overwritten')
    output.mkdir(parents=True, exist_ok=True)
    config = configuration(repeats, timing_repeats, master_seed)
    # Freeze the specification before generating any outcomes or metric files.
    (output/'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
    started = perf_counter()
    manifest = {
        'schema_version': SCHEMA_VERSION, 'status': 'running',
        'started_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'platform': platform.platform(),
        'cpu': platform.processor() or os.environ.get('PROCESSOR_IDENTIFIER', 'unknown'),
        'logical_cpu_count': os.cpu_count(),
        'script_sha256': _sha(__file__), 'engine_sha256': _sha(recommender.__file__),
        'config_sha256': _sha(output/'config.json'),
        'independent_scenario_count': config['scenario_count'],
        'paired_methods_on_identical_inputs': True,
        'database_access': False, 'parameter_tuning': False,
    }
    manifest_path = output/'manifest.json'

    def write_manifest():
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')

    write_manifest()
    try:
        cases = [generate_case(condition, load, n, repeat, master_seed)
                 for condition in CONDITIONS for load in LOADS for n in HISTORY_SIZES for repeat in range(repeats)]
        if len({case['seed'] for case in cases}) != len(cases):
            raise AssertionError('Scenario seed collision')
        with (output/'inputs.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
            for case in cases:
                stream.write(_json(case)+'\n')
        manifest['inputs_sha256'] = _sha(output/'inputs.jsonl')
        write_manifest()
        # A single common warmup stage, kept outside every measured sample.
        for method in METHODS:
            generate_plan(**planner_inputs(cases[0]), method=method)
        records, total_violations = [], 0
        with (output/'assignments.jsonl').open('w', encoding='utf-8', newline='\n') as plans_stream:
            for case_index, case in enumerate(cases):
                inputs = planner_inputs(case)
                results, timings = {}, {method: [] for method in METHODS}
                order_rng = random.Random(_seed(case['seed'], 'timing_order'))
                orders = []
                for timing_round in range(timing_repeats):
                    order = list(METHODS)
                    order_rng.shuffle(order)
                    orders.append(order)
                    for method in order:
                        before = perf_counter()
                        result = generate_plan(**inputs, method=method)
                        elapsed = (perf_counter()-before)*1000
                        timings[method].append(elapsed)
                        if method in results and _signature(result) != _signature(results[method]):
                            raise AssertionError(f'Non-deterministic plan: {case["scenario_id"]}/{method}')
                        results[method] = result
                public_hash = hashlib.sha256(_json(case['planner']).encode('utf-8')).hexdigest()
                for method in METHODS:
                    result = results[method]
                    metrics = evaluate_plan(case, result)
                    total_violations += metrics['total_violations']
                    row = {'scenario_id': case['scenario_id'], 'condition': case['condition'],
                           'load_target': case['load_target'], 'actual_load': case['actual_load'],
                           'history_n': case['history_n'], 'repeat': case['repeat'], 'seed': case['seed'],
                           'method': method, 'planner_inputs_sha256': public_hash,
                           'independently_feasible_count': case['independently_feasible_count'],
                           'independently_feasible_importance': case['independently_feasible_importance'],
                           **metrics, 'elapsed_ms': statistics.median(timings[method])}
                    records.append(row)
                    assignments = []
                    by_id = {task.id: task for task in inputs['tasks']}
                    for assignment in result.assignments:
                        values = asdict(assignment)
                        actual = simulate_duration(case, by_id[assignment.task_id], assignment.start)
                        values['simulated_duration_minutes'] = actual
                        values['simulated_fits_interval'] = actual <= (assignment.end-assignment.start).total_seconds()/60
                        assignments.append(values)
                    plans_stream.write(_json({'scenario_id': case['scenario_id'], 'method': method,
                                              'planner_inputs_sha256': public_hash, 'assignments': assignments,
                                              'unscheduled': [asdict(rejection) for rejection in result.unscheduled],
                                              'timing_samples_ms': timings[method], 'method_orders': orders})+'\n')
                if progress and ((case_index+1) % 30 == 0 or case_index+1 == len(cases)):
                    print(f'Completed {case_index+1}/{len(cases)} scenarios', flush=True)
        with (output/'case_metrics.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        manifest.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                        wall_time_seconds=perf_counter()-started,
                        scenario_count=len(cases), metric_rows=len(records), total_violations=total_violations,
                        measured_planner_calls=len(cases)*len(METHODS)*timing_repeats,
                        warmup_planner_calls=len(METHODS),
                        outputs_sha256={name: _sha(output/name) for name in ('inputs.jsonl', 'assignments.jsonl', 'case_metrics.csv')})
        write_manifest()
        return manifest
    except Exception as error:
        manifest.update(status='failed', error=repr(error), wall_time_seconds=perf_counter()-started)
        write_manifest()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('experiments/results_v1'))
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--timing-repeats', type=int, default=3)
    parser.add_argument('--master-seed', type=int, default=MASTER_SEED)
    args = parser.parse_args()
    manifest = run_experiment(args.output, args.repeats, args.timing_repeats, args.master_seed)
    print(json.dumps({'status': manifest['status'], 'scenarios': manifest['scenario_count'],
                      'rows': manifest['metric_rows'], 'violations': manifest['total_violations'],
                      'wall_time_seconds': manifest['wall_time_seconds']}, indent=2))


if __name__ == '__main__':
    main()
