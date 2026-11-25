"""Prespecified v2 comparison with dated history; no database access.

All seven modes receive the same inputs. Hidden future durations never enter the
planner. Historical outcomes can reflect a change only after its simulated date.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import statistics
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter

from app import recommender
from app.recommender import (HistoryObservation, TaskInput, WeightedHistoryCount,
                             generate_plan, recency_counts)
from experiments import run_comparison as legacy
from experiments.frozen_v1 import recommender as frozen_engine


SCHEMA_VERSION = '2.0'
MASTER_SEED = 20261106
HALF_LIFE_DAYS = 14.0
CONDITIONS = ('stable', 'no_time_effect', 'unseen_drift', 'observed_drift_7',
              'observed_drift_14', 'observed_drift_28')
CHANGE_AGES = {'unseen_drift': 0, 'observed_drift_7': 7,
               'observed_drift_14': 14, 'observed_drift_28': 28}
LOADS = legacy.LOADS
HISTORY_SIZES = legacy.HISTORY_SIZES
LEGACY_METHODS = legacy.METHODS
METHODS = (*LEGACY_METHODS, 'weighted_recent', 'personalized_recent')
METHOD_LABELS = {
    'edf': 'Найближчий строк', 'weighted': 'Лінійна оцінка',
    'weighted_history': 'Лінійна оцінка з усією історією', 'fuzzy': 'Нечітка оцінка',
    'personalized': 'Попередній повний метод',
    'weighted_recent': 'Лінійна оцінка з давністю історії',
    'personalized_recent': 'Оновлений повний метод',
}
ROOT = Path(__file__).resolve().parent.parent
_seed, _json, _sha = legacy._seed, legacy._json, legacy._sha
_utc_local, _allocated = legacy._utc_local, legacy._allocated
duration_model = legacy.duration_model
simulate_duration = legacy.simulate_duration
evaluate_plan = legacy.evaluate_plan
validate_plan = legacy.validate_plan
individually_feasible_ids = legacy.individually_feasible_ids


def configuration(repeats=40, timing_repeats=3, master_seed=MASTER_SEED):
    config = legacy.configuration(repeats, timing_repeats, master_seed)
    config.update(schema_version=SCHEMA_VERSION, conditions=list(CONDITIONS),
                  methods=list(METHODS), method_labels=METHOD_LABELS,
                  scenario_count=len(CONDITIONS)*len(LOADS)*len(HISTORY_SIZES)*repeats,
                  recency={'half_life_days': HALF_LIFE_DAYS,
                           'weight': '2**(-(as_of-planned_end).total_seconds()/86400/14)',
                           'history_score': '(sum(weight*success)+1)/(sum(weight)+2)',
                           'timestamp': 'planned_end; strictly before as_of',
                           'shared_inputs': 'all seven modes receive identical raw and weighted counts',
                           'legacy_modes': 'use raw integer successes/total and ignore weights',
                           'selection': 'fixed before v2 outcomes; not tuned'})
    config['simulator'].update(
        change_age_days=CHANGE_AGES,
        stable_conditions=['stable', 'no_time_effect'],
        change_rule='historical start >= as_of - change_age_days uses shifted peak; earlier starts use old peak',
        historical_timestamps='planned_end=start+15*ceil(estimated_minutes/15)',
        unseen_drift='change occurs at as_of; all available history precedes the change',
        future_drift_hours=8,
        fresh_history='observed drift may contain completed post-change outcomes; no future outcomes used')
    config['measurement'].update(
        warmup='one untimed invocation of each of seven methods on the first scenario',
        runtime_scope='seven modes of shared planner; excludes history aggregation; not standalone optimized algorithms',
        baseline_regression='each of five legacy modes compared with byte-frozen v1 engine outside timing')
    return config


def generate_case(condition, load_target, history_n, repeat, master_seed=MASTER_SEED):
    if condition not in CONDITIONS or load_target not in LOADS or history_n not in HISTORY_SIZES:
        raise ValueError('Unsupported experimental cell')
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 0:
        raise ValueError('repeat must be non-negative')
    seed = _seed(master_seed, condition, load_target, history_n, repeat)
    task_rng = random.Random(_seed(seed, 'tasks'))
    world_rng = random.Random(_seed(seed, 'world'))
    history_rng = random.Random(_seed(seed, 'history'))
    future_rng = random.Random(_seed(seed, 'future'))
    windows = [(_utc_local(legacy.FIRST_DAY+timedelta(days=i), 9),
                _utc_local(legacy.FIRST_DAY+timedelta(days=i), 18)) for i in range(2)]
    busy = [(_utc_local(legacy.FIRST_DAY+timedelta(days=i), a),
             _utc_local(legacy.FIRST_DAY+timedelta(days=i), b))
            for i in range(2) for a, b in ((12, 12.5), (15, 15.5))]
    as_of = _utc_local(legacy.FIRST_DAY, 8)
    change_at = as_of-timedelta(days=CHANGE_AGES[condition]) if condition in CHANGE_AGES else None
    past_peaks = {str(category): world_rng.uniform(8, 20) for category in (1, 2, 3)}
    future_peaks = {category: (peak+(8 if change_at is not None else 0)) % 24
                    for category, peak in past_peaks.items()}
    amplitude = 0 if condition == 'no_time_effect' else 0.28
    tasks, future, minutes = [], {}, 0
    while minutes < load_target*legacy.CAPACITY_MINUTES:
        task_id = len(tasks)+1
        duration = task_rng.choice(legacy.DURATIONS)
        allocated = _allocated(duration)
        importance, category = task_rng.randint(1, 5), task_rng.randint(1, 3)
        earliest = None
        if task_rng.random() >= 0.4:
            offset = 0 if task_rng.random() < 2/3 else 1
            earliest = _utc_local(legacy.FIRST_DAY+timedelta(days=offset), 9+task_rng.randint(0, 28)/4)
        release = earliest or windows[0][0]
        deadline = None
        if task_rng.random() >= 0.2:
            deadline = min(windows[-1][1], release+timedelta(minutes=allocated+task_rng.choice(legacy.SLACKS)))
        tasks.append({'id': task_id, 'title': f'Task {task_id}', 'category_id': category,
                      'importance': importance, 'duration_min': duration,
                      'earliest_start': earliest.isoformat() if earliest else None,
                      'deadline': deadline.isoformat() if deadline else None})
        future[str(task_id)] = {'difficulty': future_rng.uniform(-0.12, 0.12),
                                'noise': future_rng.gauss(0, 1)}
        minutes += allocated
    observations, records = [], []
    for category in (1, 2, 3):
        for part_index, part in enumerate(legacy.PARTS):
            for observation in range(history_n):
                hour = 6*part_index+6*history_rng.random()
                day = legacy.FIRST_DAY-timedelta(days=history_rng.randint(1, 60))
                start = _utc_local(day, hour)
                duration = history_rng.choice(legacy.DURATIONS)
                planned_end = start+timedelta(minutes=_allocated(duration))
                if planned_end >= as_of:
                    raise AssertionError('Generator produced a not-yet-observed history outcome')
                difficulty, noise = history_rng.uniform(-0.12, 0.12), history_rng.gauss(0, 1)
                post_change = change_at is not None and start >= change_at
                peak = (future_peaks if post_change else past_peaks)[str(category)]
                actual = duration_model(duration, hour, peak, amplitude, difficulty, noise)
                success = actual <= _allocated(duration)
                observations.append(HistoryObservation(category, part, planned_end, success))
                records.append({'category_id': category, 'daypart': part, 'observation': observation,
                                'start': start.isoformat(), 'planned_end': planned_end.isoformat(),
                                'hour_local': hour, 'estimated_minutes': duration,
                                'difficulty': difficulty, 'noise': noise,
                                'actual_minutes': actual, 'success': success,
                                'post_change': post_change, 'peak_used': peak})
    weighted = recency_counts(observations, as_of, HALF_LIFE_DAYS)
    history = [{'category_id': category, 'daypart': part,
                **asdict(weighted.get((category, part), WeightedHistoryCount(0, 0, 0., 0.)))}
               for category in (1, 2, 3) for part in legacy.PARTS]
    public_observations = [{'category_id': item.category_id, 'daypart': item.daypart,
                            'occurred_at': item.occurred_at.isoformat(), 'success': item.success}
                           for item in observations]
    case = {'schema_version': SCHEMA_VERSION,
            'scenario_id': f'{condition}_load{load_target}_n{history_n}_r{repeat:03}',
            'condition': condition, 'load_target': load_target, 'history_n': history_n,
            'repeat': repeat, 'seed': seed,
            'total_allocated_minutes': minutes, 'capacity_minutes': legacy.CAPACITY_MINUTES,
            'actual_load': minutes/legacy.CAPACITY_MINUTES,
            'post_change_history_count': sum(row['post_change'] for row in records),
            'planner': {'tasks': tasks, 'available_windows': [[a.isoformat(), b.isoformat()] for a, b in windows],
                        'busy_intervals': [[a.isoformat(), b.isoformat()] for a, b in busy],
                        'history': history, 'history_observations': public_observations,
                        'as_of': as_of.isoformat(), 'timezone_name': legacy.ZONE_NAME},
            'simulation': {'amplitude': amplitude, 'historical_peaks': past_peaks,
                           'future_peaks': future_peaks, 'future_task_noise': future,
                           'change_at': change_at.isoformat() if change_at else None},
            'history_records': records}
    feasible = individually_feasible_ids(case)
    case['individually_feasible_ids'] = feasible
    case['independently_feasible_count'] = len(feasible)
    case['independently_feasible_importance'] = sum(t['importance'] for t in tasks if t['id'] in feasible)
    return case


def planner_inputs(case):
    """Allow only task, calendar and already-observed historical evidence."""
    raw = case['planner']
    inputs = legacy.planner_inputs(case)
    inputs['history'] = {(row['category_id'], row['daypart']):
                         WeightedHistoryCount(row['successes'], row['total'],
                                              row['weighted_successes'], row['weighted_total'])
                         for row in raw['history']}
    return inputs


def check_legacy_equivalence(inputs, results):
    """Check decisions, scores and rejection reasons against the frozen engine."""
    for method in LEGACY_METHODS:
        expected = frozen_engine.generate_plan(**inputs, method=method)
        if legacy._signature(expected) != legacy._signature(results[method]):
            raise AssertionError(f'Legacy baseline changed: {method}')


def run_experiment(output, repeats=40, timing_repeats=3, master_seed=MASTER_SEED, progress=True):
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (repeats, timing_repeats)):
        raise ValueError('Positive integer repeat counts required')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Output must be new or empty; existing evidence will not be overwritten')
    output.mkdir(parents=True, exist_ok=True)
    config = configuration(repeats, timing_repeats, master_seed)
    (output/'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
    started = perf_counter()
    sources = ('experiments/run_comparison.py', 'experiments/frozen_v1/recommender.py',
               'experiments/protocol_v2.md')
    manifest = {'schema_version': SCHEMA_VERSION, 'status': 'running',
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'python': sys.version, 'platform': platform.platform(),
                'cpu': platform.processor() or os.environ.get('PROCESSOR_IDENTIFIER', 'unknown'),
                'logical_cpu_count': os.cpu_count(),
                'script_sha256': _sha(__file__), 'engine_sha256': _sha(recommender.__file__),
                'sources_sha256': {name: _sha(ROOT/name) for name in sources},
                'config_sha256': _sha(output/'config.json'),
                'independent_scenario_count': config['scenario_count'],
                'paired_methods_on_identical_inputs': True, 'database_access': False,
                'parameter_tuning': False, 'history_aggregation_in_timing': False}
    def write_manifest():
        (output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding='utf-8')
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
        for method in METHODS:
            generate_plan(**planner_inputs(cases[0]), method=method)
        rows, total_violations = [], 0
        with (output/'assignments.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
            for case_index, case in enumerate(cases):
                inputs = planner_inputs(case)
                results, timings = {}, {method: [] for method in METHODS}
                order_rng, orders = random.Random(_seed(case['seed'], 'timing_order')), []
                for timing_round in range(timing_repeats):
                    order = list(METHODS)
                    order_rng.shuffle(order)
                    orders.append(order)
                    for method in order:
                        before = perf_counter()
                        result = generate_plan(**inputs, method=method)
                        timings[method].append((perf_counter()-before)*1000)
                        if method in results and legacy._signature(result) != legacy._signature(results[method]):
                            raise AssertionError(f'Non-deterministic plan: {case["scenario_id"]}/{method}')
                        results[method] = result
                check_legacy_equivalence(inputs, results)
                public_hash = hashlib.sha256(_json(case['planner']).encode('utf-8')).hexdigest()
                for method in METHODS:
                    result = results[method]
                    metrics = evaluate_plan(case, result)
                    total_violations += metrics['total_violations']
                    rows.append({'scenario_id': case['scenario_id'], 'condition': case['condition'],
                                 'load_target': case['load_target'], 'actual_load': case['actual_load'],
                                 'history_n': case['history_n'], 'repeat': case['repeat'], 'seed': case['seed'],
                                 'post_change_history_count': case['post_change_history_count'],
                                 'method': method, 'planner_inputs_sha256': public_hash,
                                 'independently_feasible_count': case['independently_feasible_count'],
                                 'independently_feasible_importance': case['independently_feasible_importance'],
                                 **metrics, 'elapsed_ms': statistics.median(timings[method])})
                    assignments, tasks = [], {task.id: task for task in inputs['tasks']}
                    for assignment in result.assignments:
                        values = asdict(assignment)
                        actual = simulate_duration(case, tasks[assignment.task_id], assignment.start)
                        values['simulated_duration_minutes'] = actual
                        values['simulated_fits_interval'] = actual <= (assignment.end-assignment.start).total_seconds()/60
                        assignments.append(values)
                    stream.write(_json({'scenario_id': case['scenario_id'], 'method': method,
                                        'planner_inputs_sha256': public_hash, 'assignments': assignments,
                                        'unscheduled': [asdict(item) for item in result.unscheduled],
                                        'timing_samples_ms': timings[method], 'method_orders': orders})+'\n')
                if progress and ((case_index+1) % 40 == 0 or case_index+1 == len(cases)):
                    print(f'Completed {case_index+1}/{len(cases)} scenarios', flush=True)
        with (output/'case_metrics.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        manifest.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                        wall_time_seconds=perf_counter()-started,
                        scenario_count=len(cases), metric_rows=len(rows), total_violations=total_violations,
                        measured_planner_calls=len(cases)*len(METHODS)*timing_repeats,
                        warmup_planner_calls=len(METHODS),
                        frozen_baseline_equivalence_checks=len(cases)*len(LEGACY_METHODS),
                        outputs_sha256={name: _sha(output/name) for name in ('inputs.jsonl', 'assignments.jsonl', 'case_metrics.csv')})
        write_manifest()
        return manifest
    except Exception as error:
        manifest.update(status='failed', error=repr(error), wall_time_seconds=perf_counter()-started)
        write_manifest()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('experiments/results_v2'))
    parser.add_argument('--repeats', type=int, default=40)
    parser.add_argument('--timing-repeats', type=int, default=3)
    parser.add_argument('--master-seed', type=int, default=MASTER_SEED)
    args = parser.parse_args()
    manifest = run_experiment(args.output, args.repeats, args.timing_repeats, args.master_seed)
    print(json.dumps({key: manifest[key] for key in ('status', 'scenario_count', 'metric_rows',
                                                   'total_violations', 'wall_time_seconds')}, indent=2))


if __name__ == '__main__':
    main()
