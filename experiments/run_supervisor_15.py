"""Reconstruct the documented 15-task conditions; never fit the thesis numbers."""
import argparse
import copy
import csv
import hashlib
import json
import platform
import random
import statistics
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import perf_counter

from app import recommender
from app.recommender import generate_plan
from experiments import run_recency_comparison as v2

ROOT = Path(__file__).resolve().parent.parent
METHODS = ('edf', 'importance', *v2.METHODS[1:])
LABELS = dict(v2.METHOD_LABELS, importance='Лише важливість')
MASTER_SEED = 20261108
REPEATS = 200
CAPACITY = 1080
sha, encode = v2._sha, v2._json


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str,
                                    allow_nan=False), encoding='utf-8')


def generate_pair(repeat, master_seed=MASTER_SEED):
    if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 0:
        raise ValueError('Non-negative integer repeat required')
    case = v2.generate_case('stable', .7, 20, repeat, master_seed)
    first = v2.legacy.FIRST_DAY
    windows = [(v2._utc_local(first+timedelta(days=d), 8),
                v2._utc_local(first+timedelta(days=d), 18)) for d in range(3)]
    busy = [(v2._utc_local(first+timedelta(days=d), a),
             v2._utc_local(first+timedelta(days=d), b))
            for d in range(3) for a, b in ((10, 12), (14, 16))]
    rng = random.Random(v2._seed(master_seed, 'supervisor_tasks', repeat))
    future_rng = random.Random(v2._seed(master_seed, 'supervisor_future', repeat))
    possible_deadlines = [a+timedelta(minutes=15*k) for a, b in windows for k in range(41)]
    tasks, noise = [], {}
    for task_id in range(1, 16):
        duration = rng.choice(tuple(range(15, 121, 15)))
        importance, category = rng.choice((2, 3, 4)), rng.randint(1, 3)
        deadline = rng.choice([d for d in possible_deadlines if d >= windows[0][0]+timedelta(minutes=duration)])
        tasks.append({'id': task_id, 'title': f'Task {task_id}', 'category_id': category,
                      'importance': importance, 'duration_min': duration,
                      'earliest_start': None, 'deadline': deadline.isoformat()})
        noise[str(task_id)] = {'difficulty': future_rng.uniform(-.12, .12),
                              'noise': future_rng.gauss(0, 1)}
    case['planner'].update(tasks=tasks,
        available_windows=[[a.isoformat(), b.isoformat()] for a, b in windows],
        busy_intervals=[[a.isoformat(), b.isoformat()] for a, b in busy])
    case['simulation']['future_task_noise'] = noise
    case.update(schema_version='supervisor15-1.0', scenario_id=f'supervisor15_n20_r{repeat:03}',
                condition='supervisor_stable', load_target=.9375,
                total_allocated_minutes=sum(t['duration_min'] for t in tasks), capacity_minutes=CAPACITY)
    case['actual_load'] = case['total_allocated_minutes']/CAPACITY
    feasible = v2.individually_feasible_ids(case)
    case.update(individually_feasible_ids=feasible, independently_feasible_count=len(feasible),
                independently_feasible_importance=sum(t['importance'] for t in tasks if t['id'] in feasible))
    cold = copy.deepcopy(case)
    cold.update(history_n=0, scenario_id=f'supervisor15_n0_r{repeat:03}')
    cold['planner']['history_observations'] = []
    cold['history_records'] = []
    for row in cold['planner']['history']:
        row.update(successes=0, total=0, weighted_successes=0., weighted_total=0.)
    return case, cold


def evaluate(case, result):
    metrics = v2.evaluate_plan(case, result)
    by_id = {t['id']: t for t in case['planner']['tasks']}
    value = sum((by_id[a.task_id]['importance']-1)/4 for a in result.assignments)
    minutes = sum((a.end-a.start).total_seconds()/60 for a in result.assignments)
    return dict(metrics, normalized_value=value,
                normalized_input_value=sum((t['importance']-1)/4 for t in by_id.values()),
                utilization_percent=100*minutes/CAPACITY)


def run(output, repeats=REPEATS, timing_repeats=3, master_seed=MASTER_SEED, progress=True):
    if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in (repeats, timing_repeats)):
        raise ValueError('Positive integer repeat counts required')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Existing evidence cannot be overwritten')
    output.mkdir(parents=True, exist_ok=True)
    config = {'schema_version': 'supervisor15-1.0', 'master_seed': master_seed,
              'independent_task_sets': repeats, 'scenario_count': repeats*2,
              'history_sizes': [20, 0], 'methods': list(METHODS), 'method_labels': LABELS,
              'timing_repeats': timing_repeats, 'tasks_per_case': 15,
              'capacity_minutes': CAPACITY, 'window_minutes': 1800, 'busy_minutes': 720,
              'weights': [.8, .2], 'normalized_priorities': [.25, .5, .75],
              'illustration': 'supervisor15_n20_r000',
              'protocol': 'experiments/protocol_supervisor_15.md',
              'simulator': v2.configuration()['simulator'],
              'exact_supervisor_data_available': False, 'parameter_tuning': False}
    write_json(output/'config.json', config)
    sources = ('app/recommender.py', 'experiments/run_supervisor_15.py',
               'experiments/run_comparison.py', 'experiments/run_recency_comparison.py',
               'experiments/protocol_supervisor_15.md')
    manifest = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
                'python': sys.version, 'platform': platform.platform(),
                'sources_sha256': {p: sha(ROOT/p) for p in sources},
                'config_sha256': sha(output/'config.json'), 'database_access': False,
                'parameter_tuning': False, 'independent_task_sets': repeats}
    write_json(output/'manifest.json', manifest)
    started = perf_counter()
    try:
        cases = [case for repeat in range(repeats) for case in generate_pair(repeat, master_seed)]
        with (output/'inputs.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
            for case in cases:
                stream.write(encode(case)+'\n')
        for method in METHODS:
            generate_plan(**v2.planner_inputs(cases[0]), method=method)
        rows = []
        with (output/'assignments.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
            for index, case in enumerate(cases):
                inputs = v2.planner_inputs(case)
                results, times = {}, {m: [] for m in METHODS}
                order_rng = random.Random(v2._seed(case['seed'], case['history_n'], 'timing_order_15'))
                orders = []
                for _ in range(timing_repeats):
                    order = list(METHODS)
                    order_rng.shuffle(order)
                    orders.append(order)
                    for method in order:
                        before = perf_counter()
                        result = generate_plan(**inputs, method=method)
                        times[method].append(1000*(perf_counter()-before))
                        if method in results and v2.legacy._signature(result) != v2.legacy._signature(results[method]):
                            raise AssertionError('Non-deterministic planner')
                        results[method] = result
                metrics = {m: evaluate(case, r) for m, r in results.items()}
                edf_value = metrics['edf']['normalized_value']
                public_hash = hashlib.sha256(encode(case['planner']).encode('utf-8')).hexdigest()
                for method in METHODS:
                    result = results[method]
                    row = {'scenario_id': case['scenario_id'], 'repeat': case['repeat'],
                           'history_n': case['history_n'], 'method': method,
                           'actual_load': case['actual_load'], 'planner_inputs_sha256': public_hash,
                           **metrics[method], 'relative_value_edf': metrics[method]['normalized_value']/edf_value if edf_value else None,
                           'elapsed_ms': statistics.median(times[method]),
                           'elapsed_ms_mean': statistics.mean(times[method])}
                    rows.append(row)
                    assignments = []
                    tasks = {t.id: t for t in inputs['tasks']}
                    for assignment in result.assignments:
                        actual = v2.simulate_duration(case, tasks[assignment.task_id], assignment.start)
                        assignments.append(dict(asdict(assignment), simulated_duration_minutes=actual,
                            simulated_fits_interval=actual <= (assignment.end-assignment.start).total_seconds()/60))
                    stream.write(encode({'scenario_id': case['scenario_id'], 'method': method,
                        'planner_inputs_sha256': public_hash, 'assignments': assignments,
                        'unscheduled': [asdict(r) for r in result.unscheduled],
                        'timing_samples_ms': times[method], 'method_orders': orders})+'\n')
                if progress and (index+1) % 40 == 0:
                    print(f'Completed {index+1}/{len(cases)} scenarios', flush=True)
        with (output/'case_metrics.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        violations = sum(r['total_violations'] for r in rows)
        if violations:
            raise AssertionError(f'Constraint violations: {violations}')
        manifest.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                        scenario_count=len(cases), metric_rows=len(rows), total_violations=violations,
                        measured_planner_calls=len(rows)*timing_repeats, warmup_calls=len(METHODS),
                        wall_time_seconds=perf_counter()-started,
                        outputs_sha256={p: sha(output/p) for p in ('inputs.jsonl', 'assignments.jsonl', 'case_metrics.csv')})
        write_json(output/'manifest.json', manifest)
        return manifest
    except Exception as error:
        manifest.update(status='failed', error=repr(error))
        write_json(output/'manifest.json', manifest)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('experiments/results_supervisor_15'))
    parser.add_argument('--repeats', type=int, default=REPEATS)
    parser.add_argument('--timing-repeats', type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.repeats, args.timing_repeats), indent=2))
