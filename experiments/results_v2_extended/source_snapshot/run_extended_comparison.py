"""Replay the immutable v2 inputs with eight methods and verify archived plans."""
import argparse
import csv
import hashlib
import json
import os
import platform
import random
import shutil
import statistics
import sys
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from app import recommender
from app.recommender import generate_plan
from experiments import run_recency_comparison as v2


ROOT = Path(__file__).resolve().parent.parent
METHODS = (*v2.METHODS, 'importance')
LABELS = {**v2.METHOD_LABELS, 'importance': 'Лише за важливістю'}
SCHEMA_VERSION = '2.1-extension'
_json, _sha = v2._json, v2._sha


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    sort_keys=True, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def audit_archive(source):
    """Validate immutable outputs and the byte-exact old implementation snapshot."""
    source = Path(source)
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete':
        raise ValueError('The archived run did not complete')
    checks = {**manifest['outputs_sha256'], 'config.json': manifest['config_sha256']}
    for name, expected in checks.items():
        if _sha(source/name) != expected:
            raise ValueError(f'Archived evidence checksum mismatch: {name}')
    snapshot = ROOT/'experiments/frozen_v2/recommender.py'
    if _sha(snapshot) != manifest['engine_sha256']:
        raise ValueError('Frozen v2 engine differs from the archived run')
    for name, expected in manifest['sources_sha256'].items():
        if _sha(ROOT/name) != expected:
            raise ValueError(f'Archived dependency changed: {name}')
    if _sha(ROOT/'experiments/run_recency_comparison.py') != manifest['script_sha256']:
        raise ValueError('The original v2 runner changed')
    config = json.loads((source/'config.json').read_text(encoding='utf-8'))
    if tuple(config['methods']) != tuple(v2.METHODS):
        raise ValueError('Unexpected archived method definitions')
    return config, manifest, checks


def serialized_assignments(result):
    return json.loads(_json([asdict(a) for a in result.assignments]))


def compare_archived_result(result, archived, archived_metrics):
    """All planner fields, rejection reasons, and candidate counts must match."""
    names = {field.name for field in fields(recommender.Assignment)}
    expected = [{k: value for k, value in item.items() if k in names}
                for item in archived['assignments']]
    actual = serialized_assignments(result)
    rejected = [asdict(item) for item in result.unscheduled]
    if (actual != expected or rejected != archived['unscheduled']
            or result.candidate_count != int(float(archived_metrics['candidate_count']))):
        raise AssertionError(f'Archived decisions changed: {archived["scenario_id"]}/{archived["method"]}')


def relative_value(placed_importance, edf_importance):
    return placed_importance/edf_importance if edf_importance else None


def summarize(rows):
    summaries = []
    for method in METHODS:
        selected = [row for row in rows if row['method'] == method]
        ratios = [row['relative_scheduled_importance_edf'] for row in selected
                  if row['relative_scheduled_importance_edf'] is not None]
        summary = {'method': method, 'label': LABELS[method], 'scenarios': len(selected)}
        for name in ('placed_fraction', 'scheduled_importance_fraction',
                     'simulated_on_time_importance_fraction', 'planned_deadline_violation_fraction'):
            values = [row[name] for row in selected if row[name] is not None]
            summary[name+'_mean'] = statistics.mean(values) if values else None
        summary.update(relative_scheduled_importance_edf_mean=statistics.mean(ratios) if ratios else None,
                       relative_scheduled_importance_edf_defined_scenarios=len(ratios),
                       aggregate_importance_ratio_edf=relative_value(
                           sum(row['placed_importance'] for row in selected),
                           sum(row['edf_placed_importance'] for row in selected)),
                       elapsed_ms_median=statistics.median(row['elapsed_ms'] for row in selected),
                       total_violations=sum(row['total_violations'] for row in selected))
        summaries.append(summary)
    return summaries


def run_experiment(source, output, timing_repeats=3, progress=True):
    if isinstance(timing_repeats, bool) or not isinstance(timing_repeats, int) or timing_repeats < 1:
        raise ValueError('Positive integer timing repeats required')
    source, output = Path(source), Path(output)
    config, old_manifest, archive_checks = audit_archive(source)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Output must be new or empty; old evidence will not be overwritten')
    output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source/'inputs.jsonl', output/'inputs.jsonl')
    config.update(schema_version=SCHEMA_VERSION, methods=list(METHODS), method_labels=LABELS,
                  extension={'inputs': 'byte-identical saved v2 inputs; no newly independent scenarios',
                             'new_baseline': 'importance=(p-1)/4; same feasibility and tie breaks',
                             'relative_value': 'placed raw importance p / EDF placed raw importance p per scenario',
                             'undefined_ratio': 'null when EDF places zero importance',
                             'post_hoc_extension': True, 'parameter_tuning': False})
    config['measurement'].update(timing_repeats=timing_repeats,
                                 warmup='one untimed invocation of each of eight methods on first scenario',
                                 runtime_scope='eight modes of shared planner; excludes history aggregation and DB',
                                 baseline_regression='all seven old methods compared to archived assignments and metrics')
    write_json(output/'config.json', config)
    sources = ('experiments/run_comparison.py', 'experiments/run_recency_comparison.py',
               'experiments/frozen_v1/recommender.py', 'experiments/frozen_v2/recommender.py',
               'experiments/protocol_v2_extended.md')
    started = perf_counter()
    manifest = {'schema_version': SCHEMA_VERSION, 'status': 'running',
                'started_utc': datetime.now(timezone.utc).isoformat(),
                'python': sys.version, 'platform': platform.platform(),
                'cpu': platform.processor() or os.environ.get('PROCESSOR_IDENTIFIER', 'unknown'),
                'logical_cpu_count': os.cpu_count(), 'script_sha256': _sha(__file__),
                'engine_sha256': _sha(recommender.__file__),
                'sources_sha256': {name: _sha(ROOT/name) for name in sources},
                'config_sha256': _sha(output/'config.json'),
                'archive_manifest_sha256': _sha(source/'manifest.json'),
                'archive_checks': archive_checks, 'inputs_sha256': _sha(output/'inputs.jsonl'),
                'independent_scenario_count': config['scenario_count'],
                'paired_methods_on_identical_inputs': True, 'database_access': False,
                'parameter_tuning': False, 'history_aggregation_in_timing': False,
                'source_engine_sha256': old_manifest['engine_sha256']}
    write_json(output/'manifest.json', manifest)
    try:
        cases = [json.loads(line) for line in (output/'inputs.jsonl').read_text(encoding='utf-8').splitlines()]
        archived = {}
        with (source/'assignments.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                item = json.loads(line)
                key = (item['scenario_id'], item['method'])
                if key in archived:
                    raise ValueError('Duplicate archived assignment record')
                archived[key] = item
        with (source/'case_metrics.csv').open(encoding='utf-8', newline='') as stream:
            archived_metrics = {(row['scenario_id'], row['method']): row for row in csv.DictReader(stream)}
        if len(cases) != config['scenario_count'] or len({case['scenario_id'] for case in cases}) != len(cases):
            raise ValueError('Invalid archived scenario count or duplicate case')
        if len(archived) != len(cases)*len(v2.METHODS) or set(archived) != set(archived_metrics):
            raise ValueError('Incomplete archived result pairing')
        for method in METHODS:
            generate_plan(**v2.planner_inputs(cases[0]), method=method)
        rows, total_violations, regression_checks = [], 0, 0
        with (output/'assignments.jsonl').open('w', encoding='utf-8', newline='\n') as stream:
            for case_index, case in enumerate(cases):
                inputs = v2.planner_inputs(case)
                results, timings = {}, {method: [] for method in METHODS}
                order_rng = random.Random(v2._seed(case['seed'], 'eight_method_timing_order'))
                orders = []
                for timing_round in range(timing_repeats):
                    order = list(METHODS)
                    order_rng.shuffle(order)
                    orders.append(order)
                    for method in order:
                        before = perf_counter()
                        result = generate_plan(**inputs, method=method)
                        timings[method].append((perf_counter()-before)*1000)
                        if method in results and v2.legacy._signature(result) != v2.legacy._signature(results[method]):
                            raise AssertionError('Repeated timing calls produced different decisions')
                        results[method] = result
                metrics_by_method = {method: v2.evaluate_plan(case, result) for method, result in results.items()}
                public_hash = hashlib.sha256(_json(case['planner']).encode('utf-8')).hexdigest()
                for method in v2.METHODS:
                    key = (case['scenario_id'], method)
                    if archived[key]['planner_inputs_sha256'] != public_hash:
                        raise ValueError('Archived plan input hash differs')
                    compare_archived_result(results[method], archived[key], archived_metrics[key])
                    for metric, value in metrics_by_method[method].items():
                        previous = archived_metrics[key][metric]
                        previous = float(previous) if previous else None
                        if value != previous:
                            raise AssertionError(f'Archived metric changed: {key}/{metric}')
                    regression_checks += 1
                edf_importance = metrics_by_method['edf']['placed_importance']
                for method in METHODS:
                    result, metrics = results[method], metrics_by_method[method]
                    if metrics['total_violations']:
                        raise AssertionError(f'Invalid plan: {case["scenario_id"]}/{method}')
                    total_violations += metrics['total_violations']
                    row = {'scenario_id': case['scenario_id'], 'condition': case['condition'],
                           'load_target': case['load_target'], 'actual_load': case['actual_load'],
                           'history_n': case['history_n'], 'repeat': case['repeat'], 'seed': case['seed'],
                           'post_change_history_count': case['post_change_history_count'],
                           'method': method, 'planner_inputs_sha256': public_hash,
                           'independently_feasible_count': case['independently_feasible_count'],
                           'independently_feasible_importance': case['independently_feasible_importance'],
                           **metrics, 'edf_placed_importance': edf_importance,
                           'relative_scheduled_importance_edf': relative_value(metrics['placed_importance'], edf_importance),
                           'planned_deadline_violation_fraction': (100*metrics['deadline_violations']/metrics['placed_count']
                                                                  if metrics['placed_count'] else None),
                           'elapsed_ms': statistics.median(timings[method])}
                    rows.append(row)
                    assignments, tasks = [], {task.id: task for task in inputs['tasks']}
                    for assignment in result.assignments:
                        values = asdict(assignment)
                        actual = v2.simulate_duration(case, tasks[assignment.task_id], assignment.start)
                        values['simulated_duration_minutes'] = actual
                        values['simulated_fits_interval'] = actual <= (assignment.end-assignment.start).total_seconds()/60
                        assignments.append(values)
                    stream.write(_json({'scenario_id': case['scenario_id'], 'method': method,
                                        'planner_inputs_sha256': public_hash, 'assignments': assignments,
                                        'unscheduled': [asdict(item) for item in result.unscheduled],
                                        'timing_samples_ms': timings[method], 'method_orders': orders})+'\n')
                if progress and ((case_index+1) % 120 == 0 or case_index+1 == len(cases)):
                    print(f'Completed {case_index+1}/{len(cases)} scenarios', flush=True)
        write_csv(output/'case_metrics.csv', rows)
        write_csv(output/'descriptive_summary.csv', summarize(rows))
        # Archive the implementation used; replaying future versions remains auditable.
        (output/'source_snapshot').mkdir()
        shutil.copyfile(recommender.__file__, output/'source_snapshot/recommender.py')
        shutil.copyfile(__file__, output/'source_snapshot/run_extended_comparison.py')
        manifest.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat(),
                        wall_time_seconds=perf_counter()-started, scenario_count=len(cases),
                        metric_rows=len(rows), total_violations=total_violations,
                        measured_planner_calls=len(cases)*len(METHODS)*timing_repeats,
                        warmup_planner_calls=len(METHODS),
                        archived_result_equivalence_checks=regression_checks,
                        outputs_sha256={name: _sha(output/name) for name in
                                        ('inputs.jsonl', 'assignments.jsonl', 'case_metrics.csv',
                                         'descriptive_summary.csv', 'source_snapshot/recommender.py',
                                         'source_snapshot/run_extended_comparison.py')})
        write_json(output/'manifest.json', manifest)
        return manifest
    except Exception as error:
        manifest.update(status='failed', error=repr(error), wall_time_seconds=perf_counter()-started)
        write_json(output/'manifest.json', manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'experiments/results_v2')
    parser.add_argument('--output', type=Path, default=ROOT/'experiments/results_v2_extended')
    parser.add_argument('--timing-repeats', type=int, default=3)
    args = parser.parse_args()
    result = run_experiment(args.source, args.output, args.timing_repeats)
    print(json.dumps({key: result[key] for key in ('status', 'scenario_count', 'metric_rows',
                                                  'total_violations', 'archived_result_equivalence_checks',
                                                  'wall_time_seconds')}, indent=2))


if __name__ == '__main__':
    main()
