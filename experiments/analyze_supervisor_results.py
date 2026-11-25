"""Audit and analyze the 15-task reconstruction and eight-method v2 extension.

No planner calls. Run with the bundled Python/NumPy runtime. The two experiments
remain separate. The original seven-method Q intervals are preserved exactly.
"""
import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEED = 20261109
REPEATS = 5000
Q = 'simulated_on_time_importance_fraction'
P = 'placed_fraction'
V = 'normalized_value'
CONTRASTS = ('edf', 'importance', 'personalized', 'fuzzy', 'weighted_recent')
STRINGS = {'scenario_id', 'condition', 'method', 'planner_inputs_sha256'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify_hash(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f'Checksum mismatch: {path}')
    return actual


def audit(source, expected_runner):
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    config = json.loads((source/'config.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete':
        raise ValueError('Experiment has not completed')
    expected = dict(manifest['outputs_sha256'], **{'config.json': manifest['config_sha256']})
    checked = {name: verify_hash(source/name, checksum) for name, checksum in expected.items()}
    sources = dict(manifest.get('sources_sha256', {}))
    if 'script_sha256' in manifest:
        sources[expected_runner] = manifest['script_sha256']
    if 'engine_sha256' in manifest:
        sources['app/recommender.py'] = manifest['engine_sha256']
    source_checks = {name: verify_hash(ROOT/name, checksum) for name, checksum in sources.items()}
    with (source/'case_metrics.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = [{key: value if key in STRINGS else float(value) if value else None
                 for key, value in row.items()} for row in csv.DictReader(stream)]
    groups = defaultdict(dict)
    for row in rows:
        sid, method = row['scenario_id'], row['method']
        if method in groups[sid]:
            raise ValueError('Duplicate paired result')
        groups[sid][method] = row
        for metric in (P, Q, 'scheduled_importance_fraction'):
            if not 0 <= row[metric] <= 100:
                raise ValueError('Metric outside percentage bounds')
        if row['total_violations'] != 0:
            raise ValueError('Invalid plan in reported results')
    if len(groups) != config['scenario_count'] or len(rows) != len(groups)*len(config['methods']):
        raise ValueError('Incorrect result count')
    for sid, group in groups.items():
        if set(group) != set(config['methods']):
            raise ValueError('Missing method')
        for field in ('planner_inputs_sha256', 'input_task_count', 'input_importance', 'history_n', 'repeat'):
            if len({row[field] for row in group.values()}) != 1:
                raise ValueError(f'Nonidentical paired input: {sid}/{field}')
    cases = {}
    with (source/'inputs.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            case = json.loads(line)
            if case['scenario_id'] in cases:
                raise ValueError('Duplicate input case')
            cases[case['scenario_id']] = case
    if set(cases) != set(groups):
        raise ValueError('Input case coverage differs')
    plans = {}
    with (source/'assignments.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            plan = json.loads(line)
            key = plan['scenario_id'], plan['method']
            if key in plans or key[0] not in groups or key[1] not in groups[key[0]]:
                raise ValueError('Unexpected plan')
            plans[key] = plan
            row = groups[key[0]][key[1]]
            if plan['planner_inputs_sha256'] != row['planner_inputs_sha256']:
                raise ValueError('Plan input checksum differs')
            if len(plan['assignments']) != row['placed_count']:
                raise ValueError('Placed counts differ')
    if len(plans) != len(rows):
        raise ValueError('Incomplete plan records')
    checks = {'status': 'passed', 'source_checksums': checked, 'code_checksums': source_checks,
              'scenarios': len(groups), 'paired_rows': len(rows), 'total_constraint_violations': 0}
    return config, manifest, groups, cases, plans, checks


def bootstrap(groups, methods, cells, metrics, rng):
    """Resample complete paired sets within each fixed cell."""
    count = sum(len(ids) for ids in cells.values())
    boot = {metric: np.zeros((REPEATS, len(methods))) for metric in metrics}
    means = {metric: np.zeros(len(methods)) for metric in metrics}
    for cell in sorted(cells):
        ids = sorted(cells[cell])
        indices = rng.integers(0, len(ids), (REPEATS, len(ids)))
        weight = len(ids)/count
        for metric in metrics:
            values = np.array([[groups[sid][method][metric] for method in methods] for sid in ids])
            means[metric] += weight*values.mean(axis=0)
            boot[metric] += weight*values[indices].mean(axis=1)
    return means, boot


def interval(values):
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def descriptive(rows):
    result = {'scenarios': len(rows)}
    metrics = (P, 'scheduled_importance_fraction', Q, 'placed_count', 'placed_importance',
               'input_task_count', 'candidate_count', 'actual_load')
    for metric in metrics:
        result[metric+'_mean'] = float(np.mean([row[metric] for row in rows]))
    for metric in (V, 'normalized_input_value', 'utilization_percent', 'elapsed_ms_mean'):
        if metric in rows[0]:
            result[metric+'_mean'] = float(np.mean([row[metric] for row in rows]))
    result['elapsed_ms_median'] = float(np.median([row['elapsed_ms'] for row in rows]))
    result['total_violations'] = int(sum(row['total_violations'] for row in rows))
    return result


def save_analysis(source, output, result, checks, extra_sources=None):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Analysis folder must be new or empty')
    output.mkdir(parents=True, exist_ok=True)
    files = []
    for name in ('summary', 'paired_comparisons', 'illustration', 'illustration_tasks', 'illustration_assignments'):
        if result.get(name):
            write_csv(output/(name+'.csv'), result[name])
            files.append(name+'.csv')
    write_json(output/'analysis.json', result)
    files.append('analysis.json')
    write_json(output/'analysis_manifest.json', {
        'status': 'complete', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
        'analysis_script_sha256': sha(__file__), 'source_manifest_sha256': sha(source/'manifest.json'),
        'source_checksums': checks['source_checksums'], 'code_checksums': checks['code_checksums'],
        'extra_sources': extra_sources or {}, 'parameters': result['parameters'],
        'output_checksums': {name: sha(output/name) for name in files}})


def analyze_supervisor(source, output):
    config, manifest, groups, cases, plans, checks = audit(source, 'experiments/run_supervisor_15.py')
    methods, rng = config['methods'], np.random.default_rng(SEED)
    summary, paired = [], []
    paired_sets = defaultdict(dict)
    for sid, case in cases.items():
        paired_sets[case['repeat']][case['history_n']] = case
    if len(paired_sets) != config['independent_task_sets']:
        raise ValueError('Incorrect independent-set count')
    for pair in paired_sets.values():
        if set(pair) != {0, 20}:
            raise ValueError('Incomplete history pair')
        for name in ('tasks', 'available_windows', 'busy_intervals'):
            if pair[0]['planner'][name] != pair[20]['planner'][name]:
                raise ValueError('History sensitivity uses different tasks/calendar')
        if pair[0]['simulation'] != pair[20]['simulation']:
            raise ValueError('History sensitivity uses different latent future')
    cold_count = 0
    for sid, case in cases.items():
        if case['history_n'] == 0:
            positions = lambda method: [(a['task_id'], a['start'], a['end']) for a in plans[sid, method]['assignments']]
            if any(positions(method) != positions('fuzzy') for method in ('personalized', 'personalized_recent')):
                raise ValueError('Cold-start assignment mismatch')
            cold_count += 1
    checks.update(independent_task_sets=len(paired_sets), cold_start_identical_scenarios=cold_count)
    for n in config['history_sizes']:
        ids = sorted(sid for sid in groups if groups[sid][methods[0]]['history_n'] == n)
        if len(ids) != config['independent_task_sets']:
            raise ValueError('Incorrect history-group size')
        means, boot = bootstrap(groups, methods, {'all': ids}, (P, V, Q), rng)
        edf = methods.index('edf')
        for index, method in enumerate(methods):
            row = {'history_n': n, 'method': method, 'label': config['method_labels'][method],
                   **descriptive([groups[sid][method] for sid in ids])}
            for metric, short in ((P, 'p'), (V, 'v'), (Q, 'q')):
                low, high = interval(boot[metric][:, index])
                row.update({short+'_mean': float(means[metric][index]), short+'_ci_low': low, short+'_ci_high': high})
            denominator = means[V][edf]
            ratio_boot = np.divide(boot[V][:, index], boot[V][:, edf],
                                   out=np.full(REPEATS, np.nan), where=boot[V][:, edf] != 0)
            valid = ratio_boot[np.isfinite(ratio_boot)]
            low, high = interval(valid) if len(valid) else (None, None)
            row.update(c_ratio_of_means=float(means[V][index]/denominator) if denominator else None,
                       c_ci_low=low, c_ci_high=high, c_bootstrap_defined_repeats=len(valid))
            summary.append(row)
        left = methods.index('personalized_recent')
        for right_method in CONTRASTS:
            right = methods.index(right_method)
            for metric, units in ((P, 'percentage_points'), (V, 'normalized_value_units'), (Q, 'percentage_points')):
                low, high = interval(boot[metric][:, left]-boot[metric][:, right])
                paired.append({'history_n': n, 'independent_sets': len(ids), 'left': 'personalized_recent',
                               'right': right_method, 'metric': metric, 'units': units,
                               'difference': float(means[metric][left]-means[metric][right]),
                               'ci_low': low, 'ci_high': high})
    sid = config['illustration']
    illustration = [dict(method=method, label=config['method_labels'][method], **{k: groups[sid][method][k]
                    for k in ('placed_count', P, V, 'relative_value_edf', Q, 'utilization_percent',
                              'deadline_violations', 'total_violations', 'elapsed_ms', 'elapsed_ms_mean')})
                    for method in methods]
    local = ZoneInfo('Europe/Kyiv')
    def local_time(value):
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc).astimezone(local).strftime('%Y-%m-%d %H:%M') if value else None
    tasks = [dict(task_id=task['id'], category_id=task['category_id'], importance_p=task['importance'],
                  importance_I=(task['importance']-1)/4, duration_min=task['duration_min'],
                  earliest_start_local=local_time(task['earliest_start']), deadline_local=local_time(task['deadline']))
             for task in cases[sid]['planner']['tasks']]
    assignments = []
    for method in methods:
        by_id = {a['task_id']: a for a in plans[sid, method]['assignments']}
        rejections = {a['task_id']: a['reason'] for a in plans[sid, method]['unscheduled']}
        for task in tasks:
            a = by_id.get(task['task_id'])
            assignments.append({'method': method, 'task_id': task['task_id'], 'placed': a is not None,
                                'start_local': local_time(a['start']) if a else None,
                                'end_local': local_time(a['end']) if a else None,
                                'score': a['score'] if a else None,
                                'simulated_duration_minutes': a['simulated_duration_minutes'] if a else None,
                                'simulated_fits_interval': a['simulated_fits_interval'] if a else None,
                                'rejection': rejections.get(task['task_id'])})
    parameters = {'bootstrap_seed': SEED, 'bootstrap_repeats': REPEATS,
                  'interval': '95% paired percentile; whole independent task sets; separate N=20/N=0',
                  'multiple_comparison_adjustment': False, 'ratio': 'mean V(method) / mean V(EDF)',
                  'contrasts': list(CONTRASTS), 'metrics': [P, V, Q],
                  'uncertainty_scope': 'given synthetic generator only', 'illustration': sid,
                  'exact_supervisor_replication': False, 'parameter_tuning': False}
    result = {'parameters': parameters, 'integrity_checks': checks, 'summary': summary,
              'paired_comparisons': paired, 'illustration': illustration,
              'illustration_tasks': tasks, 'illustration_assignments': assignments}
    save_analysis(source, output, result, checks)
    return result


def analyze_extension(source, output, archive):
    config, manifest, groups, cases, plans, checks = audit(source, 'experiments/run_extended_comparison.py')
    old_manifest = json.loads((archive/'manifest.json').read_text(encoding='utf-8'))
    extra = {'archive_manifest': verify_hash(archive/'manifest.json', manifest['archive_manifest_sha256'])}
    verify_hash(archive/'inputs.jsonl', old_manifest['inputs_sha256'])
    if sha(source/'inputs.jsonl') != sha(archive/'inputs.jsonl'):
        raise ValueError('Extended inputs changed')
    analysis_manifest_path = archive/'analysis/analysis_manifest.json'
    am = json.loads(analysis_manifest_path.read_text(encoding='utf-8'))
    verify_hash(archive/'manifest.json', am['source_manifest_sha256'])
    for name, checksum in am['source_checksums'].items():
        verify_hash(archive/name, checksum)
    for name, checksum in am['output_checksums'].items():
        verify_hash(archive/'analysis'/name, checksum)
    verify_hash(ROOT/'experiments/frozen_v2/recommender.py', old_manifest['engine_sha256'])
    verify_hash(ROOT/'experiments/analyze_recency_comparison.py', am['analysis_script_sha256'])
    old = json.loads((archive/'analysis/analysis.json').read_text(encoding='utf-8'))
    extra.update(archived_analysis=sha(archive/'analysis/analysis.json'),
                 archived_analysis_manifest=sha(analysis_manifest_path))
    cells = defaultdict(list)
    methods = config['methods']
    for sid, group in groups.items():
        row = group[methods[0]]
        cells[(row['condition'], row['load_target'], row['history_n'])].append(sid)
    if len(cells) != 36 or any(len(ids) != 40 for ids in cells.values()):
        raise ValueError('Unexpected extended v2 strata')
    means, boot = bootstrap(groups, methods, cells, (Q,), np.random.default_rng(SEED))
    archived = {row['method']: row for row in old['summary']}
    summary = []
    for index, method in enumerate(methods):
        row = {'method': method, 'label': config['method_labels'][method],
               **descriptive([groups[sid][method] for sid in sorted(groups)])}
        if method in archived:
            previous = archived[method]
            if not np.isclose(means[Q][index], previous['q_mean'], rtol=0, atol=1e-12):
                raise ValueError('Archived mean changed')
            row.update({name: previous[name] for name in ('q_mean', 'q_ci_low', 'q_ci_high', 'q_monte_carlo_se')})
            row['interval_source'] = 'verified original v2 analysis; seed 20261107'
        else:
            low, high = interval(boot[Q][:, index])
            row.update(q_mean=float(means[Q][index]), q_ci_low=low, q_ci_high=high, q_monte_carlo_se=None,
                       interval_source='new paired stratified analysis; seed 20261109')
        summary.append(row)
    left, right = methods.index('personalized_recent'), methods.index('importance')
    low, high = interval(boot[Q][:, left]-boot[Q][:, right])
    paired = [{'scope': 'overall', 'scenarios': len(groups), 'left': 'personalized_recent', 'right': 'importance',
               'metric': Q, 'difference_pp': float(means[Q][left]-means[Q][right]),
               'ci_low_pp': low, 'ci_high_pp': high}]
    parameters = {'bootstrap_seed_new_interval': SEED, 'bootstrap_repeats': REPEATS,
                  'interval': '95% paired percentile, stratified by condition x load x history (36 cells)',
                  'archived_seven_method_intervals_preserved': True,
                  'archived_recency_comparisons_preserved': True, 'new_independent_inputs': False,
                  'multiple_comparison_adjustment': False, 'uncertainty_scope': 'given synthetic generator only'}
    result = {'parameters': parameters, 'integrity_checks': checks, 'summary': summary,
              'paired_comparisons': paired, 'archived_recency_comparisons': old['paired_comparisons']}
    save_analysis(source, output, result, checks, extra)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment', choices=('supervisor', 'extension'))
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    source = args.input or ROOT/'experiments'/('results_supervisor_15' if args.experiment == 'supervisor' else 'results_v2_extended')
    output = args.output or source/'analysis'
    result = (analyze_supervisor(source, output) if args.experiment == 'supervisor'
              else analyze_extension(source, output, ROOT/'experiments/results_v2'))
    print(json.dumps({'summary': result['summary'], 'paired_comparisons': result['paired_comparisons']},
                     ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
