"""Reproducible paired analysis of the frozen experiment; no planner calls.

Use the bundled Python runtime (NumPy is required). Plotting is deliberately
separate. The intervals concern Monte Carlo uncertainty under this generator,
not uncertainty about a real population of users.
"""
import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


BOOTSTRAP_SEED = 20261105
BOOTSTRAP_REPEATS = 5000
Q = 'simulated_on_time_importance_fraction'
MEAN_METRICS = ('placed_fraction', 'scheduled_importance_fraction', Q,
                'fit_fraction_of_placed', 'input_task_count', 'candidate_count')
CONTRASTS = (('personalized', 'fuzzy'), ('personalized', 'weighted_history'))


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


def audit_inputs(source):
    manifest = json.loads((source/'manifest.json').read_text(encoding='utf-8'))
    config = json.loads((source/'config.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete':
        raise ValueError('The main experiment has not completed')
    checked = {}
    expected = dict(manifest['outputs_sha256'], **{'config.json': manifest['config_sha256']})
    for filename, checksum in expected.items():
        actual = sha(source/filename)
        if actual != checksum:
            raise ValueError(f'Checksum mismatch: {filename}')
        checked[filename] = actual
    # Confirm that the scripts associated with the reported run still match.
    root = Path(__file__).resolve().parent.parent
    for path, expected_sha in ((root/'experiments/run_comparison.py', manifest['script_sha256']),
                               (root/'app/recommender.py', manifest['engine_sha256'])):
        if sha(path) != expected_sha:
            raise ValueError(f'Current implementation differs from the run: {path.name}')

    with (source/'case_metrics.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    methods = config['methods']
    string_fields = {'scenario_id', 'condition', 'method', 'planner_inputs_sha256'}
    integer_fields = {'history_n', 'repeat', 'seed'}
    grouped = defaultdict(dict)
    for row in rows:
        for field, value in row.items():
            if field in integer_fields:
                row[field] = int(value)
            elif field not in string_fields:
                row[field] = float(value) if value else None
        scenario, method = row['scenario_id'], row['method']
        if method in grouped[scenario]:
            raise ValueError(f'Duplicate paired record: {scenario}/{method}')
        grouped[scenario][method] = row
    if len(grouped) != config['scenario_count'] or len(rows) != len(grouped)*len(methods):
        raise ValueError('Incorrect scenario or result count')
    cells = defaultdict(list)
    for scenario, group in grouped.items():
        if set(group) != set(methods):
            raise ValueError(f'Missing method in {scenario}')
        fields = ('condition', 'load_target', 'history_n', 'repeat', 'seed',
                  'planner_inputs_sha256', 'input_task_count', 'input_importance',
                  'independently_feasible_count', 'independently_feasible_importance')
        for field in fields:
            if len({row[field] for row in group.values()}) != 1:
                raise ValueError(f'Nonidentical paired input {scenario}/{field}')
        first = group[methods[0]]
        cells[(first['condition'], first['load_target'], first['history_n'])].append(scenario)
        for row in group.values():
            for metric in ('placed_fraction', 'scheduled_importance_fraction', Q):
                if not 0 <= row[metric] <= 100:
                    raise ValueError(f'Out of bounds metric {scenario}/{metric}')
            if row[Q] > row['scheduled_importance_fraction'] + 1e-9:
                raise ValueError('Interval-fit importance exceeds scheduled importance')
    expected_cells = {(condition, load, n) for condition in config['conditions']
                      for load in config['load_targets']
                      for n in config['history_sizes_per_category_daypart']}
    if set(cells) != expected_cells or any(len(v) != config['repeats_per_cell'] for v in cells.values()):
        raise ValueError('Experimental cell sizes differ from the frozen specification')
    for ids in cells.values():
        ids.sort()

    cold = {}
    plans_seen = set()
    with (source/'assignments.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            plan = json.loads(line)
            key = (plan['scenario_id'], plan['method'])
            if key in plans_seen or plan['scenario_id'] not in grouped:
                raise ValueError('Unexpected or duplicate plan record')
            plans_seen.add(key)
            row = grouped[plan['scenario_id']][plan['method']]
            if row['planner_inputs_sha256'] != plan['planner_inputs_sha256']:
                raise ValueError('Plan and metrics input hashes differ')
            if row['history_n'] == 0 and plan['method'] in ('personalized', 'fuzzy'):
                signature = ([(a['task_id'], a['start'], a['end']) for a in plan['assignments']],
                             [(a['task_id'], a['reason']) for a in plan['unscheduled']])
                cold.setdefault(plan['scenario_id'], {})[plan['method']] = signature
    if len(plans_seen) != len(rows):
        raise ValueError('Missing plan records')
    if any(v.get('personalized') != v.get('fuzzy') for v in cold.values()):
        raise ValueError('Cold-start plans differ between personalized and fuzzy')
    cold_expected = sum(group[methods[0]]['history_n'] == 0 for group in grouped.values())
    if len(cold) != cold_expected:
        raise ValueError('Missing cold-start check')
    checks = {'status': 'passed', 'hashes': checked, 'scenarios': len(grouped),
              'paired_rows': len(rows), 'experimental_cells': len(cells),
              'cold_start_identical_assignment_scenarios': len(cold),
              'total_constraint_violations': int(sum(row['total_violations'] for row in rows))}
    return config, manifest, grouped, cells, checks


def stratified_estimates(groups, methods, cells, rng):
    """Whole-scenario resampling retains paired outcomes and fixed cell weights."""
    total = sum(len(ids) for ids in cells.values())
    k = len(methods)
    bootstrap = np.zeros((BOOTSTRAP_REPEATS, k))
    mean = np.zeros(k)
    variance_mean = np.zeros(k)
    contrasts = np.zeros(len(CONTRASTS))
    variance_contrasts = np.zeros(len(CONTRASTS))
    for cell in sorted(cells):
        ids = cells[cell]
        values = np.asarray([[groups[sid][method][Q] for method in methods] for sid in ids])
        n = len(values)
        weight = n/total
        mean += weight*values.mean(axis=0)
        variance_mean += weight**2*values.var(axis=0, ddof=1)/n
        indices = rng.integers(0, n, size=(BOOTSTRAP_REPEATS, n))
        bootstrap += weight*values[indices].mean(axis=1)
        for c, (left, right) in enumerate(CONTRASTS):
            differences = values[:, methods.index(left)]-values[:, methods.index(right)]
            contrasts[c] += weight*differences.mean()
            variance_contrasts[c] += weight**2*differences.var(ddof=1)/n
    intervals = np.percentile(bootstrap, [2.5, 97.5], axis=0)
    method_stats = {method: {'q_mean': float(mean[i]), 'q_ci_low': float(intervals[0, i]),
                             'q_ci_high': float(intervals[1, i]),
                             'q_monte_carlo_se': float(math.sqrt(variance_mean[i]))}
                    for i, method in enumerate(methods)}
    contrast_stats = []
    for c, (left, right) in enumerate(CONTRASTS):
        differences = bootstrap[:, methods.index(left)]-bootstrap[:, methods.index(right)]
        low, high = np.percentile(differences, [2.5, 97.5])
        contrast_stats.append({'left': left, 'right': right,
                               'difference_pp': float(contrasts[c]),
                               'ci_low_pp': float(low), 'ci_high_pp': float(high),
                               'monte_carlo_se_pp': float(math.sqrt(variance_contrasts[c]))})
    return method_stats, contrast_stats


def metric_summary(rows):
    result = {'scenarios': len(rows)}
    for metric in MEAN_METRICS:
        values = [row[metric] for row in rows if row[metric] is not None]
        result[metric+'_mean'] = float(np.mean(values)) if values else None
    result['elapsed_ms_median'] = float(np.median([row['elapsed_ms'] for row in rows]))
    result['elapsed_ms_p95'] = float(np.percentile([row['elapsed_ms'] for row in rows], 95))
    result['total_violations'] = int(sum(row['total_violations'] for row in rows))
    return result


def analyze(source, output):
    source, output = Path(source), Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Analysis output must be new or empty')
    config, manifest, groups, cells, checks = audit_inputs(source)
    methods = config['methods']
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    overall_stats, overall_contrasts = stratified_estimates(groups, methods, cells, rng)
    summary = []
    for method in methods:
        rows = [groups[sid][method] for sid in sorted(groups)]
        summary.append({'method': method, 'label': config['method_labels'][method],
                        **metric_summary(rows), **overall_stats[method]})
    paired = [{'scope': 'overall', 'condition': 'all', 'history_n': 'all',
               'scenarios': len(groups), **row} for row in overall_contrasts]
    subgroups = []
    for condition in config['conditions']:
        for n in config['history_sizes_per_category_daypart']:
            selected = {key: ids for key, ids in cells.items() if key[0] == condition and key[2] == n}
            stats, contrasts = stratified_estimates(groups, methods, selected, rng)
            ids = sorted(sid for values in selected.values() for sid in values)
            for row in contrasts:
                paired.append({'scope': 'condition_history', 'condition': condition,
                               'history_n': n, 'scenarios': len(ids), **row})
            for method in methods:
                subgroups.append({'scope': 'condition_history', 'condition': condition,
                                  'load_target': 'both', 'history_n': n, 'method': method,
                                  **metric_summary([groups[sid][method] for sid in ids]), **stats[method]})
            for key in sorted(selected):
                for method in methods:
                    rows = [groups[sid][method] for sid in selected[key]]
                    subgroups.append({'scope': 'cell', 'condition': condition,
                                      'load_target': key[1], 'history_n': n, 'method': method,
                                      **metric_summary(rows),
                                      'q_mean': float(np.mean([row[Q] for row in rows])),
                                      'q_ci_low': None, 'q_ci_high': None, 'q_monte_carlo_se': None})
    parameters = {'bootstrap_repeats': BOOTSTRAP_REPEATS, 'bootstrap_seed': BOOTSTRAP_SEED,
                  'bootstrap_rng': 'numpy.random.default_rng (PCG64)',
                  'interval': '95% percentile, paired stratified resampling of whole scenarios',
                  'strata': 'condition x load_target x history_n; 18 overall, 2 per condition/history subgroup',
                  'estimand': 'equal-scenario mean; fixed experimental-cell weights',
                  'monte_carlo_se': 'sqrt(sum_h(w_h^2 * sample_variance_h / n_h))',
                  'primary_metric': Q, 'contrast_units': 'percentage points',
                  'interval_scope': 'Monte Carlo uncertainty for this generator only',
                  'contrasts': [list(pair) for pair in CONTRASTS]}
    result = {'parameters': parameters, 'integrity_checks': checks,
              'summary': summary, 'paired_comparisons': paired, 'subgroup_metrics': subgroups}
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output/'summary.csv', summary)
    write_csv(output/'paired_comparisons.csv', paired)
    write_csv(output/'subgroup_metrics.csv', subgroups)
    write_json(output/'analysis.json', result)
    files = ('summary.csv', 'paired_comparisons.csv', 'subgroup_metrics.csv', 'analysis.json')
    analysis_manifest = {'status': 'complete', 'created_utc': datetime.now(timezone.utc).isoformat(),
                         'python': sys.version, 'numpy': np.__version__, 'platform': platform.platform(),
                         'analysis_script_sha256': sha(__file__), 'parameters': parameters,
                         'source_manifest_sha256': sha(source/'manifest.json'),
                         'source_checksums': checks['hashes'],
                         'output_checksums': {filename: sha(output/filename) for filename in files}}
    write_json(output/'analysis_manifest.json', analysis_manifest)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('experiments/results_v1'))
    parser.add_argument('--output', type=Path, default=Path('experiments/results_v1/analysis'))
    args = parser.parse_args()
    result = analyze(args.input, args.output)
    print(json.dumps({'integrity_checks': result['integrity_checks'], 'summary': result['summary'],
                      'paired_comparisons': result['paired_comparisons']}, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
