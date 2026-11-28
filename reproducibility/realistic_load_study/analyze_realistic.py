from pathlib import Path
import argparse, csv, json
import numpy as np


def mean_age_ms(run, item):
    rows = list(csv.DictReader((run / f"{item['label']}_delivered.csv").open()))
    end = 1000.0 * item['duration_s']; ages = []
    for c in range(item['cameras']):
        ev = sorted((float(r['finish_ms']), float(r['arrival_ms'])) for r in rows if int(r['camera']) == c)
        ev = [e for e in ev if e[0] <= end]
        if len(ev) < 1 or ev[0][0] >= end: ages.append(float('nan')); continue
        area = 0.0; newest = -1e18
        for k, (t, arr) in enumerate(ev):
            newest = max(newest, arr); t_next = ev[k + 1][0] if k + 1 < len(ev) else end
            a0 = t - newest; a1 = t_next - newest; area += (a0 + a1) / 2 * (t_next - t)
        ages.append(area / (end - ev[0][0]))
    return float(np.nanmean(ages)), ages


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('runs', nargs='+'); ap.add_argument('--out', required=True); a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False); arms = []; excluded = []
    for run in map(Path, a.runs):
        meta = json.loads((run / 'metadata.json').read_text())
        for item in json.loads((run / 'summaries.json').read_text()):
            if item['foreign_process_detected']:
                excluded.append(dict(run=run.name, label=item['label'], order=item['order'])); continue
            age, per_cam = mean_age_ms(run, item)
            reasons = {}
            dp = run / f"{item['label']}_dropped.csv"
            if dp.exists():
                for r in csv.DictReader(dp.open()): reasons[r['reason']] = reasons.get(r['reason'], 0) + 1
            arms.append({**{k: item[k] for k in ['label', 'tenant', 'detector', 'cameras', 'rate', 'aggregate_rate', 'policy', 'replicate', 'order', 'started_utc', 'n', 'delivered', 'dropped', 'drop_pct',
                                                 'errors', 'queue_max', 'p50_ms', 'p95_ms', 'p99_ms', 'max_ms', 'mean_service_ms', 'miss33_pct', 'miss_interval_pct', 'periodic_keep_fraction',
                                                 'calibration_mean_service_ms', 'max_arrival_lateness_ms', 'foreign_process_detected']},
                         'mean_age_ms': age, 'offered_load': item['aggregate_rate'] * item['calibration_mean_service_ms'] / 1000,
                         'discard_reasons': json.dumps(reasons, sort_keys=True), 'run': run.name})
    with (out / 'arms.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(arms[0])); w.writeheader(); w.writerows(arms)
    cells = {}
    for r in arms: cells.setdefault((r['tenant'], r['detector'], r['cameras'], r['rate'], r['policy']), []).append(r)
    rows = []
    for (tenant, det, k, rate, pol), rs in sorted(cells.items(), key=lambda x: (x[0][0], x[0][1], x[0][3], x[0][2], ['fifo', 'latest', 'drop_newest', 'periodic'].index(x[0][4]))):
        row = dict(tenant=tenant, detector=det, cameras=k, rate=rate, aggregate_rate=k * rate, policy=pol, executions=len(rs), offered_load=float(np.mean([r['offered_load'] for r in rs])))
        for m in ['p99_ms', 'drop_pct', 'mean_age_ms', 'mean_service_ms', 'miss33_pct', 'miss_interval_pct', 'queue_max', 'p50_ms']:
            v = [r[m] for r in rs]; row[m + '_mean'] = float(np.mean(v)); row[m + '_min'] = float(np.min(v)); row[m + '_max'] = float(np.max(v))
        row['errors'] = sum(r['errors'] for r in rs); row['foreign_process_any'] = any(r['foreign_process_detected'] for r in rs)
        rows.append(row)
    with (out / 'cells.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / 'analysis_summary.json').write_text(json.dumps(dict(arms=len(arms), cells=len(rows), total_arrivals=sum(r['n'] for r in arms), total_delivered=sum(r['delivered'] for r in arms),
                                                               total_errors=sum(r['errors'] for r in arms), excluded_arms=excluded, runs=[str(Path(r).name) for r in a.runs]), indent=2))
    for r in rows:
        print(f"{r['tenant'][:5]:5s} {r['detector']} K={r['cameras']} r={r['rate']:>2} load={r['offered_load']:.2f} {r['policy']:12s} p99={r['p99_ms_mean']:9.1f} [{r['p99_ms_min']:.1f}-{r['p99_ms_max']:.1f}] loss={r['drop_pct_mean']:5.1f}% age={r['mean_age_ms_mean']:8.1f} n={r['executions']}")


if __name__ == '__main__':
    main()
