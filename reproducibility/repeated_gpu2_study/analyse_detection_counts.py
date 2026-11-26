"""Detection-count consequences of frame discards in the GPU 2 repeated arms.

Uses stored logs only (no video, weights or GPU). For every source, the detector's
box count for a replay index is taken from the delivered rows of all arms; the script
first verifies that this count is identical wherever the same replay index was delivered.
Counts of discarded frames are then recovered from that per-index reference.

Outputs (written to --output, default fresh_detection_count_analysis/; copies are supplied in tables/):
  detection_count_by_arm.csv      one row per arm
  detection_count_summary.csv     mean and range over three executions

Run from this directory:  python analyse_detection_counts.py [--output DIR]
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'repeated_gpu2'
ap = argparse.ArgumentParser()
ap.add_argument('--output', default=str(HERE / 'fresh_detection_count_analysis'))
OUT = Path(ap.parse_args().output)
CYCLE = {'s0': 158, 's1': 600, 's2': 600}
SOURCE = {'s0': 'Cows2021', 's1': 'June NVR', 's2': 'November NVR'}
POLICY = {'fifo': 'FIFO', 'latest': 'Latest-pending', 'drop_newest': 'Drop-newest', 'periodic': 'Periodic'}


def arm_files():
    for f in sorted(RUNS.glob('r*_delivered.csv')):
        rep, src, fps, pol = f.stem[:-len('_delivered')].split('_', 3)
        yield f, int(rep[1:]), src, int(fps), pol


# 1. Per-replay-index reference and determinism check.
seen = {s: {} for s in CYCLE}
checked_rows = 0
for f, rep, src, fps, pol in arm_files():
    d = pd.read_csv(f)
    assert d['error'].isna().all(), f
    for idx, n in zip(d['frame'] % CYCLE[src], d['n_detections']):
        prev = seen[src].setdefault(int(idx), int(n))
        assert prev == int(n), (f.name, idx, prev, n)
    checked_rows += len(d)
ref = {}
for s, m in seen.items():
    assert len(m) == CYCLE[s], (s, len(m))
    ref[s] = np.array([m[i] for i in range(CYCLE[s])])

# 2. Per-arm measures.
rows = []
for f, rep, src, fps, pol in arm_files():
    d = pd.read_csv(f).sort_values('finish_ms')
    a = pd.read_csv(f.with_name(f.name.replace('_delivered', '_arrivals'))).sort_values('frame')
    n = len(a)
    truth = ref[src][a['frame'].to_numpy() % CYCLE[src]]
    kept = np.zeros(n, bool)
    kept[d['frame'].to_numpy()] = True
    assert kept.sum() == len(d)
    # Segments: maximal runs of consecutive scheduled frames with an identical count.
    seg_id = np.concatenate([[0], np.cumsum(truth[1:] != truth[:-1])])
    n_seg = int(seg_id[-1]) + 1
    seg_len = np.bincount(seg_id, minlength=n_seg)
    seg_kept = np.bincount(seg_id, weights=kept, minlength=n_seg) > 0
    missed = ~seg_kept
    # Displayed count: count of the frame most recently completed at each arrival instant.
    fin = d['finish_ms'].to_numpy()
    shown_all = ref[src][d['frame'].to_numpy() % CYCLE[src]]
    pos = np.searchsorted(fin, a['arrival_ms'].to_numpy(), side='right') - 1
    valid = pos >= 0
    shown = shown_all[pos[valid]]
    err = shown - truth[valid]
    dc = d['n_detections']
    # Longest run of consecutive discarded scheduled frames.
    longest = run = 0
    for k in kept:
        run = 0 if k else run + 1
        longest = max(longest, run)
    rows.append(dict(
        repetition=rep, source=SOURCE[src], arrivals_per_s=fps, policy=POLICY[pol],
        scheduled=n, delivered=int(kept.sum()), discarded=int((~kept).sum()),
        mean_count_all=truth.mean(), mean_count_delivered=truth[kept].mean(),
        mean_count_discarded=truth[~kept].mean() if (~kept).any() else np.nan,
        discard_pct_count0=100 * (~kept)[truth == 0].mean() if (truth == 0).any() else np.nan,
        discard_pct_count1=100 * (~kept)[truth == 1].mean() if (truth == 1).any() else np.nan,
        discard_pct_count2plus=100 * (~kept)[truth >= 2].mean() if (truth >= 2).any() else np.nan,
        count_segments=n_seg, segments_without_delivery=int(missed.sum()),
        segments_without_delivery_pct=100 * missed.mean(),
        longest_missed_segment_frames=int(seg_len[missed].max()) if missed.any() else 0,
        single_frame_segments=int((seg_len == 1).sum()),
        single_frame_segments_missed=int((missed & (seg_len == 1)).sum()),
        longest_discard_run_frames=longest,
        mean_service_ms_count0=d.loc[dc == 0, 'service_ms'].mean() if (dc == 0).any() else np.nan,
        mean_service_ms_count1=d.loc[dc == 1, 'service_ms'].mean() if (dc == 1).any() else np.nan,
        mean_service_ms_count2plus=d.loc[dc >= 2, 'service_ms'].mean() if (dc >= 2).any() else np.nan,
        displayed_count_mismatch_pct=100 * (err != 0).mean(),
        displayed_count_mean_abs_error=np.abs(err).mean(),
        arrivals_evaluated_for_display=int(valid.sum()),
    ))

arms = pd.DataFrame(rows).sort_values(['arrivals_per_s', 'source', 'policy', 'repetition'])
OUT.mkdir(parents=True, exist_ok=True)
arms.to_csv(OUT / 'detection_count_by_arm.csv', index=False, float_format='%.6f')

value_cols = [c for c in arms.columns if c not in ('repetition', 'source', 'arrivals_per_s', 'policy')]
g = arms.groupby(['arrivals_per_s', 'source', 'policy'], sort=False)[value_cols]
summary = g.mean().add_suffix('_mean').join(g.min().add_suffix('_min')).join(g.max().add_suffix('_max'))
summary = summary[sorted(summary.columns, key=lambda c: value_cols.index(c.rsplit('_', 1)[0]))].reset_index()
summary.to_csv(OUT / 'detection_count_summary.csv', index=False, float_format='%.6f')

print(json.dumps({
    'arms': len(arms), 'delivered_rows_checked': checked_rows,
    'replay_indices': {SOURCE[s]: int(len(v)) for s, v in ref.items()},
    'inconsistent_counts': 0,
    'reference_mean_count': {SOURCE[s]: round(float(v.mean()), 4) for s, v in ref.items()},
    'reference_count_values': {SOURCE[s]: {int(k): int(c) for k, c in zip(*np.unique(v, return_counts=True))} for s, v in ref.items()},
}, indent=1))
